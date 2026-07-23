use clap::{Parser, Subcommand};

mod autodiff;
mod benchmark_plan;
mod common;
mod contract;
mod execute_plan;
mod format;
mod optimize;
mod parse;
mod static_plan;
mod yao_tn;

#[derive(Parser)]
#[command(name = "omeinsum", version, about = "Einstein summation CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Optimize contraction order for an einsum expression
    Optimize {
        /// Einsum expression in NumPy notation (e.g., "ij,jk->ik")
        expression: String,

        /// Dimension sizes as "label=size" pairs (e.g., "i=2,j=3,k=4")
        #[arg(long)]
        sizes: String,

        /// Optimization method: "greedy" or "treesa"
        #[arg(long, default_value = "greedy")]
        method: String,

        // -- greedy parameters --
        /// [greedy] Weight for output-vs-input size balance (default: 0.0)
        #[arg(long)]
        alpha: Option<f64>,

        /// [greedy] Temperature for stochastic selection; 0 = deterministic (default: 0.0)
        #[arg(long)]
        temperature: Option<f64>,

        // -- treesa parameters --
        /// [treesa] Number of independent SA trials (default: 10)
        #[arg(long)]
        ntrials: Option<usize>,

        /// [treesa] Iterations per temperature level (default: 50)
        #[arg(long)]
        niters: Option<usize>,

        /// [treesa] Space complexity target threshold (default: 20.0)
        #[arg(long)]
        sc_target: Option<f64>,

        /// [treesa] Inverse temperature schedule as "start:step:stop" (default: "0.01:0.05:15.0")
        #[arg(long)]
        betas: Option<String>,

        /// [treesa] Time complexity weight (default: 1.0)
        #[arg(long)]
        tc_weight: Option<f64>,

        /// [treesa] Space complexity weight (default: 1.0)
        #[arg(long)]
        sc_weight: Option<f64>,

        /// [treesa] Read-write complexity weight (default: 0.0)
        #[arg(long)]
        rw_weight: Option<f64>,

        /// Output file (default: stdout)
        #[arg(short, long)]
        output: Option<String>,

        /// Pretty-print JSON (default: auto-detect TTY)
        #[arg(long)]
        pretty: Option<bool>,
    },
    /// Execute a tensor contraction
    Contract {
        /// Tensors JSON file
        tensors: String,

        /// Topology JSON file
        #[arg(short = 't', long)]
        topology: Option<String>,

        /// Parenthesized einsum expression with explicit contraction order
        #[arg(long)]
        expr: Option<String>,

        /// Output file (default: stdout)
        #[arg(short, long)]
        output: Option<String>,

        /// Pretty-print JSON (default: auto-detect TTY)
        #[arg(long)]
        pretty: Option<bool>,
    },
    /// Execute autodiff and emit the forward result plus input gradients
    Autodiff {
        /// Tensors JSON file
        tensors: String,

        /// Topology JSON file
        #[arg(short = 't', long)]
        topology: Option<String>,

        /// Parenthesized einsum expression with explicit contraction order
        #[arg(long)]
        expr: Option<String>,

        /// Gradient seed for the einsum output, using the Result JSON schema
        #[arg(long = "grad-output")]
        grad_output: Option<String>,

        /// Output file (default: stdout)
        #[arg(short, long)]
        output: Option<String>,

        /// Pretty-print JSON (default: auto-detect TTY)
        #[arg(long)]
        pretty: Option<bool>,
    },
    /// Compile an optimized yao-tn-v1 scalar network into frozen real plans
    StaticPlan {
        /// Optimized yao-tn-v1 JSON file
        input: String,

        /// Maximum imaginary magnitude for classifying a tensor as real
        #[arg(long, default_value_t = 1e-12)]
        realness_tol: f64,

        /// Output file (default: stdout)
        #[arg(short, long)]
        output: Option<String>,

        /// Pretty-print JSON (default: auto-detect TTY)
        #[arg(long)]
        pretty: Option<bool>,
    },
    /// Execute prepared static plans once and emit untimed correctness outputs
    ExecutePlan {
        /// Frozen omeinsum-static-plan-v1 JSON file
        plan: String,

        /// Execution backend
        #[arg(long, default_value = "cpu")]
        backend: String,

        /// Comma-separated representations
        #[arg(long, default_value = "real-skeleton,flat-4m,realified-rank3")]
        representations: String,

        /// Floating-point dtype
        #[arg(long, default_value = "f64")]
        dtype: String,

        /// Ascend device id (required with --backend ascend)
        #[arg(long)]
        device_id: Option<i32>,

        /// Ascend precision mode
        #[arg(long, default_value = "keep-dtype")]
        precision_mode: String,

        /// Output file (default: stdout)
        #[arg(short, long)]
        output: Option<String>,

        /// Pretty-print JSON (default: auto-detect TTY)
        #[arg(long)]
        pretty: Option<bool>,
    },
    /// Benchmark reusable prepared static plans
    BenchmarkPlan {
        /// Frozen omeinsum-static-plan-v1 JSON file
        plan: String,

        /// Execution backend
        #[arg(long, default_value = "cpu")]
        backend: String,

        /// Comma-separated representations
        #[arg(long, default_value = "real-skeleton,flat-4m,realified-rank3")]
        representations: String,

        /// Floating-point dtype
        #[arg(long, default_value = "f64")]
        dtype: String,

        /// Ascend device id (required with --backend ascend)
        #[arg(long)]
        device_id: Option<i32>,

        /// Ascend precision mode
        #[arg(long, default_value = "keep-dtype")]
        precision_mode: String,

        /// Capture policy for realified-rank3: off, auto, or required
        #[arg(long, default_value = "off")]
        capture_realified: String,

        /// Untimed warm-up rounds
        #[arg(long, default_value_t = 3)]
        warmups: usize,

        /// Timed samples per representation
        #[arg(long, default_value_t = 5)]
        samples: usize,

        /// Minimum duration used to calibrate each timed sample
        #[arg(long, default_value_t = 200)]
        min_sample_ms: u64,

        /// Seed controlling per-round representation order
        #[arg(long, default_value_t = 20260723)]
        measurement_order_seed: u64,

        /// Output file (default: stdout)
        #[arg(short, long)]
        output: Option<String>,

        /// Pretty-print JSON (default: auto-detect TTY)
        #[arg(long)]
        pretty: Option<bool>,
    },
}

fn main() {
    let cli = Cli::parse();
    let result = match cli.command {
        Commands::Optimize {
            expression,
            sizes,
            method,
            alpha,
            temperature,
            ntrials,
            niters,
            sc_target,
            betas,
            tc_weight,
            sc_weight,
            rw_weight,
            output,
            pretty,
        } => optimize::run(
            &expression,
            &sizes,
            &method,
            optimize::OptimizeParams {
                alpha,
                temperature,
                ntrials,
                niters,
                sc_target,
                betas,
                tc_weight,
                sc_weight,
                rw_weight,
            },
            output.as_deref(),
            pretty,
        ),
        Commands::Contract {
            tensors,
            topology,
            expr,
            output,
            pretty,
        } => contract::run(
            &tensors,
            topology.as_deref(),
            expr.as_deref(),
            output.as_deref(),
            pretty,
        ),
        Commands::Autodiff {
            tensors,
            topology,
            expr,
            grad_output,
            output,
            pretty,
        } => autodiff::run(
            &tensors,
            topology.as_deref(),
            expr.as_deref(),
            grad_output.as_deref(),
            output.as_deref(),
            pretty,
        ),
        Commands::StaticPlan {
            input,
            realness_tol,
            output,
            pretty,
        } => static_plan::run(&input, realness_tol, output.as_deref(), pretty),
        Commands::ExecutePlan {
            plan,
            backend,
            representations,
            dtype,
            device_id,
            precision_mode,
            output,
            pretty,
        } => execute_plan::run(
            &plan,
            &backend,
            &representations,
            &dtype,
            device_id,
            &precision_mode,
            output.as_deref(),
            pretty,
        ),
        Commands::BenchmarkPlan {
            plan,
            backend,
            representations,
            dtype,
            device_id,
            precision_mode,
            capture_realified,
            warmups,
            samples,
            min_sample_ms,
            measurement_order_seed,
            output,
            pretty,
        } => benchmark_plan::run(
            &plan,
            &backend,
            &representations,
            &dtype,
            device_id,
            &precision_mode,
            &capture_realified,
            warmups,
            samples,
            min_sample_ms,
            measurement_order_seed,
            output.as_deref(),
            pretty,
        ),
    };

    if let Err(err) = result {
        eprintln!("Error: {err}");
        std::process::exit(1);
    }
}
