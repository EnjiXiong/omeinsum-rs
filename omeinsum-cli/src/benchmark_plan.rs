use omeinsum::static_plan::{
    benchmark_prepared, contract_complex64, BenchmarkConfig, BenchmarkTarget, ComplexValue,
    ModeTiming, PreparedExecutable,
};
use serde::{Deserialize, Serialize};

use crate::execute_plan::{prepare_cpu, read_bundle, select_plans, validate_cpu_backend, CpuDtype};

#[derive(Serialize, Deserialize)]
struct BenchmarkPlanReport {
    format: String,
    tree_hash: String,
    reference_complex64: ComplexValue,
    timings: Vec<ModeTiming>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    plan_path: &str,
    backend: &str,
    representations: &str,
    dtype: &str,
    warmups: usize,
    samples: usize,
    min_sample_ms: u64,
    measurement_order_seed: u64,
    output: Option<&str>,
    pretty: Option<bool>,
) -> Result<(), String> {
    validate_cpu_backend(backend)?;
    let dtype = CpuDtype::parse(dtype)?;
    let bundle = read_bundle(plan_path)?;
    let selected = select_plans(&bundle, representations)?;
    let reference_complex64 = contract_complex64(&bundle.realified_rank3, &bundle.inputs)
        .map_err(|error| error.to_string())?;
    let config = BenchmarkConfig {
        warmups,
        samples,
        min_sample_ms,
        measurement_order_seed,
    };

    let mut executables = selected
        .into_iter()
        .map(|plan| prepare_cpu(plan, &bundle, dtype))
        .collect::<Result<Vec<Box<dyn PreparedExecutable>>, _>>()?;
    let mut targets = executables
        .iter_mut()
        .map(|executable| BenchmarkTarget {
            backend: "cpu",
            dtype: dtype.name(),
            execution_mode: "prepared",
            executable: executable.as_mut(),
        })
        .collect::<Vec<_>>();
    let timings = benchmark_prepared(&mut targets, &config).map_err(|error| error.to_string())?;

    let report = BenchmarkPlanReport {
        format: "omeinsum-benchmark-report-v1".to_string(),
        tree_hash: bundle.tree_hash,
        reference_complex64,
        timings,
    };
    crate::common::write_json_output(&report, output, pretty)
}
