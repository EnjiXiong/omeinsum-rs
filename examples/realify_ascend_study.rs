//! Correctness and performance study for complex-to-real TDVP contractions.
//!
//! `OMEINSUM_REALIFY_STUDY=quick` runs chi 32 and 64; `full` runs 32 through 256.
//! Optional `OMEINSUM_REALIFY_WARMUPS` and `OMEINSUM_REALIFY_ITERATIONS` override
//! the default repetition counts. Output is line-oriented CSV for durable capture.

use std::collections::{HashMap, HashSet};
use std::hint::black_box;
use std::time::Instant;

use num_complex::Complex32;
use omeco::NestedEinsum;
use omeinsum::backend::{Backend, BackendScalar};
use omeinsum::realify::{mul_vertex_tensor, realify_code};
use omeinsum::{realify_data, recover_complex, Ascend, Cpu, Einsum, Standard, Tensor};

const MPO_BOND_DIM: usize = 9;

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    left_shape: fn(usize) -> Vec<usize>,
    right_shape: fn(usize) -> Vec<usize>,
    left_labels: &'static [usize],
    right_labels: &'static [usize],
    output_labels: &'static [usize],
    logical_complex_macs: fn(usize) -> u64,
}

struct HostInputs {
    left_shape: Vec<usize>,
    right_shape: Vec<usize>,
    left: Vec<Complex32>,
    right: Vec<Complex32>,
}

struct Prepared<B: Backend> {
    einsum: Einsum<usize>,
    tensors: Vec<Tensor<f32, B>>,
}

fn patterned_complex(len: usize, seed: usize, scale: f32) -> Vec<Complex32> {
    (0..len)
        .map(|index| {
            let re = ((index.wrapping_mul(17) + seed.wrapping_mul(31)) % 257) as f32;
            let im = ((index.wrapping_mul(29) + seed.wrapping_mul(13)) % 251) as f32;
            Complex32::new((re - 128.0) * scale / 128.0, (im - 125.0) * scale / 125.0)
        })
        .collect()
}

fn host_inputs(case: Case, chi: usize) -> HostInputs {
    let left_shape = (case.left_shape)(chi);
    let right_shape = (case.right_shape)(chi);
    let scale = 1.0 / (chi as f32).sqrt();
    HostInputs {
        left: patterned_complex(left_shape.iter().product(), 1, scale),
        right: patterned_complex(right_shape.iter().product(), 2, scale),
        left_shape,
        right_shape,
    }
}

fn source_sizes(case: Case, inputs: &HostInputs) -> HashMap<usize, usize> {
    let mut sizes = HashMap::new();
    for (shape, labels) in [
        (&inputs.left_shape, case.left_labels),
        (&inputs.right_shape, case.right_labels),
    ] {
        for (&label, &size) in labels.iter().zip(shape) {
            if let Some(previous) = sizes.insert(label, size) {
                assert_eq!(previous, size, "inconsistent size for label {label}");
            }
        }
    }
    sizes
}

fn native_cpu(case: Case, inputs: &HostInputs) -> (Einsum<usize>, Vec<Tensor<Complex32, Cpu>>) {
    let mut einsum = Einsum::new(
        vec![case.left_labels.to_vec(), case.right_labels.to_vec()],
        case.output_labels.to_vec(),
        source_sizes(case, inputs),
    );
    einsum.optimize_greedy();
    let tensors = vec![
        Tensor::from_data(&inputs.left, &inputs.left_shape),
        Tensor::from_data(&inputs.right, &inputs.right_shape),
    ];
    (einsum, tensors)
}

fn prepare_realified<B>(case: Case, inputs: &HostInputs, backend: B) -> Prepared<B>
where
    B: Backend,
    f32: BackendScalar<B>,
{
    let mut plan = realify_code(
        &[case.left_labels.to_vec(), case.right_labels.to_vec()],
        case.output_labels,
        &source_sizes(case, inputs),
        &[true, true],
    );
    plan.einsum.optimize_greedy();

    Prepared {
        einsum: plan.einsum,
        tensors: upload_realified(inputs, backend),
    }
}

fn upload_realified<B>(inputs: &HostInputs, backend: B) -> Vec<Tensor<f32, B>>
where
    B: Backend,
    f32: BackendScalar<B>,
{
    let mut left_shape = inputs.left_shape.clone();
    left_shape.push(2);
    let mut right_shape = inputs.right_shape.clone();
    right_shape.push(2);
    vec![
        Tensor::from_data_with_backend(
            &realify_data(&inputs.left, false),
            &left_shape,
            backend.clone(),
        ),
        Tensor::from_data_with_backend(
            &realify_data(&inputs.right, false),
            &right_shape,
            backend.clone(),
        ),
        mul_vertex_tensor::<f32, B>(backend),
    ]
}

fn execute<T, B>(einsum: &Einsum<usize>, tensors: &[Tensor<T, B>]) -> Tensor<T, B>
where
    T: omeinsum::algebra::Scalar + BackendScalar<B>,
    B: Backend,
    Standard<T>: omeinsum::algebra::Algebra<Scalar = T, Index = u32>,
{
    let refs: Vec<_> = tensors.iter().collect();
    einsum.execute::<Standard<T>, T, B>(&refs)
}

fn measure<F>(
    case: Case,
    chi: usize,
    backend: &str,
    scope: &str,
    warmups: usize,
    iterations: usize,
    mut run: F,
) where
    F: FnMut(),
{
    for _ in 0..warmups {
        run();
    }
    let start = Instant::now();
    for _ in 0..iterations {
        run();
    }
    let elapsed = start.elapsed().as_secs_f64();
    let avg_ms = elapsed * 1_000.0 / iterations as f64;
    let macs_per_second = (case.logical_complex_macs)(chi) as f64 * iterations as f64 / elapsed;
    println!(
        "RESULT,{},{chi},{backend},{scope},{warmups},{iterations},{avg_ms:.6},{macs_per_second:.6}",
        case.name
    );
}

fn errors(expected: &[Complex32], actual: &[Complex32]) -> (f32, f32, f64) {
    assert_eq!(expected.len(), actual.len(), "result length mismatch");
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut checksum = 0.0f64;
    for (index, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
        assert!(
            expected.re.is_finite()
                && expected.im.is_finite()
                && actual.re.is_finite()
                && actual.im.is_finite(),
            "non-finite output at index {index}"
        );
        let abs = (actual - expected).norm();
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(abs / expected.norm().max(1.0e-6));
        checksum += actual.re as f64 + actual.im as f64 * 0.5;
    }
    (max_abs, max_rel, checksum)
}

fn max_extra_legs(tree: &NestedEinsum<usize>, source_labels: &HashSet<usize>) -> usize {
    match tree {
        NestedEinsum::Leaf { .. } => 0,
        NestedEinsum::Node { args, eins } => {
            let here = eins
                .iy
                .iter()
                .filter(|label| !source_labels.contains(label))
                .count();
            args.iter()
                .map(|arg| max_extra_legs(arg, source_labels))
                .fold(here, usize::max)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    ascend: &Ascend,
    case: Case,
    chi: usize,
    warmups: usize,
    iterations: usize,
    cpu_iterations: usize,
) {
    let inputs = host_inputs(case, chi);
    let (native_einsum, native_tensors) = native_cpu(case, &inputs);
    let real_cpu = prepare_realified(case, &inputs, Cpu);
    let real_ascend = prepare_realified(case, &inputs, ascend.clone());

    let expected = execute(&native_einsum, &native_tensors).to_vec();
    assert!(
        expected.iter().any(|value| value.im.abs() > 1.0e-6),
        "{} chi {chi}: reference output must exercise imaginary recovery",
        case.name
    );
    let cpu_real_output = execute(&real_cpu.einsum, &real_cpu.tensors);
    let cpu_actual = recover_complex(&cpu_real_output);
    let (cpu_abs, cpu_rel, cpu_checksum) = errors(&expected, &cpu_actual);

    let ascend_output = execute(&real_ascend.einsum, &real_ascend.tensors);
    ascend.synchronize();
    let ascend_actual = recover_complex(&ascend_output);
    let (ascend_abs, ascend_rel, ascend_checksum) = errors(&expected, &ascend_actual);
    let expected_scale = expected
        .iter()
        .map(|value| value.norm())
        .fold(0.0, f32::max);
    let tolerance = 5.0e-3 * (1.0 + expected_scale);
    assert!(
        cpu_abs <= tolerance,
        "{} chi {chi}: CPU Realify max abs error {cpu_abs} exceeds {tolerance}",
        case.name
    );
    assert!(
        ascend_abs <= tolerance,
        "{} chi {chi}: Ascend max abs error {ascend_abs} exceeds {tolerance}",
        case.name
    );

    println!(
        "ACCURACY,{},chi{chi},realified_cpu,{cpu_abs:.9e},{cpu_rel:.9e},{cpu_checksum:.9e}",
        case.name
    );
    println!(
        "ACCURACY,{},chi{chi},realified_ascend,{ascend_abs:.9e},{ascend_rel:.9e},{ascend_checksum:.9e}",
        case.name
    );

    let source_labels: HashSet<_> = source_sizes(case, &inputs).into_keys().collect();
    let tree = real_ascend
        .einsum
        .contraction_tree()
        .expect("optimized tree");
    println!(
        "SCHEDULE,{},chi{chi},max_extra_re_im_legs,{},tree,{tree:?}",
        case.name,
        max_extra_legs(tree, &source_labels)
    );

    measure(
        case,
        chi,
        "native_cpu_c32",
        "execution_only",
        warmups,
        cpu_iterations,
        || {
            black_box(execute(&native_einsum, &native_tensors));
        },
    );
    measure(
        case,
        chi,
        "realified_cpu_f32",
        "execution_only",
        warmups,
        cpu_iterations,
        || {
            black_box(execute(&real_cpu.einsum, &real_cpu.tensors));
        },
    );
    measure(
        case,
        chi,
        "realified_ascend_f32",
        "execution_only",
        warmups,
        iterations,
        || {
            black_box(execute(&real_ascend.einsum, &real_ascend.tensors));
            ascend.synchronize();
        },
    );
    measure(
        case,
        chi,
        "realified_ascend_f32",
        "end_to_end",
        warmups,
        iterations,
        || {
            let tensors = upload_realified(&inputs, ascend.clone());
            let output = execute(&real_ascend.einsum, &tensors);
            ascend.synchronize();
            black_box(recover_complex(&output));
        },
    );
}

fn h1_left(chi: usize) -> Vec<usize> {
    vec![chi, MPO_BOND_DIM, chi]
}
fn h1_wavefunction(chi: usize) -> Vec<usize> {
    vec![chi, 2, chi]
}
fn h1_right_intermediate(chi: usize) -> Vec<usize> {
    vec![chi, chi, 2, MPO_BOND_DIM]
}
fn right_environment(chi: usize) -> Vec<usize> {
    vec![chi, MPO_BOND_DIM, chi]
}
fn h2_wavefunction(chi: usize) -> Vec<usize> {
    vec![chi, 2, 2, chi]
}
fn h2_right_intermediate(chi: usize) -> Vec<usize> {
    vec![chi, 2, chi, 2, MPO_BOND_DIM]
}

const CASES: [Case; 4] = [
    Case {
        name: "h1-left-environment",
        left_shape: h1_left,
        right_shape: h1_wavefunction,
        left_labels: &[0, 1, 2],
        right_labels: &[0, 3, 4],
        output_labels: &[1, 2, 3, 4],
        logical_complex_macs: |chi| 2 * MPO_BOND_DIM as u64 * (chi as u64).pow(3),
    },
    Case {
        name: "h1-right-environment",
        left_shape: h1_right_intermediate,
        right_shape: right_environment,
        left_labels: &[2, 4, 5, 6],
        right_labels: &[4, 6, 7],
        output_labels: &[2, 5, 7],
        logical_complex_macs: |chi| 2 * MPO_BOND_DIM as u64 * (chi as u64).pow(3),
    },
    Case {
        name: "h2-left-environment",
        left_shape: h1_left,
        right_shape: h2_wavefunction,
        left_labels: &[0, 1, 2],
        right_labels: &[0, 3, 4, 5],
        output_labels: &[1, 2, 3, 4, 5],
        logical_complex_macs: |chi| 4 * MPO_BOND_DIM as u64 * (chi as u64).pow(3),
    },
    Case {
        name: "h2-right-environment",
        left_shape: h2_right_intermediate,
        right_shape: right_environment,
        left_labels: &[2, 6, 5, 8, 9],
        right_labels: &[5, 9, 10],
        output_labels: &[2, 6, 8, 10],
        logical_complex_macs: |chi| 4 * MPO_BOND_DIM as u64 * (chi as u64).pow(3),
    },
];

fn env_count(name: &str, default: usize) -> usize {
    std::env::var(name)
        .map(|value| {
            value
                .parse()
                .unwrap_or_else(|_| panic!("{name} must be an integer"))
        })
        .unwrap_or(default)
}

fn main() {
    let profile = std::env::var("OMEINSUM_REALIFY_STUDY").unwrap_or_else(|_| "quick".into());
    let chis: &[usize] = match profile.as_str() {
        "quick" => &[32, 64],
        "full" => &[32, 64, 128, 256],
        _ => panic!("OMEINSUM_REALIFY_STUDY must be quick or full"),
    };
    let warmups = env_count("OMEINSUM_REALIFY_WARMUPS", 3);
    let iterations = env_count("OMEINSUM_REALIFY_ITERATIONS", 20);
    assert!(
        iterations > 0,
        "OMEINSUM_REALIFY_ITERATIONS must be positive"
    );
    let ascend = Ascend::new().expect("initialize Ascend device 0");
    println!("STUDY,realify_ascend,{profile},warmups,{warmups},iterations,{iterations}");
    println!("ACCURACY_HEADER,case,size,backend,max_abs_error,max_relative_error,checksum");
    println!("RESULT_HEADER,case,chi,backend,scope,warmups,iterations,avg_ms,logical_complex_macs_per_second");
    for case in CASES {
        for &chi in chis {
            let cpu_iterations = iterations.min(10);
            run_case(&ascend, case, chi, warmups, iterations, cpu_iterations);
        }
    }
}
