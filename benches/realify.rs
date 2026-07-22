use std::collections::HashMap;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use num_complex::Complex64;
use omeinsum::realify::{mul_vertex_tensor, realify_code};
use omeinsum::{realify_data, split_re_im, Cpu, Einsum, Standard, Tensor};

const MPO_BOND_DIM: usize = 9;
const CHI_VALUES: [usize; 3] = [32, 64, 128];

#[derive(Clone, Copy)]
struct TdvpContraction {
    name: &'static str,
    left_shape: fn(usize) -> Vec<usize>,
    right_shape: fn(usize) -> Vec<usize>,
    left_labels: &'static [usize],
    right_labels: &'static [usize],
    output_labels: &'static [usize],
    output_shape: fn(usize) -> Vec<usize>,
    logical_complex_macs: fn(usize) -> u64,
}

struct PreparedContraction {
    native_einsum: Einsum<usize>,
    native_tensors: Vec<Tensor<Complex64, Cpu>>,
    realified_einsum: Einsum<usize>,
    realified_tensors: Vec<Tensor<f64, Cpu>>,
}

fn patterned_complex(len: usize, seed: usize) -> Vec<Complex64> {
    (0..len)
        .map(|index| {
            let real = ((index.wrapping_mul(17) + seed.wrapping_mul(31)) % 257) as f64;
            let imag = ((index.wrapping_mul(29) + seed.wrapping_mul(13)) % 251) as f64;
            Complex64::new((real - 128.0) / 37.0, (imag - 125.0) / 41.0)
        })
        .collect()
}

fn prepare(case: TdvpContraction, chi: usize) -> PreparedContraction {
    let left_shape = (case.left_shape)(chi);
    let right_shape = (case.right_shape)(chi);
    let left_len = left_shape.iter().product();
    let right_len = right_shape.iter().product();
    let left_data = patterned_complex(left_len, 1);
    let right_data = patterned_complex(right_len, 2);
    let source_ixs = vec![case.left_labels.to_vec(), case.right_labels.to_vec()];
    let source_iy = case.output_labels.to_vec();
    let source_sizes = infer_sizes(&[&left_shape, &right_shape], &source_ixs);

    let mut native_einsum =
        Einsum::new(source_ixs.clone(), source_iy.clone(), source_sizes.clone());
    native_einsum.optimize_greedy();
    let native_tensors = vec![
        Tensor::from_data(&left_data, &left_shape),
        Tensor::from_data(&right_data, &right_shape),
    ];

    let mut realified_plan = realify_code(&source_ixs, &source_iy, &source_sizes, &[true, true]);
    realified_plan.einsum.optimize_greedy();
    let mut left_real_shape = left_shape.clone();
    left_real_shape.push(2);
    let mut right_real_shape = right_shape.clone();
    right_real_shape.push(2);
    let realified_tensors = vec![
        Tensor::from_data(&realify_data(&left_data, false), &left_real_shape),
        Tensor::from_data(&realify_data(&right_data, false), &right_real_shape),
        mul_vertex_tensor::<f64, Cpu>(Cpu),
    ];

    PreparedContraction {
        native_einsum,
        native_tensors,
        realified_einsum: realified_plan.einsum,
        realified_tensors,
    }
}

fn infer_sizes(shapes: &[&[usize]], ixs: &[Vec<usize>]) -> HashMap<usize, usize> {
    let mut sizes = HashMap::new();
    for (shape, ix) in shapes.iter().zip(ixs.iter()) {
        for (&label, &size) in ix.iter().zip(shape.iter()) {
            if let Some(existing) = sizes.insert(label, size) {
                assert_eq!(existing, size, "label {label} has inconsistent sizes");
            }
        }
    }
    sizes
}

fn tensor_refs<T>(tensors: &[Tensor<T, Cpu>]) -> Vec<&Tensor<T, Cpu>>
where
    T: omeinsum::algebra::Scalar,
{
    tensors.iter().collect()
}

fn run_native(prepared: &PreparedContraction) -> Tensor<Complex64, Cpu> {
    prepared
        .native_einsum
        .execute::<Standard<Complex64>, Complex64, Cpu>(&tensor_refs(&prepared.native_tensors))
}

fn run_realified(prepared: &PreparedContraction) -> Tensor<f64, Cpu> {
    prepared
        .realified_einsum
        .execute::<Standard<f64>, f64, Cpu>(&tensor_refs(&prepared.realified_tensors))
}

fn assert_equivalent_outputs(native: &Tensor<Complex64, Cpu>, realified: &Tensor<f64, Cpu>) {
    let (re, im) = split_re_im(realified);
    let native = native.to_vec();
    assert_eq!(native.len(), re.len());
    for (expected, (actual_re, actual_im)) in native.into_iter().zip(re.into_iter().zip(im)) {
        let tolerance = 1e-8 * (1.0 + expected.norm());
        assert!((expected.re - actual_re).abs() <= tolerance);
        assert!((expected.im - actual_im).abs() <= tolerance);
    }
}

fn h1_left_shape(chi: usize) -> Vec<usize> {
    vec![chi, MPO_BOND_DIM, chi]
}

fn h1_wavefunction_shape(chi: usize) -> Vec<usize> {
    vec![chi, 2, chi]
}

fn h1_left_output_shape(chi: usize) -> Vec<usize> {
    vec![MPO_BOND_DIM, chi, 2, chi]
}

fn h1_right_intermediate_shape(chi: usize) -> Vec<usize> {
    vec![chi, chi, 2, MPO_BOND_DIM]
}

fn h1_right_output_shape(chi: usize) -> Vec<usize> {
    vec![chi, 2, chi]
}

fn right_environment_shape(chi: usize) -> Vec<usize> {
    vec![chi, MPO_BOND_DIM, chi]
}

fn h2_wavefunction_shape(chi: usize) -> Vec<usize> {
    vec![chi, 2, 2, chi]
}

fn h2_left_output_shape(chi: usize) -> Vec<usize> {
    vec![MPO_BOND_DIM, chi, 2, 2, chi]
}

fn h2_right_intermediate_shape(chi: usize) -> Vec<usize> {
    vec![chi, 2, chi, 2, MPO_BOND_DIM]
}

fn h2_right_output_shape(chi: usize) -> Vec<usize> {
    vec![chi, 2, 2, chi]
}

const TDVP_CONTRACTIONS: [TdvpContraction; 4] = [
    TdvpContraction {
        name: "h1-left-environment",
        left_shape: h1_left_shape,
        right_shape: h1_wavefunction_shape,
        left_labels: &[0, 1, 2],
        right_labels: &[0, 3, 4],
        output_labels: &[1, 2, 3, 4],
        output_shape: h1_left_output_shape,
        logical_complex_macs: |chi| 2 * MPO_BOND_DIM as u64 * (chi as u64).pow(3),
    },
    TdvpContraction {
        name: "h1-right-environment",
        left_shape: h1_right_intermediate_shape,
        right_shape: right_environment_shape,
        left_labels: &[2, 4, 5, 6],
        right_labels: &[4, 6, 7],
        output_labels: &[2, 5, 7],
        output_shape: h1_right_output_shape,
        logical_complex_macs: |chi| 2 * MPO_BOND_DIM as u64 * (chi as u64).pow(3),
    },
    TdvpContraction {
        name: "h2-left-environment",
        left_shape: h1_left_shape,
        right_shape: h2_wavefunction_shape,
        left_labels: &[0, 1, 2],
        right_labels: &[0, 3, 4, 5],
        output_labels: &[1, 2, 3, 4, 5],
        output_shape: h2_left_output_shape,
        logical_complex_macs: |chi| 4 * MPO_BOND_DIM as u64 * (chi as u64).pow(3),
    },
    TdvpContraction {
        name: "h2-right-environment",
        left_shape: h2_right_intermediate_shape,
        right_shape: right_environment_shape,
        left_labels: &[2, 6, 5, 8, 9],
        right_labels: &[5, 9, 10],
        output_labels: &[2, 6, 8, 10],
        output_shape: h2_right_output_shape,
        logical_complex_macs: |chi| 4 * MPO_BOND_DIM as u64 * (chi as u64).pow(3),
    },
];

fn bench_tdvp_realify_contractions(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("tdvp-realify");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(10));

    for case in TDVP_CONTRACTIONS {
        for chi in CHI_VALUES {
            let prepared = prepare(case, chi);
            let native = run_native(&prepared);
            assert_eq!(native.shape(), (case.output_shape)(chi));
            let mut realified_shape = (case.output_shape)(chi);
            realified_shape.push(2);
            let realified = run_realified(&prepared);
            assert_eq!(realified.shape(), realified_shape);
            assert_equivalent_outputs(&native, &realified);

            // Use identical source-network work for both implementations so the
            // reported rate compares elapsed time rather than backend internals.
            group.throughput(Throughput::Elements((case.logical_complex_macs)(chi)));
            group.bench_with_input(
                BenchmarkId::new(format!("{}-native", case.name), format!("chi{chi}")),
                &prepared,
                |bencher, prepared| {
                    bencher.iter(|| {
                        let output = run_native(black_box(prepared));
                        black_box(output);
                    });
                },
            );
            group.bench_with_input(
                BenchmarkId::new(format!("{}-realified", case.name), format!("chi{chi}")),
                &prepared,
                |bencher, prepared| {
                    bencher.iter(|| {
                        let output = run_realified(black_box(prepared));
                        black_box(output);
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, bench_tdvp_realify_contractions);
criterion_main!(benches);
