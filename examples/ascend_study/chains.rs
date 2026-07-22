//! Standard and max-plus chain workloads.

use super::{data, max_abs_error, measure};
use omeinsum::{Ascend, Cpu, MaxPlus, Standard, Tensor};
use std::hint::black_box;

pub(super) fn chain_case(ascend: &Ascend, n: usize, steps: usize) {
    let a_data = data(n * n, 17);
    let b_data: Vec<_> = data(n * n, 13)
        .into_iter()
        .map(|value| value / (n as f32).sqrt())
        .collect();
    let cpu_a = Tensor::<f32, Cpu>::from_data(&a_data, &[n, n]);
    let cpu_b = Tensor::<f32, Cpu>::from_data(&b_data, &[n, n]);
    let npu_a = Tensor::<f32, Ascend>::from_data_with_backend(&a_data, &[n, n], ascend.clone());
    let npu_b = Tensor::<f32, Ascend>::from_data_with_backend(&b_data, &[n, n], ascend.clone());
    let mut expected = cpu_a.clone();
    let mut actual = npu_a.clone();
    for _ in 0..steps {
        expected = expected.contract_binary::<Standard<f32>>(&cpu_b, &[0, 1], &[1, 2], &[0, 2]);
        actual = actual.contract_binary::<Standard<f32>>(&npu_b, &[0, 1], &[1, 2], &[0, 2]);
    }
    let expected = expected.to_vec();
    let actual = actual.to_vec();
    assert!(expected
        .iter()
        .chain(&actual)
        .all(|value| value.is_finite()));
    let error = max_abs_error(&expected, &actual);
    let scale = expected.iter().map(|value| value.abs()).fold(0.0, f32::max);
    assert!(
        scale > f32::MIN_POSITIVE,
        "standard chain result underflowed"
    );
    let relative_error = error / scale;
    assert!(
        relative_error <= 1.0e-3,
        "standard chain relative error {relative_error}"
    );
    println!("ACCURACY,chain,standard,standard,relative_max_error,{relative_error:.9e}");

    let operations = 2.0 * (steps * n * n * n) as f64;
    let count = 3;
    measure(
        "chain",
        "standard",
        "cpu",
        "device_resident",
        n,
        n,
        n,
        steps,
        operations,
        count,
        || {
            let mut state = cpu_a.clone();
            for _ in 0..steps {
                state = state.contract_binary::<Standard<f32>>(&cpu_b, &[0, 1], &[1, 2], &[0, 2]);
            }
            black_box(state);
        },
    );
    measure(
        "chain",
        "standard",
        "ascend",
        "device_resident",
        n,
        n,
        n,
        steps,
        operations,
        count,
        || {
            let mut state = npu_a.clone();
            for _ in 0..steps {
                state = state.contract_binary::<Standard<f32>>(&npu_b, &[0, 1], &[1, 2], &[0, 2]);
            }
            black_box(state);
        },
    );
    measure(
        "chain",
        "standard",
        "cpu",
        "result_download",
        n,
        n,
        n,
        steps,
        operations,
        count,
        || {
            let mut state = cpu_a.clone();
            for _ in 0..steps {
                state = state.contract_binary::<Standard<f32>>(&cpu_b, &[0, 1], &[1, 2], &[0, 2]);
            }
            black_box(state.to_vec()[0]);
        },
    );
    measure(
        "chain",
        "standard",
        "ascend",
        "result_download",
        n,
        n,
        n,
        steps,
        operations,
        count,
        || {
            let mut state = npu_a.clone();
            for _ in 0..steps {
                state = state.contract_binary::<Standard<f32>>(&npu_b, &[0, 1], &[1, 2], &[0, 2]);
            }
            black_box(state.to_vec()[0]);
        },
    );
}

pub(super) fn tropical_chain_case(ascend: &Ascend, n: usize, steps: usize) {
    let a_data = data(n * n, 17);
    let b_data = data(n * n, 13);
    let cpu_b = Tensor::<f32, Cpu>::from_data(&b_data, &[n, n]);
    let npu_b = Tensor::<f32, Ascend>::from_data_with_backend(&b_data, &[n, n], ascend.clone());
    let mut expected = Tensor::<f32, Cpu>::from_data(&a_data, &[n, n]);
    let mut actual =
        Tensor::<f32, Ascend>::from_data_with_backend(&a_data, &[n, n], ascend.clone());
    for _ in 0..steps {
        expected = expected.contract_binary::<MaxPlus<f32>>(&cpu_b, &[0, 1], &[1, 2], &[0, 2]);
        actual = actual.contract_binary::<MaxPlus<f32>>(&npu_b, &[0, 1], &[1, 2], &[0, 2]);
    }
    assert_eq!(actual.to_vec(), expected.to_vec());
    println!("ACCURACY,chain,maxplus,maxplus,exact_match,true");

    let operations = 2.0 * (steps * n * n * n) as f64;
    measure(
        "chain",
        "maxplus",
        "cpu",
        "result_download",
        n,
        n,
        n,
        steps,
        operations,
        3,
        || {
            let mut state = Tensor::<f32, Cpu>::from_data(&a_data, &[n, n]);
            for _ in 0..steps {
                state = state.contract_binary::<MaxPlus<f32>>(&cpu_b, &[0, 1], &[1, 2], &[0, 2]);
            }
            black_box(state.to_vec()[0]);
        },
    );
    measure(
        "chain",
        "maxplus",
        "ascend",
        "result_download",
        n,
        n,
        n,
        steps,
        operations,
        3,
        || {
            let mut state =
                Tensor::<f32, Ascend>::from_data_with_backend(&a_data, &[n, n], ascend.clone());
            for _ in 0..steps {
                state = state.contract_binary::<MaxPlus<f32>>(&npu_b, &[0, 1], &[1, 2], &[0, 2]);
            }
            black_box(state.to_vec()[0]);
        },
    );
}
