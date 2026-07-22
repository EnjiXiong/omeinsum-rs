//! Small exact-value smoke test for the experimental single-NPU Ascend backend.
//!
//! Run on a Linux host with CANN 8.5 and an allocated Ascend NPU:
//!
//! ```bash
//! source /usr/local/Ascend/cann-8.5.0/set_env.sh
//! cargo run --features ascend --example ascend_smoke
//! cargo run --features ascend-tropical --example ascend_smoke
//! ```

use omeinsum::{einsum, Ascend, Cpu, Standard, Tensor};
#[cfg(feature = "ascend-tropical")]
use omeinsum::{einsum_with_grad, MaxMul, MaxPlus, MinPlus};
use std::{hint::black_box, time::Instant};

fn benchmark<F>(name: &str, warmup: usize, iterations: usize, operations: f64, mut run: F)
where
    F: FnMut() -> f32,
{
    for _ in 0..warmup {
        black_box(run());
    }
    let start = Instant::now();
    let mut checksum = 0.0f64;
    for _ in 0..iterations {
        checksum += black_box(run()) as f64;
    }
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "{name}: iterations={iterations} avg_ms={:.3} gops={:.3} checksum={checksum:.6}",
        elapsed * 1_000.0 / iterations as f64,
        operations * iterations as f64 / elapsed / 1.0e9,
    );
}

fn run_standard_benchmark(ascend: &Ascend) {
    let n = 128;
    let data_a: Vec<f32> = (0..n * n)
        .map(|index| (index % 17) as f32 / 17.0 - 0.5)
        .collect();
    let data_b: Vec<f32> = (0..n * n)
        .map(|index| (index % 13) as f32 / 13.0 - 0.5)
        .collect();
    let a = Tensor::<f32, Ascend>::from_data_with_backend(&data_a, &[n, n], ascend.clone());
    let b = Tensor::<f32, Ascend>::from_data_with_backend(&data_b, &[n, n], ascend.clone());
    let cpu_a = Tensor::<f32, Cpu>::from_data(&data_a, &[n, n]);
    let cpu_b = Tensor::<f32, Cpu>::from_data(&data_b, &[n, n]);
    let operations = 2.0 * (n * n * n) as f64;
    let expected = cpu_a
        .contract_binary::<Standard<f32>>(&cpu_b, &[0, 1], &[1, 2], &[0, 2])
        .to_vec();
    let actual = a
        .contract_binary::<Standard<f32>>(&b, &[0, 1], &[1, 2], &[0, 2])
        .to_vec();
    let max_error = expected
        .iter()
        .zip(&actual)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max);
    assert!(max_error <= 1.0e-3, "standard max error {max_error}");
    println!("standard-f32-128 max_abs_error={max_error:.8}");

    benchmark("standard-cpu-f32-128", 1, 5, operations, || {
        cpu_a
            .contract_binary::<Standard<f32>>(&cpu_b, &[0, 1], &[1, 2], &[0, 2])
            .to_vec()[0]
    });
    benchmark("standard-ascend-f32-128", 1, 5, operations, || {
        a.contract_binary::<Standard<f32>>(&b, &[0, 1], &[1, 2], &[0, 2])
            .to_vec()[0]
    });
}

#[cfg(feature = "ascend-tropical")]
fn run_tropical_benchmark(ascend: &Ascend) {
    let n = 64;
    let data_a: Vec<f32> = (0..n * n)
        .map(|index| (index % 17) as f32 / 17.0 - 0.5)
        .collect();
    let data_b: Vec<f32> = (0..n * n)
        .map(|index| (index % 13) as f32 / 13.0 - 0.5)
        .collect();
    let a = Tensor::<f32, Ascend>::from_data_with_backend(&data_a, &[n, n], ascend.clone());
    let b = Tensor::<f32, Ascend>::from_data_with_backend(&data_b, &[n, n], ascend.clone());
    let cpu_a = Tensor::<f32, Cpu>::from_data(&data_a, &[n, n]);
    let cpu_b = Tensor::<f32, Cpu>::from_data(&data_b, &[n, n]);
    let operations = 2.0 * (n * n * n) as f64;
    let expected = cpu_a
        .contract_binary::<MaxPlus<f32>>(&cpu_b, &[0, 1], &[1, 2], &[0, 2])
        .to_vec();
    let actual = a
        .contract_binary::<MaxPlus<f32>>(&b, &[0, 1], &[1, 2], &[0, 2])
        .to_vec();
    assert_eq!(actual, expected);
    println!("maxplus-f32-64 exact_match=true");

    benchmark("maxplus-cpu-f32-64", 1, 3, operations, || {
        cpu_a
            .contract_binary::<MaxPlus<f32>>(&cpu_b, &[0, 1], &[1, 2], &[0, 2])
            .to_vec()[0]
    });
    benchmark("maxplus-ascend-f32-64", 1, 3, operations, || {
        a.contract_binary::<MaxPlus<f32>>(&b, &[0, 1], &[1, 2], &[0, 2])
            .to_vec()[0]
    });
}

fn main() {
    let ascend = Ascend::new().expect("initialize Ascend device 0");

    let roundtrip = Tensor::<f32, Ascend>::from_data_with_backend(
        &[1.0, 2.0, 3.0, 4.0],
        &[2, 2],
        ascend.clone(),
    );
    assert_eq!(roundtrip.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);

    // Column-major A = [[1, 3, 5], [2, 4, 6]].
    let a = Tensor::<f32, Ascend>::from_data_with_backend(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
        ascend.clone(),
    );
    // Column-major B = [[1, 4], [2, 5], [3, 6]].
    let b = Tensor::<f32, Ascend>::from_data_with_backend(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[3, 2],
        ascend.clone(),
    );
    let matrix = einsum::<Standard<f32>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[0, 2]);
    assert_eq!(matrix.shape(), &[2, 2]);
    assert_eq!(matrix.to_vec(), vec![22.0, 28.0, 49.0, 64.0]);

    let transposed = einsum::<Standard<f32>, _, _>(&[&a, &b], &[&[0, 1], &[1, 2]], &[2, 0]);
    assert_eq!(transposed.shape(), &[2, 2]);
    assert_eq!(transposed.to_vec(), vec![22.0, 49.0, 28.0, 64.0]);

    let x = Tensor::<f32, Ascend>::from_data_with_backend(&[1.0, 2.0, 3.0], &[3], ascend.clone());
    let y = Tensor::<f32, Ascend>::from_data_with_backend(&[4.0, 5.0, 6.0], &[3], ascend.clone());
    let dot = einsum::<Standard<f32>, _, _>(&[&x, &y], &[&[0], &[0]], &[]);
    assert!(dot.shape().is_empty());
    assert_eq!(dot.to_vec(), vec![32.0]);

    let outer = einsum::<Standard<f32>, _, _>(&[&x, &y], &[&[0], &[1]], &[0, 1]);
    assert_eq!(outer.shape(), &[3, 3]);
    assert_eq!(
        outer.to_vec(),
        vec![4.0, 8.0, 12.0, 5.0, 10.0, 15.0, 6.0, 12.0, 18.0]
    );

    // The label 2 appears only in A and not in the output, so the backend must
    // reduce that axis on the NPU before the matrix multiplication.
    let trace_a_data: Vec<f32> = (1..=24).map(|value| value as f32).collect();
    let trace_b_data: Vec<f32> = (1..=6).map(|value| value as f32).collect();
    let trace_a =
        Tensor::<f32, Ascend>::from_data_with_backend(&trace_a_data, &[2, 3, 4], ascend.clone());
    let trace_b =
        Tensor::<f32, Ascend>::from_data_with_backend(&trace_b_data, &[3, 2], ascend.clone());
    let trace_result =
        trace_a.contract_binary::<Standard<f32>>(&trace_b, &[0, 1, 2], &[1, 3], &[0, 3]);
    let cpu_a = Tensor::<f32, Cpu>::from_data(&trace_a_data, &[2, 3, 4]);
    let cpu_b = Tensor::<f32, Cpu>::from_data(&trace_b_data, &[3, 2]);
    let expected = cpu_a.contract_binary::<Standard<f32>>(&cpu_b, &[0, 1, 2], &[1, 3], &[0, 3]);
    assert_eq!(trace_result.to_vec(), expected.to_vec());

    // Direct binary contractions may carry repeated labels; normalize the
    // diagonal before dispatch instead of requiring the high-level engine.
    let diagonal = Tensor::<f32, Ascend>::from_data_with_backend(
        &[1.0, 2.0, 3.0, 4.0],
        &[2, 2],
        ascend.clone(),
    );
    let diagonal_rhs =
        Tensor::<f32, Ascend>::from_data_with_backend(&[10.0, 20.0], &[2], ascend.clone());
    let diagonal_dot = diagonal.contract_binary::<Standard<f32>>(&diagonal_rhs, &[0, 0], &[0], &[]);
    assert_eq!(diagonal_dot.to_vec(), vec![90.0]);

    #[cfg(feature = "ascend-tropical")]
    {
        let tropical_a = Tensor::<f32, Ascend>::from_data_with_backend(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
            ascend.clone(),
        );
        let tropical_b = Tensor::<f32, Ascend>::from_data_with_backend(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[3, 2],
            ascend.clone(),
        );
        let (max_plus, winners) = tropical_a.contract_binary_with_argmax::<MaxPlus<f32>>(
            &tropical_b,
            &[0, 1],
            &[1, 2],
            &[0, 2],
        );
        assert_eq!(max_plus.to_vec(), vec![8.0, 9.0, 11.0, 12.0]);
        assert_eq!(winners.to_vec(), vec![2, 2, 2, 2]);

        let (diagonal_max, diagonal_winner) =
            diagonal.contract_binary_with_argmax::<MaxPlus<f32>>(&diagonal_rhs, &[0, 0], &[0], &[]);
        assert_eq!(diagonal_max.to_vec(), vec![24.0]);
        assert_eq!(diagonal_winner.to_vec(), vec![1]);

        let min_plus =
            tropical_a.contract_binary::<MinPlus<f32>>(&tropical_b, &[0, 1], &[1, 2], &[0, 2]);
        assert_eq!(min_plus.to_vec(), vec![2.0, 3.0, 5.0, 6.0]);

        let max_mul =
            tropical_a.contract_binary::<MaxMul<f32>>(&tropical_b, &[0, 1], &[1, 2], &[0, 2]);
        assert_eq!(max_mul.to_vec(), vec![15.0, 18.0, 30.0, 36.0]);

        // Equal candidates must retain the first contracted index so backward
        // routes the gradient through the same path as the CPU backend.
        let tie_a = Tensor::<f32, Ascend>::from_data_with_backend(
            &[5.0, 1.0],
            &[2],
            tropical_a.backend().clone(),
        );
        let tie_b = Tensor::<f32, Ascend>::from_data_with_backend(
            &[1.0, 5.0],
            &[2],
            tropical_a.backend().clone(),
        );
        let (tie_result, tie_gradient) =
            einsum_with_grad::<MaxPlus<f32>, _, _>(&[&tie_a, &tie_b], &[&[0], &[0]], &[]);
        assert_eq!(tie_result.to_vec(), vec![6.0]);
        let tie_seed = Tensor::<f32, Ascend>::from_data_with_backend(
            &[1.0],
            &[],
            tropical_a.backend().clone(),
        );
        let tie_grads = tie_gradient.backward::<MaxPlus<f32>>(&tie_seed, &[&tie_a, &tie_b]);
        assert_eq!(tie_grads[0].to_vec(), vec![1.0, 0.0]);
        assert_eq!(tie_grads[1].to_vec(), vec![1.0, 0.0]);

        // Match the CPU algebra's NaN rule: a NaN candidate replaces the
        // accumulator, while a later finite candidate can replace that NaN.
        let nan_a = Tensor::<f32, Ascend>::from_data_with_backend(
            &[1.0, f32::NAN],
            &[2],
            tropical_a.backend().clone(),
        );
        let nan_b = Tensor::<f32, Ascend>::from_data_with_backend(
            &[0.0, 0.0],
            &[2],
            tropical_a.backend().clone(),
        );
        let (nan_result, nan_winner) =
            nan_a.contract_binary_with_argmax::<MaxPlus<f32>>(&nan_b, &[0], &[0], &[]);
        assert!(nan_result.to_vec()[0].is_nan());
        assert_eq!(nan_winner.to_vec(), vec![1]);

        // An empty reduction has no winning input. Forward returns the
        // semiring identity and backward must return empty, zero gradients.
        let empty_a = Tensor::<f32, Ascend>::from_data_with_backend(
            &[],
            &[2, 0],
            tropical_a.backend().clone(),
        );
        let empty_b = Tensor::<f32, Ascend>::from_data_with_backend(
            &[],
            &[0, 3],
            tropical_a.backend().clone(),
        );
        let (empty_result, empty_gradient) = einsum_with_grad::<MaxPlus<f32>, _, _>(
            &[&empty_a, &empty_b],
            &[&[0, 1], &[1, 2]],
            &[0, 2],
        );
        assert_eq!(empty_result.to_vec(), vec![f32::MIN; 6]);
        let empty_seed = Tensor::<f32, Ascend>::from_data_with_backend(
            &[1.0; 6],
            &[2, 3],
            tropical_a.backend().clone(),
        );
        let empty_grads =
            empty_gradient.backward::<MaxPlus<f32>>(&empty_seed, &[&empty_a, &empty_b]);
        assert!(empty_grads[0].to_vec().is_empty());
        assert!(empty_grads[1].to_vec().is_empty());
    }

    if std::env::var_os("OMEINSUM_ASCEND_BENCH").is_some() {
        run_standard_benchmark(&ascend);
        #[cfg(feature = "ascend-tropical")]
        run_tropical_benchmark(&ascend);
    }

    println!("Ascend smoke checks passed");
}
