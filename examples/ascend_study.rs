//! Reproducible scaling study for the experimental single-NPU Ascend backend.
//!
//! This intentionally reports both contraction-only (no result download) and
//! end-to-end latency. Set `OMEINSUM_ASCEND_STUDY=quick` for a short validation
//! run or `OMEINSUM_ASCEND_STUDY=full` for the documented study matrix.

use omeinsum::{Ascend, Cpu, MaxPlus, Standard, Tensor};
use std::{hint::black_box, time::Instant};

#[path = "ascend_study/chains.rs"]
mod chains;
use chains::{chain_case, tropical_chain_case};

fn data(len: usize, period: usize) -> Vec<f32> {
    (0..len)
        .map(|index| (index % period) as f32 / period as f32 - 0.5)
        .collect()
}

fn iterations(operations: f64, tropical: bool) -> usize {
    let threshold = if tropical { 2.0e7 } else { 5.0e8 };
    if operations <= threshold {
        5
    } else if operations <= threshold * 8.0 {
        3
    } else {
        1
    }
}

#[allow(clippy::too_many_arguments)]
fn measure<F>(
    category: &str,
    name: &str,
    backend: &str,
    scope: &str,
    m: usize,
    k: usize,
    n: usize,
    batch: usize,
    operations: f64,
    iterations: usize,
    mut run: F,
) where
    F: FnMut(),
{
    run();
    let start = Instant::now();
    for _ in 0..iterations {
        run();
    }
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "RESULT,{category},{name},{backend},{scope},{m},{k},{n},{batch},{iterations},{:.6},{:.6}",
        elapsed * 1_000.0 / iterations as f64,
        operations * iterations as f64 / elapsed / 1.0e9,
    );
}

fn max_abs_error(expected: &[f32], actual: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len(), "result length mismatch");
    assert!(
        expected.iter().chain(actual).all(|value| value.is_finite()),
        "accuracy comparison contains a non-finite value"
    );
    expected
        .iter()
        .zip(actual)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f32::max)
}

fn standard_case(ascend: &Ascend, category: &str, name: &str, m: usize, k: usize, n: usize) {
    let a_data = data(m * k, 17);
    let b_data = data(k * n, 13);
    let cpu_a = Tensor::<f32, Cpu>::from_data(&a_data, &[m, k]);
    let cpu_b = Tensor::<f32, Cpu>::from_data(&b_data, &[k, n]);
    let npu_a = Tensor::<f32, Ascend>::from_data_with_backend(&a_data, &[m, k], ascend.clone());
    let npu_b = Tensor::<f32, Ascend>::from_data_with_backend(&b_data, &[k, n], ascend.clone());
    let expected = cpu_a
        .contract_binary::<Standard<f32>>(&cpu_b, &[0, 1], &[1, 2], &[0, 2])
        .to_vec();
    let actual = npu_a
        .contract_binary::<Standard<f32>>(&npu_b, &[0, 1], &[1, 2], &[0, 2])
        .to_vec();
    let error = max_abs_error(&expected, &actual);
    assert!(error <= 1.0e-3, "{name} max absolute error {error}");
    println!("ACCURACY,{category},{name},standard,max_abs_error,{error:.9}");

    let operations = 2.0 * (m * k * n) as f64;
    let count = iterations(operations, false);
    measure(
        category,
        name,
        "cpu",
        "contraction",
        m,
        k,
        n,
        1,
        operations,
        count,
        || {
            black_box(cpu_a.contract_binary::<Standard<f32>>(&cpu_b, &[0, 1], &[1, 2], &[0, 2]));
        },
    );
    measure(
        category,
        name,
        "ascend",
        "contraction",
        m,
        k,
        n,
        1,
        operations,
        count,
        || {
            black_box(npu_a.contract_binary::<Standard<f32>>(&npu_b, &[0, 1], &[1, 2], &[0, 2]));
        },
    );
    measure(
        category,
        name,
        "cpu",
        "result_download",
        m,
        k,
        n,
        1,
        operations,
        count,
        || {
            black_box(
                cpu_a
                    .contract_binary::<Standard<f32>>(&cpu_b, &[0, 1], &[1, 2], &[0, 2])
                    .to_vec()[0],
            );
        },
    );
    measure(
        category,
        name,
        "ascend",
        "result_download",
        m,
        k,
        n,
        1,
        operations,
        count,
        || {
            black_box(
                npu_a
                    .contract_binary::<Standard<f32>>(&npu_b, &[0, 1], &[1, 2], &[0, 2])
                    .to_vec()[0],
            );
        },
    );
    measure(
        category,
        name,
        "cpu",
        "host_to_host",
        m,
        k,
        n,
        1,
        operations,
        count,
        || {
            let a = Tensor::<f32, Cpu>::from_data(&a_data, &[m, k]);
            let b = Tensor::<f32, Cpu>::from_data(&b_data, &[k, n]);
            black_box(
                a.contract_binary::<Standard<f32>>(&b, &[0, 1], &[1, 2], &[0, 2])
                    .to_vec()[0],
            );
        },
    );
    measure(
        category,
        name,
        "ascend",
        "host_to_host",
        m,
        k,
        n,
        1,
        operations,
        count,
        || {
            let a = Tensor::<f32, Ascend>::from_data_with_backend(&a_data, &[m, k], ascend.clone());
            let b = Tensor::<f32, Ascend>::from_data_with_backend(&b_data, &[k, n], ascend.clone());
            black_box(
                a.contract_binary::<Standard<f32>>(&b, &[0, 1], &[1, 2], &[0, 2])
                    .to_vec()[0],
            );
        },
    );
}

fn tropical_case(ascend: &Ascend, name: &str, m: usize, k: usize, n: usize) {
    let a_data = data(m * k, 17);
    let b_data = data(k * n, 13);
    let cpu_a = Tensor::<f32, Cpu>::from_data(&a_data, &[m, k]);
    let cpu_b = Tensor::<f32, Cpu>::from_data(&b_data, &[k, n]);
    let npu_a = Tensor::<f32, Ascend>::from_data_with_backend(&a_data, &[m, k], ascend.clone());
    let npu_b = Tensor::<f32, Ascend>::from_data_with_backend(&b_data, &[k, n], ascend.clone());
    let expected = cpu_a
        .contract_binary::<MaxPlus<f32>>(&cpu_b, &[0, 1], &[1, 2], &[0, 2])
        .to_vec();
    let actual = npu_a
        .contract_binary::<MaxPlus<f32>>(&npu_b, &[0, 1], &[1, 2], &[0, 2])
        .to_vec();
    assert_eq!(actual, expected, "{name} must exactly match CPU max-plus");
    println!("ACCURACY,scaling,{name},maxplus,exact_match,true");

    let operations = 2.0 * (m * k * n) as f64;
    let count = iterations(operations, true);
    measure(
        "scaling",
        name,
        "cpu",
        "contraction",
        m,
        k,
        n,
        1,
        operations,
        count,
        || {
            black_box(cpu_a.contract_binary::<MaxPlus<f32>>(&cpu_b, &[0, 1], &[1, 2], &[0, 2]));
        },
    );
    measure(
        "scaling",
        name,
        "ascend",
        "contraction",
        m,
        k,
        n,
        1,
        operations,
        count,
        || {
            black_box(npu_a.contract_binary::<MaxPlus<f32>>(&npu_b, &[0, 1], &[1, 2], &[0, 2]));
        },
    );
    measure(
        "scaling",
        name,
        "cpu",
        "result_download",
        m,
        k,
        n,
        1,
        operations,
        count,
        || {
            black_box(
                cpu_a
                    .contract_binary::<MaxPlus<f32>>(&cpu_b, &[0, 1], &[1, 2], &[0, 2])
                    .to_vec()[0],
            );
        },
    );
    measure(
        "scaling",
        name,
        "ascend",
        "result_download",
        m,
        k,
        n,
        1,
        operations,
        count,
        || {
            black_box(
                npu_a
                    .contract_binary::<MaxPlus<f32>>(&npu_b, &[0, 1], &[1, 2], &[0, 2])
                    .to_vec()[0],
            );
        },
    );
}

fn batched_case(ascend: &Ascend, batch: usize, n: usize, batch_first: bool) {
    let len = batch * n * n;
    let a_data = data(len, 17);
    let b_data = data(len, 13);
    let (shape_a, modes_a, shape_b, modes_b, shape_c, modes_c, name) = if batch_first {
        (
            vec![batch, n, n],
            vec![0, 1, 2],
            vec![batch, n, n],
            vec![0, 2, 3],
            vec![batch, n, n],
            vec![0, 1, 3],
            "batch_first",
        )
    } else {
        (
            vec![n, n, batch],
            vec![1, 2, 0],
            vec![n, n, batch],
            vec![2, 3, 0],
            vec![n, n, batch],
            vec![1, 3, 0],
            "batch_last_canonical",
        )
    };
    let cpu_a = Tensor::<f32, Cpu>::from_data(&a_data, &shape_a);
    let cpu_b = Tensor::<f32, Cpu>::from_data(&b_data, &shape_b);
    let npu_a = Tensor::<f32, Ascend>::from_data_with_backend(&a_data, &shape_a, ascend.clone());
    let npu_b = Tensor::<f32, Ascend>::from_data_with_backend(&b_data, &shape_b, ascend.clone());
    let expected = cpu_a
        .contract_binary::<Standard<f32>>(&cpu_b, &modes_a, &modes_b, &modes_c)
        .to_vec();
    let actual = npu_a
        .contract_binary::<Standard<f32>>(&npu_b, &modes_a, &modes_b, &modes_c)
        .to_vec();
    let error = max_abs_error(&expected, &actual);
    assert!(error <= 1.0e-3, "{name} max absolute error {error}");
    println!("ACCURACY,batched,{name},standard,max_abs_error,{error:.9}");
    assert_eq!(actual.len(), shape_c.iter().product::<usize>());

    let operations = 2.0 * (batch * n * n * n) as f64;
    let count = iterations(operations, false);
    measure(
        "batched",
        name,
        "cpu",
        "result_download",
        n,
        n,
        n,
        batch,
        operations,
        count,
        || {
            black_box(
                cpu_a
                    .contract_binary::<Standard<f32>>(&cpu_b, &modes_a, &modes_b, &modes_c)
                    .to_vec()[0],
            );
        },
    );
    measure(
        "batched",
        name,
        "ascend",
        "result_download",
        n,
        n,
        n,
        batch,
        operations,
        count,
        || {
            black_box(
                npu_a
                    .contract_binary::<Standard<f32>>(&npu_b, &modes_a, &modes_b, &modes_c)
                    .to_vec()[0],
            );
        },
    );
}

fn layout_case(ascend: &Ascend, n: usize) {
    let a_data = data(n * n, 17);
    let b_data = data(n * n, 13);
    let cpu_a = Tensor::<f32, Cpu>::from_data(&a_data, &[n, n]).permute(&[1, 0]);
    let cpu_b = Tensor::<f32, Cpu>::from_data(&b_data, &[n, n]);
    let npu_a = Tensor::<f32, Ascend>::from_data_with_backend(&a_data, &[n, n], ascend.clone())
        .permute(&[1, 0]);
    let npu_b = Tensor::<f32, Ascend>::from_data_with_backend(&b_data, &[n, n], ascend.clone());
    let expected = cpu_a
        .contract_binary::<Standard<f32>>(&cpu_b, &[1, 0], &[0, 2], &[2, 1])
        .to_vec();
    let actual = npu_a
        .contract_binary::<Standard<f32>>(&npu_b, &[1, 0], &[0, 2], &[2, 1])
        .to_vec();
    let error = max_abs_error(&expected, &actual);
    assert!(
        error <= 1.0e-3,
        "permuted layout max absolute error {error}"
    );
    println!("ACCURACY,layout,permuted_both,standard,max_abs_error,{error:.9}");

    let operations = 2.0 * (n * n * n) as f64;
    let count = iterations(operations, false);
    measure(
        "layout",
        "permuted_both",
        "cpu",
        "result_download",
        n,
        n,
        n,
        1,
        operations,
        count,
        || {
            black_box(
                cpu_a
                    .contract_binary::<Standard<f32>>(&cpu_b, &[1, 0], &[0, 2], &[2, 1])
                    .to_vec()[0],
            );
        },
    );
    measure(
        "layout",
        "permuted_both",
        "ascend",
        "result_download",
        n,
        n,
        n,
        1,
        operations,
        count,
        || {
            black_box(
                npu_a
                    .contract_binary::<Standard<f32>>(&npu_b, &[1, 0], &[0, 2], &[2, 1])
                    .to_vec()[0],
            );
        },
    );
}

fn main() {
    let profile = std::env::var("OMEINSUM_ASCEND_STUDY").unwrap_or_else(|_| "quick".into());
    assert!(
        profile == "quick" || profile == "full",
        "study profile must be quick or full"
    );
    let ascend = Ascend::new().expect("initialize Ascend device 0");
    println!("STUDY,profile,{profile}");
    println!("HEADER,category,name,backend,scope,m,k,n,batch,iterations,avg_ms,gops");

    let standard_sizes: &[usize] = if profile == "full" {
        &[64, 128, 256, 512, 1024, 2048]
    } else {
        &[64, 128]
    };
    for &n in standard_sizes {
        standard_case(&ascend, "scaling", &format!("standard_square_{n}"), n, n, n);
    }

    let tropical_sizes: &[usize] = if profile == "full" {
        &[32, 64, 128, 256, 512]
    } else {
        &[32, 64]
    };
    for &n in tropical_sizes {
        tropical_case(&ascend, &format!("maxplus_square_{n}"), n, n, n);
    }

    if profile == "full" {
        standard_case(&ascend, "rectangular", "skinny_16x1024x16", 16, 1024, 16);
        standard_case(&ascend, "rectangular", "skinny_64x1024x64", 64, 1024, 64);
        standard_case(&ascend, "rectangular", "wide_256x64x256", 256, 64, 256);
        for batch in [1, 8, 32, 128] {
            batched_case(&ascend, batch, 64, false);
        }
        batched_case(&ascend, 8, 64, true);
        layout_case(&ascend, 256);
        chain_case(&ascend, 128, 20);
        tropical_chain_case(&ascend, 64, 20);
    }

    println!("STUDY,complete,true");
}
