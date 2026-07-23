//! Matched real-f32 tensor-network benchmark for one CUDA GPU or Ascend NPU.

#[cfg(all(feature = "cuda", feature = "ascend"))]
compile_error!("network_benchmark requires exactly one of `cuda` or `ascend`");

#[path = "support/yao_benchmark.rs"]
mod format;

use format::BenchmarkNetwork;
use omeinsum::{Backend, Cpu, Standard};
use std::{fs::File, hint::black_box, io::BufReader, time::Instant};

#[cfg(all(feature = "ascend", not(feature = "cuda")))]
use omeinsum::Ascend as Device;
#[cfg(not(any(feature = "cuda", feature = "ascend")))]
use omeinsum::Cpu as Device;
#[cfg(all(feature = "cuda", not(feature = "ascend")))]
use omeinsum::Cuda as Device;

struct Args {
    path: String,
    warmup: usize,
    repeats: usize,
    check_cpu: bool,
}

fn parse_args() -> Args {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .unwrap_or_else(|| "benches/network_small.json".to_string());
    let mut warmup = 3;
    let mut repeats = 20;
    let mut check_cpu = false;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--check-cpu" => check_cpu = true,
            "--warmup" | "--repeats" => {
                let value = args.next().expect("flag requires a value");
                match flag.as_str() {
                    "--warmup" => warmup = value.parse().expect("--warmup requires an integer"),
                    "--repeats" => repeats = value.parse().expect("--repeats requires an integer"),
                    _ => unreachable!(),
                }
            }
            _ => panic!("unknown flag {flag}"),
        }
    }
    assert!(repeats > 0, "--repeats must be positive");
    Args {
        path,
        warmup,
        repeats,
        check_cpu,
    }
}

#[cfg(any(feature = "cuda", feature = "ascend"))]
fn device() -> Device {
    Device::new().expect("failed to initialize accelerator device 0")
}

#[cfg(not(any(feature = "cuda", feature = "ascend")))]
fn device() -> Device {
    Cpu
}

fn max_abs_error(expected: &[f32], actual: &[f32]) -> f32 {
    assert_eq!(
        expected.len(),
        actual.len(),
        "CPU/device result length mismatch"
    );
    expected
        .iter()
        .zip(actual)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max)
}

fn main() {
    let args = parse_args();
    let network: BenchmarkNetwork = serde_json::from_reader(BufReader::new(
        File::open(&args.path)
            .unwrap_or_else(|error| panic!("failed to open {}: {error}", args.path)),
    ))
    .expect("invalid benchmark network JSON");
    assert_eq!(network.format, "omeinsum-real-f32-benchmark-v1");
    let einsum = network.einsum();

    let expected = args.check_cpu.then(|| {
        let tensors = network.tensors(Cpu);
        let refs = tensors.iter().collect::<Vec<_>>();
        einsum.execute::<Standard<f32>, f32, Cpu>(&refs).to_vec()
    });

    let device = device();
    let tensors = network.tensors(device.clone());
    let refs = tensors.iter().collect::<Vec<_>>();
    let initial = einsum.execute::<Standard<f32>, f32, Device>(&refs);
    device.synchronize();
    let initial_values = initial.to_vec();
    let error = expected
        .as_ref()
        .map(|values| max_abs_error(values, &initial_values));

    for _ in 0..args.warmup {
        let result = einsum.execute::<Standard<f32>, f32, Device>(&refs);
        device.synchronize();
        black_box(result);
    }

    let mut times_ms = Vec::with_capacity(args.repeats);
    for _ in 0..args.repeats {
        let start = Instant::now();
        let result = einsum.execute::<Standard<f32>, f32, Device>(&refs);
        device.synchronize();
        times_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
        black_box(result);
    }
    times_ms.sort_by(f64::total_cmp);
    let mean = times_ms.iter().sum::<f64>() / args.repeats as f64;
    let median = times_ms[times_ms.len() / 2];
    let checksum = initial_values
        .iter()
        .enumerate()
        .map(|(index, value)| (index + 1) as f64 * *value as f64)
        .sum::<f64>();
    assert!(
        initial_values.iter().all(|value| value.is_finite()),
        "device result contains a non-finite value"
    );
    println!(
        "backend={} network={} tensors={} dtype=f32 scope=contraction warmup={} repeats={} cpu_check={} max_abs_error={}",
        Device::name(),
        args.path,
        network.tensors.len(),
        args.warmup,
        args.repeats,
        args.check_cpu,
        error.map_or_else(|| "not-run".to_string(), |value| format!("{value:.9}"))
    );
    println!(
        "complexity log2_flops={:.6} log2_peak_elements={:.6} estimated_peak_bytes_f32={:.0}",
        network.complexity.log2_flops,
        network.complexity.log2_peak_elements,
        network.complexity.estimated_peak_bytes_f32
    );
    println!(
        "result elements={} checksum={checksum:.12e}",
        initial_values.len()
    );
    println!(
        "result_values={}",
        initial_values
            .iter()
            .map(|value| format!("{value:.12e}"))
            .collect::<Vec<_>>()
            .join(",")
    );
    println!(
        "wall_clock_ms mean={mean:.6} median={median:.6} min={:.6} max={:.6}",
        times_ms[0],
        times_ms[times_ms.len() - 1]
    );
    println!(
        "samples_ms={}",
        times_ms
            .iter()
            .map(|value| format!("{value:.6}"))
            .collect::<Vec<_>>()
            .join(",")
    );
}
