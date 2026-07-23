//! Matched standard-f32 tensor-network benchmark for one CUDA GPU or Ascend NPU.
//!
//! Build exactly one backend at a time:
//! `cargo build --release --features cuda --example network_benchmark`
//! `cargo build --release --features ascend --example network_benchmark`

#[cfg(all(feature = "cuda", feature = "ascend"))]
compile_error!("network_benchmark requires exactly one of `cuda` or `ascend`");
#[cfg(not(any(feature = "cuda", feature = "ascend")))]
compile_error!("network_benchmark requires exactly one of `cuda` or `ascend`");

use omeco::{EinCode, NestedEinsum};
use omeinsum::{Backend, BackendScalar, Cpu, Einsum, Standard, Tensor};
use serde::Deserialize;
use std::{collections::HashMap, fs::File, hint::black_box, io::BufReader, time::Instant};

#[cfg(feature = "ascend")]
use omeinsum::Ascend as Device;
#[cfg(feature = "cuda")]
use omeinsum::Cuda as Device;

#[derive(Deserialize)]
struct NetworkJson {
    #[serde(rename = "n_vertices")]
    _n_vertices: usize,
    bond_dim: usize,
    edges: Vec<(usize, usize)>,
    tree: TreeNodeJson,
}

#[derive(Deserialize)]
struct TreeNodeJson {
    #[serde(rename = "isleaf")]
    is_leaf: bool,
    tensorindex: Option<usize>,
    eins: Option<EinsJson>,
    args: Option<Vec<TreeNodeJson>>,
}

#[derive(Deserialize)]
struct EinsJson {
    ixs: Vec<Vec<usize>>,
    iy: Vec<usize>,
}

fn nested(node: &TreeNodeJson) -> NestedEinsum<usize> {
    if node.is_leaf {
        NestedEinsum::leaf(node.tensorindex.expect("leaf missing tensor index"))
    } else {
        let eins = node.eins.as_ref().expect("internal node missing einsum");
        NestedEinsum::node(
            node.args
                .as_ref()
                .expect("internal node missing arguments")
                .iter()
                .map(nested)
                .collect(),
            EinCode::new(eins.ixs.clone(), eins.iy.clone()),
        )
    }
}

fn parse_args() -> (String, usize, usize) {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .unwrap_or_else(|| "benches/network_small.json".to_string());
    let mut warmup = 3;
    let mut repeats = 20;
    while let Some(flag) = args.next() {
        let value = args.next().expect("flag requires a value");
        match flag.as_str() {
            "--warmup" => warmup = value.parse().expect("--warmup requires an integer"),
            "--repeats" => repeats = value.parse().expect("--repeats requires an integer"),
            _ => panic!("unknown flag {flag}"),
        }
    }
    assert!(repeats > 0, "--repeats must be positive");
    (path, warmup, repeats)
}

fn build_einsum(network: &NetworkJson) -> Einsum<usize> {
    let ixs = network.edges.iter().map(|&(u, v)| vec![u, v]).collect();
    let sizes = network
        .edges
        .iter()
        .flat_map(|&(u, v)| [(u, network.bond_dim), (v, network.bond_dim)])
        .collect::<HashMap<_, _>>();
    let mut einsum = Einsum::new(ixs, vec![], sizes);
    einsum.set_contraction_tree(nested(&network.tree));
    einsum
}

fn tensors<B>(network: &NetworkJson, backend: B) -> Vec<Tensor<f32, B>>
where
    B: Backend + Clone,
    f32: BackendScalar<B>,
{
    let shape = [network.bond_dim, network.bond_dim];
    let data: Vec<f32> = (0..shape.iter().product())
        .map(|index| (index + 1) as f32 / 8.0)
        .collect();
    network
        .edges
        .iter()
        .map(|_| Tensor::from_data_with_backend(&data, &shape, backend.clone()))
        .collect()
}

fn main() {
    let (path, warmup, repeats) = parse_args();
    let network: NetworkJson = serde_json::from_reader(BufReader::new(
        File::open(&path).unwrap_or_else(|error| panic!("failed to open {path}: {error}")),
    ))
    .expect("invalid network JSON");
    let einsum = build_einsum(&network);

    let cpu_tensors = tensors(&network, Cpu);
    let cpu_refs = cpu_tensors.iter().collect::<Vec<_>>();
    let expected = einsum
        .execute::<Standard<f32>, f32, Cpu>(&cpu_refs)
        .to_vec();

    let device = Device::new().expect("failed to initialize accelerator device 0");
    let device_tensors = tensors(&network, device.clone());
    let device_refs = device_tensors.iter().collect::<Vec<_>>();
    let actual = einsum
        .execute::<Standard<f32>, f32, Device>(&device_refs)
        .to_vec();
    let max_abs_error = expected
        .iter()
        .zip(&actual)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs_error <= 1.0e-3,
        "max absolute error {max_abs_error}"
    );

    for _ in 0..warmup {
        let result = einsum.execute::<Standard<f32>, f32, Device>(&device_refs);
        device.synchronize();
        black_box(result);
    }

    let mut times_ms = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = Instant::now();
        let result = einsum.execute::<Standard<f32>, f32, Device>(&device_refs);
        device.synchronize();
        times_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
        black_box(result);
    }
    times_ms.sort_by(f64::total_cmp);
    let mean = times_ms.iter().sum::<f64>() / repeats as f64;
    let median = times_ms[times_ms.len() / 2];
    println!(
        "backend={} network={} tensors={} bond_dim={} dtype=f32 scope=contraction warmup={} repeats={} max_abs_error={:.9}",
        Device::name(), path, network.edges.len(), network.bond_dim, warmup, repeats, max_abs_error
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
