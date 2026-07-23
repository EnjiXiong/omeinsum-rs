//! Convert a yao-tn-v1 complex network into one backend-neutral real-f32 artifact.

#[path = "support/yao_benchmark.rs"]
mod format;

use format::{
    BenchmarkEinCode, BenchmarkNetwork, BenchmarkTensor, Complexity, TreeNode, YaoNetwork,
};
use num_complex::Complex32;
use omeco::{contraction_complexity, optimize_code, TreeSA};
use omeinsum::realify::constants;
use omeinsum::{realify_code, realify_data};
use std::{collections::HashMap, fs::File, io::BufReader};

fn parse_label(label: &str) -> usize {
    label
        .parse()
        .unwrap_or_else(|error| panic!("label '{label}' is not a non-negative integer: {error}"))
}

fn main() {
    let mut args = std::env::args().skip(1);
    let input = args
        .next()
        .expect("usage: prepare_yao_benchmark INPUT OUTPUT");
    let output = args
        .next()
        .expect("usage: prepare_yao_benchmark INPUT OUTPUT");
    assert!(args.next().is_none(), "unexpected extra argument");

    let source: YaoNetwork = serde_json::from_reader(BufReader::new(
        File::open(&input).unwrap_or_else(|error| panic!("failed to open {input}: {error}")),
    ))
    .unwrap_or_else(|error| panic!("invalid yao network {input}: {error}"));
    assert_eq!(source.format, "yao-tn-v1", "unsupported source format");
    assert_eq!(
        source.eincode.input_indices.len(),
        source.tensors.len(),
        "one tensor is required for every input index list"
    );

    let input_indices: Vec<Vec<usize>> = source
        .eincode
        .input_indices
        .iter()
        .map(|indices| indices.iter().map(|label| parse_label(label)).collect())
        .collect();
    let output_indices = source
        .eincode
        .output_indices
        .iter()
        .map(|label| parse_label(label))
        .collect::<Vec<_>>();
    let size_dict = source
        .size_dict
        .iter()
        .map(|(label, &size)| (parse_label(label), size))
        .collect::<HashMap<_, _>>();

    let is_complex = source
        .tensors
        .iter()
        .map(|tensor| tensor.data_im.iter().any(|value| *value != 0.0))
        .collect::<Vec<_>>();
    let plan = realify_code(&input_indices, &output_indices, &size_dict, &is_complex);
    let complex_inputs = is_complex.iter().filter(|&&value| value).count();

    let mut tensors = source
        .tensors
        .iter()
        .zip(&is_complex)
        .map(|(tensor, &complex)| {
            let expected_len = tensor.shape.iter().product::<usize>();
            assert_eq!(
                tensor.data_re.len(),
                expected_len,
                "invalid real data length"
            );
            assert_eq!(
                tensor.data_im.len(),
                expected_len,
                "invalid imaginary data length"
            );
            if complex {
                let data = tensor
                    .data_re
                    .iter()
                    .zip(&tensor.data_im)
                    .map(|(&re, &im)| Complex32::new(re as f32, im as f32))
                    .collect::<Vec<_>>();
                let mut shape = tensor.shape.clone();
                shape.push(2);
                BenchmarkTensor {
                    shape,
                    data: realify_data(&data, false),
                }
            } else {
                BenchmarkTensor {
                    shape: tensor.shape.clone(),
                    data: tensor.data_re.iter().map(|value| *value as f32).collect(),
                }
            }
        })
        .collect::<Vec<_>>();
    tensors.extend((0..plan.num_mul_vertices).map(|_| BenchmarkTensor {
        shape: vec![2, 2, 2],
        data: constants::M_DATA.map(|value| value as f32).to_vec(),
    }));

    // TreeSA seeds trial 0 with 42, so this one-trial configuration is reproducible.
    // The 2^28-element target leaves headroom beyond the largest intermediate on
    // both the 64 GiB Ascend device and 80 GiB A800.
    let optimizer = TreeSA::fast().with_sc_target(28.0);
    let code = plan.einsum.code();
    let tree = optimize_code(&code, &plan.einsum.size_dict, &optimizer)
        .expect("deterministic TreeSA optimizer produced no contraction tree");
    let complexity = contraction_complexity(&tree, &plan.einsum.size_dict, &plan.einsum.ixs);
    let artifact = BenchmarkNetwork {
        format: "omeinsum-real-f32-benchmark-v1".to_string(),
        source_format: source.format,
        source_mode: source.mode,
        optimizer: "omeco-treesa-fast-ntrials1-niters20-seed42-sc-target28".to_string(),
        realification: format!(
            "omeinsum-realify-v1;complex_inputs={complex_inputs};multiplication_vertices={}",
            plan.num_mul_vertices
        ),
        eincode: BenchmarkEinCode {
            input_indices: plan.einsum.ixs.clone(),
            output_indices: plan.einsum.iy.clone(),
        },
        tensors,
        size_dict: plan.einsum.size_dict,
        contraction_order: TreeNode::from_nested(&tree),
        complexity: Complexity {
            log2_flops: complexity.tc,
            log2_peak_elements: complexity.sc,
            log2_readwrites: complexity.rwc,
            estimated_peak_bytes_f32: 2_f64.powf(complexity.sc) * 4.0,
        },
    };
    assert_eq!(
        artifact.tensors.len(),
        artifact.eincode.input_indices.len(),
        "realified tensor/index count mismatch"
    );

    serde_json::to_writer(
        File::create(&output).unwrap_or_else(|error| panic!("failed to create {output}: {error}")),
        &artifact,
    )
    .unwrap_or_else(|error| panic!("failed to write {output}: {error}"));
    println!(
        "prepared={} tensors={} complex_inputs={} log2_flops={:.6} log2_peak_elements={:.6} estimated_peak_bytes_f32={:.0}",
        output,
        artifact.tensors.len(),
        complex_inputs,
        artifact.complexity.log2_flops,
        artifact.complexity.log2_peak_elements,
        artifact.complexity.estimated_peak_bytes_f32
    );
}
