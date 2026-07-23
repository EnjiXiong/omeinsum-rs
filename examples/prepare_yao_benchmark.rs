//! Archive an optimized and sliced original complex Yao network.
#[path = "support/yao_benchmark.rs"]
mod format;
use format::*;
use omeco::{contraction_complexity, optimize_code, slice_code, EinCode, TreeSA, TreeSASlicer};
use std::{collections::HashMap, fs::File, io::BufReader};

struct Args {
    input: String,
    output: String,
    sc: f64,
    max: usize,
    opt_profile: String,
    opt_trials: usize,
    opt_iters: usize,
    slice_trials: usize,
    slice_iters: usize,
}
fn args() -> Args {
    let mut a = std::env::args().skip(1);
    let input = a
        .next()
        .expect("usage: prepare_yao_benchmark INPUT OUTPUT [options]");
    let output = a.next().expect("missing OUTPUT");
    let mut x = Args {
        input,
        output,
        sc: 28.0,
        max: 1 << 24,
        opt_profile: "fast".into(),
        opt_trials: 1,
        opt_iters: 20,
        slice_trials: 1,
        slice_iters: 5,
    };
    while let Some(f) = a.next() {
        let v = a.next().unwrap_or_else(|| panic!("{f} requires a value"));
        match f.as_str() {
            "--sc-target" => x.sc = v.parse().unwrap(),
            "--max-assignments" => x.max = v.parse().unwrap(),
            "--optimizer-profile" => x.opt_profile = v,
            "--optimizer-trials" => x.opt_trials = v.parse().unwrap(),
            "--optimizer-iters" => x.opt_iters = v.parse().unwrap(),
            "--slicer-trials" => x.slice_trials = v.parse().unwrap(),
            "--slicer-iters" => x.slice_iters = v.parse().unwrap(),
            _ => panic!("unknown option {f}"),
        }
    }
    x
}
fn label(x: &str) -> usize {
    x.parse()
        .unwrap_or_else(|e| panic!("invalid label {x}: {e}"))
}
fn column_major<T: Copy>(data: &[T], shape: &[usize]) -> Vec<T> {
    assert_eq!(data.len(), shape.iter().product::<usize>());
    (0..data.len())
        .map(|mut ci| {
            let mut c = Vec::new();
            for &n in shape {
                c.push(ci % n);
                ci /= n;
            }
            c.iter().zip(shape).fold(0, |i, (&v, &n)| i * n + v)
        })
        .map(|i| data[i])
        .collect()
}
fn metric(x: omeco::ContractionComplexity) -> Complexity {
    Complexity {
        log2_flops: x.tc,
        log2_peak_elements: x.sc,
        log2_readwrites: x.rwc,
    }
}

fn main() {
    let a = args();
    let source: YaoNetwork = serde_json::from_reader(BufReader::new(File::open(&a.input).unwrap()))
        .expect("invalid Yao JSON");
    assert_eq!(source.format, "yao-tn-v1");
    let ixs = source
        .eincode
        .input_indices
        .iter()
        .map(|x| x.iter().map(|s| label(s)).collect())
        .collect::<Vec<Vec<_>>>();
    let iy = source
        .eincode
        .output_indices
        .iter()
        .map(|s| label(s))
        .collect::<Vec<_>>();
    let sizes = source
        .size_dict
        .iter()
        .map(|(k, v)| (label(k), *v))
        .collect::<HashMap<_, _>>();
    assert_eq!(ixs.len(), source.tensors.len());
    let mut tensors = Vec::new();
    for (i, t) in source.tensors.iter().enumerate() {
        let n = t.shape.iter().product::<usize>();
        assert_eq!(t.shape, ixs[i].iter().map(|l| sizes[l]).collect::<Vec<_>>());
        assert_eq!(t.data_re.len(), n);
        assert_eq!(t.data_im.len(), n);
        tensors.push(BenchmarkTensor {
            shape: t.shape.clone(),
            data_re: column_major(
                &t.data_re.iter().map(|x| *x as f32).collect::<Vec<_>>(),
                &t.shape,
            ),
            data_im: column_major(
                &t.data_im.iter().map(|x| *x as f32).collect::<Vec<_>>(),
                &t.shape,
            ),
            structurally_complex: t.data_im.iter().any(|x| *x != 0.0),
        });
    }
    let code = EinCode::new(ixs.clone(), iy.clone());
    let optimizer = match a.opt_profile.as_str() {
        "default" => TreeSA::default(),
        "fast" => TreeSA::fast(),
        profile => panic!("unknown optimizer profile {profile}"),
    }
    .with_ntrials(a.opt_trials)
    .with_niters(a.opt_iters)
    .with_sc_target(a.sc);
    let original = optimize_code(&code, &sizes, &optimizer).expect("TreeSA produced no tree");
    let unsliced = contraction_complexity(&original, &sizes, &ixs);
    let slicer = TreeSASlicer::fast()
        .with_ntrials(a.slice_trials)
        .with_niters(a.slice_iters)
        .with_sc_target(a.sc);
    let sliced = slice_code(&original, &sizes, &slicer, &ixs).expect("TreeSASlicer failed");
    let mut cuts = sliced.slicing;
    cuts.sort_unstable();
    cuts.dedup();
    for c in &cuts {
        assert!(sizes.contains_key(c), "invalid physical cut {c}");
        assert!(!iy.contains(c), "cannot cut output label {c}");
    }
    let assignments = cuts
        .iter()
        .try_fold(1usize, |n, c| n.checked_mul(sizes[c]))
        .expect("assignment count overflow");
    assert!(
        assignments <= a.max,
        "assignment cap exceeded: {assignments} > {}",
        a.max
    );
    let tree = TreeNode::from_nested(&sliced.eins);
    let leaves = tree.validate(tensors.len()).expect("invalid tree");
    assert_eq!(
        leaves.len(),
        tensors.len(),
        "tree does not contain every tensor"
    );
    let adjusted = cut_sizes(&sizes, &cuts);
    let per = contraction_complexity(&sliced.eins, &adjusted, &ixs);
    let total = Complexity {
        log2_flops: per.tc + (assignments as f64).log2(),
        log2_peak_elements: per.sc,
        log2_readwrites: per.rwc + (assignments as f64).log2(),
    };
    let mut audit = audit_tree(
        &tree,
        &tensors
            .iter()
            .map(|t| t.structurally_complex)
            .collect::<Vec<_>>(),
        &adjusted,
        per.sc,
    );
    audit.native_resident_input_cache_elements = tensors
        .iter()
        .map(|t| t.shape.iter().product::<usize>())
        .sum();
    audit.native_resident_input_cache_bytes = audit.native_resident_input_cache_elements * 8;
    audit.tree_real_resident_input_cache_elements = tensors
        .iter()
        .map(|t| t.shape.iter().product::<usize>() * (1 + usize::from(t.structurally_complex)))
        .sum();
    audit.tree_real_resident_input_cache_bytes = audit.tree_real_resident_input_cache_elements * 4;
    let artifact = BenchmarkNetwork {
        format: "omeinsum-yao-benchmark-v2".into(),
        source_format: source.format,
        source_mode: source.mode,
        optimizer: OptimizerConfig {
            algorithm: "omeco::TreeSA".into(),
            version: "0.2.6".into(),
            ntrials: a.opt_trials,
            niters: a.opt_iters,
            sc_target: a.sc,
            seed_base: 42,
        },
        slicer: SlicerConfig {
            algorithm: "omeco::TreeSASlicer".into(),
            ntrials: a.slice_trials,
            niters: a.slice_iters,
            sc_target: a.sc,
            optimization_ratio: 1.0,
            seed_base: 42,
        },
        eincode: BenchmarkEinCode {
            input_indices: ixs,
            output_indices: iy,
        },
        tensors,
        size_dict: sizes,
        contraction_order: tree,
        cuts,
        assignment_count: assignments,
        complexity: ComplexityReport {
            physical_unsliced: metric(unsliced),
            physical_per_slice: metric(per),
            physical_total: total,
        },
        tree_real_audit: audit,
    };
    artifact.validate().expect("prepared artifact is invalid");
    serde_json::to_writer(File::create(&a.output).unwrap(), &artifact).unwrap();
    println!(
        "prepared={} cuts={} assignments={}",
        a.output,
        artifact
            .cuts
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(","),
        assignments
    );
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn layout() {
        assert_eq!(
            column_major(&[0, 1, 2, 3, 4, 5], &[2, 3]),
            vec![0, 3, 1, 4, 2, 5]
        );
    }
}
