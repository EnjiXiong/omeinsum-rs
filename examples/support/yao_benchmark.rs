#![allow(dead_code)]
use omeco::{EinCode, NestedEinsum};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Deserialize)]
pub struct YaoNetwork {
    pub format: String,
    pub mode: String,
    pub eincode: YaoEinCode,
    pub tensors: Vec<YaoTensor>,
    pub size_dict: HashMap<String, usize>,
}
#[derive(Deserialize)]
pub struct YaoEinCode {
    pub input_indices: Vec<Vec<String>>,
    pub output_indices: Vec<String>,
}
#[derive(Deserialize)]
pub struct YaoTensor {
    pub shape: Vec<usize>,
    pub data_re: Vec<f64>,
    pub data_im: Vec<f64>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct BenchmarkNetwork {
    pub format: String,
    pub source_format: String,
    pub source_mode: String,
    pub optimizer: OptimizerConfig,
    pub slicer: SlicerConfig,
    pub eincode: BenchmarkEinCode,
    pub tensors: Vec<BenchmarkTensor>,
    pub size_dict: HashMap<usize, usize>,
    pub contraction_order: TreeNode,
    pub cuts: Vec<usize>,
    pub assignment_count: usize,
    pub complexity: ComplexityReport,
    pub tree_real_audit: TreeRealAudit,
}
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct OptimizerConfig {
    pub algorithm: String,
    pub version: String,
    pub ntrials: usize,
    pub niters: usize,
    pub sc_target: f64,
    pub seed_base: u64,
}
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct SlicerConfig {
    pub algorithm: String,
    pub ntrials: usize,
    pub niters: usize,
    pub sc_target: f64,
    pub optimization_ratio: f64,
    pub seed_base: u64,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct BenchmarkEinCode {
    pub input_indices: Vec<Vec<usize>>,
    pub output_indices: Vec<usize>,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct BenchmarkTensor {
    pub shape: Vec<usize>,
    pub data_re: Vec<f32>,
    pub data_im: Vec<f32>,
    pub structurally_complex: bool,
}
#[derive(Default, Serialize, Deserialize, Clone)]
pub struct Complexity {
    pub log2_flops: f64,
    pub log2_peak_elements: f64,
    pub log2_readwrites: f64,
}
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct ComplexityReport {
    pub physical_unsliced: Complexity,
    pub physical_per_slice: Complexity,
    pub physical_total: Complexity,
}
#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct TreeRealAudit {
    pub pass_nodes: usize,
    pub ride_nodes: usize,
    pub merge_nodes: usize,
    pub pass_volume: f64,
    pub ride_volume: f64,
    pub merge_volume: f64,
    pub base_volume: f64,
    pub real_volume: f64,
    pub m: f64,
    pub r: f64,
    pub predicted_overhead: f64,
    pub native_contraction_sc_elements: f64,
    pub native_contraction_sc_bytes: f64,
    pub tree_real_logical_sc_elements: f64,
    pub tree_real_logical_sc_bytes: f64,
    pub native_resident_input_cache_elements: usize,
    pub native_resident_input_cache_bytes: usize,
    pub tree_real_resident_input_cache_elements: usize,
    pub tree_real_resident_input_cache_bytes: usize,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TreeNode {
    Leaf {
        tensor_index: usize,
    },
    Node {
        args: Vec<TreeNode>,
        input_indices: Vec<Vec<usize>>,
        output_indices: Vec<usize>,
    },
}
impl TreeNode {
    pub fn from_nested(tree: &NestedEinsum<usize>) -> Self {
        match tree {
            NestedEinsum::Leaf { tensor_index } => Self::Leaf {
                tensor_index: *tensor_index,
            },
            NestedEinsum::Node { args, eins } => Self::Node {
                args: args.iter().map(Self::from_nested).collect(),
                input_indices: eins.ixs.clone(),
                output_indices: eins.iy.clone(),
            },
        }
    }
    #[allow(dead_code)]
    pub fn to_nested(&self) -> NestedEinsum<usize> {
        match self {
            Self::Leaf { tensor_index } => NestedEinsum::leaf(*tensor_index),
            Self::Node {
                args,
                input_indices,
                output_indices,
            } => NestedEinsum::node(
                args.iter().map(Self::to_nested).collect(),
                EinCode::new(input_indices.clone(), output_indices.clone()),
            ),
        }
    }
    pub fn validate(&self, leaves: usize) -> Result<HashSet<usize>, String> {
        match self {
            Self::Leaf { tensor_index } if *tensor_index < leaves => {
                Ok(HashSet::from([*tensor_index]))
            }
            Self::Leaf { tensor_index } => Err(format!("leaf {tensor_index} is out of range")),
            Self::Node {
                args,
                input_indices,
                ..
            } => {
                if args.len() != 2 || input_indices.len() != 2 {
                    return Err("contraction tree must be binary".into());
                }
                let mut seen = args[0].validate(leaves)?;
                for leaf in args[1].validate(leaves)? {
                    if !seen.insert(leaf) {
                        return Err(format!("duplicate leaf {leaf}"));
                    }
                }
                Ok(seen)
            }
        }
    }
}

impl BenchmarkNetwork {
    pub fn validate(&self) -> Result<(), String> {
        if self.tensors.len() != self.eincode.input_indices.len() {
            return Err("tensor and input-label counts differ".into());
        }
        for (i, (tensor, labels)) in self
            .tensors
            .iter()
            .zip(&self.eincode.input_indices)
            .enumerate()
        {
            if tensor.shape.len() != labels.len() {
                return Err(format!("leaf {i} shape rank does not match its labels"));
            }
            for (&extent, label) in tensor.shape.iter().zip(labels) {
                if self.size_dict.get(label) != Some(&extent) {
                    return Err(format!(
                        "leaf {i} has an unknown label or wrong extent for {label}"
                    ));
                }
            }
            let elements = tensor.shape.iter().product::<usize>();
            if tensor.data_re.len() != elements || tensor.data_im.len() != elements {
                return Err(format!("leaf {i} data length does not match its shape"));
            }
        }
        fn validate_labels(
            network: &BenchmarkNetwork,
            labels: &[usize],
            context: &str,
        ) -> Result<(), String> {
            for label in labels {
                if !network.size_dict.contains_key(label) {
                    return Err(format!("{context} contains unknown label {label}"));
                }
            }
            Ok(())
        }
        validate_labels(self, &self.eincode.output_indices, "eincode output")?;
        fn walk(
            network: &BenchmarkNetwork,
            node: &TreeNode,
            seen: &mut HashSet<usize>,
        ) -> Result<Vec<usize>, String> {
            match node {
                TreeNode::Leaf { tensor_index } => {
                    if *tensor_index >= network.tensors.len() {
                        return Err(format!("leaf {tensor_index} is out of range"));
                    }
                    if !seen.insert(*tensor_index) {
                        return Err(format!("duplicate leaf {tensor_index}"));
                    }
                    Ok(network.eincode.input_indices[*tensor_index].clone())
                }
                TreeNode::Node {
                    args,
                    input_indices,
                    output_indices,
                } => {
                    if args.len() != 2 || input_indices.len() != 2 {
                        return Err("contraction tree must be binary".into());
                    }
                    for labels in input_indices {
                        validate_labels(network, labels, "tree input")?;
                    }
                    validate_labels(network, output_indices, "tree output")?;
                    for i in 0..2 {
                        let child_output = walk(network, &args[i], seen)?;
                        if child_output != input_indices[i] {
                            return Err(format!(
                                "child {i} output labels do not exactly match parent input labels"
                            ));
                        }
                    }
                    let inputs = input_indices
                        .iter()
                        .flatten()
                        .copied()
                        .collect::<HashSet<_>>();
                    if output_indices.iter().any(|label| !inputs.contains(label)) {
                        return Err("tree output contains a label absent from its inputs".into());
                    }
                    Ok(output_indices.clone())
                }
            }
        }
        let mut seen = HashSet::new();
        let root = walk(self, &self.contraction_order, &mut seen)?;
        if seen.len() != self.tensors.len() {
            return Err("tree does not contain every tensor".into());
        }
        if root != self.eincode.output_indices {
            return Err("root output labels do not exactly match eincode output".into());
        }
        for cut in &self.cuts {
            if !self.size_dict.contains_key(cut) || self.eincode.output_indices.contains(cut) {
                return Err(format!("invalid physical cut {cut}"));
            }
        }
        let assignments = self
            .cuts
            .iter()
            .try_fold(1usize, |n, cut| n.checked_mul(self.size_dict[cut]))
            .ok_or_else(|| "assignment count overflow".to_string())?;
        if assignments != self.assignment_count {
            return Err("assignment count does not match cuts".into());
        }
        Ok(())
    }
}

pub fn cut_sizes(sizes: &HashMap<usize, usize>, cuts: &[usize]) -> HashMap<usize, usize> {
    let mut out = sizes.clone();
    for cut in cuts {
        out.insert(*cut, 1);
    }
    out
}

pub fn audit_tree(
    tree: &TreeNode,
    complex: &[bool],
    sizes: &HashMap<usize, usize>,
    peak: f64,
) -> TreeRealAudit {
    fn walk(
        node: &TreeNode,
        complex: &[bool],
        sizes: &HashMap<usize, usize>,
        a: &mut TreeRealAudit,
    ) -> bool {
        match node {
            TreeNode::Leaf { tensor_index } => complex[*tensor_index],
            TreeNode::Node {
                args,
                input_indices,
                ..
            } => {
                let left = walk(&args[0], complex, sizes, a);
                let right = walk(&args[1], complex, sizes, a);
                let labels: HashSet<_> = input_indices.iter().flatten().copied().collect();
                let volume = labels.iter().map(|x| sizes[x] as f64).product::<f64>();
                a.base_volume += volume;
                match (left, right) {
                    (false, false) => {
                        a.pass_nodes += 1;
                        a.pass_volume += volume;
                    }
                    (true, true) => {
                        a.merge_nodes += 1;
                        a.merge_volume += volume;
                    }
                    _ => {
                        a.ride_nodes += 1;
                        a.ride_volume += volume;
                    }
                }
                left || right
            }
        }
    }
    let mut a = TreeRealAudit::default();
    walk(tree, complex, sizes, &mut a);
    a.real_volume = a.pass_volume + 2.0 * a.ride_volume + 3.0 * a.merge_volume;
    if a.base_volume > 0.0 {
        a.m = a.merge_volume / a.base_volume;
        a.r = a.ride_volume / a.base_volume;
        a.predicted_overhead = a.real_volume / a.base_volume;
    }
    a.native_contraction_sc_elements = 2f64.powf(peak);
    a.native_contraction_sc_bytes = a.native_contraction_sc_elements * 8.0;
    a.tree_real_logical_sc_elements = a.native_contraction_sc_elements * 2.0;
    a.tree_real_logical_sc_bytes = a.tree_real_logical_sc_elements * 4.0;
    a
}

pub fn slice_column_major<T: Copy>(
    data: &[T],
    shape: &[usize],
    labels: &[usize],
    cuts: &HashMap<usize, usize>,
) -> (Vec<T>, Vec<usize>) {
    let mut out_shape = shape.to_vec();
    for (axis, label) in labels.iter().enumerate() {
        if cuts.contains_key(label) {
            out_shape[axis] = 1;
        }
    }
    let len: usize = out_shape.iter().product();
    let mut out = Vec::with_capacity(len);
    for mut linear in 0..len {
        let mut source = 0;
        let mut stride = 1;
        for axis in 0..shape.len() {
            let coordinate = if let Some(&value) = cuts.get(&labels[axis]) {
                value
            } else {
                let v = linear % out_shape[axis];
                linear /= out_shape[axis];
                v
            };
            source += coordinate * stride;
            stride *= shape[axis];
        }
        out.push(data[source]);
    }
    (out, out_shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn slicing_retains_axes() {
        let (v, s) = slice_column_major(
            &[0, 1, 2, 3, 4, 5],
            &[2, 3],
            &[0, 1],
            &HashMap::from([(1, 1)]),
        );
        assert_eq!(s, vec![2, 1]);
        assert_eq!(v, vec![2, 3]);
    }
    #[test]
    fn audit_identity_and_counts() {
        let tree = TreeNode::Node {
            args: vec![
                TreeNode::Node {
                    args: vec![
                        TreeNode::Leaf { tensor_index: 0 },
                        TreeNode::Leaf { tensor_index: 1 },
                    ],
                    input_indices: vec![vec![0], vec![0]],
                    output_indices: vec![],
                },
                TreeNode::Node {
                    args: vec![
                        TreeNode::Leaf { tensor_index: 2 },
                        TreeNode::Leaf { tensor_index: 3 },
                    ],
                    input_indices: vec![vec![0], vec![0]],
                    output_indices: vec![],
                },
            ],
            input_indices: vec![vec![], vec![]],
            output_indices: vec![],
        };
        let a = audit_tree(
            &tree,
            &[false, false, true, true],
            &HashMap::from([(0, 2)]),
            0.0,
        );
        assert_eq!((a.pass_nodes, a.ride_nodes, a.merge_nodes), (1, 1, 1));
        assert!((a.predicted_overhead - (1.0 + 2.0 * a.m + a.r)).abs() < 1e-12);
    }

    #[test]
    fn validation_rejects_malformed_child_label_order() {
        let mut network = BenchmarkNetwork {
            format: "omeinsum-yao-benchmark-v2".into(),
            source_format: "x".into(),
            source_mode: "x".into(),
            optimizer: Default::default(),
            slicer: Default::default(),
            eincode: BenchmarkEinCode {
                input_indices: vec![vec![0, 1], vec![1]],
                output_indices: vec![0],
            },
            tensors: vec![
                BenchmarkTensor {
                    shape: vec![2, 3],
                    data_re: vec![0.0; 6],
                    data_im: vec![0.0; 6],
                    structurally_complex: false,
                },
                BenchmarkTensor {
                    shape: vec![3],
                    data_re: vec![0.0; 3],
                    data_im: vec![0.0; 3],
                    structurally_complex: false,
                },
            ],
            size_dict: HashMap::from([(0, 2), (1, 3)]),
            contraction_order: TreeNode::Node {
                args: vec![
                    TreeNode::Leaf { tensor_index: 0 },
                    TreeNode::Leaf { tensor_index: 1 },
                ],
                input_indices: vec![vec![0, 1], vec![1]],
                output_indices: vec![0],
            },
            cuts: vec![],
            assignment_count: 1,
            complexity: Default::default(),
            tree_real_audit: Default::default(),
        };
        assert!(network.validate().is_ok());
        if let TreeNode::Node { input_indices, .. } = &mut network.contraction_order {
            input_indices[0] = vec![1, 0];
        }
        assert!(network.validate().unwrap_err().contains("exactly match"));
    }
}
