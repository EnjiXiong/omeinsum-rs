use omeco::{EinCode, NestedEinsum};
use omeinsum::{Einsum, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[allow(dead_code)]
#[derive(Deserialize)]
pub struct YaoNetwork {
    pub format: String,
    pub mode: String,
    pub eincode: YaoEinCode,
    pub tensors: Vec<YaoTensor>,
    pub size_dict: HashMap<String, usize>,
}

#[allow(dead_code)]
#[derive(Deserialize)]
pub struct YaoEinCode {
    pub input_indices: Vec<Vec<String>>,
    pub output_indices: Vec<String>,
}

#[allow(dead_code)]
#[derive(Deserialize)]
pub struct YaoTensor {
    pub shape: Vec<usize>,
    pub data_re: Vec<f64>,
    pub data_im: Vec<f64>,
}

#[derive(Serialize, Deserialize)]
pub struct BenchmarkNetwork {
    pub format: String,
    pub source_format: String,
    pub source_mode: String,
    pub optimizer: String,
    pub realification: String,
    pub eincode: BenchmarkEinCode,
    pub tensors: Vec<BenchmarkTensor>,
    pub size_dict: HashMap<usize, usize>,
    pub contraction_order: TreeNode,
    pub complexity: Complexity,
}

#[derive(Serialize, Deserialize)]
pub struct BenchmarkEinCode {
    pub input_indices: Vec<Vec<usize>>,
    pub output_indices: Vec<usize>,
}

#[derive(Serialize, Deserialize)]
pub struct BenchmarkTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
pub struct Complexity {
    pub log2_flops: f64,
    pub log2_peak_elements: f64,
    pub log2_readwrites: f64,
    pub estimated_peak_bytes_f32: f64,
}

#[derive(Serialize, Deserialize)]
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
    #[allow(dead_code)]
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
}

impl BenchmarkNetwork {
    #[allow(dead_code)]
    pub fn einsum(&self) -> Einsum<usize> {
        let mut einsum = Einsum::new(
            self.eincode.input_indices.clone(),
            self.eincode.output_indices.clone(),
            self.size_dict.clone(),
        );
        einsum.set_contraction_tree(self.contraction_order.to_nested());
        einsum
    }

    #[allow(dead_code)]
    pub fn tensors<B: omeinsum::Backend + Clone>(&self, backend: B) -> Vec<Tensor<f32, B>>
    where
        f32: omeinsum::BackendScalar<B>,
    {
        self.tensors
            .iter()
            .map(|tensor| {
                Tensor::from_data_with_backend(&tensor.data, &tensor.shape, backend.clone())
            })
            .collect()
    }
}
