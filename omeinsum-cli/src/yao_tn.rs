use std::collections::{HashMap, HashSet};

use omeco::json::NestedEinsumTree;
use omeco::NestedEinsum;
use omeinsum::static_plan::{
    BinaryContractionTree, ComplexNetwork, ComplexTensor, PlanError, TensorSpec,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct YaoTensorDto {
    shape: Vec<usize>,
    data_re: Vec<f64>,
    data_im: Vec<f64>,
}

#[derive(Deserialize)]
struct YaoEinCodeDto {
    input_indices: Vec<Vec<String>>,
    output_indices: Vec<String>,
}

#[derive(Deserialize)]
struct YaoTensorNetworkDto {
    format: String,
    mode: String,
    eincode: YaoEinCodeDto,
    tensors: Vec<YaoTensorDto>,
    size_dict: HashMap<String, usize>,
    contraction_order: Option<NestedEinsumTree<i32>>,
}

pub(crate) fn parse_yao_tn(json: &str) -> Result<ComplexNetwork<f64>, PlanError> {
    let dto: YaoTensorNetworkDto = serde_json::from_str(json)
        .map_err(|error| PlanError::InvalidNetwork(format!("invalid JSON: {error}")))?;
    if dto.format != "yao-tn-v1" {
        return Err(PlanError::InvalidFormat(format!(
            "expected yao-tn-v1, got {:?}",
            dto.format
        )));
    }
    if dto.mode != "overlap" {
        return Err(PlanError::InvalidNetwork(format!(
            "expected overlap mode, got {:?}",
            dto.mode
        )));
    }
    if !dto.eincode.output_indices.is_empty() {
        return Err(PlanError::NonScalarOutput(parse_labels(
            &dto.eincode.output_indices,
            "output indices",
        )?));
    }
    if dto.tensors.len() != dto.eincode.input_indices.len() {
        return Err(PlanError::InvalidNetwork(format!(
            "tensor count {} differs from input-index count {}",
            dto.tensors.len(),
            dto.eincode.input_indices.len()
        )));
    }

    let size_dict = normalize_size_dict(&dto.size_dict)?;
    let sizes: HashMap<_, _> = size_dict.iter().copied().collect();
    let mut tensors = Vec::with_capacity(dto.tensors.len());
    for (index, (tensor, raw_modes)) in dto
        .tensors
        .into_iter()
        .zip(dto.eincode.input_indices)
        .enumerate()
    {
        let modes = parse_labels(&raw_modes, &format!("tensor {index} modes"))?;
        if modes.len() != tensor.shape.len() {
            return Err(PlanError::InvalidTensor {
                index,
                detail: format!(
                    "rank {} differs from mode count {}",
                    tensor.shape.len(),
                    modes.len()
                ),
            });
        }
        let mut unique = HashSet::new();
        if let Some(mode) = modes.iter().find(|mode| !unique.insert(**mode)) {
            return Err(PlanError::InvalidTensor {
                index,
                detail: format!("mode {mode} is repeated"),
            });
        }
        let elements = tensor
            .shape
            .iter()
            .try_fold(1usize, |product, size| product.checked_mul(*size));
        let elements = elements.ok_or_else(|| PlanError::InvalidTensor {
            index,
            detail: "shape product overflows usize".to_string(),
        })?;
        if tensor.data_re.len() != elements || tensor.data_im.len() != elements {
            return Err(PlanError::InvalidTensor {
                index,
                detail: format!(
                    "shape has {elements} elements but real/imag lengths are {}/{}",
                    tensor.data_re.len(),
                    tensor.data_im.len()
                ),
            });
        }
        for ((mode, actual), dimension) in modes.iter().zip(tensor.shape.iter()).zip(0usize..) {
            let expected = sizes.get(mode).ok_or_else(|| PlanError::InvalidTensor {
                index,
                detail: format!("mode {mode} is missing from size_dict"),
            })?;
            if expected != actual {
                return Err(PlanError::InvalidTensor {
                    index,
                    detail: format!(
                        "shape[{dimension}] is {actual} but size_dict[{mode}] is {expected}"
                    ),
                });
            }
        }
        tensors.push(ComplexTensor {
            spec: TensorSpec {
                modes,
                shape: tensor.shape,
            },
            real: tensor.data_re,
            imag: tensor.data_im,
        });
    }

    let tree = dto
        .contraction_order
        .ok_or(PlanError::MissingContractionOrder)?;
    validate_tree_flags(&tree)?;
    let nested: NestedEinsum<i32> = tree.into();
    let mut seen = vec![false; tensors.len()];
    let (tree, root_modes) = normalize_tree(nested, &tensors, &sizes, &mut seen)?;
    if seen.iter().any(|seen| !seen) {
        let missing: Vec<_> = seen
            .iter()
            .enumerate()
            .filter_map(|(index, seen)| (!seen).then_some(index))
            .collect();
        return Err(PlanError::InvalidTree {
            detail: format!("tree does not reference tensors {missing:?}"),
        });
    }
    if !root_modes.is_empty() {
        return Err(PlanError::NonScalarOutput(root_modes));
    }

    Ok(ComplexNetwork {
        tensors,
        output_modes: vec![],
        size_dict,
        tree,
    })
}

fn parse_labels(labels: &[String], context: &str) -> Result<Vec<i32>, PlanError> {
    labels
        .iter()
        .map(|label| {
            label.parse::<i32>().map_err(|_| {
                PlanError::InvalidNetwork(format!("{context} contains non-i32 label {label:?}"))
            })
        })
        .collect()
}

fn normalize_size_dict(raw: &HashMap<String, usize>) -> Result<Vec<(i32, usize)>, PlanError> {
    let mut normalized = Vec::with_capacity(raw.len());
    let mut labels = HashSet::new();
    for (label, size) in raw {
        let label = label.parse::<i32>().map_err(|_| {
            PlanError::InvalidNetwork(format!("size_dict contains non-i32 label {label:?}"))
        })?;
        if !labels.insert(label) {
            return Err(PlanError::InvalidNetwork(format!(
                "size_dict contains duplicate normalized label {label}"
            )));
        }
        if *size == 0 {
            return Err(PlanError::InvalidNetwork(format!(
                "size_dict[{label}] must be positive"
            )));
        }
        normalized.push((label, *size));
    }
    normalized.sort_unstable_by_key(|(label, _)| *label);
    Ok(normalized)
}

fn validate_tree_flags(tree: &NestedEinsumTree<i32>) -> Result<(), PlanError> {
    match tree {
        NestedEinsumTree::Leaf { isleaf, .. } => {
            if !isleaf {
                return Err(PlanError::InvalidTree {
                    detail: "leaf node has isleaf=false".to_string(),
                });
            }
        }
        NestedEinsumTree::Node { isleaf, args, .. } => {
            if *isleaf {
                return Err(PlanError::InvalidTree {
                    detail: "internal node has isleaf=true".to_string(),
                });
            }
            for child in args {
                validate_tree_flags(child)?;
            }
        }
    }
    Ok(())
}

fn normalize_tree(
    tree: NestedEinsum<i32>,
    tensors: &[ComplexTensor<f64>],
    sizes: &HashMap<i32, usize>,
    seen: &mut [bool],
) -> Result<(BinaryContractionTree, Vec<i32>), PlanError> {
    match tree {
        NestedEinsum::Leaf { tensor_index } => {
            let tensor = tensors
                .get(tensor_index)
                .ok_or_else(|| PlanError::InvalidTree {
                    detail: format!("leaf tensor index {tensor_index} is out of range"),
                })?;
            if std::mem::replace(&mut seen[tensor_index], true) {
                return Err(PlanError::InvalidTree {
                    detail: format!("leaf tensor index {tensor_index} appears more than once"),
                });
            }
            Ok((
                BinaryContractionTree::Leaf { tensor_index },
                tensor.spec.modes.clone(),
            ))
        }
        NestedEinsum::Node { args, eins } => {
            if args.len() != 2 {
                return Err(PlanError::NonBinaryNode {
                    children: args.len(),
                });
            }
            if eins.ixs.len() != 2 {
                return Err(PlanError::InvalidTree {
                    detail: format!(
                        "binary node has {} input-index groups instead of 2",
                        eins.ixs.len()
                    ),
                });
            }
            let mut args = args.into_iter();
            let (left, left_modes) = normalize_tree(args.next().unwrap(), tensors, sizes, seen)?;
            let (right, right_modes) = normalize_tree(args.next().unwrap(), tensors, sizes, seen)?;
            if eins.ixs[0] != left_modes || eins.ixs[1] != right_modes {
                return Err(PlanError::InvalidTree {
                    detail: format!(
                        "node input modes {:?} do not match child modes {:?}",
                        eins.ixs,
                        [left_modes, right_modes]
                    ),
                });
            }
            let mut unique = HashSet::new();
            for mode in &eins.iy {
                if !sizes.contains_key(mode) {
                    return Err(PlanError::InvalidTree {
                        detail: format!("node output mode {mode} is missing from size_dict"),
                    });
                }
                if !unique.insert(*mode) {
                    return Err(PlanError::InvalidTree {
                        detail: format!("node output mode {mode} is repeated"),
                    });
                }
            }
            let output_modes = eins.iy;
            Ok((
                BinaryContractionTree::Node {
                    output_modes: output_modes.clone(),
                    left: Box::new(left),
                    right: Box::new(right),
                },
                output_modes,
            ))
        }
    }
}
