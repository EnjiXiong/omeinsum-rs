use std::collections::{HashMap, HashSet};

use crate::backend::contract_plan::plan_contraction;

use super::hash::{plan_hash, tree_hash};
use super::{
    BinaryContractionTree, ComplexNetwork, ContractionSpec, KernelKind, PlanError, PlanNode,
    PlanStats, Plane, Representation, StaticPlan, TensorSpec, ValueId, ValueSpec,
};

pub fn build_geometry_plan(network: &ComplexNetwork<f64>) -> Result<StaticPlan, PlanError> {
    validate_network_geometry(network)?;
    let sizes: HashMap<_, _> = network.size_dict.iter().copied().collect();
    let mut state = BuildState {
        network,
        sizes: &sizes,
        values: network
            .tensors
            .iter()
            .enumerate()
            .map(|(index, tensor)| ValueSpec {
                id: ValueId(index),
                tensor: tensor.spec.clone(),
                planes: vec![Plane::Real],
            })
            .collect(),
        nodes: Vec::new(),
        seen_leaves: vec![false; network.tensors.len()],
        total_volume: 0,
    };
    let output = state.walk(&network.tree)?;
    if state.seen_leaves.iter().any(|seen| !seen) {
        let missing: Vec<_> = state
            .seen_leaves
            .iter()
            .enumerate()
            .filter_map(|(index, seen)| (!seen).then_some(index))
            .collect();
        return Err(PlanError::InvalidTree {
            detail: format!("tree does not reference tensors {missing:?}"),
        });
    }
    let root = &state.values[output.0].tensor;
    if root.modes != network.output_modes {
        return Err(PlanError::InvalidTree {
            detail: format!(
                "tree root modes {:?} differ from network output {:?}",
                root.modes, network.output_modes
            ),
        });
    }
    if !root.modes.is_empty() {
        return Err(PlanError::NonScalarOutput(root.modes.clone()));
    }

    let tree_hash = tree_hash(&network.tree)?;
    let node_count = state.nodes.len();
    let mut plan = StaticPlan {
        representation: Representation::RealSkeleton,
        tree_hash,
        plan_hash: String::new(),
        leaf_values: (0..network.tensors.len()).map(ValueId).collect(),
        values: state.values,
        nodes: state.nodes,
        output,
        stats: PlanStats {
            real_leaf_count: network.tensors.len(),
            complex_leaf_count: 0,
            real_real_nodes: node_count,
            ride_left_nodes: 0,
            ride_right_nodes: 0,
            merge_3m_nodes: 0,
            flat_4m_nodes: 0,
            real_skeleton_volume: state.total_volume,
            real_matmul_volume: state.total_volume,
            realification_cost: None,
        },
    };
    plan.plan_hash = plan_hash(&plan)?;
    Ok(plan)
}

struct BuildState<'a> {
    network: &'a ComplexNetwork<f64>,
    sizes: &'a HashMap<i32, usize>,
    values: Vec<ValueSpec>,
    nodes: Vec<PlanNode>,
    seen_leaves: Vec<bool>,
    total_volume: u128,
}

impl BuildState<'_> {
    fn walk(&mut self, tree: &BinaryContractionTree) -> Result<ValueId, PlanError> {
        match tree {
            BinaryContractionTree::Leaf { tensor_index } => {
                if *tensor_index >= self.network.tensors.len() {
                    return Err(PlanError::InvalidTree {
                        detail: format!("leaf tensor index {tensor_index} is out of range"),
                    });
                }
                if std::mem::replace(&mut self.seen_leaves[*tensor_index], true) {
                    return Err(PlanError::InvalidTree {
                        detail: format!("leaf tensor index {tensor_index} appears more than once"),
                    });
                }
                Ok(ValueId(*tensor_index))
            }
            BinaryContractionTree::Node {
                output_modes,
                left,
                right,
            } => {
                let left_id = self.walk(left)?;
                let right_id = self.walk(right)?;
                let left_spec = self.values[left_id.0].tensor.clone();
                let right_spec = self.values[right_id.0].tensor.clone();
                validate_output_modes(output_modes, &left_spec, &right_spec, self.sizes)?;

                let geometry = plan_contraction(
                    &left_spec.modes,
                    &left_spec.shape,
                    &right_spec.modes,
                    &right_spec.shape,
                    output_modes,
                );
                let node_id = self.nodes.len();
                if !geometry.left_trace.is_empty() || !geometry.right_trace.is_empty() {
                    return Err(PlanError::UnsupportedTrace {
                        node: node_id,
                        left_trace_modes: geometry.left_trace,
                        right_trace_modes: geometry.right_trace,
                    });
                }

                let volume = [
                    geometry.batch_size,
                    geometry.left_size,
                    geometry.contract_size,
                    geometry.right_size,
                ]
                .into_iter()
                .try_fold(1u128, |product, value| product.checked_mul(value as u128))
                .ok_or_else(|| PlanError::InvalidTree {
                    detail: format!("node {node_id} contraction volume overflows u128"),
                })?;
                self.total_volume = self.total_volume.checked_add(volume).ok_or_else(|| {
                    PlanError::InvalidTree {
                        detail: "total contraction volume overflows u128".to_string(),
                    }
                })?;

                let output_shape = output_modes
                    .iter()
                    .map(|mode| self.sizes[mode])
                    .collect::<Vec<_>>();
                let output = ValueId(self.values.len());
                self.values.push(ValueSpec {
                    id: output,
                    tensor: TensorSpec {
                        modes: output_modes.clone(),
                        shape: output_shape,
                    },
                    planes: vec![Plane::Real],
                });
                let left_permutation = geometry.a_permutation(&left_spec.modes);
                let right_permutation = geometry.b_permutation(&right_spec.modes);
                self.nodes.push(PlanNode {
                    id: node_id,
                    left: left_id,
                    right: right_id,
                    output,
                    kind: KernelKind::RealReal,
                    contraction: ContractionSpec {
                        batch_modes: geometry.batch_modes,
                        left_modes: geometry.left_modes,
                        right_modes: geometry.right_modes,
                        contracted_modes: geometry.contracted_modes,
                        left_trace_modes: geometry.left_trace,
                        right_trace_modes: geometry.right_trace,
                        batch: geometry.batch_size,
                        m: geometry.left_size,
                        k: geometry.contract_size,
                        n: geometry.right_size,
                        left_permutation,
                        right_permutation,
                        output_permutation: geometry.output_perm,
                    },
                    scratch: vec![],
                    real_skeleton_volume: volume,
                });
                Ok(output)
            }
        }
    }
}

fn validate_network_geometry(network: &ComplexNetwork<f64>) -> Result<(), PlanError> {
    if !network.output_modes.is_empty() {
        return Err(PlanError::NonScalarOutput(network.output_modes.clone()));
    }
    if network.tensors.is_empty() {
        return Err(PlanError::InvalidNetwork(
            "at least one input tensor is required".to_string(),
        ));
    }
    let mut sizes = HashMap::new();
    for (label, size) in &network.size_dict {
        if *size == 0 {
            return Err(PlanError::InvalidNetwork(format!(
                "size_dict[{label}] must be positive"
            )));
        }
        if sizes.insert(*label, *size).is_some() {
            return Err(PlanError::InvalidNetwork(format!(
                "size_dict repeats label {label}"
            )));
        }
    }
    for (index, tensor) in network.tensors.iter().enumerate() {
        if tensor.spec.modes.len() != tensor.spec.shape.len() {
            return Err(PlanError::InvalidTensor {
                index,
                detail: "mode count differs from rank".to_string(),
            });
        }
        let mut unique = HashSet::new();
        for (axis, (mode, dimension)) in tensor
            .spec
            .modes
            .iter()
            .zip(tensor.spec.shape.iter())
            .enumerate()
        {
            if !unique.insert(*mode) {
                return Err(PlanError::InvalidTensor {
                    index,
                    detail: format!("mode {mode} is repeated"),
                });
            }
            let expected = sizes.get(mode).ok_or_else(|| PlanError::InvalidTensor {
                index,
                detail: format!("mode {mode} is missing from size_dict"),
            })?;
            if expected != dimension {
                return Err(PlanError::InvalidTensor {
                    index,
                    detail: format!(
                        "shape[{axis}] is {dimension} but size_dict[{mode}] is {expected}"
                    ),
                });
            }
        }
    }
    Ok(())
}

fn validate_output_modes(
    output_modes: &[i32],
    left: &TensorSpec,
    right: &TensorSpec,
    sizes: &HashMap<i32, usize>,
) -> Result<(), PlanError> {
    let available: HashSet<_> = left
        .modes
        .iter()
        .chain(right.modes.iter())
        .copied()
        .collect();
    let mut unique = HashSet::new();
    for mode in output_modes {
        if !available.contains(mode) {
            return Err(PlanError::InvalidTree {
                detail: format!("node output mode {mode} is absent from both children"),
            });
        }
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
    Ok(())
}
