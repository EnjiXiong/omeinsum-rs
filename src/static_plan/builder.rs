use std::collections::{HashMap, HashSet};

use crate::backend::contract_plan::plan_contraction;

use super::hash::{plan_hash, tree_hash};
use super::{
    BinaryContractionTree, ComplexNetwork, ContractionSpec, InputSet, InputTensor, KernelKind,
    LeafClass, PlanBundle, PlanError, PlanNode, PlanStats, Plane, RealificationCost,
    Representation, ScratchId, ScratchRole, ScratchSpec, StaticPlan, TensorSpec, ValueId,
    ValueSpec,
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

pub fn build_plan_bundle(
    network: &ComplexNetwork<f64>,
    realness_tol: f64,
) -> Result<PlanBundle, PlanError> {
    let inputs = classify_inputs(network, realness_tol)?;
    let geometry = build_geometry_plan(network)?;
    let real_leaf_count = inputs
        .tensors
        .iter()
        .filter(|tensor| tensor.class == LeafClass::Real)
        .count();
    let complex_leaf_count = inputs.tensors.len() - real_leaf_count;

    let real_skeleton =
        configure_real_skeleton(geometry.clone(), real_leaf_count, complex_leaf_count)?;
    let flat_4m = configure_flat_4m(geometry.clone(), real_leaf_count, complex_leaf_count)?;
    let realified_rank3 =
        configure_selective(geometry, &inputs, real_leaf_count, complex_leaf_count)?;
    let bundle = PlanBundle {
        format: "omeinsum-static-plan-v1".to_string(),
        realness_tol,
        tree_hash: real_skeleton.tree_hash.clone(),
        inputs,
        real_skeleton,
        flat_4m,
        realified_rank3,
    };
    bundle.validate()?;
    Ok(bundle)
}

fn classify_inputs(
    network: &ComplexNetwork<f64>,
    realness_tol: f64,
) -> Result<InputSet<f64>, PlanError> {
    if !realness_tol.is_finite() || realness_tol <= 0.0 {
        return Err(PlanError::InvalidNetwork(format!(
            "realness tolerance must be positive and finite, got {realness_tol}"
        )));
    }
    let tensors = network
        .tensors
        .iter()
        .enumerate()
        .map(|(index, tensor)| {
            let elements = tensor
                .spec
                .shape
                .iter()
                .try_fold(1usize, |product, size| product.checked_mul(*size));
            let elements = elements.ok_or_else(|| PlanError::InvalidTensor {
                index,
                detail: "shape product overflows usize".to_string(),
            })?;
            if tensor.real.len() != elements || tensor.imag.len() != elements {
                return Err(PlanError::InvalidTensor {
                    index,
                    detail: format!(
                        "shape has {elements} elements but real/imag lengths are {}/{}",
                        tensor.real.len(),
                        tensor.imag.len()
                    ),
                });
            }
            if let Some((plane, position, value)) = tensor
                .real
                .iter()
                .enumerate()
                .map(|(position, value)| ("real", position, *value))
                .chain(
                    tensor
                        .imag
                        .iter()
                        .enumerate()
                        .map(|(position, value)| ("imag", position, *value)),
                )
                .find(|(_, _, value)| !value.is_finite())
            {
                return Err(PlanError::InvalidTensor {
                    index,
                    detail: format!("{plane}[{position}] is non-finite: {value}"),
                });
            }
            let imag_max = tensor
                .imag
                .iter()
                .map(|value| value.abs())
                .fold(0.0_f64, f64::max);
            let class = if imag_max <= realness_tol {
                LeafClass::Real
            } else {
                LeafClass::Complex
            };
            let imag = if class == LeafClass::Real {
                vec![0.0; elements]
            } else {
                tensor.imag.clone()
            };
            Ok(InputTensor {
                spec: tensor.spec.clone(),
                real: tensor.real.clone(),
                imag,
                class,
                imag_max,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(InputSet { tensors })
}

fn configure_real_skeleton(
    mut plan: StaticPlan,
    real_leaf_count: usize,
    complex_leaf_count: usize,
) -> Result<StaticPlan, PlanError> {
    plan.representation = Representation::RealSkeleton;
    for value in &mut plan.values {
        value.planes = vec![Plane::Real];
    }
    for node in &mut plan.nodes {
        node.kind = KernelKind::RealReal;
        node.scratch.clear();
    }
    plan.stats = PlanStats {
        real_leaf_count,
        complex_leaf_count,
        real_real_nodes: plan.nodes.len(),
        ride_left_nodes: 0,
        ride_right_nodes: 0,
        merge_3m_nodes: 0,
        flat_4m_nodes: 0,
        real_skeleton_volume: sum_node_volume(&plan)?,
        real_matmul_volume: sum_node_volume(&plan)?,
        realification_cost: None,
    };
    plan.plan_hash = plan_hash(&plan)?;
    Ok(plan)
}

fn configure_flat_4m(
    mut plan: StaticPlan,
    real_leaf_count: usize,
    complex_leaf_count: usize,
) -> Result<StaticPlan, PlanError> {
    plan.representation = Representation::Flat4M;
    for value in &mut plan.values {
        value.planes = vec![Plane::Real, Plane::Imag];
    }
    let mut next_scratch = 0usize;
    for node in &mut plan.nodes {
        node.kind = KernelKind::Flat4M;
        let output_elements = tensor_elements(&plan.values[node.output.0].tensor)?;
        node.scratch = scratch_specs(
            &[
                ScratchRole::Product1,
                ScratchRole::Product2,
                ScratchRole::Product3,
                ScratchRole::Product4,
            ],
            &mut next_scratch,
            output_elements,
            output_elements,
            output_elements,
        );
    }
    let volume = sum_node_volume(&plan)?;
    let real_matmul_volume = volume.checked_mul(4).ok_or_else(|| {
        PlanError::InvalidNetwork("flat-4m real matmul volume overflows u128".to_string())
    })?;
    plan.stats = PlanStats {
        real_leaf_count,
        complex_leaf_count,
        real_real_nodes: 0,
        ride_left_nodes: 0,
        ride_right_nodes: 0,
        merge_3m_nodes: 0,
        flat_4m_nodes: plan.nodes.len(),
        real_skeleton_volume: volume,
        real_matmul_volume,
        realification_cost: None,
    };
    plan.plan_hash = plan_hash(&plan)?;
    Ok(plan)
}

fn configure_selective(
    mut plan: StaticPlan,
    inputs: &InputSet<f64>,
    real_leaf_count: usize,
    complex_leaf_count: usize,
) -> Result<StaticPlan, PlanError> {
    plan.representation = Representation::RealifiedRank3;
    for (value, input) in plan.values.iter_mut().zip(&inputs.tensors) {
        value.planes = planes_for_class(&input.class);
    }

    let mut next_scratch = 0usize;
    let mut real_real_nodes = 0usize;
    let mut ride_left_nodes = 0usize;
    let mut ride_right_nodes = 0usize;
    let mut merge_3m_nodes = 0usize;
    let mut real_real_volume = 0u128;
    let mut ride_volume = 0u128;
    let mut merge_volume = 0u128;
    for node in &mut plan.nodes {
        let left_complex = plan.values[node.left.0].planes.len() == 2;
        let right_complex = plan.values[node.right.0].planes.len() == 2;
        let output_elements = tensor_elements(&plan.values[node.output.0].tensor)?;
        let left_elements = tensor_elements(&plan.values[node.left.0].tensor)?;
        let right_elements = tensor_elements(&plan.values[node.right.0].tensor)?;
        match (left_complex, right_complex) {
            (false, false) => {
                node.kind = KernelKind::RealReal;
                plan.values[node.output.0].planes = vec![Plane::Real];
                node.scratch.clear();
                real_real_nodes += 1;
                real_real_volume = checked_add_volume(real_real_volume, node)?;
            }
            (true, false) => {
                node.kind = KernelKind::RideLeft;
                plan.values[node.output.0].planes = vec![Plane::Real, Plane::Imag];
                node.scratch.clear();
                ride_left_nodes += 1;
                ride_volume = checked_add_volume(ride_volume, node)?;
            }
            (false, true) => {
                node.kind = KernelKind::RideRight;
                plan.values[node.output.0].planes = vec![Plane::Real, Plane::Imag];
                node.scratch.clear();
                ride_right_nodes += 1;
                ride_volume = checked_add_volume(ride_volume, node)?;
            }
            (true, true) => {
                node.kind = KernelKind::Merge3M;
                plan.values[node.output.0].planes = vec![Plane::Real, Plane::Imag];
                node.scratch = scratch_specs(
                    &[
                        ScratchRole::LeftSum,
                        ScratchRole::RightSum,
                        ScratchRole::Product1,
                        ScratchRole::Product2,
                        ScratchRole::Product3,
                    ],
                    &mut next_scratch,
                    left_elements,
                    right_elements,
                    output_elements,
                );
                merge_3m_nodes += 1;
                merge_volume = checked_add_volume(merge_volume, node)?;
            }
        }
    }
    let total_volume = real_real_volume
        .checked_add(ride_volume)
        .and_then(|value| value.checked_add(merge_volume))
        .ok_or_else(|| {
            PlanError::InvalidNetwork("selective contraction volume overflows u128".to_string())
        })?;
    if total_volume == 0 {
        return Err(PlanError::InvalidNetwork(
            "selective plan has zero contraction volume".to_string(),
        ));
    }
    let real_matmul_volume = real_real_volume
        .checked_add(ride_volume.checked_mul(2).ok_or_else(|| {
            PlanError::InvalidNetwork("ride volume accounting overflows u128".to_string())
        })?)
        .and_then(|value| {
            merge_volume
                .checked_mul(3)
                .and_then(|merge| value.checked_add(merge))
        })
        .ok_or_else(|| {
            PlanError::InvalidNetwork("selective real matmul volume overflows u128".to_string())
        })?;
    let denominator = total_volume as f64;
    let real_real_fraction = real_real_volume as f64 / denominator;
    let ride_fraction = ride_volume as f64 / denominator;
    let merge_fraction = merge_volume as f64 / denominator;
    plan.stats = PlanStats {
        real_leaf_count,
        complex_leaf_count,
        real_real_nodes,
        ride_left_nodes,
        ride_right_nodes,
        merge_3m_nodes,
        flat_4m_nodes: 0,
        real_skeleton_volume: total_volume,
        real_matmul_volume,
        realification_cost: Some(RealificationCost {
            real_real_volume,
            ride_volume,
            merge_volume,
            real_real_fraction,
            ride_fraction,
            merge_fraction,
            predicted_arithmetic_overhead: 1.0 + ride_fraction + 2.0 * merge_fraction,
        }),
    };
    plan.plan_hash = plan_hash(&plan)?;
    Ok(plan)
}

fn planes_for_class(class: &LeafClass) -> Vec<Plane> {
    match class {
        LeafClass::Real => vec![Plane::Real],
        LeafClass::Complex => vec![Plane::Real, Plane::Imag],
    }
}

fn scratch_specs(
    roles: &[ScratchRole],
    next_id: &mut usize,
    left_elements: usize,
    right_elements: usize,
    output_elements: usize,
) -> Vec<ScratchSpec> {
    roles
        .iter()
        .map(|role| {
            let elements = match role {
                ScratchRole::LeftSum => left_elements,
                ScratchRole::RightSum => right_elements,
                ScratchRole::Product1
                | ScratchRole::Product2
                | ScratchRole::Product3
                | ScratchRole::Product4 => output_elements,
            };
            let spec = ScratchSpec {
                id: ScratchId(*next_id),
                role: role.clone(),
                elements,
            };
            *next_id += 1;
            spec
        })
        .collect()
}

fn tensor_elements(spec: &TensorSpec) -> Result<usize, PlanError> {
    spec.shape
        .iter()
        .try_fold(1usize, |product, size| product.checked_mul(*size))
        .ok_or_else(|| {
            PlanError::InvalidNetwork("tensor element count overflows usize".to_string())
        })
}

fn sum_node_volume(plan: &StaticPlan) -> Result<u128, PlanError> {
    plan.nodes.iter().try_fold(0u128, |sum, node| {
        sum.checked_add(node.real_skeleton_volume)
            .ok_or_else(|| PlanError::InvalidNetwork("node volume sum overflows u128".to_string()))
    })
}

fn checked_add_volume(current: u128, node: &PlanNode) -> Result<u128, PlanError> {
    current
        .checked_add(node.real_skeleton_volume)
        .ok_or_else(|| PlanError::InvalidNetwork("kernel volume sum overflows u128".to_string()))
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

impl StaticPlan {
    pub fn validate(&self) -> Result<(), PlanError> {
        validate_sha256("tree_hash", &self.tree_hash)?;
        let leaf_count = self.leaf_values.len();
        if self.values.len() != leaf_count + self.nodes.len() {
            return Err(PlanError::InvalidTree {
                detail: format!(
                    "value count {} must equal leaf count {leaf_count} plus node count {}",
                    self.values.len(),
                    self.nodes.len()
                ),
            });
        }
        let expected_leaves: Vec<_> = (0..leaf_count).map(ValueId).collect();
        if self.leaf_values != expected_leaves {
            return Err(PlanError::InvalidTree {
                detail: "leaf ValueIds are not in original tensor order".to_string(),
            });
        }

        let mut sizes = HashMap::new();
        for (index, value) in self.values.iter().enumerate() {
            if value.id != ValueId(index) {
                return Err(PlanError::InvalidTree {
                    detail: format!("value {index} carries id {:?}", value.id),
                });
            }
            if value.tensor.modes.len() != value.tensor.shape.len() {
                return Err(PlanError::InvalidTree {
                    detail: format!("value {index} mode count differs from rank"),
                });
            }
            let mut unique = HashSet::new();
            for (mode, dimension) in value.tensor.modes.iter().zip(value.tensor.shape.iter()) {
                if !unique.insert(*mode) {
                    return Err(PlanError::InvalidTree {
                        detail: format!("value {index} repeats mode {mode}"),
                    });
                }
                if let Some(previous) = sizes.insert(*mode, *dimension) {
                    if previous != *dimension {
                        return Err(PlanError::InvalidTree {
                            detail: format!(
                                "mode {mode} has inconsistent dimensions {previous} and {dimension}"
                            ),
                        });
                    }
                }
            }
            if value.planes != vec![Plane::Real] && value.planes != vec![Plane::Real, Plane::Imag] {
                return Err(PlanError::InvalidTree {
                    detail: format!(
                        "value {index} has invalid semantic planes {:?}",
                        value.planes
                    ),
                });
            }
            match self.representation {
                Representation::RealSkeleton if value.planes != vec![Plane::Real] => {
                    return Err(PlanError::InvalidTree {
                        detail: format!("real-skeleton value {index} has an imaginary plane"),
                    });
                }
                Representation::Flat4M if value.planes != vec![Plane::Real, Plane::Imag] => {
                    return Err(PlanError::InvalidTree {
                        detail: format!("flat-4m value {index} does not have two planes"),
                    });
                }
                _ => {}
            }
        }

        let mut next_scratch = 0usize;
        let mut real_real_nodes = 0usize;
        let mut ride_left_nodes = 0usize;
        let mut ride_right_nodes = 0usize;
        let mut merge_3m_nodes = 0usize;
        let mut flat_4m_nodes = 0usize;
        let mut total_volume = 0u128;
        let mut real_real_volume = 0u128;
        let mut ride_volume = 0u128;
        let mut merge_volume = 0u128;
        for (node_index, node) in self.nodes.iter().enumerate() {
            if node.id != node_index {
                return Err(PlanError::InvalidTree {
                    detail: format!("node {node_index} carries id {}", node.id),
                });
            }
            let expected_output = ValueId(leaf_count + node_index);
            if node.output != expected_output {
                return Err(PlanError::InvalidTree {
                    detail: format!(
                        "node {node_index} output {:?} differs from postorder id {expected_output:?}",
                        node.output
                    ),
                });
            }
            if node.left.0 >= node.output.0 || node.right.0 >= node.output.0 {
                return Err(PlanError::InvalidTree {
                    detail: format!("node {node_index} references a non-topological input"),
                });
            }
            let left = self
                .values
                .get(node.left.0)
                .ok_or_else(|| PlanError::InvalidTree {
                    detail: format!("node {node_index} left value is out of range"),
                })?;
            let right = self
                .values
                .get(node.right.0)
                .ok_or_else(|| PlanError::InvalidTree {
                    detail: format!("node {node_index} right value is out of range"),
                })?;
            let output = self
                .values
                .get(node.output.0)
                .ok_or_else(|| PlanError::InvalidTree {
                    detail: format!("node {node_index} output value is out of range"),
                })?;
            validate_output_modes(&output.tensor.modes, &left.tensor, &right.tensor, &sizes)?;
            let geometry = plan_contraction(
                &left.tensor.modes,
                &left.tensor.shape,
                &right.tensor.modes,
                &right.tensor.shape,
                &output.tensor.modes,
            );
            if !geometry.left_trace.is_empty() || !geometry.right_trace.is_empty() {
                return Err(PlanError::UnsupportedTrace {
                    node: node_index,
                    left_trace_modes: geometry.left_trace,
                    right_trace_modes: geometry.right_trace,
                });
            }
            let expected_shape = output
                .tensor
                .modes
                .iter()
                .map(|mode| sizes[mode])
                .collect::<Vec<_>>();
            if output.tensor.shape != expected_shape {
                return Err(PlanError::InvalidTree {
                    detail: format!(
                        "node {node_index} output shape {:?} differs from {expected_shape:?}",
                        output.tensor.shape
                    ),
                });
            }
            let left_permutation = geometry.a_permutation(&left.tensor.modes);
            let right_permutation = geometry.b_permutation(&right.tensor.modes);
            let expected_contraction = ContractionSpec {
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
            };
            if node.contraction != expected_contraction {
                return Err(PlanError::InvalidTree {
                    detail: format!("node {node_index} contraction geometry is not canonical"),
                });
            }
            let expected_volume = [
                node.contraction.batch,
                node.contraction.m,
                node.contraction.k,
                node.contraction.n,
            ]
            .into_iter()
            .try_fold(1u128, |product, value| product.checked_mul(value as u128))
            .ok_or_else(|| PlanError::InvalidTree {
                detail: format!("node {node_index} volume overflows u128"),
            })?;
            if node.real_skeleton_volume != expected_volume {
                return Err(PlanError::InvalidTree {
                    detail: format!("node {node_index} has an incorrect contraction volume"),
                });
            }
            total_volume = total_volume.checked_add(expected_volume).ok_or_else(|| {
                PlanError::InvalidTree {
                    detail: "total contraction volume overflows u128".to_string(),
                }
            })?;

            let left_complex = left.planes.len() == 2;
            let right_complex = right.planes.len() == 2;
            let (expected_kind, expected_planes, expected_roles) = match self.representation {
                Representation::RealSkeleton => (KernelKind::RealReal, vec![Plane::Real], vec![]),
                Representation::Flat4M => (
                    KernelKind::Flat4M,
                    vec![Plane::Real, Plane::Imag],
                    vec![
                        ScratchRole::Product1,
                        ScratchRole::Product2,
                        ScratchRole::Product3,
                        ScratchRole::Product4,
                    ],
                ),
                Representation::RealifiedRank3 => match (left_complex, right_complex) {
                    (false, false) => (KernelKind::RealReal, vec![Plane::Real], vec![]),
                    (true, false) => (KernelKind::RideLeft, vec![Plane::Real, Plane::Imag], vec![]),
                    (false, true) => (
                        KernelKind::RideRight,
                        vec![Plane::Real, Plane::Imag],
                        vec![],
                    ),
                    (true, true) => (
                        KernelKind::Merge3M,
                        vec![Plane::Real, Plane::Imag],
                        vec![
                            ScratchRole::LeftSum,
                            ScratchRole::RightSum,
                            ScratchRole::Product1,
                            ScratchRole::Product2,
                            ScratchRole::Product3,
                        ],
                    ),
                },
            };
            if node.kind != expected_kind || output.planes != expected_planes {
                return Err(PlanError::InvalidTree {
                    detail: format!("node {node_index} kernel/plane transition is invalid"),
                });
            }
            if node.scratch.len() != expected_roles.len() {
                return Err(PlanError::InvalidTree {
                    detail: format!("node {node_index} has an invalid scratch count"),
                });
            }
            let left_elements = tensor_elements(&left.tensor)?;
            let right_elements = tensor_elements(&right.tensor)?;
            let output_elements = tensor_elements(&output.tensor)?;
            for (scratch, role) in node.scratch.iter().zip(expected_roles) {
                let expected_elements = match role {
                    ScratchRole::LeftSum => left_elements,
                    ScratchRole::RightSum => right_elements,
                    ScratchRole::Product1
                    | ScratchRole::Product2
                    | ScratchRole::Product3
                    | ScratchRole::Product4 => output_elements,
                };
                if scratch.id != ScratchId(next_scratch)
                    || scratch.role != role
                    || scratch.elements != expected_elements
                {
                    return Err(PlanError::InvalidTree {
                        detail: format!("node {node_index} scratch layout is invalid"),
                    });
                }
                next_scratch += 1;
            }

            match node.kind {
                KernelKind::RealReal => {
                    real_real_nodes += 1;
                    real_real_volume =
                        real_real_volume
                            .checked_add(expected_volume)
                            .ok_or_else(|| PlanError::InvalidTree {
                                detail: "real-real volume overflows u128".to_string(),
                            })?;
                }
                KernelKind::RideLeft => {
                    ride_left_nodes += 1;
                    ride_volume = ride_volume.checked_add(expected_volume).ok_or_else(|| {
                        PlanError::InvalidTree {
                            detail: "ride volume overflows u128".to_string(),
                        }
                    })?;
                }
                KernelKind::RideRight => {
                    ride_right_nodes += 1;
                    ride_volume = ride_volume.checked_add(expected_volume).ok_or_else(|| {
                        PlanError::InvalidTree {
                            detail: "ride volume overflows u128".to_string(),
                        }
                    })?;
                }
                KernelKind::Merge3M => {
                    merge_3m_nodes += 1;
                    merge_volume = merge_volume.checked_add(expected_volume).ok_or_else(|| {
                        PlanError::InvalidTree {
                            detail: "merge volume overflows u128".to_string(),
                        }
                    })?;
                }
                KernelKind::Flat4M => flat_4m_nodes += 1,
            }
        }

        let root = self
            .values
            .get(self.output.0)
            .ok_or_else(|| PlanError::InvalidTree {
                detail: "plan output ValueId is out of range".to_string(),
            })?;
        if !root.tensor.modes.is_empty() || !root.tensor.shape.is_empty() {
            return Err(PlanError::NonScalarOutput(root.tensor.modes.clone()));
        }
        if !self.nodes.is_empty() && self.output != ValueId(self.values.len() - 1) {
            return Err(PlanError::InvalidTree {
                detail: "plan output is not the root postorder value".to_string(),
            });
        }
        validate_reachability(self, leaf_count)?;
        if self.stats.real_leaf_count + self.stats.complex_leaf_count != leaf_count {
            return Err(PlanError::InvalidTree {
                detail: "leaf class counts do not match leaf values".to_string(),
            });
        }
        if self.stats.real_real_nodes != real_real_nodes
            || self.stats.ride_left_nodes != ride_left_nodes
            || self.stats.ride_right_nodes != ride_right_nodes
            || self.stats.merge_3m_nodes != merge_3m_nodes
            || self.stats.flat_4m_nodes != flat_4m_nodes
            || self.stats.real_skeleton_volume != total_volume
        {
            return Err(PlanError::InvalidTree {
                detail: "plan statistics do not match node topology".to_string(),
            });
        }
        let expected_real_matmul = real_real_volume
            .checked_add(
                ride_volume
                    .checked_mul(2)
                    .ok_or_else(|| PlanError::InvalidTree {
                        detail: "ride accounting overflows u128".to_string(),
                    })?,
            )
            .and_then(|value| {
                merge_volume
                    .checked_mul(3)
                    .and_then(|merge| value.checked_add(merge))
            })
            .and_then(|value| {
                if flat_4m_nodes == 0 {
                    Some(value)
                } else {
                    total_volume.checked_mul(4)
                }
            })
            .ok_or_else(|| PlanError::InvalidTree {
                detail: "real matmul accounting overflows u128".to_string(),
            })?;
        if self.stats.real_matmul_volume != expected_real_matmul {
            return Err(PlanError::InvalidTree {
                detail: "real matmul volume is inconsistent".to_string(),
            });
        }
        match self.representation {
            Representation::RealSkeleton | Representation::Flat4M => {
                if self.stats.realification_cost.is_some() {
                    return Err(PlanError::InvalidTree {
                        detail: "baseline plan unexpectedly has selective cost data".to_string(),
                    });
                }
            }
            Representation::RealifiedRank3 => {
                if total_volume == 0 {
                    return Err(PlanError::InvalidTree {
                        detail: "selective plan has zero contraction volume".to_string(),
                    });
                }
                let expected_merges = self.stats.complex_leaf_count.saturating_sub(1);
                if merge_3m_nodes != expected_merges {
                    return Err(PlanError::InvalidTree {
                        detail: format!(
                            "selective merge count {merge_3m_nodes} differs from {expected_merges}"
                        ),
                    });
                }
                validate_realification_cost(
                    self.stats.realification_cost.as_ref(),
                    real_real_volume,
                    ride_volume,
                    merge_volume,
                    total_volume,
                )?;
            }
        }
        let expected_hash = plan_hash(self)?;
        if self.plan_hash != expected_hash {
            return Err(PlanError::Hash(format!(
                "plan_hash mismatch: stored {}, expected {expected_hash}",
                self.plan_hash
            )));
        }
        Ok(())
    }
}

impl PlanBundle {
    pub fn validate(&self) -> Result<(), PlanError> {
        if self.format != "omeinsum-static-plan-v1" {
            return Err(PlanError::InvalidFormat(format!(
                "expected omeinsum-static-plan-v1, got {:?}",
                self.format
            )));
        }
        if !self.realness_tol.is_finite() || self.realness_tol <= 0.0 {
            return Err(PlanError::InvalidNetwork(
                "realness tolerance must be positive and finite".to_string(),
            ));
        }
        validate_sha256("tree_hash", &self.tree_hash)?;
        let mut real_count = 0usize;
        let mut complex_count = 0usize;
        for (index, input) in self.inputs.tensors.iter().enumerate() {
            let elements = tensor_elements(&input.spec)?;
            if input.real.len() != elements || input.imag.len() != elements {
                return Err(PlanError::InvalidTensor {
                    index,
                    detail: "input plane lengths do not match shape".to_string(),
                });
            }
            if input
                .real
                .iter()
                .chain(input.imag.iter())
                .any(|value| !value.is_finite())
                || !input.imag_max.is_finite()
                || input.imag_max < 0.0
            {
                return Err(PlanError::InvalidTensor {
                    index,
                    detail: "input contains non-finite data or imag_max".to_string(),
                });
            }
            match input.class {
                LeafClass::Real => {
                    real_count += 1;
                    if input.imag_max > self.realness_tol
                        || input.imag.iter().any(|value| *value != 0.0)
                    {
                        return Err(PlanError::InvalidTensor {
                            index,
                            detail: "real leaf classification or zero plane is invalid".to_string(),
                        });
                    }
                }
                LeafClass::Complex => {
                    complex_count += 1;
                    let observed = input
                        .imag
                        .iter()
                        .map(|value| value.abs())
                        .fold(0.0_f64, f64::max);
                    if input.imag_max <= self.realness_tol || observed != input.imag_max {
                        return Err(PlanError::InvalidTensor {
                            index,
                            detail: "complex leaf imag_max is inconsistent".to_string(),
                        });
                    }
                }
            }
        }

        let plans = [&self.real_skeleton, &self.flat_4m, &self.realified_rank3];
        if self.real_skeleton.representation != Representation::RealSkeleton
            || self.flat_4m.representation != Representation::Flat4M
            || self.realified_rank3.representation != Representation::RealifiedRank3
        {
            return Err(PlanError::InvalidNetwork(
                "plan representations are in the wrong bundle fields".to_string(),
            ));
        }
        for plan in plans {
            plan.validate()?;
            if plan.tree_hash != self.tree_hash {
                return Err(PlanError::Hash(
                    "bundle plans do not share the declared tree hash".to_string(),
                ));
            }
            if plan.leaf_values.len() != self.inputs.tensors.len() {
                return Err(PlanError::InvalidNetwork(
                    "plan leaf count differs from input count".to_string(),
                ));
            }
            if plan.stats.real_leaf_count != real_count
                || plan.stats.complex_leaf_count != complex_count
            {
                return Err(PlanError::InvalidNetwork(
                    "plan leaf statistics differ from input classes".to_string(),
                ));
            }
            for (input, value) in self.inputs.tensors.iter().zip(&plan.values) {
                if input.spec != value.tensor {
                    return Err(PlanError::InvalidNetwork(
                        "plan leaf TensorSpec differs from immutable inputs".to_string(),
                    ));
                }
            }
        }
        if !same_topology(&self.real_skeleton, &self.flat_4m)
            || !same_topology(&self.real_skeleton, &self.realified_rank3)
        {
            return Err(PlanError::InvalidTree {
                detail: "plan variants do not share identical node topology".to_string(),
            });
        }
        let hashes = [
            &self.real_skeleton.plan_hash,
            &self.flat_4m.plan_hash,
            &self.realified_rank3.plan_hash,
        ];
        if hashes[0] == hashes[1] || hashes[0] == hashes[2] || hashes[1] == hashes[2] {
            return Err(PlanError::Hash(
                "plan variants must have distinct plan hashes".to_string(),
            ));
        }
        Ok(())
    }
}

fn validate_sha256(name: &str, value: &str) -> Result<(), PlanError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(PlanError::Hash(format!(
            "{name} is not a lowercase SHA-256 hex digest: {value:?}"
        )));
    }
    Ok(())
}

fn validate_reachability(plan: &StaticPlan, leaf_count: usize) -> Result<(), PlanError> {
    let mut reached_values = vec![false; plan.values.len()];
    let mut reached_nodes = vec![false; plan.nodes.len()];
    let mut stack = vec![plan.output];
    while let Some(value) = stack.pop() {
        if value.0 >= reached_values.len() {
            return Err(PlanError::InvalidTree {
                detail: "reachable value is out of range".to_string(),
            });
        }
        if std::mem::replace(&mut reached_values[value.0], true) {
            continue;
        }
        if value.0 >= leaf_count {
            let node_index = value.0 - leaf_count;
            let node = plan
                .nodes
                .get(node_index)
                .ok_or_else(|| PlanError::InvalidTree {
                    detail: "intermediate value has no producer".to_string(),
                })?;
            reached_nodes[node_index] = true;
            stack.push(node.left);
            stack.push(node.right);
        }
    }
    if reached_values.iter().any(|reached| !reached) || reached_nodes.iter().any(|reached| !reached)
    {
        return Err(PlanError::InvalidTree {
            detail: "plan contains values or nodes disconnected from the root".to_string(),
        });
    }
    Ok(())
}

fn validate_realification_cost(
    cost: Option<&RealificationCost>,
    real_real_volume: u128,
    ride_volume: u128,
    merge_volume: u128,
    total_volume: u128,
) -> Result<(), PlanError> {
    let cost = cost.ok_or_else(|| PlanError::InvalidTree {
        detail: "selective plan is missing realification cost".to_string(),
    })?;
    let denominator = total_volume as f64;
    let expected_real = real_real_volume as f64 / denominator;
    let expected_ride = ride_volume as f64 / denominator;
    let expected_merge = merge_volume as f64 / denominator;
    let expected_overhead = 1.0 + expected_ride + 2.0 * expected_merge;
    if cost.real_real_volume != real_real_volume
        || cost.ride_volume != ride_volume
        || cost.merge_volume != merge_volume
        || (cost.real_real_fraction - expected_real).abs() > 1e-15
        || (cost.ride_fraction - expected_ride).abs() > 1e-15
        || (cost.merge_fraction - expected_merge).abs() > 1e-15
        || (cost.predicted_arithmetic_overhead - expected_overhead).abs() > 1e-15
        || (cost.real_real_fraction + cost.ride_fraction + cost.merge_fraction - 1.0).abs() > 1e-12
    {
        return Err(PlanError::InvalidTree {
            detail: "realification cost accounting is inconsistent".to_string(),
        });
    }
    Ok(())
}

fn same_topology(left: &StaticPlan, right: &StaticPlan) -> bool {
    left.tree_hash == right.tree_hash
        && left.leaf_values == right.leaf_values
        && left.output == right.output
        && left.values.len() == right.values.len()
        && left
            .values
            .iter()
            .zip(&right.values)
            .all(|(left, right)| left.id == right.id && left.tensor == right.tensor)
        && left.nodes.len() == right.nodes.len()
        && left.nodes.iter().zip(&right.nodes).all(|(left, right)| {
            left.id == right.id
                && left.left == right.left
                && left.right == right.right
                && left.output == right.output
                && left.contraction == right.contraction
                && left.real_skeleton_volume == right.real_skeleton_volume
        })
}
