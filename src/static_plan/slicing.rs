#![allow(clippy::result_large_err)] // Frozen execution diagnostics intentionally retain context.

use std::collections::{HashMap, HashSet};

use super::{
    BinaryContractionTree, ComplexNetwork, ComplexTensor, ComplexValue, ExecutionError, InputSet,
    InputTensor, InputUpdate, LeafClass, PlanBundle, PlanError, PreparedExecutable, Representation,
    SliceAssignmentOrder, SliceSpec, SlicedPlanBundle, SlicedVolume, StaticPlan, TensorSpec,
};

use super::builder::{configure_flat_4m, configure_real_skeleton, configure_selective};

pub fn build_sliced_plan_bundle(
    source: &PlanBundle,
    modes: &[i32],
) -> Result<SlicedPlanBundle, PlanError> {
    source.validate()?;
    let dimensions = slice_dimensions(&source.real_skeleton, modes)?;
    let slice_count = dimensions
        .iter()
        .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
        .ok_or_else(|| PlanError::InvalidNetwork("slice count overflows usize".to_string()))?;
    let slice = SliceSpec {
        modes: modes.to_vec(),
        dimensions,
        assignment_order: SliceAssignmentOrder::BinaryReflectedGray,
        slice_count,
    };
    gray_assignments(&slice)?;
    let zero_assignment = vec![0usize; modes.len()];
    let reduced_inputs = slice_inputs(&source.inputs, &slice, &zero_assignment)?;
    let tree = reconstruct_tree(&source.real_skeleton)?;
    let size_dict = reduced_size_dict(&reduced_inputs)?;
    let network = ComplexNetwork {
        tensors: reduced_inputs
            .tensors
            .iter()
            .map(|tensor| ComplexTensor {
                spec: tensor.spec.clone(),
                real: tensor.real.clone(),
                imag: tensor.imag.clone(),
            })
            .collect(),
        output_modes: vec![],
        size_dict,
        tree,
    };
    let geometry = super::build_geometry_plan(&network)?;
    if geometry.tree_hash != source.tree_hash {
        return Err(PlanError::InvalidTree {
            detail: format!(
                "reconstructed tree hash {} differs from source {}",
                geometry.tree_hash, source.tree_hash
            ),
        });
    }
    let real_leaf_count = source
        .inputs
        .tensors
        .iter()
        .filter(|tensor| tensor.class == LeafClass::Real)
        .count();
    let complex_leaf_count = source.inputs.tensors.len() - real_leaf_count;
    let real_skeleton =
        configure_real_skeleton(geometry.clone(), real_leaf_count, complex_leaf_count)?;
    let flat_4m = configure_flat_4m(geometry.clone(), real_leaf_count, complex_leaf_count)?;
    let realified_rank3 = configure_selective(
        geometry,
        &reduced_inputs,
        real_leaf_count,
        complex_leaf_count,
    )?;
    let reduced = PlanBundle {
        format: "omeinsum-static-plan-v1".to_string(),
        realness_tol: source.realness_tol,
        leaf_preprocessing: source.leaf_preprocessing,
        phase_canonicalization: source.phase_canonicalization.clone(),
        tree_hash: source.tree_hash.clone(),
        inputs: reduced_inputs,
        real_skeleton,
        flat_4m,
        realified_rank3,
    };
    reduced.validate()?;
    let source_plans = [
        &source.real_skeleton,
        &source.flat_4m,
        &source.realified_rank3,
    ];
    let reduced_plans = [
        &reduced.real_skeleton,
        &reduced.flat_4m,
        &reduced.realified_rank3,
    ];
    let source_plan_hashes = source_plans
        .iter()
        .map(|plan| (plan.representation.clone(), plan.plan_hash.clone()))
        .collect();
    let aggregate_volumes = source_plans
        .iter()
        .zip(reduced_plans)
        .map(|(source_plan, reduced_plan)| {
            let sliced_real_matmul_volume = reduced_plan
                .stats
                .real_matmul_volume
                .checked_mul(slice_count as u128)
                .ok_or_else(|| {
                    PlanError::InvalidNetwork(
                        "aggregate sliced real matmul volume overflows u128".to_string(),
                    )
                })?;
            Ok(SlicedVolume {
                representation: source_plan.representation.clone(),
                source_real_matmul_volume: source_plan.stats.real_matmul_volume,
                sliced_real_matmul_volume,
            })
        })
        .collect::<Result<Vec<_>, PlanError>>()?;
    let sliced = SlicedPlanBundle {
        format: "omeinsum-sliced-plan-v1".to_string(),
        source_tree_hash: source.tree_hash.clone(),
        source_plan_hashes,
        source_inputs: source.inputs.clone(),
        slice,
        reduced,
        aggregate_volumes,
    };
    sliced.validate()?;
    Ok(sliced)
}

pub fn gray_assignments(spec: &SliceSpec) -> Result<Vec<Vec<usize>>, PlanError> {
    if spec.modes.is_empty() || spec.modes.len() != spec.dimensions.len() {
        return Err(PlanError::InvalidNetwork(
            "slice modes and dimensions must have the same nonzero length".to_string(),
        ));
    }
    if spec.dimensions.iter().any(|dimension| *dimension != 2) {
        return Err(PlanError::InvalidNetwork(
            "binary-reflected Gray order requires dimension-two slice modes".to_string(),
        ));
    }
    if spec.modes.iter().collect::<HashSet<_>>().len() != spec.modes.len() {
        return Err(PlanError::InvalidNetwork(
            "slice modes must be unique".to_string(),
        ));
    }
    let expected_count =
        1usize
            .checked_shl(u32::try_from(spec.modes.len()).map_err(|_| {
                PlanError::InvalidNetwork("slice-mode count overflows u32".to_string())
            })?)
            .ok_or_else(|| PlanError::InvalidNetwork("slice count overflows usize".to_string()))?;
    if spec.slice_count != expected_count {
        return Err(PlanError::InvalidNetwork(format!(
            "slice_count {} differs from expected {expected_count}",
            spec.slice_count
        )));
    }
    match spec.assignment_order {
        SliceAssignmentOrder::BinaryReflectedGray => {
            let assignments = (0..expected_count)
                .map(|index| {
                    let gray = index ^ (index >> 1);
                    (0..spec.modes.len())
                        .map(|axis| (gray >> (spec.modes.len() - axis - 1)) & 1)
                        .collect()
                })
                .collect();
            Ok(assignments)
        }
    }
}

pub struct SlicedExecutable<E> {
    inner: E,
    update_batches: Vec<Vec<InputUpdate<f64>>>,
    output: Option<ComplexValue>,
}

impl<E: PreparedExecutable> SlicedExecutable<E> {
    pub fn new(
        inner: E,
        source_inputs: &InputSet<f64>,
        sliced_plan: &SlicedPlanBundle,
    ) -> Result<Self, ExecutionError> {
        sliced_plan
            .validate()
            .map_err(|error| ExecutionError::InvalidPlan(error.to_string()))?;
        if source_inputs != &sliced_plan.source_inputs {
            return Err(ExecutionError::InvalidPlan(
                "sliced executable source inputs differ from the sliced plan".to_string(),
            ));
        }
        let assignments = gray_assignments(&sliced_plan.slice)
            .map_err(|error| ExecutionError::InvalidPlan(error.to_string()))?;
        let leaf_indices_by_mode = sliced_plan
            .slice
            .modes
            .iter()
            .map(|mode| {
                let indices = source_inputs
                    .tensors
                    .iter()
                    .enumerate()
                    .filter_map(|(index, tensor)| tensor.spec.modes.contains(mode).then_some(index))
                    .collect::<Vec<_>>();
                if indices.is_empty() {
                    Err(ExecutionError::InvalidPlan(format!(
                        "slice mode {mode} is absent from source leaves"
                    )))
                } else {
                    Ok(indices)
                }
            })
            .collect::<Result<Vec<_>, ExecutionError>>()?;
        let declared_dimensions = sliced_plan
            .slice
            .modes
            .iter()
            .copied()
            .zip(sliced_plan.slice.dimensions.iter().copied())
            .collect::<HashMap<_, _>>();
        let mut previous = assignments.last().cloned().ok_or_else(|| {
            ExecutionError::InvalidPlan("sliced assignment schedule is empty".to_string())
        })?;
        let mut update_batches = Vec::with_capacity(assignments.len());
        for current in &assignments {
            let changed_axes = previous
                .iter()
                .zip(current)
                .enumerate()
                .filter_map(|(axis, (before, after))| (before != after).then_some(axis))
                .collect::<Vec<_>>();
            if changed_axes.len() != 1 {
                return Err(ExecutionError::InvalidPlan(format!(
                    "cyclic Gray edge changes {} modes; expected exactly one",
                    changed_axes.len()
                )));
            }
            let changed_axis = changed_axes[0];
            let selected = sliced_plan
                .slice
                .modes
                .iter()
                .copied()
                .zip(current.iter().copied())
                .collect::<HashMap<_, _>>();
            let updates = leaf_indices_by_mode[changed_axis]
                .iter()
                .map(|index| {
                    let tensor = slice_tensor(
                        *index,
                        &source_inputs.tensors[*index],
                        &selected,
                        &declared_dimensions,
                    )
                    .map_err(|error| ExecutionError::InvalidPlan(error.to_string()))?;
                    Ok(InputUpdate {
                        index: *index,
                        tensor,
                    })
                })
                .collect::<Result<Vec<_>, ExecutionError>>()?;
            update_batches.push(updates);
            previous.clone_from(current);
        }
        Ok(Self {
            inner,
            update_batches,
            output: None,
        })
    }

    pub fn execution_mode_label(&self) -> &'static str {
        "sliced-host-f64-accumulated"
    }

    pub fn inner(&self) -> &E {
        &self.inner
    }
}

impl<E: PreparedExecutable> PreparedExecutable for SlicedExecutable<E> {
    fn representation(&self) -> Representation {
        self.inner.representation()
    }

    fn update_inputs(&mut self, updates: &[InputUpdate<f64>]) -> Result<(), ExecutionError> {
        if updates.is_empty() {
            Ok(())
        } else {
            Err(ExecutionError::Unsupported(
                "a sliced executable owns its frozen input-update schedule".to_string(),
            ))
        }
    }

    fn enqueue(&mut self) -> Result<(), ExecutionError> {
        let mut sum = ComplexValue { re: 0.0, im: 0.0 };
        for updates in &self.update_batches {
            self.inner.update_inputs(updates)?;
            self.inner.enqueue()?;
            self.inner.synchronize()?;
            let value = self.inner.output()?;
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(ExecutionError::NonFiniteOutput(value));
            }
            sum.re += value.re;
            sum.im += value.im;
            if !sum.re.is_finite() || !sum.im.is_finite() {
                return Err(ExecutionError::NonFiniteOutput(sum));
            }
        }
        self.output = Some(sum);
        Ok(())
    }

    fn synchronize(&mut self) -> Result<(), ExecutionError> {
        Ok(())
    }

    fn output(&mut self) -> Result<ComplexValue, ExecutionError> {
        self.output.ok_or_else(|| {
            ExecutionError::InvalidPlan(
                "sliced output requested before a complete contraction".to_string(),
            )
        })
    }
}

impl SlicedPlanBundle {
    pub fn validate(&self) -> Result<(), PlanError> {
        if self.format != "omeinsum-sliced-plan-v1" {
            return Err(PlanError::InvalidFormat(format!(
                "expected omeinsum-sliced-plan-v1, got {:?}",
                self.format
            )));
        }
        self.reduced.validate()?;
        gray_assignments(&self.slice)?;
        if self.source_tree_hash != self.reduced.tree_hash {
            return Err(PlanError::InvalidTree {
                detail: "sliced source and reduced tree hashes differ".to_string(),
            });
        }
        let expected_representations = [
            Representation::RealSkeleton,
            Representation::Flat4M,
            Representation::RealifiedRank3,
        ];
        if self.source_plan_hashes.len() != expected_representations.len()
            || self.aggregate_volumes.len() != expected_representations.len()
        {
            return Err(PlanError::InvalidNetwork(
                "sliced plan identity set is incomplete".to_string(),
            ));
        }
        let reduced_plans = [
            &self.reduced.real_skeleton,
            &self.reduced.flat_4m,
            &self.reduced.realified_rank3,
        ];
        for (index, representation) in expected_representations.iter().enumerate() {
            let (stored_representation, plan_hash) = &self.source_plan_hashes[index];
            if stored_representation != representation || !is_sha256(plan_hash) {
                return Err(PlanError::InvalidNetwork(format!(
                    "source plan identity {index} is invalid"
                )));
            }
            let volume = &self.aggregate_volumes[index];
            let expected_sliced = reduced_plans[index]
                .stats
                .real_matmul_volume
                .checked_mul(self.slice.slice_count as u128)
                .ok_or_else(|| {
                    PlanError::InvalidNetwork(
                        "aggregate sliced real matmul volume overflows u128".to_string(),
                    )
                })?;
            if volume.representation != *representation
                || volume.source_real_matmul_volume == 0
                || volume.sliced_real_matmul_volume != expected_sliced
            {
                return Err(PlanError::InvalidNetwork(format!(
                    "aggregate sliced volume {index} is invalid"
                )));
            }
        }
        let expected_zero = slice_inputs(
            &self.source_inputs,
            &self.slice,
            &vec![0usize; self.slice.modes.len()],
        )?;
        if self.reduced.inputs != expected_zero {
            return Err(PlanError::InvalidNetwork(
                "reduced inputs differ from the zero slice".to_string(),
            ));
        }
        Ok(())
    }
}

fn slice_dimensions(plan: &StaticPlan, modes: &[i32]) -> Result<Vec<usize>, PlanError> {
    if modes.is_empty() {
        return Err(PlanError::InvalidNetwork(
            "at least one slice mode is required".to_string(),
        ));
    }
    if modes.iter().collect::<HashSet<_>>().len() != modes.len() {
        return Err(PlanError::InvalidNetwork(
            "slice modes must be unique".to_string(),
        ));
    }
    let mut sizes = HashMap::new();
    for value in &plan.values {
        for (mode, dimension) in value.tensor.modes.iter().zip(&value.tensor.shape) {
            if let Some(previous) = sizes.insert(*mode, *dimension) {
                if previous != *dimension {
                    return Err(PlanError::InvalidTree {
                        detail: format!("mode {mode} has inconsistent dimensions"),
                    });
                }
            }
        }
    }
    modes
        .iter()
        .map(|mode| {
            let dimension = sizes
                .get(mode)
                .copied()
                .ok_or_else(|| PlanError::InvalidNetwork(format!("slice mode {mode} is absent")))?;
            if dimension != 2 {
                return Err(PlanError::InvalidNetwork(format!(
                    "slice mode {mode} has dimension {dimension}; expected 2"
                )));
            }
            if !plan
                .nodes
                .iter()
                .any(|node| node.contraction.contracted_modes.contains(mode))
            {
                return Err(PlanError::InvalidNetwork(format!(
                    "slice mode {mode} is not an internal contracted mode"
                )));
            }
            Ok(dimension)
        })
        .collect()
}

fn reconstruct_tree(plan: &StaticPlan) -> Result<BinaryContractionTree, PlanError> {
    let mut trees = vec![None; plan.values.len()];
    for (tensor_index, value) in plan.leaf_values.iter().enumerate() {
        trees[value.0] = Some(BinaryContractionTree::Leaf { tensor_index });
    }
    for node in &plan.nodes {
        let left = trees[node.left.0]
            .clone()
            .ok_or_else(|| PlanError::InvalidTree {
                detail: format!("node {} left subtree is unavailable", node.id),
            })?;
        let right = trees[node.right.0]
            .clone()
            .ok_or_else(|| PlanError::InvalidTree {
                detail: format!("node {} right subtree is unavailable", node.id),
            })?;
        trees[node.output.0] = Some(BinaryContractionTree::Node {
            output_modes: plan.values[node.output.0].tensor.modes.clone(),
            left: Box::new(left),
            right: Box::new(right),
        });
    }
    trees[plan.output.0]
        .clone()
        .ok_or_else(|| PlanError::InvalidTree {
            detail: "plan output tree is unavailable".to_string(),
        })
}

fn reduced_size_dict(inputs: &InputSet<f64>) -> Result<Vec<(i32, usize)>, PlanError> {
    let mut sizes = HashMap::new();
    for (index, tensor) in inputs.tensors.iter().enumerate() {
        for (mode, dimension) in tensor.spec.modes.iter().zip(&tensor.spec.shape) {
            if let Some(previous) = sizes.insert(*mode, *dimension) {
                if previous != *dimension {
                    return Err(PlanError::InvalidTensor {
                        index,
                        detail: format!("mode {mode} has inconsistent reduced dimensions"),
                    });
                }
            }
        }
    }
    let mut sizes = sizes.into_iter().collect::<Vec<_>>();
    sizes.sort_unstable_by_key(|(mode, _)| *mode);
    Ok(sizes)
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

pub fn slice_inputs(
    source: &InputSet<f64>,
    spec: &SliceSpec,
    assignment: &[usize],
) -> Result<InputSet<f64>, PlanError> {
    gray_assignments(spec)?;
    if assignment.len() != spec.modes.len() {
        return Err(PlanError::InvalidNetwork(format!(
            "slice assignment has {} indices for {} modes",
            assignment.len(),
            spec.modes.len()
        )));
    }
    if let Some((axis, (index, dimension))) = assignment
        .iter()
        .zip(&spec.dimensions)
        .enumerate()
        .find(|(_, (index, dimension))| **index >= **dimension)
    {
        return Err(PlanError::InvalidNetwork(format!(
            "slice assignment axis {axis} index {index} is outside dimension {dimension}"
        )));
    }
    let selected = spec
        .modes
        .iter()
        .copied()
        .zip(assignment.iter().copied())
        .collect::<HashMap<_, _>>();
    let declared_dimensions = spec
        .modes
        .iter()
        .copied()
        .zip(spec.dimensions.iter().copied())
        .collect::<HashMap<_, _>>();
    let tensors = source
        .tensors
        .iter()
        .enumerate()
        .map(|(index, tensor)| slice_tensor(index, tensor, &selected, &declared_dimensions))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(InputSet { tensors })
}

fn slice_tensor(
    index: usize,
    source: &InputTensor<f64>,
    selected: &HashMap<i32, usize>,
    declared_dimensions: &HashMap<i32, usize>,
) -> Result<InputTensor<f64>, PlanError> {
    if !source
        .spec
        .modes
        .iter()
        .any(|mode| selected.contains_key(mode))
    {
        return Ok(source.clone());
    }
    if source.spec.modes.len() != source.spec.shape.len() {
        return Err(PlanError::InvalidTensor {
            index,
            detail: "mode count differs from rank".to_string(),
        });
    }
    for (mode, source_dimension) in source.spec.modes.iter().zip(&source.spec.shape) {
        if let Some(declared_dimension) = declared_dimensions.get(mode) {
            if source_dimension != declared_dimension {
                return Err(PlanError::InvalidTensor {
                    index,
                    detail: format!(
                        "slice mode {mode} has source dimension {source_dimension} \
                         but the slice spec declares {declared_dimension}"
                    ),
                });
            }
        }
    }
    let source_elements =
        checked_elements(&source.spec.shape).ok_or_else(|| PlanError::InvalidTensor {
            index,
            detail: "source shape product overflows usize".to_string(),
        })?;
    if source.real.len() != source_elements || source.imag.len() != source_elements {
        return Err(PlanError::InvalidTensor {
            index,
            detail: "input plane lengths do not match shape".to_string(),
        });
    }
    let target_shape = source
        .spec
        .modes
        .iter()
        .zip(&source.spec.shape)
        .map(|(mode, dimension)| {
            if selected.contains_key(mode) {
                1
            } else {
                *dimension
            }
        })
        .collect::<Vec<_>>();
    let target_elements =
        checked_elements(&target_shape).ok_or_else(|| PlanError::InvalidTensor {
            index,
            detail: "sliced shape product overflows usize".to_string(),
        })?;
    let source_strides = column_major_strides(&source.spec.shape);
    let target_strides = column_major_strides(&target_shape);
    let mut real = Vec::with_capacity(target_elements);
    let mut imag = Vec::with_capacity(target_elements);
    for linear in 0..target_elements {
        let mut source_offset = 0usize;
        for axis in 0..target_shape.len() {
            let target_coordinate = (linear / target_strides[axis]) % target_shape[axis];
            let coordinate = selected
                .get(&source.spec.modes[axis])
                .copied()
                .unwrap_or(target_coordinate);
            source_offset = source_offset
                .checked_add(
                    coordinate
                        .checked_mul(source_strides[axis])
                        .ok_or_else(|| PlanError::InvalidTensor {
                            index,
                            detail: "sliced source offset overflows usize".to_string(),
                        })?,
                )
                .ok_or_else(|| PlanError::InvalidTensor {
                    index,
                    detail: "sliced source offset overflows usize".to_string(),
                })?;
        }
        real.push(source.real[source_offset]);
        imag.push(source.imag[source_offset]);
    }
    let imag_max = imag.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
    let classification_imag_max = match source.class {
        LeafClass::Real => None,
        LeafClass::Complex => Some(source.classification_imag_max.unwrap_or(source.imag_max)),
    };
    Ok(InputTensor {
        spec: TensorSpec {
            modes: source.spec.modes.clone(),
            shape: target_shape,
        },
        real,
        imag,
        class: source.class.clone(),
        imag_max,
        classification_imag_max,
    })
}

fn checked_elements(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1usize, |product, dimension| product.checked_mul(*dimension))
}

fn column_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut stride = 1usize;
    let mut strides = vec![1usize; shape.len()];
    for (axis, dimension) in shape.iter().enumerate() {
        strides[axis] = stride;
        stride *= *dimension;
    }
    strides
}
