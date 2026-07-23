#![allow(clippy::result_large_err)] // Public execution APIs use the frozen diagnostic schema.

use std::ops::{Add, Mul, Sub};

use num_traits::{One, Zero};

use crate::algebra::{Algebra, Scalar, Standard};
use crate::{BackendScalar, Cpu, Tensor};

use super::{
    ComplexValue, ExecutionError, InputSet, KernelKind, PlanNode, Plane, PreparedExecutable,
    Representation, ScratchRole, StaticPlan, TensorSpec,
};

pub fn prepare_cpu_f64(
    plan: &StaticPlan,
    inputs: &InputSet<f64>,
) -> Result<Box<dyn PreparedExecutable>, ExecutionError> {
    Ok(Box::new(CpuExecutable::<f64>::prepare(plan, inputs)?))
}

pub fn prepare_cpu_f32(
    plan: &StaticPlan,
    inputs: &InputSet<f64>,
) -> Result<Box<dyn PreparedExecutable>, ExecutionError> {
    Ok(Box::new(CpuExecutable::<f32>::prepare(plan, inputs)?))
}

trait CpuReal:
    Scalar
    + BackendScalar<Cpu>
    + Zero
    + One
    + PartialEq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
{
    fn from_f64(value: f64) -> Option<Self>;
    fn to_f64(self) -> f64;
}

impl CpuReal for f64 {
    fn from_f64(value: f64) -> Option<Self> {
        value.is_finite().then_some(value)
    }

    fn to_f64(self) -> f64 {
        self
    }
}

impl CpuReal for f32 {
    fn from_f64(value: f64) -> Option<Self> {
        let converted = value as f32;
        converted.is_finite().then_some(converted)
    }

    fn to_f64(self) -> f64 {
        self as f64
    }
}

#[derive(Clone)]
struct PlaneValue<T> {
    real: Vec<T>,
    imag: Option<Vec<T>>,
}

struct ComputedNode<T> {
    real: Vec<T>,
    imag: Option<Vec<T>>,
    scratch: Vec<(ScratchRole, Vec<T>)>,
}

struct CpuExecutable<T: CpuReal> {
    representation: Representation,
    plan: StaticPlan,
    values: Vec<PlaneValue<T>>,
    scratch: Vec<Vec<T>>,
    ready: bool,
}

impl<T: CpuReal> CpuExecutable<T>
where
    Standard<T>: Algebra<Scalar = T, Index = u32>,
{
    fn prepare(plan: &StaticPlan, inputs: &InputSet<f64>) -> Result<Self, ExecutionError> {
        plan.validate()
            .map_err(|error| ExecutionError::InvalidPlan(error.to_string()))?;
        if plan.leaf_values.len() != inputs.tensors.len() {
            return Err(ExecutionError::InvalidPlan(format!(
                "plan has {} leaves but input set has {} tensors",
                plan.leaf_values.len(),
                inputs.tensors.len()
            )));
        }

        let mut values = Vec::with_capacity(plan.values.len());
        for (index, value) in plan.values.iter().enumerate() {
            let elements = elements(&value.tensor)?;
            if index < inputs.tensors.len() {
                let input = &inputs.tensors[index];
                if input.spec != value.tensor
                    || input.real.len() != elements
                    || input.imag.len() != elements
                {
                    return Err(ExecutionError::InvalidPlan(format!(
                        "input tensor {index} does not match leaf ValueSpec"
                    )));
                }
                let real = convert_plane::<T>(&input.real, index, "real")?;
                let imag = if value.planes == vec![Plane::Real, Plane::Imag] {
                    Some(convert_plane::<T>(&input.imag, index, "imag")?)
                } else {
                    None
                };
                values.push(PlaneValue { real, imag });
            } else {
                values.push(PlaneValue {
                    real: vec![T::zero(); elements],
                    imag: (value.planes == vec![Plane::Real, Plane::Imag])
                        .then(|| vec![T::zero(); elements]),
                });
            }
        }

        let scratch_count = plan
            .nodes
            .iter()
            .flat_map(|node| node.scratch.iter())
            .map(|scratch| scratch.id.0 + 1)
            .max()
            .unwrap_or(0);
        let mut scratch = vec![Vec::new(); scratch_count];
        for spec in plan.nodes.iter().flat_map(|node| node.scratch.iter()) {
            scratch[spec.id.0] = vec![T::zero(); spec.elements];
        }

        Ok(Self {
            representation: plan.representation.clone(),
            plan: plan.clone(),
            values,
            scratch,
            ready: false,
        })
    }

    fn compute_node(&self, node: &PlanNode) -> Result<ComputedNode<T>, ExecutionError> {
        let left = &self.values[node.left.0];
        let right = &self.values[node.right.0];
        let left_spec = &self.plan.values[node.left.0].tensor;
        let right_spec = &self.plan.values[node.right.0].tensor;
        let output_spec = &self.plan.values[node.output.0].tensor;
        let contract = |left_plane: &[T], right_plane: &[T]| {
            contract_real(
                left_plane,
                left_spec,
                right_plane,
                right_spec,
                output_spec,
                node,
            )
        };
        match node.kind {
            KernelKind::RealReal => Ok(ComputedNode {
                real: contract(&left.real, &right.real)?,
                imag: None,
                scratch: vec![],
            }),
            KernelKind::RideLeft => Ok(ComputedNode {
                real: contract(&left.real, &right.real)?,
                imag: Some(contract(
                    required_imag(left, node.id, "left")?,
                    &right.real,
                )?),
                scratch: vec![],
            }),
            KernelKind::RideRight => Ok(ComputedNode {
                real: contract(&left.real, &right.real)?,
                imag: Some(contract(
                    &left.real,
                    required_imag(right, node.id, "right")?,
                )?),
                scratch: vec![],
            }),
            KernelKind::Merge3M => {
                let left_imag = required_imag(left, node.id, "left")?;
                let right_imag = required_imag(right, node.id, "right")?;
                let left_sum = zip_map(&left.real, left_imag, |real, imag| real + imag)?;
                let right_sum = zip_map(&right.real, right_imag, |real, imag| real + imag)?;
                let product1 = contract(&left.real, &right.real)?;
                let product2 = contract(left_imag, right_imag)?;
                let product3 = contract(&left_sum, &right_sum)?;
                let real = zip_map(&product1, &product2, |p1, p2| p1 - p2)?;
                let imag = zip3_map(&product3, &product1, &product2, |p3, p1, p2| p3 - p1 - p2)?;
                Ok(ComputedNode {
                    real,
                    imag: Some(imag),
                    scratch: vec![
                        (ScratchRole::LeftSum, left_sum),
                        (ScratchRole::RightSum, right_sum),
                        (ScratchRole::Product1, product1),
                        (ScratchRole::Product2, product2),
                        (ScratchRole::Product3, product3),
                    ],
                })
            }
            KernelKind::Flat4M => {
                let left_imag = required_imag(left, node.id, "left")?;
                let right_imag = required_imag(right, node.id, "right")?;
                let product1 = contract(&left.real, &right.real)?;
                let product2 = contract(left_imag, right_imag)?;
                let product3 = contract(&left.real, right_imag)?;
                let product4 = contract(left_imag, &right.real)?;
                let real = zip_map(&product1, &product2, |p1, p2| p1 - p2)?;
                let imag = zip_map(&product3, &product4, |p3, p4| p3 + p4)?;
                Ok(ComputedNode {
                    real,
                    imag: Some(imag),
                    scratch: vec![
                        (ScratchRole::Product1, product1),
                        (ScratchRole::Product2, product2),
                        (ScratchRole::Product3, product3),
                        (ScratchRole::Product4, product4),
                    ],
                })
            }
        }
    }

    fn commit_node(
        &mut self,
        node: &PlanNode,
        computed: ComputedNode<T>,
    ) -> Result<(), ExecutionError> {
        for (role, data) in computed.scratch {
            let spec = node
                .scratch
                .iter()
                .find(|scratch| scratch.role == role)
                .ok_or_else(|| {
                    ExecutionError::InvalidPlan(format!(
                        "node {} is missing scratch role {role:?}",
                        node.id
                    ))
                })?;
            self.scratch[spec.id.0].copy_from_slice(&data);
        }
        let output = &mut self.values[node.output.0];
        output.real.copy_from_slice(&computed.real);
        match (&mut output.imag, computed.imag) {
            (Some(target), Some(source)) => target.copy_from_slice(&source),
            (None, None) => {}
            _ => {
                return Err(ExecutionError::InvalidPlan(format!(
                    "node {} output plane allocation is inconsistent",
                    node.id
                )));
            }
        }
        Ok(())
    }
}

impl<T: CpuReal> PreparedExecutable for CpuExecutable<T>
where
    Standard<T>: Algebra<Scalar = T, Index = u32>,
{
    fn representation(&self) -> Representation {
        self.representation.clone()
    }

    fn enqueue(&mut self) -> Result<(), ExecutionError> {
        for index in 0..self.plan.nodes.len() {
            let node = self.plan.nodes[index].clone();
            let computed = self.compute_node(&node)?;
            self.commit_node(&node, computed)?;
        }
        self.ready = true;
        Ok(())
    }

    fn synchronize(&mut self) -> Result<(), ExecutionError> {
        Ok(())
    }

    fn output(&mut self) -> Result<ComplexValue, ExecutionError> {
        if !self.ready {
            return Err(ExecutionError::InvalidPlan(
                "output requested before enqueue".to_string(),
            ));
        }
        let output = &self.values[self.plan.output.0];
        if output.real.len() != 1 {
            return Err(ExecutionError::InvalidPlan(
                "prepared CPU output is not scalar".to_string(),
            ));
        }
        let value = ComplexValue {
            re: output.real[0].to_f64(),
            im: output
                .imag
                .as_ref()
                .map_or(0.0, |imaginary| imaginary[0].to_f64()),
        };
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(ExecutionError::NonFiniteOutput(value));
        }
        Ok(value)
    }
}

fn required_imag<'a, T>(
    value: &'a PlaneValue<T>,
    node_id: usize,
    side: &str,
) -> Result<&'a [T], ExecutionError> {
    value.imag.as_deref().ok_or_else(|| {
        ExecutionError::InvalidPlan(format!("node {node_id} expects an imaginary {side} plane"))
    })
}

fn convert_plane<T: CpuReal>(
    values: &[f64],
    tensor_index: usize,
    plane: &str,
) -> Result<Vec<T>, ExecutionError> {
    values
        .iter()
        .enumerate()
        .map(|(position, value)| {
            T::from_f64(*value).ok_or_else(|| {
                ExecutionError::InvalidPlan(format!(
                    "input tensor {tensor_index} {plane}[{position}] cannot be represented by the CPU dtype"
                ))
            })
        })
        .collect()
}

fn contract_real<T: CpuReal>(
    left: &[T],
    left_spec: &TensorSpec,
    right: &[T],
    right_spec: &TensorSpec,
    output_spec: &TensorSpec,
    node: &PlanNode,
) -> Result<Vec<T>, ExecutionError>
where
    Standard<T>: Algebra<Scalar = T, Index = u32>,
{
    let left_packed =
        permute_col_major(left, &left_spec.shape, &node.contraction.left_permutation)?;
    let right_packed = permute_col_major(
        right,
        &right_spec.shape,
        &node.contraction.right_permutation,
    )?;
    let left_tensor = Tensor::<T, Cpu>::from_data(
        &left_packed,
        &[
            node.contraction.m,
            node.contraction.k,
            node.contraction.batch,
        ],
    );
    let right_tensor = Tensor::<T, Cpu>::from_data(
        &right_packed,
        &[
            node.contraction.k,
            node.contraction.n,
            node.contraction.batch,
        ],
    );
    let product = left_tensor.contract_binary::<Standard<T>>(
        &right_tensor,
        &[0, 1, 3],
        &[1, 2, 3],
        &[0, 2, 3],
    );
    let canonical = product.to_vec();
    let canonical_modes: Vec<_> = node
        .contraction
        .left_modes
        .iter()
        .chain(node.contraction.right_modes.iter())
        .chain(node.contraction.batch_modes.iter())
        .copied()
        .collect();
    let canonical_shape = canonical_modes
        .iter()
        .map(|mode| {
            output_spec
                .modes
                .iter()
                .position(|candidate| candidate == mode)
                .map(|axis| output_spec.shape[axis])
                .ok_or_else(|| {
                    ExecutionError::InvalidPlan(format!(
                        "node {} canonical output mode {mode} is missing",
                        node.id
                    ))
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let result = if let Some(permutation) = &node.contraction.output_permutation {
        permute_col_major(&canonical, &canonical_shape, permutation)?
    } else {
        canonical
    };
    if result.len() != elements(output_spec)? {
        return Err(ExecutionError::InvalidPlan(format!(
            "node {} CPU contraction produced the wrong output length",
            node.id
        )));
    }
    Ok(result)
}

fn permute_col_major<T: Copy + Default>(
    data: &[T],
    shape: &[usize],
    permutation: &[usize],
) -> Result<Vec<T>, ExecutionError> {
    if permutation.len() != shape.len() {
        return Err(ExecutionError::InvalidPlan(format!(
            "permutation rank {} differs from tensor rank {}",
            permutation.len(),
            shape.len()
        )));
    }
    let mut seen = vec![false; shape.len()];
    for axis in permutation {
        if *axis >= shape.len() || std::mem::replace(&mut seen[*axis], true) {
            return Err(ExecutionError::InvalidPlan(
                "invalid axis permutation".to_string(),
            ));
        }
    }
    let expected = shape.iter().product::<usize>();
    if data.len() != expected {
        return Err(ExecutionError::InvalidPlan(
            "permutation input length does not match shape".to_string(),
        ));
    }
    let new_shape: Vec<_> = permutation.iter().map(|axis| shape[*axis]).collect();
    let old_strides = col_major_strides(shape);
    let mut coordinates = vec![0usize; new_shape.len()];
    let mut output = vec![T::default(); data.len()];
    for target in &mut output {
        let source = coordinates
            .iter()
            .enumerate()
            .map(|(new_axis, coordinate)| coordinate * old_strides[permutation[new_axis]])
            .sum::<usize>();
        *target = data[source];
        for axis in 0..coordinates.len() {
            coordinates[axis] += 1;
            if coordinates[axis] < new_shape[axis] {
                break;
            }
            coordinates[axis] = 0;
        }
    }
    Ok(output)
}

fn col_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for axis in 1..shape.len() {
        strides[axis] = strides[axis - 1] * shape[axis - 1];
    }
    strides
}

fn zip_map<T: Copy>(
    left: &[T],
    right: &[T],
    mut operation: impl FnMut(T, T) -> T,
) -> Result<Vec<T>, ExecutionError> {
    if left.len() != right.len() {
        return Err(ExecutionError::InvalidPlan(
            "plane operation received mismatched lengths".to_string(),
        ));
    }
    Ok(left
        .iter()
        .zip(right)
        .map(|(left, right)| operation(*left, *right))
        .collect())
}

fn zip3_map<T: Copy>(
    first: &[T],
    second: &[T],
    third: &[T],
    mut operation: impl FnMut(T, T, T) -> T,
) -> Result<Vec<T>, ExecutionError> {
    if first.len() != second.len() || first.len() != third.len() {
        return Err(ExecutionError::InvalidPlan(
            "three-plane operation received mismatched lengths".to_string(),
        ));
    }
    Ok(first
        .iter()
        .zip(second)
        .zip(third)
        .map(|((first, second), third)| operation(*first, *second, *third))
        .collect())
}

fn elements(spec: &TensorSpec) -> Result<usize, ExecutionError> {
    spec.shape
        .iter()
        .try_fold(1usize, |product, size| product.checked_mul(*size))
        .ok_or_else(|| ExecutionError::InvalidPlan("tensor size overflows usize".to_string()))
}
