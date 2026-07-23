use crate::static_plan::{plan_f32_arena, ExecutionError, InputSet, StaticPlan};

use super::storage::DeviceBuffer;
use super::{AscendMemoryStats, AscendSession};

pub(crate) struct ExecutableState<'session> {
    pub(crate) semantic_arena: DeviceBuffer<'session>,
    pub(crate) scratch: DeviceBuffer<'session>,
    pub(crate) workspace: DeviceBuffer<'session>,
}

impl<'session> ExecutableState<'session> {
    pub(crate) fn reserve(
        session: &'session AscendSession,
        plan: &StaticPlan,
        inputs: &InputSet<f64>,
    ) -> Result<(Self, AscendMemoryStats), ExecutionError> {
        plan.validate()
            .map_err(|error| ExecutionError::InvalidPlan(error.to_string()))?;
        validate_inputs(plan, inputs)?;
        let arena =
            plan_f32_arena(plan).map_err(|error| ExecutionError::InvalidPlan(error.to_string()))?;
        let scratch_bytes = scratch_bytes(plan)?;
        // Operator preparation in Task 4 replaces this zero-byte placeholder
        // after querying every ACLNN executor.
        let workspace_bytes = 0;
        let peak_device_bytes = arena
            .arena_bytes
            .checked_add(scratch_bytes)
            .and_then(|bytes| bytes.checked_add(workspace_bytes))
            .ok_or_else(|| {
                ExecutionError::Unsupported(
                    "Ascend peak device-byte accounting overflowed u64".to_string(),
                )
            })?;

        let semantic_arena = session.context.allocate(arena.arena_bytes)?;
        let scratch = session.context.allocate(scratch_bytes)?;
        let workspace = session.context.allocate(workspace_bytes)?;
        let stats = AscendMemoryStats {
            semantic_bytes: arena.semantic_peak_bytes,
            device_arena_bytes: arena.arena_bytes,
            scratch_bytes,
            workspace_bytes,
            peak_device_bytes,
        };
        Ok((
            Self {
                semantic_arena,
                scratch,
                workspace,
            },
            stats,
        ))
    }
}

fn validate_inputs(plan: &StaticPlan, inputs: &InputSet<f64>) -> Result<(), ExecutionError> {
    if plan.leaf_values.len() != inputs.tensors.len() {
        return Err(ExecutionError::InvalidPlan(format!(
            "plan has {} leaves but input set has {} tensors",
            plan.leaf_values.len(),
            inputs.tensors.len()
        )));
    }
    for (index, (value, input)) in plan
        .leaf_values
        .iter()
        .map(|value| &plan.values[value.0])
        .zip(&inputs.tensors)
        .enumerate()
    {
        if value.tensor != input.spec {
            return Err(ExecutionError::InvalidPlan(format!(
                "input tensor {index} does not match plan leaf geometry"
            )));
        }
        let expected = input
            .spec
            .shape
            .iter()
            .try_fold(1usize, |product, dimension| product.checked_mul(*dimension))
            .ok_or_else(|| {
                ExecutionError::InvalidPlan(format!(
                    "input tensor {index} element count overflows usize"
                ))
            })?;
        if input.real.len() != expected || input.imag.len() != expected {
            return Err(ExecutionError::InvalidPlan(format!(
                "input tensor {index} plane length differs from its shape"
            )));
        }
    }
    Ok(())
}

fn scratch_bytes(plan: &StaticPlan) -> Result<u64, ExecutionError> {
    plan.nodes.iter().try_fold(0u64, |maximum, node| {
        let elements = node.scratch.iter().try_fold(0u64, |sum, scratch| {
            let elements = u64::try_from(scratch.elements).map_err(|_| {
                ExecutionError::Unsupported(format!(
                    "node {} scratch element count exceeds u64",
                    node.id
                ))
            })?;
            sum.checked_add(elements).ok_or_else(|| {
                ExecutionError::Unsupported(format!(
                    "node {} scratch element sum overflows u64",
                    node.id
                ))
            })
        })?;
        let bytes = elements.checked_mul(4).ok_or_else(|| {
            ExecutionError::Unsupported(format!(
                "node {} Float32 scratch bytes overflow u64",
                node.id
            ))
        })?;
        Ok(maximum.max(bytes))
    })
}
