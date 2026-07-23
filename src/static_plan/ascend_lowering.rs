#![allow(clippy::result_large_err)] // Frozen backend diagnostics retain full context.

use super::{ExecutionError, KernelKind, StaticPlan};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoalescedPermutation {
    pub input_shape: Vec<usize>,
    pub axes: Vec<i64>,
    pub output_shape: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LoweredNodeTrace {
    pub node_id: usize,
    pub kind: KernelKind,
    pub matmul_calls: usize,
    pub real_matmul_volume: u128,
    pub elementwise_calls: usize,
    pub permute_calls: usize,
    pub green_batch: usize,
}

pub fn green_operand_plane_batches(kind: KernelKind) -> (usize, usize) {
    match kind {
        KernelKind::RealReal => (1, 1),
        KernelKind::RideLeft => (2, 1),
        KernelKind::RideRight => (1, 2),
        KernelKind::Merge3M | KernelKind::Flat4M => (2, 2),
    }
}

pub fn lower_plan_traces(plan: &StaticPlan) -> Result<Vec<LoweredNodeTrace>, ExecutionError> {
    plan.validate()
        .map_err(|error| ExecutionError::InvalidPlan(error.to_string()))?;
    let traces = plan
        .nodes
        .iter()
        .map(|node| {
            let (matmul_calls, volume_multiplier, elementwise_calls, green_batch) = match node.kind
            {
                KernelKind::RealReal => (1usize, 1u128, 0usize, 1usize),
                KernelKind::RideLeft | KernelKind::RideRight => (1, 2, 0, 2),
                KernelKind::Merge3M => (3, 3, 5, 1),
                KernelKind::Flat4M => (4, 4, 2, 1),
            };
            let real_matmul_volume = node
                .real_skeleton_volume
                .checked_mul(volume_multiplier)
                .ok_or_else(|| {
                    ExecutionError::InvalidPlan(format!(
                        "node {} real Matmul volume overflows u128",
                        node.id
                    ))
                })?;
            Ok(LoweredNodeTrace {
                node_id: node.id,
                kind: node.kind.clone(),
                matmul_calls,
                real_matmul_volume,
                elementwise_calls,
                permute_calls: 0,
                green_batch,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let lowered_volume = traces.iter().try_fold(0u128, |sum, trace| {
        sum.checked_add(trace.real_matmul_volume).ok_or_else(|| {
            ExecutionError::InvalidPlan("lowered real Matmul volume overflows u128".to_string())
        })
    })?;
    if lowered_volume != plan.stats.real_matmul_volume {
        return Err(ExecutionError::InvalidPlan(format!(
            "lowered real Matmul volume {lowered_volume} differs from plan accounting {}",
            plan.stats.real_matmul_volume
        )));
    }
    Ok(traces)
}

pub fn coalesce_permutation(
    current_modes: &[i32],
    desired_modes: &[i32],
    dimensions: &[(i32, usize)],
    rank_limit: usize,
) -> Result<CoalescedPermutation, ExecutionError> {
    if current_modes.len() != desired_modes.len() {
        return Err(ExecutionError::Unsupported(format!(
            "permutation changes rank {} to {}",
            current_modes.len(),
            desired_modes.len()
        )));
    }
    let sizes = dimensions
        .iter()
        .copied()
        .collect::<std::collections::HashMap<_, _>>();
    let current = current_modes
        .iter()
        .copied()
        .filter(|mode| sizes.get(mode).copied().unwrap_or(0) != 1)
        .collect::<Vec<_>>();
    let desired = desired_modes
        .iter()
        .copied()
        .filter(|mode| sizes.get(mode).copied().unwrap_or(0) != 1)
        .collect::<Vec<_>>();
    let mut sorted_current = current.clone();
    let mut sorted_desired = desired.clone();
    sorted_current.sort_unstable();
    sorted_desired.sort_unstable();
    if sorted_current != sorted_desired {
        return Err(ExecutionError::Unsupported(
            "permutation mode sets differ".to_string(),
        ));
    }
    if current.is_empty() {
        return Ok(CoalescedPermutation {
            input_shape: vec![],
            axes: vec![],
            output_shape: vec![],
        });
    }

    let target_positions = desired
        .iter()
        .enumerate()
        .map(|(position, mode)| (*mode, position))
        .collect::<std::collections::HashMap<_, _>>();
    let mut groups = Vec::<Vec<i32>>::new();
    for mode in current {
        if let Some(group) = groups.last_mut() {
            let previous = *group.last().expect("group is non-empty");
            if target_positions[&mode] == target_positions[&previous] + 1 {
                group.push(mode);
                continue;
            }
        }
        groups.push(vec![mode]);
    }
    if groups.len() > rank_limit {
        return Err(ExecutionError::Unsupported(format!(
            "permutation requires rank {} after coalescing, live limit is {rank_limit}",
            groups.len()
        )));
    }
    let input_shape = groups
        .iter()
        .map(|group| checked_group_size(group, &sizes))
        .collect::<Result<Vec<_>, _>>()?;
    let mut desired_groups = (0..groups.len()).collect::<Vec<_>>();
    desired_groups.sort_unstable_by_key(|group| target_positions[&groups[*group][0]]);
    let axes = desired_groups
        .iter()
        .map(|group| {
            i64::try_from(*group).map_err(|_| {
                ExecutionError::Unsupported("coalesced permutation axis exceeds i64".to_string())
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let output_shape = desired_groups
        .iter()
        .map(|group| input_shape[*group])
        .collect();
    Ok(CoalescedPermutation {
        input_shape,
        axes,
        output_shape,
    })
}

fn checked_group_size(
    group: &[i32],
    sizes: &std::collections::HashMap<i32, usize>,
) -> Result<usize, ExecutionError> {
    group.iter().try_fold(1usize, |product, mode| {
        let size = sizes.get(mode).ok_or_else(|| {
            ExecutionError::Unsupported(format!(
                "mode {mode} is missing from permutation dimensions"
            ))
        })?;
        product.checked_mul(*size).ok_or_else(|| {
            ExecutionError::Unsupported(
                "coalesced permutation dimension overflows usize".to_string(),
            )
        })
    })
}
