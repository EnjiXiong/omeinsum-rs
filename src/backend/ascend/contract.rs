use super::{
    ffi::{checked_product, tensor},
    normalize::normalize_repeated_labels,
    reduce::reduce_trace,
    runtime::Runtime,
    storage::AscendStorage,
    sys, AscendError,
};
use crate::backend::{
    contract_plan::{materialize_strided, plan_contraction},
    Storage,
};
use std::{collections::HashMap, ptr, sync::Arc};

pub(crate) fn validate_operand(
    storage_len: usize,
    shape: &[usize],
    strides: &[usize],
    modes: &[i32],
) -> Result<(), AscendError> {
    if shape.len() != strides.len() || shape.len() != modes.len() {
        return Err(AscendError::status("Ascend metadata rank mismatch", -1));
    }
    let numel = checked_product(shape, "Ascend shape product overflow")?;
    if numel == 0 {
        return Ok(());
    }
    let max_offset = shape
        .iter()
        .zip(strides)
        .try_fold(0usize, |offset, (&dim, &stride)| {
            (dim - 1)
                .checked_mul(stride)
                .and_then(|term| offset.checked_add(term))
                .ok_or_else(|| AscendError::status("Ascend strided offset overflow", -1))
        })?;
    if max_offset >= storage_len {
        return Err(AscendError::status(
            "Ascend strided view exceeds storage",
            -1,
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_metadata(
    a_len: usize,
    shape_a: &[usize],
    strides_a: &[usize],
    modes_a: &[i32],
    b_len: usize,
    shape_b: &[usize],
    strides_b: &[usize],
    modes_b: &[i32],
    shape_c: &[usize],
    modes_c: &[i32],
) -> Result<usize, AscendError> {
    validate_operand(a_len, shape_a, strides_a, modes_a)?;
    validate_operand(b_len, shape_b, strides_b, modes_b)?;
    if shape_c.len() != modes_c.len() {
        return Err(AscendError::status("Ascend output rank mismatch", -1));
    }
    if modes_c
        .iter()
        .enumerate()
        .any(|(i, mode)| modes_c[..i].contains(mode))
    {
        return Err(AscendError::status("Ascend repeated output mode", -1));
    }
    let mut extents = HashMap::new();
    for (&mode, &extent) in modes_a.iter().zip(shape_a) {
        extents.insert(mode, extent);
    }
    for (&mode, &extent) in modes_b.iter().zip(shape_b) {
        if extents.get(&mode).is_some_and(|&known| known != extent) {
            return Err(AscendError::status("Ascend mode extent mismatch", -1));
        }
        extents.insert(mode, extent);
    }
    for (&mode, &extent) in modes_c.iter().zip(shape_c) {
        if extents.get(&mode) != Some(&extent) {
            return Err(AscendError::status("Ascend output shape mismatch", -1));
        }
    }
    checked_product(shape_c, "Ascend output shape product overflow")
}

#[allow(clippy::too_many_arguments)]
fn launch(
    runtime: &Arc<Runtime>,
    a: AscendStorage<f32>,
    b: AscendStorage<f32>,
    output: AscendStorage<f32>,
    a_shape: &[usize],
    b_shape: &[usize],
    c_shape: &[usize],
    batched: bool,
) -> Result<AscendStorage<f32>, AscendError> {
    let mut stream_state_unknown = false;
    let result = runtime.with_transaction(|runtime| {
        let ta = tensor(a.as_mut_ptr(), a_shape)?;
        let tb = tensor(b.as_mut_ptr(), b_shape)?;
        let tc = tensor(output.as_mut_ptr(), c_shape)?;
        let mut workspace_size = 0u64;
        let mut executor = ptr::null_mut();
        let status = unsafe {
            if batched {
                sys::aclnnBatchMatMulGetWorkspaceSize(
                    tb.0,
                    ta.0,
                    tc.0,
                    0,
                    &mut workspace_size,
                    &mut executor,
                )
            } else {
                sys::aclnnMatmulGetWorkspaceSize(
                    tb.0,
                    ta.0,
                    tc.0,
                    0,
                    &mut workspace_size,
                    &mut executor,
                )
            }
        };
        if status != sys::ACL_SUCCESS {
            return Err(AscendError::status("aclnnMatmulGetWorkspaceSize", status));
        }
        if executor.is_null() {
            return Err(AscendError::null("aclnnMatmulGetWorkspaceSize executor"));
        }
        let bytes = usize::try_from(workspace_size)
            .map_err(|_| AscendError::status("aclnnMatmul workspace overflow", -1))?;
        let workspace = runtime.alloc_locked(bytes, false)?;
        let status = unsafe {
            if batched {
                sys::aclnnBatchMatMul(workspace, workspace_size, executor, runtime.stream)
            } else {
                sys::aclnnMatmul(workspace, workspace_size, executor, runtime.stream)
            }
        };
        if status != sys::ACL_SUCCESS {
            if !workspace.is_null() {
                unsafe { sys::aclrtFree(workspace) };
            }
            return Err(AscendError::status("aclnnMatmul", status));
        }
        stream_state_unknown = true;
        if let Err(error) = runtime.synchronize_locked() {
            // The stream may still reference every temporary. Conservatively leak them.
            std::mem::forget(ta);
            std::mem::forget(tb);
            std::mem::forget(tc);
            return Err(error);
        }
        stream_state_unknown = false;
        if !workspace.is_null() {
            unsafe {
                let _ = sys::aclrtFree(workspace);
            }
        }
        Ok(())
    });
    if let Err(error) = result {
        if stream_state_unknown {
            std::mem::forget(a);
            std::mem::forget(b);
            std::mem::forget(output);
        }
        return Err(error);
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn contract(
    runtime: &Arc<Runtime>,
    a: &AscendStorage<f32>,
    shape_a: &[usize],
    strides_a: &[usize],
    modes_a: &[i32],
    b: &AscendStorage<f32>,
    shape_b: &[usize],
    strides_b: &[usize],
    modes_b: &[i32],
    shape_c: &[usize],
    modes_c: &[i32],
) -> Result<AscendStorage<f32>, AscendError> {
    if !Arc::ptr_eq(runtime, a.runtime()) || !Arc::ptr_eq(runtime, b.runtime()) {
        return Err(AscendError::status("Ascend runtime/device mismatch", -1));
    }
    if a.runtime().device != b.runtime().device {
        return Err(AscendError::status("Ascend device mismatch", -1));
    }
    let normalized_a = normalize_repeated_labels(runtime, a, shape_a, strides_a, modes_a)?;
    let normalized_b = normalize_repeated_labels(runtime, b, shape_b, strides_b, modes_b)?;
    let (a, shape_a, strides_a, modes_a) =
        normalized_a
            .as_ref()
            .map_or((a, shape_a, strides_a, modes_a), |operand| {
                (
                    &operand.storage,
                    operand.shape.as_slice(),
                    operand.strides.as_slice(),
                    operand.modes.as_slice(),
                )
            });
    let (b, shape_b, strides_b, modes_b) =
        normalized_b
            .as_ref()
            .map_or((b, shape_b, strides_b, modes_b), |operand| {
                (
                    &operand.storage,
                    operand.shape.as_slice(),
                    operand.strides.as_slice(),
                    operand.modes.as_slice(),
                )
            });
    let output_len = validate_metadata(
        a.len(),
        shape_a,
        strides_a,
        modes_a,
        b.len(),
        shape_b,
        strides_b,
        modes_b,
        shape_c,
        modes_c,
    )?;
    if output_len == 0 {
        return AscendStorage::allocate(runtime.clone(), 0, false);
    }
    let initial_plan = plan_contraction(modes_a, shape_a, modes_b, shape_b, modes_c);
    let reduced_a = (!initial_plan.left_trace.is_empty()).then(|| {
        reduce_trace(
            runtime,
            a,
            shape_a,
            strides_a,
            modes_a,
            &initial_plan.left_trace,
        )
    });
    let reduced_a = reduced_a.transpose()?;
    let reduced_b = (!initial_plan.right_trace.is_empty()).then(|| {
        reduce_trace(
            runtime,
            b,
            shape_b,
            strides_b,
            modes_b,
            &initial_plan.right_trace,
        )
    });
    let reduced_b = reduced_b.transpose()?;
    let (a, shape_a, strides_a, modes_a) =
        reduced_a
            .as_ref()
            .map_or((a, shape_a, strides_a, modes_a), |operand| {
                (
                    &operand.storage,
                    operand.shape.as_slice(),
                    operand.strides.as_slice(),
                    operand.modes.as_slice(),
                )
            });
    let (b, shape_b, strides_b, modes_b) =
        reduced_b
            .as_ref()
            .map_or((b, shape_b, strides_b, modes_b), |operand| {
                (
                    &operand.storage,
                    operand.shape.as_slice(),
                    operand.strides.as_slice(),
                    operand.modes.as_slice(),
                )
            });
    let plan = plan_contraction(modes_a, shape_a, modes_b, shape_b, modes_c);
    let zero_contract_extent = plan.contracted_modes.iter().any(|mode| {
        let axis = modes_a
            .iter()
            .position(|candidate| candidate == mode)
            .expect("planner contracted mode must belong to A");
        shape_a[axis] == 0
    });
    if zero_contract_extent {
        return AscendStorage::allocate(runtime.clone(), output_len.max(1), true);
    }

    let host_a = a.to_vec()?;
    let host_b = b.to_vec()?;
    let canonical_a =
        materialize_strided(&host_a, shape_a, strides_a, &plan.a_permutation(modes_a));
    let canonical_b =
        materialize_strided(&host_b, shape_b, strides_b, &plan.b_permutation(modes_b));
    let a = AscendStorage::upload(runtime.clone(), &canonical_a)?;
    let b = AscendStorage::upload(runtime.clone(), &canonical_b)?;
    let output = AscendStorage::allocate(runtime.clone(), output_len.max(1), false)?;
    let (a_shape, b_shape, c_shape, batched) = if plan.batch_size > 1 {
        (
            vec![plan.batch_size, plan.contract_size, plan.left_size],
            vec![plan.batch_size, plan.right_size, plan.contract_size],
            vec![plan.batch_size, plan.right_size, plan.left_size],
            true,
        )
    } else {
        (
            vec![plan.contract_size, plan.left_size],
            vec![plan.right_size, plan.contract_size],
            vec![plan.right_size, plan.left_size],
            false,
        )
    };
    let output = launch(runtime, a, b, output, &a_shape, &b_shape, &c_shape, batched)?;
    if let Some(perm) = plan.output_perm {
        let host = output.to_vec()?;
        let canonical_shape: Vec<usize> = plan
            .left_modes
            .iter()
            .chain(&plan.right_modes)
            .chain(&plan.batch_modes)
            .map(|mode| {
                let axis = modes_c
                    .iter()
                    .position(|candidate| candidate == mode)
                    .unwrap();
                shape_c[axis]
            })
            .collect();
        let mut stride = 1usize;
        let canonical_strides: Vec<usize> = canonical_shape
            .iter()
            .map(|&extent| {
                let current = stride;
                stride *= extent;
                current
            })
            .collect();
        let reordered = materialize_strided(&host, &canonical_shape, &canonical_strides, &perm);
        return AscendStorage::upload(runtime.clone(), &reordered);
    }
    Ok(output)
}
