use super::{
    ffi::{
        checked_product, column_major_strides, finish_operation, tensor, IntArray, PendingOperation,
    },
    runtime::Runtime,
    storage::AscendStorage,
    sys, AscendError,
};
use crate::backend::contract_plan::materialize_strided;
use std::{ptr, sync::Arc};

pub(crate) struct ReducedOperand {
    pub(crate) storage: AscendStorage<f32>,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    pub(crate) modes: Vec<i32>,
}

pub(crate) fn reduce_trace(
    runtime: &Arc<Runtime>,
    source: &AscendStorage<f32>,
    shape: &[usize],
    strides: &[usize],
    modes: &[i32],
    trace_modes: &[i32],
) -> Result<ReducedOperand, AscendError> {
    let identity: Vec<usize> = (0..shape.len()).collect();
    let contiguous = materialize_strided(&source.to_vec()?, shape, strides, &identity);
    let input = AscendStorage::upload(runtime.clone(), &contiguous)?;
    let trace_axes: Vec<usize> = trace_modes
        .iter()
        .map(|mode| {
            modes
                .iter()
                .position(|candidate| candidate == mode)
                .ok_or_else(|| AscendError::status("Ascend trace mode missing", -1))
        })
        .collect::<Result<_, _>>()?;
    let output_shape: Vec<usize> = shape
        .iter()
        .enumerate()
        .filter_map(|(axis, &extent)| (!trace_axes.contains(&axis)).then_some(extent))
        .collect();
    let output_modes: Vec<i32> = modes
        .iter()
        .enumerate()
        .filter_map(|(axis, &mode)| (!trace_axes.contains(&axis)).then_some(mode))
        .collect();
    let output_len = checked_product(&output_shape, "Ascend trace output size overflow")?;
    if contiguous.is_empty() {
        return Ok(ReducedOperand {
            storage: AscendStorage::allocate(runtime.clone(), output_len, true)?,
            strides: column_major_strides(&output_shape)?,
            shape: output_shape,
            modes: output_modes,
        });
    }
    let output = AscendStorage::allocate(runtime.clone(), output_len, false)?;
    let input_shape: Vec<usize> = shape.iter().rev().copied().collect();
    let reduced_shape: Vec<usize> = output_shape.iter().rev().copied().collect();
    let dims: Vec<i64> = trace_axes
        .iter()
        .map(|&axis| i64::try_from(shape.len() - 1 - axis).expect("trace rank exceeds i64"))
        .collect();
    runtime.with_transaction(|runtime| {
        let input_tensor = tensor(input.as_mut_ptr(), &input_shape)?;
        let output_tensor = tensor(output.as_mut_ptr(), &reduced_shape)?;
        let dims = IntArray::new(&dims, "aclCreateIntArray(reduce)")?;
        let mut workspace_size = 0u64;
        let mut executor = ptr::null_mut();
        let status = unsafe {
            sys::aclnnReduceSumGetWorkspaceSize(
                input_tensor.0,
                dims.0,
                false,
                sys::ACL_FLOAT,
                output_tensor.0,
                &mut workspace_size,
                &mut executor,
            )
        };
        if status != sys::ACL_SUCCESS {
            return Err(AscendError::status(
                "aclnnReduceSumGetWorkspaceSize",
                status,
            ));
        }
        if executor.is_null() {
            return Err(AscendError::null("aclnnReduceSumGetWorkspaceSize executor"));
        }
        let bytes = usize::try_from(workspace_size)
            .map_err(|_| AscendError::status("aclnnReduceSum workspace overflow", -1))?;
        let workspace = runtime.alloc_locked(bytes, false)?;
        let operation =
            PendingOperation::new(vec![input_tensor, output_tensor], vec![dims], workspace);
        let status =
            unsafe { sys::aclnnReduceSum(workspace, workspace_size, executor, runtime.stream) };
        let launch_result = (status == sys::ACL_SUCCESS)
            .then_some(())
            .ok_or_else(|| AscendError::status("aclnnReduceSum", status));
        finish_operation(
            operation,
            launch_result,
            || runtime.synchronize_locked(),
            |operation| operation.release(runtime),
        )
    })?;
    Ok(ReducedOperand {
        storage: output,
        strides: column_major_strides(&output_shape)?,
        shape: output_shape,
        modes: output_modes,
    })
}
