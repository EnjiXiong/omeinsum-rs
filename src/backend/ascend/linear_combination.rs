use super::{
    ffi::{finish_operation, tensor, PendingOperation, Scalar},
    runtime::Runtime,
    storage::AscendStorage,
    sys, AscendError,
};
use crate::backend::Storage;
use std::{ptr, sync::Arc};

pub(crate) fn linear_combination(
    runtime: &Arc<Runtime>,
    x: &AscendStorage<f32>,
    y: &AscendStorage<f32>,
    alpha: f32,
) -> Result<AscendStorage<f32>, AscendError> {
    if x.len() != y.len() {
        return Err(AscendError::status(
            "Ascend linear combination length mismatch",
            -1,
        ));
    }
    let output = AscendStorage::allocate(runtime.clone(), x.len(), false)?;
    runtime.with_transaction(|runtime| {
        let shape = [x.len()];
        let tx = tensor(x.as_mut_ptr(), &shape)?;
        let ty = tensor(y.as_mut_ptr(), &shape)?;
        let tout = tensor(output.as_mut_ptr(), &shape)?;
        let scalar = Scalar::f32(alpha)?;
        let mut workspace_size = 0u64;
        let mut executor = ptr::null_mut();
        let status = unsafe {
            sys::aclnnAddGetWorkspaceSize(
                tx.0,
                ty.0,
                scalar.0,
                tout.0,
                &mut workspace_size,
                &mut executor,
            )
        };
        if status != sys::ACL_SUCCESS {
            return Err(AscendError::status("aclnnAddGetWorkspaceSize", status));
        }
        if executor.is_null() {
            return Err(AscendError::null("aclnnAddGetWorkspaceSize executor"));
        }
        let bytes = usize::try_from(workspace_size)
            .map_err(|_| AscendError::status("aclnnAdd workspace overflow", -1))?;
        let workspace = runtime.alloc_locked(bytes, false)?;
        let operation = PendingOperation::new(vec![tx, ty, tout], Vec::new(), workspace)
            .with_scalars(vec![scalar]);
        let status = unsafe { sys::aclnnAdd(workspace, workspace_size, executor, runtime.stream) };
        let result = (status == sys::ACL_SUCCESS)
            .then_some(())
            .ok_or_else(|| AscendError::status("aclnnAdd", status));
        finish_operation(
            operation,
            result,
            || runtime.synchronize_locked(),
            |operation| operation.release(runtime),
        )
    })?;
    Ok(output)
}
