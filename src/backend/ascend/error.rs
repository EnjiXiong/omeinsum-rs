use std::ffi::CStr;

use crate::static_plan::ExecutionError;

use super::ffi;

pub(crate) struct ErrorContext<'a> {
    pub(crate) stage: &'a str,
    pub(crate) node_id: Option<usize>,
    pub(crate) operation: &'a str,
    pub(crate) logical_shape: &'a [usize],
    pub(crate) physical_strides: &'a [i64],
    pub(crate) device: Option<String>,
}

pub(crate) fn check(status: ffi::Status, context: ErrorContext<'_>) -> Result<(), ExecutionError> {
    if status.category == 0 && status.cann_code == 0 {
        return Ok(());
    }
    // The shim's error is thread-local and may be replaced by the next FFI
    // call, so copy it before doing anything else.
    let detail = unsafe {
        let pointer = ffi::ome_ascend_last_error();
        if pointer.is_null() {
            "native Ascend shim returned no diagnostic".to_string()
        } else {
            CStr::from_ptr(pointer).to_string_lossy().into_owned()
        }
    };
    Err(ExecutionError::Backend {
        stage: context.stage.to_string(),
        node_id: context.node_id,
        operation: context.operation.to_string(),
        status_code: Some(i64::from(status.cann_code)),
        logical_shape: context.logical_shape.to_vec(),
        physical_strides: context.physical_strides.to_vec(),
        device: context.device,
        detail: format!("category={}; {detail}", status.category),
    })
}

pub(crate) fn invalid_request(
    stage: &str,
    operation: &str,
    logical_shape: &[usize],
    physical_strides: &[i64],
    device: Option<String>,
    detail: impl Into<String>,
) -> ExecutionError {
    ExecutionError::Backend {
        stage: stage.to_string(),
        node_id: None,
        operation: operation.to_string(),
        status_code: None,
        logical_shape: logical_shape.to_vec(),
        physical_strides: physical_strides.to_vec(),
        device,
        detail: detail.into(),
    }
}
