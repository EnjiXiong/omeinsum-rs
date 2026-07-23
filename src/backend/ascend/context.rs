//! Ascend context ownership.
//!
//! Context-bound resources cannot outlive their context:
//!
//! ```compile_fail
//! use omeinsum::backend::ascend::context::Context;
//! let buffer = {
//!     let context = Context::new(0).unwrap();
//!     context.allocate(16).unwrap()
//! };
//! drop(buffer);
//! ```
//!
//! Contexts and their resources are deliberately neither `Send` nor `Sync`:
//!
//! ```compile_fail
//! use omeinsum::backend::ascend::context::Context;
//! let context = Context::new(0).unwrap();
//! std::thread::spawn(move || drop(context));
//! ```

use std::ffi::CStr;
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::rc::Rc;

use crate::static_plan::ExecutionError;

use super::error::{check, ErrorContext};
use super::ffi;
use super::storage::DeviceBuffer;

pub struct Context {
    raw: NonNull<ffi::Context>,
    device_id: i32,
    soc_name: String,
    _not_send_sync: PhantomData<Rc<()>>,
}

impl Context {
    pub fn new(device_id: i32) -> Result<Self, ExecutionError> {
        let mut raw = std::ptr::null_mut();
        let status = unsafe { ffi::ome_ascend_context_create(device_id, &mut raw) };
        check(
            status,
            ErrorContext {
                stage: "context-create",
                node_id: None,
                operation: "ome_ascend_context_create",
                logical_shape: &[],
                physical_strides: &[],
                device: Some(format!("device-id={device_id}")),
            },
        )?;
        let raw = NonNull::new(raw).ok_or_else(|| {
            super::error::invalid_request(
                "context-create",
                "ome_ascend_context_create",
                &[],
                &[],
                Some(format!("device-id={device_id}")),
                "native shim returned success with a null context",
            )
        })?;
        let soc_name = unsafe {
            let pointer = ffi::ome_ascend_context_soc_name(raw.as_ptr());
            if pointer.is_null() {
                "unknown".to_string()
            } else {
                CStr::from_ptr(pointer).to_string_lossy().into_owned()
            }
        };
        Ok(Self {
            raw,
            device_id,
            soc_name,
            _not_send_sync: PhantomData,
        })
    }

    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    pub fn soc_name(&self) -> &str {
        &self.soc_name
    }

    pub fn synchronize(&self) -> Result<(), ExecutionError> {
        let status = unsafe { ffi::ome_ascend_context_synchronize(self.raw.as_ptr()) };
        check(
            status,
            ErrorContext {
                stage: "synchronize",
                node_id: None,
                operation: "ome_ascend_context_synchronize",
                logical_shape: &[],
                physical_strides: &[],
                device: Some(self.soc_name.clone()),
            },
        )
    }

    pub fn allocate(&self, bytes: u64) -> Result<DeviceBuffer<'_>, ExecutionError> {
        DeviceBuffer::allocate(self, bytes)
    }

    pub(crate) fn raw(&self) -> *mut ffi::Context {
        self.raw.as_ptr()
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { ffi::ome_ascend_context_destroy(self.raw.as_ptr()) };
    }
}

#[cfg(test)]
mod tests {
    const CREATE: [&str; 5] = [
        "aclInit",
        "aclrtSetDevice",
        "aclrtCreateContext",
        "aclrtSetCurrentContext",
        "aclrtCreateStream",
    ];
    const DESTROY: [&str; 5] = [
        "aclrtSynchronizeStream",
        "aclrtDestroyStream",
        "aclrtDestroyContext",
        "aclrtResetDevice",
        "aclFinalize",
    ];

    fn lifecycle_log(fail_at: Option<usize>) -> Vec<&'static str> {
        let completed = fail_at.unwrap_or(CREATE.len());
        let mut log = CREATE[..completed].to_vec();
        if completed == CREATE.len() {
            log.extend(DESTROY);
            return log;
        }
        if completed >= 4 {
            // Stream creation itself failed, so there is no stream to destroy.
        }
        if completed >= 3 {
            log.push("aclrtDestroyContext");
        }
        if completed >= 2 {
            log.push("aclrtResetDevice");
        }
        if completed >= 1 {
            log.push("aclFinalize");
        }
        log
    }

    #[test]
    fn lifecycle_order_and_every_partial_failure_are_explicit() {
        assert_eq!(
            lifecycle_log(None),
            CREATE.into_iter().chain(DESTROY).collect::<Vec<_>>()
        );
        assert_eq!(lifecycle_log(Some(0)), Vec::<&str>::new());
        assert_eq!(lifecycle_log(Some(1)), vec!["aclInit", "aclFinalize"]);
        assert_eq!(
            lifecycle_log(Some(2)),
            vec![
                "aclInit",
                "aclrtSetDevice",
                "aclrtResetDevice",
                "aclFinalize"
            ]
        );
        assert_eq!(
            lifecycle_log(Some(3)),
            vec![
                "aclInit",
                "aclrtSetDevice",
                "aclrtCreateContext",
                "aclrtDestroyContext",
                "aclrtResetDevice",
                "aclFinalize"
            ]
        );
        assert_eq!(
            lifecycle_log(Some(4)),
            vec![
                "aclInit",
                "aclrtSetDevice",
                "aclrtCreateContext",
                "aclrtSetCurrentContext",
                "aclrtDestroyContext",
                "aclrtResetDevice",
                "aclFinalize"
            ]
        );
    }
}
