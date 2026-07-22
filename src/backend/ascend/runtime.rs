use super::sys;
use std::{
    ffi::c_void,
    fmt, ptr,
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex, OnceLock,
    },
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AscendError {
    operation: &'static str,
    status: i32,
}

impl AscendError {
    pub(crate) fn status(operation: &'static str, status: i32) -> Self {
        Self { operation, status }
    }

    pub(crate) fn null(operation: &'static str) -> Self {
        Self::status(operation, -1)
    }
}

impl fmt::Display for AscendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Ascend {} failed with status {}",
            self.operation, self.status
        )
    }
}

impl std::error::Error for AscendError {}

fn check(operation: &'static str, status: i32) -> Result<(), AscendError> {
    if status == sys::ACL_SUCCESS {
        Ok(())
    } else {
        Err(AscendError::status(operation, status))
    }
}

// ACL remains initialized for process lifetime. Calling aclFinalize while any
// independently owned context or operator resource exists is unsafe.
static ACL_INIT: OnceLock<i32> = OnceLock::new();

pub(crate) struct Runtime {
    pub(crate) context: sys::AclrtContext,
    pub(crate) stream: sys::AclrtStream,
    pub(crate) device: i32,
    transaction: Mutex<()>,
    stream_failed: AtomicBool,
}

unsafe impl Send for Runtime {}
unsafe impl Sync for Runtime {}

impl Runtime {
    pub(crate) fn new(device: i32) -> Result<Self, AscendError> {
        let init = *ACL_INIT.get_or_init(|| unsafe { sys::aclInit(ptr::null()) });
        if init != sys::ACL_SUCCESS && init != sys::ACL_ERROR_REPEAT_INITIALIZE {
            return Err(AscendError::status("aclInit", init));
        }
        check("aclrtSetDevice", unsafe { sys::aclrtSetDevice(device) })?;
        let mut context = ptr::null_mut();
        check("aclrtCreateContext", unsafe {
            sys::aclrtCreateContext(&mut context, device)
        })?;
        if context.is_null() {
            return Err(AscendError::null("aclrtCreateContext"));
        }
        let mut stream = ptr::null_mut();
        if let Err(error) = check("aclrtCreateStream", unsafe {
            sys::aclrtCreateStream(&mut stream)
        }) {
            unsafe { sys::aclrtDestroyContext(context) };
            return Err(error);
        }
        if stream.is_null() {
            unsafe { sys::aclrtDestroyContext(context) };
            return Err(AscendError::null("aclrtCreateStream"));
        }
        Ok(Self {
            context,
            stream,
            device,
            transaction: Mutex::new(()),
            stream_failed: AtomicBool::new(false),
        })
    }

    pub(crate) fn synchronize(&self) -> Result<(), AscendError> {
        self.with_transaction(|runtime| runtime.synchronize_locked())
    }

    pub(crate) fn alloc(&self, bytes: usize, zero: bool) -> Result<*mut c_void, AscendError> {
        self.with_transaction(|runtime| runtime.alloc_locked(bytes, zero))
    }

    pub(crate) fn with_transaction<T>(
        &self,
        operation: impl FnOnce(&Self) -> Result<T, AscendError>,
    ) -> Result<T, AscendError> {
        let _guard = self.transaction.lock().unwrap_or_else(|e| e.into_inner());
        if self.stream_failed.load(Ordering::Acquire) {
            return Err(AscendError::status(
                "runtime unavailable after stream synchronization failure",
                -1,
            ));
        }
        check("aclrtSetCurrentContext", unsafe {
            sys::aclrtSetCurrentContext(self.context)
        })?;
        operation(self)
    }

    pub(crate) fn synchronize_locked(&self) -> Result<(), AscendError> {
        let result = check("aclrtSynchronizeStream", unsafe {
            sys::aclrtSynchronizeStream(self.stream)
        });
        if result.is_err() {
            // The stream may still reference any submitted allocation. Poison
            // the runtime so storage drops leak those pointers until context
            // teardown rather than freeing memory that could still be in use.
            self.stream_failed.store(true, Ordering::Release);
        }
        result
    }

    pub(crate) fn has_failed_stream(&self) -> bool {
        self.stream_failed.load(Ordering::Acquire)
    }

    pub(crate) fn alloc_locked(
        &self,
        bytes: usize,
        zero: bool,
    ) -> Result<*mut c_void, AscendError> {
        if bytes == 0 {
            return Ok(ptr::null_mut());
        }
        let mut allocation = ptr::null_mut();
        check("aclrtMalloc", unsafe {
            sys::aclrtMalloc(&mut allocation, bytes, sys::ACL_MEM_MALLOC_HUGE_FIRST)
        })?;
        if zero {
            if let Err(error) = check("aclrtMemset", unsafe {
                sys::aclrtMemset(allocation, bytes, 0, bytes)
            }) {
                unsafe { sys::aclrtFree(allocation) };
                return Err(error);
            }
        }
        Ok(allocation)
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        let _guard = self.transaction.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            let _ = sys::aclrtSetCurrentContext(self.context);
            let _ = sys::aclrtSynchronizeStream(self.stream);
            let _ = sys::aclrtDestroyStream(self.stream);
            let _ = sys::aclrtDestroyContext(self.context);
        }
    }
}
