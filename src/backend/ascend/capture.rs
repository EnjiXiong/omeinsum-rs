#![allow(clippy::result_large_err)] // Frozen backend diagnostics retain full context.

use std::ptr::NonNull;

use crate::static_plan::ExecutionError;

use super::context::Context;
use super::error::{check, invalid_request, ErrorContext};
use super::ffi;

pub(crate) enum CaptureAttempt<Handle> {
    Ready(Handle),
    SkippedUnsupported { reason: String },
}

pub(crate) trait CaptureRuntime {
    type Handle;

    fn supported(&mut self) -> Result<bool, ExecutionError>;
    fn synchronize(&mut self) -> Result<(), ExecutionError>;
    fn begin(&mut self) -> Result<(), ExecutionError>;
    fn enqueue_repeatable(&mut self) -> Result<(), ExecutionError>;
    fn end(&mut self) -> Result<Self::Handle, ExecutionError>;
}

pub(crate) fn configure_capture<Runtime>(
    runtime: &mut Runtime,
    required: bool,
) -> Result<CaptureAttempt<Runtime::Handle>, ExecutionError>
where
    Runtime: CaptureRuntime,
{
    let supported = match runtime.supported() {
        Ok(supported) => supported,
        Err(error) => return unsupported(required, error.to_string()),
    };
    if !supported {
        return unsupported(
            required,
            "native shim was built without capture support".to_string(),
        );
    }

    runtime.synchronize()?;
    if let Err(error) = runtime.begin() {
        // A failed begin is a live capability failure. Synchronize before
        // returning so the already-prepared repeatable path remains usable.
        let _ = runtime.synchronize();
        return unsupported(required, error.to_string());
    }

    if let Err(error) = runtime.enqueue_repeatable() {
        // End the active capture even when an operator rejects capture. A
        // successfully returned partial model is dropped immediately.
        let _ = runtime.end();
        let _ = runtime.synchronize();
        return Err(error);
    }

    let handle = match runtime.end() {
        Ok(handle) => handle,
        Err(error) => {
            let _ = runtime.synchronize();
            return unsupported(required, error.to_string());
        }
    };
    runtime.synchronize()?;
    Ok(CaptureAttempt::Ready(handle))
}

fn unsupported<Handle>(
    required: bool,
    reason: String,
) -> Result<CaptureAttempt<Handle>, ExecutionError> {
    if required {
        Err(ExecutionError::Unsupported(reason))
    } else {
        Ok(CaptureAttempt::SkippedUnsupported { reason })
    }
}

pub(crate) struct Capture<'context> {
    raw: NonNull<ffi::Capture>,
    context: &'context Context,
}

impl<'context> Capture<'context> {
    pub(crate) fn supported(context: &Context) -> Result<bool, ExecutionError> {
        let mut supported = 0;
        let status = unsafe { ffi::ome_ascend_capture_supported(context.raw(), &mut supported) };
        check(
            status,
            ErrorContext {
                stage: "capture-probe",
                node_id: None,
                operation: "ome_ascend_capture_supported",
                logical_shape: &[],
                physical_strides: &[],
                device: Some(context.soc_name().to_string()),
            },
        )?;
        Ok(supported != 0)
    }

    pub(crate) fn begin(context: &Context) -> Result<(), ExecutionError> {
        let status = unsafe { ffi::ome_ascend_capture_begin(context.raw()) };
        check(
            status,
            ErrorContext {
                stage: "capture-begin",
                node_id: None,
                operation: "ome_ascend_capture_begin",
                logical_shape: &[],
                physical_strides: &[],
                device: Some(context.soc_name().to_string()),
            },
        )
    }

    pub(crate) fn end(context: &'context Context) -> Result<Self, ExecutionError> {
        let mut raw = std::ptr::null_mut();
        let status = unsafe { ffi::ome_ascend_capture_end(context.raw(), &mut raw) };
        check(
            status,
            ErrorContext {
                stage: "capture-end",
                node_id: None,
                operation: "ome_ascend_capture_end",
                logical_shape: &[],
                physical_strides: &[],
                device: Some(context.soc_name().to_string()),
            },
        )?;
        let raw = NonNull::new(raw).ok_or_else(|| {
            invalid_request(
                "capture-end",
                "ome_ascend_capture_end",
                &[],
                &[],
                Some(context.soc_name().to_string()),
                "native shim returned success with a null capture",
            )
        })?;
        Ok(Self { raw, context })
    }

    pub(crate) fn run(&mut self) -> Result<(), ExecutionError> {
        let status = unsafe { ffi::ome_ascend_capture_run(self.context.raw(), self.raw.as_ptr()) };
        check(
            status,
            ErrorContext {
                stage: "capture-replay",
                node_id: None,
                operation: "ome_ascend_capture_run",
                logical_shape: &[],
                physical_strides: &[],
                device: Some(self.context.soc_name().to_string()),
            },
        )
    }
}

impl Drop for Capture<'_> {
    fn drop(&mut self) {
        unsafe { ffi::ome_ascend_capture_destroy(self.raw.as_ptr()) };
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use crate::static_plan::ExecutionError;

    use super::{configure_capture, CaptureAttempt, CaptureRuntime};

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Event {
        Probe,
        Synchronize,
        Begin,
        Enqueue,
        End,
        Replay,
        DestroyCapture,
        DestroyOps,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Failure {
        Begin,
        Enqueue,
        End,
    }

    struct MockHandle {
        events: Rc<RefCell<Vec<Event>>>,
    }

    impl MockHandle {
        fn replay(&self) {
            self.events.borrow_mut().push(Event::Replay);
        }
    }

    impl Drop for MockHandle {
        fn drop(&mut self) {
            self.events.borrow_mut().push(Event::DestroyCapture);
        }
    }

    struct MockRuntime {
        supported: bool,
        failure: Option<Failure>,
        capture_active: bool,
        events: Rc<RefCell<Vec<Event>>>,
    }

    impl MockRuntime {
        fn new(supported: bool, failure: Option<Failure>) -> Self {
            Self {
                supported,
                failure,
                capture_active: false,
                events: Rc::new(RefCell::new(Vec::new())),
            }
        }

        fn backend_error(stage: &str) -> ExecutionError {
            ExecutionError::Backend {
                stage: stage.to_string(),
                node_id: None,
                operation: stage.to_string(),
                status_code: Some(-1),
                logical_shape: vec![],
                physical_strides: vec![],
                device: Some("mock".to_string()),
                detail: "injected failure".to_string(),
            }
        }
    }

    impl CaptureRuntime for MockRuntime {
        type Handle = MockHandle;

        fn supported(&mut self) -> Result<bool, ExecutionError> {
            self.events.borrow_mut().push(Event::Probe);
            Ok(self.supported)
        }

        fn synchronize(&mut self) -> Result<(), ExecutionError> {
            self.events.borrow_mut().push(Event::Synchronize);
            Ok(())
        }

        fn begin(&mut self) -> Result<(), ExecutionError> {
            self.events.borrow_mut().push(Event::Begin);
            if self.failure == Some(Failure::Begin) {
                return Err(Self::backend_error("capture-begin"));
            }
            self.capture_active = true;
            Ok(())
        }

        fn enqueue_repeatable(&mut self) -> Result<(), ExecutionError> {
            self.events.borrow_mut().push(Event::Enqueue);
            if self.failure == Some(Failure::Enqueue) {
                self.failure = None;
                return Err(Self::backend_error("operator"));
            }
            Ok(())
        }

        fn end(&mut self) -> Result<Self::Handle, ExecutionError> {
            self.events.borrow_mut().push(Event::End);
            self.capture_active = false;
            if self.failure == Some(Failure::End) {
                return Err(Self::backend_error("capture-end"));
            }
            Ok(MockHandle {
                events: Rc::clone(&self.events),
            })
        }
    }

    #[test]
    fn unsupported_auto_skips_and_required_errors_without_beginning() {
        let mut auto = MockRuntime::new(false, None);
        assert!(matches!(
            configure_capture(&mut auto, false).unwrap(),
            CaptureAttempt::SkippedUnsupported { .. }
        ));
        assert_eq!(&*auto.events.borrow(), &[Event::Probe]);

        let mut required = MockRuntime::new(false, None);
        assert!(matches!(
            configure_capture(&mut required, true),
            Err(ExecutionError::Unsupported(_))
        ));
        assert_eq!(&*required.events.borrow(), &[Event::Probe]);
    }

    #[test]
    fn begin_failure_auto_skips_and_leaves_repeatable_path_usable() {
        let mut runtime = MockRuntime::new(true, Some(Failure::Begin));
        assert!(matches!(
            configure_capture(&mut runtime, false).unwrap(),
            CaptureAttempt::SkippedUnsupported { .. }
        ));
        runtime.failure = None;
        runtime.enqueue_repeatable().unwrap();
        assert_eq!(
            &*runtime.events.borrow(),
            &[
                Event::Probe,
                Event::Synchronize,
                Event::Begin,
                Event::Synchronize,
                Event::Enqueue,
            ]
        );
    }

    #[test]
    fn mid_capture_operator_failure_cleans_up_and_propagates() {
        let mut runtime = MockRuntime::new(true, Some(Failure::Enqueue));
        assert!(matches!(
            configure_capture(&mut runtime, false),
            Err(ExecutionError::Backend { stage, .. }) if stage == "operator"
        ));
        assert!(!runtime.capture_active);
        runtime.enqueue_repeatable().unwrap();
        assert_eq!(
            &*runtime.events.borrow(),
            &[
                Event::Probe,
                Event::Synchronize,
                Event::Begin,
                Event::Enqueue,
                Event::End,
                Event::DestroyCapture,
                Event::Synchronize,
                Event::Enqueue,
            ]
        );
    }

    #[test]
    fn end_failure_auto_skips_and_required_errors_after_cleanup() {
        for required in [false, true] {
            let mut runtime = MockRuntime::new(true, Some(Failure::End));
            let result = configure_capture(&mut runtime, required);
            if required {
                assert!(matches!(result, Err(ExecutionError::Unsupported(_))));
            } else {
                assert!(matches!(
                    result.unwrap(),
                    CaptureAttempt::SkippedUnsupported { .. }
                ));
            }
            assert!(!runtime.capture_active);
            runtime.failure = None;
            runtime.enqueue_repeatable().unwrap();
            assert_eq!(runtime.events.borrow().last(), Some(&Event::Enqueue));
        }
    }

    #[test]
    fn successful_capture_replays_and_destroys_before_operators() {
        let mut runtime = MockRuntime::new(true, None);
        let capture = match configure_capture(&mut runtime, true).unwrap() {
            CaptureAttempt::Ready(capture) => capture,
            CaptureAttempt::SkippedUnsupported { reason } => {
                panic!("unexpected skip: {reason}")
            }
        };
        capture.replay();
        drop(capture);
        runtime.events.borrow_mut().push(Event::DestroyOps);
        assert_eq!(
            &*runtime.events.borrow(),
            &[
                Event::Probe,
                Event::Synchronize,
                Event::Begin,
                Event::Enqueue,
                Event::End,
                Event::Synchronize,
                Event::Replay,
                Event::DestroyCapture,
                Event::DestroyOps,
            ]
        );
    }
}
