use std::marker::PhantomData;
use std::mem::size_of_val;
use std::ptr::NonNull;
use std::rc::Rc;

use bytemuck::Pod;

use crate::static_plan::ExecutionError;

use super::context::Context;
use super::error::{check, invalid_request, ErrorContext};
use super::ffi;

struct BufferInner<'context> {
    raw: NonNull<ffi::Buffer>,
    bytes: u64,
    context: &'context Context,
    _not_send_sync: PhantomData<Rc<()>>,
}

impl Drop for BufferInner<'_> {
    fn drop(&mut self) {
        unsafe { ffi::ome_ascend_buffer_destroy(self.raw.as_ptr()) };
    }
}

pub struct DeviceBuffer<'context> {
    inner: Rc<BufferInner<'context>>,
}

impl<'context> DeviceBuffer<'context> {
    pub(crate) fn allocate(context: &'context Context, bytes: u64) -> Result<Self, ExecutionError> {
        let mut raw = std::ptr::null_mut();
        let status = unsafe { ffi::ome_ascend_buffer_alloc(context.raw(), bytes, &mut raw) };
        check(
            status,
            ErrorContext {
                stage: "allocation",
                node_id: None,
                operation: "ome_ascend_buffer_alloc",
                logical_shape: &[],
                physical_strides: &[],
                device: Some(context.soc_name().to_string()),
            },
        )?;
        let raw = NonNull::new(raw).ok_or_else(|| {
            invalid_request(
                "allocation",
                "ome_ascend_buffer_alloc",
                &[],
                &[],
                Some(context.soc_name().to_string()),
                "native shim returned success with a null buffer",
            )
        })?;
        Ok(Self {
            inner: Rc::new(BufferInner {
                raw,
                bytes,
                context,
                _not_send_sync: PhantomData,
            }),
        })
    }

    pub fn bytes(&self) -> u64 {
        self.inner.bytes
    }

    pub fn copy_h2d<T: Pod>(&self, byte_offset: u64, values: &[T]) -> Result<(), ExecutionError> {
        let bytes = u64::try_from(size_of_val(values)).map_err(|_| {
            self.range_error("copy-h2d", byte_offset, u64::MAX, "host slice is too large")
        })?;
        checked_range(self.inner.bytes, byte_offset, bytes)
            .map_err(|detail| self.range_error("copy-h2d", byte_offset, bytes, detail))?;
        let status = unsafe {
            ffi::ome_ascend_buffer_copy_h2d(
                self.inner.context.raw(),
                self.inner.raw.as_ptr(),
                byte_offset,
                values.as_ptr().cast(),
                bytes,
            )
        };
        check(
            status,
            self.error_context("copy-h2d", "ome_ascend_buffer_copy_h2d", &[], &[]),
        )
    }

    pub fn copy_d2h<T: Pod>(
        &self,
        byte_offset: u64,
        values: &mut [T],
    ) -> Result<(), ExecutionError> {
        let bytes = u64::try_from(size_of_val(values)).map_err(|_| {
            self.range_error("copy-d2h", byte_offset, u64::MAX, "host slice is too large")
        })?;
        checked_range(self.inner.bytes, byte_offset, bytes)
            .map_err(|detail| self.range_error("copy-d2h", byte_offset, bytes, detail))?;
        let status = unsafe {
            ffi::ome_ascend_buffer_copy_d2h(
                self.inner.context.raw(),
                self.inner.raw.as_ptr(),
                byte_offset,
                values.as_mut_ptr().cast(),
                bytes,
            )
        };
        check(
            status,
            self.error_context("copy-d2h", "ome_ascend_buffer_copy_d2h", &[], &[]),
        )
    }

    pub fn tensor_f32(
        &self,
        byte_offset: u64,
        shape: &[usize],
        strides: &[i64],
    ) -> Result<TensorDescriptor<'context>, ExecutionError> {
        let bytes = tensor_span_bytes(shape, strides).map_err(|detail| {
            invalid_request(
                "descriptor-create",
                "ome_ascend_tensor_f32",
                shape,
                strides,
                Some(self.inner.context.soc_name().to_string()),
                detail,
            )
        })?;
        checked_range(self.inner.bytes, byte_offset, bytes)
            .map_err(|detail| self.range_error("descriptor-create", byte_offset, bytes, detail))?;
        let shape_i64 = shape
            .iter()
            .map(|dimension| {
                i64::try_from(*dimension).map_err(|_| {
                    invalid_request(
                        "descriptor-create",
                        "ome_ascend_tensor_f32",
                        shape,
                        strides,
                        Some(self.inner.context.soc_name().to_string()),
                        format!("dimension {dimension} exceeds i64"),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            ffi::ome_ascend_tensor_f32(
                self.inner.raw.as_ptr(),
                byte_offset,
                shape_i64.as_ptr(),
                strides.as_ptr(),
                u64::try_from(shape.len()).unwrap_or(u64::MAX),
                &mut raw,
            )
        };
        check(
            status,
            self.error_context("descriptor-create", "ome_ascend_tensor_f32", shape, strides),
        )?;
        let raw = NonNull::new(raw).ok_or_else(|| {
            invalid_request(
                "descriptor-create",
                "ome_ascend_tensor_f32",
                shape,
                strides,
                Some(self.inner.context.soc_name().to_string()),
                "native shim returned success with a null tensor descriptor",
            )
        })?;
        Ok(TensorDescriptor {
            inner: Rc::new(TensorInner {
                raw,
                shape: shape.to_vec(),
                strides: strides.to_vec(),
                _buffer: Rc::clone(&self.inner),
            }),
        })
    }

    pub(crate) fn raw(&self) -> *mut ffi::Buffer {
        self.inner.raw.as_ptr()
    }

    fn error_context<'a>(
        &'a self,
        stage: &'a str,
        operation: &'a str,
        logical_shape: &'a [usize],
        physical_strides: &'a [i64],
    ) -> ErrorContext<'a> {
        ErrorContext {
            stage,
            node_id: None,
            operation,
            logical_shape,
            physical_strides,
            device: Some(self.inner.context.soc_name().to_string()),
        }
    }

    fn range_error(
        &self,
        stage: &str,
        offset: u64,
        bytes: u64,
        detail: impl Into<String>,
    ) -> ExecutionError {
        invalid_request(
            stage,
            "device-buffer-range",
            &[],
            &[],
            Some(self.inner.context.soc_name().to_string()),
            format!(
                "offset={offset}, bytes={bytes}, capacity={}: {}",
                self.inner.bytes,
                detail.into()
            ),
        )
    }
}

struct TensorInner<'context> {
    raw: NonNull<ffi::Tensor>,
    shape: Vec<usize>,
    strides: Vec<i64>,
    _buffer: Rc<BufferInner<'context>>,
}

pub struct TensorDescriptor<'context> {
    inner: Rc<TensorInner<'context>>,
}

impl Clone for TensorDescriptor<'_> {
    fn clone(&self) -> Self {
        Self {
            inner: Rc::clone(&self.inner),
        }
    }
}

impl TensorDescriptor<'_> {
    pub fn shape(&self) -> &[usize] {
        &self.inner.shape
    }

    pub fn strides(&self) -> &[i64] {
        &self.inner.strides
    }

    pub(crate) fn raw(&self) -> *mut ffi::Tensor {
        self.inner.raw.as_ptr()
    }
}

impl Drop for TensorInner<'_> {
    fn drop(&mut self) {
        unsafe { ffi::ome_ascend_tensor_destroy(self.raw.as_ptr()) };
    }
}

fn checked_range(capacity: u64, offset: u64, bytes: u64) -> Result<(), &'static str> {
    let end = offset.checked_add(bytes).ok_or("byte range overflow")?;
    if end > capacity {
        return Err("byte range exceeds buffer");
    }
    Ok(())
}

fn tensor_span_bytes(shape: &[usize], strides: &[i64]) -> Result<u64, String> {
    if shape.len() != strides.len() {
        return Err(format!(
            "shape rank {} differs from stride rank {}",
            shape.len(),
            strides.len()
        ));
    }
    let mut elements = 1u64;
    for (axis, (&dimension, &stride)) in shape.iter().zip(strides).enumerate() {
        if dimension == 0 {
            return Err(format!("shape[{axis}] must be positive"));
        }
        let stride =
            u64::try_from(stride).map_err(|_| format!("stride[{axis}] must be nonnegative"))?;
        let extent = u64::try_from(dimension - 1)
            .ok()
            .and_then(|value| value.checked_mul(stride))
            .ok_or_else(|| format!("axis {axis} span overflows u64"))?;
        elements = elements
            .checked_add(extent)
            .ok_or_else(|| "tensor element span overflows u64".to_string())?;
    }
    elements
        .checked_mul(4)
        .ok_or_else(|| "tensor byte span overflows u64".to_string())
}

#[cfg(test)]
mod tests {
    use super::{checked_range, tensor_span_bytes};

    #[test]
    fn host_range_checks_reject_overflow_and_out_of_bounds() {
        assert_eq!(checked_range(16, 4, 12), Ok(()));
        assert!(checked_range(16, 5, 12).is_err());
        assert!(checked_range(u64::MAX, u64::MAX, 1).is_err());
    }

    #[test]
    fn tensor_span_checks_strided_and_broadcast_views() {
        assert_eq!(tensor_span_bytes(&[2, 3], &[3, 1]).unwrap(), 24);
        assert_eq!(tensor_span_bytes(&[2, 3], &[0, 1]).unwrap(), 12);
        assert!(tensor_span_bytes(&[2], &[-1]).is_err());
        assert!(tensor_span_bytes(&[2, 3], &[1]).is_err());
    }
}
