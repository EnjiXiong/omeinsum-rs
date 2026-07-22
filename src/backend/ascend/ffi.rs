use super::{sys, AscendError};

pub(crate) struct Tensor(pub(crate) *mut sys::AclTensor);

impl Drop for Tensor {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe {
                let _ = sys::aclDestroyTensor(self.0);
            }
        }
    }
}

pub(crate) struct IntArray(pub(crate) *mut sys::AclIntArray);

impl IntArray {
    pub(crate) fn new(values: &[i64], operation: &'static str) -> Result<Self, AscendError> {
        let raw = unsafe { sys::aclCreateIntArray(values.as_ptr(), values.len() as u64) };
        (!raw.is_null())
            .then_some(Self(raw))
            .ok_or_else(|| AscendError::null(operation))
    }
}

impl Drop for IntArray {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe {
                let _ = sys::aclDestroyIntArray(self.0);
            }
        }
    }
}

pub(crate) fn checked_product(
    shape: &[usize],
    operation: &'static str,
) -> Result<usize, AscendError> {
    shape.iter().try_fold(1usize, |n, &d| {
        n.checked_mul(d)
            .ok_or_else(|| AscendError::status(operation, -1))
    })
}

pub(crate) fn row_major_strides(shape: &[usize]) -> Result<Vec<usize>, AscendError> {
    let mut stride = 1usize;
    let mut result = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        result[axis] = stride;
        stride = stride
            .checked_mul(shape[axis])
            .ok_or_else(|| AscendError::status("Ascend tensor stride overflow", -1))?;
    }
    Ok(result)
}

pub(crate) fn column_major_strides(shape: &[usize]) -> Result<Vec<usize>, AscendError> {
    let mut stride = 1usize;
    shape
        .iter()
        .map(|&extent| {
            let current = stride;
            stride = stride
                .checked_mul(extent)
                .ok_or_else(|| AscendError::status("Ascend tensor stride overflow", -1))?;
            Ok(current)
        })
        .collect()
}

pub(crate) fn tensor(data: *mut std::ffi::c_void, shape: &[usize]) -> Result<Tensor, AscendError> {
    let strides = row_major_strides(shape)?;
    let dims: Vec<i64> = shape
        .iter()
        .map(|&x| i64::try_from(x).map_err(|_| AscendError::status("Ascend dimension", -1)))
        .collect::<Result<_, _>>()?;
    let strides: Vec<i64> = strides
        .iter()
        .map(|&x| i64::try_from(x).map_err(|_| AscendError::status("Ascend stride", -1)))
        .collect::<Result<_, _>>()?;
    let raw = unsafe {
        sys::aclCreateTensor(
            dims.as_ptr(),
            dims.len() as u64,
            sys::ACL_FLOAT,
            strides.as_ptr(),
            0,
            sys::ACL_FORMAT_ND,
            dims.as_ptr(),
            dims.len() as u64,
            data,
        )
    };
    (!raw.is_null())
        .then_some(Tensor(raw))
        .ok_or_else(|| AscendError::null("aclCreateTensor"))
}
