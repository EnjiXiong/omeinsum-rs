use super::{runtime::Runtime, sys, AscendError};
use crate::{algebra::Scalar, backend::Storage};
use std::{ffi::c_void, marker::PhantomData, mem::size_of, sync::Arc};

pub struct AscendStorage<T: Scalar> {
    ptr: *mut c_void,
    len: usize,
    runtime: Arc<Runtime>,
    marker: PhantomData<T>,
}

unsafe impl<T: Scalar> Send for AscendStorage<T> {}
unsafe impl<T: Scalar> Sync for AscendStorage<T> {}

impl<T: Scalar> AscendStorage<T> {
    pub(crate) fn allocate(
        runtime: Arc<Runtime>,
        len: usize,
        zero: bool,
    ) -> Result<Self, AscendError> {
        let bytes = len
            .checked_mul(size_of::<T>())
            .expect("Ascend allocation size overflow");
        let ptr = runtime.alloc(bytes, zero)?;
        Ok(Self {
            ptr,
            len,
            runtime,
            marker: PhantomData,
        })
    }

    pub(crate) fn upload(runtime: Arc<Runtime>, data: &[T]) -> Result<Self, AscendError> {
        let storage = Self::allocate(runtime, data.len(), false)?;
        let bytes = std::mem::size_of_val(data);
        if bytes != 0 {
            storage.runtime.with_transaction(|_| {
                let status = unsafe {
                    sys::aclrtMemcpy(
                        storage.ptr,
                        bytes,
                        data.as_ptr().cast(),
                        bytes,
                        sys::ACL_MEMCPY_HOST_TO_DEVICE,
                    )
                };
                (status == sys::ACL_SUCCESS)
                    .then_some(())
                    .ok_or_else(|| AscendError::status("aclrtMemcpy(H2D)", status))
            })?;
        }
        Ok(storage)
    }

    pub(crate) fn as_mut_ptr(&self) -> *mut c_void {
        self.ptr
    }

    pub(crate) fn runtime(&self) -> &Arc<Runtime> {
        &self.runtime
    }

    pub fn to_vec(&self) -> Result<Vec<T>, AscendError> {
        let mut host = vec![T::default(); self.len];
        let bytes = self.len * size_of::<T>();
        if bytes != 0 {
            self.runtime.with_transaction(|runtime| {
                let status = unsafe {
                    sys::aclrtMemcpy(
                        host.as_mut_ptr().cast(),
                        bytes,
                        self.ptr,
                        bytes,
                        sys::ACL_MEMCPY_DEVICE_TO_HOST,
                    )
                };
                if status != sys::ACL_SUCCESS {
                    return Err(AscendError::status("aclrtMemcpy(D2H)", status));
                }
                runtime.synchronize_locked()
            })?;
        }
        Ok(host)
    }
}

impl<T: Scalar> Drop for AscendStorage<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let ptr = self.ptr;
            let _ = self.runtime.with_transaction(|_| {
                unsafe { sys::aclrtFree(ptr) };
                Ok(())
            });
        }
    }
}

impl<T: Scalar> Clone for AscendStorage<T> {
    fn clone(&self) -> Self {
        let copy = Self::allocate(self.runtime.clone(), self.len, false)
            .expect("Ascend aclrtMalloc during storage clone failed");
        let bytes = self.len * size_of::<T>();
        if bytes != 0 {
            self.runtime
                .with_transaction(|_| {
                    let status = unsafe {
                        sys::aclrtMemcpy(
                            copy.ptr,
                            bytes,
                            self.ptr,
                            bytes,
                            sys::ACL_MEMCPY_DEVICE_TO_DEVICE,
                        )
                    };
                    (status == sys::ACL_SUCCESS)
                        .then_some(())
                        .ok_or_else(|| AscendError::status("aclrtMemcpy(D2D)", status))
                })
                .expect("Ascend aclrtMemcpy(D2D) during storage clone failed");
        }
        copy
    }
}

impl<T: Scalar> Storage<T> for AscendStorage<T> {
    fn len(&self) -> usize {
        self.len
    }
    fn get(&self, index: usize) -> T {
        self.to_vec()
            .expect("Ascend aclrtMemcpy(D2H) in get failed")[index]
    }
    fn set(&mut self, index: usize, value: T) {
        let mut data = self
            .to_vec()
            .expect("Ascend aclrtMemcpy(D2H) in set failed");
        data[index] = value;
        *self = Self::upload(self.runtime.clone(), &data)
            .expect("Ascend aclrtMemcpy(H2D) in set failed");
    }
    fn to_vec(&self) -> Vec<T> {
        AscendStorage::to_vec(self).expect("Ascend aclrtMemcpy(D2H) failed")
    }
    fn from_slice(_: &[T]) -> Self {
        panic!("AscendStorage::from_slice requires an Ascend context; use Ascend::from_slice")
    }
    fn zeros(_: usize) -> Self {
        panic!("AscendStorage::zeros requires an Ascend context; use Ascend::alloc")
    }
}
