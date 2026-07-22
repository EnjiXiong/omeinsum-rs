mod contract;
mod ffi;
mod normalize;
mod permute;
mod reduce;
mod runtime;
mod storage;
mod sys;
#[cfg(feature = "ascend-tropical")]
mod tropical;

pub use runtime::AscendError;
pub use storage::AscendStorage;

use crate::{
    algebra::{Algebra, Scalar},
    backend::{Backend, BackendScalar},
};
use runtime::Runtime;
use std::{any::TypeId, sync::Arc};

#[derive(Clone)]
pub struct Ascend {
    runtime: Arc<Runtime>,
}

impl Ascend {
    pub fn new() -> Result<Self, AscendError> {
        Self::on_device(0)
    }
    pub fn on_device(ordinal: i32) -> Result<Self, AscendError> {
        Ok(Self {
            runtime: Arc::new(Runtime::new(ordinal)?),
        })
    }
}

impl Default for Ascend {
    fn default() -> Self {
        Self::new().expect("Ascend initialization failed")
    }
}

impl Backend for Ascend {
    type Storage<T: Scalar> = AscendStorage<T>;
    fn name() -> &'static str {
        "ascend"
    }
    fn synchronize(&self) {
        self.runtime
            .synchronize()
            .expect("Ascend aclrtSynchronizeStream failed")
    }
    fn alloc<T: Scalar>(&self, len: usize) -> AscendStorage<T> {
        AscendStorage::allocate(self.runtime.clone(), len, true)
            .expect("Ascend aclrtMalloc/aclrtMemset failed")
    }
    fn from_slice<T: Scalar>(&self, data: &[T]) -> AscendStorage<T> {
        AscendStorage::upload(self.runtime.clone(), data)
            .expect("Ascend aclrtMalloc/aclrtMemcpy(H2D) failed")
    }
    fn copy_strided<T: Scalar>(
        &self,
        src: &AscendStorage<T>,
        shape: &[usize],
        strides: &[usize],
        offset: usize,
    ) -> AscendStorage<T> {
        let source = src
            .to_vec()
            .expect("Ascend aclrtMemcpy(D2H) in copy_strided failed");
        let mut output = vec![T::default(); shape.iter().product()];
        let mut index = vec![0usize; shape.len()];
        for value in &mut output {
            let source_index =
                offset + index.iter().zip(strides).map(|(i, s)| i * s).sum::<usize>();
            *value = source[source_index];
            for axis in 0..shape.len() {
                index[axis] += 1;
                if index[axis] < shape[axis] {
                    break;
                }
                index[axis] = 0;
            }
        }
        self.from_slice(&output)
    }
    fn contract<A: Algebra>(
        &self,
        a: &AscendStorage<A::Scalar>,
        shape_a: &[usize],
        strides_a: &[usize],
        modes_a: &[i32],
        b: &AscendStorage<A::Scalar>,
        shape_b: &[usize],
        strides_b: &[usize],
        modes_b: &[i32],
        shape_c: &[usize],
        modes_c: &[i32],
    ) -> AscendStorage<A::Scalar>
    where
        A::Scalar: BackendScalar<Self>,
    {
        if A::needs_argmax() {
            #[cfg(feature = "ascend-tropical")]
            {
                use crate::algebra::{MaxMul, MaxPlus, MinPlus};

                let a: &AscendStorage<f32> = unsafe { &*(a as *const _ as *const _) };
                let b: &AscendStorage<f32> = unsafe { &*(b as *const _ as *const _) };
                macro_rules! dispatch {
                    ($algebra:ty, $mode:expr) => {{
                        let (output, _) = tropical::contract::<$algebra>(
                            &self.runtime,
                            a,
                            shape_a,
                            strides_a,
                            modes_a,
                            b,
                            shape_b,
                            strides_b,
                            modes_b,
                            shape_c,
                            modes_c,
                            $mode,
                        )
                        .unwrap_or_else(|error| {
                            panic!("Ascend tropical contraction failed: {error}")
                        });
                        return unsafe {
                            std::mem::transmute::<AscendStorage<f32>, AscendStorage<A::Scalar>>(
                                output,
                            )
                        };
                    }};
                }
                if TypeId::of::<A>() == TypeId::of::<MaxPlus<f32>>() {
                    dispatch!(MaxPlus<f32>, 0)
                } else if TypeId::of::<A>() == TypeId::of::<MinPlus<f32>>() {
                    dispatch!(MinPlus<f32>, 1)
                } else if TypeId::of::<A>() == TypeId::of::<MaxMul<f32>>() {
                    dispatch!(MaxMul<f32>, 2)
                }
                panic!(
                    "Ascend tropical contraction supports MaxPlus/MinPlus/MaxMul<f32>; got {}",
                    std::any::type_name::<A>()
                )
            }
            #[cfg(not(feature = "ascend-tropical"))]
            panic!("Ascend tropical contraction requires the `ascend-tropical` feature")
        }
        if TypeId::of::<A>() != TypeId::of::<crate::algebra::Standard<f32>>() {
            panic!(
                "Ascend supports exactly Standard<f32> and feature-enabled f32 tropical algebras"
            )
        }
        let a: &AscendStorage<f32> = unsafe { &*(a as *const _ as *const _) };
        let b: &AscendStorage<f32> = unsafe { &*(b as *const _ as *const _) };
        let output = contract::contract(
            &self.runtime,
            a,
            shape_a,
            strides_a,
            modes_a,
            b,
            shape_b,
            strides_b,
            modes_b,
            shape_c,
            modes_c,
        )
        .unwrap_or_else(|error| panic!("Ascend ACLNN binary contraction failed: {error}"));
        unsafe { std::mem::transmute(output) }
    }
    fn contract_with_argmax<A: Algebra<Index = u32>>(
        &self,
        a: &AscendStorage<A::Scalar>,
        shape_a: &[usize],
        strides_a: &[usize],
        modes_a: &[i32],
        b: &AscendStorage<A::Scalar>,
        shape_b: &[usize],
        strides_b: &[usize],
        modes_b: &[i32],
        shape_c: &[usize],
        modes_c: &[i32],
    ) -> (AscendStorage<A::Scalar>, AscendStorage<u32>)
    where
        A::Scalar: BackendScalar<Self>,
    {
        #[cfg(feature = "ascend-tropical")]
        {
            use crate::algebra::{MaxMul, MaxPlus, MinPlus};

            let a: &AscendStorage<f32> = unsafe { &*(a as *const _ as *const _) };
            let b: &AscendStorage<f32> = unsafe { &*(b as *const _ as *const _) };
            macro_rules! dispatch {
                ($algebra:ty, $mode:expr) => {{
                    let (output, argmax) = tropical::contract::<$algebra>(
                        &self.runtime,
                        a,
                        shape_a,
                        strides_a,
                        modes_a,
                        b,
                        shape_b,
                        strides_b,
                        modes_b,
                        shape_c,
                        modes_c,
                        $mode,
                    )
                    .unwrap_or_else(|error| {
                        panic!("Ascend tropical argmax contraction failed: {error}")
                    });
                    return (
                        unsafe {
                            std::mem::transmute::<AscendStorage<f32>, AscendStorage<A::Scalar>>(
                                output,
                            )
                        },
                        argmax,
                    );
                }};
            }
            if TypeId::of::<A>() == TypeId::of::<MaxPlus<f32>>() {
                dispatch!(MaxPlus<f32>, 0)
            } else if TypeId::of::<A>() == TypeId::of::<MinPlus<f32>>() {
                dispatch!(MinPlus<f32>, 1)
            } else if TypeId::of::<A>() == TypeId::of::<MaxMul<f32>>() {
                dispatch!(MaxMul<f32>, 2)
            }
            panic!(
                "Ascend tropical argmax contraction supports MaxPlus/MinPlus/MaxMul<f32>; got {}",
                std::any::type_name::<A>()
            )
        }
        #[cfg(not(feature = "ascend-tropical"))]
        {
            let _ = (
                a, shape_a, strides_a, modes_a, b, shape_b, strides_b, modes_b, shape_c, modes_c,
            );
            panic!("Ascend tropical argmax contraction requires the `ascend-tropical` feature")
        }
    }
}
