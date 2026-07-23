//! Tensor operations for contraction.

use std::collections::HashMap;

use super::Tensor;
use crate::algebra::{Algebra, Scalar};
use crate::backend::{Backend, BackendScalar};

#[derive(Default)]
pub(crate) struct BinaryContractOptions {
    pub preferred_output_indices: Option<Vec<usize>>,
}

pub(crate) fn native_binary_output_indices(ia: &[usize], ib: &[usize], iy: &[usize]) -> Vec<usize> {
    let output_set: std::collections::HashSet<usize> = iy.iter().copied().collect();
    let ia_set: std::collections::HashSet<usize> = ia.iter().copied().collect();
    let ib_set: std::collections::HashSet<usize> = ib.iter().copied().collect();
    let mut native = Vec::with_capacity(iy.len());

    for &idx in ia {
        if output_set.contains(&idx) && !ib_set.contains(&idx) && !native.contains(&idx) {
            native.push(idx);
        }
    }
    for &idx in ib {
        if output_set.contains(&idx) && !ia_set.contains(&idx) && !native.contains(&idx) {
            native.push(idx);
        }
    }
    for &idx in ia {
        if output_set.contains(&idx) && ib_set.contains(&idx) && !native.contains(&idx) {
            native.push(idx);
        }
    }

    native
}

/// Compute output shape from input shapes and modes.
fn compute_output_shape(
    shape_a: &[usize],
    modes_a: &[i32],
    shape_b: &[usize],
    modes_b: &[i32],
    modes_c: &[i32],
) -> Vec<usize> {
    let mut shape_map = std::collections::HashMap::new();
    for (idx, &m) in modes_a.iter().enumerate() {
        shape_map.insert(m, shape_a[idx]);
    }
    for (idx, &m) in modes_b.iter().enumerate() {
        shape_map.insert(m, shape_b[idx]);
    }
    modes_c.iter().map(|m| shape_map[m]).collect()
}

fn dense_modes(ia: &[usize], ib: &[usize], iy: &[usize]) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    fn map_modes(labels: &[usize], modes: &mut HashMap<usize, i32>) -> Vec<i32> {
        labels
            .iter()
            .map(|&label| {
                if let Some(&mode) = modes.get(&label) {
                    mode
                } else {
                    let mode = i32::try_from(modes.len())
                        .expect("binary contraction has too many distinct index labels");
                    modes.insert(label, mode);
                    mode
                }
            })
            .collect()
    }

    let mut modes = HashMap::new();
    let modes_a = map_modes(ia, &mut modes);
    let modes_b = map_modes(ib, &mut modes);
    let modes_c = map_modes(iy, &mut modes);
    (modes_a, modes_b, modes_c)
}

impl<B: Backend> Tensor<f32, B> {
    /// Return `self + alpha * other`, materializing non-contiguous inputs first.
    ///
    /// # Panics
    /// Panics when the input shapes differ.
    pub fn linear_combination(&self, other: &Self, alpha: f32) -> Self {
        assert_eq!(self.shape(), other.shape(), "tensor shapes must match");
        let x = self.contiguous();
        let y = other.contiguous();
        let storage = self.backend.linear_combination_f32(
            x.storage().expect("contiguous tensor has storage"),
            y.storage().expect("contiguous tensor has storage"),
            alpha,
        );
        Self::from_storage(storage, self.shape(), self.backend.clone())
    }
}

impl<T: Scalar, B: Backend> Tensor<T, B> {
    /// Binary tensor contraction using reshape-to-GEMM strategy.
    ///
    /// # Arguments
    ///
    /// * `other` - The other tensor to contract with
    /// * `ia` - Index labels for self
    /// * `ib` - Index labels for other
    /// * `iy` - Output index labels
    ///
    /// # Example
    ///
    /// ```rust
    /// use omeinsum::{Tensor, Cpu};
    /// use omeinsum::algebra::Standard;
    ///
    /// // A[i,j,k] × B[j,k,l] → C[i,l]
    /// let a = Tensor::<f32, Cpu>::from_data(&(0..24).map(|x| x as f32).collect::<Vec<_>>(), &[2, 3, 4]);
    /// let b = Tensor::<f32, Cpu>::from_data(&(0..60).map(|x| x as f32).collect::<Vec<_>>(), &[3, 4, 5]);
    /// let c = a.contract_binary::<Standard<f32>>(&b, &[0, 1, 2], &[1, 2, 3], &[0, 3]);
    /// assert_eq!(c.shape(), &[2, 5]);
    /// ```
    pub fn contract_binary<A: Algebra<Scalar = T, Index = u32>>(
        &self,
        other: &Self,
        ia: &[usize],
        ib: &[usize],
        iy: &[usize],
    ) -> Self
    where
        T: BackendScalar<B>,
    {
        let (result, _) = self.contract_binary_impl::<A>(other, ia, ib, iy, false);
        result
    }

    /// Binary contraction with argmax tracking.
    pub fn contract_binary_with_argmax<A: Algebra<Scalar = T, Index = u32>>(
        &self,
        other: &Self,
        ia: &[usize],
        ib: &[usize],
        iy: &[usize],
    ) -> (Self, Tensor<u32, B>)
    where
        T: BackendScalar<B>,
    {
        let (result, argmax) = self.contract_binary_impl::<A>(other, ia, ib, iy, true);
        (result, argmax.expect("argmax requested but not returned"))
    }

    pub(crate) fn contract_binary_with_options<A: Algebra<Scalar = T, Index = u32>>(
        &self,
        other: &Self,
        ia: &[usize],
        ib: &[usize],
        iy: &[usize],
        options: &BinaryContractOptions,
    ) -> Self
    where
        T: BackendScalar<B>,
    {
        let (result, _) =
            self.contract_binary_impl_with_options::<A>(other, ia, ib, iy, false, options);
        result
    }

    pub(crate) fn contract_binary_native_order<A: Algebra<Scalar = T, Index = u32>>(
        &self,
        other: &Self,
        ia: &[usize],
        ib: &[usize],
        iy: &[usize],
    ) -> (Self, Vec<usize>)
    where
        T: BackendScalar<B>,
    {
        let mut output_indices = native_binary_output_indices(ia, ib, iy);
        if output_indices.len() != iy.len() {
            output_indices = iy.to_vec();
        }
        let options = BinaryContractOptions {
            preferred_output_indices: Some(output_indices.clone()),
        };
        (
            self.contract_binary_with_options::<A>(other, ia, ib, iy, &options),
            output_indices,
        )
    }

    pub(crate) fn contract_binary_with_argmax_with_options<A: Algebra<Scalar = T, Index = u32>>(
        &self,
        other: &Self,
        ia: &[usize],
        ib: &[usize],
        iy: &[usize],
        options: &BinaryContractOptions,
    ) -> (Self, Tensor<u32, B>)
    where
        T: BackendScalar<B>,
    {
        let (result, argmax) =
            self.contract_binary_impl_with_options::<A>(other, ia, ib, iy, true, options);
        (result, argmax.expect("argmax requested but not returned"))
    }

    fn contract_binary_impl<A: Algebra<Scalar = T, Index = u32>>(
        &self,
        other: &Self,
        ia: &[usize],
        ib: &[usize],
        iy: &[usize],
        track_argmax: bool,
    ) -> (Self, Option<Tensor<u32, B>>)
    where
        T: BackendScalar<B>,
    {
        self.contract_binary_impl_with_options::<A>(
            other,
            ia,
            ib,
            iy,
            track_argmax,
            &BinaryContractOptions::default(),
        )
    }

    pub(crate) fn contract_binary_impl_with_options<A: Algebra<Scalar = T, Index = u32>>(
        &self,
        other: &Self,
        ia: &[usize],
        ib: &[usize],
        iy: &[usize],
        track_argmax: bool,
        options: &BinaryContractOptions,
    ) -> (Self, Option<Tensor<u32, B>>)
    where
        T: BackendScalar<B>,
    {
        assert_eq!(ia.len(), self.ndim(), "ia length must match self.ndim()");
        assert_eq!(ib.len(), other.ndim(), "ib length must match other.ndim()");

        let output_indices = options.preferred_output_indices.as_deref().unwrap_or(iy);
        if let Some(preferred_output_indices) = &options.preferred_output_indices {
            let mut preferred_sorted = preferred_output_indices.clone();
            preferred_sorted.sort_unstable();
            let mut output_sorted = iy.to_vec();
            output_sorted.sort_unstable();
            debug_assert_eq!(
                preferred_sorted, output_sorted,
                "preferred output indices must be a permutation of iy"
            );
        }

        // Backends use i32 modes, while the public API permits arbitrary usize
        // labels. Dense remapping preserves label identity without narrowing.
        let (modes_a, modes_b, modes_c) = dense_modes(ia, ib, output_indices);

        // Compute output shape
        let shape_c =
            compute_output_shape(self.shape(), &modes_a, other.shape(), &modes_b, &modes_c);

        if track_argmax {
            let (c_storage, argmax_storage) = self.backend.contract_with_argmax::<A>(
                self.storage.as_ref(),
                self.shape(),
                self.strides(),
                &modes_a,
                other.storage.as_ref(),
                other.shape(),
                other.strides(),
                &modes_b,
                &shape_c,
                &modes_c,
            );

            let c = Self::from_storage(c_storage, &shape_c, self.backend.clone());
            let argmax =
                Tensor::<u32, B>::from_storage(argmax_storage, &shape_c, self.backend.clone());
            (c, Some(argmax))
        } else {
            let c_storage = self.backend.contract::<A>(
                self.storage.as_ref(),
                self.shape(),
                self.strides(),
                &modes_a,
                other.storage.as_ref(),
                other.shape(),
                other.strides(),
                &modes_b,
                &shape_c,
                &modes_c,
            );

            let c = Self::from_storage(c_storage, &shape_c, self.backend.clone());
            (c, None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::Standard;
    use crate::backend::Cpu;

    #[test]
    fn linear_combination_computes_concrete_values() {
        let x = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let y = Tensor::<f32, Cpu>::from_data(&[4.0, 3.0, 2.0, 1.0], &[2, 2]);

        assert_eq!(x.linear_combination(&y, 1.0).to_vec(), vec![5.0; 4]);
        assert_eq!(
            x.linear_combination(&y, -1.0).to_vec(),
            vec![-3.0, -1.0, 1.0, 3.0]
        );
    }

    #[test]
    fn linear_combination_materializes_non_contiguous_inputs() {
        let x = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let y = Tensor::<f32, Cpu>::from_data(&[6.0, 5.0, 4.0, 3.0, 2.0, 1.0], &[2, 3]);

        let result = x
            .permute(&[1, 0])
            .linear_combination(&y.permute(&[1, 0]), -1.0);
        assert_eq!(result.shape(), &[3, 2]);
        assert_eq!(result.to_vec(), vec![-5.0, -1.0, 3.0, -3.0, 1.0, 5.0]);
    }

    #[cfg(feature = "tropical")]
    use crate::algebra::MaxPlus;

    #[test]
    fn test_contract_binary_matmul_standard() {
        // A[i,j] × B[j,k] → C[i,k] (matrix multiplication)
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let c = a.contract_binary::<Standard<f32>>(&b, &[0, 1], &[1, 2], &[0, 2]);

        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.to_vec(), vec![7.0, 10.0, 15.0, 22.0]);
    }

    #[test]
    fn test_contract_binary_preserves_distinct_large_labels() {
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::<f32, Cpu>::from_data(&[5.0, 6.0], &[2]);
        let aliased_if_narrowed = u32::MAX as usize;

        let c = a.contract_binary::<Standard<f32>>(
            &b,
            &[aliased_if_narrowed, usize::MAX],
            &[usize::MAX],
            &[aliased_if_narrowed],
        );

        assert_eq!(c.shape(), &[2]);
        assert_eq!(c.to_vec(), vec![23.0, 34.0]);
    }

    #[cfg(feature = "tropical")]
    #[test]
    fn test_contract_binary_matmul_maxplus() {
        // A[i,j] × B[j,k] → C[i,k] (tropical matrix multiplication)
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let c = a.contract_binary::<MaxPlus<f32>>(&b, &[0, 1], &[1, 2], &[0, 2]);

        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.to_vec(), vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_contract_binary() {
        // A[i,j] × B[j,k] → C[i,k]
        let a = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let c = a.contract_binary::<Standard<f32>>(&b, &[0, 1], &[1, 2], &[0, 2]);

        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.to_vec(), vec![7.0, 10.0, 15.0, 22.0]);
    }

    #[test]
    fn test_contract_binary_batched() {
        // A[b,i,j] × B[b,j,k] → C[b,i,k]
        // 2 batches, 2x2 matrices
        // Column-major layout: A[b,i,j] at position b + 2*i + 4*j
        let a =
            Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
        let b =
            Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 1.0, 0.0, 0.0, 1.0], &[2, 2, 2]);

        let c = a.contract_binary::<Standard<f32>>(&b, &[0, 1, 2], &[0, 2, 3], &[0, 1, 3]);

        assert_eq!(c.shape(), &[2, 2, 2]);
        // In column-major [2,2,2]:
        // Batch 0 of A: [[1,5],[3,7]], Batch 1 of A: [[2,6],[4,8]]
        // Batch 0 of B: [[1,1],[3,0]], Batch 1 of B: [[2,0],[4,1]]
        // Batch 0 result: [[16,1],[24,3]], Batch 1 result: [[28,6],[40,8]]
        // Column-major output: [16, 28, 24, 40, 1, 6, 3, 8]
        assert_eq!(c.to_vec(), vec![16.0, 28.0, 24.0, 40.0, 1.0, 6.0, 3.0, 8.0]);
    }

    #[test]
    fn test_contract_binary_native_order_returns_backend_order() {
        // A[b,i,j] x B[b,j,k] -> C[b,i,k].
        // The CPU backend naturally emits [i,k,b], which avoids a materializing
        // output permutation when this is only an intermediate contraction.
        let a =
            Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
        let b =
            Tensor::<f32, Cpu>::from_data(&[1.0, 2.0, 3.0, 4.0, 1.0, 0.0, 0.0, 1.0], &[2, 2, 2]);

        let final_order =
            a.contract_binary::<Standard<f32>>(&b, &[0, 1, 2], &[0, 2, 3], &[0, 1, 3]);
        let (native, native_indices) =
            a.contract_binary_native_order::<Standard<f32>>(&b, &[0, 1, 2], &[0, 2, 3], &[0, 1, 3]);

        assert_eq!(native_indices, vec![1, 3, 0]);
        assert_eq!(native.shape(), &[2, 2, 2]);
        assert_eq!(native.permute(&[2, 0, 1]).to_vec(), final_order.to_vec());
    }
}
