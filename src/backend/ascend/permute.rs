use super::{
    ffi::{checked_product, finish_operation, tensor, EnqueueAttempt, IntArray, PendingOperation},
    runtime::Runtime,
    storage::AscendStorage,
    sys, AscendError,
};
use crate::backend::{
    contract_plan::{materialize_strided, plan_permutation_steps},
    Storage,
};
use std::{ptr, sync::Arc};

const ACLNN_PERMUTE_MAX_RANK: usize = 8;

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct DensePermutation {
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    dims: Vec<i64>,
}

pub(crate) fn dense_permutation(
    storage_len: usize,
    shape: &[usize],
    strides: &[usize],
    permutation: &[usize],
) -> Result<Option<DensePermutation>, AscendError> {
    if shape.len() != strides.len() || shape.len() != permutation.len() {
        return Err(AscendError::status(
            "Ascend permutation metadata rank mismatch",
            -1,
        ));
    }
    if permutation
        .iter()
        .enumerate()
        .any(|(axis, &source)| source >= shape.len() || permutation[..axis].contains(&source))
    {
        return Err(AscendError::status("Ascend invalid permutation", -1));
    }
    let numel = checked_product(shape, "Ascend permutation size overflow")?;
    // CANN aclnnPermute rejects tensors above rank 8 with ACLNN_ERR_PARAM_INVALID.
    // Let the caller use the existing host materialization path for those views.
    if shape.len() > ACLNN_PERMUTE_MAX_RANK {
        return Ok(None);
    }
    if numel != storage_len {
        return Ok(None);
    }

    // Tensor storage is column-major. Sorting logical axes by stride recovers
    // the physical contiguous axis order without copying the input.
    let mut physical_axes: Vec<usize> = (0..shape.len()).collect();
    physical_axes.sort_by_key(|&axis| (strides[axis], axis));
    let mut expected_stride = 1usize;
    for &axis in &physical_axes {
        if strides[axis] != expected_stride {
            return Ok(None);
        }
        expected_stride = expected_stride
            .checked_mul(shape[axis])
            .ok_or_else(|| AscendError::status("Ascend permutation stride overflow", -1))?;
    }

    // ACLNN describes row-major tensors, so reverse both the physical input
    // axes and the requested column-major output axes.
    let input_axes: Vec<usize> = physical_axes.into_iter().rev().collect();
    let output_axes: Vec<usize> = permutation.iter().rev().copied().collect();
    let input_shape = input_axes.iter().map(|&axis| shape[axis]).collect();
    let output_shape = output_axes.iter().map(|&axis| shape[axis]).collect();
    let dims = output_axes
        .iter()
        .map(|axis| {
            input_axes
                .iter()
                .position(|candidate| candidate == axis)
                .and_then(|position| i64::try_from(position).ok())
                .ok_or_else(|| AscendError::status("Ascend permutation axis overflow", -1))
        })
        .collect::<Result<_, _>>()?;
    Ok(Some(DensePermutation {
        input_shape,
        output_shape,
        dims,
    }))
}

pub(crate) fn enqueue(
    runtime: &Runtime,
    source: &AscendStorage<f32>,
    output: &AscendStorage<f32>,
    plan: &DensePermutation,
) -> Result<EnqueueAttempt, AscendError> {
    let input_tensor = tensor(source.as_mut_ptr(), &plan.input_shape)?;
    let output_tensor = tensor(output.as_mut_ptr(), &plan.output_shape)?;
    let dims = IntArray::new(&plan.dims, "aclCreateIntArray(permute)")?;
    let mut workspace_size = 0u64;
    let mut executor = ptr::null_mut();
    let status = unsafe {
        sys::aclnnPermuteGetWorkspaceSize(
            input_tensor.0,
            dims.0,
            output_tensor.0,
            &mut workspace_size,
            &mut executor,
        )
    };
    if status != sys::ACL_SUCCESS {
        return Err(AscendError::status("aclnnPermuteGetWorkspaceSize", status));
    }
    if executor.is_null() {
        return Err(AscendError::null("aclnnPermuteGetWorkspaceSize executor"));
    }
    let bytes = usize::try_from(workspace_size)
        .map_err(|_| AscendError::status("aclnnPermute workspace overflow", -1))?;
    let workspace = runtime.alloc_locked(bytes, false)?;
    let operation = PendingOperation::new(vec![input_tensor, output_tensor], vec![dims], workspace);
    let status = unsafe { sys::aclnnPermute(workspace, workspace_size, executor, runtime.stream) };
    let result = (status == sys::ACL_SUCCESS)
        .then_some(())
        .ok_or_else(|| AscendError::status("aclnnPermute", status));
    Ok(EnqueueAttempt::new(operation, result))
}

fn materialize_dense(
    runtime: &Arc<Runtime>,
    source: &AscendStorage<f32>,
    plan: DensePermutation,
) -> Result<AscendStorage<f32>, AscendError> {
    let output = AscendStorage::allocate(runtime.clone(), source.len(), false)?;
    runtime.with_transaction(|runtime| {
        let (pending, launch_result) = enqueue(runtime, source, &output, &plan)?.into_parts();
        finish_operation(
            pending,
            launch_result,
            || runtime.synchronize_locked(),
            |operation| operation.release(runtime),
        )
    })?;
    Ok(output)
}

pub(crate) fn materialize(
    runtime: &Arc<Runtime>,
    source: &AscendStorage<f32>,
    shape: &[usize],
    strides: &[usize],
    permutation: &[usize],
) -> Result<AscendStorage<f32>, AscendError> {
    match route_permutation(source.len(), shape, strides, permutation)? {
        PermuteRoute::Single(plan) => materialize_dense(runtime, source, plan),
        PermuteRoute::Steps(steps) => {
            let verify = std::env::var_os("OMEINSUM_PERMUTE_VERIFY").is_some();
            let mut iter = steps.into_iter();
            let first = iter
                .next()
                .expect("decomposed permutation has at least one step");
            let mut current = materialize_dense(runtime, source, first)?;
            for step in iter {
                current = materialize_dense(runtime, &current, step)?;
            }
            if verify {
                let device = current.to_vec()?;
                let host_source = source.to_vec()?;
                let expected = materialize_strided(&host_source, shape, strides, permutation);
                assert_eq!(
                    device, expected,
                    "decomposed permute diverged on device: shape={shape:?} strides={strides:?} permutation={permutation:?}"
                );
            }
            Ok(current)
        }
        PermuteRoute::Host => {
            let host = source.to_vec()?;
            AscendStorage::upload(
                runtime.clone(),
                &materialize_strided(&host, shape, strides, permutation),
            )
        }
    }
}

/// How a strided-view materialization can be executed.
enum PermuteRoute {
    /// One aclnnPermute call (rank <= 8).
    Single(DensePermutation),
    /// A chain of aclnnPermute calls, each rank <= 8, for high-rank views
    /// (W8: replaces the host round trip through the CPU).
    Steps(Vec<DensePermutation>),
    /// Storage does not match the view (non-contiguous strides or wrong
    /// element count): keep the host strided-gather path.
    Host,
}

fn route_permutation(
    storage_len: usize,
    shape: &[usize],
    strides: &[usize],
    permutation: &[usize],
) -> Result<PermuteRoute, AscendError> {
    if std::env::var_os("OMEINSUM_PERMUTE_FORCE_HOST").is_some() {
        return Ok(PermuteRoute::Host);
    }
    if let Some(plan) = dense_permutation(storage_len, shape, strides, permutation)? {
        return Ok(PermuteRoute::Single(plan));
    }
    // dense_permutation declines three cases; only rank > 8 is decomposable.
    if shape.len() <= ACLNN_PERMUTE_MAX_RANK {
        return Ok(PermuteRoute::Host);
    }
    let numel = checked_product(shape, "Ascend permutation size overflow")?;
    if numel != storage_len {
        return Ok(PermuteRoute::Host);
    }
    let mut physical_axes: Vec<usize> = (0..shape.len()).collect();
    physical_axes.sort_by_key(|&axis| (strides[axis], axis));
    let mut expected_stride = 1usize;
    for &axis in &physical_axes {
        if strides[axis] != expected_stride {
            return Ok(PermuteRoute::Host);
        }
        expected_stride = expected_stride
            .checked_mul(shape[axis])
            .ok_or_else(|| AscendError::status("Ascend permutation stride overflow", -1))?;
    }
    // Row-major picture, as in dense_permutation: physical input order and the
    // requested output order, both in original axis ids.
    let input_axes: Vec<usize> = physical_axes.into_iter().rev().collect();
    let output_axes: Vec<usize> = permutation.iter().rev().copied().collect();
    if input_axes == output_axes {
        // Identity permutations are handled by callers; keep the host path as
        // the defensive answer.
        return Ok(PermuteRoute::Host);
    }
    let steps = plan_permutation_steps(shape, &input_axes, &output_axes, ACLNN_PERMUTE_MAX_RANK)
        .into_iter()
        .map(|step| DensePermutation {
            input_shape: step.input_shape,
            output_shape: step.output_shape,
            dims: step.dims,
        })
        .collect::<Vec<_>>();
    if std::env::var_os("OMEINSUM_DEBUG_PERMUTE").is_some() {
        eprintln!(
            "permute rank {}: shape={shape:?} strides={strides:?} permutation={permutation:?} steps={}",
            shape.len(),
            steps.len()
        );
        for (i, step) in steps.iter().enumerate() {
            eprintln!(
                "  step {i}: in={:?} out={:?} dims={:?}",
                step.input_shape, step.output_shape, step.dims
            );
        }
    }
    Ok(PermuteRoute::Steps(steps))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apply_row_major_permutation<T: Copy + Default>(
        input: &[T],
        input_shape: &[usize],
        dims: &[i64],
    ) -> Vec<T> {
        let output_shape: Vec<usize> = dims
            .iter()
            .map(|&axis| input_shape[usize::try_from(axis).unwrap()])
            .collect();
        let mut output = vec![T::default(); input.len()];
        for (output_linear, value) in output.iter_mut().enumerate() {
            let mut remaining = output_linear;
            let mut input_coordinates = vec![0usize; input_shape.len()];
            for output_axis in (0..output_shape.len()).rev() {
                let coordinate = remaining % output_shape[output_axis];
                remaining /= output_shape[output_axis];
                input_coordinates[usize::try_from(dims[output_axis]).unwrap()] = coordinate;
            }
            let input_linear = input_coordinates
                .iter()
                .zip(input_shape)
                .fold(0usize, |linear, (&coordinate, &extent)| {
                    linear * extent + coordinate
                });
            *value = input[input_linear];
        }
        output
    }

    #[test]
    fn maps_column_major_views_to_acl_permutations() {
        let cases = [
            (
                1,
                &[][..],
                &[][..],
                &[][..],
                DensePermutation {
                    input_shape: vec![],
                    output_shape: vec![],
                    dims: vec![],
                },
            ),
            (
                3,
                &[3][..],
                &[1][..],
                &[0][..],
                DensePermutation {
                    input_shape: vec![3],
                    output_shape: vec![3],
                    dims: vec![0],
                },
            ),
            (
                24,
                &[2, 3, 4][..],
                &[1, 8, 2][..],
                &[2, 0, 1][..],
                DensePermutation {
                    input_shape: vec![3, 4, 2],
                    output_shape: vec![3, 2, 4],
                    dims: vec![0, 2, 1],
                },
            ),
            (
                6,
                &[2, 1, 3][..],
                &[1, 2, 2][..],
                &[2, 0, 1][..],
                DensePermutation {
                    input_shape: vec![3, 1, 2],
                    output_shape: vec![1, 2, 3],
                    dims: vec![1, 2, 0],
                },
            ),
        ];
        for (storage_len, shape, strides, permutation, expected) in cases {
            assert_eq!(
                dense_permutation(storage_len, shape, strides, permutation).unwrap(),
                Some(expected)
            );
        }
    }

    #[test]
    fn acl_mapping_matches_host_materialization_values() {
        let input: Vec<i32> = (0..24).collect();
        let shape = [2, 3, 4];
        let strides = [1, 8, 2];
        let permutation = [2, 0, 1];
        let plan = dense_permutation(input.len(), &shape, &strides, &permutation)
            .unwrap()
            .unwrap();
        let actual = apply_row_major_permutation(&input, &plan.input_shape, &plan.dims);
        let expected = materialize_strided(&input, &shape, &strides, &permutation);
        assert_eq!(actual, expected);
    }

    #[test]
    fn rejects_views_that_require_host_fallback() {
        assert_eq!(
            dense_permutation(12, &[2, 3], &[1, 4], &[0, 1]).unwrap(),
            None
        );
        assert_eq!(
            dense_permutation(7, &[2, 3], &[1, 2], &[0, 1]).unwrap(),
            None
        );
        assert_eq!(
            dense_permutation(0, &[2, 0, 3], &[1, 2, 0], &[0, 1, 2]).unwrap(),
            None
        );
        assert_eq!(
            dense_permutation(
                512,
                &[2; 9],
                &[1, 2, 4, 8, 16, 32, 64, 128, 256],
                &[1, 2, 3, 4, 5, 6, 7, 8, 0],
            )
            .unwrap(),
            None
        );
    }

    #[test]
    fn rejects_invalid_or_overflowing_metadata() {
        assert!(dense_permutation(6, &[2, 3], &[1, 2], &[0]).is_err());
        assert!(dense_permutation(6, &[2, 3], &[1, 2], &[0, 0]).is_err());
        assert!(dense_permutation(6, &[2, 3], &[1, 2], &[0, 2]).is_err());
        assert!(
            dense_permutation(usize::MAX, &[usize::MAX, 2], &[1, usize::MAX], &[0, 1]).is_err()
        );
    }

    fn canonical_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = Vec::with_capacity(shape.len());
        let mut stride = 1usize;
        for &extent in shape {
            strides.push(stride);
            stride *= extent;
        }
        strides
    }

    #[test]
    fn non_contiguous_or_mismatched_high_rank_stays_host() {
        let shape = [2usize; 9];
        let strides = canonical_strides(&shape);
        let permutation: Vec<usize> = (0..9).rev().collect();
        // wrong element count
        assert!(matches!(
            route_permutation(7, &shape, &strides, &permutation).unwrap(),
            PermuteRoute::Host
        ));
        // non-contiguous strides
        let mut bad_strides = strides.clone();
        bad_strides[3] = 99;
        assert!(matches!(
            route_permutation(512, &shape, &bad_strides, &permutation).unwrap(),
            PermuteRoute::Host
        ));
        // identity stays host (callers handle it upstream)
        let identity: Vec<usize> = (0..9).collect();
        assert!(matches!(
            route_permutation(512, &shape, &strides, &identity).unwrap(),
            PermuteRoute::Host
        ));
    }
}
