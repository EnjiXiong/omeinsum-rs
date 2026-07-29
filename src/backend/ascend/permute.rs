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
use std::{
    collections::HashMap,
    ptr,
    sync::{Arc, Mutex, OnceLock},
};

const ACLNN_PERMUTE_API_MAX_RANK: usize = 8;
// CANN 8.5 accepts rank-8 aclnnPermute calls, but one merged-view
// shape/permutation used by the tensor-network workload is known to
// mis-execute silently. Keep ordinary rank-8 tensors on the direct API path,
// Rank-7 merged views also proved unstable in the end-to-end workload, so
// decompose high-rank views into steps no larger than rank 6.
const ACLNN_DECOMPOSED_PERMUTE_MAX_RANK: usize = 6;

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
    if shape.len() > ACLNN_PERMUTE_API_MAX_RANK {
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
            // CANN silently mis-executes aclnnPermute for specific shape/dims
            // combinations at its supported rank limit (observed: rank-8 merged-view
            // plan [2,2,2,2,16,4,2,128] x dims [4,0,5,1,6,2,7,3] on CANN 8.5,
            // 99.9% of elements misplaced). The mis-fire region is a black
            // box, so the first execution of every unique high-rank
            // permutation is verified bitwise against the host materializer;
            // failing signatures permanently fall back to the host path.
            let key = (
                shape.to_vec(),
                strides.to_vec(),
                permutation.to_vec(),
            );
            let verdict = verify_cache()
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get(&key)
                .copied();
            if verdict == Some(false) {
                let host = source.to_vec()?;
                return AscendStorage::upload(
                    runtime.clone(),
                    &materialize_strided(&host, shape, strides, permutation),
                );
            }
            let mut iter = steps.into_iter();
            let first = iter
                .next()
                .expect("decomposed permutation has at least one step");
            let mut current = materialize_dense(runtime, source, first)?;
            for step in iter {
                current = materialize_dense(runtime, &current, step)?;
            }
            if verdict.is_none() {
                let device = current.to_vec()?;
                let host_source = source.to_vec()?;
                let expected = materialize_strided(&host_source, shape, strides, permutation);
                let good = device == expected;
                verify_cache()
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .insert(key, good);
                if !good {
                    eprintln!(
                        "aclnnPermute mis-executed a decomposed plan; falling back to host for this permutation"
                    );
                    return AscendStorage::upload(runtime.clone(), &expected);
                }
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

/// Per-process verdicts of the first execution of each unique high-rank
/// permutation (see the Steps arm of `materialize`).
type Signature = (Vec<usize>, Vec<usize>, Vec<usize>);
static VERIFY_CACHE: OnceLock<Mutex<HashMap<Signature, bool>>> = OnceLock::new();
fn verify_cache() -> &'static Mutex<HashMap<Signature, bool>> {
    VERIFY_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// How a strided-view materialization can be executed.
enum PermuteRoute {
    /// One aclnnPermute call (rank <= 8).
    Single(DensePermutation),
    /// A chain of aclnnPermute calls, each rank <= 6, for high-rank views
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
    if shape.len() <= ACLNN_PERMUTE_API_MAX_RANK {
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
    let steps = plan_permutation_steps(
        shape,
        &input_axes,
        &output_axes,
        ACLNN_DECOMPOSED_PERMUTE_MAX_RANK,
    )
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

    #[test]
    fn high_rank_routes_avoid_rank_eight_steps() {
        let shape = [2usize; 10];
        let strides = canonical_strides(&shape);
        // In the backend's reversed row-major picture this places three
        // separated target axes and makes the old rank-8 planner emit an
        // eight-super-axis step.
        let permutation = [0, 1, 2, 4, 6, 8, 3, 5, 7, 9];
        let steps = match route_permutation(1024, &shape, &strides, &permutation).unwrap() {
            PermuteRoute::Steps(steps) => steps,
            _ => panic!("high-rank contiguous permutation must use device steps"),
        };
        assert!(!steps.is_empty());
        assert!(
            steps.iter().all(|step| step.input_shape.len() <= 6),
            "high-rank route emitted a device step above rank 6: {steps:?}"
        );
    }

    /// Device regression for the exact CANN 8.5 rank-8 signature observed to
    /// mis-execute. Decomposing it into rank-7 and rank-6 calls must reproduce
    /// the direct host permutation without invoking the faulty rank-8 call.
    #[test]
    fn device_bad_rank_eight_signature_works_when_decomposed() {
        let runtime = match Runtime::new(0) {
            Ok(runtime) => std::sync::Arc::new(runtime),
            Err(error) => {
                eprintln!("no Ascend device available, skipping probe: {error}");
                return;
            }
        };
        let shape = vec![2usize, 2, 2, 2, 16, 4, 2, 128];
        let dims = vec![4i64, 0, 5, 1, 6, 2, 7, 3];
        let input_axes: Vec<usize> = (0..shape.len()).collect();
        let output_axes: Vec<usize> = dims
            .iter()
            .map(|&axis| usize::try_from(axis).unwrap())
            .collect();
        let numel: usize = shape.iter().product();
        let data: Vec<f32> = (0..numel).map(|x| x as f32).collect();
        let expected = apply_row_major_permutation(&data, &shape, &dims);

        for max_rank in [7usize, 6] {
            let steps = plan_permutation_steps(&shape, &input_axes, &output_axes, max_rank);
            assert!(!steps.is_empty());
            assert!(
                steps.iter().all(|step| step.input_shape.len() <= max_rank),
                "rank-{max_rank} plan exceeded its limit: {steps:?}"
            );
            let step_ranks = steps
                .iter()
                .map(|step| step.input_shape.len())
                .collect::<Vec<_>>();
            let mut current =
                AscendStorage::upload(runtime.clone(), &data).expect("upload probe tensor");
            for step in steps {
                current = materialize_dense(
                    &runtime,
                    &current,
                    DensePermutation {
                        input_shape: step.input_shape,
                        output_shape: step.output_shape,
                        dims: step.dims,
                    },
                )
                .expect("execute decomposed device permute");
            }
            let device = current.to_vec().expect("download decomposed result");
            assert_eq!(
                device, expected,
                "rank-{max_rank} decomposition diverged from host"
            );
            println!(
                "rank-{max_rank} decomposition PASS: {} steps, ranks={step_ranks:?}",
                step_ranks.len()
            );
        }
    }

    /// Device probe (runs only where an Ascend device exists): execute single
    /// aclnnPermute plans through materialize_dense and compare with the host
    /// reference. Maps which (input_shape, dims) CANN executes incorrectly —
    /// the W8 cliff fix depends on knowing this constraint exactly.
    #[test]
    fn device_permute_sweep_matches_host() {
        let runtime = match Runtime::new(0) {
            Ok(runtime) => std::sync::Arc::new(runtime),
            Err(error) => {
                eprintln!("no Ascend device available, skipping probe: {error}");
                return;
            }
        };
        let cases: Vec<(Vec<usize>, Vec<i64>)> = vec![
            // A: the exact case caught by OMEINSUM_PERMUTE_VERIFY (FAIL expected)
            (vec![2, 2, 2, 2, 16, 4, 2, 128], vec![4, 0, 5, 1, 6, 2, 7, 3]),
            // B: same dims, smaller merged extents
            (vec![2, 2, 2, 2, 4, 4, 2, 8], vec![4, 0, 5, 1, 6, 2, 7, 3]),
            // C: same dims, shrink only the 128-extent axis
            (vec![2, 2, 2, 2, 16, 4, 2, 16], vec![4, 0, 5, 1, 6, 2, 7, 3]),
            // D: same dims, 128 -> 64
            (vec![2, 2, 2, 2, 16, 4, 2, 64], vec![4, 0, 5, 1, 6, 2, 7, 3]),
            // E: same shape as A, identity dims (control)
            (vec![2, 2, 2, 2, 16, 4, 2, 128], vec![0, 1, 2, 3, 4, 5, 6, 7]),
            // F: same dims as A, all-2 extents (W7-style full axes)
            (vec![2, 2, 2, 2, 2, 2, 2, 2], vec![4, 0, 5, 1, 6, 2, 7, 3]),
            // G: same shape as A, full reverse dims
            (vec![2, 2, 2, 2, 16, 4, 2, 128], vec![7, 6, 5, 4, 3, 2, 1, 0]),
            // H: rank-8 case that verified OK on device in the verify run
            (vec![2, 2, 64, 2, 2, 2, 2, 2], vec![1, 5, 7, 0, 2, 4, 6, 3]),
            // I: big axis is the innermost input and stays innermost
            (vec![2, 2, 2, 2, 16, 4, 2, 128], vec![0, 1, 2, 3, 5, 4, 6, 7]),
            // J: big innermost axis moves out (extent 2 in, big stays)
            (vec![2, 2, 2, 2, 16, 4, 128, 2], vec![4, 0, 5, 1, 6, 2, 7, 3]),
        ];
        let mut failures = 0;
        for (index, (input_shape, dims)) in cases.iter().enumerate() {
            let numel: usize = input_shape.iter().product();
            let data: Vec<f32> = (0..numel).map(|x| x as f32).collect();
            let source = AscendStorage::upload(runtime.clone(), &data)
                .expect("upload probe tensor");
            let output_shape: Vec<usize> =
                dims.iter().map(|&d| input_shape[d as usize]).collect();
            let plan = DensePermutation {
                input_shape: input_shape.clone(),
                output_shape,
                dims: dims.clone(),
            };
            let device = materialize_dense(&runtime, &source, plan)
                .and_then(|output| output.to_vec())
                .expect("device permute");
            let expected = apply_row_major_permutation(&data, input_shape, dims);
            if device == expected {
                println!("case {index}: PASS shape={input_shape:?} dims={dims:?}");
            } else {
                failures += 1;
                let mismatches = device
                    .iter()
                    .zip(&expected)
                    .filter(|(a, b)| a != b)
                    .count();
                println!(
                    "case {index}: FAIL shape={input_shape:?} dims={dims:?} mismatches={mismatches}/{numel}"
                );
            }
        }
        assert_eq!(failures, 0, "{failures} probe cases diverged on device");
    }
}
