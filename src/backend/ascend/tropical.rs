use super::{
    contract::validate_metadata, normalize::normalize_repeated_labels, runtime::Runtime,
    storage::AscendStorage, sys, AscendError,
};
use crate::{
    algebra::Algebra,
    backend::{
        contract_plan::{materialize_strided, plan_contraction, reduce_trace},
        Storage,
    },
    tensor::compute_contiguous_strides,
};
use std::sync::Arc;

fn u32_size(value: usize, operation: &'static str) -> Result<u32, AscendError> {
    u32::try_from(value).map_err(|_| AscendError::status(operation, -1))
}

#[allow(clippy::too_many_arguments)]
fn launch(
    runtime: &Arc<Runtime>,
    a: AscendStorage<f32>,
    b: AscendStorage<f32>,
    output: AscendStorage<f32>,
    argmax: AscendStorage<u32>,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    mode: u32,
) -> Result<(AscendStorage<f32>, AscendStorage<u32>), AscendError> {
    let dimensions = [
        u32_size(batch, "Ascend tropical batch exceeds u32")?,
        u32_size(m, "Ascend tropical M exceeds u32")?,
        u32_size(k, "Ascend tropical K exceeds u32")?,
        u32_size(n, "Ascend tropical N exceeds u32")?,
        mode,
    ];
    batch
        .checked_mul(m)
        .and_then(|value| value.checked_mul(n))
        .ok_or_else(|| AscendError::status("Ascend tropical output overflow", -1))?;
    let mut stream_state_unknown = false;
    let result = runtime.with_transaction(|runtime| {
        // A single logical worker avoids target-specific sparse AIV core numbering.
        // Multi-core tiling requires SoC-specific launch metadata.
        let workers = 1;
        let block_dim = workers * 2;
        let status = unsafe {
            sys::aclrtlaunch_omeinsum_tropical_gemm(
                block_dim,
                runtime.stream,
                a.as_mut_ptr().cast(),
                b.as_mut_ptr().cast(),
                output.as_mut_ptr().cast(),
                argmax.as_mut_ptr().cast(),
                dimensions[0],
                dimensions[1],
                dimensions[2],
                dimensions[3],
                workers,
                dimensions[4],
            )
        };
        if status != sys::ACL_SUCCESS as u32 {
            return Err(AscendError::status(
                "aclrtlaunch_omeinsum_tropical_gemm",
                status as i32,
            ));
        }
        stream_state_unknown = true;
        runtime.synchronize_locked()?;
        stream_state_unknown = false;
        Ok(())
    });
    if let Err(error) = result {
        if stream_state_unknown {
            std::mem::forget(a);
            std::mem::forget(b);
            std::mem::forget(output);
            std::mem::forget(argmax);
        }
        return Err(error);
    }
    Ok((output, argmax))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn contract<A: Algebra<Scalar = f32>>(
    runtime: &Arc<Runtime>,
    a: &AscendStorage<f32>,
    shape_a: &[usize],
    strides_a: &[usize],
    modes_a: &[i32],
    b: &AscendStorage<f32>,
    shape_b: &[usize],
    strides_b: &[usize],
    modes_b: &[i32],
    shape_c: &[usize],
    modes_c: &[i32],
    mode: u32,
) -> Result<(AscendStorage<f32>, AscendStorage<u32>), AscendError> {
    if !Arc::ptr_eq(runtime, a.runtime()) || !Arc::ptr_eq(runtime, b.runtime()) {
        return Err(AscendError::status("Ascend runtime/device mismatch", -1));
    }
    let normalized_a = normalize_repeated_labels(runtime, a, shape_a, strides_a, modes_a)?;
    let normalized_b = normalize_repeated_labels(runtime, b, shape_b, strides_b, modes_b)?;
    let (a, shape_a, strides_a, modes_a) =
        normalized_a
            .as_ref()
            .map_or((a, shape_a, strides_a, modes_a), |operand| {
                (
                    &operand.storage,
                    operand.shape.as_slice(),
                    operand.strides.as_slice(),
                    operand.modes.as_slice(),
                )
            });
    let (b, shape_b, strides_b, modes_b) =
        normalized_b
            .as_ref()
            .map_or((b, shape_b, strides_b, modes_b), |operand| {
                (
                    &operand.storage,
                    operand.shape.as_slice(),
                    operand.strides.as_slice(),
                    operand.modes.as_slice(),
                )
            });
    let output_len = validate_metadata(
        a.len(),
        shape_a,
        strides_a,
        modes_a,
        b.len(),
        shape_b,
        strides_b,
        modes_b,
        shape_c,
        modes_c,
    )?;
    if output_len == 0 {
        return Ok((
            AscendStorage::allocate(runtime.clone(), 0, false)?,
            AscendStorage::allocate(runtime.clone(), 0, false)?,
        ));
    }

    let probe = plan_contraction(modes_a, shape_a, modes_b, shape_b, modes_c);
    let identity_a: Vec<usize> = (0..shape_a.len()).collect();
    let identity_b: Vec<usize> = (0..shape_b.len()).collect();
    let a_contiguous = materialize_strided(&a.to_vec()?, shape_a, strides_a, &identity_a);
    let b_contiguous = materialize_strided(&b.to_vec()?, shape_b, strides_b, &identity_b);
    let (a_data, a_shape, a_modes) = if probe.left_trace.is_empty() {
        (a_contiguous, shape_a.to_vec(), modes_a.to_vec())
    } else {
        reduce_trace::<A>(&a_contiguous, shape_a, modes_a, &probe.left_trace)
    };
    let (b_data, b_shape, b_modes) = if probe.right_trace.is_empty() {
        (b_contiguous, shape_b.to_vec(), modes_b.to_vec())
    } else {
        reduce_trace::<A>(&b_contiguous, shape_b, modes_b, &probe.right_trace)
    };
    let plan = plan_contraction(&a_modes, &a_shape, &b_modes, &b_shape, modes_c);
    let zero_contract_extent = plan.contracted_modes.iter().any(|contracted| {
        let axis = a_modes.iter().position(|mode| mode == contracted).unwrap();
        a_shape[axis] == 0
    });
    if zero_contract_extent {
        return Ok((
            AscendStorage::upload(runtime.clone(), &vec![A::zero().to_scalar(); output_len])?,
            AscendStorage::upload(runtime.clone(), &vec![u32::MAX; output_len])?,
        ));
    }
    let a_canonical = materialize_strided(
        &a_data,
        &a_shape,
        &compute_contiguous_strides(&a_shape),
        &plan.a_permutation(&a_modes),
    );
    let b_canonical = materialize_strided(
        &b_data,
        &b_shape,
        &compute_contiguous_strides(&b_shape),
        &plan.b_permutation(&b_modes),
    );
    let output = AscendStorage::allocate(runtime.clone(), output_len, false)?;
    let argmax = AscendStorage::allocate(runtime.clone(), output_len, false)?;
    let result = launch(
        runtime,
        AscendStorage::upload(runtime.clone(), &a_canonical)?,
        AscendStorage::upload(runtime.clone(), &b_canonical)?,
        output,
        argmax,
        plan.batch_size.max(1),
        plan.left_size,
        plan.contract_size,
        plan.right_size,
        mode,
    )?;
    let Some(permutation) = plan.output_perm else {
        return Ok(result);
    };
    let canonical_shape: Vec<usize> = plan
        .left_modes
        .iter()
        .chain(&plan.right_modes)
        .chain(&plan.batch_modes)
        .map(|mode| {
            shape_c[modes_c
                .iter()
                .position(|candidate| candidate == mode)
                .unwrap()]
        })
        .collect();
    let strides = compute_contiguous_strides(&canonical_shape);
    let values = materialize_strided(
        &result.0.to_vec()?,
        &canonical_shape,
        &strides,
        &permutation,
    );
    let winners = materialize_strided(
        &result.1.to_vec()?,
        &canonical_shape,
        &strides,
        &permutation,
    );
    Ok((
        AscendStorage::upload(runtime.clone(), &values)?,
        AscendStorage::upload(runtime.clone(), &winners)?,
    ))
}
