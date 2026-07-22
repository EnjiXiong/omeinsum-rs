use super::{
    contract::validate_operand, ffi::column_major_strides, runtime::Runtime,
    storage::AscendStorage, AscendError,
};
use crate::backend::Storage;
use std::sync::Arc;

pub(crate) struct NormalizedOperand {
    pub(crate) storage: AscendStorage<f32>,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    pub(crate) modes: Vec<i32>,
}

pub(crate) fn normalize_repeated_labels(
    runtime: &Arc<Runtime>,
    source: &AscendStorage<f32>,
    shape: &[usize],
    strides: &[usize],
    modes: &[i32],
) -> Result<Option<NormalizedOperand>, AscendError> {
    validate_operand(source.len(), shape, strides, modes)?;
    let has_repeated = modes
        .iter()
        .enumerate()
        .any(|(axis, mode)| modes[..axis].contains(mode));
    if !has_repeated {
        return Ok(None);
    }

    let mut unique_modes = Vec::new();
    let mut unique_shape = Vec::new();
    for (&mode, &extent) in modes.iter().zip(shape) {
        if let Some(axis) = unique_modes.iter().position(|candidate| *candidate == mode) {
            if unique_shape[axis] != extent {
                return Err(AscendError::status(
                    "Ascend repeated-label extent mismatch",
                    -1,
                ));
            }
        } else {
            unique_modes.push(mode);
            unique_shape.push(extent);
        }
    }

    let source_data = source.to_vec()?;
    let output_len = unique_shape.iter().try_fold(1usize, |size, &extent| {
        size.checked_mul(extent)
            .ok_or_else(|| AscendError::status("Ascend diagonal size overflow", -1))
    })?;
    let mut output = Vec::with_capacity(output_len);
    let mut coordinates = vec![0usize; unique_shape.len()];
    for _ in 0..output_len {
        let source_offset = modes
            .iter()
            .zip(strides)
            .map(|(mode, stride)| {
                let axis = unique_modes
                    .iter()
                    .position(|candidate| candidate == mode)
                    .expect("normalized mode must exist");
                coordinates[axis] * stride
            })
            .sum::<usize>();
        output.push(source_data[source_offset]);
        for axis in 0..coordinates.len() {
            coordinates[axis] += 1;
            if coordinates[axis] < unique_shape[axis] {
                break;
            }
            coordinates[axis] = 0;
        }
    }

    Ok(Some(NormalizedOperand {
        storage: AscendStorage::upload(runtime.clone(), &output)?,
        strides: column_major_strides(&unique_shape)?,
        shape: unique_shape,
        modes: unique_modes,
    }))
}
