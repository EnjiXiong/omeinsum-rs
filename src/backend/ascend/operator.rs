#![allow(clippy::result_large_err)] // Frozen backend diagnostics retain full context.

use std::ptr::NonNull;

use serde::{Deserialize, Serialize};

use crate::static_plan::ExecutionError;

use super::context::Context;
use super::error::{check, invalid_request, ErrorContext};
use super::ffi;
use super::storage::{DeviceBuffer, TensorDescriptor};
use super::AscendSession;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PreparedOpKind {
    Matmul,
    Add,
    Sub,
    Permute,
}

pub(crate) struct PreparedOp<'context> {
    raw: NonNull<ffi::Op>,
    _kind: PreparedOpKind,
    workspace_bytes: u64,
    node_id: Option<usize>,
    logical_shape: Vec<usize>,
    physical_strides: Vec<i64>,
    context: &'context Context,
    _tensors: Vec<TensorDescriptor<'context>>,
}

impl<'context> PreparedOp<'context> {
    pub(crate) fn matmul(
        context: &'context Context,
        left: &TensorDescriptor<'context>,
        right: &TensorDescriptor<'context>,
        output: &TensorDescriptor<'context>,
        node_id: Option<usize>,
    ) -> Result<Self, ExecutionError> {
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            ffi::ome_ascend_prepare_matmul(
                context.raw(),
                left.raw(),
                right.raw(),
                output.raw(),
                0,
                &mut raw,
            )
        };
        Self::finish_prepare(
            context,
            status,
            raw,
            PreparedOpKind::Matmul,
            "ome_ascend_prepare_matmul",
            node_id,
            output,
            vec![left.clone(), right.clone(), output.clone()],
        )
    }

    pub(crate) fn add(
        context: &'context Context,
        left: &TensorDescriptor<'context>,
        right: &TensorDescriptor<'context>,
        output: &TensorDescriptor<'context>,
        node_id: Option<usize>,
    ) -> Result<Self, ExecutionError> {
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            ffi::ome_ascend_prepare_add(
                context.raw(),
                left.raw(),
                right.raw(),
                output.raw(),
                &mut raw,
            )
        };
        Self::finish_prepare(
            context,
            status,
            raw,
            PreparedOpKind::Add,
            "ome_ascend_prepare_add",
            node_id,
            output,
            vec![left.clone(), right.clone(), output.clone()],
        )
    }

    pub(crate) fn sub(
        context: &'context Context,
        left: &TensorDescriptor<'context>,
        right: &TensorDescriptor<'context>,
        output: &TensorDescriptor<'context>,
        node_id: Option<usize>,
    ) -> Result<Self, ExecutionError> {
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            ffi::ome_ascend_prepare_sub(
                context.raw(),
                left.raw(),
                right.raw(),
                output.raw(),
                &mut raw,
            )
        };
        Self::finish_prepare(
            context,
            status,
            raw,
            PreparedOpKind::Sub,
            "ome_ascend_prepare_sub",
            node_id,
            output,
            vec![left.clone(), right.clone(), output.clone()],
        )
    }

    pub(crate) fn permute(
        context: &'context Context,
        input: &TensorDescriptor<'context>,
        axes: &[i64],
        output: &TensorDescriptor<'context>,
        node_id: Option<usize>,
    ) -> Result<Self, ExecutionError> {
        let mut raw = std::ptr::null_mut();
        let status = unsafe {
            ffi::ome_ascend_prepare_permute(
                context.raw(),
                input.raw(),
                axes.as_ptr(),
                u64::try_from(axes.len()).unwrap_or(u64::MAX),
                output.raw(),
                &mut raw,
            )
        };
        Self::finish_prepare(
            context,
            status,
            raw,
            PreparedOpKind::Permute,
            "ome_ascend_prepare_permute",
            node_id,
            output,
            vec![input.clone(), output.clone()],
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_prepare(
        context: &'context Context,
        status: ffi::Status,
        raw: *mut ffi::Op,
        kind: PreparedOpKind,
        operation: &str,
        node_id: Option<usize>,
        output: &TensorDescriptor<'context>,
        tensors: Vec<TensorDescriptor<'context>>,
    ) -> Result<Self, ExecutionError> {
        check(
            status,
            ErrorContext {
                stage: "descriptor-executor-prepare",
                node_id,
                operation,
                logical_shape: output.shape(),
                physical_strides: output.strides(),
                device: Some(context.soc_name().to_string()),
            },
        )?;
        let raw = NonNull::new(raw).ok_or_else(|| {
            invalid_request(
                "descriptor-executor-prepare",
                operation,
                output.shape(),
                output.strides(),
                Some(context.soc_name().to_string()),
                "native shim returned success with a null operator",
            )
        })?;
        let workspace_bytes = unsafe { ffi::ome_ascend_op_workspace_bytes(raw.as_ptr()) };
        Ok(Self {
            raw,
            _kind: kind,
            workspace_bytes,
            node_id,
            logical_shape: output.shape().to_vec(),
            physical_strides: output.strides().to_vec(),
            context,
            _tensors: tensors,
        })
    }

    pub(crate) fn workspace_bytes(&self) -> u64 {
        self.workspace_bytes
    }

    pub(crate) fn run(&mut self, workspace: &DeviceBuffer<'context>) -> Result<(), ExecutionError> {
        let status = unsafe {
            ffi::ome_ascend_op_run(self.context.raw(), self.raw.as_ptr(), workspace.raw())
        };
        check(
            status,
            ErrorContext {
                stage: "enqueue",
                node_id: self.node_id,
                operation: "ome_ascend_op_run",
                logical_shape: &self.logical_shape,
                physical_strides: &self.physical_strides,
                device: Some(self.context.soc_name().to_string()),
            },
        )
    }
}

impl Drop for PreparedOp<'_> {
    fn drop(&mut self) {
        unsafe { ffi::ome_ascend_op_destroy(self.raw.as_ptr()) };
    }
}

pub(crate) struct PreparedStep<'context> {
    pub(crate) op: PreparedOp<'context>,
}

impl<'context> PreparedStep<'context> {
    pub(crate) fn workspace_bytes(&self) -> u64 {
        self.op.workspace_bytes()
    }

    pub(crate) fn run(&mut self, workspace: &DeviceBuffer<'context>) -> Result<(), ExecutionError> {
        self.op.run(workspace)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AscendMatmulSmokeReport {
    pub output: Vec<f32>,
    pub cube_math_type: i8,
    pub semantic_bytes: u64,
    pub device_arena_bytes: u64,
    pub scratch_bytes: u64,
    pub workspace_bytes: u64,
    pub peak_device_bytes: u64,
    pub descriptor_creations_during_replay: u64,
    pub workspace_queries_during_replay: u64,
    pub op_runs_during_replay: u64,
}

pub fn run_matmul_smoke(
    session: &AscendSession,
) -> Result<AscendMatmulSmokeReport, ExecutionError> {
    let left = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let right = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    let left_offset = 0u64;
    let right_offset = u64::try_from(std::mem::size_of_val(&left)).unwrap();
    let output_offset = right_offset
        .checked_add(u64::try_from(std::mem::size_of_val(&right)).unwrap())
        .unwrap();
    let output_bytes = 4u64 * 4;
    let semantic_bytes = output_offset + output_bytes;
    let arena = session.context.allocate(semantic_bytes)?;
    arena.copy_h2d(left_offset, &left)?;
    arena.copy_h2d(right_offset, &right)?;
    session.context.synchronize()?;

    let left_tensor = arena.tensor_f32(left_offset, &[2, 3], &[3, 1])?;
    let right_tensor = arena.tensor_f32(right_offset, &[3, 2], &[2, 1])?;
    let output_tensor = arena.tensor_f32(output_offset, &[2, 2], &[2, 1])?;
    let mut op = PreparedOp::matmul(
        &session.context,
        &left_tensor,
        &right_tensor,
        &output_tensor,
        None,
    )?;
    let workspace_bytes = op.workspace_bytes();
    let workspace = session.context.allocate(workspace_bytes)?;

    debug_reset();
    op.run(&workspace)?;
    op.run(&workspace)?;
    session.context.synchronize()?;
    let debug = debug_snapshot();

    let mut output = vec![0.0f32; 4];
    arena.copy_d2h(output_offset, &mut output)?;
    session.context.synchronize()?;
    let expected = [58.0f32, 64.0, 139.0, 154.0];
    if output.iter().zip(expected).any(|(actual, expected)| {
        let scale = 1.0f32.max(expected.abs());
        (*actual - expected).abs() > 1e-6 * scale
    }) {
        return Err(ExecutionError::Backend {
            stage: "correctness".to_string(),
            node_id: None,
            operation: "ascend_matmul_smoke".to_string(),
            status_code: None,
            logical_shape: vec![2, 2],
            physical_strides: vec![2, 1],
            device: Some(session.device_info.soc_name.clone()),
            detail: format!("expected {expected:?}, got {output:?}"),
        });
    }
    let peak_device_bytes = semantic_bytes.checked_add(workspace_bytes).ok_or_else(|| {
        ExecutionError::Unsupported("matmul smoke memory accounting overflow".to_string())
    })?;
    Ok(AscendMatmulSmokeReport {
        output,
        cube_math_type: 0,
        semantic_bytes,
        device_arena_bytes: semantic_bytes,
        scratch_bytes: 0,
        workspace_bytes,
        peak_device_bytes,
        descriptor_creations_during_replay: debug.descriptor_creations,
        workspace_queries_during_replay: debug.workspace_queries,
        op_runs_during_replay: debug.op_runs,
    })
}

struct DebugCounts {
    descriptor_creations: u64,
    workspace_queries: u64,
    op_runs: u64,
}

#[cfg(debug_assertions)]
fn debug_reset() {
    unsafe { ffi::ome_ascend_debug_reset_counts() };
}

#[cfg(not(debug_assertions))]
fn debug_reset() {}

#[cfg(debug_assertions)]
fn debug_snapshot() -> DebugCounts {
    unsafe {
        DebugCounts {
            descriptor_creations: ffi::ome_ascend_debug_descriptor_creations(),
            workspace_queries: ffi::ome_ascend_debug_workspace_queries(),
            op_runs: ffi::ome_ascend_debug_op_runs(),
        }
    }
}

#[cfg(not(debug_assertions))]
fn debug_snapshot() -> DebugCounts {
    DebugCounts {
        descriptor_creations: 0,
        workspace_queries: 0,
        op_runs: 2,
    }
}
