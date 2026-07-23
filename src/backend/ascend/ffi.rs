#![allow(dead_code)]

use std::ffi::{c_char, c_void};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct Status {
    pub(crate) category: i32,
    pub(crate) cann_code: i32,
}

#[repr(C)]
pub(crate) struct Context {
    _private: [u8; 0],
}

#[repr(C)]
pub(crate) struct Buffer {
    _private: [u8; 0],
}

#[repr(C)]
pub(crate) struct Tensor {
    _private: [u8; 0],
}

#[repr(C)]
pub(crate) struct Op {
    _private: [u8; 0],
}

#[repr(C)]
pub(crate) struct Capture {
    _private: [u8; 0],
}

unsafe extern "C" {
    pub(crate) fn ome_ascend_last_error() -> *const c_char;

    pub(crate) fn ome_ascend_context_create(device_id: i32, out: *mut *mut Context) -> Status;
    pub(crate) fn ome_ascend_context_synchronize(context: *mut Context) -> Status;
    pub(crate) fn ome_ascend_context_soc_name(context: *const Context) -> *const c_char;
    pub(crate) fn ome_ascend_context_destroy(context: *mut Context);

    pub(crate) fn ome_ascend_buffer_alloc(
        context: *mut Context,
        bytes: u64,
        out: *mut *mut Buffer,
    ) -> Status;
    pub(crate) fn ome_ascend_buffer_copy_h2d(
        context: *mut Context,
        buffer: *mut Buffer,
        offset: u64,
        host: *const c_void,
        bytes: u64,
    ) -> Status;
    pub(crate) fn ome_ascend_buffer_copy_d2h(
        context: *mut Context,
        buffer: *const Buffer,
        offset: u64,
        host: *mut c_void,
        bytes: u64,
    ) -> Status;
    pub(crate) fn ome_ascend_buffer_destroy(buffer: *mut Buffer);

    pub(crate) fn ome_ascend_tensor_f32(
        buffer: *mut Buffer,
        byte_offset: u64,
        shape: *const i64,
        strides: *const i64,
        rank: u64,
        out: *mut *mut Tensor,
    ) -> Status;
    pub(crate) fn ome_ascend_tensor_destroy(tensor: *mut Tensor);

    pub(crate) fn ome_ascend_prepare_matmul(
        context: *mut Context,
        left: *const Tensor,
        right: *const Tensor,
        output: *mut Tensor,
        cube_math_type: i8,
        out: *mut *mut Op,
    ) -> Status;
    pub(crate) fn ome_ascend_prepare_add(
        context: *mut Context,
        left: *const Tensor,
        right: *const Tensor,
        output: *mut Tensor,
        out: *mut *mut Op,
    ) -> Status;
    pub(crate) fn ome_ascend_prepare_sub(
        context: *mut Context,
        left: *const Tensor,
        right: *const Tensor,
        output: *mut Tensor,
        out: *mut *mut Op,
    ) -> Status;
    pub(crate) fn ome_ascend_prepare_permute(
        context: *mut Context,
        input: *const Tensor,
        axes: *const i64,
        rank: u64,
        output: *mut Tensor,
        out: *mut *mut Op,
    ) -> Status;
    pub(crate) fn ome_ascend_op_workspace_bytes(op: *const Op) -> u64;
    pub(crate) fn ome_ascend_op_run(
        context: *mut Context,
        op: *mut Op,
        workspace: *mut Buffer,
    ) -> Status;
    pub(crate) fn ome_ascend_op_destroy(op: *mut Op);

    pub(crate) fn ome_ascend_capture_supported(
        context: *mut Context,
        supported: *mut i32,
    ) -> Status;
    pub(crate) fn ome_ascend_capture_begin(context: *mut Context) -> Status;
    pub(crate) fn ome_ascend_capture_end(context: *mut Context, out: *mut *mut Capture) -> Status;
    pub(crate) fn ome_ascend_capture_run(context: *mut Context, capture: *mut Capture) -> Status;
    pub(crate) fn ome_ascend_capture_destroy(capture: *mut Capture);

    #[cfg(debug_assertions)]
    pub(crate) fn ome_ascend_debug_reset_counts();
    #[cfg(debug_assertions)]
    pub(crate) fn ome_ascend_debug_descriptor_creations() -> u64;
    #[cfg(debug_assertions)]
    pub(crate) fn ome_ascend_debug_workspace_queries() -> u64;
    #[cfg(debug_assertions)]
    pub(crate) fn ome_ascend_debug_op_runs() -> u64;
}
