use std::ffi::{c_char, c_void};

pub type AclError = i32;
pub type AclnnStatus = i32;
pub type AclrtContext = *mut c_void;
pub type AclrtStream = *mut c_void;
pub type AclTensor = c_void;
pub type AclIntArray = c_void;
pub type AclOpExecutor = c_void;

pub const ACL_SUCCESS: i32 = 0;
pub const ACL_ERROR_REPEAT_INITIALIZE: i32 = 100002;
pub const ACL_MEM_MALLOC_HUGE_FIRST: i32 = 0;
pub const ACL_MEMCPY_HOST_TO_DEVICE: i32 = 1;
pub const ACL_MEMCPY_DEVICE_TO_HOST: i32 = 2;
pub const ACL_MEMCPY_DEVICE_TO_DEVICE: i32 = 3;
pub const ACL_FLOAT: i32 = 0;
#[allow(dead_code)]
pub const ACL_UINT32: i32 = 8;
pub const ACL_FORMAT_ND: i32 = 2;

extern "C" {
    pub fn aclInit(config: *const c_char) -> AclError;
    pub fn aclrtSetDevice(device: i32) -> AclError;
    pub fn aclrtCreateContext(context: *mut AclrtContext, device: i32) -> AclError;
    pub fn aclrtSetCurrentContext(context: AclrtContext) -> AclError;
    pub fn aclrtDestroyContext(context: AclrtContext) -> AclError;
    pub fn aclrtCreateStream(stream: *mut AclrtStream) -> AclError;
    pub fn aclrtDestroyStream(stream: AclrtStream) -> AclError;
    pub fn aclrtSynchronizeStream(stream: AclrtStream) -> AclError;
    pub fn aclrtMalloc(ptr: *mut *mut c_void, size: usize, policy: i32) -> AclError;
    pub fn aclrtFree(ptr: *mut c_void) -> AclError;
    pub fn aclrtMemcpy(
        dst: *mut c_void,
        dst_max: usize,
        src: *const c_void,
        count: usize,
        kind: i32,
    ) -> AclError;
    pub fn aclrtMemset(dst: *mut c_void, max: usize, value: i32, count: usize) -> AclError;

    #[cfg(feature = "ascend-tropical")]
    pub fn aclrtlaunch_omeinsum_tropical_gemm(
        block_dim: u32,
        stream: AclrtStream,
        a: *mut c_void,
        b: *mut c_void,
        c: *mut c_void,
        argmax: *mut c_void,
        batch: u32,
        m: u32,
        k: u32,
        n: u32,
        workers: u32,
        mode: u32,
    ) -> u32;

    pub fn aclCreateTensor(
        view_dims: *const i64,
        view_dims_num: u64,
        data_type: i32,
        stride: *const i64,
        offset: i64,
        format: i32,
        storage_dims: *const i64,
        storage_dims_num: u64,
        data: *mut c_void,
    ) -> *mut AclTensor;
    pub fn aclDestroyTensor(tensor: *mut AclTensor) -> AclError;
    pub fn aclCreateIntArray(values: *const i64, len: u64) -> *mut AclIntArray;
    pub fn aclDestroyIntArray(array: *mut AclIntArray) -> AclError;
    pub fn aclnnMatmulGetWorkspaceSize(
        a: *const AclTensor,
        b: *const AclTensor,
        output: *mut AclTensor,
        cube_math_type: i8,
        workspace_size: *mut u64,
        executor: *mut *mut AclOpExecutor,
    ) -> AclnnStatus;
    pub fn aclnnMatmul(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut AclOpExecutor,
        stream: AclrtStream,
    ) -> AclnnStatus;
    pub fn aclnnBatchMatMulGetWorkspaceSize(
        a: *const AclTensor,
        b: *const AclTensor,
        output: *mut AclTensor,
        cube_math_type: i8,
        workspace_size: *mut u64,
        executor: *mut *mut AclOpExecutor,
    ) -> AclnnStatus;
    pub fn aclnnBatchMatMul(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut AclOpExecutor,
        stream: AclrtStream,
    ) -> AclnnStatus;
    pub fn aclnnPermuteGetWorkspaceSize(
        input: *const AclTensor,
        dims: *const AclIntArray,
        output: *mut AclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut AclOpExecutor,
    ) -> AclnnStatus;
    pub fn aclnnPermute(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut AclOpExecutor,
        stream: AclrtStream,
    ) -> AclnnStatus;
    pub fn aclnnReduceSumGetWorkspaceSize(
        input: *const AclTensor,
        dims: *const AclIntArray,
        keep_dims: bool,
        dtype: i32,
        output: *mut AclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut AclOpExecutor,
    ) -> AclnnStatus;
    pub fn aclnnReduceSum(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut AclOpExecutor,
        stream: AclrtStream,
    ) -> AclnnStatus;
}
