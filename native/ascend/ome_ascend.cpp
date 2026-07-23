#include "ome_ascend.h"

#include <acl/acl.h>
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_matmul.h>
#include <aclnnop/aclnn_permute.h>
#include <aclnnop/aclnn_sub.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <string>

namespace {
thread_local std::string last_error;

#ifdef OME_ASCEND_DEBUG_DIAGNOSTICS
thread_local uint64_t descriptor_creations = 0;
thread_local uint64_t workspace_queries = 0;
thread_local uint64_t op_runs = 0;
#endif

enum StatusCategory : int32_t {
    kSuccess = 0,
    kInvalidArgument = 1,
    kRuntime = 2,
    kAclnn = 3,
    kUnsupported = 4,
    kInternal = 5,
};

ome_ascend_status_t ok() {
    last_error.clear();
    return {kSuccess, 0};
}

ome_ascend_status_t error(
    StatusCategory category, int32_t code, const char *operation,
    const std::string &detail = {}) {
    last_error = operation;
    if (!detail.empty()) {
        last_error += ": ";
        last_error += detail;
    }
    return {category, code};
}

template <typename Function>
ome_ascend_status_t guarded(const char *operation, Function &&function) noexcept {
    try {
        last_error.clear();
        return function();
    } catch (const std::exception &exception) {
        return error(kInternal, -1, operation, exception.what());
    } catch (...) {
        return error(kInternal, -1, operation, "unknown C++ exception");
    }
}

bool checked_add(uint64_t left, uint64_t right, uint64_t *out) {
    if (left > std::numeric_limits<uint64_t>::max() - right) {
        return false;
    }
    *out = left + right;
    return true;
}

bool checked_mul(uint64_t left, uint64_t right, uint64_t *out) {
    if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) {
        return false;
    }
    *out = left * right;
    return true;
}
}

struct ome_ascend_context {
    int32_t device_id = -1;
    bool initialized = false;
    bool device_set = false;
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    std::string soc_name;
};

struct ome_ascend_buffer {
    void *device = nullptr;
    uint64_t bytes = 0;
};

struct ome_ascend_tensor {
    aclTensor *tensor = nullptr;
};

enum class OpKind {
    kMatmul,
    kAdd,
    kSub,
    kPermute,
};

struct ome_ascend_op {
    OpKind kind = OpKind::kMatmul;
    aclOpExecutor *executor = nullptr;
    aclScalar *scalar = nullptr;
    aclIntArray *int_array = nullptr;
    float alpha = 1.0F;
    uint64_t workspace_bytes = 0;
};

struct ome_ascend_capture {
#ifdef OME_ASCEND_ENABLE_CAPTURE
    aclmdlRI model = nullptr;
#endif
};

namespace {
void cleanup_context(ome_ascend_context *context) noexcept {
    if (context == nullptr) {
        return;
    }
    if (context->stream != nullptr) {
        (void)aclrtSynchronizeStream(context->stream);
        (void)aclrtDestroyStream(context->stream);
        context->stream = nullptr;
    }
    if (context->context != nullptr) {
        (void)aclrtDestroyContext(context->context);
        context->context = nullptr;
    }
    if (context->device_set) {
        (void)aclrtResetDevice(context->device_id);
        context->device_set = false;
    }
    if (context->initialized) {
        (void)aclFinalize();
        context->initialized = false;
    }
}

ome_ascend_status_t runtime_error(const char *operation, aclError code) {
    return error(kRuntime, static_cast<int32_t>(code), operation);
}
}

extern "C" const char *ome_ascend_last_error(void) {
    return last_error.c_str();
}

extern "C" ome_ascend_status_t ome_ascend_context_create(
    int32_t device_id, ome_ascend_context_t **out) {
    return guarded("ome_ascend_context_create", [&]() {
        if (out == nullptr) {
            return error(kInvalidArgument, -1, "ome_ascend_context_create",
                         "out is null");
        }
        *out = nullptr;
        auto context = std::make_unique<ome_ascend_context>();
        context->device_id = device_id;

        aclError code = aclInit(nullptr);
        if (code != ACL_SUCCESS) {
            return runtime_error("aclInit", code);
        }
        context->initialized = true;

        code = aclrtSetDevice(device_id);
        if (code != ACL_SUCCESS) {
            auto status = runtime_error("aclrtSetDevice", code);
            cleanup_context(context.get());
            return status;
        }
        context->device_set = true;

        code = aclrtCreateContext(&context->context, device_id);
        if (code != ACL_SUCCESS) {
            auto status = runtime_error("aclrtCreateContext", code);
            cleanup_context(context.get());
            return status;
        }
        code = aclrtSetCurrentContext(context->context);
        if (code != ACL_SUCCESS) {
            auto status = runtime_error("aclrtSetCurrentContext", code);
            cleanup_context(context.get());
            return status;
        }
        code = aclrtCreateStream(&context->stream);
        if (code != ACL_SUCCESS) {
            auto status = runtime_error("aclrtCreateStream", code);
            cleanup_context(context.get());
            return status;
        }
        const char *soc_name = aclrtGetSocName();
        context->soc_name = soc_name == nullptr ? "unknown" : soc_name;
        *out = context.release();
        return ok();
    });
}

extern "C" ome_ascend_status_t ome_ascend_context_synchronize(
    ome_ascend_context_t *context) {
    return guarded("ome_ascend_context_synchronize", [&]() {
        if (context == nullptr || context->stream == nullptr) {
            return error(kInvalidArgument, -1,
                         "ome_ascend_context_synchronize",
                         "context or stream is null");
        }
        const aclError code = aclrtSynchronizeStream(context->stream);
        return code == ACL_SUCCESS
                   ? ok()
                   : runtime_error("aclrtSynchronizeStream", code);
    });
}

extern "C" const char *ome_ascend_context_soc_name(
    const ome_ascend_context_t *context) {
    return context == nullptr ? "" : context->soc_name.c_str();
}

extern "C" void ome_ascend_context_destroy(
    ome_ascend_context_t *context) {
    try {
        cleanup_context(context);
        delete context;
    } catch (...) {
        // Destruction is best-effort and must never unwind across the C ABI.
    }
}

extern "C" ome_ascend_status_t ome_ascend_buffer_alloc(
    ome_ascend_context_t *context, uint64_t bytes,
    ome_ascend_buffer_t **out) {
    return guarded("ome_ascend_buffer_alloc", [&]() {
        if (context == nullptr || out == nullptr) {
            return error(kInvalidArgument, -1, "ome_ascend_buffer_alloc",
                         "context or out is null");
        }
        *out = nullptr;
        auto buffer = std::make_unique<ome_ascend_buffer>();
        buffer->bytes = bytes;
        if (bytes > 0) {
            if (bytes > static_cast<uint64_t>(
                            std::numeric_limits<size_t>::max())) {
                return error(kInvalidArgument, -1, "ome_ascend_buffer_alloc",
                             "byte count exceeds size_t");
            }
            const aclError code =
                aclrtMalloc(&buffer->device, static_cast<size_t>(bytes),
                            ACL_MEM_MALLOC_HUGE_FIRST);
            if (code != ACL_SUCCESS) {
                return runtime_error("aclrtMalloc", code);
            }
        }
        *out = buffer.release();
        return ok();
    });
}

extern "C" ome_ascend_status_t ome_ascend_buffer_copy_h2d(
    ome_ascend_context_t *context, ome_ascend_buffer_t *buffer,
    uint64_t offset, const void *host, uint64_t bytes) {
    return guarded("ome_ascend_buffer_copy_h2d", [&]() {
        if (context == nullptr || buffer == nullptr ||
            (bytes > 0 && host == nullptr)) {
            return error(kInvalidArgument, -1,
                         "ome_ascend_buffer_copy_h2d",
                         "context, buffer, or host is null");
        }
        uint64_t end = 0;
        if (!checked_add(offset, bytes, &end) || end > buffer->bytes) {
            return error(kInvalidArgument, -1,
                         "ome_ascend_buffer_copy_h2d",
                         "copy exceeds device buffer");
        }
        if (bytes == 0) {
            return ok();
        }
        auto *destination = static_cast<uint8_t *>(buffer->device) + offset;
        const aclError code = aclrtMemcpy(
            destination, static_cast<size_t>(buffer->bytes - offset), host,
            static_cast<size_t>(bytes), ACL_MEMCPY_HOST_TO_DEVICE);
        return code == ACL_SUCCESS
                   ? ok()
                   : runtime_error("aclrtMemcpy(H2D)", code);
    });
}

extern "C" ome_ascend_status_t ome_ascend_buffer_copy_d2h(
    ome_ascend_context_t *context, const ome_ascend_buffer_t *buffer,
    uint64_t offset, void *host, uint64_t bytes) {
    return guarded("ome_ascend_buffer_copy_d2h", [&]() {
        if (context == nullptr || buffer == nullptr ||
            (bytes > 0 && host == nullptr)) {
            return error(kInvalidArgument, -1,
                         "ome_ascend_buffer_copy_d2h",
                         "context, buffer, or host is null");
        }
        uint64_t end = 0;
        if (!checked_add(offset, bytes, &end) || end > buffer->bytes) {
            return error(kInvalidArgument, -1,
                         "ome_ascend_buffer_copy_d2h",
                         "copy exceeds device buffer");
        }
        if (bytes == 0) {
            return ok();
        }
        const auto *source =
            static_cast<const uint8_t *>(buffer->device) + offset;
        const aclError code = aclrtMemcpy(
            host, static_cast<size_t>(bytes), source,
            static_cast<size_t>(bytes), ACL_MEMCPY_DEVICE_TO_HOST);
        return code == ACL_SUCCESS
                   ? ok()
                   : runtime_error("aclrtMemcpy(D2H)", code);
    });
}

extern "C" void ome_ascend_buffer_destroy(
    ome_ascend_buffer_t *buffer) {
    try {
        if (buffer != nullptr && buffer->device != nullptr) {
            (void)aclrtFree(buffer->device);
        }
        delete buffer;
    } catch (...) {
    }
}

extern "C" ome_ascend_status_t ome_ascend_tensor_f32(
    ome_ascend_buffer_t *buffer, uint64_t byte_offset,
    const int64_t *shape, const int64_t *strides, uint64_t rank,
    ome_ascend_tensor_t **out) {
    return guarded("ome_ascend_tensor_f32", [&]() {
        if (buffer == nullptr || out == nullptr ||
            (rank > 0 && (shape == nullptr || strides == nullptr))) {
            return error(kInvalidArgument, -1, "ome_ascend_tensor_f32",
                         "invalid null argument");
        }
        *out = nullptr;
        uint64_t elements = 1;
        for (uint64_t axis = 0; axis < rank; ++axis) {
            if (shape[axis] <= 0 || strides[axis] < 0) {
                return error(kInvalidArgument, -1,
                             "ome_ascend_tensor_f32",
                             "shape must be positive and strides nonnegative");
            }
            uint64_t extent = 0;
            if (!checked_mul(static_cast<uint64_t>(shape[axis] - 1),
                             static_cast<uint64_t>(strides[axis]), &extent) ||
                !checked_add(elements, extent, &elements)) {
                return error(kInvalidArgument, -1,
                             "ome_ascend_tensor_f32",
                             "tensor span overflow");
            }
        }
        uint64_t tensor_bytes = 0;
        uint64_t end = 0;
        if (!checked_mul(elements, sizeof(float), &tensor_bytes) ||
            !checked_add(byte_offset, tensor_bytes, &end) ||
            end > buffer->bytes) {
            return error(kInvalidArgument, -1, "ome_ascend_tensor_f32",
                         "tensor view exceeds device buffer");
        }
        auto descriptor = std::make_unique<ome_ascend_tensor>();
        auto *data = static_cast<uint8_t *>(buffer->device) + byte_offset;
        descriptor->tensor = aclCreateTensor(
            shape, rank, ACL_FLOAT, strides, 0, ACL_FORMAT_ND, shape, rank,
            data);
        if (descriptor->tensor == nullptr) {
            return error(kAclnn, -1, "aclCreateTensor",
                         "returned a null descriptor");
        }
#ifdef OME_ASCEND_DEBUG_DIAGNOSTICS
        ++descriptor_creations;
#endif
        *out = descriptor.release();
        return ok();
    });
}

extern "C" void ome_ascend_tensor_destroy(
    ome_ascend_tensor_t *tensor) {
    try {
        if (tensor != nullptr && tensor->tensor != nullptr) {
            (void)aclDestroyTensor(tensor->tensor);
        }
        delete tensor;
    } catch (...) {
    }
}

namespace {
void cleanup_op(ome_ascend_op *op) noexcept {
    if (op == nullptr) {
        return;
    }
    if (op->executor != nullptr) {
        (void)aclDestroyAclOpExecutor(op->executor);
        op->executor = nullptr;
    }
    if (op->scalar != nullptr) {
        (void)aclDestroyScalar(op->scalar);
        op->scalar = nullptr;
    }
    if (op->int_array != nullptr) {
        (void)aclDestroyIntArray(op->int_array);
        op->int_array = nullptr;
    }
}

ome_ascend_status_t finalize_prepared_op(
    std::unique_ptr<ome_ascend_op> op, aclnnStatus query_status,
    const char *query_operation, ome_ascend_op_t **out) {
#ifdef OME_ASCEND_DEBUG_DIAGNOSTICS
    ++workspace_queries;
#endif
    if (query_status != 0) {
        const auto status =
            error(kAclnn, query_status, query_operation);
        cleanup_op(op.get());
        return status;
    }
    const aclnnStatus repeatable_status =
        aclSetAclOpExecutorRepeatable(op->executor);
    if (repeatable_status != 0) {
        const auto status = error(kAclnn, repeatable_status,
                                  "aclSetAclOpExecutorRepeatable");
        cleanup_op(op.get());
        return status;
    }
    *out = op.release();
    return ok();
}

bool valid_prepare_arguments(
    ome_ascend_context_t *context, const ome_ascend_tensor_t *left,
    const ome_ascend_tensor_t *right, ome_ascend_tensor_t *output,
    ome_ascend_op_t **out) {
    return context != nullptr && left != nullptr && left->tensor != nullptr &&
           right != nullptr && right->tensor != nullptr && output != nullptr &&
           output->tensor != nullptr && out != nullptr;
}
}

extern "C" ome_ascend_status_t ome_ascend_prepare_matmul(
    ome_ascend_context_t *context, const ome_ascend_tensor_t *left,
    const ome_ascend_tensor_t *right, ome_ascend_tensor_t *output,
    int8_t cube_math_type, ome_ascend_op_t **out) {
    return guarded("ome_ascend_prepare_matmul", [&]() {
        if (!valid_prepare_arguments(context, left, right, output, out)) {
            return error(kInvalidArgument, -1,
                         "ome_ascend_prepare_matmul",
                         "invalid null argument");
        }
        if (cube_math_type != 0) {
            return error(kInvalidArgument, -1,
                         "ome_ascend_prepare_matmul",
                         "only cubeMathType=0 KEEP_DTYPE is supported");
        }
        *out = nullptr;
        auto op = std::make_unique<ome_ascend_op>();
        op->kind = OpKind::kMatmul;
        const aclnnStatus status = aclnnMatmulGetWorkspaceSize(
            left->tensor, right->tensor, output->tensor, cube_math_type,
            &op->workspace_bytes, &op->executor);
        return finalize_prepared_op(std::move(op), status,
                                    "aclnnMatmulGetWorkspaceSize", out);
    });
}

extern "C" ome_ascend_status_t ome_ascend_prepare_add(
    ome_ascend_context_t *context, const ome_ascend_tensor_t *left,
    const ome_ascend_tensor_t *right, ome_ascend_tensor_t *output,
    ome_ascend_op_t **out) {
    return guarded("ome_ascend_prepare_add", [&]() {
        if (!valid_prepare_arguments(context, left, right, output, out)) {
            return error(kInvalidArgument, -1, "ome_ascend_prepare_add",
                         "invalid null argument");
        }
        *out = nullptr;
        auto op = std::make_unique<ome_ascend_op>();
        op->kind = OpKind::kAdd;
        op->scalar = aclCreateScalar(&op->alpha, ACL_FLOAT);
        if (op->scalar == nullptr) {
            return error(kAclnn, -1, "aclCreateScalar",
                         "returned a null scalar");
        }
        const aclnnStatus status = aclnnAddGetWorkspaceSize(
            left->tensor, right->tensor, op->scalar, output->tensor,
            &op->workspace_bytes, &op->executor);
        return finalize_prepared_op(std::move(op), status,
                                    "aclnnAddGetWorkspaceSize", out);
    });
}

extern "C" ome_ascend_status_t ome_ascend_prepare_sub(
    ome_ascend_context_t *context, const ome_ascend_tensor_t *left,
    const ome_ascend_tensor_t *right, ome_ascend_tensor_t *output,
    ome_ascend_op_t **out) {
    return guarded("ome_ascend_prepare_sub", [&]() {
        if (!valid_prepare_arguments(context, left, right, output, out)) {
            return error(kInvalidArgument, -1, "ome_ascend_prepare_sub",
                         "invalid null argument");
        }
        *out = nullptr;
        auto op = std::make_unique<ome_ascend_op>();
        op->kind = OpKind::kSub;
        op->scalar = aclCreateScalar(&op->alpha, ACL_FLOAT);
        if (op->scalar == nullptr) {
            return error(kAclnn, -1, "aclCreateScalar",
                         "returned a null scalar");
        }
        const aclnnStatus status = aclnnSubGetWorkspaceSize(
            left->tensor, right->tensor, op->scalar, output->tensor,
            &op->workspace_bytes, &op->executor);
        return finalize_prepared_op(std::move(op), status,
                                    "aclnnSubGetWorkspaceSize", out);
    });
}

extern "C" ome_ascend_status_t ome_ascend_prepare_permute(
    ome_ascend_context_t *context, const ome_ascend_tensor_t *input,
    const int64_t *axes, uint64_t rank, ome_ascend_tensor_t *output,
    ome_ascend_op_t **out) {
    return guarded("ome_ascend_prepare_permute", [&]() {
        if (context == nullptr || input == nullptr ||
            input->tensor == nullptr || (rank > 0 && axes == nullptr) ||
            output == nullptr || output->tensor == nullptr || out == nullptr) {
            return error(kInvalidArgument, -1,
                         "ome_ascend_prepare_permute",
                         "invalid null argument");
        }
        *out = nullptr;
        auto op = std::make_unique<ome_ascend_op>();
        op->kind = OpKind::kPermute;
        op->int_array = aclCreateIntArray(axes, rank);
        if (op->int_array == nullptr) {
            return error(kAclnn, -1, "aclCreateIntArray",
                         "returned a null array");
        }
        const aclnnStatus status = aclnnPermuteGetWorkspaceSize(
            input->tensor, op->int_array, output->tensor,
            &op->workspace_bytes, &op->executor);
        return finalize_prepared_op(std::move(op), status,
                                    "aclnnPermuteGetWorkspaceSize", out);
    });
}

extern "C" uint64_t ome_ascend_op_workspace_bytes(
    const ome_ascend_op_t *op) {
    return op == nullptr ? 0 : op->workspace_bytes;
}

extern "C" ome_ascend_status_t ome_ascend_op_run(
    ome_ascend_context_t *context, ome_ascend_op_t *op,
    ome_ascend_buffer_t *workspace) {
    return guarded("ome_ascend_op_run", [&]() {
        if (context == nullptr || op == nullptr || op->executor == nullptr ||
            workspace == nullptr || workspace->bytes < op->workspace_bytes) {
            return error(kInvalidArgument, -1, "ome_ascend_op_run",
                         "invalid context, op, or workspace");
        }
        void *workspace_address =
            op->workspace_bytes == 0 ? nullptr : workspace->device;
        aclnnStatus status = 0;
        switch (op->kind) {
            case OpKind::kMatmul:
                status = aclnnMatmul(workspace_address, op->workspace_bytes,
                                     op->executor, context->stream);
                break;
            case OpKind::kAdd:
                status = aclnnAdd(workspace_address, op->workspace_bytes,
                                  op->executor, context->stream);
                break;
            case OpKind::kSub:
                status = aclnnSub(workspace_address, op->workspace_bytes,
                                  op->executor, context->stream);
                break;
            case OpKind::kPermute:
                status = aclnnPermute(workspace_address, op->workspace_bytes,
                                      op->executor, context->stream);
                break;
        }
#ifdef OME_ASCEND_DEBUG_DIAGNOSTICS
        ++op_runs;
#endif
        return status == 0
                   ? ok()
                   : error(kAclnn, status, "ACLNN second-stage execution");
    });
}

extern "C" void ome_ascend_op_destroy(ome_ascend_op_t *op) {
    try {
        cleanup_op(op);
        delete op;
    } catch (...) {
    }
}

extern "C" ome_ascend_status_t ome_ascend_capture_supported(
    ome_ascend_context_t *context, int32_t *supported) {
    return guarded("ome_ascend_capture_supported", [&]() {
        if (context == nullptr || supported == nullptr) {
            return error(kInvalidArgument, -1,
                         "ome_ascend_capture_supported",
                         "context or supported is null");
        }
#ifdef OME_ASCEND_ENABLE_CAPTURE
        aclmdlRICaptureStatus status = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
        aclmdlRI model = nullptr;
        const aclError code =
            aclmdlRICaptureGetInfo(context->stream, &status, &model);
        if (code != ACL_SUCCESS) {
            *supported = 0;
            return error(kUnsupported, static_cast<int32_t>(code),
                         "aclmdlRICaptureGetInfo",
                         "live runtime/device probe rejected model-RI capture");
        }
        *supported = 1;
#else
        *supported = 0;
#endif
        return ok();
    });
}

extern "C" ome_ascend_status_t ome_ascend_capture_begin(
    ome_ascend_context_t *context) {
    return guarded("ome_ascend_capture_begin", [&]() {
        if (context == nullptr) {
            return error(kInvalidArgument, -1,
                         "ome_ascend_capture_begin", "context is null");
        }
#ifdef OME_ASCEND_ENABLE_CAPTURE
        const aclError code = aclmdlRICaptureBegin(
            context->stream, ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL);
        return code == ACL_SUCCESS
                   ? ok()
                   : runtime_error("aclmdlRICaptureBegin", code);
#else
        return error(kUnsupported, -1, "ome_ascend_capture_begin",
                     "shim was built without OME_ASCEND_ENABLE_CAPTURE=1");
#endif
    });
}

extern "C" ome_ascend_status_t ome_ascend_capture_end(
    ome_ascend_context_t *context, ome_ascend_capture_t **out) {
    return guarded("ome_ascend_capture_end", [&]() {
        if (context == nullptr || out == nullptr) {
            return error(kInvalidArgument, -1,
                         "ome_ascend_capture_end",
                         "context or out is null");
        }
        *out = nullptr;
#ifdef OME_ASCEND_ENABLE_CAPTURE
        auto capture = std::make_unique<ome_ascend_capture>();
        const aclError code =
            aclmdlRICaptureEnd(context->stream, &capture->model);
        if (code != ACL_SUCCESS) {
            return runtime_error("aclmdlRICaptureEnd", code);
        }
        if (capture->model == nullptr) {
            return error(kRuntime, -1, "aclmdlRICaptureEnd",
                         "returned a null model RI");
        }
        *out = capture.release();
        return ok();
#else
        return error(kUnsupported, -1, "ome_ascend_capture_end",
                     "shim was built without OME_ASCEND_ENABLE_CAPTURE=1");
#endif
    });
}

extern "C" ome_ascend_status_t ome_ascend_capture_run(
    ome_ascend_context_t *context, ome_ascend_capture_t *capture) {
    return guarded("ome_ascend_capture_run", [&]() {
        if (context == nullptr || capture == nullptr) {
            return error(kInvalidArgument, -1,
                         "ome_ascend_capture_run",
                         "context or capture is null");
        }
#ifdef OME_ASCEND_ENABLE_CAPTURE
        if (capture->model == nullptr) {
            return error(kInvalidArgument, -1,
                         "ome_ascend_capture_run",
                         "capture model is null");
        }
        const aclError code =
            aclmdlRIExecuteAsync(capture->model, context->stream);
        return code == ACL_SUCCESS
                   ? ok()
                   : runtime_error("aclmdlRIExecuteAsync", code);
#else
        return error(kUnsupported, -1, "ome_ascend_capture_run",
                     "shim was built without OME_ASCEND_ENABLE_CAPTURE=1");
#endif
    });
}

extern "C" void ome_ascend_capture_destroy(
    ome_ascend_capture_t *capture) {
    try {
#ifdef OME_ASCEND_ENABLE_CAPTURE
        if (capture != nullptr && capture->model != nullptr) {
            (void)aclmdlRIDestroy(capture->model);
            capture->model = nullptr;
        }
#endif
        delete capture;
    } catch (...) {
    }
}

#ifdef OME_ASCEND_DEBUG_DIAGNOSTICS
extern "C" void ome_ascend_debug_reset_counts(void) {
    descriptor_creations = 0;
    workspace_queries = 0;
    op_runs = 0;
}

extern "C" uint64_t ome_ascend_debug_descriptor_creations(void) {
    return descriptor_creations;
}

extern "C" uint64_t ome_ascend_debug_workspace_queries(void) {
    return workspace_queries;
}

extern "C" uint64_t ome_ascend_debug_op_runs(void) {
    return op_runs;
}
#endif
