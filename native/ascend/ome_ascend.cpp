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
        const aclError code = aclrtMemcpyAsync(
            destination, static_cast<size_t>(buffer->bytes - offset), host,
            static_cast<size_t>(bytes), ACL_MEMCPY_HOST_TO_DEVICE,
            context->stream);
        return code == ACL_SUCCESS
                   ? ok()
                   : runtime_error("aclrtMemcpyAsync(H2D)", code);
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
        const aclError code = aclrtMemcpyAsync(
            host, static_cast<size_t>(bytes), source,
            static_cast<size_t>(bytes), ACL_MEMCPY_DEVICE_TO_HOST,
            context->stream);
        return code == ACL_SUCCESS
                   ? ok()
                   : runtime_error("aclrtMemcpyAsync(D2H)", code);
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
