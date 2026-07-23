#ifndef OME_ASCEND_H
#define OME_ASCEND_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ome_ascend_context ome_ascend_context_t;
typedef struct ome_ascend_buffer ome_ascend_buffer_t;
typedef struct ome_ascend_tensor ome_ascend_tensor_t;
typedef struct ome_ascend_op ome_ascend_op_t;
typedef struct ome_ascend_capture ome_ascend_capture_t;

typedef struct {
    int32_t category;
    int32_t cann_code;
} ome_ascend_status_t;

const char *ome_ascend_last_error(void);

ome_ascend_status_t ome_ascend_context_create(
    int32_t device_id, ome_ascend_context_t **out);
ome_ascend_status_t ome_ascend_context_synchronize(
    ome_ascend_context_t *context);
const char *ome_ascend_context_soc_name(
    const ome_ascend_context_t *context);
void ome_ascend_context_destroy(ome_ascend_context_t *context);

ome_ascend_status_t ome_ascend_buffer_alloc(
    ome_ascend_context_t *context, uint64_t bytes,
    ome_ascend_buffer_t **out);
ome_ascend_status_t ome_ascend_buffer_copy_h2d(
    ome_ascend_context_t *context, ome_ascend_buffer_t *buffer,
    uint64_t offset, const void *host, uint64_t bytes);
ome_ascend_status_t ome_ascend_buffer_copy_d2h(
    ome_ascend_context_t *context, const ome_ascend_buffer_t *buffer,
    uint64_t offset, void *host, uint64_t bytes);
void ome_ascend_buffer_destroy(ome_ascend_buffer_t *buffer);

ome_ascend_status_t ome_ascend_tensor_f32(
    ome_ascend_buffer_t *buffer, uint64_t byte_offset,
    const int64_t *shape, const int64_t *strides, uint64_t rank,
    ome_ascend_tensor_t **out);
void ome_ascend_tensor_destroy(ome_ascend_tensor_t *tensor);

ome_ascend_status_t ome_ascend_prepare_matmul(
    ome_ascend_context_t *context, const ome_ascend_tensor_t *left,
    const ome_ascend_tensor_t *right, ome_ascend_tensor_t *output,
    int8_t cube_math_type, ome_ascend_op_t **out);
ome_ascend_status_t ome_ascend_prepare_add(
    ome_ascend_context_t *context, const ome_ascend_tensor_t *left,
    const ome_ascend_tensor_t *right, ome_ascend_tensor_t *output,
    ome_ascend_op_t **out);
ome_ascend_status_t ome_ascend_prepare_sub(
    ome_ascend_context_t *context, const ome_ascend_tensor_t *left,
    const ome_ascend_tensor_t *right, ome_ascend_tensor_t *output,
    ome_ascend_op_t **out);
ome_ascend_status_t ome_ascend_prepare_permute(
    ome_ascend_context_t *context, const ome_ascend_tensor_t *input,
    const int64_t *axes, uint64_t rank, ome_ascend_tensor_t *output,
    ome_ascend_op_t **out);
uint64_t ome_ascend_op_workspace_bytes(const ome_ascend_op_t *op);
ome_ascend_status_t ome_ascend_op_run(
    ome_ascend_context_t *context, ome_ascend_op_t *op,
    ome_ascend_buffer_t *workspace);
void ome_ascend_op_destroy(ome_ascend_op_t *op);

ome_ascend_status_t ome_ascend_capture_supported(
    ome_ascend_context_t *context, int32_t *supported);
ome_ascend_status_t ome_ascend_capture_begin(
    ome_ascend_context_t *context);
ome_ascend_status_t ome_ascend_capture_end(
    ome_ascend_context_t *context, ome_ascend_capture_t **out);
ome_ascend_status_t ome_ascend_capture_run(
    ome_ascend_context_t *context, ome_ascend_capture_t *capture);
void ome_ascend_capture_destroy(ome_ascend_capture_t *capture);

#ifdef __cplusplus
}
#endif

#endif
