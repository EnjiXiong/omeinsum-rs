#include "kernel_operator.h"

using namespace AscendC;

// Canonical buffers are contiguous column-major batches:
// A[batch, m, k], B[batch, k, n], C[batch, m, n].
// mode: 0 = max-plus, 1 = min-plus, 2 = max-mul.
extern "C" __global__ __aicore__ void omeinsum_tropical_gemm(
    GM_ADDR a,
    GM_ADDR b,
    GM_ADDR c,
    GM_ADDR argmax,
    uint32_t batch,
    uint32_t m,
    uint32_t k,
    uint32_t n,
    uint32_t workers,
    uint32_t mode) {
    GlobalTensor<float> a_global;
    GlobalTensor<float> b_global;
    GlobalTensor<float> c_global;
    GlobalTensor<uint32_t> argmax_global;
    a_global.SetGlobalBuffer((__gm__ float *)a);
    b_global.SetGlobalBuffer((__gm__ float *)b);
    c_global.SetGlobalBuffer((__gm__ float *)c);
    argmax_global.SetGlobalBuffer((__gm__ uint32_t *)argmax);

    const uint64_t block = GetBlockIdx() / GetTaskRatio();
    if (block >= workers) {
        return;
    }
    const uint64_t matrix_size = static_cast<uint64_t>(m) * n;
    const uint64_t output_size = static_cast<uint64_t>(batch) * matrix_size;
    for (uint64_t output = block; output < output_size; output += workers) {
        const uint64_t batch_index = output / matrix_size;
        const uint64_t matrix_index = output - batch_index * matrix_size;
        const uint64_t row = matrix_index % m;
        const uint64_t column = matrix_index / m;
        const uint64_t a_base = batch_index * m * k + row;
        const uint64_t b_base = batch_index * k * n + column * k;

        float best = mode == 1 ? 3.402823466e+38F
                               : (mode == 2 ? 0.0F : -3.402823466e+38F);
        uint32_t winner = 0;
        for (uint32_t contracted = 0; contracted < k; ++contracted) {
            const float left = a_global.GetValue(a_base + contracted * m);
            const float right = b_global.GetValue(b_base + contracted);
            const float candidate = mode == 2 ? left * right : left + right;
            const bool replace = mode == 1 ? !(best <= candidate) : !(best >= candidate);
            if (replace) {
                best = candidate;
                winner = contracted;
            }
        }
        c_global.SetValue(output, best);
        argmax_global.SetValue(output, winner);
    }
}
