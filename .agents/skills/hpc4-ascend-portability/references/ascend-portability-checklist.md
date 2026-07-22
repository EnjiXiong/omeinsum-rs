# Ascend and Kunpeng Portability Checklist

## Environment Stack

Verify the active stack instead of relying on memory:

- CPU architecture: `aarch64` for Kunpeng-oriented wheels/images unless the environment says otherwise.
- Driver and firmware: `npu-smi info` in an NPU environment.
- CANN toolkit and kernels: source `ascend-toolkit/set_env.sh` or image-provided equivalent.
- Framework pairings: Python, `torch`, `torch_npu`, CANN, and driver/firmware must be mutually compatible.
- Distributed jobs: HCCL configuration replaces NCCL assumptions.

Old experience such as "torch only supports 2.1.0" is not a rule. Current HKUST-GZ docs have listed newer images, including PyTorch 2.8.0 with Python 3.11. Always inspect the active module/image.

## Code Patterns to Block or Review

Block until ported or justified:

- `torch.cuda`, `tensor.cuda()`, `device="cuda"`.
- `CUDA_VISIBLE_DEVICES`, `CUDA_HOME`.
- `nvidia-smi`.
- NCCL-only distributed launch/config.
- CuPy, Numba CUDA, custom CUDA extensions.
- Triton kernels that assume NVIDIA semantics without Triton-Ascend support.

Acceptable or positive markers:

- `import torch_npu`, `torch.npu`, `transfer_to_npu`.
- `ASCEND_VISIBLE_DEVICES`.
- `npu-smi info`.
- CANN `set_env.sh`.
- MindSpore, MindIE, vLLM-ascend, MindSpeed, Triton-Ascend when version-paired.

## Minimal Smoke Tests

PyTorch NPU:

```python
import torch
import torch_npu

print("torch", torch.__version__)
print("npu available", torch.npu.is_available())
print("device count", torch.npu.device_count())
x = torch.ones(4, device="npu")
print((x + 1).cpu())
```

Runtime:

```sh
npu-smi info
python -c 'import torch, torch_npu; print(torch.__version__); print(torch.npu.is_available())'
```

For CPU-only Kunpeng work, the correct result is often "Ascend not applicable"; still check architecture-sensitive compiled dependencies.
