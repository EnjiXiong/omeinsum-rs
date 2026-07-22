# Strict CUDA to Ascend Porting Rules

This skill performs narrow migration edits only. Preserve the original program structure, algorithm, names, data flow, CLI, config, and comments unless a comment explicitly names a CUDA/NVIDIA runtime element that must change for correctness.

## Allowed Mechanical Edits

- Add `import torch_npu` immediately after `import torch` when PyTorch NPU APIs are introduced.
- Replace `torch.cuda` with `torch.npu`.
- Replace `.cuda()` and `.cuda(...)` with `.npu()` and `.npu(...)`.
- Replace device string literals: `cuda`, `cuda:0`, etc. with `npu`, `npu:0`, etc.
- Replace `CUDA_VISIBLE_DEVICES` with `ASCEND_VISIBLE_DEVICES`.
- Replace `nvidia-smi` probes with `npu-smi info`.
- Replace CANN environment sourcing only when the target path is known; otherwise insert a review note rather than inventing a path.

## Manual Review Required

Do not auto-rewrite these without explicit evidence:

- NCCL to HCCL distributed launch semantics.
- Custom CUDA extensions, C++/CUDA kernels, CuPy, Numba CUDA.
- Triton kernels unless Triton-Ascend support is confirmed.
- Docker device mapping.
- Performance flags, fused kernels, precision modes, and graph capture.

## Completion Criteria

- Static portability scan no longer reports CUDA markers.
- The diff is explainable as CUDA/NVIDIA runtime replacement only.
- A smoke test is available for `torch_npu` import, NPU visibility, and one tiny tensor operation.
