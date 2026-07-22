---
name: hpc4-ascend-portability
description: Use when checking whether code, dependencies, or runtime commands can run on Ascend/Kunpeng instead of NVIDIA CUDA, especially for torch-npu, CANN, HCCL, MindSpore, MindIE, vLLM-ascend, Triton-Ascend, or CUDA-to-NPU migration.
---

# HPC4 Ascend Portability

Detect CUDA/NVIDIA assumptions and verify the declared Ascend/Kunpeng software stack before NPU work proceeds.

## Inputs

- Code/job directory.
- Environment source: module list, Conda env, container image, or AIStudio image.
- Session directory from `hpc4-domestic-workflow`, if available.

Read `references/ascend-portability-checklist.md` before interpreting version or framework findings.

## Static Scan

Run:

```sh
python .agents/skills/hpc4-ascend-portability/scripts/check_ascend_portability.py <job-dir>
```

Save the JSON to `$SESSION/portability.json` when a session exists.

## Version Discipline

Do not hard-code old constraints. Verify the active combination:

- Python version and architecture.
- CANN toolkit/kernels.
- Driver and firmware, usually via `npu-smi info`.
- `torch` and `torch_npu` versions when using PyTorch.
- HCCL/MindSpore/MindIE/vLLM-ascend/MindSpeed/Triton-Ascend versions when relevant.

If the user cites an old rule such as "torch only supports 2.1.0", treat it as historical evidence, not current truth. Re-check current HKUST-GZ docs and the active image/module.

## Blockers

Block NPU submission until resolved:

- CUDA device strings or `torch.cuda` are still used.
- `CUDA_VISIBLE_DEVICES`, `nvidia-smi`, `CUDA_HOME`, NCCL-only launch, CuPy, Numba CUDA, or custom CUDA extensions are required.
- `torch_npu` is expected but cannot be imported in the target environment.
- CANN/driver/firmware/framework versions are not known and no smoke test has passed.
- Distributed training assumes NCCL rather than HCCL.

## Acceptable Outcomes

- `portable-or-not-applicable`: no CUDA blockers; CPU-only Kunpeng work may proceed without Ascend markers.
- `needs-review`: possible issue, but user can justify it.
- `needs-porting`: CUDA/NVIDIA-specific blockers remain.
- `blocked`: target directory or environment facts are absent.

## Handoff

If status is not blocking, return to `hpc4-domestic-workflow` and next use `hpc4-domestic-submit`. If blockers remain, report exact files/lines and next use `hpc4-cuda-to-ascend-port` unless the user chooses CPU-only fallback.
