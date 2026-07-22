---
name: hpc4-cuda-to-ascend-port
description: Use when code must be changed from NVIDIA CUDA assumptions to Ascend/Kunpeng equivalents while preserving the original task, algorithm, data flow, CLI, and behavior as closely as possible.
---

# HPC4 CUDA to Ascend Port

Strictly port CUDA/NVIDIA runtime references to Ascend/NPU equivalents. Do not redesign the program, optimize, change the task, or introduce model/framework-specific behavior beyond the requested port.

## Inputs

- Code/job directory.
- `portability.json` from `hpc4-ascend-portability`, if available.
- Target framework: PyTorch NPU, MindSpore, MindIE, vLLM-ascend, Triton-Ascend, or unknown.

Read `references/porting-rules.md` before editing.

## Workflow

1. Run or inspect `hpc4-ascend-portability` findings.
2. Make only mechanical CUDA/NVIDIA-to-Ascend edits that are justified by the original code.
3. Use the helper in dry-run mode first:

   ```sh
   python .agents/skills/hpc4-cuda-to-ascend-port/scripts/cuda_to_ascend_port.py <job-dir> --dry-run
   ```

4. Review the diff. If it preserves behavior, apply:

   ```sh
   python .agents/skills/hpc4-cuda-to-ascend-port/scripts/cuda_to_ascend_port.py <job-dir> --apply
   ```

5. Re-run `hpc4-ascend-portability`.
6. Add or confirm a minimal smoke test for the target Ascend environment.

## Boundaries

Allowed: device API/string/env/probe replacement, `torch_npu` import insertion, narrow launch/env edits.

Not allowed without explicit user approval: algorithm changes, batching changes, precision changes, model changes, data path changes, distributed topology changes, performance tuning, or replacing the framework.

## Output

Report:

- Files changed.
- Exact CUDA markers removed.
- Manual-review items that remain.
- Smoke test command.
- Next skill: `hpc4-domestic-readiness` if operational details changed, otherwise `hpc4-domestic-submit`.
