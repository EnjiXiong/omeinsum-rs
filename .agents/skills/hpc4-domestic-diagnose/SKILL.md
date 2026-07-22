---
name: hpc4-domestic-diagnose
description: Use when a HKUST-GZ HPC4 domestic Slurm or AIStudio/container job is pending, failed, slow, missing outputs, cannot see NPU devices, has torch-npu/CANN/HCCL/MindIE/vLLM-ascend errors, or needs post-submit status and log interpretation.
---

# HPC4 Domestic Diagnose

Classify post-submit and post-launch problems using scheduler state, logs, environment probes, and Ascend/Kunpeng-specific symptoms.

## Inputs

- Session directory from `hpc4-domestic-workflow`, if available.
- `submit-record.md`, job ID, container/session ID, or log paths.
- Expected success artifact from `readiness.json`.

Read `references/failure-taxonomy.md` before classifying failures.

## Evidence Order

1. Scheduler/container state: `squeue`, `sacct`, AIStudio status, or container logs.
2. Stdout/stderr and application logs.
3. Environment probes: modules, Python, `npu-smi info`, CANN env, `torch_npu` import.
4. Application-level success artifact.

Do not report success from scheduler completion alone. A completed job with no expected output is an application failure.

## Slurm Checks

For Slurm jobs, use scheduler commands directly or through `hpc4-domestic-submit/scripts/hpc4_harness_slurm.sh`:

```sh
squeue -j <jobid>
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

Then inspect logs for the taxonomy categories.

## Ascend Checks

Use the smallest available probes:

```sh
npu-smi info
python -c 'import torch, torch_npu; print(torch.__version__); print(torch.npu.is_available())'
env | grep -E 'ASCEND|CANN|HCCL|NPU'
```

Common categories:

- Device not allocated or not mapped.
- CANN/ATB/HCCL shared library missing.
- Wrong Python/architecture/wheel combination.
- CUDA/NCCL code path still active.
- Unsupported operator or custom CUDA extension.
- HBM/system memory OOM.
- Walltime exceeded.
- Data path missing in remote/container context.

## diagnose.md

Write or report:

```text
state:
classification:
evidence:
blocking cause:
recommended next action:
rerun safe: yes/no
```

If rerun is unsafe, state what must change first: resource request, environment, code port, data mount, or output/checkpoint plan.
