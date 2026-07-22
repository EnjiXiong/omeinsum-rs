---
name: hpc4-domestic-readiness
description: Use when checking whether a HKUST-GZ HPC4 domestic job is complete enough to submit or launch, including Slurm scripts, container commands, paths, logs, dependency manifests, resource requests, and smoke tests.
---

# HPC4 Domestic Readiness

Check operational completeness before any Slurm submission or AIStudio/container launch. This skill does not decide Ascend code portability; use `hpc4-ascend-portability` for that.

## Inputs

- Job directory and entrypoint script/command.
- Intended target: Kunpeng CPU, Ascend/NPU, or unknown.
- Session directory from `hpc4-domestic-workflow`, if available.

Read `references/readiness-checklist.md` for the full checklist and report contract.

## Static Scan

Run the helper when a local job directory exists:

```sh
python .agents/skills/hpc4-domestic-readiness/scripts/check_job_readiness.py <job-dir> \
  --script <optional-entry-script>
```

Save the JSON to `$SESSION/readiness.json` when a session exists. Treat script output as a first pass; inspect important findings manually.

## Required Checks

- Entrypoint exists and is runnable in the intended environment.
- Input data and model paths have a local, remote, or mounted location.
- Output directory, stdout, stderr, and application logs are declared.
- Environment activation is explicit: `module load`, Conda, CANN `set_env.sh`, or container image.
- Resource intent is explicit: walltime, CPU cores, memory, node count, and NPU count if relevant.
- A short smoke test exists for imports, device discovery, and minimal computation.

## Status

- `submit-ready`: no blocking findings; warnings are minor and understood.
- `needs-review`: no hard blocker, but user or downstream skill must ratify warnings.
- `blocked`: missing entrypoint, data, environment, resource intent, or output plan.

## Handoff

- If target is Ascend/NPU or code may contain CUDA assumptions, next use `hpc4-ascend-portability`; if it finds blockers, use `hpc4-cuda-to-ascend-port`.
- If target is confirmed CPU-only Kunpeng and status is not `blocked`, next use `hpc4-domestic-submit`.
- Return to `hpc4-domestic-workflow` after writing or summarizing `readiness.json`.
