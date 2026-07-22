# HPC4 Job Readiness Checklist

Use this checklist before submission or container launch.

## Required Inputs

- User intent: CPU/Kunpeng, Ascend/NPU, or unknown.
- Entrypoint script or command.
- Working directory and data paths.
- Output path, log path, and expected success artifact.
- Runtime environment: module, Conda, container image, or manually sourced stack.
- Resource request: walltime, CPU cores, memory, node count, and NPU count when relevant.
- Minimal smoke test that runs quickly and exercises imports/device discovery.

## Blocking Findings

- No runnable entrypoint.
- Input data path is absent or remote-only with no staging plan.
- Output directory is not declared.
- No environment activation or module/container declaration.
- Resource type is unclear.
- NPU requested but no portability check has been run.
- Slurm submission requested before partition/resource probe.

## Review Findings

- Missing `#SBATCH --output` or `#SBATCH --error`.
- Missing dependency manifest.
- Hard-coded absolute paths from a different machine.
- No small test case before a long run.
- No checkpoint/restart plan for long training.

## Report Contract

Write `readiness.json` in the session directory with:

```json
{
  "status": "submit-ready | needs-review | blocked",
  "blocking": [],
  "warnings": [],
  "facts": {}
}
```
