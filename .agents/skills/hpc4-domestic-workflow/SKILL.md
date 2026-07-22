---
name: hpc4-domestic-workflow
description: Use when preparing, submitting, monitoring, or diagnosing a HKUST-GZ HPC4 domestic computing job, especially when the user mentions Ascend, NPU, Kunpeng, torch-npu, CANN, MindSpore, MindIE, vLLM-ascend, AIStudio, or uncertainty about whether to use Slurm or container resources.
---

# HPC4 Domestic Workflow

Single entrypoint for HKUST-GZ HPC4 domestic jobs. Keep the whole run recorded, route each stage to the right skill, and prevent accidental NVIDIA/CUDA or wrong-resource submissions.

## Session Record

Create a session before substantial work:

```sh
python .agents/skills/hpc4-domestic-workflow/scripts/new_session.py \
  --title "<short-title>" \
  --request "<user request>"
```

Use the printed directory as `SESSION`. Write stage outputs there:

```text
$SESSION/request.md
$SESSION/readiness.json
$SESSION/portability.json
$SESSION/submit-record.md
$SESSION/diagnose.md
$SESSION/logs/
```

Read `references/hpc4-domestic-routing.md` when routing between Slurm and AIStudio/container matters, or when version/platform facts may have changed.

## Pipeline

1. **Clarify target.** Determine whether the user wants Kunpeng CPU, Ascend/NPU, or an unknown domestic target. Ask only if the code/request cannot reveal it.
2. **Profile discovery.** If submission/probing is likely, invoke `hpc4-domestic-submit`'s profile setup gate early. First inspect existing profiles; only if none is active ask one compact setup question for username, remote staging directory, and whether to install SSH-key access.
3. **Readiness.** Invoke `hpc4-domestic-readiness` to check entrypoint, paths, resources, logs, dependency manifests, and smoke-test plan. Save `readiness.json`.
4. **Portability.** If NPU/Ascend is requested, or if code may have CUDA assumptions, invoke `hpc4-ascend-portability`. Save `portability.json`.
5. **Port if needed.** If portability reports CUDA/NVIDIA blockers and the user wants an Ascend run, invoke `hpc4-cuda-to-ascend-port` for strict CUDA-to-Ascend edits, then re-run portability.
6. **Route and submit.** Invoke `hpc4-domestic-submit` to verify whether the job should use Slurm or AIStudio/container, then assist with the actual SSH/sbatch workflow when Slurm is confirmed.
7. **Diagnose.** After submission or container launch, invoke `hpc4-domestic-diagnose` for pending, failure, timeout, import, device, and output-artifact problems.

## Gates

Stop before submission when:

- `readiness.json.status` is `blocked`.
- `portability.json.status` is `needs-porting` and `hpc4-cuda-to-ascend-port` has not cleared the blockers or the user has not approved a CPU-only run.
- The user requests NPU Slurm submission but probes do not show a valid Slurm NPU resource path.
- The active environment/version combination is unknown for an NPU job and no smoke test has passed.

Do not treat `sbatch` success as scientific or application success. Require logs and the user's expected success artifact.

## Handoff Prompts

- After readiness: "Readiness check complete. Next use `hpc4-ascend-portability` if this touches NPU/Ascend, otherwise use `hpc4-domestic-submit`."
- After portability with blockers: "Portability blockers remain. Use `hpc4-cuda-to-ascend-port` for strict CUDA-to-Ascend edits, then rerun portability."
- After portability without blockers: "Portability check complete. Use `hpc4-domestic-submit` to probe and assist the actual HPC4 route."
- After submit: "Submission route recorded. Use `hpc4-domestic-diagnose` for status, logs, failures, or missing outputs."

## Output

Report:

- Session path.
- Current stage and next skill.
- Blocking issues, if any.
- Confirmed route: Slurm, AIStudio/container, or unresolved.
- Commands run or commands proposed for confirmation.
