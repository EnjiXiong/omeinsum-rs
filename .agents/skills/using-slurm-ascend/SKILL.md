---
name: using-slurm-ascend
description: Use when preparing, checking, porting, submitting, monitoring, or diagnosing Ascend/NPU or Kunpeng CPU jobs for the HKUST-GZ HPC4 domestic cluster.
---

# Using Slurm Ascend

Public entrypoint for the HKUST-GZ HPC4 Ascend/Kunpeng workflow. Users should
be able to start with this skill plus their HPC4 username and intended remote
staging directory; the component skills handle the detailed checks.

## First Step

Create a workflow session:

```sh
python .agents/skills/hpc4-domestic-workflow/scripts/new_session.py \
  --title "<short-title>" \
  --request "<user request>"
```

Then run the profile setup gate before asking for more cluster details:

```sh
python .agents/skills/hpc4-domestic-submit/scripts/cluster_profile_setup.py inspect
```

If no active profile exists, ask one compact question for:

- HPC4 username.
- Remote staging directory, defaulting to `/data/user/<username>/hpc4-workdir`.
- Whether the user wants SSH-key setup for unattended probes and submissions.

Create and activate the profile:

```sh
python .agents/skills/hpc4-domestic-submit/scripts/cluster_profile_setup.py create \
  --username <username> \
  --remote-dir </data/user/<username>/hpc4-workdir> \
  --activate
```

If the user agrees to SSH-key setup, configure the printed `Host` block and
have the user run the printed `ssh-copy-id` command locally. Never ask for a
password in chat.

## Route

After profile setup, follow `hpc4-domestic-workflow`:

1. Run `hpc4-domestic-readiness`.
2. Run `hpc4-ascend-portability` for Ascend/NPU or CUDA-sensitive jobs.
3. Run `hpc4-cuda-to-ascend-port` only when CUDA/NVIDIA blockers remain.
4. Run `hpc4-domestic-submit` for Slurm or AIStudio/container route probing.
5. Run `hpc4-domestic-diagnose` for queued, failed, timed-out, or missing-output jobs.

## Submission Rule

Do not submit NPU jobs until a live probe or trusted profile confirms the exact
partition/QOS/GRES request and the target CANN/torch_npu environment passes a
tiny NPU tensor smoke test.
