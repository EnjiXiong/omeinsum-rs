---
name: hpc4-domestic-submit
description: Use when deciding whether a HKUST-GZ HPC4 domestic job should be submitted through Slurm or launched through AIStudio/container resources, including probing partitions, modules, NPU visibility, SSH access, sbatch submission, status, cancellation, and log checks.
---

# HPC4 Domestic Submit

Route and submit only after readiness and portability checks. This skill is self-contained: it carries the profile parser, guardrails, SSH, partition probe, `sbatch`, `squeue`, `sacct`, fetch, array wrapper, log-tail, cancel, and smoke-test mechanics needed for ordinary Slurm use.

## Inputs

- Session directory with `readiness.json` and, for NPU work, `portability.json`.
- SSH host/alias or explicit note that only web/AIStudio access is available.
- Target resource: Kunpeng CPU, Ascend/NPU, or unknown.
- Entrypoint and resource request.

Read `references/submission-routing.md` before probing or recommending a route.
Read `references/cluster-profiles.md` before creating or editing a profile.

## Profile Setup Gate

Before saying "we need a profile", detect whether one already exists:

```sh
python .agents/skills/hpc4-domestic-submit/scripts/cluster_profile_setup.py inspect
```

The profile is the TOML file under `profiles/`; do not describe TOML as a
second thing to configure after the profile.

If `needs_setup` is false, use the active profile and continue. If setup is
needed, ask a single compact question for:

- HPC4 username.
- Remote staging directory, defaulting to `/data/user/<username>/hpc4-workdir` unless the user wants another path.
- Whether to set up SSH-key access so the agent can run non-interactive probes and submissions.

After the username is known, create and activate the profile mechanically:

```sh
python .agents/skills/hpc4-domestic-submit/scripts/cluster_profile_setup.py create \
  --username <username> \
  --remote-dir </data/user/<username>/hpc4-workdir> \
  --activate
```

Then ensure local `~/.ssh/config` has the printed `Host` block. If the user
approves SSH-key access, have them run the printed `ssh-copy-id` command in an
interactive terminal, then verify:

```sh
ssh <alias> 'echo key-login-ok; hostname; whoami'
```

Only fall back to manual profile editing when the helper cannot represent the
cluster or the user supplies a nonstandard host/path policy.

## Probe First

On the login node, collect:

```sh
sinfo -o "%P %a %.10l %.6D %.6t %G"
scontrol show partition
module av
```

For a supposed NPU environment, collect:

```sh
npu-smi info
python -c 'import torch, torch_npu; print(torch.__version__); print(torch.npu.is_available())'
```

## Routing

- **Kunpeng CPU Slurm:** if `hpc` or another confirmed CPU partition matches the request, assist the user with SSH precheck, script upload, `sbatch`, status, and log inspection.
- **Ascend/NPU Slurm:** submit through Slurm only if probes or an administrator profile expose the exact partition/QOS/GRES/directive for NPU allocation.
- **AIStudio/container:** if NPU resources are exposed through AIStudio/container rather than Slurm, stop Slurm submission and produce a container launch checklist.
- **Unresolved:** if probes cannot be run and no trusted profile exists, report missing facts and do not invent directives.

## Primary Slurm Driver

Use `scripts/hpc4_harness_slurm.sh` for profile-driven work. It supports dry-run, precheck, partition probing, single jobs, array jobs, status, wait, fetch, classify, pending-cells, and smoke tests.

```sh
bash .agents/skills/hpc4-domestic-submit/scripts/hpc4_harness_slurm.sh --profile <profile.toml> --dry-run precheck
bash .agents/skills/hpc4-domestic-submit/scripts/hpc4_harness_slurm.sh --profile <profile.toml> --dry-run probe-partitions
bash .agents/skills/hpc4-domestic-submit/scripts/hpc4_harness_slurm.sh --profile <profile.toml> --dry-run submit --script <script> --partition <p> --time <t> --cpus <n>
bash .agents/skills/hpc4-domestic-submit/scripts/hpc4_harness_slurm.sh --profile <profile.toml> status <jobid>
bash .agents/skills/hpc4-domestic-submit/scripts/hpc4_harness_slurm.sh --profile <profile.toml> fetch <run>
bash .agents/skills/hpc4-domestic-submit/scripts/hpc4_harness_slurm.sh --profile <profile.toml> classify <run> <jobid>
```

Use `--dry-run` for preview. Remove `--dry-run` only after the user confirms the host, profile, remote repository path, resource request, and script.

For one-off script upload without a profile, `scripts/hpc4_slurm.py` remains available. Prefer the profile-driven driver for reusable workflows.

Before upload or submission:

- Capture local changes if the job comes from a repository; do not silently mutate or ship unrelated files.
- Prefer uploading the exact sbatch script first, not broad project sync.
- If code/data staging is needed, ask for explicit source and remote target.
- Record every executed command in `$SESSION/submit-record.md`.
- If no profile exists, use the Profile Setup Gate above. Avoid asking about TOML internals unless the helper is insufficient.

## submit-record.md

Record:

- Route: Slurm, AIStudio/container, or unresolved.
- Probe outputs or source of truth.
- Partition/resource directive, if Slurm.
- Image/container choice, if AIStudio/container.
- Final command submitted or proposed.
- Job ID or container/session identifier.

## Handoff

After a job is queued, running, failed, or launched in a container, next use `hpc4-domestic-diagnose`.
