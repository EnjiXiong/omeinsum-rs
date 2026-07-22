# HPC4 Submission Routing

## Pre-submit Probes

Run these on the login node when SSH is available:

```sh
hostname
sinfo -o "%P %a %.10l %.6D %.6t %G"
scontrol show partition
module av 2>&1 | head -200
```

If probing an NPU-capable environment, also run in that environment:

```sh
npu-smi info
python -c 'import sys, platform; print(sys.version); print(platform.machine())'
python -c 'import torch, torch_npu; print(torch.__version__); print(torch.npu.is_available())'
```

## Slurm Path

Submit through Slurm only after a real Slurm resource path is confirmed. Required facts:

- SSH alias/host and remote working directory.
- Partition/QOS/GRES/resource directive.
- Walltime, CPU, memory, node count.
- Whether the job is single-run or array.
- Clean shipping plan: git or explicit rsync.

Use the profile-driven bundled helper:

```sh
bash .agents/skills/hpc4-domestic-submit/scripts/hpc4_harness_slurm.sh --profile <profile.toml> --dry-run precheck
bash .agents/skills/hpc4-domestic-submit/scripts/hpc4_harness_slurm.sh --profile <profile.toml> --dry-run probe-partitions
bash .agents/skills/hpc4-domestic-submit/scripts/hpc4_harness_slurm.sh --profile <profile.toml> --dry-run submit --script <script> --partition <p> --time <t> --cpus <n>
```

The driver runs for real unless `--dry-run` is passed. Always preview with `--dry-run` before submission.

## AIStudio/Container Path

If NPU resources are only available through AIStudio/container:

- Do not fabricate sbatch directives.
- Tell the user that the current public HPC4 docs route NPU development through AIStudio/container.
- Prepare a container launch checklist: image, NPU count, mounted code/data, model path, smoke test, and logs.
- If the user provides an internal Slurm NPU example, re-run resource probing and then reconsider Slurm.
