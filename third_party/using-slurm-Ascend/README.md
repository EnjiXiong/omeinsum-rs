# using-slurm-Ascend

`using-slurm-Ascend` is a Codex skill suite for preparing, porting, submitting,
and diagnosing Ascend/NPU jobs on the HKUST-GZ HPC4 domestic cluster. It is
adapted for the cluster's Kunpeng CPU and Ascend NPU Slurm routes, including
profile setup, readiness checks, CUDA-to-Ascend migration checks, Slurm probing,
runtime smoke tests, and failure diagnosis.

This is a community-maintained workflow aid, not an official HKUST-GZ HPC4
service.

## Install

Manual install:

```sh
git clone https://github.com/EnjiXiong/using-slurm-Ascend.git
cp -R using-slurm-Ascend/skills/* ~/.codex/skills/
```

macOS host notes:

- Use Python 3.11 or newer. The profile parser uses the standard-library
  `tomllib` module, which was added in Python 3.11.
- Ensure `bash`, `ssh`, `scp`, and `python3` are on `PATH`.
- `rsync` is only required for helper flows that fetch results or run the
  bundled smoke-test transport.
- `ssh-copy-id` is convenient but may need to be installed separately; the
  workflow also prints a manual `authorized_keys` fallback.

On macOS with Homebrew, the usual dependency setup is:

```sh
brew install python@3.12 rsync ssh-copy-id
```

See [macOS Compatibility Notes](docs/macos-compatibility.md) for the
source-backed command-line compatibility audit.

Agent install:

Ask your coding agent to clone this repository and copy `skills/*` into your
Codex skills directory. A typical prompt is:

```text
Install the Codex skills from https://github.com/EnjiXiong/using-slurm-Ascend
by copying the repository's skills/* directories into ~/.codex/skills/.
```

## Basic Use

Start with the entrypoint skill and provide the minimum cluster setup facts:

```text
/using-slurm-ascend
username: <your-hpc4-username>
remote upload path: /data/user/<your-hpc4-username>/hpc4-workdir
```

The workflow first checks whether an active profile already exists. If not, it
creates one from your username and remote upload path. It can also help set up
SSH-key access so the agent can run non-interactive probes, upload scripts, and
monitor jobs without repeatedly asking you to run commands manually.

For NPU jobs, the workflow does not invent resource flags. It probes Slurm
partitions, NPU GRES, CANN/torch_npu runtime availability, and a tiny NPU tensor
smoke test before treating a job as submit-ready.

## Some Basic Command on Ascend

```bash
# Least Smoke Test
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -c 'import torch, torch_npu; x=torch.ones(4, device="npu"); print(torch.__version__, torch.npu.is_available(), torch.npu.device_count(), (x+1).cpu())'
```

```bash
# Basic info
npu-smi info
npu-smi info -l
npu-smi info -t health

# CANN Environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh
find /usr/local/Ascend -maxdepth 3 -name set_env.sh

# Pytorch NPU Check
python -c 'import torch, torch_npu; print(torch.__version__); print(torch_npu.__version__)'
python -c 'import torch, torch_npu; print(torch.npu.is_available()); print(torch.npu.device_count())'
python -c 'import torch, torch_npu; x=torch.ones(4, device="npu"); print((x+1).cpu())'

# Slurm Resources Check
sinfo -o "%P %a %.10l %.6D %.6t %G"
scontrol show partition
squeue -u $USER
sacct -j <jobid> --format=JobID,JobName,Partition,State,ExitCode,Elapsed,ReqTRES%80,AllocTRES%80 -P

# Other Useful Lines
which npu-smi
which python
python -m pip show torch torch-npu torch_npu
ldconfig -p | grep -E 'hccl|ascend|acl'
echo $LD_LIBRARY_PATH | tr ':' '\n' | grep Ascend
```

## Components

Although users normally start with `/using-slurm-ascend`, the suite is split
into focused skills:

- `using-slurm-ascend`: public entrypoint and first-run profile setup flow.
- `hpc4-domestic-workflow`: session record and end-to-end routing.
- `hpc4-domestic-readiness`: entrypoint, data, logs, dependency, resource, and smoke-test checks.
- `hpc4-ascend-portability`: detects CUDA/NVIDIA assumptions and Ascend runtime markers.
- `hpc4-cuda-to-ascend-port`: strict mechanical CUDA-to-Ascend edits.
- `hpc4-domestic-submit`: profile management, SSH, Slurm probing, guardrails, submission, status, and log fetch.
- `hpc4-domestic-diagnose`: pending/failure/runtime diagnosis guidance.

The included scripts provide JSON/static checks and deterministic guardrails so
the agent can explain why a job is or is not ready to submit.

## Feedback

Issues and pull requests are welcome, especially for updated HKUST-GZ HPC4
partition facts, CANN/torch_npu environment changes, AIStudio/container routes,
and additional Ascend portability patterns.

## License

MIT License. See [LICENSE](LICENSE).
