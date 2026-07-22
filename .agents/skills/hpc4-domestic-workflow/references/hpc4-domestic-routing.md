# HKUST-GZ HPC4 Domestic Routing Notes

## Sources to Re-check

Re-check these when platform routing, images, or version constraints matter:

- HPC4 domestic manual: `https://docs.hpc.hkust-gz.edu.cn/docs/hpc4/domestic/`
- Connection: `https://docs.hpc.hkust-gz.edu.cn/docs/hpc4/domestic/connection/`
- Slurm job page: `https://docs.hpc.hkust-gz.edu.cn/docs/hpc4/domestic/quickstart-submit-slurm-job/`
- Container/AIStudio job page: `https://docs.hpc.hkust-gz.edu.cn/docs/hpc4/domestic/quickstart-submit-ai-job/`
- Built-in images/models: `https://docs.hpc.hkust-gz.edu.cn/docs/hpc4/domestic/models-images/`
- Software/model adaptation lists under the HPC4 domestic manual.

## Current Platform Facts Observed on 2026-07-08

- SSH login host is `hpc4login.hpc.hkust-gz.edu.cn`.
- The public Slurm quickstart says HPC4 jobs are submitted to the Kunpeng CPU cluster and lists partition `hpc`.
- The public container job page describes AIStudio/container development for NPU resources. It says resource type `NPU`, counts should be multiples of 2, and one 910C corresponds to two 910B chips.
- The built-in image page lists current Ascend-oriented images including `ascend-pytorch 2.8.0-A3-ubuntu22.03-py311`, vLLM Ascend, MindIE, MindSpore, and LLaMA Factory images.

These facts are not permanent. Verify against current docs and actual cluster commands before making a submission decision.

## Routing Rule

Do not infer `sbatch` NPU support from the existence of NPU containers. Before submitting an accelerator job through Slurm, verify at least one of:

- `sinfo` or `scontrol show partition` exposes a partition/QOS/GRES that can allocate Ascend/NPU resources.
- Local HPC documentation or an administrator-provided profile explicitly gives the Slurm directive for NPU allocation.
- A known working sbatch example from the same cluster requests Ascend/NPU resources and matches the current account.

If none are true, route NPU work to AIStudio/container guidance and do not attempt NPU `sbatch` submission.
