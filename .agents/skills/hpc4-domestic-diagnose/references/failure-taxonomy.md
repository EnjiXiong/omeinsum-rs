# HPC4 Domestic Failure Taxonomy

## Scheduler and Resource

- Pending `Priority` or `Resources`: queue/resource wait.
- Account/QOS limits: waiting may not help; reduce resource request or ask admin.
- `TIMEOUT`: walltime too short or startup hung.
- `OUT_OF_MEMORY`: memory or NPU HBM pressure; inspect framework logs.

## Environment

- `ModuleNotFoundError: torch_npu`: wrong Python environment/image.
- `libhccl.so`, `libascendcl.so`, `libatb.so` missing: CANN/ATB/MindIE env not sourced or wrong image.
- `Failed to infer device type`: framework cannot see Ascend backend.
- Python wheel architecture mismatch: x86 wheel on aarch64 or incompatible Python ABI.

## Device and Container Mapping

- `npu-smi info` missing or empty in a supposed NPU container: device not allocated/mapped.
- `torch.npu.is_available() == False`: backend import or device visibility failure.
- Requested NPU count not matching 910C/910B chip semantics: re-check allocation unit.

## Portability

- CUDA API calls remain in code.
- NCCL-only launch used on HCCL environment.
- Unsupported operator in CANN migration report.
- Custom CUDA extension has no Ascend equivalent.

## Data and Logic

- Input path exists locally but not remotely/container-mounted.
- Output/log path not writable.
- Job exits 0 but no success artifact: application-level failure, not scheduler success.
