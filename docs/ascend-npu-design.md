# Ascend NPU backend design Spec

## Source prompt

Explore adding an optional Ascend NPU backend feature to omeinsum-rs. Determine the useful first scope, integration architecture, dependency/toolchain strategy, API behavior, validation approach, and explicit non-goals before implementation.

## Approved design

### Goals & constraints

**Decision:** Cover all three outcomes as sequential milestones: A standard-f32 feasibility spike, B a usable standard-algebra backend, then C tropical acceleration with custom kernels.

Milestone A proves AscendCL/ACLNN integration with f32 allocation, copies, and representative contractions. Milestone B hardens layouts, supported low-precision types, backward execution, workspace reuse, documentation, and hardware parity. Milestone C adds max-plus/min-plus (and potentially max-mul) custom kernels after the shared runtime is stable. The stages are ordered because tropical kernels reuse the device storage, stream, layout, error, and test infrastructure from A/B; attempting all layers simultaneously would make failures hard to localize.

**Rejected alternatives:**
- Only a narrow standard-algebra prototype, because the user wants the complete opportunity explored.
- Treat standard and tropical support as one initial milestone, because ACLNN handles standard matmul while tropical contraction requires a different custom-kernel toolchain.

### API surface

**Decision:** Mirror the CUDA public API with Ascend::new(), Ascend::on_device(ordinal), AscendStorage<T>, to_vec(), and optional ascend/ascend-tropical features.

Existing generic einsum and Tensor APIs remain unchanged. Enabling ascend exports Ascend, AscendError, and AscendStorage; enabling ascend-tropical reuses the same backend/runtime and adds tropical contraction support. Both features are included in full. Device selection remains a single ordinal, matching Cuda::on_device.

**Rejected alternatives:**
- A separate Ascend-specific einsum API, because the Backend generic already preserves device type.
- Public communicator or distributed APIs, because scope is single NPU.
- One monolithic Ascend feature, because standard ACLNN support should not depend on custom tropical kernels.

### Architecture

**Decision:** Implement a single-NPU Ascend backend by mirroring the existing CUDA backend's ownership model and curated-FFI approach.

Ascend owns one device/context/stream; AscendStorage owns a device allocation tied to that execution context; cloned backends share runtime state and reusable caches. Curated Rust FFI targets ACL/ACLRT for lifecycle, memory, copies, and synchronization, and ACLNN for standard tensor operators. Tropical support later adds custom Ascend kernels while reusing the same storage/runtime layer. No HCCL, communicator, cross-device transfer, tensor sharding, NCCL, or NVSHMEM is included.

**Rejected alternatives:**
- Multi-NPU HCCL support, because the user explicitly selected single-NPU parity with the current CUDA backend.
- A C/C++ shim, because the selected approach is curated Rust FFI analogous to the repository's handwritten cuTENSOR bindings.
- Depending on an incomplete general-purpose Rust CANN wrapper.

### Data contracts

**Decision:** Start with f32 standard and tropical values plus u32 tropical argmax indices; preserve existing column-major tensor metadata and materialize device layouts only where ACLNN/custom kernels require it.

The current Scalar model exposes f32/f64/complex but no f16/BF16. ACLNN's core matmul path supports f32 but not the package's f64 expectation, so f32 is the honest initial intersection. Tensor shapes, strides, modes, and backend ownership remain unchanged. Host/device copies preserve element order. Standard contractions lower through the existing contraction planner to ACLNN-supported operations; tropical contractions use the same lowering and return u32 winner indices for backward routing.

**Rejected alternatives:**
- Pretend f64 support exists through downcasting, because that silently changes numerical semantics.
- Add f16/BF16 scalar types in the first backend patch, because that expands the public algebra/type system independently of Ascend runtime feasibility.
- Use host fallback inside Ascend contractions, because it would violate device preservation and obscure whether the NPU path works.

### Failure handling

**Decision:** Use AscendError for initialization, device/context/stream, ACLNN, allocation, copy, and unsupported-operation failures; do not silently fall back to CPU.

Constructors return Result so missing CANN libraries, unavailable devices, and setup failures are inspectable. Native status codes are mapped with operation context. Resource wrappers use RAII and avoid panicking in Drop. Existing Backend trait methods cannot return Result, so failures there follow the CUDA precedent and panic with explicit Ascend operation/status context. Unsupported scalar/algebra combinations fail clearly at the boundary. Synchronization is explicit for host-visible copies and Backend::synchronize.

**Rejected alternatives:**
- Automatic CPU fallback, because it breaks backend/device preservation and can hide severe performance or correctness failures.
- Ignore cleanup failures by panicking in Drop, because panic during unwinding is unsafe operationally.
- Redesign the shared Backend trait to return Result in this feature, because that is a cross-package API change beyond the backend scope.

### Testing & rollout

**Decision:** Deliver in three gated milestones on one branch: f32 ACLNN feasibility, hardened standard backend, then f32 custom tropical kernels; validate locally where possible and on hpc4 through reviewed Slurm jobs.

Milestone A tests initialization, allocation/copies, matrix/batched/scalar/transpose/trace contractions, and CPU parity. Milestone B adds device layout handling, public einsum and backward parity, cache reuse, documentation, and explicit capability errors. Milestone C adds max-plus/min-plus/max-mul forward and u32 argmax/backward parity using custom Ascend kernels. Local checks cover default/tropical builds and feature-gate structure without requiring CANN. Toolkit compile/link and all NPU numerical tests run on hpc4's Linux aarch64 CANN 8.5.0 environment. Remote scripts/resources must be reviewed before sbatch; every hardware performance experiment must follow the repository's runscribe hypothesis gate. The feature remains experimental until all required NPU parity tests pass.

**Rejected alternatives:**
- Mark support stable after compile-only validation, because native ABI linkage does not prove device correctness.
- Run device work on the login node, because it lacks NPU visibility and toolchain commands.
- Bundle multi-NPU or performance tuning before single-NPU correctness.

## Open questions

_None._

## Decision log

- 2026-07-21T15:13:17.500Z — **Goals & constraints** (probing): The exploration should preserve the existing generic einsum API and isolate Ascend behind an optional feature. Real execution is Linux-only and requires CANN plus Ascend hardware; this macOS workspace can support design and mock/CPU validation but not hardware validation.
- 2026-07-21T15:14:45.030Z — **Goals & constraints** (done): Cover all three outcomes as sequential milestones: A standard-f32 feasibility spike, B a usable standard-algebra backend, then C tropical acceleration with custom kernels.
- 2026-07-21T15:22:47.519Z — **Testing & rollout** (probing): Use hpc4 as the real-hardware validation environment through Slurm, with local CPU/mock checks on macOS and NPU jobs only after exact scripts and resources are reviewed.
- 2026-07-21T15:24:59.232Z — **Architecture** (probing): Mirror the existing CUDA backend's ownership and feature structure, using curated native Rust FFI, but translate NVIDIA-specific libraries to their Ascend equivalents.
- 2026-07-21T15:30:36.503Z — **Architecture** (done): Implement a single-NPU Ascend backend by mirroring the existing CUDA backend's ownership model and curated-FFI approach.
- 2026-07-21T15:31:34.288Z — **API surface** (done): Mirror the CUDA public API with Ascend::new(), Ascend::on_device(ordinal), AscendStorage<T>, to_vec(), and optional ascend/ascend-tropical features.
- 2026-07-21T15:31:41.175Z — **Data contracts** (done): Start with f32 standard and tropical values plus u32 tropical argmax indices; preserve existing column-major tensor metadata and materialize device layouts only where ACLNN/custom kernels require it.
- 2026-07-21T15:31:50.285Z — **Failure handling** (done): Use AscendError for initialization, device/context/stream, ACLNN, allocation, copy, and unsupported-operation failures; do not silently fall back to CPU.
- 2026-07-21T15:32:01.315Z — **Testing & rollout** (done): Deliver in three gated milestones on one branch: f32 ACLNN feasibility, hardened standard backend, then f32 custom tropical kernels; validate locally where possible and on hpc4 through reviewed Slurm jobs.
