# Realify: Complex → Real Tensor Network Conversion

**Status:** core implemented (M0–M2), with an M3 benchmark harness, M4 CLI
support, and an M5 feature-gated Ascend integration test. Performance measurements
and the remaining stretch work remain future work.
**Goal:** contract complex-valued tensor networks on backends without native complex
support (today: Ascend, which is f32-only; also CUDA builds without cuTENSOR) by
mechanically rewriting the network into an equivalent real-valued network, with no
asymptotic FLOP or memory overhead versus native complex contraction.

This document is self-contained: the math is stated with verified constants, the
relevant codebase facts are listed, all design decisions are settled with rationale,
and the work is broken into milestones with test gates. An implementing session
should not need to re-derive anything.

---

## 1. Mathematical background (verified)

Source: *tnet.pdf* (Typst lecture notes), section "Complex Numbers: A Tensor-Network
Perspective", pp. 23–25. Every identity below was re-verified numerically with numpy
to machine precision before being written down here.

### 1.1 Realification of a tensor

ℂ is a 2-dimensional algebra over ℝ. Make that explicit as network structure: each
complex tensor `A` with indices `i1..in` becomes a real tensor `T_A` with **one extra
trailing index of dimension 2**:

```text
(T_A)[i1..in, 0] = Re(A[i1..in])
(T_A)[i1..in, 1] = Im(A[i1..in])
```

In column-major layout (what this crate uses) the extra index being *last* means the
data is simply `[ all Re entries (col-major) ; then all Im entries (col-major) ]`.

- **Conjugation** = applying `Z = diag(1, -1)` to the extra leg. Implementation-wise:
  negate the Im block during conversion. No extra tensor needed.
- **Phase multiplication** `e^{iφ}·A` = rotation `R_φ = [[cosφ, -sinφ],[sinφ, cosφ]]`
  on the extra leg (column-major data `[cosφ, sinφ, -sinφ, cosφ]`). Not needed for
  the core feature; export as a utility constant.

### 1.2 The multiplication vertex

Complex multiplication `(x0 + i·x1)(y0 + i·y1)` is a bilinear map ℝ²×ℝ² → ℝ², i.e. a
rank-3 tensor `M[a,b,c]` with `result[c] = Σ_ab x[a]·y[b]·M[a,b,c]`:

```text
M[:,:,0] = [[1, 0], [0, -1]]   # Re = x0·y0 − x1·y1
M[:,:,1] = [[0, 1], [1,  0]]   # Im = x0·y1 + x1·y0
```

Column-major flat data for shape `[2,2,2]`:

```rust
// index (a,b,c) at position a + 2b + 4c
const M_DATA: [f64; 8] = [1.0, 0.0, 0.0, -1.0,   0.0, 1.0, 1.0, 0.0];
```

The lecture notes use a variant `𝒞 = M·Z` (multiplication with conjugated output),
whose nonzeros are `𝒞_000 = 1`, `𝒞_011 = 𝒞_101 = 𝒞_110 = −1`:

```rust
const C_DATA: [f64; 8] = [1.0, 0.0, 0.0, -1.0,   0.0, -1.0, -1.0, 0.0];
```

`𝒞` is **invariant under any permutation of its three legs** and invariant under
`Z⊗Z⊗Z` (both verified). These symmetries make it the nicer object for *tests and
graphical identities*, but the plain `M` is the vertex we insert at runtime (one
fewer tensor; eq. 26 of the notes is `T_A, T_B, 𝒞, Z` chained — pre-contracting
`𝒞·Z = M` is exactly the same network with the trailing `Z` absorbed).

Matrix-product ground truth (notes eq. 26, rewritten with `M`):

```text
D = A·B   ⟺   (T_D)[i,k,d] = Σ_{j,a,b} (T_A)[i,j,a] · (T_B)[j,k,b] · M[a,b,d]
einsum: "ija,jkb,abd->ikd"
```

### 1.3 Whole-network recipe

Given a complex einsum `(ixs, iy, size_dict)` with input tensors `t_1..t_n`, of which
`m` are genuinely complex:

1. **Realify each complex tensor**: tensor `k` gets a fresh label `a_k` appended to
   its `ix` (dimension 2). Real tensors are left untouched — their extra leg would be
   pinned to `e0 = (1,0)`, and `M · e0 = I₂` (verified), so it cancels exactly.
2. **Insert `m − 1` copies of `M`** as ordinary constant input tensors, merging the
   extra labels pairwise: `M(a_1, a_2 → b_2), M(b_2, a_3 → b_3), …, M(b_{m-1}, a_m → b_m)`.
   Associativity/permutation symmetry (the notes' "cascade rule") guarantees any
   merge tree gives the same result; the contraction-order optimizer remains free to
   schedule the merges anywhere in the tree.
3. **Output**: `iy' = iy ++ [b_m]` (or `++ [a_1]` when `m == 1`). Slicing the result
   at the last axis gives Re (index 0) and Im (index 1). A scalar-output network
   becomes a shape-`[2]` vector `[Re, Im]`.
4. `size_dict' = size_dict ∪ { a_k ↦ 2, b_k ↦ 2 }`.

Special case `m == 0`: the network is already real; the transform is the identity and
the caller must know no trailing axis was added (see `RealifiedOutput` below).

A 3-tensor chain (`ija,jkb,abe,klc,ecf->ilf` ≡ `A·B·B'`) was verified numerically.

### 1.4 Cost

Algebraically, a binary contraction of two realified tensors followed by an
`M`-merge requires the same four real products as a native complex contraction
(Re·Re, Im·Im, Re·Im, Im·Re). Memory is 2× real (same as complex), so
realification has **no asymptotic overhead**. Whether the generic optimizer and
backend realize that ideal schedule is a benchmark question; the constant-factor
risks are:

- an optimizer that *defers* `M`-merges leaves intermediates carrying ≥2 extra dim-2
  legs (4× data instead of 2×). This is a **cost issue only, never a correctness
  issue**: omeco keeps every extra label alive until its `M` vertex consumes it (the
  label appears downstream, so node outputs retain it and
  `normalize_binary_operand` will not strip it). Greedy cost penalizes deferral, but
  there is no hard guarantee → benchmark in M3, tree-aligned fallback in M5.
- Gauss's trick (3 real multiplications instead of 4) is a possible later fusion; not
  in scope for v1.

### 1.5 Exact integer test vector (paste into tests)

```text
A = [[1+2i, 3+0i], [0−1i, 2−1i]]        B = [[2+0i, 0+1i], [1−1i, 4+0i]]
D = A·B = [[5+1i, 10+1i], [1−5i, 9−4i]]

Column-major blocks (shape [2,2,2], Re block then Im block):
T_A data = [1, 0, 3, 2,   2, -1, 0, -1]
T_B data = [2, 1, 0, 4,   0, -1, 1, 0]
T_D data = [5, 1, 10, 9,  1, -5, 1, -4]

Network: ixs = [[0,1,10], [1,2,11], [10,11,12]]  (labels 10,11,12 are the extra legs)
         iy  = [0,2,12]
         tensors = [T_A, T_B, M]
         sizes: 0↦2, 1↦2, 2↦2, 10↦2, 11↦2, 12↦2
```

---

## 2. Codebase facts an implementer needs

Repo: `omeinsum` v0.1.1, single crate + `omeinsum-cli` workspace member, edition
2021, column-major tensors throughout. Canonical agent instructions:
`.claude/CLAUDE.md` (read it; `make check` is the pre-PR gate: fmt + clippy +
non-GPU tests).

Key types and where they live:

| Item | Location | Facts that matter here |
|---|---|---|
| `Tensor<T, B>` | `src/tensor/mod.rs` | column-major; `from_data(&[T], &[usize])`, `from_data_with_backend`, `to_vec()`, `permute`, `reshape`, `get(linear)`; `contract_binary::<A>` in `src/tensor/ops.rs` |
| `Scalar` | `src/algebra/mod.rs` | marker trait; impls include `f32, f64, Complex32, Complex64`; requires `bytemuck::Pod` |
| `Standard<T>` | `src/algebra/standard.rs` | the `(+,×)` algebra; realified execution runs `Standard<f64>`/`Standard<f32>` |
| `Einsum<L=usize>` | `src/einsum/engine.rs` | public fields `ixs: Vec<Vec<usize>>`, `iy: Vec<usize>`, `size_dict: HashMap<usize, usize>`; `optimize_greedy()`, `optimize_treesa()`, `set_contraction_tree`, `execute::<A,T,B>(&[&Tensor<T,B>])` |
| one-shot `einsum` | `src/einsum/mod.rs` | infers size_dict from tensors; we construct `Einsum` directly instead |
| `EinBuilder` | `src/einsum/builder.rs` | builder for `Einsum`; not required but keep API style consistent |
| `BackendScalar<B>` | `src/backend/traits.rs` | **Cpu: all scalars. Cuda: f32/f64 always, complex only with cuTENSOR (`cuda` feature). Ascend: `f32` only** ("Milestone A intentionally exposes only CANN's native f32 matmul path") |
| backward | `src/einsum/backward.rs` | gradient tape over the same execute path; realified networks get AD for free since they are ordinary real einsums |

Conventions (from `.claude/CLAUDE.md`):

- unit tests inline under `#[cfg(test)]`; integration suites in `tests/suites/*.rs`
  wired through `tests/main.rs` (do **not** add new top-level `tests/*.rs` crates);
- tests must prove **values**, not shapes;
- topology and tensor data are separate concerns — the design below honors this;
- `num-complex` already has the `bytemuck` feature: `Complex64` is Pod, `#[repr(C)]`
  `{re, im}`, so `bytemuck::cast_slice::<Complex64, f64>` yields interleaved
  `[re0, im0, re1, im1, …]` in column-major element order.

**Trap:** the doc comment on `Tensor::is_contiguous` (`src/tensor/mod.rs`) says
"contiguous in memory (row-major)" — that comment is wrong. The crate is
column-major everywhere: `compute_contiguous_strides` and the unit test
`test_from_data` (strides `[1, 2]` for shape `[2, 3]`) are the ground truth. Trust
the tests, not that comment.

Motivating consumer: `rydbergsim-rs/crates/rydberg-tn` calls omeinsum directly with
`Standard<Complex64>` on CPU (TDVP/GSE contractions; shapes mirrored in
`benches/complex_tdvp.rs`). `yao-rs` (`circuit_to_einsum`) is a second consumer.
Neither should need code changes to *use* the feature — it is a pure omeinsum API.

---

## 3. Design decisions (settled)

- **D1 — Code+data transform, not a new algebra.** Realification is a preprocessing
  step producing an ordinary real einsum. No changes to `Semiring`/`Algebra`, no
  backend-specific code. Every existing optimizer, backend, and the backward tape
  work unchanged. (Rejected alternative: a `RealifiedComplex` algebra type — touches
  every backend kernel, violates "topology and data are separate concerns".)
- **D2 — Insert `M`, export `𝒞`/`Z`/`R_φ` as constants.** `M = 𝒞·Z` is the runtime
  vertex (one tensor per merge). `𝒞`, `Z`, `R_φ`, `E0` are exported for tests and
  identity checks (permutation invariance, `Z⊗Z⊗Z·𝒞 = 𝒞`, `M·e0 = I`).
- **D3 — Extra leg is appended last.** Column-major ⇒ contiguous Re block then Im
  block; conversion from `Complex<T>` storage is a de-interleave; recovery is a
  slice at the last axis. Matches the notes' `cat(real(A), imag(A), dims=n+1)`.
- **D4 — Conjugation is a per-input flag**, applied by negating the Im block during
  conversion. No `Z` tensors in the network. This makes bra-side tensors in
  `⟨ψ|O|ψ⟩` sandwiches share data with ket-side tensors.
- **D5 — Only genuinely complex inputs get an extra leg.** Mixed real/complex
  networks are first-class: real tensors pass through untouched (their `M` would
  collapse to identity). The caller declares which inputs are complex via the input
  enum below; an `is_real` auto-detect (all Im ≈ 0) is deliberately **not** done in
  v1 (silent behavior changes from noisy zeros; revisit later as an explicit opt-in).
- **D6 — Planning is separated from data conversion.** The label/topology transform
  (`RealifyPlan`) is pure and backend-agnostic. Data conversion runs **host-side**
  on slices, *then* uploads via `Tensor::from_data_with_backend`. This is forced:
  `Tensor<Complex64, Ascend>` cannot exist (`BackendScalar` gate), so complex data
  can never touch a real-only backend. CPU-side conversion + real upload is the only
  possible dataflow, and the API should make it the natural one.
- **D7 — Generic over `Complex<T>` with `T ∈ {f32, f64}`.** c64 ↦ f64 network,
  c32 ↦ f32 network. Note for Ascend users: the backend is f32-only today, so c64
  networks targeting Ascend must be explicitly downcast by the caller (precision
  loss is their informed choice); document this, do not silently downcast.
- **D8 — Merge topology in v1 is a left-deep chain** of `M` vertices in input order,
  handed to the optimizer as ordinary inputs. Rationale: simplest correct thing;
  the optimizer can still schedule merges anywhere. If M3 benchmarks show
  intermediates accumulating multiple dim-2 legs, M5 adds a tree-aligned variant
  (optimize the *complex* network first, then attach one `M` per binary node joining
  two complex-carrying subtrees, via `set_contraction_tree`).
- **D9 — Output is explicit about whether a trailing Re/Im axis exists** (see
  `RealifiedOutput`), so `m == 0` (all-real network) is not a silent special case.

---

## 4. Proposed API

New module `src/realify.rs` (single file; split only if it grows past ~600 lines),
re-exported from `lib.rs` as `pub mod realify` plus top-level re-exports of the main
entry points.

```rust
//! src/realify.rs — complex → real tensor network conversion.

use crate::{Backend, BackendScalar, Einsum, Scalar, Tensor};
use num_complex::Complex;

/// Constant tensors (all shape [2,2,2] except Z/E0), column-major data.
pub mod constants {
    /// Multiplication vertex M: result[c] = Σ_ab x[a] y[b] M[a,b,c].
    pub const M_DATA: [f64; 8] = [1.0, 0.0, 0.0, -1.0, 0.0, 1.0, 1.0, 0.0];
    /// Fully symmetric conjugated-output variant 𝒞 = M·Z (tests/identities).
    pub const C_DATA: [f64; 8] = [1.0, 0.0, 0.0, -1.0, 0.0, -1.0, -1.0, 0.0];
    /// Conjugation Z = diag(1, -1).
    pub const Z_DATA: [f64; 4] = [1.0, 0.0, 0.0, -1.0];
    /// Real-unit vector e0; pins the extra leg of a real tensor.
    pub const E0_DATA: [f64; 2] = [1.0, 0.0];
    // plus `pub fn r_phi(phi: f64) -> [f64; 4]` for phase rotations
}

/// Declares one input of the complex network, host-side.
/// `T` is the *real* scalar type of the realified network (f32 or f64).
pub enum RealifyInput<'a, T> {
    /// Already-real tensor: passes through unchanged (no extra leg).
    Real { data: &'a [T], shape: &'a [usize] },
    /// Complex tensor, column-major `Complex<T>` data.
    Complex { data: &'a [Complex<T>], shape: &'a [usize], conjugate: bool },
}

/// Result of the topology transform. Pure labels/sizes — no tensor data.
pub struct RealifyPlan {
    /// The realified einsum spec: original ixs with extra labels appended to
    /// complex inputs, followed by the ixs of the inserted M vertices;
    /// iy' = iy ++ [result_leg] when `output` is `ReImAxis`.
    pub einsum: Einsum<usize>,          // not yet optimized
    /// How many M vertices were appended (== max(m−1, 0)).
    pub num_mul_vertices: usize,
    /// Positions (into the *new* tensor list) of the M vertices.
    pub mul_vertex_positions: Vec<usize>,
    pub output: RealifiedOutput,
}

pub enum RealifiedOutput {
    /// No complex inputs: result is the plain real result, no trailing axis.
    Real,
    /// Result carries a trailing dim-2 axis; index 0 = Re, 1 = Im.
    ReImAxis,
}

/// Pure topology transform. `is_complex[k]` ⇔ input k gets an extra leg.
/// Fresh labels start at max(all labels in ixs/iy/size_dict) + 1.
pub fn realify_code(
    ixs: &[Vec<usize>],
    iy: &[usize],
    size_dict: &std::collections::HashMap<usize, usize>,
    is_complex: &[bool],
) -> RealifyPlan;

/// Host-side data conversion for one complex tensor:
/// interleaved Complex<T> (column-major) → [Re block ; ±Im block], shape ++ [2].
pub fn realify_data<T: Scalar + num_traits::Float>(
    data: &[Complex<T>],
    conjugate: bool,
) -> Vec<T>;

/// One-shot convenience on a chosen backend: plans, converts, uploads,
/// optimizes (greedy), executes with Standard<T>, returns the raw real result
/// plus the output descriptor.
pub fn realify_einsum<T, B>(
    inputs: &[RealifyInput<'_, T>],
    ixs: &[Vec<usize>],
    iy: &[usize],
    backend: B,
) -> (Tensor<T, B>, RealifiedOutput)
where
    T: Scalar + num_traits::Float,
    B: Backend,
    T: BackendScalar<B>;

/// Recover a complex result host-side (CPU use / tests).
/// Splits the trailing axis: returns (re, im) with the original output shape.
pub fn split_re_im<T: Scalar, B: Backend>(
    result: &Tensor<T, B>,
) -> (Vec<T>, Vec<T>);
```

Notes:

- `realify_code` must **not** consume tensor data — keeps it testable in isolation
  and usable by callers (rydberg-tn) that manage their own device tensors.
- `realify_einsum` is deliberately thin sugar over plan + convert + `Einsum::execute`.
  Callers with custom optimization (TreeSA, precomputed trees) use the pieces.
- `realify_data` is a de-interleave loop; use `bytemuck::cast_slice` only as an
  input view (`&[Complex<T>] → &[T]` interleaved), then copy out blocks. Conjugate ⇒
  negate while writing the Im block. No in-place trickery.
- The M vertex tensor for `T = f32` is `M_DATA.map(|x| x as f32)`; keep a small
  helper `mul_vertex_tensor::<T, B>(backend) -> Tensor<T, B>`.

## 5. Milestones

Each milestone ends green under `make check`. Wire the new integration suite into
`tests/main.rs` as `#[path = "suites/realify.rs"] mod realify;` (M2).

### M0 — constants + topology transform (no tensor data)

Files: `src/realify.rs` (new), `src/lib.rs` (module + re-exports).

- Implement `constants`, `realify_code`, `RealifyPlan`, `RealifiedOutput`.
- Fresh-label allocation: `max(labels in ixs ∪ iy ∪ size_dict keys) + 1`. Extra
  labels and merge labels all get size 2 in `size_dict`.
- Handle: `m == 0` (identity plan, `RealifiedOutput::Real`), `m == 1` (no M vertex,
  `iy' = iy ++ [a_1]`), `m ≥ 2` (chain of M vertices per §1.3).
- Repeated labels inside one `ix` (diagonals, e.g. `[0,0]`) are untouched — the
  extra label is simply appended; add a unit test asserting the plan for `[[0,0]]`.
- Unit tests (inline): plan shapes for m ∈ {0,1,2,3}; label freshness (collision
  with a sparse size_dict containing a high unused label); `C_DATA` permutation
  invariance and `Z⊗Z⊗Z·𝒞 = 𝒞` checked numerically against `constants`; `M·e0 = I₂`.

### M1 — data conversion + recovery

Files: `src/realify.rs`.

- `realify_data` (+ conjugate), `split_re_im`, `mul_vertex_tensor`.
- Unit tests with exact integers: the §1.5 vectors for `T_A`/`T_B` (including a
  `conjugate: true` case: `T_A*` data = `[1, 0, 3, 2, -2, 1, 0, 1]`); round-trip
  `Complex → realify_data → split` equality.

### M2 — end-to-end execution + oracle suite

Files: `src/realify.rs` (`realify_einsum`), `tests/suites/realify.rs` (new),
`tests/main.rs` (wire-in).

Oracle: the same network contracted natively with `Standard<Complex64>` on `Cpu`.
Tests prove **values** (`approx::assert_relative_eq`, tol 1e-10 for f64):

1. §1.5 matmul with exact integer asserts (no oracle needed).
2. Random 4–6 tensor chains/networks, f64, mixed real/complex inputs, including a
   permuted / partially traced output `iy`.
3. TDVP-shaped contraction copied from `benches/complex_tdvp.rs` (χ = 32 slice).
4. Sandwich `⟨ψ|O|ψ⟩`: bra = same data with `conjugate: true`; assert Im ≈ 0 for
   Hermitian `O` **and** the value matches the oracle (catches sign errors that
   Im-only checks miss).
5. `m == 0`: all-real network returns `RealifiedOutput::Real`, bit-identical to a
   plain real einsum.
6. `m == 1`: single complex tensor, unary-style network (e.g. partial trace).
7. Scalar output network → shape `[2]` `[Re, Im]`.
8. Both unoptimized (`execute_pairwise`) and `optimize_greedy` paths, and once via
   `optimize_treesa` on the 6-tensor network (guards tree-shape assumptions).
9. Backward smoke test: `einsum_with_grad` on a realified scalar network runs and
   gradients match finite differences on 2–3 entries (AD-for-free claim, verified).
   Note `einsum_with_grad` re-infers `size_dict` from tensors and greedy-optimizes
   internally — fine here, since every size (including the dim-2 legs) is inferable
   from the realified tensors; pass the plan's `ixs`/`iy` slices directly.

### M3 — benchmark + docs

Files: `benches/realify.rs` (new, criterion, mirror `complex_tdvp.rs` cases),
`Cargo.toml` (`[[bench]]`), this document (update status), `docs/src` book page if
the mdbook lists features.

- Compare: native `Standard<Complex64>` CPU vs realified `Standard<f64>` CPU on the
  TDVP shapes. Acceptance: realified within ~1.3× of native complex CPU time (both
  reduce to faer GEMMs; the gap is permute/fold overhead). Record numbers here.
- Inspect (debug print or `contraction_tree()`) whether greedy defers M merges on
  the bench networks. If intermediates carry ≥2 extra legs → schedule M5.
- Implemented: `benches/realify.rs` mirrors the TDVP cases and registers as the
  `realify` Criterion bench. Measurements are intentionally not recorded here yet;
  run the benchmark under the repo's `runscribe` experiment protocol before using
  numbers for a design decision.

### M4 — CLI support (implemented)

Files: `omeinsum-cli/src/contract.rs`, `autodiff.rs`, `format.rs`.

- `--realify` is valid for c32/c64 `contract` and `autodiff`: it plans, converts,
  executes a real network, then reassembles complex result and gradient JSON.
- The CLI remains CPU-only; the flag exercises the backend-independent transform
  but does not yet select Ascend or CUDA. A source topology/expression is validated,
  then the transformed network is greedily replanned because inserted multiplication
  vertices add leaves that do not exist in the source contraction tree.
- For autodiff, a complex seed `u+iv` differentiates the real scalar objective
  `u*Re(y) + v*Im(y)`. Omitting the seed for an original scalar output uses `[1,0]`,
  so gradients are for `Re(y)`. Gradients for internal multiplication vertices are
  discarded before writing one complex gradient per source input.

### M5 — stretch (each independent, do only when justified)

- **Tree-aligned merges**: optimize the complex code first, walk the
  `NestedEinsum`, insert one M per node joining two complex-carrying subtrees,
  install via `set_contraction_tree`. Guarantees every intermediate carries exactly
  one dim-2 leg.
- **Gauss 3-mult fusion** at binary-contract level (needs kernel-side work; only if
  profiling shows the 4th GEMM matters).
- **`R_φ` / phase utilities** and an explicit-opt-in `detect_real` helper.
- **Ascend integration test** behind the `ascend` feature flag, c32 → f32 network,
  guarded like the CUDA suites. Implemented in `tests/suites/ascend.rs`; it requires
  a CANN-enabled host with an allocated NPU to link and execute.

## 6. Correctness invariants (checklist for review)

- [x] Contracting the realified network with `Standard<T>` equals the native
      `Standard<Complex<T>>` result, Re and Im, on every test network.
- [x] Real inputs are byte-identical pass-throughs (no copy beyond what the engine
      does anyway; no extra leg).
- [x] `conjugate: true` equals conjugating the data first (test both paths once).
- [x] Labels: no collisions; every extra/merge label has size 2 in `size_dict`;
      merge labels appear exactly twice (or once in `iy'` for the last one).
- [x] `m == 0` adds no axis; `m == 1` adds axis but no M vertex.
- [x] Result recovery preserves the engine's output ordering (`iy'` order — the
      trailing extra label stays last through `finalize_ordered_result`; assert on
      values after permuted-`iy` tests, which is where this would break).
- [x] No `unsafe`, no backend-specific code paths in `realify.rs`.

## 7. Verification commands

```bash
make check                      # canonical gate: fmt + clippy + non-GPU tests
cargo test --test main realify  # integration suite only
cargo test realify              # + inline unit tests
cargo bench --bench realify     # M3
```

Implementation verification on 2026-07-22:

- `cargo test realify` passed.
- `cargo test --test main realify` passed.
- `cargo check --benches` passed.
- `cargo test -p omeinsum-cli realify` passed (nine M4 contract/autodiff and
  topology-validation cases).
- `cargo check --features ascend --tests` passed locally.
- The feature-gated c32 → f32 Ascend integration test passed on two allocated
  Ascend 910 NPUs in HPC4 Slurm job `109214` (exit `0:0`, empty stderr).
- `make check` passed.
- `cargo bench --bench realify` was not run because benchmark measurements should be
  logged under the repo's `runscribe` experiment protocol.

## 8. References

- *tnet.pdf* pp. 23–25 ("Complex Numbers: A Tensor-Network Perspective"): eq. 23
  (realification), eqs. 24–25 (`𝒞`), eq. 26 (matmul network), graphical properties
  (permutation invariance, conjugation invariance, cascade rule), eq. 27 (backward
  rule `T_B = (T_A ∗ T_{D*})∗` — the Wirtinger conjugations emerge from the real
  network; motivates M2 test 9).
- The notes' own Julia/OMEinsum verification snippet (p. 24) uses
  `ein"ija,jkb,abc,cd->ikd"(T_A, T_B, C, Z)` — equivalent to our `M` form.
- Consumer shapes: `benches/complex_tdvp.rs` (this repo);
  `rydbergsim-rs/crates/rydberg-tn/src/{contraction,gse}.rs` (CPU complex today);
  `yao-rs/src/einsum.rs` (`circuit_to_einsum`, `ArrayD<Complex64>`).
