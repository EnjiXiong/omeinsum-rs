# Ascend NPU performance study

This exploratory study asks where the experimental single-NPU Ascend backend is
useful beyond small square matrix multiplication. It measures standard `f32` and
max-plus `f32` contractions across scale, shape, batch layout, and repeated
device-resident chains. The evidence supports standard `f32` acceleration once
there is enough arithmetic or batching, but it does **not** support a blanket
claim that every contraction is faster on the NPU.

## Result summary

- Standard square `f32` becomes faster than the sequential CPU baseline at
  `128×128×128` when inputs are already resident, at `256×256×256` when host
  upload and result download are included, and scales to a 70.8× host-to-host
  speedup at size 2048. This follows directly from the timing table below.
- Shape and layout determine whether offload pays. `16×1024×16` is 4.1× slower
  host-to-host, while `64×1024×64` is 1.31× faster. A batch-last workload grows
  from 0.39× CPU speed at batch 1 to 5.93× at batch 128, whereas batch-first at
  batch 8 is 0.33×, consistent with the backend's known host-assisted layout
  materialization cost.
- A 20-step standard `128×128` chain is 1.96× faster while device-resident and
  1.89× faster with the final download. This is a closer proxy for tensor-network
  pipelines than isolated GEMM because intermediate tensors remain on device.
- Max-plus does not scale competitively with the current custom kernel. It wins
  at size 64 and in the tested 20-step size-64 recurrence, but loses from size
  128 through 512. The larger cases remain near 0.48 GOP/s on Ascend while the
  sequential CPU remains near 0.7 GOP/s. Contraction-only timings show the same
  trend, so the custom contraction path—not result transfer—is the next
  optimization target.
- Every standard case agreed with the CPU reference. Maximum absolute error was
  at most `4.292e-6` for isolated cases; the 20-step standard chain had
  `2.057e-6` maximum relative error. Every max-plus result matched exactly.

These are crossover observations for one machine and one sequential CPU
implementation, not general hardware rankings.

## Measurement scopes

The harness reports three scopes because “NPU time” can otherwise hide transfer
costs:

| Scope | Inputs | Output | Included work |
|---|---|---|---|
| `contraction` | Already resident | Not downloaded | Synchronized contraction, output allocation, and output destruction |
| `result_download` | Already resident | Downloaded | `contraction` plus `to_vec()` device-to-host copy |
| `host_to_host` | Host slices | Downloaded | Two tensor constructions/uploads, contraction, and result download |

The backend synchronizes each contraction, so `contraction` is not an asynchronous
kernel-only event. Inputs are constructed outside the timed region in the first
two scopes. The chain's `device_resident` scope performs all intermediate
contractions on device and excludes only the final result download.

Each case receives one untimed warm-up. Timed results are arithmetic means over
five iterations for small cases, three for larger cases, and one for the largest
cases. Reported GOP/s uses `2mnk` operations (multiplies plus adds, or additions
plus maxima). This convention is useful for scaling comparisons but does not
establish hardware peak utilization.

## Hardware and software

| Item | Value |
|---|---|
| Slurm job | `109096`, `COMPLETED`, exit `0:0`, elapsed 16 s |
| Host | `npu1-1`, Linux aarch64, kernel `5.10.0-216.0.0.115.oe2203sp4.aarch64` |
| NPU | Ascend 910, runtime SoC `Ascend910_9382` |
| CANN | 8.5.0 |
| `npu-smi` | 25.5.1 |
| Backend implementation checkpoint | `44a878f` |
| Study harness source | The repository commit containing this document |
| Study binary SHA-256 | `ea991744bed99234133c3cf3d756df67ab5bc599374b33814f09000f150fff90` |
| CPU baseline | omeinsum CPU backend; standard GEMM uses `faer::Par::Seq` |
| NPU allocation | Slurm requires `--gres=npu:2`; omeinsum uses device ordinal 0 only |

The CPU comparison is deliberately called **sequential**. Allocating one CPU task
does not make it a whole-node or all-core CPU baseline, and standard CPU GEMM is
explicitly configured with `faer::Par::Seq`.

## Standard square scaling

Times are milliseconds. “Speedup” is CPU time divided by Ascend time; values
below 1 mean Ascend is slower.

| Size | Scope | CPU ms | Ascend ms | Speedup |
|---:|---|---:|---:|---:|
| 64 | contraction | 0.0167 | 0.0585 | 0.29× |
| 64 | result download | 0.0191 | 0.0572 | 0.33× |
| 64 | host to host | 0.0226 | 0.0893 | 0.25× |
| 128 | contraction | 0.0737 | 0.0341 | 2.16× |
| 128 | result download | 0.0717 | 0.0569 | 1.26× |
| 128 | host to host | 0.0764 | 0.0947 | 0.81× |
| 256 | contraction | 0.5177 | 0.0754 | 6.87× |
| 256 | result download | 0.5173 | 0.0963 | 5.37× |
| 256 | host to host | 0.5384 | 0.1584 | 3.40× |
| 512 | contraction | 4.0038 | 0.0967 | 41.42× |
| 512 | result download | 3.9882 | 0.1606 | 24.84× |
| 512 | host to host | 4.0428 | 0.3322 | 12.17× |
| 1024 | contraction | 31.6019 | 0.1175 | 268.87× |
| 1024 | result download | 31.4533 | 0.4506 | 69.80× |
| 1024 | host to host | 31.4681 | 1.1264 | 27.94× |
| 2048 | contraction | 248.5357 | 0.2626 | 946.37× |
| 2048 | result download | 247.9966 | 1.5490 | 160.10× |
| 2048 | host to host | 253.2964 | 3.5786 | 70.78× |

The very large contraction-only ratios compare ACLNN on an Ascend 910 against a
single sequential CPU path. They should not be read as an NPU-versus-server-CPU
claim. The host-to-host column is the safer estimate for isolated calls with host
inputs, while resident pipelines lie between the contraction and result-download
scopes.

## Max-plus square scaling

| Size | CPU ms | Ascend ms | Speedup | Exact match |
|---:|---:|---:|---:|---:|
| 32 | 0.0814 | 0.1070 | 0.76× | yes |
| 64 | 0.6567 | 0.3773 | 1.74× | yes |
| 128 | 5.5355 | 9.5169 | 0.58× | yes |
| 256 | 45.6395 | 70.1200 | 0.65× | yes |
| 512 | 392.1705 | 557.2992 | 0.70× | yes |

These are `result_download` measurements. Since download is small relative to the
128–512 runtimes and contraction-only measurements show the same trend, the
current 50-worker scalar/tiled Ascend C kernel is the bottleneck at larger sizes.

## Applied workload proxies

| Workload proxy | Shape/configuration | Scope | CPU ms | Ascend ms | Speedup |
|---|---|---|---:|---:|---:|
| Narrow scientific contraction | `16×1024×16` | host to host | 0.0239 | 0.0971 | 0.25× |
| Rectangular tensor contraction | `64×1024×64` | host to host | 0.1915 | 0.1462 | 1.31× |
| Wide scientific contraction | `256×64×256` | host to host | 0.1474 | 0.1275 | 1.16× |
| Batched attention-like GEMM | `B=1`, batch last, `N=64` | result download | 0.0226 | 0.0573 | 0.39× |
| Batched attention-like GEMM | `B=8`, batch last, `N=64` | result download | 0.0960 | 0.0719 | 1.33× |
| Batched attention-like GEMM | `B=32`, batch last, `N=64` | result download | 0.3755 | 0.1172 | 3.20× |
| Batched attention-like GEMM | `B=128`, batch last, `N=64` | result download | 1.4784 | 0.2492 | 5.93× |
| Noncanonical batch layout | `B=8`, batch first, `N=64` | result download | 0.1794 | 0.5432 | 0.33× |
| Permuted input and output | `N=256` | result download | 0.7147 | 0.6104 | 1.17× |
| Tensor-network-style chain | 20 standard `128×128` steps | device resident | 1.3939 | 0.7122 | 1.96× |
| Tensor-network-style chain | 20 standard `128×128` steps | result download | 1.3941 | 0.7393 | 1.89× |
| Viterbi-style recurrence | 20 max-plus `64×64` steps | result download | 12.9125 | 7.0011 | 1.84× |

The batch-last case maps the batch label to ACLNN's canonical batch dimension.
The batch-first and permuted cases intentionally force layout handling. Their
contrast demonstrates that operation count alone is insufficient: whether modes
are already in the backend's preferred order materially changes the result.

## Accuracy

For each workload, the harness computes a CPU reference before timing and fails
the job if the comparison exceeds its tolerance.

| Group | Cases | Observed result |
|---|---|---|
| Standard square | sizes 64–2048 | maximum absolute error `3.58e-7` to `4.292e-6` |
| Standard rectangular | three shapes | maximum absolute error ≤ `2.742e-6` |
| Standard batched/layout | five batch cases plus permutation | maximum absolute error ≤ `8.34e-7` |
| Standard chain | 20 steps | maximum relative error `2.057e-6`; all values finite |
| Max-plus square and chain | six cases | exact elementwise equality |

Only `f32` is exercised because the Ascend backend rejects `f64` and complex
scalars at the backend-scalar boundary. There is no silent CPU fallback, so these
results cannot accidentally measure unsupported types on the CPU.

## Reproduction

Cross-build the benchmark on the development machine:

```bash
ASCEND_HOME_PATH=/tmp/omeinsum-cann \
ASCEND_TROPICAL_KERNEL=/tmp/libomeinsum_tropical_gemm-9382-final.so \
RUSTFLAGS='-C link-arg=-Wl,--allow-shlib-undefined' \
cargo zigbuild --release \
  --target aarch64-unknown-linux-gnu.2.28 \
  --features ascend-tropical \
  --example ascend_study
```

The submitted Slurm script was:

```bash
#!/bin/bash
#SBATCH --job-name=omeinsum-ascend-study
#SBATCH --partition=a320m2tn910cu
#SBATCH --gres=npu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

source /usr/local/Ascend/cann-8.5.0/set_env.sh
set -euo pipefail
cd /data/user/yzhao053/omeinsum-ascend-smoke
mkdir -p results
export LD_LIBRARY_PATH="$PWD/multicore:${LD_LIBRARY_PATH:-}"
export OMEINSUM_ASCEND_STUDY="${OMEINSUM_ASCEND_STUDY:-quick}"

{
    printf "METADATA,job_id,%s\n" "$SLURM_JOB_ID"
    printf "METADATA,profile,%s\n" "$OMEINSUM_ASCEND_STUDY"
    printf "METADATA,hostname,%s\n" "$(hostname)"
    printf "METADATA,uname,%s\n" "$(uname -a)"
    printf "METADATA,cann,8.5.0\n"
    printf "METADATA,implementation_commit,44a878f\n"
    npu-smi info
} > "results/ascend-study-${OMEINSUM_ASCEND_STUDY}-${SLURM_JOB_ID}.metadata"

./ascend_study | tee "results/ascend-study-${OMEINSUM_ASCEND_STUDY}-${SLURM_JOB_ID}.csv"
printf 'ok\n' > "results/ascend-study-${OMEINSUM_ASCEND_STUDY}-${SLURM_JOB_ID}.ok"
```

Submit the documented matrix with:

```bash
ssh hpc4
cd /data/user/yzhao053/omeinsum-ascend-smoke
sbatch --export=ALL,OMEINSUM_ASCEND_STUDY=full ascend-study.sbatch
```

The complete raw CSV, metadata, stdout, empty stderr, completion marker, staged
script, and binary hash remain under
`/data/user/yzhao053/omeinsum-ascend-smoke` for job `109096`. The repository's
preferred `runscribe` recorder was unavailable both locally and remotely, so job
ID, script, scheduler status, logs, marker, metadata, and binary hash were
preserved directly rather than fabricating a runscribe record.

## Limitations and next work

1. **CPU comparison:** the baseline is one sequential faer path, not a tuned
   all-core CPU implementation. A hardware purchasing or backend-ranking study
   must add a fair multicore baseline.
2. **Statistical strength:** this is one Slurm run with 1–5 timed iterations per
   point. Sub-millisecond NPU timings varied across validation jobs, so robust
   claims need repeated jobs, medians, percentiles, and confidence intervals.
3. **Single NPU:** no multi-NPU scaling, HCCL, sharding, overlap, or communication
   is implemented. The scheduler allocates two NPUs but this backend uses only
   device 0, so allocation efficiency is 50% for this cluster route.
4. **Data types:** only `f32` is supported. Scientific workloads requiring `f64`
   or complex arithmetic cannot use this backend today.
5. **Layout costs:** noncanonical and host-assisted layouts can erase the NPU
   benefit. Device-native transpose/materialization and better lowering should be
   optimized before expecting broad einsum speedups.
6. **Tropical kernel:** large max-plus GEMM is slower than the sequential CPU.
   Profiling and redesigning its tiling/vectorization is higher priority than
   expanding tropical workload coverage.
7. **Workload fidelity:** attention, tensor-network, scientific, and Viterbi
   entries are contraction proxies with representative shapes, not complete
   applications. End-to-end application studies should include contraction-order
   optimization, surrounding operators, memory pressure, and I/O.

The evidence therefore supports continued development for standard `f32`
resident or sufficiently large/batched workloads. It does not yet support
production claims for arbitrary layouts, max-plus scaling, unsupported scalar
types, multicore CPU competition, or multiple NPUs.
