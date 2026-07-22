#!/bin/bash
#SBATCH --job-name=harness-array
#SBATCH --output=slurm-%A_%a.out

# Generic harness array wrapper. The submitter supplies:
#   HARNESS_RUN_SPEC=<path to run_spec.json>
#   HARNESS_COMMAND=<shell command>
# or, as a convenience fallback:
#   HARNESS_ENTRYPOINT=<script path>
#   HARNESS_RUNNER=<optional runner command>
# and supplies --array=1-N plus cluster resources at sbatch time. Partition,
# account, QoS, memory, wall-clock, node, task, and CPU choices come from the
# active cluster profile or the submitter's explicit sbatch flags, not this
# wrapper.
#
# Example:
#   sbatch --partition=<queue> --time=<walltime> --cpus-per-task=<n> --array=1-<n_cells> --export=ALL,HARNESS_RUN_SPEC=results/run/run_spec.json,HARNESS_COMMAND='julia --project=julia-env scripts/foo.jl' scripts/hpc4_array_sbatch.sh

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

: "${HARNESS_RUN_SPEC:?Set HARNESS_RUN_SPEC to a run_spec.json path}"
if [[ -z "${HARNESS_COMMAND:-}" && -z "${HARNESS_ENTRYPOINT:-}" ]]; then
  echo "Set HARNESS_COMMAND, or set HARNESS_ENTRYPOINT with optional HARNESS_RUNNER" >&2
  exit 2
fi
CELL_SELECTOR="${SLURM_ARRAY_TASK_ID:-${HARNESS_CELL_INDEX:-${HARNESS_CELL_ID:-single}}}"

echo "Cell selector: ${CELL_SELECTOR}"
echo "Run spec:   ${HARNESS_RUN_SPEC}"
[[ -n "${HARNESS_COMMAND:-}" ]] && echo "Command:    ${HARNESS_COMMAND}"
[[ -n "${HARNESS_ENTRYPOINT:-}" ]] && echo "Entrypoint: ${HARNESS_ENTRYPOINT}"
[[ -n "${HARNESS_RUNNER:-}" ]] && echo "Runner:     ${HARNESS_RUNNER}"
echo "Started:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [[ -n "${HARNESS_COMMAND:-}" ]]; then
  stdbuf -oL bash -lc "$HARNESS_COMMAND"
elif [[ -n "${HARNESS_RUNNER:-}" ]]; then
  stdbuf -oL bash -lc "$HARNESS_RUNNER \"\$1\"" _ "$HARNESS_ENTRYPOINT"
elif [[ "${HARNESS_ENTRYPOINT}" == *.jl ]]; then
  if [[ -n "${HARNESS_JULIA_BIN:-}" ]]; then
    JULIA_CMD="$HARNESS_JULIA_BIN"
  elif [[ -d "$HOME/.juliaup/bin" ]]; then
    export PATH="$HOME/.juliaup/bin:$PATH"
    JULIA_CMD="julia"
  else
    JULIA_CMD="julia"
  fi
  stdbuf -oL "$JULIA_CMD" --project="${HARNESS_JULIA_PROJECT:-julia-env}" "$HARNESS_ENTRYPOINT"
else
  stdbuf -oL "$HARNESS_ENTRYPOINT"
fi

echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
