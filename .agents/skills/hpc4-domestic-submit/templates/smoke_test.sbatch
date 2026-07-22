#!/bin/bash
#SBATCH --job-name=hpc4-smoke
#SBATCH --output=results/smoke-%j.out
#SBATCH --error=results/smoke-%j.err
#SBATCH --time=00:05:00

set -euo pipefail
mkdir -p results/smoke
python - <<'PY'
import json
import platform
import sys

manifest = {
    "status": "ok",
    "python": sys.version.split()[0],
    "machine": platform.machine(),
}
with open("results/smoke/manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
print("hpc4 smoke ok", manifest, flush=True)
PY
