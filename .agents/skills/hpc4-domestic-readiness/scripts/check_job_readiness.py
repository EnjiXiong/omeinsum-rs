#!/usr/bin/env python3
"""Static readiness checks for a Slurm/container job directory."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Iterable


SBATCH_KEYS = {
    "--job-name": "job_name",
    "--output": "stdout",
    "--error": "stderr",
    "--time": "time",
    "--partition": "partition",
    "--nodes": "nodes",
    "--ntasks": "ntasks",
    "--ntasks-per-node": "ntasks_per_node",
    "--cpus-per-task": "cpus_per_task",
    "--mem": "memory",
    "--gres": "gres",
}


def read_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def parse_sbatch(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#SBATCH"):
            if stripped and not stripped.startswith("#"):
                break
            continue
        rest = stripped[len("#SBATCH") :].strip()
        for key, label in SBATCH_KEYS.items():
            if rest.startswith(key):
                value = rest[len(key) :].strip()
                if value.startswith("="):
                    value = value[1:].strip()
                out[label] = value or "<present>"
    return out


def find_files(root: pathlib.Path, patterns: Iterable[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        paths.extend(str(p.relative_to(root)) for p in root.rglob(pattern) if p.is_file())
    return sorted(set(paths))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="Project/job directory")
    parser.add_argument("--script", help="Primary sbatch or entry script")
    args = parser.parse_args()

    root = pathlib.Path(args.target).resolve()
    report: dict[str, object] = {
        "target": str(root),
        "status": "submit-ready",
        "blocking": [],
        "warnings": [],
        "facts": {},
    }

    if not root.exists():
        report["status"] = "blocked"
        report["blocking"].append("target directory does not exist")
        print(json.dumps(report, indent=2))
        return 1

    scripts = [pathlib.Path(args.script)] if args.script else []
    if not scripts:
        scripts = [root / p for p in find_files(root, ["*.sh", "*.slurm", "*.sbatch"])]
    scripts = [p if p.is_absolute() else root / p for p in scripts]
    existing_scripts = [p for p in scripts if p.exists()]
    report["facts"]["candidate_scripts"] = [str(p.relative_to(root)) for p in existing_scripts]

    if not existing_scripts:
        report["status"] = "blocked"
        report["blocking"].append("no shell/slurm entry script found; provide --script")
    else:
        text = read_text(existing_scripts[0])
        directives = parse_sbatch(text)
        report["facts"]["sbatch_directives"] = directives
        for label in ["job_name", "stdout", "stderr", "time"]:
            if label not in directives:
                report["warnings"].append(f"missing SBATCH {label} directive")
        if "partition" not in directives:
            report["warnings"].append("no partition declared; must be chosen after HPC4 resource probe")
        if not re.search(r"\b(module load|conda activate|source .*(set_env|activate)|apptainer|singularity|docker)\b", text):
            report["warnings"].append("no environment activation/module/container command detected")
        if not re.search(r"\b(python|python3|bash|srun|mpirun|torchrun|deepspeed|mindie|vllm)\b", text):
            report["warnings"].append("no obvious compute command detected")

    data_markers = find_files(root, ["*.json", "*.yaml", "*.yml", "*.toml", "*.txt"])
    report["facts"]["config_like_files"] = data_markers[:50]
    if not any(name in data_markers for name in ["requirements.txt", "environment.yml", "pyproject.toml"]):
        report["warnings"].append("no dependency manifest found")

    if report["blocking"]:
        report["status"] = "blocked"
    elif report["warnings"]:
        report["status"] = "needs-review"

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if report["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
