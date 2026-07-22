#!/usr/bin/env python3
"""Static CUDA/NVIDIA-to-Ascend portability scan."""

from __future__ import annotations

import argparse
import json
import pathlib
import re


TEXT_SUFFIXES = {
    ".py",
    ".sh",
    ".slurm",
    ".sbatch",
    ".yaml",
    ".yml",
    ".toml",
    ".txt",
    ".md",
    ".json",
}

BLOCKING_PATTERNS = {
    "torch.cuda": r"\btorch\.cuda\b",
    "cuda device strings": r"['\"]cuda(?::\d+)?['\"]",
    "CUDA_VISIBLE_DEVICES": r"\bCUDA_VISIBLE_DEVICES\b",
    "nvidia-smi": r"\bnvidia-smi\b",
    "NCCL": r"\bNCCL\b",
    "CUDA_HOME": r"\bCUDA_HOME\b",
    "cupy": r"\bcupy\b",
    "numba.cuda": r"\bnumba\.cuda\b",
}

ASCEND_MARKERS = {
    "torch_npu": r"\btorch_npu\b",
    "torch.npu": r"\btorch\.npu\b",
    "transfer_to_npu": r"\btransfer_to_npu\b",
    "ASCEND_VISIBLE_DEVICES": r"\bASCEND_VISIBLE_DEVICES\b",
    "npu-smi": r"\bnpu-smi\b",
    "CANN set_env": r"Ascend/.*/set_env\.sh|ascend-toolkit/set_env\.sh",
    "HCCL": r"\bHCCL\b",
    "MindSpore": r"\bmindspore\b",
    "MindIE": r"\bmindie\b",
    "vLLM Ascend": r"\bvllm[_-]ascend\b",
}


def iter_text_files(root: pathlib.Path):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            if any(part.startswith(".") and part not in {".env"} for part in path.relative_to(root).parts):
                continue
            yield path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target")
    args = parser.parse_args()
    root = pathlib.Path(args.target).resolve()
    report: dict[str, object] = {
        "target": str(root),
        "status": "portable-or-not-applicable",
        "blocking": [],
        "warnings": [],
        "ascend_markers": [],
        "hits": [],
    }

    if not root.exists():
        report["status"] = "blocked"
        report["blocking"].append("target directory does not exist")
        print(json.dumps(report, indent=2))
        return 1

    markers = set()
    for path in iter_text_files(root):
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = str(path.relative_to(root))
        for label, pattern in BLOCKING_PATTERNS.items():
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                line = text[: match.start()].count("\n") + 1
                report["hits"].append({"file": rel, "line": line, "pattern": label})
        for label, pattern in ASCEND_MARKERS.items():
            if re.search(pattern, text, flags=re.IGNORECASE):
                markers.add(label)

    report["ascend_markers"] = sorted(markers)
    if report["hits"]:
        report["status"] = "needs-porting"
        report["blocking"].append("CUDA/NVIDIA-specific code or environment markers detected")
    if not markers:
        report["warnings"].append("no Ascend runtime markers detected; verify this is CPU-only or add explicit NPU environment setup")
    if "torch_npu" in markers and "CANN set_env" not in markers:
        report["warnings"].append("torch_npu is referenced but no CANN set_env command was detected")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if report["status"] in {"blocked", "needs-porting"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
