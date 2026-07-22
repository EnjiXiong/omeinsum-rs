#!/usr/bin/env python3
"""Conservative CUDA-to-Ascend source rewriter.

The helper only performs narrow mechanical edits and prints a unified diff by
default. Use --apply to write changes.
"""

from __future__ import annotations

import argparse
import difflib
import pathlib
import re
import sys


TEXT_SUFFIXES = {".py", ".sh", ".slurm", ".sbatch", ".yaml", ".yml", ".toml", ".txt", ".md"}


REPLACEMENTS: list[tuple[str, str]] = [
    (r"\btorch\.cuda\b", "torch.npu"),
    (r"\.cuda\(\)", ".npu()"),
    (r"\.cuda\(", ".npu("),
    (r"(['\"])cuda(:\d+)?\1", lambda m: f"{m.group(1)}npu{m.group(2) or ''}{m.group(1)}"),
    (r"\bCUDA_VISIBLE_DEVICES\b", "ASCEND_VISIBLE_DEVICES"),
    (r"\bCUDA_HOME\b", "ASCEND_HOME"),
    (r"\bnvidia-smi\b", "npu-smi info"),
]


def iter_files(root: pathlib.Path):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            if any(part in {".git", "__pycache__"} for part in path.parts):
                continue
            yield path


def rewrite_python_imports(text: str) -> str:
    if "import torch_npu" in text or "from torch_npu" in text:
        return text
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if re.match(r"\s*import torch(\s|$|,)", line):
            newline = "\n" if line.endswith("\n") else ""
            lines.insert(i + 1, f"import torch_npu{newline}")
            return "".join(lines)
    return text


def rewrite_text(path: pathlib.Path, text: str) -> str:
    updated = text
    for pattern, replacement in REPLACEMENTS:
        updated = re.sub(pattern, replacement, updated)
    if path.suffix == ".py" and updated != text:
        updated = rewrite_python_imports(updated)
    return updated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="File or directory to rewrite")
    parser.add_argument("--apply", action="store_true", help="Write changes instead of printing only")
    parser.add_argument("--dry-run", action="store_true", help="Print diff only; default behavior")
    args = parser.parse_args()

    target = pathlib.Path(args.target).resolve()
    if not target.exists():
        print(f"missing target: {target}", file=sys.stderr)
        return 2

    files = [target] if target.is_file() else list(iter_files(target))
    changed = 0
    for path in files:
        old = path.read_text(encoding="utf-8", errors="replace")
        new = rewrite_text(path, old)
        if new == old:
            continue
        changed += 1
        rel = str(path.relative_to(target if target.is_dir() else target.parent))
        diff = difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"{rel}:before",
            tofile=f"{rel}:after",
        )
        sys.stdout.writelines(diff)
        if args.apply:
            path.write_text(new, encoding="utf-8")

    print(f"\nfiles_changed: {changed}")
    if not args.apply:
        print("mode: dry-run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
