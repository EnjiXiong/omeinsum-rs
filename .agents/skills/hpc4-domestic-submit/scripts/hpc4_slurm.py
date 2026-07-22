#!/usr/bin/env python3
"""Self-contained SSH/Slurm helper for HKUST-GZ HPC4 workflows."""

from __future__ import annotations

import argparse
import json
import pathlib
import shlex
import subprocess
import sys


def run(cmd: list[str], *, execute: bool) -> subprocess.CompletedProcess[str] | None:
    if not execute:
        print("+ " + " ".join(shlex.quote(x) for x in cmd))
        return None
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def emit(result: subprocess.CompletedProcess[str] | None) -> int:
    if result is None:
        return 0
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return result.returncode


def ssh(host: str, command: str, *, execute: bool):
    return run(["ssh", host, command], execute=execute)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    pre = sub.add_parser("precheck")
    pre.add_argument("--host", required=True)
    pre.add_argument("--yes", action="store_true")

    probe = sub.add_parser("probe")
    probe.add_argument("--host", required=True)
    probe.add_argument("--yes", action="store_true")

    submit = sub.add_parser("submit")
    submit.add_argument("--host", required=True)
    submit.add_argument("--remote-dir", required=True)
    submit.add_argument("--script", required=True)
    submit.add_argument("--yes", action="store_true")

    status = sub.add_parser("status")
    status.add_argument("--host", required=True)
    status.add_argument("--jobid", required=True)
    status.add_argument("--yes", action="store_true")

    tail = sub.add_parser("tail")
    tail.add_argument("--host", required=True)
    tail.add_argument("--remote-path", required=True)
    tail.add_argument("--lines", default="80")
    tail.add_argument("--yes", action="store_true")

    cancel = sub.add_parser("cancel")
    cancel.add_argument("--host", required=True)
    cancel.add_argument("--jobid", required=True)
    cancel.add_argument("--yes", action="store_true")

    args = parser.parse_args()
    execute = bool(args.yes)

    if args.cmd == "precheck":
        result = ssh(args.host, "hostname && whoami && command -v sbatch && command -v sinfo", execute=execute)
        return emit(result)

    if args.cmd == "probe":
        command = (
            "printf '== sinfo ==\\n'; sinfo -o '%P %a %.10l %.6D %.6t %G'; "
            "printf '\\n== partitions ==\\n'; scontrol show partition; "
            "printf '\\n== modules ==\\n'; module av 2>&1 | head -200"
        )
        result = ssh(args.host, command, execute=execute)
        return emit(result)

    if args.cmd == "submit":
        script = pathlib.Path(args.script).resolve()
        if not script.exists():
            print(f"missing script: {script}", file=sys.stderr)
            return 2
        remote_dir = args.remote_dir.rstrip("/")
        remote_script = f"{remote_dir}/{script.name}"
        commands = [
            ["ssh", args.host, f"mkdir -p {shlex.quote(remote_dir)}"],
            ["scp", str(script), f"{args.host}:{remote_script}"],
            ["ssh", args.host, f"cd {shlex.quote(remote_dir)} && sbatch {shlex.quote(script.name)}"],
        ]
        for cmd in commands:
            result = run(cmd, execute=execute)
            rc = emit(result)
            if rc:
                return rc
        if not execute:
            print(json.dumps({"dry_run": True, "remote_script": remote_script}, indent=2))
        return 0

    if args.cmd == "status":
        command = (
            f"squeue -j {shlex.quote(args.jobid)} -o '%i %T %M %D %R'; "
            f"sacct -j {shlex.quote(args.jobid)} --format=JobID,State,ExitCode,Elapsed,MaxRSS 2>/dev/null || true"
        )
        result = ssh(args.host, command, execute=execute)
        return emit(result)

    if args.cmd == "tail":
        result = ssh(args.host, f"tail -n {shlex.quote(str(args.lines))} {shlex.quote(args.remote_path)}", execute=execute)
        return emit(result)

    if args.cmd == "cancel":
        result = ssh(args.host, f"scancel {shlex.quote(args.jobid)}", execute=execute)
        return emit(result)

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
