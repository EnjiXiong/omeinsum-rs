#!/usr/bin/env python3
"""First-run helper for HPC4 domestic cluster profiles.

The profile is the TOML file. This helper keeps setup mechanical:

* inspect existing active/candidate profiles before asking the user anything;
* create a user-specific HPC4 profile from a username and remote repo path;
* optionally activate it through profiles/active.toml;
* print the SSH-key bootstrap commands needed for unattended probes.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import cluster_profile as cp


PROFILES_DIR = pathlib.Path(cp.PROFILES_DIR)
ACTIVE = PROFILES_DIR / "active.toml"
EXAMPLE = "hpc4-domestic.example.toml"


def _profile_files() -> list[pathlib.Path]:
    if not PROFILES_DIR.is_dir():
        return []
    return sorted(
        p
        for p in PROFILES_DIR.glob("*.toml")
        if p.name != EXAMPLE and p.name != "active.toml"
    )


def inspect_profiles() -> dict:
    active_target = None
    active_exists = False
    active_valid = False
    active_error = None
    if ACTIVE.exists() or ACTIVE.is_symlink():
        active_target = os.readlink(ACTIVE) if ACTIVE.is_symlink() else str(ACTIVE)
        try:
            profile = cp.load_profile(ACTIVE)
            active_exists = True
            active_valid = not cp.validate(profile)
        except cp.ProfileError as exc:
            active_error = str(exc)

    candidates = []
    for path in _profile_files():
        item = {"path": str(path), "valid": False, "user": None, "repo_path_remote": None}
        try:
            profile = cp.load_profile(path)
            item["valid"] = not cp.validate(profile)
            item["user"] = cp.get_field(profile, "connection.ssh.user")
            item["repo_path_remote"] = cp.get_field(profile, "connection.repo_path_remote")
        except cp.ProfileError as exc:
            item["error"] = str(exc)
        candidates.append(item)

    return {
        "active": {
            "path": str(ACTIVE),
            "target": active_target,
            "exists": active_exists,
            "valid": active_valid,
            "error": active_error,
        },
        "candidates": candidates,
        "needs_setup": not active_valid,
    }


def default_remote_dir(username: str) -> str:
    return f"/data/user/{username}/hpc4-workdir"


def render_profile(username: str, remote_dir: str, alias: str, host: str, identity_file: str) -> str:
    allowed_roots = [
        "~/scratch",
        "~/results",
        "~/hpc4-workdir",
        remote_dir,
    ]
    roots = ", ".join(json.dumps(x) for x in allowed_roots)
    return f'''# HKUST-GZ HPC4 domestic profile for {username}.
# The profile is this TOML file; activate it via profiles/active.toml.

[identity]
name = "hkust-gz-hpc4-domestic-{username}"
purpose = "HKUST-GZ HPC4 domestic Kunpeng/Ascend environment"
maintainer = "{username}"

[connection]
repo_path_remote = "{remote_dir}"

[connection.ssh]
alias = "{alias}"
host = "{host}"
user = "{username}"
identity_file = "{identity_file}"
port = 22

[scheduler]
type = "slurm"
default_partition = "a128m512u"

[[partitions]]
name = "a128m512u"
class = "default-cpu"
cores = 128
memory = "500G"
max_wall = "7-00:00:00"
gpu = ""

[[partitions]]
name = "debug"
class = "debug"
cores = 128
memory = "500G"
max_wall = "00:30:00"
gpu = ""

[[partitions]]
name = "a320m2tn910cu"
class = "ascend-npu"
cores = 512
memory = ""
max_wall = "7-00:00:00"
gpu = "npu:16"

[[partitions]]
name = "a320m2tn910cue"
class = "ascend-npu"
cores = 512
memory = ""
max_wall = "7-00:00:00"
gpu = "npu:16"

[network]
internet_from_login = true
internet_from_compute = false

[region]
region = "mainland_china"

[limits.hard]
max_walltime = "24:00:00"
max_nodes = 1
max_cpus = 192
max_array_size = 200

[limits.soft]
warn_walltime = "08:00:00"
warn_cpus = 64
unusual_partitions = []

[limits.paths]
allowed_roots = [{roots}]

[[documentation]]
url = "https://docs.hpc.hkust-gz.edu.cn/docs/hpc4/domestic/"
documents = ["connection", "quickstart-submit-slurm-job", "quickstart-submit-ai-job", "models-images"]

[[gotchas]]
symptom = "NPU runtime tools are absent on the login node"
cause = "NPU devices and torch_npu runtime must be checked inside an allocated NPU job or container"
fix = "Submit a tiny Slurm probe to a320m2tn910cu with --gres=npu:2, or use the approved AIStudio/container route"

[[gotchas]]
symptom = "sbatch rejects --gres=npu:1"
cause = "HPC4 Slurm may require NPU counts to be 2, 4, 6, 8, 10, 12, 14, or 16"
fix = "Use --gres=npu:2 for the smallest Ascend/NPU smoke jobs"

[commands]
quota_command = ""

[notes]
text = """
Seeded from the HKUST-GZ HPC4 domestic skill defaults. Run live probes before relying on partition/resource facts:
sinfo -o "%P %a %.10l %.6D %.6t %G"
scontrol show partition
npu-smi info inside an allocated NPU job
"""
'''


def create_profile(args: argparse.Namespace) -> dict:
    username = args.username
    remote_dir = args.remote_dir or default_remote_dir(username)
    path = PROFILES_DIR / f"hpc4-domestic.{username}.toml"
    if path.exists() and not args.force:
        raise SystemExit(f"profile already exists: {path} (use --force to overwrite)")
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_profile(username, remote_dir, args.alias, args.host, args.identity_file),
        encoding="utf-8",
    )
    if args.activate:
        if ACTIVE.exists() or ACTIVE.is_symlink():
            ACTIVE.unlink()
        ACTIVE.symlink_to(path.name)
    return {
        "profile": str(path),
        "activated": args.activate,
        "active": str(ACTIVE) if args.activate else None,
        "ssh_alias": args.alias,
        "ssh_host": args.host,
        "username": username,
        "repo_path_remote": remote_dir,
        "identity_file": args.identity_file,
        "ssh_config_block": "\n".join(
            [
                f"Host {args.alias}",
                f"  HostName {args.host}",
                f"  User {username}",
                f"  IdentityFile {args.identity_file}",
                "  IdentitiesOnly yes",
            ]
        ),
        "ssh_key_setup": [
            f"ssh-copy-id -i {args.identity_file}.pub {args.alias}",
            f"cat {args.identity_file}.pub | ssh {args.alias} 'mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys'",
            f"ssh {args.alias} 'echo key-login-ok; hostname; whoami'",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect or create an HPC4 domestic profile.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("inspect", help="Report active and candidate profiles.")
    create = sub.add_parser("create", help="Create a user-specific HPC4 profile TOML.")
    create.add_argument("--username", required=True)
    create.add_argument("--remote-dir", default=None)
    create.add_argument("--alias", default="hpc4")
    create.add_argument("--host", default="hpc4login.hpc.hkust-gz.edu.cn")
    create.add_argument("--identity-file", default="~/.ssh/id_ed25519")
    create.add_argument("--activate", action="store_true")
    create.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.cmd == "inspect":
        print(json.dumps(inspect_profiles(), indent=2))
        return 0
    if args.cmd == "create":
        print(json.dumps(create_profile(args), indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
