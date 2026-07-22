# Cluster Profiles

Per-cluster profile describing **generic remote-execution conventions** — ssh access, scheduler, partitions, filesystem, internet reach, region, and safety **limits** — needed to drive `hpc4-domestic-submit` without hard-coding site specifics into the skill. Language-specific setup is not in this schema.

**The TOML file is the profile.** Do not present "profile" and "TOML" as two separate setup objects. Each cluster/account gets one unified TOML file under `.agents/skills/hpc4-domestic-submit/profiles/<name>.toml`. The harness reads it through one parser — `scripts/cluster_profile.py` — so skills never parse TOML by hand. `hpc4_harness_slurm.sh` shells out to that parser for the few fields it needs; Python code imports it.

## First-run flow

Always inspect before asking the user for setup information:

```sh
python .agents/skills/hpc4-domestic-submit/scripts/cluster_profile_setup.py inspect
```

If the report shows a valid active profile, use it. If not, ask one compact setup question:

```text
What is your HPC4 username, where should I stage the repo on the cluster
(default: /data/user/<username>/hpc4-workdir), and should I set up SSH-key
access so I can run non-interactive probes?
```

After the username is known, create and activate the profile:

```sh
python .agents/skills/hpc4-domestic-submit/scripts/cluster_profile_setup.py create \
  --username <username> \
  --remote-dir /data/user/<username>/hpc4-workdir \
  --activate
```

The helper prints the `~/.ssh/config` block, an `ssh-copy-id` command, a manual
`authorized_keys` fallback, and a verification command. Prefer this over walking
the user through manual TOML edits.

## SSH-key bootstrap

Once a username exists, ask whether the user wants unattended agent access for probes/submission. If yes:

1. Reuse an existing public key such as `~/.ssh/id_ed25519.pub`; do not read or display private keys.
2. Ensure `~/.ssh/config` has the printed `Host <alias>` block.
3. Have the user run the printed `ssh-copy-id -i <identity>.pub <alias>` in an interactive terminal so they enter the password locally.
4. If `ssh-copy-id` is unavailable, use the printed fallback that pipes the public key into `~/.ssh/authorized_keys` over `ssh`.
5. Verify with `ssh <alias> 'echo key-login-ok; hostname; whoami'`.

Do not ask for the user's password in chat.

Skills consult the active profile via either:

- environment variable `HARNESS_CLUSTER_PROFILE=<name>` (→ `<name>.toml`);
- symlink `.agents/skills/hpc4-domestic-submit/profiles/active.toml → <name>.toml` (preferred when the user wants the choice persisted across sessions).

A skill that needs cluster information resolves the active profile (`cluster_profile.resolve_profile_path`) and falls back to a built-in minimal-Slurm default if neither is present.

## Profile schema (TOML tables)

The schema is **additive**: new fields land as new keys/tables; parsers that don't read them ignore them. Required anchors (`cluster_profile.validate` warns if absent): `[identity]`, `[connection]`, `[scheduler]`.

| Table | Keys | Purpose |
|---|---|---|
| `[identity]` | `name`, `purpose`, `maintainer` | One-line who/what. |
| `[connection]` | `repo_path_remote` | Where the harness checkout lives on the cluster. |
| `[connection.ssh]` | `alias`, `host`, `user`, `identity_file`, `port` | ssh handle + the source-of-truth fields to reconstruct `~/.ssh/config`. The harness uses `alias` as the handle. |
| `[scheduler]` | `type` (`slurm`/`pbs`/`lsf`/`none`), `default_partition` | How jobs are submitted. |
| `[[partitions]]` | `name`, `class`, `cores`, `memory`, `max_wall`, `gpu` | Array of partition rows. `class` (`default-cpu`, `gpu`, `high-mem`, `debug`, `long`, `emergency`) is how skills pick, not by name. |
| `[filesystem]` | `home`, `scratch`, `project`, `quota` | Paths + whether `/scratch` exists. |
| `[network]` | `internet_from_login`, `internet_from_compute` | Booleans controlling ship strategy + in-job installs. |
| `[region]` | `region` (`mainland_china` / blank) | Downstream mirror defaults. |
| `[limits]` | see below | Safety ceilings consumed by `cluster_guardrail.py`. |
| `[[documentation]]` | `url`, `documents` | Array of every relevant docs sub-page (login, scheduler, partitions, modules, filesystem, network). Built by `/setup-cluster`'s docs crawl; the table is the spec, not a fallback. |
| `[[gotchas]]` | `symptom`, `cause`, `fix` | Harness-side issues not explicit in cluster docs (e.g., non-interactive ssh not sourcing `/etc/profile`; two `sbatch` binaries). |
| `[bootstrap]` | `one_time` (multi-line string) | Cluster-specific first-checkout quirks; idempotent. |
| `[sbatch]` | `single`, `array` (multi-line templates), `array_id_var`, `output_pattern` | Submission idioms. |
| `[commands]` | `squeue`, `sacct`, `sinfo`, `quota_command` | Scheduler-flavor + the optional read-only allocation/quota probe `/cluster-jobs` runs at setup. |
| `[notes]` | `text` | Anything else (group ownership, GPU exclusivity, egress). |

Language-specific tooling (`julia.provider`, `python.distribution`, …) does **not** belong here.

## The `[limits]` section (student safety)

Seeded by `/setup-cluster` from the probed `[[partitions]]` caps, then student-editable. Two tiers + path roots:

```toml
[limits.hard]            # exceed → /cluster-jobs refuses; student must lower to submit
max_walltime = "24:00:00"
max_nodes = 4
max_cpus = 256
max_array_size = 200

[limits.soft]            # exceed → warn + explicit confirm; may proceed
warn_walltime = "08:00:00"
warn_cpus = 64
unusual_partitions = ["gpu-large"]

[limits.paths]           # download/delete confined to these roots
allowed_roots = ["~/scratch", "~/results"]
```

`cluster_guardrail.py inspect` grades a job script against `[limits.hard]`/`[limits.soft]`; `check-path` enforces `[limits.paths].allowed_roots`. A profile with **no** `[limits]` is treated fail-closed.

## Full example

```toml
[identity]
name = "demohpc"
purpose = "teaching cluster"
maintainer = "hpc-help@example.edu"

[connection]
repo_path_remote = "/home/student07/hpc4-workdir"
[connection.ssh]
alias = "demohpc"
host = "login.demohpc.example.edu"
user = "student07"
identity_file = "~/.ssh/id_ed25519"
port = 22

[scheduler]
type = "slurm"
default_partition = "cpu"

[[partitions]]
name = "cpu"
class = "default-cpu"
cores = 64
memory = "256G"
max_wall = "24:00:00"
gpu = ""

[[partitions]]
name = "gpu-large"
class = "gpu"
cores = 32
memory = "512G"
max_wall = "12:00:00"
gpu = "a100:4"

[network]
internet_from_login = true
internet_from_compute = false

[region]
region = ""

[limits.hard]
max_walltime = "24:00:00"
max_nodes = 4
max_cpus = 256
max_array_size = 200
[limits.soft]
warn_walltime = "08:00:00"
warn_cpus = 64
unusual_partitions = ["gpu-large"]
[limits.paths]
allowed_roots = ["~/scratch", "~/results"]

[commands]
quota_command = "sshare -U -u student07"
```

## Picking the active profile

| Situation | Action |
|---|---|
| Single-cluster user | `ln -s <name>.toml .agents/skills/hpc4-domestic-submit/profiles/active.toml` once. |
| Multi-cluster user | `HARNESS_CLUSTER_PROFILE=<name>` per shell or in `.envrc`; env var wins over the symlink. |
| First-time user | Run `cluster_profile_setup.py inspect`; if needed, ask username/remote-dir/SSH-key preference once, then run `cluster_profile_setup.py create --activate`. |
| Profile contains secrets | `.gitignore` it locally; commit only public profiles. |

## Authoring a new profile

For manual authoring: first try `cluster_profile_setup.py create`. If the cluster is nonstandard, probe the cluster (`sinfo`, `scontrol show partition`, `sacctmgr show accounts`), write `.agents/skills/hpc4-domestic-submit/profiles/<name>.toml` following the tables above, activate it (`ln -s <name>.toml active.toml` or the env var), and test with a tiny job. Validate shape with `python .agents/skills/hpc4-domestic-submit/scripts/cluster_profile.py --field connection.ssh.alias --profile <name>.toml`.

## Cards in this folder

Profiles are optional and user/site-specific. Public profiles may be committed; private profiles should stay gitignored locally.
