# MIGOps

MIGOps is a Python CLI for safe NVIDIA Multi-Instance GPU (MIG) operations on Linux GPU hosts.

It complements NVIDIA's native MIG tooling with workload-aware safety checks, smart partition recommendations, desired-state YAML, dry-run planning, drift detection, snapshots, restore, and verified apply workflows.

## Features

- GPU and MIG status
- MIG profile discovery
- Active GPU/MIG workload discovery
- MIG diagnostics
- Smart equal-memory split recommendations
- Easy MIG creation (`GI + default CI`)
- Safe MIG destruction (`CI -> GI`)
- MIG mode enable/disable
- Advanced GI and CI operations
- YAML desired-state configuration
- Configuration validation
- Desired-vs-actual drift detection
- Change planning and risk reporting
- Configuration snapshots
- Restore workflow
- Safe apply with automatic pre-change snapshot
- JSON output for automation-oriented commands
- Automated unit tests and GitHub Actions CI

## Requirements

- Linux for real MIG operations
- Python 3.9+
- NVIDIA driver
- `nvidia-smi`
- NVIDIA MIG-capable GPU for MIG operations

Development and parser tests can run without an NVIDIA GPU.

## Installation

```bash
git clone https://github.com/aminmsalimi/migops.git
cd migops

python3 -m venv .venv
source .venv/bin/activate

python -m pip install -e .
```

## Quick Start

Inspect the host:

```bash
migops doctor
migops status
migops profiles
migops users
```

Smart split planning:

```bash
migops split --gpu 0 --instances 2
migops split --gpu 0 --instances 4
```

Preview actually applying a smart split:

```bash
migops split --gpu 0 --instances 4 --apply --dry-run
```

Execute it deliberately:

```bash
sudo migops split --gpu 0 --instances 4 --apply --yes
```

MIGOps selects the best identical profile reported by the installed NVIDIA driver; it does not invent unsupported memory sizes.

## Easy Lifecycle Commands

Enable or disable MIG mode:

```bash
sudo migops mode enable --gpu 0
sudo migops mode disable --gpu 0
```

Create complete usable MIG instances (GI + default CI):

```bash
sudo migops create --gpu 0 --profile 3g.40gb
sudo migops create --gpu 0 --profile 3g.40gb --count 2
```

Safely destroy MIG instances:

```bash
sudo migops destroy --gpu 0 --gi 2
sudo migops destroy --gpu 0 --all
```

Use `--dry-run` before destructive operations whenever possible.

## Desired-State Configuration

Example:

```yaml
version: 1

gpus:
  - gpu: "0"
    mig_enabled: true

    instances:
      - profile: "3g.40gb"
        count: 1

      - profile: "2g.20gb"
        count: 1

      - profile: "1g.10gb"
        count: 2
```

Validate, diff, and plan:

```bash
migops validate examples/h100-mixed.yaml
migops diff examples/h100-mixed.yaml
migops plan examples/h100-mixed.yaml
```

Preview an apply:

```bash
sudo migops apply examples/h100-mixed.yaml --dry-run
```

Execute:

```bash
sudo migops apply examples/h100-mixed.yaml --yes
```

Before a real apply, MIGOps creates a snapshot of the current state and blocks destructive changes when active GPU workloads are detected unless `--force` is explicitly used.

## Snapshots and Restore

Create a snapshot:

```bash
migops snapshot
```

Or choose an output path:

```bash
migops snapshot --gpu 0 --output before-maintenance.yaml
```

Preview restore:

```bash
sudo migops restore before-maintenance.yaml --dry-run
```

Execute restore:

```bash
sudo migops restore before-maintenance.yaml --yes
```

## Advanced GI / CI Commands

The simple workflow is recommended for normal administration. Advanced commands remain available when direct control is required:

```bash
migops gi list --gpu 0
migops gi create --gpu 0 --profile 3g.40gb --with-ci
migops gi delete --gpu 0 --gi 2

migops ci list --gpu 0
migops ci create --gpu 0 --gi 2 --profile 0
migops ci delete --gpu 0 --gi 2 --ci 0
```

## Safety Model

MIGOps is intentionally conservative:

- destructive changes are workload-checked
- `--dry-run` previews operations
- desired-state `apply` requires `--yes` for real changes
- a snapshot is taken before a real desired-state apply
- CIs are removed before GIs
- final state is verified after apply
- NVIDIA's driver remains the final authority on supported geometry and placement

## Tests

```bash
python -m unittest discover -s tests -v
```

GitHub Actions runs the test suite automatically on pushes and pull requests.

## Status

MIGOps is under active development.
