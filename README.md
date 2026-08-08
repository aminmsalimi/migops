# MIGOps

MIGOps is a Python CLI for safe NVIDIA Multi-Instance GPU (MIG) operations on Linux GPU hosts.

It complements NVIDIA's native MIG tooling with workload-aware safety checks, smart partition recommendations, configuration planning, snapshots, drift detection, and safe apply workflows.

## Features

- GPU and MIG status
- MIG profile discovery
- Active GPU/MIG workload detection
- Smart equal-memory MIG splitting
- Easy MIG creation and destruction
- MIG mode enable/disable
- Advanced GI and CI management
- YAML desired-state configuration
- Validate, diff, and plan workflows
- Snapshots and restore
- Safe apply with workload checks and dry-run
- JSON output for automation
- Automated tests and GitHub Actions CI

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

Inspect the system:

```bash
migops status
migops profiles
migops users
```

Recommend an equal MIG split:

```bash
migops split gpu 0 4
```

Preview applying it:

```bash
sudo migops split gpu 0 4 --apply --dry-run
```

Apply it:

```bash
sudo migops split gpu 0 4 --apply --yes
```

MIGOps selects from MIG profiles reported by the installed NVIDIA driver rather than inventing unsupported partition sizes.

## MIG Management

Enable or disable MIG mode:

```bash
sudo migops enable gpu 0
sudo migops disable gpu 0
```

Create complete MIG instances:

```bash
sudo migops create gpu 0 3g.40gb
sudo migops create gpu 0 3g.40gb --count 2
```

Destroy them safely:

```bash
sudo migops destroy gpu 0 --gi 2
sudo migops destroy gpu 0 --all
```

Advanced GI and CI commands are also available through:

```bash
migops gi --help
migops ci --help
```

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
```

Validate, compare, and plan:

```bash
migops validate config.yaml
migops diff config.yaml
migops plan config.yaml
```

Preview and apply:

```bash
sudo migops apply config.yaml --dry-run
sudo migops apply config.yaml
```

## Snapshots

Save the current MIG configuration:

```bash
migops snapshot
```

Restore a previous snapshot:

```bash
sudo migops restore snapshot.yaml --dry-run
sudo migops restore snapshot.yaml
```

## Safety

MIGOps is intentionally conservative:

- detects active GPU workloads before destructive changes
- supports `--dry-run`
- requires `--yes` for desired-state changes
- creates a snapshot before real apply operations
- removes CIs before GIs
- verifies the final configuration after apply
- leaves final hardware validation to the NVIDIA driver

## Tests

```bash
python -m unittest discover -s tests -v
```

GitHub Actions runs the test suite automatically on pushes and pull requests.


## Status

MIGOps v0.1.0 is an early alpha release.

Core workflows are implemented and covered by automated tests. Real-world validation across different NVIDIA MIG-capable GPU generations and driver versions is still ongoing.