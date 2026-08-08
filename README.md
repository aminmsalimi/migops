# MIGOps

MIGOps is a Linux CLI for safe operations, planning, validation, and troubleshooting of NVIDIA Multi-Instance GPU (MIG) environments.

## Goals

MIGOps is designed to complement NVIDIA's native MIG tooling by providing administrator-focused workflows such as:

- GPU and MIG inventory
- MIG profile discovery
- Active workload detection
- Configuration planning
- Dry-run operations
- Desired vs. actual configuration comparison
- Configuration snapshots
- Pre-flight safety checks
- MIG diagnostics and troubleshooting

## Planned Commands

```bash
migops status
migops profiles
migops users
migops plan config.yaml
migops diff config.yaml
migops apply config.yaml --dry-run
migops snapshot
migops doctor
```

## Requirements

- Linux
- Python 3.9+
- NVIDIA driver
- `nvidia-smi`
- NVIDIA MIG-capable GPU for MIG operations

## Status

MIGOps is currently under active development.