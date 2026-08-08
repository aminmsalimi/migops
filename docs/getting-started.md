# Getting Started

MIGOps is intended for Linux GPU hosts with an NVIDIA driver, `nvidia-smi`, and a MIG-capable GPU.

Start with read-only inspection:

```bash
migops status
migops profiles gpu 0
migops users gpu 0
```

For an equal split recommendation:

```bash
migops split gpu 0 4
```

Before any real desired-state change, preview it:

```bash
migops apply config.yaml --dry-run
```

A real desired-state apply requires `--yes` and may require elevated permissions:

```bash
sudo migops apply config.yaml
```

Use `--force` only when you deliberately intend to override MIGOps workload protection.
