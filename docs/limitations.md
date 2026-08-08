# Current Limitations

MIGOps is still an early-stage project.

Important current limits:

- Real MIG operations require a supported Linux/NVIDIA environment.
- Human-readable `nvidia-smi` output can vary across driver and GPU generations; real-hardware testing remains important.
- Mixed-profile validation checks individual profile limits, while the NVIDIA driver performs final placement validation.
- Snapshot/restore currently preserves GI profile counts and recreates one default CI per GI. It does not preserve advanced custom CI sub-partitioning.
- `--force` bypasses MIGOps workload protection but cannot override NVIDIA driver restrictions.
- Kubernetes, GPU Operator awareness, DCGM integration, multi-node orchestration, and time-slicing management are not part of the first release.
