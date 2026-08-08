# Safety Model

MIG reconfiguration can interrupt GPU workloads. MIGOps therefore treats destructive or disruptive changes conservatively.

For desired-state apply, MIGOps:

1. validates the configuration
2. compares desired and actual state
3. checks active workloads before disruptive operations
4. creates a pre-change snapshot for real applies
5. removes Compute Instances before GPU Instances
6. applies the requested state
7. verifies the final state

`--dry-run` never changes the GPU and reports active workloads as warnings.

Real desired-state changes ask for confirmation by default; `--yes` skips the prompt for automation.

`--force` bypasses MIGOps workload protection. It does not guarantee that the NVIDIA driver will accept the operation.

MIG mode transitions can behave differently across GPU generations and driver versions. The NVIDIA driver remains the final authority.

## Direct MIG mode commands

`migops enable gpu <GPU>` and `migops disable gpu <GPU>` use workload detection as an advisory warning rather than a hard MIGOps block. They still require explicit confirmation, and the NVIDIA driver remains the final authority. `--force` is not required for these direct mode commands.
