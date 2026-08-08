"""MIGOps environment and MIG diagnostic checks."""

from __future__ import annotations

import platform

from migops.nvidia import (
    NvidiaSmiError,
    find_nvidia_smi,
)
from migops.status import (
    GPU,
    query_gpus,
    query_mig_devices,
)
from migops.workloads import (
    query_compute_instances,
    query_workloads,
)


def print_check(status: str, message: str) -> None:
    """Print a diagnostic result."""
    print(f"[{status}] {message}")


def mig_supported_from_mode(mig_mode: str) -> bool:
    """
    Determine whether the GPU appears to support MIG.

    NVIDIA GPUs that support MIG normally report the current MIG mode
    as Enabled or Disabled. Non-MIG GPUs generally report N/A or an
    unsupported value.
    """

    normalized = mig_mode.strip().lower()

    return normalized in {
        "enabled",
        "disabled",
    }


def command_doctor() -> int:
    """Run MIGOps environment and MIG diagnostics."""

    print()
    print("MIGOps Diagnostic")
    print("=================")
    print()

    # ---------------------------------------------------------
    # System
    # ---------------------------------------------------------

    print("System")
    print("------")

    system = platform.system()

    if system == "Linux":
        print_check("PASS", "Operating system: Linux")
    else:
        print_check(
            "WARN",
            f"Operating system: {system} "
            "(MIG operations require a supported Linux environment)",
        )

    print_check(
        "PASS",
        f"Python: {platform.python_version()}",
    )

    print()

    # ---------------------------------------------------------
    # NVIDIA tooling
    # ---------------------------------------------------------

    print("NVIDIA")
    print("------")

    nvidia_smi = find_nvidia_smi()

    if not nvidia_smi:
        print_check(
            "FAIL",
            "nvidia-smi not found",
        )

        print()
        print(
            "Install a supported NVIDIA driver and ensure "
            "nvidia-smi is available in PATH."
        )

        return 1

    print_check(
        "PASS",
        f"nvidia-smi found: {nvidia_smi}",
    )

    try:
        gpus = query_gpus()

    except NvidiaSmiError as exc:
        print_check(
            "FAIL",
            "Unable to query NVIDIA GPUs",
        )

        print()
        print(str(exc))

        return 1

    if not gpus:
        print_check(
            "FAIL",
            "No NVIDIA GPUs detected",
        )

        return 1

    print_check(
        "PASS",
        f"NVIDIA GPUs detected: {len(gpus)}",
    )

    driver_versions = sorted(
        {gpu.driver_version for gpu in gpus}
    )

    if driver_versions:
        print_check(
            "PASS",
            f"NVIDIA driver: {', '.join(driver_versions)}",
        )

    # ---------------------------------------------------------
    # Gather optional MIG information
    # ---------------------------------------------------------

    try:
        mig_devices = query_mig_devices()
    except NvidiaSmiError:
        mig_devices = {}

    compute_instances = query_compute_instances()

    try:
        workloads = query_workloads()
    except NvidiaSmiError:
        workloads = []

    print()

    mig_capable_count = 0
    warning_count = 0

    # ---------------------------------------------------------
    # Individual GPUs
    # ---------------------------------------------------------

    for gpu in gpus:

        print(f"GPU {gpu.index}")
        print("-" * 60)

        print(f"Model:      {gpu.name}")
        print(f"UUID:       {gpu.uuid}")
        print(f"PCI Bus:    {gpu.pci_bus_id}")
        print(f"MIG Mode:   {gpu.mig_mode}")

        print()

        mig_supported = mig_supported_from_mode(
            gpu.mig_mode
        )

        if mig_supported:
            mig_capable_count += 1

            print_check(
                "PASS",
                "MIG capability detected",
            )

        else:
            print_check(
                "WARN",
                "MIG capability not detected",
            )

            warning_count += 1

        if gpu.mig_mode.strip().lower() == "enabled":
            print_check(
                "PASS",
                "MIG mode enabled",
            )

        elif gpu.mig_mode.strip().lower() == "disabled":
            print_check(
                "WARN",
                "MIG supported but currently disabled",
            )

            warning_count += 1

        else:
            print_check(
                "INFO",
                f"MIG state: {gpu.mig_mode}",
            )

        gpu_mig_devices = mig_devices.get(
            gpu.index,
            [],
        )

        gpu_compute_instances = [
            instance
            for instance in compute_instances
            if instance.gpu == gpu.index
        ]

        gpu_workloads = [
            process
            for process in workloads
            if process.gpu == gpu.index
        ]

        if gpu.mig_mode.strip().lower() == "enabled":

            if gpu_mig_devices:
                print_check(
                    "PASS",
                    (
                        "MIG devices detected: "
                        f"{len(gpu_mig_devices)}"
                    ),
                )
            else:
                print_check(
                    "INFO",
                    "MIG enabled but no MIG devices created",
                )

            if gpu_compute_instances:
                print_check(
                    "PASS",
                    (
                        "Compute instances detected: "
                        f"{len(gpu_compute_instances)}"
                    ),
                )
            else:
                print_check(
                    "INFO",
                    "No MIG compute instances detected",
                )

        if gpu_workloads:
            print_check(
                "WARN",
                (
                    "Active GPU processes: "
                    f"{len(gpu_workloads)}"
                ),
            )

            warning_count += 1

        else:
            print_check(
                "PASS",
                "No active GPU processes detected",
            )

        print()

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print("Summary")
    print("-------")

    print(
        f"Total NVIDIA GPUs:       {len(gpus)}"
    )

    print(
        f"MIG-capable GPUs:        {mig_capable_count}"
    )

    print(
        f"Warnings:                {warning_count}"
    )

    print()

    if mig_capable_count == 0:
        print("MIGOps readiness: MIG NOT AVAILABLE")
        return 1

    if warning_count:
        print("MIGOps readiness: READY WITH WARNINGS")
        return 0

    print("MIGOps readiness: READY")

    return 0