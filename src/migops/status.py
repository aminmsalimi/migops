"""Combined system, NVIDIA GPU, MIG, and workload status reporting."""

from __future__ import annotations

import csv
import io
import platform
import re
from dataclasses import dataclass, field

from migops.nvidia import (
    NvidiaSmiError,
    find_nvidia_smi,
    run_nvidia_smi,
)


@dataclass
class MigDevice:
    profile: str
    device_id: str
    uuid: str


@dataclass
class GPU:
    index: str
    name: str
    uuid: str
    driver_version: str
    pci_bus_id: str
    mig_mode: str
    mig_devices: list[MigDevice] = field(default_factory=list)


GPU_LINE_RE = re.compile(
    r"^GPU\s+(?P<index>\d+):\s+"
    r"(?P<name>.+?)\s+"
    r"\(UUID:\s+(?P<uuid>[^)]+)\)"
)

MIG_LINE_RE = re.compile(
    r"^\s*MIG\s+"
    r"(?P<profile>.+?)\s+"
    r"Device\s+(?P<device>\d+):\s+"
    r"\(UUID:\s+(?P<uuid>[^)]+)\)"
)


def mig_supported_from_mode(mig_mode: str) -> bool:
    """Return True when the driver reports a normal MIG mode state."""

    return mig_mode.strip().lower() in {
        "enabled",
        "disabled",
    }


def query_gpus() -> list[GPU]:
    """Query physical NVIDIA GPU inventory and current MIG mode."""

    output = run_nvidia_smi(
        [
            "--query-gpu=index,name,uuid,driver_version,pci.bus_id,mig.mode.current",
            "--format=csv,noheader,nounits",
        ]
    )

    reader = csv.reader(io.StringIO(output))
    gpus: list[GPU] = []

    for row in reader:
        if len(row) != 6:
            continue

        values = [value.strip() for value in row]

        gpus.append(
            GPU(
                index=values[0],
                name=values[1],
                uuid=values[2],
                driver_version=values[3],
                pci_bus_id=values[4],
                mig_mode=values[5],
            )
        )

    return gpus


def query_mig_devices() -> dict[str, list[MigDevice]]:
    """Return MIG devices grouped by physical GPU index."""

    output = run_nvidia_smi(["-L"])

    devices: dict[str, list[MigDevice]] = {}
    current_gpu: str | None = None

    for line in output.splitlines():
        gpu_match = GPU_LINE_RE.match(line)

        if gpu_match:
            current_gpu = gpu_match.group("index")
            devices.setdefault(current_gpu, [])
            continue

        mig_match = MIG_LINE_RE.match(line)

        if mig_match and current_gpu is not None:
            devices[current_gpu].append(
                MigDevice(
                    profile=mig_match.group("profile").strip(),
                    device_id=mig_match.group("device"),
                    uuid=mig_match.group("uuid").strip(),
                )
            )

    return devices


def collect_status() -> list[GPU]:
    """Collect physical GPU and MIG-device inventory."""

    gpus = query_gpus()

    try:
        mig_devices = query_mig_devices()
    except NvidiaSmiError:
        mig_devices = {}

    for gpu in gpus:
        gpu.mig_devices = mig_devices.get(gpu.index, [])

    return gpus


def print_check(status: str, message: str) -> None:
    print(f"[{status}] {message}")


def print_status() -> int:
    """Print one combined system, NVIDIA, MIG, and workload status report."""

    print()
    print("MIGOps Status")
    print("=============")
    print()

    print("System")
    print("------")

    system = platform.system()

    if system == "Linux":
        print_check("PASS", "Operating system: Linux")
    else:
        print_check(
            "WARN",
            f"Operating system: {system} "
            "(real MIG operations are intended for Linux GPU hosts)",
        )

    print_check("PASS", f"Python: {platform.python_version()}")

    print()
    print("NVIDIA")
    print("------")

    nvidia_smi = find_nvidia_smi()

    if not nvidia_smi:
        print_check("FAIL", "nvidia-smi not found")
        print()
        print(
            "Install a supported NVIDIA driver and ensure "
            "nvidia-smi is available in PATH."
        )
        return 1

    print_check("PASS", f"nvidia-smi found: {nvidia_smi}")

    try:
        gpus = collect_status()
    except NvidiaSmiError as exc:
        print_check("FAIL", "Unable to query NVIDIA GPUs")
        print()
        print(str(exc))
        return 1

    if not gpus:
        print_check("FAIL", "No NVIDIA GPUs detected")
        return 1

    driver_versions = sorted(
        {gpu.driver_version for gpu in gpus}
    )

    print_check("PASS", f"NVIDIA GPUs detected: {len(gpus)}")

    if driver_versions:
        print_check(
            "PASS",
            f"NVIDIA driver: {', '.join(driver_versions)}",
        )

    try:
        from migops.workloads import (
            query_compute_instances,
            query_workloads,
        )

        compute_instances = query_compute_instances()

        try:
            workloads = query_workloads()
        except NvidiaSmiError:
            workloads = []

    except Exception:
        compute_instances = []
        workloads = []

    print()

    warning_count = 0
    mig_capable_count = 0

    for gpu in gpus:
        print(f"GPU {gpu.index}")
        print("-" * 60)
        print(f"Model:      {gpu.name}")
        print(f"UUID:       {gpu.uuid}")
        print(f"PCI Bus:    {gpu.pci_bus_id}")
        print(f"MIG Mode:   {gpu.mig_mode}")
        print()

        if mig_supported_from_mode(gpu.mig_mode):
            mig_capable_count += 1
            print_check("PASS", "MIG capability detected")
        else:
            warning_count += 1
            print_check("WARN", "MIG capability not detected")

        mode = gpu.mig_mode.strip().lower()

        if mode == "enabled":
            print_check("PASS", "MIG mode enabled")
        elif mode == "disabled":
            warning_count += 1
            print_check("WARN", "MIG supported but currently disabled")
        else:
            print_check("INFO", f"MIG state: {gpu.mig_mode}")

        if mode == "enabled":
            if gpu.mig_devices:
                print_check(
                    "PASS",
                    f"MIG devices detected: {len(gpu.mig_devices)}",
                )
            else:
                print_check(
                    "INFO",
                    "MIG enabled but no MIG devices created",
                )

            gpu_compute_instances = [
                instance
                for instance in compute_instances
                if instance.gpu == gpu.index
            ]

            if gpu_compute_instances:
                print_check(
                    "PASS",
                    "Compute Instances detected: "
                    f"{len(gpu_compute_instances)}",
                )
            else:
                print_check(
                    "INFO",
                    "No MIG Compute Instances detected",
                )

        gpu_workloads = [
            process
            for process in workloads
            if process.gpu == gpu.index
        ]

        if gpu_workloads:
            warning_count += 1
            print_check(
                "WARN",
                f"Active GPU processes: {len(gpu_workloads)}",
            )

            for process in gpu_workloads:
                user = process.username or "unknown"
                memory = (
                    f"{process.memory_mib} MiB"
                    if process.memory_mib is not None
                    else "N/A"
                )

                print(
                    f"       PID {process.pid} | "
                    f"{user} | {memory} | "
                    f"{process.process_name}"
                )
        else:
            print_check("PASS", "No active GPU processes detected")

        print()

    print("Summary")
    print("-------")
    print(f"Total NVIDIA GPUs: {len(gpus)}")
    print(f"MIG-capable GPUs:  {mig_capable_count}")
    print(f"Warnings:          {warning_count}")
    print()

    if mig_capable_count == 0:
        print("MIGOps readiness: MIG NOT AVAILABLE")
        return 1

    if warning_count:
        print("MIGOps readiness: READY WITH WARNINGS")
    else:
        print("MIGOps readiness: READY")

    return 0
