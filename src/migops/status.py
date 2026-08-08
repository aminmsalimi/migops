"""GPU and MIG status reporting."""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field

from migops.nvidia import NvidiaSmiError, run_nvidia_smi


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


def query_gpus() -> list[GPU]:
    """Query GPU inventory and MIG mode."""

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
    """
    Return MIG devices grouped by their parent GPU index.

    Uses `nvidia-smi -L`.
    """

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
    """Collect GPU and MIG inventory."""

    gpus = query_gpus()

    try:
        mig_devices = query_mig_devices()
    except NvidiaSmiError:
        mig_devices = {}

    for gpu in gpus:
        gpu.mig_devices = mig_devices.get(gpu.index, [])

    return gpus


def print_status() -> int:
    """Print a human-readable GPU/MIG status report."""

    print()
    print("MIGOps Status")
    print("=============")
    print()

    try:
        gpus = collect_status()
    except NvidiaSmiError as exc:
        print("[FAIL] Unable to query NVIDIA GPUs")
        print()
        print(str(exc))
        return 1

    if not gpus:
        print("[FAIL] No NVIDIA GPUs detected.")
        return 1

    print(f"GPUs detected: {len(gpus)}")

    driver_versions = sorted(
        {gpu.driver_version for gpu in gpus}
    )

    if driver_versions:
        print(f"NVIDIA Driver: {', '.join(driver_versions)}")

    for gpu in gpus:
        print()
        print("-" * 60)
        print(f"GPU {gpu.index}: {gpu.name}")
        print("-" * 60)

        print(f"UUID:       {gpu.uuid}")
        print(f"PCI Bus:    {gpu.pci_bus_id}")
        print(f"MIG Mode:   {gpu.mig_mode}")

        if gpu.mig_mode.lower() == "enabled":

            if gpu.mig_devices:
                print()
                print("MIG Devices")
                print()

                for device in gpu.mig_devices:
                    print(
                        f"  Device {device.device_id:<3} "
                        f"{device.profile:<15} "
                        f"{device.uuid}"
                    )

                print()
                print(
                    f"MIG instances detected: "
                    f"{len(gpu.mig_devices)}"
                )

            else:
                print()
                print(
                    "MIG mode is enabled, but no MIG devices "
                    "were detected."
                )

        elif gpu.mig_mode.lower() == "disabled":
            print()
            print("MIG is currently disabled on this GPU.")

        else:
            print()
            print(
                "MIG state could not be determined "
                f"({gpu.mig_mode})."
            )

    print()

    return 0