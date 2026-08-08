"""GPU and MIG workload discovery."""

from __future__ import annotations

import json
import os
import platform
import re
from dataclasses import asdict, dataclass

from migops.nvidia import NvidiaSmiError, run_nvidia_smi


@dataclass
class ComputeInstance:
    gpu: str
    gi_id: str
    ci_id: str
    profile: str


@dataclass
class GpuProcess:
    gpu: str
    gi_id: str | None
    ci_id: str | None
    pid: int
    process_type: str
    process_name: str
    memory_mib: int | None
    username: str | None = None
    mig_profile: str | None = None


# Modern nvidia-smi process table with MIG GI/CI columns.
MIG_PROCESS_RE = re.compile(
    r"^\|\s*"
    r"(?P<gpu>\d+)\s+"
    r"(?P<gi>\d+|N/A)\s+"
    r"(?P<ci>\d+|N/A)\s+"
    r"(?P<pid>\d+)\s+"
    r"(?P<type>C\+G|M\+C|[CGMO])\s+"
    r"(?P<name>.+?)\s+"
    r"(?P<memory>\d+\s*MiB|N/A)\s*"
    r"\|$"
)


# Some driver/output formats without GI/CI columns.
LEGACY_PROCESS_RE = re.compile(
    r"^\|\s*"
    r"(?P<gpu>\d+)\s+"
    r"(?P<pid>\d+)\s+"
    r"(?P<type>C\+G|M\+C|[CGMO])\s+"
    r"(?P<name>.+?)\s+"
    r"(?P<memory>\d+\s*MiB|N/A)\s*"
    r"\|$"
)


CI_RE = re.compile(
    r"^\|\s*"
    r"(?P<gpu>\d+)\s+"
    r"(?P<gi>\d+)\s+"
    r"MIG\s+"
    r"(?P<profile>\S+)\s+"
    r"(?P<profile_id>\d+\*?)\s+"
    r"(?P<ci>\d+)\s*"
    r"\|$"
)


def parse_memory(value: str) -> int | None:
    """Convert an nvidia-smi memory field such as 153MiB to MiB."""

    value = value.strip()

    if value == "N/A":
        return None

    match = re.search(r"(\d+)", value)

    if not match:
        return None

    return int(match.group(1))


def get_process_username(pid: int) -> str | None:
    """
    Resolve a PID to its Linux username.

    Returns None on non-Linux systems or when the process cannot be resolved.
    """

    if platform.system() != "Linux":
        return None

    try:
        import pwd

        process_stat = os.stat(f"/proc/{pid}")
        return pwd.getpwuid(process_stat.st_uid).pw_name

    except (ImportError, KeyError, OSError):
        return None


def parse_compute_instances(output: str) -> list[ComputeInstance]:
    """Parse `nvidia-smi mig -lci` output."""

    instances: list[ComputeInstance] = []

    for line in output.splitlines():
        match = CI_RE.match(line)

        if not match:
            continue

        instances.append(
            ComputeInstance(
                gpu=match.group("gpu"),
                gi_id=match.group("gi"),
                ci_id=match.group("ci"),
                profile=match.group("profile"),
            )
        )

    return instances


def parse_processes(output: str) -> list[GpuProcess]:
    """Parse process rows from regular `nvidia-smi` output."""

    processes: list[GpuProcess] = []

    for line in output.splitlines():

        match = MIG_PROCESS_RE.match(line)

        if match:
            gi = match.group("gi")
            ci = match.group("ci")

            processes.append(
                GpuProcess(
                    gpu=match.group("gpu"),
                    gi_id=None if gi == "N/A" else gi,
                    ci_id=None if ci == "N/A" else ci,
                    pid=int(match.group("pid")),
                    process_type=match.group("type"),
                    process_name=match.group("name").strip(),
                    memory_mib=parse_memory(match.group("memory")),
                )
            )

            continue

        match = LEGACY_PROCESS_RE.match(line)

        if match:
            processes.append(
                GpuProcess(
                    gpu=match.group("gpu"),
                    gi_id=None,
                    ci_id=None,
                    pid=int(match.group("pid")),
                    process_type=match.group("type"),
                    process_name=match.group("name").strip(),
                    memory_mib=parse_memory(match.group("memory")),
                )
            )

    return processes


def query_compute_instances() -> list[ComputeInstance]:
    """
    Query current MIG compute instances.

    Returns an empty list when MIG compute instances are not available.
    """

    try:
        output = run_nvidia_smi(["mig", "-lci"])
    except NvidiaSmiError:
        return []

    return parse_compute_instances(output)


def query_workloads() -> list[GpuProcess]:
    """Return active NVIDIA GPU processes with MIG information where available."""

    output = run_nvidia_smi([])

    processes = parse_processes(output)
    compute_instances = query_compute_instances()

    ci_map = {
        (instance.gpu, instance.gi_id, instance.ci_id): instance.profile
        for instance in compute_instances
    }

    for process in processes:

        process.username = get_process_username(process.pid)

        if process.gi_id is not None and process.ci_id is not None:
            process.mig_profile = ci_map.get(
                (
                    process.gpu,
                    process.gi_id,
                    process.ci_id,
                )
            )

    return processes


def print_users(
    gpu: str | None = None,
    json_output: bool = False,
) -> int:
    """Print processes currently using NVIDIA GPUs."""

    try:
        processes = query_workloads()

    except NvidiaSmiError as exc:

        if json_output:
            print(
                json.dumps(
                    {
                        "error": str(exc),
                        "processes": [],
                    },
                    indent=2,
                )
            )

        else:
            print()
            print("MIGOps GPU Users")
            print("================")
            print()
            print("[FAIL] Unable to query GPU workloads")
            print()
            print(str(exc))

        return 1

    if gpu is not None:
        processes = [
            process
            for process in processes
            if process.gpu == gpu
        ]

    if json_output:
        print(
            json.dumps(
                {
                    "processes": [
                        asdict(process)
                        for process in processes
                    ]
                },
                indent=2,
            )
        )

        return 0

    print()
    print("MIGOps GPU Users")
    print("================")
    print()

    if not processes:
        print("No active NVIDIA GPU compute processes detected.")
        print()
        return 0

    print(
        f"{'GPU':<5}"
        f"{'GI':<5}"
        f"{'CI':<5}"
        f"{'PROFILE':<18}"
        f"{'PID':<9}"
        f"{'USER':<16}"
        f"{'MEMORY':<12}"
        f"PROCESS"
    )

    print("-" * 100)

    for process in processes:

        gi = process.gi_id or "-"
        ci = process.ci_id or "-"
        profile = process.mig_profile or "-"
        username = process.username or "unknown"

        memory = (
            f"{process.memory_mib} MiB"
            if process.memory_mib is not None
            else "N/A"
        )

        print(
            f"{process.gpu:<5}"
            f"{gi:<5}"
            f"{ci:<5}"
            f"{profile:<18}"
            f"{process.pid:<9}"
            f"{username:<16}"
            f"{memory:<12}"
            f"{process.process_name}"
        )

    print()
    print(f"Active GPU processes: {len(processes)}")
    print()

    return 0