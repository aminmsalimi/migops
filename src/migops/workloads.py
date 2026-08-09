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


def parse_compute_instances(output: str):
    # Parse current NVIDIA `nvidia-smi mig -lci` table output.

    instances = []

    for raw_line in output.splitlines():
        line = raw_line.strip()

        if not (line.startswith("|") and line.endswith("|")):
            continue

        body = line[1:-1].strip()

        if (
            not body
            or body.startswith("=")
            or body.lower().startswith("compute instances")
            or body.lower().startswith("gpu ")
            or body.lower().startswith("instance")
            or body.lower() == "id"
        ):
            continue

        parts = body.split()

        # Current H100:
        # 0  6  MIG  1g.24gb  7  0  0:2
        if len(parts) >= 7 and parts[2].upper() == "MIG":
            gpu = parts[0]
            gi = parts[1]
            profile = parts[3]
            profile_id = parts[4]
            ci = parts[5]
            placement = parts[6]

        # Older table forms may omit the literal MIG token.
        elif len(parts) >= 6:
            gpu = parts[0]
            gi = parts[1]
            profile = parts[2]
            profile_id = parts[3]
            ci = parts[4]
            placement = parts[5]

        else:
            continue

        if not (
            gpu.isdigit()
            and gi.isdigit()
            and profile_id.isdigit()
            and ci.isdigit()
        ):
            continue

        placement_start = None
        placement_size = None

        if ":" in placement:
            start_text, size_text = placement.split(":", 1)

            if start_text.isdigit():
                placement_start = start_text

            if size_text.isdigit():
                placement_size = size_text

        instances.append(
            ComputeInstance(
            gpu=gpu,
            gi_id=gi,
            ci_id=ci,
            profile=profile
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

def query_compute_instances():
    """
    Return current MIG Compute Instances.

    NVIDIA can return a non-zero exit status with:
    "No compute instances found: Not Found"

    That means zero CIs exist; it is not a fatal error.
    """

    try:
        output = run_nvidia_smi(
            ["mig", "-lci"]
        )
    except NvidiaSmiError as exc:
        message = str(exc).strip().lower()

        if (
            "no compute instances found" in message
            or "no compute instance found" in message
        ):
            return []

        raise

    return parse_compute_instances(output)

def _query_workloads_strict() -> list[GpuProcess]:
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

def _query_workloads_fallback():
    """
    Best-effort workload discovery that does not require MIG GI/CI listing.

    Some driver/permission combinations allow normal nvidia-smi process
    queries but deny `nvidia-smi mig -lci`. In that case MIGOps should still
    be able to report running GPU processes.
    """

    import csv
    import io
    import os
    from types import SimpleNamespace

    try:
        import pwd
    except ImportError:
        pwd = None

    # Build UUID -> physical GPU index mapping. `nvidia-smi -L` includes
    # both physical GPU UUIDs and MIG device UUIDs.
    uuid_to_gpu = {}
    physical_gpu_indexes = []

    gpu_csv = run_nvidia_smi(
        [
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ]
    )

    for row in csv.reader(io.StringIO(gpu_csv)):
        if len(row) < 2:
            continue

        index = row[0].strip()
        uuid = row[1].strip()
        physical_gpu_indexes.append(index)
        uuid_to_gpu[uuid] = index

    try:
        inventory = run_nvidia_smi(["-L"])
    except NvidiaSmiError:
        inventory = ""

    current_gpu = None

    for raw_line in inventory.splitlines():
        line = raw_line.strip()

        if line.startswith("GPU ") and "(UUID:" in line:
            try:
                index = line.split(":", 1)[0].split()[1]
                uuid = line.split("(UUID:", 1)[1].rstrip(")").strip()
            except (IndexError, ValueError):
                continue

            current_gpu = index
            uuid_to_gpu[uuid] = index
            continue

        if (
            current_gpu is not None
            and line.startswith("MIG ")
            and "(UUID:" in line
        ):
            try:
                mig_uuid = (
                    line.split("(UUID:", 1)[1]
                    .rstrip(")")
                    .strip()
                )
            except (IndexError, ValueError):
                continue

            uuid_to_gpu[mig_uuid] = current_gpu

    output = run_nvidia_smi(
        [
            "--query-compute-apps=pid,process_name,used_memory,gpu_uuid",
            "--format=csv,noheader,nounits",
        ]
    )

    if not output.strip():
        return []

    workloads = []

    for row in csv.reader(io.StringIO(output)):
        if len(row) < 4:
            continue

        pid_text = row[0].strip()
        process_name = row[1].strip()
        memory_text = row[2].strip()
        gpu_uuid = row[3].strip()

        try:
            pid = int(pid_text)
        except ValueError:
            continue

        try:
            memory_mib = int(float(memory_text))
        except ValueError:
            memory_mib = None

        gpu = uuid_to_gpu.get(gpu_uuid)

        # On a one-GPU host, an unmapped MIG UUID still belongs to GPU 0
        # (or whatever the only physical index is).
        if gpu is None and len(physical_gpu_indexes) == 1:
            gpu = physical_gpu_indexes[0]

        username = None

        if pwd is not None:
            try:
                uid = os.stat(f"/proc/{pid}").st_uid
                username = pwd.getpwuid(uid).pw_name
            except (FileNotFoundError, KeyError, PermissionError, OSError):
                username = None

        workloads.append(
            SimpleNamespace(
                gpu=gpu or "unknown",
                pid=pid,
                process_name=process_name,
                memory_mib=memory_mib,
                username=username,
                gi=None,
                ci=None,
                gpu_uuid=gpu_uuid,
            )
        )

    return workloads


def query_workloads():
    """
    Return active NVIDIA workloads.

    Use the normal MIG-aware implementation first. If that implementation
    fails only because MIG instance inspection is permission-restricted,
    fall back to ordinary nvidia-smi compute-process discovery.
    """

    try:
        return _query_workloads_strict()
    except NvidiaSmiError as exc:
        message = str(exc).strip().lower()

        permission_limited = (
            "insufficient permissions" in message
            or "permission denied" in message
        )

        if not permission_limited:
            raise

        try:
            return _query_workloads_fallback()
        except NvidiaSmiError:
            raise exc
