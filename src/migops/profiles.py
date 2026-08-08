"""MIG GPU instance profile discovery."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from migops.nvidia import NvidiaSmiError, run_nvidia_smi


@dataclass
class MigProfile:
    gpu: str
    name: str
    profile_id: str
    free: int
    total: int
    memory_gib: float | None


PROFILE_RE = re.compile(
    r"^\|\s*"
    r"(?P<gpu>\d+)\s+"
    r"MIG\s+(?P<name>\S+)\s+"
    r"(?P<id>\d+)\s+"
    r"(?P<free>\d+)/(?P<total>\d+)\s+"
    r"(?P<memory>\d+(?:\.\d+)?)"
)


def parse_profiles(output: str) -> list[MigProfile]:
    """Parse `nvidia-smi mig -lgip` output."""

    profiles: list[MigProfile] = []

    for line in output.splitlines():
        match = PROFILE_RE.match(line)

        if not match:
            continue

        profiles.append(
            MigProfile(
                gpu=match.group("gpu"),
                name=match.group("name"),
                profile_id=match.group("id"),
                free=int(match.group("free")),
                total=int(match.group("total")),
                memory_gib=float(match.group("memory")),
            )
        )

    return profiles


def query_profiles(gpu: str | None = None) -> list[MigProfile]:
    """Query supported MIG GPU instance profiles."""

    arguments = ["mig", "-lgip"]

    if gpu is not None:
        arguments.extend(["-i", gpu])

    output = run_nvidia_smi(arguments)

    return parse_profiles(output)


def print_profiles(
    gpu: str | None = None,
    json_output: bool = False,
) -> int:
    """Print supported MIG profiles."""

    try:
        profiles = query_profiles(gpu)
    except NvidiaSmiError as exc:
        if json_output:
            print(
                json.dumps(
                    {
                        "error": str(exc),
                        "profiles": [],
                    },
                    indent=2,
                )
            )
        else:
            print()
            print("MIGOps Profiles")
            print("===============")
            print()
            print("[FAIL] Unable to query MIG profiles")
            print()
            print(str(exc))

        return 1

    if json_output:
        print(
            json.dumps(
                {
                    "profiles": [
                        asdict(profile)
                        for profile in profiles
                    ]
                },
                indent=2,
            )
        )
        return 0

    print()
    print("MIGOps Profiles")
    print("===============")
    print()

    if not profiles:
        print("No MIG GPU instance profiles were detected.")
        print()
        print(
            "The GPU may not support MIG, MIG may be unavailable, "
            "or the NVIDIA driver returned no profile information."
        )
        return 1

    current_gpu: str | None = None

    for profile in profiles:
        if profile.gpu != current_gpu:
            current_gpu = profile.gpu

            print()
            print(f"GPU {current_gpu}")
            print("-" * 58)
            print(
                f"{'Profile':<18}"
                f"{'ID':<8}"
                f"{'Available':<14}"
                f"{'Memory':<12}"
            )
            print("-" * 58)

        memory = (
            f"{profile.memory_gib:.2f} GiB"
            if profile.memory_gib is not None
            else "N/A"
        )

        availability = f"{profile.free}/{profile.total}"

        print(
            f"{profile.name:<18}"
            f"{profile.profile_id:<8}"
            f"{availability:<14}"
            f"{memory:<12}"
        )

    print()

    return 0