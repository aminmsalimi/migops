"""Automatic MIG partition recommendation and optional safe application."""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import asdict, dataclass

from migops.apply import apply_config_object
from migops.config import (
    GPUConfig,
    MigOpsConfig,
    ProfileRequest,
)
from migops.nvidia import (
    NvidiaSmiError,
    run_nvidia_smi,
)
from migops.profiles import (
    MigProfile,
    query_profiles,
)


STANDARD_PROFILE_RE = re.compile(
    r"^\d+g\.\d+gb$"
)


@dataclass
class PhysicalGPU:
    index: str
    name: str
    uuid: str
    pci_bus_id: str
    memory_mib: int
    mig_mode: str

    @property
    def memory_gib(self) -> float:
        return self.memory_mib / 1024


@dataclass
class SplitRecommendation:
    gpu_index: str
    gpu_name: str
    gpu_memory_gib: float
    mig_mode: str
    requested_instances: int
    target_memory_gib: float
    profile: str
    profile_id: str
    profile_memory_gib: float
    max_instances: int
    allocated_memory_gib: float
    memory_coverage_percent: float
    acceptable: bool
    message: str


def query_physical_gpus() -> list[PhysicalGPU]:
    output = run_nvidia_smi(
        [
            (
                "--query-gpu="
                "index,name,uuid,pci.bus_id,"
                "memory.total,mig.mode.current"
            ),
            "--format=csv,noheader,nounits",
        ]
    )

    reader = csv.reader(
        io.StringIO(output)
    )

    gpus: list[PhysicalGPU] = []

    for row in reader:
        if len(row) != 6:
            continue

        values = [
            value.strip()
            for value in row
        ]

        try:
            memory_mib = int(
                float(values[4])
            )
        except ValueError:
            continue

        gpus.append(
            PhysicalGPU(
                index=values[0],
                name=values[1],
                uuid=values[2],
                pci_bus_id=values[3],
                memory_mib=memory_mib,
                mig_mode=values[5],
            )
        )

    return gpus


def select_gpu(
    gpus: list[PhysicalGPU],
    selector: str,
) -> PhysicalGPU:
    selector_lower = selector.lower()

    for gpu in gpus:
        if gpu.index == selector:
            return gpu

        if gpu.uuid.lower() == selector_lower:
            return gpu

        if gpu.pci_bus_id.lower() == selector_lower:
            return gpu

    raise ValueError(
        f"GPU '{selector}' was not found."
    )


def is_standard_profile(
    profile: MigProfile,
) -> bool:
    return bool(
        STANDARD_PROFILE_RE.match(
            profile.name
        )
    )


def recommend_split(
    gpu: PhysicalGPU,
    profiles: list[MigProfile],
    instances: int,
) -> SplitRecommendation:
    if instances < 1:
        raise ValueError(
            "Instance count must be at least 1."
        )

    target_memory = (
        gpu.memory_gib
        / instances
    )

    candidates: list[MigProfile] = []

    for profile in profiles:
        if not is_standard_profile(profile):
            continue

        if profile.total < instances:
            continue

        if profile.memory_gib is None:
            continue

        allocated = (
            profile.memory_gib
            * instances
        )

        if (
            allocated
            > gpu.memory_gib * 1.02
        ):
            continue

        candidates.append(
            profile
        )

    if not candidates:
        raise ValueError(
            "No MIG profile reported by the "
            f"driver supports {instances} "
            "identical instances."
        )

    candidates.sort(
        key=lambda profile: (
            abs(
                profile.memory_gib
                - target_memory
            ),
            -profile.memory_gib,
        )
    )

    best = candidates[0]

    allocated_memory = (
        best.memory_gib
        * instances
    )

    coverage = (
        allocated_memory
        / gpu.memory_gib
        * 100
    )

    per_instance_error = (
        abs(
            best.memory_gib
            - target_memory
        )
        / target_memory
    )

    acceptable = (
        coverage >= 85.0
        and per_instance_error <= 0.20
    )

    message = (
        "A suitable equal-memory MIG geometry was found."
        if acceptable
        else (
            "No close equal-memory geometry exists. "
            "The best native MIG profile would leave "
            "a significant amount of GPU memory unused."
        )
    )

    return SplitRecommendation(
        gpu_index=gpu.index,
        gpu_name=gpu.name,
        gpu_memory_gib=gpu.memory_gib,
        mig_mode=gpu.mig_mode,
        requested_instances=instances,
        target_memory_gib=target_memory,
        profile=best.name,
        profile_id=best.profile_id,
        profile_memory_gib=best.memory_gib,
        max_instances=best.total,
        allocated_memory_gib=allocated_memory,
        memory_coverage_percent=coverage,
        acceptable=acceptable,
        message=message,
    )


def plan_split(
    gpu_selector: str,
    instances: int,
    json_output: bool = False,
    apply: bool = False,
    dry_run: bool = False,
    yes: bool = False,
    force: bool = False,
) -> int:
    try:
        gpu = select_gpu(
            query_physical_gpus(),
            gpu_selector,
        )

        recommendation = recommend_split(
            gpu,
            query_profiles(gpu.index),
            instances,
        )

    except (
        NvidiaSmiError,
        ValueError,
    ) as exc:
        if json_output:
            print(
                json.dumps(
                    {
                        "error": str(exc),
                    },
                    indent=2,
                )
            )
        else:
            print()
            print("MIGOps Smart Split")
            print("==================")
            print()
            print(
                "[FAIL] Unable to generate split plan"
            )
            print()
            print(str(exc))

        return 1

    if json_output and not apply:
        print(
            json.dumps(
                asdict(recommendation),
                indent=2,
            )
        )

        return (
            0
            if recommendation.acceptable
            else 2
        )

    if not json_output:
        print()
        print("MIGOps Smart Split")
        print("==================")
        print()
        print(
            f"GPU:          "
            f"{recommendation.gpu_index}"
        )
        print(
            f"Model:        "
            f"{recommendation.gpu_name}"
        )
        print(
            f"Total VRAM:   "
            f"{recommendation.gpu_memory_gib:.2f} GiB"
        )
        print(
            f"MIG Mode:     "
            f"{recommendation.mig_mode}"
        )
        print()
        print("Request")
        print("-------")
        print(
            f"Instances:            "
            f"{recommendation.requested_instances}"
        )
        print(
            f"Target per instance:  "
            f"{recommendation.target_memory_gib:.2f} GiB"
        )
        print()
        print("Recommended MIG Geometry")
        print("------------------------")
        print(
            f"{recommendation.requested_instances} x "
            f"MIG {recommendation.profile}"
        )
        print()
        print(
            f"Profile memory:       "
            f"{recommendation.profile_memory_gib:.2f} GiB"
        )
        print(
            f"Allocated memory:     "
            f"{recommendation.allocated_memory_gib:.2f} GiB"
        )
        print(
            f"Memory coverage:      "
            f"{recommendation.memory_coverage_percent:.1f}%"
        )
        print(
            f"Maximum instances:    "
            f"{recommendation.max_instances}"
        )
        print()

        print(
            "[PASS] Suitable equal split found."
            if recommendation.acceptable
            else (
                "[WARN] This is not a good "
                "equal-memory split."
            )
        )

        print()
        print(
            recommendation.message
        )

    if not recommendation.acceptable:
        print()
        print(
            "No changes have been made."
        )
        return 2

    if not apply:
        print()
        print(
            "No changes have been made. "
            "Use --apply --dry-run to preview "
            "execution, or --apply --yes to execute."
        )
        return 0

    config = MigOpsConfig(
        version=1,
        gpus=[
            GPUConfig(
                gpu=gpu.index,
                mig_enabled=True,
                instances=[
                    ProfileRequest(
                        profile=recommendation.profile,
                        count=instances,
                    )
                ],
            )
        ],
    )

    return apply_config_object(
        config,
        dry_run=dry_run,
        yes=yes,
        force=force,
        source_label=(
            f"smart split: GPU {gpu.index} into "
            f"{instances} x {recommendation.profile}"
        ),
    )
