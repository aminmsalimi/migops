"""MIGOps change planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json

from migops.config import (
    ConfigError,
    MigOpsConfig,
    load_config,
    resolve_gpu,
    validate_config,
)
from migops.diffing import diff_config_object
from migops.nvidia import NvidiaSmiError
from migops.status import query_gpus
from migops.workloads import query_workloads


@dataclass
class GPUPlan:
    selector: str
    gpu_index: str | None
    gpu_name: str | None
    actions: list[str]
    active_processes: int
    risk: str
    changed: bool


@dataclass
class PlanResult:
    valid: bool
    changed: bool
    gpus: list[GPUPlan]


def build_plan(
    config: MigOpsConfig,
) -> PlanResult:
    validation = validate_config(
        config
    )

    if not validation.valid:
        return PlanResult(
            valid=False,
            changed=True,
            gpus=[],
        )

    diff = diff_config_object(
        config
    )

    actual_gpus = query_gpus()

    try:
        workloads = query_workloads()
    except NvidiaSmiError:
        workloads = []

    plans: list[GPUPlan] = []

    for desired, item in zip(
        config.gpus,
        diff.gpus,
    ):
        actual = resolve_gpu(
            desired.gpu,
            actual_gpus,
        )

        actions: list[str] = []

        active = len(
            [
                process
                for process in workloads
                if (
                    actual is not None
                    and process.gpu == actual.index
                )
            ]
        )

        if actual is None:
            plans.append(
                GPUPlan(
                    selector=desired.gpu,
                    gpu_index=None,
                    gpu_name=None,
                    actions=["GPU not found"],
                    active_processes=active,
                    risk="HIGH",
                    changed=True,
                )
            )
            continue

        if not item.changed:
            actions.append(
                "No changes required"
            )
            risk = "LOW"

        elif not desired.mig_enabled:
            if item.actual_profiles:
                actions.append(
                    "Destroy all existing Compute Instances "
                    "and GPU Instances"
                )

            if item.actual_mig_enabled:
                actions.append(
                    "Disable MIG mode"
                )

            risk = (
                "HIGH"
                if item.actual_profiles or active
                else "MEDIUM"
            )

        else:
            if not item.actual_mig_enabled:
                actions.append(
                    "Enable MIG mode"
                )

            if (
                item.actual_profiles
                != item.desired_profiles
            ):
                if item.actual_profiles:
                    actions.append(
                        "Destroy existing Compute Instances "
                        "and GPU Instances"
                    )

                for request in desired.instances:
                    actions.append(
                        f"Create {request.count} x "
                        f"{request.profile} with default "
                        "Compute Instance(s)"
                    )

            risk = (
                "HIGH"
                if item.actual_profiles or active
                else "MEDIUM"
            )

        if active:
            actions.append(
                f"WARNING: {active} active GPU "
                "process(es) detected"
            )

        plans.append(
            GPUPlan(
                selector=desired.gpu,
                gpu_index=actual.index,
                gpu_name=actual.name,
                actions=actions,
                active_processes=active,
                risk=risk,
                changed=item.changed,
            )
        )

    return PlanResult(
        valid=True,
        changed=diff.changed,
        gpus=plans,
    )


def print_plan(
    path: str,
    json_output: bool = False,
) -> int:
    try:
        config = load_config(path)
        result = build_plan(config)

    except (
        ConfigError,
        NvidiaSmiError,
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
            print("MIGOps Plan")
            print("===========")
            print()
            print(f"[FAIL] {exc}")

        return 1

    if json_output:
        print(
            json.dumps(
                asdict(result),
                indent=2,
            )
        )

        return 0 if result.valid else 1

    print()
    print("MIGOps Plan")
    print("===========")
    print()
    print(f"Configuration: {path}")
    print()

    if not result.valid:
        print(
            "[FAIL] Configuration is invalid. "
            "Run `migops validate` first."
        )
        return 1

    for gpu_plan in result.gpus:
        title = f"GPU {gpu_plan.selector}"

        if gpu_plan.gpu_name:
            title += f" ({gpu_plan.gpu_name})"

        print(title)
        print("-" * 60)

        for number, action in enumerate(
            gpu_plan.actions,
            start=1,
        ):
            print(
                f"{number}. {action}"
            )

        print(
            f"Risk: {gpu_plan.risk}"
        )
        print()

    print("No changes have been made.")

    return 0
