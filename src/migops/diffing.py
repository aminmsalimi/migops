"""Desired-vs-actual MIG configuration diffing."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json

from migops.config import (
    ConfigError,
    MigOpsConfig,
    load_config,
    resolve_gpu,
)
from migops.lifecycle import query_gpu_instances
from migops.nvidia import NvidiaSmiError
from migops.status import query_gpus


@dataclass
class GPUDiff:
    selector: str
    gpu_index: str | None
    gpu_name: str | None
    desired_mig_enabled: bool
    actual_mig_enabled: bool | None
    desired_profiles: dict[str, int]
    actual_profiles: dict[str, int]
    changed: bool


@dataclass
class DiffResult:
    changed: bool
    gpus: list[GPUDiff]


def desired_counts(
    config_gpu,
) -> dict[str, int]:
    return {
        request.profile: request.count
        for request in config_gpu.instances
    }


def current_counts(
    gpu_index: str,
) -> dict[str, int]:
    instances = query_gpu_instances(
        gpu_index
    )

    return dict(
        Counter(
            instance.profile
            for instance in instances
        )
    )


def diff_config_object(
    config: MigOpsConfig,
) -> DiffResult:
    try:
        actual_gpus = query_gpus()
    except NvidiaSmiError as exc:
        raise ConfigError(
            f"Unable to query NVIDIA GPUs: {exc}"
        ) from exc

    results: list[GPUDiff] = []

    for desired in config.gpus:
        actual = resolve_gpu(
            desired.gpu,
            actual_gpus,
        )

        if actual is None:
            results.append(
                GPUDiff(
                    selector=desired.gpu,
                    gpu_index=None,
                    gpu_name=None,
                    desired_mig_enabled=desired.mig_enabled,
                    actual_mig_enabled=None,
                    desired_profiles=desired_counts(
                        desired
                    ),
                    actual_profiles={},
                    changed=True,
                )
            )
            continue

        actual_enabled = (
            actual.mig_mode
            .strip()
            .lower()
            == "enabled"
        )

        actual_profiles: dict[str, int] = {}

        if actual_enabled:
            try:
                actual_profiles = current_counts(
                    actual.index
                )
            except NvidiaSmiError as exc:
                raise ConfigError(
                    "Unable to query current MIG layout "
                    f"for GPU {actual.index}: {exc}"
                ) from exc

        desired_profiles = (
            desired_counts(desired)
            if desired.mig_enabled
            else {}
        )

        changed = (
            desired.mig_enabled != actual_enabled
            or desired_profiles != actual_profiles
        )

        results.append(
            GPUDiff(
                selector=desired.gpu,
                gpu_index=actual.index,
                gpu_name=actual.name,
                desired_mig_enabled=desired.mig_enabled,
                actual_mig_enabled=actual_enabled,
                desired_profiles=desired_profiles,
                actual_profiles=actual_profiles,
                changed=changed,
            )
        )

    return DiffResult(
        changed=any(
            item.changed
            for item in results
        ),
        gpus=results,
    )


def print_diff(
    path: str,
    json_output: bool = False,
) -> int:
    try:
        result = diff_config_object(
            load_config(path)
        )

    except ConfigError as exc:
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
            print("MIGOps Diff")
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

        return 2 if result.changed else 0

    print()
    print("MIGOps Diff")
    print("===========")
    print()
    print(f"Configuration: {path}")
    print()

    for item in result.gpus:
        title = f"GPU {item.selector}"

        if item.gpu_name:
            title += f" ({item.gpu_name})"

        print(title)
        print("-" * 60)

        actual_mode = (
            "Unknown"
            if item.actual_mig_enabled is None
            else (
                "Enabled"
                if item.actual_mig_enabled
                else "Disabled"
            )
        )

        print(
            "MIG mode: "
            f"desired="
            f"{'Enabled' if item.desired_mig_enabled else 'Disabled'}  "
            f"actual={actual_mode}"
        )

        keys = sorted(
            set(item.desired_profiles)
            | set(item.actual_profiles)
        )

        if keys:
            print()
            print(
                f"{'Profile':<20}"
                f"{'Desired':<10}"
                f"{'Actual':<10}"
                f"State"
            )

            for profile in keys:
                desired = item.desired_profiles.get(
                    profile,
                    0,
                )

                actual = item.actual_profiles.get(
                    profile,
                    0,
                )

                print(
                    f"{profile:<20}"
                    f"{desired:<10}"
                    f"{actual:<10}"
                    f"{'OK' if desired == actual else 'DRIFT'}"
                )

        print()
        print(
            "Result: "
            f"{'DRIFT' if item.changed else 'MATCH'}"
        )
        print()

    print(
        "Overall: DRIFT DETECTED"
        if result.changed
        else "Overall: MATCH"
    )

    return 2 if result.changed else 0
