"""Snapshot current MIG state into reusable MIGOps YAML."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import platform

import yaml

from migops.lifecycle import query_gpu_instances
from migops.nvidia import NvidiaSmiError
from migops.status import query_gpus


class SnapshotError(RuntimeError):
    """Raised when a MIG snapshot cannot be created."""


def snapshot_data(
    gpu_selector: str | None = None,
) -> dict:
    try:
        gpus = query_gpus()
    except NvidiaSmiError as exc:
        raise SnapshotError(
            f"Unable to query NVIDIA GPUs: {exc}"
        ) from exc

    if gpu_selector is not None:
        selector_lower = gpu_selector.lower()

        gpus = [
            gpu
            for gpu in gpus
            if (
                gpu.index == gpu_selector
                or gpu.uuid.lower() == selector_lower
                or gpu.pci_bus_id.lower() == selector_lower
            )
        ]

        if not gpus:
            raise SnapshotError(
                f"GPU '{gpu_selector}' was not found."
            )

    entries = []

    for gpu in gpus:
        instances = []

        if gpu.mig_mode.strip().lower() == "enabled":
            try:
                gpu_instances = query_gpu_instances(
                    gpu.index
                )
            except NvidiaSmiError as exc:
                raise SnapshotError(
                    "Unable to query GPU Instances for "
                    f"GPU {gpu.index}: {exc}"
                ) from exc

            counts = Counter(
                instance.profile
                for instance in gpu_instances
            )

            instances = [
                {
                    "profile": profile,
                    "count": count,
                }
                for profile, count in sorted(
                    counts.items()
                )
            ]

        entries.append(
            {
                "gpu": gpu.index,
                "mig_enabled": (
                    gpu.mig_mode.strip().lower()
                    == "enabled"
                ),
                "instances": instances,
            }
        )

    return {
        "version": 1,
        "snapshot": {
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "host": platform.node() or "unknown",
        },
        "gpus": entries,
    }


def write_snapshot(
    output: str | None = None,
    gpu: str | None = None,
) -> Path:
    data = snapshot_data(gpu)

    if output:
        path = Path(output)
    else:
        host = (
            platform.node()
            or "host"
        ).replace(" ", "-")

        stamp = datetime.now().strftime(
            "%Y%m%d-%H%M%S"
        )

        path = (
            Path("snapshots")
            / f"{host}-{stamp}.yaml"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        yaml.safe_dump(
            data,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    return path


def print_snapshot(
    output: str | None = None,
    gpu: str | None = None,
) -> int:
    try:
        path = write_snapshot(
            output,
            gpu,
        )

    except (
        SnapshotError,
        OSError,
    ) as exc:
        print()
        print("MIGOps Snapshot")
        print("===============")
        print()
        print(f"[FAIL] {exc}")
        return 1

    print()
    print("MIGOps Snapshot")
    print("===============")
    print()
    print(
        f"[PASS] Snapshot saved: {path}"
    )

    return 0
