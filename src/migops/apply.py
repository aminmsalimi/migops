"""Safe desired-state application and restore workflows."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from migops.config import (
    ConfigError,
    MigOpsConfig,
    canonical_profile_counts,
    load_config,
    resolve_gpu,
    validate_config,
)
from migops.diffing import diff_config_object
from migops.lifecycle import (
    MigSafetyError,
    build_destroy_sequence,
    build_easy_create_command,
    build_mode_command,
    check_workload_safety,
    execute_operation,
    execute_sequence,
    get_matching_workloads,
    query_gpu_instances,
)
from migops.nvidia import NvidiaSmiError
from migops.profiles import query_profiles
from migops.snapshot import (
    SnapshotError,
    write_snapshot,
)
from migops.status import query_gpus


def requires_workload_check(
    *,
    current_enabled: bool,
    desired_enabled: bool,
    current_profiles: dict[str, int],
    desired_profiles: dict[str, int],
) -> bool:
    """
    Return True when an operation may disrupt current GPU workloads.

    Mode changes and replacement/removal of an existing MIG layout are
    considered disruptive.
    """

    if current_enabled != desired_enabled:
        return True

    if current_profiles and current_profiles != desired_profiles:
        return True

    return False


def _apply_object(
    config: MigOpsConfig,
    *,
    dry_run: bool,
    yes: bool,
    force: bool,
    snapshot_dir: str,
    source_label: str,
) -> int:
    try:
        validation = validate_config(
            config
        )
    except ConfigError as exc:
        print()
        print("MIGOps Apply")
        print("============")
        print()
        print(f"[FAIL] {exc}")
        return 1

    if not validation.valid:
        print()
        print("MIGOps Apply")
        print("============")
        print()
        print(
            "[BLOCKED] Configuration validation failed."
        )

        for gpu_result in validation.gpu_results:
            for message in gpu_result.messages:
                if message.level == "FAIL":
                    print(
                        f"[FAIL] GPU {gpu_result.selector}: "
                        f"{message.message}"
                    )

        return 1

    try:
        diff = diff_config_object(
            config
        )
    except ConfigError as exc:
        print()
        print(f"[FAIL] {exc}")
        return 1

    print()
    print("MIGOps Apply")
    print("============")
    print()
    print(f"Source: {source_label}")
    print()

    if not diff.changed:
        print(
            "[PASS] System already matches desired state."
        )
        return 0

    if dry_run:
        print(
            "DRY RUN - planned native operations:"
        )
        print()

    elif not yes:
        print(
            "[BLOCKED] Real changes require --yes."
        )
        print(
            "Preview first with --dry-run, then "
            "re-run with --yes."
        )
        return 1

    # Full pre-flight before any real change. This avoids changing GPU 0
    # and only then discovering a safety problem on GPU 1.
    try:
        actual_gpus = query_gpus()

        for desired in config.gpus:
            actual = resolve_gpu(
                desired.gpu,
                actual_gpus,
            )

            if actual is None:
                raise ConfigError(
                    f"GPU '{desired.gpu}' was not found."
                )

            current_enabled = (
                actual.mig_mode
                .strip()
                .lower()
                == "enabled"
            )

            current_gis = (
                query_gpu_instances(actual.index)
                if current_enabled
                else []
            )

            current_profiles: dict[str, int] = {}

            for instance in current_gis:
                current_profiles[instance.profile] = (
                    current_profiles.get(
                        instance.profile,
                        0,
                    )
                    + 1
                )

            desired_profiles: dict[str, int] = {}

            if desired.mig_enabled:
                desired_profiles = canonical_profile_counts(
                    desired,
                    query_profiles(actual.index),
                )

            disruptive = requires_workload_check(
                current_enabled=current_enabled,
                desired_enabled=desired.mig_enabled,
                current_profiles=current_profiles,
                desired_profiles=desired_profiles,
            )

            if not disruptive:
                continue

            if dry_run:
                workloads = get_matching_workloads(
                    actual.index
                )

                if workloads:
                    print(
                        f"[WARN] GPU {actual.index}: "
                        f"{len(workloads)} active workload(s) "
                        "would block a real apply unless --force "
                        "is used intentionally."
                    )

            else:
                check_workload_safety(
                    actual.index,
                    force=force,
                )

    except (
        NvidiaSmiError,
        ConfigError,
        MigSafetyError,
    ) as exc:
        print(
            f"[BLOCKED] {exc}"
        )
        return 1

    snapshot_path: Path | None = None

    if not dry_run:
        try:
            directory = Path(snapshot_dir)

            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            stamp = datetime.now().strftime(
                "%Y%m%d-%H%M%S"
            )

            snapshot_path = write_snapshot(
                str(
                    directory
                    / f"pre-apply-{stamp}.yaml"
                )
            )

            print(
                f"Safety snapshot: {snapshot_path}"
            )
            print()

        except (
            SnapshotError,
            OSError,
        ) as exc:
            print(
                "[BLOCKED] Unable to create "
                f"safety snapshot: {exc}"
            )
            return 1

    for desired in config.gpus:
        try:
            actual_gpus = query_gpus()

            actual = resolve_gpu(
                desired.gpu,
                actual_gpus,
            )

            if actual is None:
                raise ConfigError(
                    f"GPU '{desired.gpu}' was not found."
                )

            current_enabled = (
                actual.mig_mode
                .strip()
                .lower()
                == "enabled"
            )

            current_gis = (
                query_gpu_instances(
                    actual.index
                )
                if current_enabled
                else []
            )

            current_profiles: dict[str, int] = {}

            for instance in current_gis:
                current_profiles[instance.profile] = (
                    current_profiles.get(
                        instance.profile,
                        0,
                    )
                    + 1
                )

            desired_profiles: dict[str, int] = {}

            if desired.mig_enabled:
                desired_profiles = canonical_profile_counts(
                    desired,
                    query_profiles(actual.index),
                )

            if not desired.mig_enabled:
                if current_gis:
                    result = execute_sequence(
                        build_destroy_sequence(
                            actual.index
                        ),
                        dry_run=dry_run,
                    )

                    if result != 0:
                        return 1

                if current_enabled:
                    result = execute_operation(
                        build_mode_command(
                            actual.index,
                            False,
                        ),
                        dry_run=dry_run,
                    )

                    if result != 0:
                        return 1

                continue

            if not current_enabled:
                result = execute_operation(
                    build_mode_command(
                        actual.index,
                        True,
                    ),
                    dry_run=dry_run,
                )

                if result != 0:
                    return 1

                if not dry_run:
                    refreshed = resolve_gpu(
                        desired.gpu,
                        query_gpus(),
                    )

                    if (
                        refreshed is None
                        or (
                            refreshed.mig_mode
                            .strip()
                            .lower()
                            != "enabled"
                        )
                    ):
                        print(
                            "[BLOCKED] MIG mode is not active "
                            "after the enable request. A GPU "
                            "reset or reboot may be required "
                            "before continuing."
                        )
                        return 1

            if current_profiles != desired_profiles:
                if current_gis:
                    result = execute_sequence(
                        build_destroy_sequence(
                            actual.index
                        ),
                        dry_run=dry_run,
                    )

                    if result != 0:
                        return 1

                for profile_name, count in desired_profiles.items():
                    result = execute_operation(
                        build_easy_create_command(
                            actual.index,
                            profile_name,
                            count,
                        ),
                        dry_run=dry_run,
                    )

                    if result != 0:
                        return 1

        except (
            NvidiaSmiError,
            ConfigError,
            ValueError,
        ) as exc:
            print(
                f"[FAIL] {exc}"
            )
            return 1

    if dry_run:
        print()
        print(
            "DRY RUN complete - no changes were made."
        )
        return 0

    try:
        verification = diff_config_object(
            config
        )
    except ConfigError as exc:
        print(
            "[WARN] Changes were executed, but "
            f"verification failed: {exc}"
        )
        return 1

    if verification.changed:
        print()
        print(
            "[FAIL] Apply completed but final state "
            "does not match desired configuration."
        )

        if snapshot_path:
            print(
                f"Recovery snapshot: {snapshot_path}"
            )

        return 1

    print()
    print(
        "[PASS] Desired MIG state applied and verified."
    )

    if snapshot_path:
        print(
            f"Previous state snapshot: {snapshot_path}"
        )

    return 0


def apply_config(
    path: str,
    *,
    dry_run: bool = False,
    yes: bool = False,
    force: bool = False,
    snapshot_dir: str = "snapshots",
) -> int:
    """Apply a desired-state YAML configuration."""

    try:
        config = load_config(path)
    except ConfigError as exc:
        print()
        print("MIGOps Apply")
        print("============")
        print()
        print(f"[FAIL] {exc}")
        return 1

    return _apply_object(
        config,
        dry_run=dry_run,
        yes=yes,
        force=force,
        snapshot_dir=snapshot_dir,
        source_label=path,
    )


def apply_config_object(
    config: MigOpsConfig,
    *,
    dry_run: bool = False,
    yes: bool = False,
    force: bool = False,
    snapshot_dir: str = "snapshots",
    source_label: str = "generated configuration",
) -> int:
    """Apply an in-memory desired-state configuration."""

    return _apply_object(
        config,
        dry_run=dry_run,
        yes=yes,
        force=force,
        snapshot_dir=snapshot_dir,
        source_label=source_label,
    )


def restore_snapshot(
    path: str,
    *,
    dry_run: bool = False,
    yes: bool = False,
    force: bool = False,
) -> int:
    """
    Restore a standard MIGOps snapshot.

    Snapshots currently restore GI profile counts and recreate one default
    CI per GI. They are not intended to preserve advanced custom CI
    sub-partitioning.
    """

    print()
    print(
        "[INFO] Restore recreates GI profile counts with "
        "default Compute Instances."
    )

    return apply_config(
        path,
        dry_run=dry_run,
        yes=yes,
        force=force,
        snapshot_dir="snapshots",
    )
