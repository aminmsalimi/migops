"""NVIDIA MIG lifecycle management.

Provides both:
- high-level MIGOps workflows such as `create` and `destroy`
- low-level GI / CI operations for advanced administrators
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from migops.nvidia import NvidiaSmiError, run_nvidia_smi
from migops.profiles import MigProfile, query_profiles
from migops.status import GPU, query_gpus
from migops.workloads import (
    GpuProcess,
    parse_compute_instances,
    query_workloads,
)


class MigSafetyError(RuntimeError):
    """Raised when MIGOps blocks an unsafe MIG operation."""


@dataclass
class GpuInstance:
    gpu: str
    profile: str
    profile_id: str
    gi_id: str
    placement: str


GI_RE = re.compile(
    r"^\|\s*"
    r"(?P<gpu>\d+)\s+"
    r"MIG\s+"
    r"(?P<profile>\S+)\s+"
    r"(?P<profile_id>\d+\*?)\s+"
    r"(?P<gi_id>\d+)\s+"
    r"(?P<placement>\d+:\d+)\s*"
    r"\|$"
)


# ============================================================
# QUERY / PARSING
# ============================================================


def parse_gpu_instances(output: str) -> list[GpuInstance]:
    """Parse `nvidia-smi mig -lgi` output."""

    instances: list[GpuInstance] = []

    for line in output.splitlines():
        match = GI_RE.match(line)

        if not match:
            continue

        instances.append(
            GpuInstance(
                gpu=match.group("gpu"),
                profile=match.group("profile"),
                profile_id=match.group("profile_id"),
                gi_id=match.group("gi_id"),
                placement=match.group("placement"),
            )
        )

    return instances


def resolve_gpu(selector: str) -> GPU:
    """Resolve GPU index, UUID, or PCI bus ID."""

    selector_lower = selector.lower()

    for gpu in query_gpus():
        if gpu.index == selector:
            return gpu

        if gpu.uuid.lower() == selector_lower:
            return gpu

        if gpu.pci_bus_id.lower() == selector_lower:
            return gpu

    raise ValueError(
        f"GPU '{selector}' was not found."
    )


def query_gpu_instances(
    gpu: str | None = None,
) -> list[GpuInstance]:
    """Return current GPU Instances."""

    arguments = ["mig", "-lgi"]

    if gpu is not None:
        arguments.extend(["-i", gpu])

    output = run_nvidia_smi(arguments)

    return parse_gpu_instances(output)


def query_ci_instances(
    gpu: str | None = None,
    gi: str | None = None,
):
    """Return current Compute Instances."""

    arguments = ["mig", "-lci"]

    if gi is not None:
        arguments.extend(["-gi", gi])

    if gpu is not None:
        arguments.extend(["-i", gpu])

    output = run_nvidia_smi(arguments)

    return parse_compute_instances(output)


# ============================================================
# COMMAND BUILDERS
# ============================================================


def build_mode_command(
    gpu: str,
    enabled: bool,
) -> list[str]:
    """Build native MIG mode command."""

    return [
        "-i",
        gpu,
        "-mig",
        "1" if enabled else "0",
    ]


def build_gi_create_command(
    gpu: str,
    profile: str,
    with_ci: bool = False,
) -> list[str]:
    """Build low-level GPU Instance creation command."""

    command = [
        "mig",
        "-cgi",
        profile,
    ]

    if with_ci:
        command.append("-C")

    command.extend(
        ["-i", gpu]
    )

    return command


def build_easy_create_command(
    gpu: str,
    profile: str,
    count: int,
) -> list[str]:
    """
    Build high-level create command.

    `-C` tells nvidia-smi to create corresponding CIs
    automatically.
    """

    if count < 1:
        raise ValueError(
            "Count must be at least 1."
        )

    profiles = ",".join(
        profile
        for _ in range(count)
    )

    return [
        "mig",
        "-cgi",
        profiles,
        "-C",
        "-i",
        gpu,
    ]


def build_gi_delete_command(
    gpu: str,
    gi: str | None = None,
) -> list[str]:
    """Build low-level GI deletion command."""

    command = [
        "mig",
        "-dgi",
    ]

    if gi is not None:
        command.extend(
            ["-gi", gi]
        )

    command.extend(
        ["-i", gpu]
    )

    return command


def build_ci_create_command(
    gpu: str,
    gi: str,
    profile: str,
) -> list[str]:
    """Build low-level CI creation command."""

    return [
        "mig",
        "-cci",
        profile,
        "-gi",
        gi,
        "-i",
        gpu,
    ]


def build_ci_delete_command(
    gpu: str,
    gi: str | None = None,
    ci: str | None = None,
) -> list[str]:
    """Build low-level CI deletion command."""

    command = [
        "mig",
        "-dci",
    ]

    if ci is not None:
        command.extend(
            ["-ci", ci]
        )

    if gi is not None:
        command.extend(
            ["-gi", gi]
        )

    command.extend(
        ["-i", gpu]
    )

    return command


def build_destroy_sequence(
    gpu: str,
    *,
    gi: str | None = None,
    ci_ids: list[str] | None = None,
) -> list[list[str]]:
    """
    Build safe destruction sequence.

    Compute Instances are always removed before GPU Instances.
    """

    commands: list[list[str]] = []

    if gi is None:
        # All CIs, then all GIs.
        commands.append(
            [
                "mig",
                "-dci",
                "-i",
                gpu,
            ]
        )

        commands.append(
            [
                "mig",
                "-dgi",
                "-i",
                gpu,
            ]
        )

        return commands

    if ci_ids:
        commands.append(
            [
                "mig",
                "-dci",
                "-ci",
                ",".join(ci_ids),
                "-gi",
                gi,
                "-i",
                gpu,
            ]
        )

    commands.append(
        [
            "mig",
            "-dgi",
            "-gi",
            gi,
            "-i",
            gpu,
        ]
    )

    return commands


# ============================================================
# SAFETY
# ============================================================


def get_matching_workloads(
    gpu_index: str,
    gi: str | None = None,
    ci: str | None = None,
) -> list[GpuProcess]:
    """Return workloads affected by an operation."""

    processes = query_workloads()

    matches = [
        process
        for process in processes
        if process.gpu == gpu_index
    ]

    if gi is not None:
        matches = [
            process
            for process in matches
            if process.gi_id == gi
        ]

    if ci is not None:
        matches = [
            process
            for process in matches
            if process.ci_id == ci
        ]

    return matches


def check_workload_safety(
    gpu_index: str,
    *,
    gi: str | None = None,
    ci: str | None = None,
    force: bool = False,
) -> list[GpuProcess]:
    """Block destructive operations when workloads exist."""

    try:
        workloads = get_matching_workloads(
            gpu_index=gpu_index,
            gi=gi,
            ci=ci,
        )

    except NvidiaSmiError as exc:
        if force:
            return []

        raise MigSafetyError(
            "MIGOps could not verify whether GPU workloads are active. "
            "Operation blocked. Use --force only if you intentionally "
            f"want to bypass this check. Reason: {exc}"
        ) from exc

    if workloads and not force:
        details = ", ".join(
            f"{process.pid}"
            for process in workloads
        )

        raise MigSafetyError(
            f"{len(workloads)} active GPU workload(s) detected "
            f"(PID: {details}). Operation blocked. "
            "Run `migops users` first or explicitly use --force."
        )

    return workloads


def find_profile(
    profiles: list[MigProfile],
    requested: str,
) -> MigProfile:
    """Find a profile using either its name or ID."""

    requested_lower = requested.lower()

    for profile in profiles:
        if profile.name.lower() == requested_lower:
            return profile

        if profile.profile_id == requested:
            return profile

    raise ValueError(
        f"MIG profile '{requested}' is not supported "
        "or was not reported by the NVIDIA driver."
    )


# ============================================================
# EXECUTION
# ============================================================


def format_native_command(
    arguments: list[str],
) -> str:
    """Return readable native command."""

    return (
        "nvidia-smi "
        + " ".join(arguments)
    )


def execute_operation(
    arguments: list[str],
    *,
    dry_run: bool = False,
) -> int:
    """Execute or preview one NVIDIA operation."""

    print(
        f"  {format_native_command(arguments)}"
    )

    if dry_run:
        return 0

    try:
        output = run_nvidia_smi(
            arguments
        )

    except NvidiaSmiError as exc:
        print()
        print("[FAIL] NVIDIA operation failed")
        print()
        print(str(exc))
        return 1

    if output:
        print()
        print(output)

    return 0


def execute_sequence(
    commands: list[list[str]],
    *,
    dry_run: bool = False,
) -> int:
    """Execute a sequence, stopping on the first failure."""

    for number, command in enumerate(
        commands,
        start=1,
    ):
        print(
            f"[{number}/{len(commands)}] "
            f"{format_native_command(command)}"
        )

        if dry_run:
            continue

        try:
            output = run_nvidia_smi(
                command
            )

        except NvidiaSmiError as exc:
            print()
            print("[FAIL] Operation sequence stopped.")
            print()
            print(str(exc))
            return 1

        if output:
            print(output)

    if dry_run:
        print()
        print("DRY RUN - no changes were made.")

    else:
        print()
        print("[PASS] Operation completed.")

    return 0


# ============================================================
# HIGH-LEVEL EASY WORKFLOW
# ============================================================


def create_mig(
    gpu: str,
    profile: str,
    *,
    count: int = 1,
    dry_run: bool = False,
) -> int:
    """
    Create complete usable MIG instances.

    Each requested GI receives its corresponding default CI.
    """

    print()
    print("MIGOps Create")
    print("=============")
    print()

    if count < 1:
        print("[FAIL] Count must be at least 1.")
        return 1

    try:
        selected = resolve_gpu(
            gpu
        )

        if selected.mig_mode.strip().lower() != "enabled":
            raise MigSafetyError(
                f"MIG mode is currently '{selected.mig_mode}'. "
                "Enable MIG first with: "
                f"`migops mode enable --gpu {selected.index}`"
            )

        profiles = query_profiles(
            selected.index
        )

        selected_profile = find_profile(
            profiles,
            profile,
        )

        if selected_profile.free < count:
            raise MigSafetyError(
                f"Profile {selected_profile.name} has only "
                f"{selected_profile.free} currently available "
                f"placement(s), but {count} were requested."
            )

    except (
        NvidiaSmiError,
        ValueError,
        MigSafetyError,
    ) as exc:
        print("[BLOCKED]")
        print()
        print(str(exc))
        return 1

    print(f"GPU:          {selected.index}")
    print(f"Model:        {selected.name}")
    print(f"Profile:      {selected_profile.name}")
    print(f"Count:        {count}")
    print("Compute CI:   automatic")
    print()

    command = build_easy_create_command(
        gpu=selected.index,
        profile=selected_profile.name,
        count=count,
    )

    if dry_run:
        print("Planned operation")
        print("-----------------")
    else:
        print("Executing")
        print("---------")

    result = execute_operation(
        command,
        dry_run=dry_run,
    )

    if dry_run:
        print()
        print("DRY RUN - no changes were made.")

    elif result == 0:
        print()
        print(
            "[PASS] MIG GPU Instance(s) and corresponding "
            "Compute Instance(s) created."
        )

    return result


def destroy_mig(
    gpu: str,
    *,
    gi: str | None = None,
    destroy_all: bool = False,
    dry_run: bool = False,
    force: bool = False,
) -> int:
    """
    Safely destroy complete MIG instances.

    CIs are removed before GIs automatically.
    """

    print()
    print("MIGOps Destroy")
    print("==============")
    print()

    try:
        selected = resolve_gpu(
            gpu
        )

        if selected.mig_mode.strip().lower() != "enabled":
            raise MigSafetyError(
                "MIG mode is not currently enabled."
            )

        check_workload_safety(
            gpu_index=selected.index,
            gi=None if destroy_all else gi,
            force=force,
        )

        existing_gis = query_gpu_instances(
            selected.index
        )

        if destroy_all:
            target_gis = existing_gis

        else:
            target_gis = [
                instance
                for instance in existing_gis
                if instance.gi_id == gi
            ]

        if not target_gis:
            if destroy_all:
                print("No GPU Instances exist on this GPU.")
            else:
                print(
                    f"GPU Instance {gi} was not found."
                )

            return 0

        if destroy_all:
            commands = build_destroy_sequence(
                selected.index
            )

        else:
            compute_instances = query_ci_instances(
                gpu=selected.index,
                gi=gi,
            )

            ci_ids = [
                instance.ci_id
                for instance in compute_instances
            ]

            commands = build_destroy_sequence(
                selected.index,
                gi=gi,
                ci_ids=ci_ids,
            )

    except (
        NvidiaSmiError,
        ValueError,
        MigSafetyError,
    ) as exc:
        print("[BLOCKED]")
        print()
        print(str(exc))
        return 1

    print(f"GPU:       {selected.index}")
    print(f"Model:     {selected.name}")

    if destroy_all:
        print(
            f"Target:    all {len(target_gis)} GPU Instance(s)"
        )
    else:
        print(
            f"Target:    GPU Instance {gi}"
        )

    print()
    print(
        "MIGOps will remove Compute Instances first, "
        "then GPU Instances."
    )
    print()

    return execute_sequence(
        commands,
        dry_run=dry_run,
    )


# ============================================================
# MIG MODE
# ============================================================


def mode_status(
    gpu: str,
) -> int:
    """Show MIG mode."""

    try:
        selected = resolve_gpu(
            gpu
        )

    except (
        NvidiaSmiError,
        ValueError,
    ) as exc:
        print()
        print("[FAIL] Unable to query MIG mode")
        print()
        print(str(exc))
        return 1

    print()
    print("MIGOps MIG Mode")
    print("===============")
    print()

    print(f"GPU:       {selected.index}")
    print(f"Model:     {selected.name}")
    print(f"MIG Mode:  {selected.mig_mode}")
    print()

    return 0


def set_mig_mode(
    gpu: str,
    *,
    enabled: bool,
    dry_run: bool = False,
    force: bool = False,
) -> int:
    """Enable or disable MIG mode safely."""

    desired = (
        "Enabled"
        if enabled
        else "Disabled"
    )

    try:
        selected = resolve_gpu(
            gpu
        )

        current = (
            selected.mig_mode
            .strip()
            .lower()
        )

        if current == desired.lower():
            print()
            print(
                f"MIG mode is already {desired.lower()} "
                f"on GPU {selected.index}."
            )
            return 0

        check_workload_safety(
            gpu_index=selected.index,
            force=force,
        )

        if not enabled:
            instances = query_gpu_instances(
                selected.index
            )

            if instances and not force:
                raise MigSafetyError(
                    f"{len(instances)} GPU Instance(s) still exist. "
                    "Run `migops destroy --gpu "
                    f"{selected.index} --all` first."
                )

    except (
        NvidiaSmiError,
        ValueError,
        MigSafetyError,
    ) as exc:
        print()
        print("[BLOCKED]")
        print()
        print(str(exc))
        return 1

    command = build_mode_command(
        selected.index,
        enabled,
    )

    print()
    print("MIGOps MIG Mode")
    print("===============")
    print()

    print(
        f"{selected.mig_mode} -> {desired}"
    )

    print()

    result = execute_operation(
        command,
        dry_run=dry_run,
    )

    if dry_run:
        print()
        print("DRY RUN - no changes were made.")

    return result


# ============================================================
# ADVANCED GI OPERATIONS
# ============================================================


def list_gi(
    gpu: str,
) -> int:
    """List GPU Instances."""

    print()
    print("MIGOps GPU Instances")
    print("====================")
    print()

    try:
        selected = resolve_gpu(
            gpu
        )

        instances = query_gpu_instances(
            selected.index
        )

    except (
        NvidiaSmiError,
        ValueError,
    ) as exc:
        print("[FAIL] Unable to list GPU Instances")
        print()
        print(str(exc))
        return 1

    if not instances:
        print("No GPU Instances detected.")
        print()
        return 0

    print(
        f"{'GPU':<6}"
        f"{'GI':<6}"
        f"{'PROFILE':<18}"
        f"{'PROFILE ID':<12}"
        f"PLACEMENT"
    )

    print("-" * 60)

    for instance in instances:
        print(
            f"{instance.gpu:<6}"
            f"{instance.gi_id:<6}"
            f"{instance.profile:<18}"
            f"{instance.profile_id:<12}"
            f"{instance.placement}"
        )

    print()

    return 0


def create_gi(
    gpu: str,
    profile: str,
    *,
    with_ci: bool = False,
    dry_run: bool = False,
) -> int:
    """Advanced low-level GI creation."""

    try:
        selected = resolve_gpu(
            gpu
        )

    except (
        NvidiaSmiError,
        ValueError,
    ) as exc:
        print()
        print("[FAIL]")
        print(str(exc))
        return 1

    command = build_gi_create_command(
        gpu=selected.index,
        profile=profile,
        with_ci=with_ci,
    )

    print()
    print("Advanced GI Create")
    print("==================")
    print()

    result = execute_operation(
        command,
        dry_run=dry_run,
    )

    if dry_run:
        print()
        print("DRY RUN - no changes were made.")

    return result


def delete_gi(
    gpu: str,
    *,
    gi: str | None,
    dry_run: bool = False,
    force: bool = False,
) -> int:
    """Advanced low-level GI deletion."""

    try:
        selected = resolve_gpu(
            gpu
        )

        check_workload_safety(
            gpu_index=selected.index,
            gi=gi,
            force=force,
        )

    except (
        NvidiaSmiError,
        ValueError,
        MigSafetyError,
    ) as exc:
        print()
        print("[BLOCKED]")
        print()
        print(str(exc))
        return 1

    command = build_gi_delete_command(
        gpu=selected.index,
        gi=gi,
    )

    print()

    result = execute_operation(
        command,
        dry_run=dry_run,
    )

    if dry_run:
        print()
        print("DRY RUN - no changes were made.")

    return result


# ============================================================
# ADVANCED CI OPERATIONS
# ============================================================


def list_ci(
    gpu: str,
    gi: str | None = None,
) -> int:
    """List Compute Instances."""

    print()
    print("MIGOps Compute Instances")
    print("========================")
    print()

    try:
        selected = resolve_gpu(
            gpu
        )

        instances = query_ci_instances(
            gpu=selected.index,
            gi=gi,
        )

    except (
        NvidiaSmiError,
        ValueError,
    ) as exc:
        print("[FAIL] Unable to list Compute Instances")
        print()
        print(str(exc))
        return 1

    if not instances:
        print("No Compute Instances detected.")
        print()
        return 0

    print(
        f"{'GPU':<6}"
        f"{'GI':<6}"
        f"{'CI':<6}"
        f"PROFILE"
    )

    print("-" * 50)

    for instance in instances:
        print(
            f"{instance.gpu:<6}"
            f"{instance.gi_id:<6}"
            f"{instance.ci_id:<6}"
            f"{instance.profile}"
        )

    print()

    return 0


def create_ci(
    gpu: str,
    gi: str,
    profile: str,
    *,
    dry_run: bool = False,
) -> int:
    """Advanced low-level CI creation."""

    try:
        selected = resolve_gpu(
            gpu
        )

    except (
        NvidiaSmiError,
        ValueError,
    ) as exc:
        print()
        print("[FAIL]")
        print(str(exc))
        return 1

    command = build_ci_create_command(
        gpu=selected.index,
        gi=gi,
        profile=profile,
    )

    print()

    result = execute_operation(
        command,
        dry_run=dry_run,
    )

    if dry_run:
        print()
        print("DRY RUN - no changes were made.")

    return result


def delete_ci(
    gpu: str,
    *,
    gi: str | None,
    ci: str | None,
    dry_run: bool = False,
    force: bool = False,
) -> int:
    """Advanced low-level CI deletion."""

    try:
        selected = resolve_gpu(
            gpu
        )

        check_workload_safety(
            gpu_index=selected.index,
            gi=gi,
            ci=ci,
            force=force,
        )

    except (
        NvidiaSmiError,
        ValueError,
        MigSafetyError,
    ) as exc:
        print()
        print("[BLOCKED]")
        print()
        print(str(exc))
        return 1

    command = build_ci_delete_command(
        gpu=selected.index,
        gi=gi,
        ci=ci,
    )

    print()

    result = execute_operation(
        command,
        dry_run=dry_run,
    )

    if dry_run:
        print()
        print("DRY RUN - no changes were made.")

    return result