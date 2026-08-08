"""MIGOps desired-state configuration handling."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from migops.nvidia import NvidiaSmiError
from migops.profiles import MigProfile, query_profiles
from migops.status import GPU, query_gpus


class ConfigError(RuntimeError):
    """Raised when a MIGOps configuration is invalid."""


@dataclass
class ProfileRequest:
    profile: str
    count: int


@dataclass
class GPUConfig:
    gpu: str
    mig_enabled: bool = True
    instances: list[ProfileRequest] = field(default_factory=list)


@dataclass
class MigOpsConfig:
    version: int
    gpus: list[GPUConfig]


@dataclass
class ValidationMessage:
    level: str
    message: str


@dataclass
class GPUValidationResult:
    selector: str
    gpu_index: str | None
    gpu_name: str | None
    valid: bool
    requested_memory_gib: float
    messages: list[ValidationMessage]


@dataclass
class ValidationResult:
    valid: bool
    config_version: int
    gpu_results: list[GPUValidationResult]


def parse_config_data(data: object) -> MigOpsConfig:
    """Convert raw YAML data to a validated MIGOps configuration."""

    if not isinstance(data, dict):
        raise ConfigError("Configuration root must be a YAML mapping.")

    version = data.get("version")

    if version != 1:
        raise ConfigError(
            "Unsupported configuration version. "
            "MIGOps currently supports version: 1."
        )

    raw_gpus = data.get("gpus")

    if not isinstance(raw_gpus, list) or not raw_gpus:
        raise ConfigError("`gpus` must be a non-empty list.")

    parsed_gpus: list[GPUConfig] = []
    seen_selectors: set[str] = set()

    for number, raw_gpu in enumerate(raw_gpus, start=1):
        if not isinstance(raw_gpu, dict):
            raise ConfigError(
                f"GPU entry #{number} must be a mapping."
            )

        selector = raw_gpu.get("gpu")

        if selector is None:
            raise ConfigError(
                f"GPU entry #{number} is missing `gpu`."
            )

        selector = str(selector)

        if selector in seen_selectors:
            raise ConfigError(
                f"GPU '{selector}' is defined more than once."
            )

        seen_selectors.add(selector)

        mig_enabled = raw_gpu.get("mig_enabled", True)

        if not isinstance(mig_enabled, bool):
            raise ConfigError(
                f"`mig_enabled` for GPU '{selector}' "
                "must be true or false."
            )

        raw_instances = raw_gpu.get("instances", [])

        if not isinstance(raw_instances, list):
            raise ConfigError(
                f"`instances` for GPU '{selector}' must be a list."
            )

        requests: list[ProfileRequest] = []
        seen_profiles: set[str] = set()

        for instance_number, raw_instance in enumerate(
            raw_instances,
            start=1,
        ):
            if not isinstance(raw_instance, dict):
                raise ConfigError(
                    f"Instance entry #{instance_number} "
                    f"for GPU '{selector}' must be a mapping."
                )

            profile = raw_instance.get("profile")
            count = raw_instance.get("count")

            if not isinstance(profile, str) or not profile.strip():
                raise ConfigError(
                    f"Instance entry #{instance_number} "
                    f"for GPU '{selector}' requires a profile."
                )

            profile = profile.strip()

            if profile in seen_profiles:
                raise ConfigError(
                    f"Profile '{profile}' is defined more than once "
                    f"for GPU '{selector}'. Combine the counts instead."
                )

            seen_profiles.add(profile)

            if (
                not isinstance(count, int)
                or isinstance(count, bool)
                or count < 1
            ):
                raise ConfigError(
                    f"Count for profile '{profile}' "
                    f"on GPU '{selector}' must be a positive integer."
                )

            requests.append(
                ProfileRequest(
                    profile=profile,
                    count=count,
                )
            )

        if not mig_enabled and requests:
            raise ConfigError(
                f"GPU '{selector}' has `mig_enabled: false` "
                "but also defines MIG instances."
            )

        parsed_gpus.append(
            GPUConfig(
                gpu=selector,
                mig_enabled=mig_enabled,
                instances=requests,
            )
        )

    return MigOpsConfig(
        version=version,
        gpus=parsed_gpus,
    )


def load_config(path: str | Path) -> MigOpsConfig:
    """Load a MIGOps YAML configuration."""

    config_path = Path(path)

    if not config_path.exists():
        raise ConfigError(
            f"Configuration file not found: {config_path}"
        )

    if not config_path.is_file():
        raise ConfigError(
            f"Configuration path is not a file: {config_path}"
        )

    try:
        content = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(
            f"Unable to read configuration: {exc}"
        ) from exc

    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise ConfigError(
            f"Invalid YAML: {exc}"
        ) from exc

    return parse_config_data(data)


def resolve_gpu(
    selector: str,
    available_gpus: list[GPU],
) -> GPU | None:
    """Resolve a GPU by index, UUID, or PCI bus ID."""

    selector_lower = selector.lower()

    for gpu in available_gpus:
        if gpu.index == selector:
            return gpu

        if gpu.uuid.lower() == selector_lower:
            return gpu

        if gpu.pci_bus_id.lower() == selector_lower:
            return gpu

    return None


def find_profile(
    requested: str,
    profiles: list[MigProfile],
) -> MigProfile | None:
    """Resolve a profile by profile name or NVIDIA profile ID."""

    requested_lower = requested.lower()

    for profile in profiles:
        if profile.name.lower() == requested_lower:
            return profile

        if profile.profile_id == requested:
            return profile

    return None


def canonical_profile_counts(
    desired: GPUConfig,
    profiles: list[MigProfile],
) -> dict[str, int]:
    """
    Return desired profile counts using canonical NVIDIA profile names.

    This keeps profile IDs and profile names equivalent throughout diff,
    plan, apply, and verification.
    """

    counts: dict[str, int] = {}

    for request in desired.instances:
        supported = find_profile(
            request.profile,
            profiles,
        )

        if supported is None:
            raise ConfigError(
                f"Profile '{request.profile}' is not supported "
                "or was not reported by the NVIDIA driver."
            )

        counts[supported.name] = (
            counts.get(supported.name, 0)
            + request.count
        )

    return counts


def validate_gpu_config(
    desired: GPUConfig,
    actual_gpu: GPU,
    profiles: list[MigProfile],
) -> GPUValidationResult:
    """Validate one desired GPU configuration."""

    messages: list[ValidationMessage] = []
    valid = True
    requested_memory = 0.0
    current_mode = actual_gpu.mig_mode.strip().lower()

    if desired.mig_enabled:
        if current_mode == "enabled":
            messages.append(
                ValidationMessage(
                    level="PASS",
                    message="MIG mode is enabled.",
                )
            )

        elif current_mode == "disabled":
            messages.append(
                ValidationMessage(
                    level="INFO",
                    message=(
                        "MIG mode is currently disabled and must be "
                        "enabled before applying this configuration."
                    ),
                )
            )

        else:
            valid = False
            messages.append(
                ValidationMessage(
                    level="FAIL",
                    message=(
                        f"GPU reports MIG state '{actual_gpu.mig_mode}'. "
                        "MIG capability could not be confirmed."
                    ),
                )
            )

    else:
        messages.append(
            ValidationMessage(
                level="PASS" if current_mode == "disabled" else "INFO",
                message=(
                    "MIG mode is already disabled."
                    if current_mode == "disabled"
                    else "Desired state requires MIG mode to be disabled."
                ),
            )
        )

        return GPUValidationResult(
            selector=desired.gpu,
            gpu_index=actual_gpu.index,
            gpu_name=actual_gpu.name,
            valid=valid,
            requested_memory_gib=0.0,
            messages=messages,
        )

    canonical_seen: set[str] = set()

    for request in desired.instances:
        supported = find_profile(
            request.profile,
            profiles,
        )

        if supported is None:
            valid = False
            messages.append(
                ValidationMessage(
                    level="FAIL",
                    message=(
                        f"Profile '{request.profile}' is not supported "
                        "or was not reported by the NVIDIA driver."
                    ),
                )
            )
            continue

        if supported.name in canonical_seen:
            valid = False
            messages.append(
                ValidationMessage(
                    level="FAIL",
                    message=(
                        f"Profile '{request.profile}' resolves to "
                        f"'{supported.name}', which is already defined. "
                        "Combine the counts into one entry."
                    ),
                )
            )
            continue

        canonical_seen.add(supported.name)

        if request.count > supported.total:
            valid = False
            messages.append(
                ValidationMessage(
                    level="FAIL",
                    message=(
                        f"Profile {supported.name} supports at most "
                        f"{supported.total} instance(s), but "
                        f"{request.count} were requested."
                    ),
                )
            )
        else:
            messages.append(
                ValidationMessage(
                    level="PASS",
                    message=(
                        f"{request.count} x {supported.name} is within "
                        f"the profile's reported maximum of "
                        f"{supported.total}."
                    ),
                )
            )

        if supported.memory_gib is not None:
            requested_memory += (
                supported.memory_gib
                * request.count
            )

    if len(desired.instances) > 1:
        messages.append(
            ValidationMessage(
                level="INFO",
                message=(
                    "Mixed-profile layout detected. Exact placement "
                    "compatibility is ultimately verified by the "
                    "NVIDIA driver during apply."
                ),
            )
        )

    return GPUValidationResult(
        selector=desired.gpu,
        gpu_index=actual_gpu.index,
        gpu_name=actual_gpu.name,
        valid=valid,
        requested_memory_gib=requested_memory,
        messages=messages,
    )


def validate_config(
    config: MigOpsConfig,
) -> ValidationResult:
    """Validate desired configuration against detected NVIDIA hardware."""

    try:
        available_gpus = query_gpus()
    except NvidiaSmiError as exc:
        raise ConfigError(
            f"Unable to query NVIDIA GPUs: {exc}"
        ) from exc

    results: list[GPUValidationResult] = []

    for desired_gpu in config.gpus:
        actual_gpu = resolve_gpu(
            desired_gpu.gpu,
            available_gpus,
        )

        if actual_gpu is None:
            results.append(
                GPUValidationResult(
                    selector=desired_gpu.gpu,
                    gpu_index=None,
                    gpu_name=None,
                    valid=False,
                    requested_memory_gib=0.0,
                    messages=[
                        ValidationMessage(
                            level="FAIL",
                            message=(
                                f"GPU '{desired_gpu.gpu}' was not found."
                            ),
                        )
                    ],
                )
            )
            continue

        profiles: list[MigProfile] = []

        if desired_gpu.mig_enabled:
            try:
                profiles = query_profiles(actual_gpu.index)
            except NvidiaSmiError as exc:
                results.append(
                    GPUValidationResult(
                        selector=desired_gpu.gpu,
                        gpu_index=actual_gpu.index,
                        gpu_name=actual_gpu.name,
                        valid=False,
                        requested_memory_gib=0.0,
                        messages=[
                            ValidationMessage(
                                level="FAIL",
                                message=(
                                    "Unable to query MIG profiles: "
                                    f"{exc}"
                                ),
                            )
                        ],
                    )
                )
                continue

        results.append(
            validate_gpu_config(
                desired=desired_gpu,
                actual_gpu=actual_gpu,
                profiles=profiles,
            )
        )

    return ValidationResult(
        valid=all(result.valid for result in results),
        config_version=config.version,
        gpu_results=results,
    )


def print_validation(
    path: str,
    json_output: bool = False,
) -> int:
    """Load, validate, and print a desired-state configuration."""

    try:
        config = load_config(path)
        result = validate_config(config)

    except ConfigError as exc:
        if json_output:
            print(
                json.dumps(
                    {
                        "valid": False,
                        "error": str(exc),
                    },
                    indent=2,
                )
            )
        else:
            print()
            print("MIGOps Validate")
            print("===============")
            print()
            print("[FAIL]")
            print(str(exc))

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
    print("MIGOps Validate")
    print("===============")
    print()
    print(f"Configuration: {path}")
    print(f"Version:       {config.version}")
    print()

    for gpu_result in result.gpu_results:
        print(f"GPU {gpu_result.selector}")
        print("-" * 60)

        if gpu_result.gpu_name:
            print(f"Detected: {gpu_result.gpu_name}")

        if gpu_result.gpu_index is not None:
            print(f"Index:    {gpu_result.gpu_index}")

        if gpu_result.requested_memory_gib:
            print(
                "Requested profile memory: "
                f"{gpu_result.requested_memory_gib:.2f} GiB"
            )

        print()

        for message in gpu_result.messages:
            print(
                f"[{message.level}] {message.message}"
            )

        print()

    print(
        "Configuration result: VALID"
        if result.valid
        else "Configuration result: INVALID"
    )

    return 0 if result.valid else 1
