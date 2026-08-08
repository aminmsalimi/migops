import argparse
import sys

from migops import __version__
from migops.config import print_validation
from migops.doctor import command_doctor
from migops.lifecycle import (
    create_ci,
    create_gi,
    create_mig,
    delete_ci,
    delete_gi,
    destroy_mig,
    list_ci,
    list_gi,
    mode_status,
    set_mig_mode,
)
from migops.profiles import print_profiles
from migops.split import plan_split
from migops.status import print_status
from migops.workloads import print_users


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="migops",
        description=(
            "Safe operations, planning, validation and troubleshooting "
            "for NVIDIA MIG environments."
        ),
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"MIGOps {__version__}",
    )

    subparsers = parser.add_subparsers(
        dest="command",
    )

    # Inspection

    subparsers.add_parser(
        "doctor",
        help="Diagnose the NVIDIA GPU and MIG environment.",
    )

    subparsers.add_parser(
        "status",
        help="Show NVIDIA GPU and MIG status.",
    )

    profiles_parser = subparsers.add_parser(
        "profiles",
        help="Show supported MIG profiles.",
    )

    profiles_parser.add_argument(
        "--gpu",
        help="GPU index, UUID, or PCI bus ID.",
    )

    profiles_parser.add_argument(
        "--json",
        action="store_true",
    )

    users_parser = subparsers.add_parser(
        "users",
        help="Show processes using NVIDIA GPUs or MIG devices.",
    )

    users_parser.add_argument(
        "--gpu",
    )

    users_parser.add_argument(
        "--json",
        action="store_true",
    )

    # Validate configuration

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a MIGOps YAML configuration.",
    )

    validate_parser.add_argument(
        "config",
        help="Path to MIGOps YAML configuration.",
    )

    validate_parser.add_argument(
        "--json",
        action="store_true",
        help="Return validation results as JSON.",
    )

    # Smart split

    split_parser = subparsers.add_parser(
        "split",
        help="Recommend an equal MIG partition layout.",
    )

    split_parser.add_argument(
        "--gpu",
        default="0",
    )

    split_parser.add_argument(
        "--instances",
        type=int,
        required=True,
    )

    split_parser.add_argument(
        "--json",
        action="store_true",
    )

    # Easy create

    create_parser = subparsers.add_parser(
        "create",
        help="Create complete MIG instances (GI + CI automatically).",
    )

    create_parser.add_argument(
        "--gpu",
        default="0",
    )

    create_parser.add_argument(
        "--profile",
        required=True,
    )

    create_parser.add_argument(
        "--count",
        type=int,
        default=1,
    )

    create_parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    # Easy destroy

    destroy_parser = subparsers.add_parser(
        "destroy",
        help="Safely destroy complete MIG instances.",
    )

    destroy_parser.add_argument(
        "--gpu",
        default="0",
    )

    destroy_target = (
        destroy_parser.add_mutually_exclusive_group(
            required=True
        )
    )

    destroy_target.add_argument(
        "--gi",
    )

    destroy_target.add_argument(
        "--all",
        action="store_true",
    )

    destroy_parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    destroy_parser.add_argument(
        "--force",
        action="store_true",
    )

    # MIG mode

    mode_parser = subparsers.add_parser(
        "mode",
        help="Inspect, enable or disable MIG mode.",
    )

    mode_subparsers = mode_parser.add_subparsers(
        dest="mode_action",
        required=True,
    )

    mode_status_parser = mode_subparsers.add_parser(
        "status",
    )

    mode_status_parser.add_argument(
        "--gpu",
        default="0",
    )

    for action in (
        "enable",
        "disable",
    ):
        action_parser = mode_subparsers.add_parser(
            action
        )

        action_parser.add_argument(
            "--gpu",
            default="0",
        )

        action_parser.add_argument(
            "--dry-run",
            action="store_true",
        )

        action_parser.add_argument(
            "--force",
            action="store_true",
        )

    # Advanced GI

    gi_parser = subparsers.add_parser(
        "gi",
        help="Advanced GPU Instance management.",
    )

    gi_subparsers = gi_parser.add_subparsers(
        dest="gi_action",
        required=True,
    )

    gi_list = gi_subparsers.add_parser(
        "list"
    )

    gi_list.add_argument(
        "--gpu",
        default="0",
    )

    gi_create = gi_subparsers.add_parser(
        "create"
    )

    gi_create.add_argument(
        "--gpu",
        default="0",
    )

    gi_create.add_argument(
        "--profile",
        required=True,
    )

    gi_create.add_argument(
        "--with-ci",
        action="store_true",
    )

    gi_create.add_argument(
        "--dry-run",
        action="store_true",
    )

    gi_delete = gi_subparsers.add_parser(
        "delete"
    )

    gi_delete.add_argument(
        "--gpu",
        default="0",
    )

    gi_delete_group = (
        gi_delete.add_mutually_exclusive_group(
            required=True
        )
    )

    gi_delete_group.add_argument(
        "--gi",
    )

    gi_delete_group.add_argument(
        "--all",
        action="store_true",
    )

    gi_delete.add_argument(
        "--dry-run",
        action="store_true",
    )

    gi_delete.add_argument(
        "--force",
        action="store_true",
    )

    # Advanced CI

    ci_parser = subparsers.add_parser(
        "ci",
        help="Advanced Compute Instance management.",
    )

    ci_subparsers = ci_parser.add_subparsers(
        dest="ci_action",
        required=True,
    )

    ci_list = ci_subparsers.add_parser(
        "list"
    )

    ci_list.add_argument(
        "--gpu",
        default="0",
    )

    ci_list.add_argument(
        "--gi",
    )

    ci_create = ci_subparsers.add_parser(
        "create"
    )

    ci_create.add_argument(
        "--gpu",
        default="0",
    )

    ci_create.add_argument(
        "--gi",
        required=True,
    )

    ci_create.add_argument(
        "--profile",
        required=True,
    )

    ci_create.add_argument(
        "--dry-run",
        action="store_true",
    )

    ci_delete = ci_subparsers.add_parser(
        "delete"
    )

    ci_delete.add_argument(
        "--gpu",
        default="0",
    )

    ci_delete.add_argument(
        "--gi",
    )

    ci_delete_target = (
        ci_delete.add_mutually_exclusive_group(
            required=True
        )
    )

    ci_delete_target.add_argument(
        "--ci",
    )

    ci_delete_target.add_argument(
        "--all",
        action="store_true",
    )

    ci_delete.add_argument(
        "--dry-run",
        action="store_true",
    )

    ci_delete.add_argument(
        "--force",
        action="store_true",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "doctor":
        sys.exit(command_doctor())

    if args.command == "status":
        sys.exit(print_status())

    if args.command == "profiles":
        sys.exit(
            print_profiles(
                gpu=args.gpu,
                json_output=args.json,
            )
        )

    if args.command == "users":
        sys.exit(
            print_users(
                gpu=args.gpu,
                json_output=args.json,
            )
        )

    if args.command == "validate":
        sys.exit(
            print_validation(
                path=args.config,
                json_output=args.json,
            )
        )

    if args.command == "split":
        sys.exit(
            plan_split(
                gpu_selector=args.gpu,
                instances=args.instances,
                json_output=args.json,
            )
        )

    if args.command == "create":
        sys.exit(
            create_mig(
                gpu=args.gpu,
                profile=args.profile,
                count=args.count,
                dry_run=args.dry_run,
            )
        )

    if args.command == "destroy":
        sys.exit(
            destroy_mig(
                gpu=args.gpu,
                gi=args.gi,
                destroy_all=args.all,
                dry_run=args.dry_run,
                force=args.force,
            )
        )

    if args.command == "mode":

        if args.mode_action == "status":
            sys.exit(
                mode_status(
                    args.gpu
                )
            )

        if args.mode_action == "enable":
            sys.exit(
                set_mig_mode(
                    args.gpu,
                    enabled=True,
                    dry_run=args.dry_run,
                    force=args.force,
                )
            )

        if args.mode_action == "disable":
            sys.exit(
                set_mig_mode(
                    args.gpu,
                    enabled=False,
                    dry_run=args.dry_run,
                    force=args.force,
                )
            )

    if args.command == "gi":

        if args.gi_action == "list":
            sys.exit(
                list_gi(
                    args.gpu
                )
            )

        if args.gi_action == "create":
            sys.exit(
                create_gi(
                    gpu=args.gpu,
                    profile=args.profile,
                    with_ci=args.with_ci,
                    dry_run=args.dry_run,
                )
            )

        if args.gi_action == "delete":
            sys.exit(
                delete_gi(
                    gpu=args.gpu,
                    gi=None if args.all else args.gi,
                    dry_run=args.dry_run,
                    force=args.force,
                )
            )

    if args.command == "ci":

        if args.ci_action == "list":
            sys.exit(
                list_ci(
                    gpu=args.gpu,
                    gi=args.gi,
                )
            )

        if args.ci_action == "create":
            sys.exit(
                create_ci(
                    gpu=args.gpu,
                    gi=args.gi,
                    profile=args.profile,
                    dry_run=args.dry_run,
                )
            )

        if args.ci_action == "delete":

            if (
                args.ci is not None
                and args.gi is None
            ):
                parser.error(
                    "--gi is required when deleting a specific --ci"
                )

            sys.exit(
                delete_ci(
                    gpu=args.gpu,
                    gi=args.gi,
                    ci=None if args.all else args.ci,
                    dry_run=args.dry_run,
                    force=args.force,
                )
            )

    parser.print_help()


if __name__ == "__main__":
    main()