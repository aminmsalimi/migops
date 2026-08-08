"""MIGOps command-line interface."""

from __future__ import annotations

import argparse
import sys

from migops import __version__
from migops.apply import (
    apply_config,
    restore_snapshot,
)
from migops.config import print_validation
from migops.diffing import print_diff
from migops.lifecycle import (
    create_ci,
    create_gi,
    create_mig,
    delete_ci,
    delete_gi,
    destroy_mig,
    list_ci,
    list_gi,
    set_mig_mode,
)
from migops.planner import print_plan
from migops.profiles import print_profiles
from migops.snapshot import print_snapshot
from migops.split import plan_split
from migops.status import print_status
from migops.workloads import print_users


def _add_required_gpu(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "target",
        choices=("gpu",),
        metavar="gpu",
        help="Target type. Currently: gpu",
    )
    parser.add_argument(
        "gpu",
        help="GPU index, UUID, or supported selector.",
    )


def _add_optional_gpu(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "target",
        nargs="?",
        choices=("gpu",),
        metavar="gpu",
        help="Optional target type: gpu",
    )
    parser.add_argument(
        "gpu",
        nargs="?",
        help="GPU index, UUID, or supported selector.",
    )


def _optional_gpu_value(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> str | None:
    if args.target is None and args.gpu is None:
        return None

    if args.target == "gpu" and args.gpu is not None:
        return args.gpu

    parser.error("GPU selector must use: gpu <GPU>")
    return None


def _confirm(
    message: str,
    *,
    yes: bool,
) -> bool:
    if yes:
        return True

    if not sys.stdin.isatty():
        print(
            "[BLOCKED] Confirmation required. "
            "Use --yes for non-interactive execution."
        )
        return False

    answer = input(f"{message} [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


def _parse_ci_list_scope(
    rest: list[str],
    parser: argparse.ArgumentParser,
) -> str | None:
    if not rest:
        return None

    if len(rest) == 2 and rest[0] == "gi":
        return rest[1]

    parser.error("Use: migops ci list gpu <GPU> [gi <GI>]")
    return None


def _parse_ci_delete_scope(
    rest: list[str],
    destroy_all: bool,
    parser: argparse.ArgumentParser,
) -> tuple[str | None, str | None]:
    if destroy_all:
        if not rest:
            return None, None

        if len(rest) == 2 and rest[0] == "gi":
            return rest[1], None

        parser.error(
            "Use: migops ci delete gpu <GPU> [gi <GI>] --all"
        )

    if (
        len(rest) == 4
        and rest[0] == "gi"
        and rest[2] == "ci"
    ):
        return rest[1], rest[3]

    parser.error(
        "Use: migops ci delete gpu <GPU> gi <GI> ci <CI> "
        "or add --all"
    )
    return None, None


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

    subparsers.add_parser(
        "status",
        help="Show system, NVIDIA GPU, MIG and workload status.",
    )

    profiles_parser = subparsers.add_parser(
        "profiles",
        help="Show supported MIG profiles.",
    )
    _add_optional_gpu(profiles_parser)
    profiles_parser.add_argument(
        "--json",
        action="store_true",
    )

    users_parser = subparsers.add_parser(
        "users",
        help="Show processes using NVIDIA GPUs or MIG devices.",
    )
    _add_optional_gpu(users_parser)
    users_parser.add_argument(
        "--json",
        action="store_true",
    )

    enable_parser = subparsers.add_parser(
        "enable",
        help="Enable MIG mode on a GPU.",
    )
    _add_required_gpu(enable_parser)
    enable_parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    enable_parser.add_argument(
        "--force",
        action="store_true",
    )
    enable_parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive confirmation.",
    )

    disable_parser = subparsers.add_parser(
        "disable",
        help="Disable MIG mode on a GPU.",
    )
    _add_required_gpu(disable_parser)
    disable_parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    disable_parser.add_argument(
        "--force",
        action="store_true",
    )
    disable_parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive confirmation.",
    )

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a MIGOps YAML configuration.",
    )
    validate_parser.add_argument("config")
    validate_parser.add_argument(
        "--json",
        action="store_true",
    )

    diff_parser = subparsers.add_parser(
        "diff",
        help="Compare desired YAML with current MIG state.",
    )
    diff_parser.add_argument("config")
    diff_parser.add_argument(
        "--json",
        action="store_true",
    )

    plan_parser = subparsers.add_parser(
        "plan",
        help="Show changes required to reach desired state.",
    )
    plan_parser.add_argument("config")
    plan_parser.add_argument(
        "--json",
        action="store_true",
    )

    snapshot_parser = subparsers.add_parser(
        "snapshot",
        help="Save current MIG state as reusable YAML.",
    )
    _add_optional_gpu(snapshot_parser)
    snapshot_parser.add_argument("--output")

    apply_parser = subparsers.add_parser(
        "apply",
        help="Apply a desired MIGOps YAML configuration safely.",
    )
    apply_parser.add_argument("config")
    apply_parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    apply_parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive confirmation.",
    )
    apply_parser.add_argument(
        "--force",
        action="store_true",
    )
    apply_parser.add_argument(
        "--snapshot-dir",
        default="snapshots",
    )

    restore_parser = subparsers.add_parser(
        "restore",
        help="Restore a MIGOps snapshot YAML.",
    )
    restore_parser.add_argument("snapshot")
    restore_parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    restore_parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive confirmation.",
    )
    restore_parser.add_argument(
        "--force",
        action="store_true",
    )

    recommend_parser = subparsers.add_parser(
        "recommend",
        help="Recommend an equal MIG split without changing the GPU.",
    )
    _add_required_gpu(recommend_parser)
    recommend_parser.add_argument(
        "instances",
        type=int,
        help="Requested number of equal MIG instances.",
    )
    recommend_parser.add_argument(
        "--json",
        action="store_true",
    )

    split_parser = subparsers.add_parser(
        "split",
        help="Split a GPU into equal MIG instances.",
    )
    _add_required_gpu(split_parser)
    split_parser.add_argument(
        "instances",
        type=int,
        help="Number of equal MIG instances to create.",
    )
    split_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the split without changing the GPU.",
    )
    split_parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive confirmation.",
    )
    split_parser.add_argument(
        "--force",
        action="store_true",
    )

    create_parser = subparsers.add_parser(
        "create",
        help="Create complete MIG instances (GI + CI automatically).",
    )
    _add_required_gpu(create_parser)
    create_parser.add_argument(
        "profile",
        help="MIG profile name or supported profile ID.",
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

    destroy_parser = subparsers.add_parser(
        "destroy",
        help="Safely destroy complete MIG instances.",
    )
    _add_required_gpu(destroy_parser)
    destroy_target = destroy_parser.add_mutually_exclusive_group(
        required=True
    )
    destroy_target.add_argument(
        "--gi",
        help="GPU Instance ID to destroy.",
    )
    destroy_target.add_argument(
        "--all",
        action="store_true",
        help="Destroy all MIG instances on the GPU.",
    )
    destroy_parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    destroy_parser.add_argument(
        "--force",
        action="store_true",
    )
    destroy_parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive confirmation.",
    )

    gi_parser = subparsers.add_parser(
        "gi",
        help="Advanced GPU Instance management.",
    )
    gi_subparsers = gi_parser.add_subparsers(
        dest="gi_action",
        required=True,
    )

    gi_list = gi_subparsers.add_parser(
        "list",
        help="List GPU Instances.",
    )
    _add_required_gpu(gi_list)

    gi_create = gi_subparsers.add_parser(
        "create",
        help="Create a GPU Instance.",
    )
    _add_required_gpu(gi_create)
    gi_create.add_argument("profile")
    gi_create.add_argument(
        "--with-ci",
        action="store_true",
    )
    gi_create.add_argument(
        "--dry-run",
        action="store_true",
    )

    gi_delete = gi_subparsers.add_parser(
        "delete",
        help="Delete a GPU Instance.",
    )
    _add_required_gpu(gi_delete)
    gi_delete.add_argument(
        "gi",
        nargs="?",
        help="GPU Instance ID.",
    )
    gi_delete.add_argument(
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
    gi_delete.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive confirmation.",
    )

    ci_parser = subparsers.add_parser(
        "ci",
        help="Advanced Compute Instance management.",
    )
    ci_subparsers = ci_parser.add_subparsers(
        dest="ci_action",
        required=True,
    )

    ci_list = ci_subparsers.add_parser(
        "list",
        help="List Compute Instances.",
    )
    _add_required_gpu(ci_list)
    ci_list.add_argument(
        "scope",
        nargs="*",
        metavar="gi",
    )

    ci_create = ci_subparsers.add_parser(
        "create",
        help="Create a Compute Instance.",
    )
    _add_required_gpu(ci_create)
    ci_create.add_argument(
        "gi_word",
        choices=("gi",),
        metavar="gi",
    )
    ci_create.add_argument("gi")
    ci_create.add_argument("profile")
    ci_create.add_argument(
        "--dry-run",
        action="store_true",
    )

    ci_delete = ci_subparsers.add_parser(
        "delete",
        help="Delete Compute Instances.",
    )
    _add_required_gpu(ci_delete)
    ci_delete.add_argument(
        "scope",
        nargs="*",
    )
    ci_delete.add_argument(
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
    ci_delete.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive confirmation.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "status":
        sys.exit(print_status())

    if args.command == "profiles":
        gpu = _optional_gpu_value(args, parser)
        sys.exit(print_profiles(gpu, args.json))

    if args.command == "users":
        gpu = _optional_gpu_value(args, parser)
        sys.exit(print_users(gpu, args.json))

    if args.command in {"enable", "disable"}:
        enabled = args.command == "enable"

        if (
            not args.dry_run
            and not _confirm(
                f"{'Enable' if enabled else 'Disable'} MIG mode "
                f"on GPU {args.gpu}?",
                yes=args.yes,
            )
        ):
            print("No changes have been made.")
            sys.exit(1)

        sys.exit(
            set_mig_mode(
                args.gpu,
                enabled=enabled,
                dry_run=args.dry_run,
                force=args.force,
            )
        )

    if args.command == "validate":
        sys.exit(print_validation(args.config, args.json))

    if args.command == "diff":
        sys.exit(print_diff(args.config, args.json))

    if args.command == "plan":
        sys.exit(print_plan(args.config, args.json))

    if args.command == "snapshot":
        gpu = _optional_gpu_value(args, parser)
        sys.exit(print_snapshot(args.output, gpu))

    if args.command == "apply":
        approved = args.yes

        if not args.dry_run and not approved:
            approved = _confirm(
                f"Apply desired state from {args.config}?",
                yes=False,
            )

            if not approved:
                print("No changes have been made.")
                sys.exit(1)

        sys.exit(
            apply_config(
                args.config,
                dry_run=args.dry_run,
                yes=approved,
                force=args.force,
                snapshot_dir=args.snapshot_dir,
            )
        )

    if args.command == "restore":
        approved = args.yes

        if not args.dry_run and not approved:
            approved = _confirm(
                f"Restore MIG state from {args.snapshot}?",
                yes=False,
            )

            if not approved:
                print("No changes have been made.")
                sys.exit(1)

        sys.exit(
            restore_snapshot(
                args.snapshot,
                dry_run=args.dry_run,
                yes=approved,
                force=args.force,
            )
        )

    if args.command == "recommend":
        sys.exit(
            plan_split(
                args.gpu,
                args.instances,
                args.json,
                False,
                False,
                False,
                False,
            )
        )

    if args.command == "split":
        approved = args.yes

        if not args.dry_run and not approved:
            approved = _confirm(
                f"Split GPU {args.gpu} into "
                f"{args.instances} MIG instances?",
                yes=False,
            )

            if not approved:
                print("No changes have been made.")
                sys.exit(1)

        sys.exit(
            plan_split(
                args.gpu,
                args.instances,
                False,
                True,
                args.dry_run,
                approved,
                args.force,
            )
        )

    if args.command == "create":
        sys.exit(
            create_mig(
                args.gpu,
                args.profile,
                count=args.count,
                dry_run=args.dry_run,
            )
        )

    if args.command == "destroy":
        if (
            not args.dry_run
            and not _confirm(
                (
                    f"Destroy all MIG instances on GPU {args.gpu}?"
                    if args.all
                    else f"Destroy GI {args.gi} on GPU {args.gpu}?"
                ),
                yes=args.yes,
            )
        ):
            print("No changes have been made.")
            sys.exit(1)

        sys.exit(
            destroy_mig(
                args.gpu,
                gi=args.gi,
                destroy_all=args.all,
                dry_run=args.dry_run,
                force=args.force,
            )
        )

    if args.command == "gi":
        if args.gi_action == "list":
            sys.exit(list_gi(args.gpu))

        if args.gi_action == "create":
            sys.exit(
                create_gi(
                    args.gpu,
                    args.profile,
                    with_ci=args.with_ci,
                    dry_run=args.dry_run,
                )
            )

        if bool(args.gi) == bool(args.all):
            parser.error(
                "Use either a GI ID or --all, for example: "
                "migops gi delete gpu 0 1"
            )

        if (
            not args.dry_run
            and not _confirm(
                (
                    f"Delete all GPU Instances on GPU {args.gpu}?"
                    if args.all
                    else f"Delete GI {args.gi} on GPU {args.gpu}?"
                ),
                yes=args.yes,
            )
        ):
            print("No changes have been made.")
            sys.exit(1)

        sys.exit(
            delete_gi(
                args.gpu,
                gi=None if args.all else args.gi,
                dry_run=args.dry_run,
                force=args.force,
            )
        )

    if args.command == "ci":
        if args.ci_action == "list":
            gi = _parse_ci_list_scope(
                args.scope,
                parser,
            )
            sys.exit(list_ci(args.gpu, gi))

        if args.ci_action == "create":
            sys.exit(
                create_ci(
                    args.gpu,
                    args.gi,
                    args.profile,
                    dry_run=args.dry_run,
                )
            )

        gi, ci = _parse_ci_delete_scope(
            args.scope,
            args.all,
            parser,
        )

        if (
            not args.dry_run
            and not _confirm(
                (
                    f"Delete matching Compute Instances on GPU {args.gpu}?"
                    if args.all
                    else f"Delete CI {ci} from GI {gi} on GPU {args.gpu}?"
                ),
                yes=args.yes,
            )
        ):
            print("No changes have been made.")
            sys.exit(1)

        sys.exit(
            delete_ci(
                args.gpu,
                gi=gi,
                ci=ci,
                dry_run=args.dry_run,
                force=args.force,
            )
        )

    parser.print_help()


if __name__ == "__main__":
    main()
