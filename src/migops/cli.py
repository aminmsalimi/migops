import argparse
import sys

from migops import __version__
from migops.doctor import command_doctor
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

    # doctor
    subparsers.add_parser(
        "doctor",
        help="Diagnose the local NVIDIA GPU and MIG environment.",
    )

    # status
    subparsers.add_parser(
        "status",
        help="Show NVIDIA GPU and MIG status.",
    )

    # profiles
    profiles_parser = subparsers.add_parser(
        "profiles",
        help="Show supported NVIDIA MIG GPU instance profiles.",
    )

    profiles_parser.add_argument(
        "--gpu",
        help="GPU index, UUID, or PCI bus ID.",
    )

    profiles_parser.add_argument(
        "--json",
        action="store_true",
        help="Output profile information as JSON.",
    )

    # users
    users_parser = subparsers.add_parser(
        "users",
        help=(
            "Show processes currently using NVIDIA GPUs "
            "or MIG instances."
        ),
    )

    users_parser.add_argument(
        "--gpu",
        help="Only show processes using this GPU index.",
    )

    users_parser.add_argument(
        "--json",
        action="store_true",
        help="Output workload information as JSON.",
    )

    # split
    split_parser = subparsers.add_parser(
        "split",
        help=(
            "Recommend an equal MIG partition layout "
            "for a GPU."
        ),
    )

    split_parser.add_argument(
        "--gpu",
        default="0",
        help=(
            "GPU index, UUID, or PCI bus ID "
            "(default: 0)."
        ),
    )

    split_parser.add_argument(
        "--instances",
        type=int,
        required=True,
        help="Number of equal MIG instances requested.",
    )

    split_parser.add_argument(
        "--json",
        action="store_true",
        help="Output the split recommendation as JSON.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "doctor":
        sys.exit(
            command_doctor()
        )

    if args.command == "status":
        sys.exit(
            print_status()
        )

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

    if args.command == "split":
        sys.exit(
            plan_split(
                gpu_selector=args.gpu,
                instances=args.instances,
                json_output=args.json,
            )
        )

    parser.print_help()


if __name__ == "__main__":
    main()