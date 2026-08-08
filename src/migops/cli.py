import argparse
import platform
import shutil
import subprocess
import sys

from migops import __version__
from migops.profiles import print_profiles
from migops.status import print_status


def print_check(status: str, message: str) -> None:
    print(f"[{status}] {message}")


def command_doctor() -> int:
    """Check whether the local system is ready for MIGOps."""

    print()
    print("MIGOps Environment Check")
    print("========================")
    print()

    system = platform.system()

    if system == "Linux":
        print_check("OK", f"Operating system: {system}")
    else:
        print_check(
            "WARN",
            f"Operating system: {system} "
            "(MIGOps is intended for Linux GPU hosts)",
        )

    print_check("OK", f"Python: {platform.python_version()}")

    nvidia_smi = shutil.which("nvidia-smi")

    if not nvidia_smi:
        print_check("FAIL", "nvidia-smi not found")
        print()
        print(
            "NVIDIA drivers may not be installed "
            "or nvidia-smi is not in PATH."
        )
        return 1

    print_check("OK", f"nvidia-smi found: {nvidia_smi}")

    try:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,name,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        print_check(
            "FAIL",
            f"Could not execute nvidia-smi: {exc}",
        )
        return 1

    if result.returncode != 0:
        print_check("FAIL", "nvidia-smi returned an error")

        if result.stderr:
            print()
            print(result.stderr.strip())

        return 1

    gpu_output = result.stdout.strip()

    if not gpu_output:
        print_check("FAIL", "No NVIDIA GPUs detected")
        return 1

    print_check("OK", "NVIDIA GPU detected")

    print()
    print("Detected GPUs")
    print("-------------")

    for line in gpu_output.splitlines():
        print(line)

    print()
    print("Environment looks ready for further MIG checks.")

    return 0


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

    subparsers = parser.add_subparsers(dest="command")

    # doctor
    subparsers.add_parser(
        "doctor",
        help="Check the local NVIDIA GPU environment.",
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

    parser.print_help()


if __name__ == "__main__":
    main()