"""Helpers for interacting with NVIDIA's nvidia-smi utility."""

from __future__ import annotations

import shutil
import subprocess


class NvidiaSmiError(RuntimeError):
    """Raised when nvidia-smi cannot be executed successfully."""


def find_nvidia_smi() -> str | None:
    """Return the path to nvidia-smi if available."""
    return shutil.which("nvidia-smi")


def run_nvidia_smi(arguments: list[str]) -> str:
    """
    Execute nvidia-smi and return stdout.

    Raises NvidiaSmiError if nvidia-smi is unavailable or returns an error.
    """

    executable = find_nvidia_smi()

    if executable is None:
        raise NvidiaSmiError(
            "nvidia-smi was not found. "
            "Make sure the NVIDIA driver is installed and nvidia-smi is in PATH."
        )

    try:
        result = subprocess.run(
            [executable, *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise NvidiaSmiError(
            f"Could not execute nvidia-smi: {exc}"
        ) from exc

    if result.returncode != 0:
        error = result.stderr.strip() or result.stdout.strip()

        if not error:
            error = f"nvidia-smi exited with code {result.returncode}"

        raise NvidiaSmiError(error)

    return result.stdout.strip()