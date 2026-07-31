"""Shared helpers for the packaged interactive applications."""

from __future__ import annotations

import os
import sys
from importlib.resources import files
from pathlib import Path


def default_project_root() -> Path:
    """Return the configured project root, falling back to the launch directory."""

    return Path(os.environ.get("SPATIOEV_PROJECT_ROOT", Path.cwd())).expanduser().resolve()


def module_command(module: str, *arguments: object) -> list[str]:
    """Build a subprocess command that also works after wheel installation."""

    return [sys.executable, "-m", module, *(str(argument) for argument in arguments)]


def resource_path(filename: str) -> Path:
    """Return the installed path of a bundled example resource."""

    return Path(str(files("spatioev.resources").joinpath(filename)))
