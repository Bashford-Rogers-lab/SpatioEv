"""Command-line entry points for SpatioEv."""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from importlib.resources import files
from pathlib import Path


def _ui_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("ui", help="Launch the staged analysis interface")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--address", default="localhost")
    parser.add_argument("--no-browser", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spatioev")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _ui_parser(subparsers)
    return parser


def launch_ui(args: argparse.Namespace) -> int:
    if importlib.util.find_spec("streamlit") is None:
        raise SystemExit(
            "The interface dependencies are not installed. "
            "Install them with: pip install 'spatioev[apps]'"
        )
    os.environ["SPATIOEV_PROJECT_ROOT"] = str(args.project_root.expanduser().resolve())
    home = files("spatioev.apps").joinpath("Home.py")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(home),
        "--server.port",
        str(args.port),
        "--server.address",
        args.address,
        "--server.headless",
        "true" if args.no_browser else "false",
    ]
    try:
        return subprocess.call(command)
    except KeyboardInterrupt:
        return 130


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "ui":
        return launch_ui(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
