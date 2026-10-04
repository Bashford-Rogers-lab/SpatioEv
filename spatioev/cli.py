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


def _qupath_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "qupath",
        help="Exchange cells and classes with QuPath (one import, one export per core)",
        description="Bridge between SpatioEv segmentation and QuPath, keyed on cell_id. "
        "Typical use: 'prepare' -> import in QuPath -> classify -> export in QuPath -> 'collect'.",
    )
    actions = parser.add_subparsers(dest="qupath_command", required=True)

    def add_roots(sub):
        sub.add_argument("roots", nargs="+", type=Path, help="Core folders, or folders containing core folders")
        sub.add_argument("--cores", nargs="*", default=None, help="Only these core names")
        sub.add_argument(
            "--table", default="arcsinh_transformed",
            help="Cell table for measurements: arcsinh_transformed (default), size_normalized or raw",
        )

    status = actions.add_parser("status", help="List cores and which bridge files exist")
    add_roots(status)
    prepare = actions.add_parser("prepare", help="Write <core>/qupath/<core>_cells.geojson for QuPath to import")
    add_roots(prepare)
    prepare.add_argument("--overwrite", action="store_true", help="Rewrite cell files that already exist")
    prepare.add_argument("--no-nuclear", action="store_true", help="Leave out the '<marker> nucleus' measurements")
    collect = actions.add_parser("collect", help="Build <core>/qupath/<core>_phenotyped.h5ad from QuPath's export")
    add_roots(collect)
    collect.add_argument("--keep-excluded", action="store_true", help="Flag, rather than drop, cells in Exclude annotations")
    collect.add_argument("--counts", type=Path, default=None, help="Write the class-count table here (CSV)")
    scripts = actions.add_parser("scripts", help="Copy the QuPath scripts into a folder (e.g. QuPath's scripts folder)")
    scripts.add_argument("destination", type=Path)


def _batch_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "batch",
        help="Repeat one core's fiber segmentation, tissue regions or co-localisation run for every core",
        description="Run a stage once for one core, check it, then repeat it for every core with the same settings. "
        "Cores that are already done are skipped, so an interrupted batch can be started again.",
    )
    parser.add_argument("run", type=Path, help="Output folder of the finished run (or its *_config.json)")
    parser.add_argument("--roots", nargs="*", type=Path, default=None, help="Folders holding the core folders (default: the run's project folder)")
    parser.add_argument("--cores", nargs="*", default=None, help="Only these core names")
    parser.add_argument("--overwrite", action="store_true", help="Re-run cores that are already done")
    parser.add_argument("--jobs", type=int, default=1, help="Cores to process at the same time (each needs as much memory as one run)")


def _demo_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "demo",
        help="Create synthetic practice cores in the same layout as real data",
        description="Write synthetic TMA cores (image, segmentation masks, cell tables) and a sample sheet, "
        "for trying every step of the workflow. Contains no real patient data.",
    )
    parser.add_argument("destination", type=Path, help="Folder to create the demo cores in")
    parser.add_argument("--cores", type=int, default=6, help="Number of cores (half in group A, half in group B)")
    parser.add_argument("--size", type=int, default=1500, help="Core image size in pixels")
    parser.add_argument("--with-classes", action="store_true", help="Also write ready-made QuPath classes, to skip QuPath")
    parser.add_argument("--overwrite", action="store_true", help="Replace demo cores that already exist")
    parser.add_argument("--seed", type=int, default=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spatioev")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _ui_parser(subparsers)
    _qupath_parser(subparsers)
    _batch_parser(subparsers)
    _demo_parser(subparsers)
    return parser


def run_batch(args: argparse.Namespace) -> int:
    import pandas as pd

    from spatioev.workflows import batch

    config = {
        "template": str(args.run),
        "roots": [str(root) for root in args.roots] if args.roots else None,
        "cores": args.cores,
        "overwrite": args.overwrite,
        "jobs": args.jobs,
    }
    outputs = batch.run_workflow(config, None)
    summary = pd.read_csv(outputs["summary"]).fillna("")
    with pd.option_context("display.width", 200, "display.max_rows", 1000, "display.max_colwidth", 80):
        print(summary.to_string(index=False))
    print(f"Summary: {outputs['summary']}")
    return 1 if (summary["status"] == "failed").any() else 0


def run_demo(args: argparse.Namespace) -> int:
    from spatioev.io.demo import make_demo_tma

    cores = make_demo_tma(
        args.destination, n_cores=args.cores, size=args.size, seed=args.seed,
        with_classes=args.with_classes, overwrite=args.overwrite,
    )
    print(f"Created {len(cores)} demo cores in {args.destination.expanduser().resolve()}:")
    for core in cores:
        print(f"  {core.name}")
    print("Sample sheet: demo_sample_sheet.csv")
    return 0


def run_qupath(args: argparse.Namespace) -> int:
    import pandas as pd

    from spatioev.io.qupath import write_qupath_scripts
    from spatioev.workflows import qupath_bridge

    if args.qupath_command == "scripts":
        for path in write_qupath_scripts(args.destination):
            print(path)
        return 0
    config = {"roots": [str(root) for root in args.roots], "cores": args.cores, "table": args.table}
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        if args.qupath_command == "status":
            table = qupath_bridge.scan(config["roots"], args.table)
            if table.empty:
                print("No core folders (with segmentation/ and quantification/) found.")
                return 1
            print(table.drop(columns="path").to_string(index=False))
            return 0
        if args.qupath_command == "prepare":
            summary = qupath_bridge.prepare({**config, "overwrite": args.overwrite, "include_nuclear": not args.no_nuclear})
        else:
            summary, counts = qupath_bridge.collect({**config, "drop_excluded": not args.keep_excluded})
            if args.counts is not None and not counts.empty:
                counts.to_csv(args.counts, index=False)
                print(f"Class counts: {args.counts}")
            if not counts.empty:
                print(counts.pivot_table(index="core", columns="qupath_class", values="count", aggfunc="sum", fill_value=0).astype(int).to_string())
    if summary.empty:
        print("No core folders found.")
        return 1
    print(summary["status"].value_counts().to_string())
    return 1 if (summary["status"] == "failed").any() else 0


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
    if args.command == "qupath":
        return run_qupath(args)
    if args.command == "batch":
        return run_batch(args)
    if args.command == "demo":
        return run_demo(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
