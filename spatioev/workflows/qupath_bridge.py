#!/usr/bin/env python3
"""Background worker: QuPath bridge for many cores at once.

``prepare``  write ``<core>/qupath/<core>_cells.geojson`` for every core, ready
             for ``spatioev_import_cells.groovy`` (Run for project).
``collect``  after ``spatioev_export_classes.groovy``, turn every core's QuPath
             classes into ``<core>/qupath/<core>_phenotyped.h5ad`` and write a
             class-count table across cores.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import pandas as pd

from spatioev.io.qupath import (
    core_status,
    find_cores,
    qupath_phenotyped_anndata,
    write_qupath_cells,
)

from ._io import now, write_json


def update_status(path: Path | None, state: str, message: str, *, stage: str, progress: float, **extra) -> None:
    if path is None:
        return
    write_json(path, {"state": state, "message": message, "stage": stage, "progress": progress, "updated_at": now(), **extra})


def scan(roots, table: str = "arcsinh_transformed") -> pd.DataFrame:
    """One row per core found under ``roots`` with which files exist."""
    return pd.DataFrame([core_status(core, table) for core in find_cores(roots)])


def _selected(config: dict) -> list[Path]:
    cores = find_cores(config["roots"])
    wanted = set(config.get("cores") or [])
    return [core for core in cores if not wanted or core.name in wanted]


def prepare(config: dict, status_path: Path | None = None, log=print) -> pd.DataFrame:
    """Write the QuPath cell GeoJSON of every selected core."""
    cores = _selected(config)
    rows = []
    for index, core in enumerate(cores):
        target = core / "qupath" / f"{core.name}_cells.geojson"
        update_status(status_path, "running", f"{core.name} ({index + 1}/{len(cores)})", stage="prepare", progress=round(0.02 + 0.96 * index / max(1, len(cores)), 4))
        if target.exists() and not config.get("overwrite", False):
            rows.append({"core": core.name, "status": "exists", "path": str(target)})
            continue
        try:
            result = write_qupath_cells(
                core,
                table=config.get("table", "arcsinh_transformed"),
                include_nuclear=bool(config.get("include_nuclear", True)),
            )
            result.pop("markers", None)
            rows.append({**result, "status": "written"})
            log(f"{core.name}: {result['cells']} cells ({result['cells_without_nucleus']} without a nucleus)")
        except Exception as error:  # keep going: one bad core must not stop 200
            rows.append({"core": core.name, "status": "failed", "error": str(error)})
            log(f"{core.name}: FAILED - {error}")
    return pd.DataFrame(rows)


def collect(config: dict, status_path: Path | None = None, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build every selected core's phenotyped AnnData from its QuPath export."""
    from spatioev.workflows.cellsam import write_h5ad_atomically

    cores = _selected(config)
    rows, counts = [], []
    for index, core in enumerate(cores):
        update_status(status_path, "running", f"{core.name} ({index + 1}/{len(cores)})", stage="collect", progress=round(0.02 + 0.96 * index / max(1, len(cores)), 4))
        classes = core / "qupath" / f"{core.name}_cell_classes.csv"
        if not classes.exists():
            rows.append({"core": core.name, "status": "no QuPath export"})
            continue
        try:
            adata = qupath_phenotyped_anndata(
                core,
                table=config.get("table", "arcsinh_transformed"),
                drop_excluded=bool(config.get("drop_excluded", True)),
            )
            target = core / "qupath" / f"{core.name}_phenotyped.h5ad"
            write_h5ad_atomically(adata, target)
            info = adata.uns["qupath_bridge"]
            rows.append({
                "core": core.name, "status": "written", "path": str(target), "cells": int(adata.n_obs),
                "excluded": info["n_excluded"], "classified": info["n_cells_classified"],
            })
            for name, count in adata.obs["qupath_class"].value_counts().items():
                counts.append({"core": core.name, "qupath_class": name, "count": int(count)})
            log(f"{core.name}: {adata.n_obs} cells, {info['n_excluded']} excluded")
        except Exception as error:
            rows.append({"core": core.name, "status": "failed", "error": str(error)})
            log(f"{core.name}: FAILED - {error}")
    return pd.DataFrame(rows), pd.DataFrame(counts, columns=["core", "qupath_class", "count"])


def run_workflow(config: dict, status_path: Path) -> dict:
    output_dir = Path(config["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = config["mode"]
    if mode == "prepare":
        summary = prepare(config, status_path)
        summary_path = output_dir / "qupath_prepare_summary.csv"
        summary.to_csv(summary_path, index=False)
        written = int((summary.get("status") == "written").sum()) if len(summary) else 0
        failed = int((summary.get("status") == "failed").sum()) if len(summary) else 0
        outputs = {"mode": mode, "summary": str(summary_path), "written": written, "failed": failed, "cores": int(len(summary))}
        message = f"Wrote QuPath cell files for {written} core{'' if written == 1 else 's'}" + (f"; {failed} failed" if failed else "")
    elif mode == "collect":
        summary, counts = collect(config, status_path)
        summary_path = output_dir / "qupath_collect_summary.csv"
        counts_path = output_dir / "qupath_class_counts.csv"
        summary.to_csv(summary_path, index=False)
        counts.to_csv(counts_path, index=False)
        written = int((summary.get("status") == "written").sum()) if len(summary) else 0
        failed = int((summary.get("status") == "failed").sum()) if len(summary) else 0
        outputs = {"mode": mode, "summary": str(summary_path), "class_counts": str(counts_path), "written": written, "failed": failed, "cores": int(len(summary))}
        message = f"Built phenotyped AnnData for {written} core{'' if written == 1 else 's'}" + (f"; {failed} failed" if failed else "")
    else:
        raise ValueError(f"Unknown mode {mode!r}; expected 'prepare' or 'collect'")
    write_json(output_dir / f"qupath_{mode}_manifest.json", {"created_at": now(), "config": config, "outputs": outputs})
    update_status(status_path, "complete", message, stage="complete", progress=1.0, outputs=outputs)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    try:
        run_workflow(config, args.status)
    except Exception as error:
        update_status(args.status, "failed", str(error), stage="failed", progress=1.0, traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
