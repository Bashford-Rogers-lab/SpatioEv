#!/usr/bin/env python3
"""Background worker: repeat one core's run of stage 04, 05 or 06 for every core.

Run a stage once (in the interface, or by hand), check the result, then point
this at that run. Every other core folder is run with exactly the same
settings: only the core name in the paths and the sample ID change. Cores
whose output already exists are skipped, so an interrupted batch can simply
be started again.

The run's settings come from the config file the interface saves in the
output folder (``<core>_fiber_config.json``, ``<core>_tissue_regions_config.json``,
``<core>_colocalization_config.json``) or any config passed to a workflow with
``--config``.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from spatioev.io.qupath import find_cores

from ._io import now, read_json, write_json

__all__ = ["STAGES", "config_for_core", "detect_stage", "find_template", "run_batch"]

#: Per stage: workflow module, the file that marks a finished run, the inputs a
#: core must have, and optional inputs that are required when the template used them.
STAGES = {
    "fiber": {
        "label": "Fiber segmentation",
        "module": "spatioev.workflows.fiber_segmentation",
        "done": "{core}_fiber_manifest.json",
        "required": ("image_path",),
        "optional": ("adata_path",),
    },
    "regions": {
        "label": "Tissue regions",
        "module": "spatioev.workflows.tissue_regions",
        "done": "{core}_tissue_regions_manifest.json",
        "required": ("adata_path",),
        "optional": ("fiber_manifest",),
    },
    "coloc": {
        "label": "Co-localisation",
        "module": "spatioev.workflows.colocalization",
        "done": "{core}_colocalization_manifest.json",
        "required": ("adata_path",),
        "optional": ("fiber_manifest",),
    },
}
PATH_KEYS = ("image_path", "adata_path", "fiber_manifest", "output_dir")
_CONFIG_SUFFIXES = {"_fiber_config.json": "fiber", "_tissue_regions_config.json": "regions", "_colocalization_config.json": "coloc"}


def update_status(path: Path | None, state: str, message: str, *, stage: str, progress: float, **extra) -> None:
    if path is None:
        return
    write_json(path, {"state": state, "message": message, "stage": stage, "progress": progress, "updated_at": now(), **extra})


def detect_stage(config: dict) -> str:
    """Which stage a workflow config belongs to: ``fiber``, ``regions`` or ``coloc``."""
    if "channels" in config and "image_path" in config:
        return "fiber"
    if "tumour_labels" in config:
        return "regions"
    if "pairs" in config or "radii_um" in config or "matrix_phenotypes" in config:
        return "coloc"
    raise ValueError("Not a fiber segmentation, tissue regions or co-localisation config")


def find_template(path: str | Path) -> Path:
    """The config file for ``path``: the file itself, or the newest config in a run's output folder."""
    path = Path(path).expanduser().resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"No such file or folder: {path}")
    candidates = [p for p in path.glob("*_config.json") if any(p.name.endswith(s) for s in _CONFIG_SUFFIXES)]
    candidates += [p for p in [path / "config.json"] if p.exists()]
    if not candidates:
        raise FileNotFoundError(
            f"No run settings found in {path}. Point at the output folder of a finished fiber segmentation, "
            "tissue regions or co-localisation run (it holds a *_config.json file)."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def project_root(template: dict) -> Path:
    """The folder that holds ``results/``: the output folder is ``<root>/results/<core>_<stage>``."""
    return Path(template["output_dir"]).expanduser().resolve().parent.parent


def _swap(value: str, old: str, new: str, root: Path) -> str:
    """Replace the core name in a path, but only below ``root``."""
    path = Path(value).expanduser()
    try:
        relative = path.resolve().relative_to(root)
    except ValueError:
        return str(path.parent / path.name.replace(old, new))
    return str(root.joinpath(*(part.replace(old, new) for part in relative.parts)))


def config_for_core(template: dict, core: str, root: Path | None = None) -> dict:
    """``template`` with its core name replaced by ``core`` in the sample ID and every path."""
    old = str(template["sample_id"])
    root = root or project_root(template)
    config = copy.deepcopy(template)
    config["sample_id"] = core
    for key in PATH_KEYS:
        if isinstance(config.get(key), str) and config[key]:
            config[key] = _swap(config[key], old, core, root)
    return config


def _missing_inputs(config: dict, stage: str) -> list[str]:
    spec = STAGES[stage]
    missing = [key for key in spec["required"] if not config.get(key) or not Path(config[key]).expanduser().exists()]
    for key in spec["optional"]:
        if config.get(key) and not Path(config[key]).expanduser().exists():
            if stage == "fiber":
                config[key] = None  # only supplies image IDs; the image file name is used instead
            else:
                missing.append(key)
    return missing


_INPUT_HINTS = {
    "image_path": "image (background/<core>.ome.tiff)",
    "adata_path": "cell AnnData (QuPath collect, or Tissue regions for co-localisation)",
    "fiber_manifest": "fiber segmentation result (run stage 04 for this core first)",
}


def _run_one(core: str, config: dict, stage: str) -> tuple[str, float, str]:
    output_dir = Path(config["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = {"fiber": "fiber", "regions": "tissue_regions", "coloc": "colocalization"}[stage]
    config_path = output_dir / f"{core}_{prefix}_config.json"
    status_path = output_dir / f"{core}_{prefix}_status.json"
    log_path = output_dir / f"{core}_{prefix}_worker.log"
    write_json(config_path, config)
    started = time.monotonic()
    with log_path.open("ab") as log:
        code = subprocess.call(
            [sys.executable, "-m", STAGES[stage]["module"], "--config", str(config_path), "--status", str(status_path)],
            stdout=log, stderr=subprocess.STDOUT,
        )
    seconds = time.monotonic() - started
    status = read_json(status_path) or {}
    if code == 0 and status.get("state") == "complete":
        return "done", seconds, status.get("message", "")
    return "failed", seconds, status.get("message") or f"exit code {code}; see {log_path.name}"


def run_batch(
    template_path: str | Path,
    roots=None,
    cores=None,
    overwrite: bool = False,
    jobs: int = 1,
    status_path: Path | None = None,
    log=print,
) -> pd.DataFrame:
    """Run the template's stage for every core folder under ``roots``.

    Parameters
    ----------
    template_path : path
        A run's config file, or the output folder of a finished run.
    roots : path or list of paths, optional
        Folders holding the core folders. Default: the template's project root.
    cores : list of str, optional
        Only these core names.
    overwrite : bool
        Re-run cores whose output already exists (otherwise they are skipped).
    jobs : int
        Cores processed at the same time. Each one needs as much memory as a
        single run, so raise this only with memory to spare.

    Returns
    -------
    pandas.DataFrame
        One row per core: ``core``, ``status`` (``done``, ``skipped``,
        ``missing input`` or ``failed``), ``seconds``, ``message``.
    """
    template_file = find_template(template_path)
    template = json.loads(template_file.read_text(encoding="utf-8"))
    stage = detect_stage(template)
    root = project_root(template)
    unnamed = [key for key in PATH_KEYS if template.get(key) and str(template["sample_id"]) not in Path(template[key]).name]
    if unnamed:
        raise ValueError(
            f"The run for {template['sample_id']} used paths without the core name ({', '.join(unnamed)}), so every core "
            "would read or write the same files. Run that core again with the standard paths (the "
            "'Fill standard sample paths' button) and start the batch from that run."
        )
    found = find_cores(roots if roots else [root])
    wanted = set(cores or [])
    names = [core.name for core in found if not wanted or core.name in wanted]
    if not names:
        raise FileNotFoundError(f"No core folders (with segmentation/ and quantification/) under {roots or root}")
    label = STAGES[stage]["label"]
    log(f"{label}: {len(names)} cores, settings from {template_file}")

    rows, todo = [], []
    for name in names:
        config = config_for_core(template, name, root)
        done_file = Path(config["output_dir"]) / STAGES[stage]["done"].format(core=name)
        if done_file.exists() and not overwrite:
            rows.append({"core": name, "status": "skipped", "seconds": 0.0, "message": "already done"})
            continue
        missing = _missing_inputs(config, stage)
        if missing:
            message = "no " + "; no ".join(_INPUT_HINTS.get(key, key) for key in missing)
            rows.append({"core": name, "status": "missing input", "seconds": 0.0, "message": message})
            log(f"{name}: skipped - {message}")
            continue
        todo.append((name, config))

    total = len(todo)
    update_status(status_path, "running", f"{label}: 0 of {total} cores", stage="batch", progress=0.02)
    with ThreadPoolExecutor(max_workers=max(1, int(jobs))) as pool:
        futures = {pool.submit(_run_one, name, config, stage): name for name, config in todo}
        for finished, future in enumerate(as_completed(futures), start=1):
            name = futures[future]
            try:
                state, seconds, message = future.result()
            except Exception as error:  # the subprocess could not even start
                state, seconds, message = "failed", 0.0, str(error)
            rows.append({"core": name, "status": state, "seconds": round(seconds, 1), "message": message})
            log(f"[{finished}/{total}] {name}: {state} ({seconds:.0f} s){' - ' + message if state == 'failed' else ''}")
            update_status(status_path, "running", f"{label}: {finished} of {total} cores ({name} {state})", stage="batch", progress=0.02 + 0.96 * finished / max(total, 1))

    order = {name: index for index, name in enumerate(names)}
    summary = pd.DataFrame(rows, columns=["core", "status", "seconds", "message"])
    summary = summary.sort_values("core", key=lambda column: column.map(order)).reset_index(drop=True)
    summary.attrs.update({"stage": stage, "label": label, "template": str(template_file), "root": str(root)})
    return summary


def run_workflow(config: dict, status_path: Path) -> dict:
    """Entry point for the interface: ``config`` holds ``template`` and optional ``roots``, ``cores``, ``overwrite``, ``jobs``."""
    summary = run_batch(
        config["template"], config.get("roots"), config.get("cores"), bool(config.get("overwrite", False)),
        int(config.get("jobs", 1)), status_path,
    )
    stage = summary.attrs["stage"]
    output_dir = Path(summary.attrs["root"]) / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"batch_{stage}_summary.csv"
    summary.to_csv(summary_path, index=False)
    counts = summary["status"].value_counts().to_dict()
    outputs = {"stage": stage, "label": summary.attrs["label"], "summary": str(summary_path), "template": summary.attrs["template"], **{f"n_{k.replace(' ', '_')}": int(v) for k, v in counts.items()}}
    parts = [f"{counts.get(s, 0)} {s}" for s in ("done", "skipped", "missing input", "failed") if counts.get(s)]
    update_status(status_path, "complete", f"{summary.attrs['label']} for every core: " + ", ".join(parts), stage="complete", progress=1.0, outputs=outputs)
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
