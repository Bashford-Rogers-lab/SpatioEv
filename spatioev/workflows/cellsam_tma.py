#!/usr/bin/env python3
"""Convert multiple ARK/CellSAM TMA batches into one multi-FOV AnnData file."""

from __future__ import annotations

import argparse
import json
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

# read_marker_manifest lives in cellsam so both conversion paths apply one
# rule: the marker order CSV's row order defines the channel order.
from spatioev.workflows.cellsam import (
    now,
    read_marker_manifest,
    resolve_marker_columns,
    write_h5ad_atomically,
    write_json,
    write_qc,
)
from spatioev.workflows.image_collection import (
    channel_names,
    collection_manifest,
    image_files,
    natural_key,
)


@dataclass(frozen=True)
class TMAConversionPlan:
    project_root: Path
    image_dir: Path
    marker_manifest: Path
    dataset_id: str
    output_path: Path
    primary_filename: str = "cell_table_arcsinh_transformed.csv"
    secondary_filename: str = "cell_table_size_normalized.csv"
    layer_name: str = "size_normalized"
    mask_type: str = "whole_cell"
    make_qc: bool = True


StatusCallback = Callable[[str, str, float], None]


def _table_path(table_dir: Path, filename: str) -> Path:
    relative = Path(str(filename).strip())
    if not relative.name or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Table source must be a relative filename: {filename!r}")
    return table_dir / relative


def discover_table_pairs(
    project_root: Path,
    primary_filename: str = "cell_table_arcsinh_transformed.csv",
    secondary_filename: str = "cell_table_size_normalized.csv",
) -> list[dict[str, Path | str]]:
    root = Path(project_root).expanduser().resolve()
    pairs = []
    for ark_dir in sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir() and path.name.startswith("ark_wdir")
        ),
        key=lambda path: natural_key(path.name),
    ):
        table_dir = ark_dir / "segmentation" / "cell_table"
        primary = _table_path(table_dir, primary_filename)
        secondary = _table_path(table_dir, secondary_filename)
        if primary.exists() and secondary.exists():
            pairs.append(
                {
                    "batch": ark_dir.name,
                    "table_dir": table_dir,
                    "primary": primary,
                    "secondary": secondary,
                }
            )
    if not pairs:
        raise FileNotFoundError(
            "No complete ark_wdir*/segmentation/cell_table pairs found under "
            f"{root} using {primary_filename!r} and {secondary_filename!r}"
        )
    return pairs


def _header(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def inspect_tma(plan: TMAConversionPlan) -> dict:
    if not plan.dataset_id.strip():
        raise ValueError("Dataset ID cannot be empty")
    pairs = discover_table_pairs(
        plan.project_root, plan.primary_filename, plan.secondary_filename
    )
    markers = read_marker_manifest(plan.marker_manifest)
    marker_order = markers["marker_name"].tolist()
    images = image_files(plan.image_dir)
    if not images:
        raise FileNotFoundError(f"No OME-TIFF files found under {plan.image_dir}")
    # The marker order CSV is authoritative: TMA exports routinely lose their
    # OME channel names, so the image cannot arbitrate its own marker order. A
    # named image that disagrees is reported as a warning rather than a hard
    # failure -- but a different *number* of planes is a different panel, which
    # is never safe to reconcile silently.
    warnings: list[str] = []
    first_channels = channel_names(images[0], fallback=marker_order)
    if len(first_channels) != len(marker_order):
        raise ValueError(
            f"Marker order CSV lists {len(marker_order)} markers but "
            f"{images[0].name} has {len(first_channels)} image channels"
        )

    batch_rows = []
    all_fovs: list[str] = []
    reference_header = None
    for pair in pairs:
        primary = Path(pair["primary"])
        secondary = Path(pair["secondary"])
        primary_header = _header(primary)
        secondary_header = _header(secondary)
        if primary_header != secondary_header:
            raise ValueError(f"Primary and secondary columns differ in {pair['batch']}")
        if reference_header is None:
            reference_header = primary_header
        elif primary_header != reference_header:
            raise ValueError(
                f"Cell-table columns differ across ARK batches at {pair['batch']}"
            )
        resolved, missing_markers, ambiguous = resolve_marker_columns(
            marker_order, primary_header
        )
        if ambiguous:
            raise ValueError(
                f"Marker names are ambiguous in {pair['batch']}: "
                + "; ".join(
                    f"{marker!r} matches {columns}" for marker, columns in ambiguous
                )
            )
        required = [
            column
            for column in ["label", "fov", "mask_type"]
            if column not in primary_header
        ]
        if missing_markers or required:
            raise ValueError(
                f"Invalid columns in {pair['batch']}: missing markers={missing_markers}; missing metadata={required}"
            )
        renamed = {
            marker: column for marker, column in resolved.items() if marker != column
        }
        if renamed:
            warnings.append(
                f"These marker names differ from the cell-table column spelling "
                f"in {pair['batch']} and were matched case-insensitively: "
                + ", ".join(
                    f"{marker!r} -> {column!r}" for marker, column in renamed.items()
                )
            )
        summary = pd.read_csv(primary, usecols=["fov", "mask_type"])
        selected = summary["mask_type"].astype(str).eq(plan.mask_type)
        fovs = sorted(
            summary.loc[selected, "fov"].astype(str).unique(), key=natural_key
        )
        all_fovs.extend(fovs)
        batch_rows.append(
            {
                "batch": pair["batch"],
                "primary_csv": str(primary),
                "secondary_csv": str(secondary),
                "input_rows": int(len(summary)),
                "selected_cells": int(selected.sum()),
                "fovs": ", ".join(fovs),
            }
        )

    if len(all_fovs) != len(set(all_fovs)):
        duplicates = sorted({fov for fov in all_fovs if all_fovs.count(fov) > 1})
        raise ValueError(f"FOVs occur in more than one ARK batch: {duplicates}")
    image_manifest = collection_manifest(
        plan.image_dir, sorted(all_fovs, key=natural_key)
    )
    for row in image_manifest:
        channels = channel_names(Path(row["image_path"]), fallback=marker_order)
        if len(channels) != len(marker_order):
            raise ValueError(
                f"Image panel size differs for {row['imageid']}: "
                f"{len(channels)} channels against {len(marker_order)} markers "
                f"in the marker order CSV ({row['image_path']})"
            )
        if channels != marker_order:
            warnings.append(
                f"The marker order CSV overrides the channel names in "
                f"{row['imageid']}. Image order: {channels}. "
                f"Marker order CSV: {marker_order}."
            )

    return {
        "dataset_id": plan.dataset_id.strip(),
        "n_batches": len(pairs),
        "n_fovs": len(all_fovs),
        "n_markers": len(marker_order),
        "n_cells": int(sum(row["selected_cells"] for row in batch_rows)),
        "marker_order": marker_order,
        # Marker name -> cell-table column. Every batch is checked against the
        # same reference header above, so the last resolution covers them all.
        "marker_columns": resolved,
        "warnings": warnings,
        "marker_manifest": str(Path(plan.marker_manifest).expanduser().resolve()),
        "image_dir": str(Path(plan.image_dir).expanduser().resolve()),
        "output_path": str(Path(plan.output_path).expanduser().resolve()),
        "mask_type": plan.mask_type,
        "primary_filename": plan.primary_filename,
        "secondary_filename": plan.secondary_filename,
        "batches": batch_rows,
        "image_manifest": image_manifest,
    }


def _merge_mask_metadata(
    frame: pd.DataFrame, batch: str, mask_type: str
) -> pd.DataFrame:
    frame = frame.copy()
    frame["source_ark_wdir"] = batch
    if "mask_type" not in frame:
        return frame
    selected = frame.loc[frame["mask_type"].astype(str).eq(mask_type)].copy()
    if selected.empty:
        raise ValueError(f"No rows with mask_type={mask_type!r} in {batch}")
    keys = ["source_ark_wdir", "fov", "label"]
    if selected.duplicated(keys).any():
        raise ValueError(f"Selected cell keys are not unique in {batch}")
    for other_type in (
        frame.loc[~frame["mask_type"].astype(str).eq(mask_type), "mask_type"]
        .dropna()
        .unique()
    ):
        suffix = str(other_type).strip().lower().replace(" ", "_")
        other = frame.loc[frame["mask_type"].astype(str).eq(str(other_type))].copy()
        if other.empty:
            continue
        value_columns = [column for column in other.columns if column not in keys]
        other = other[keys + value_columns].rename(
            columns={column: f"{column}_{suffix}" for column in value_columns}
        )
        selected = selected.merge(
            other, on=keys, how="left", validate="one_to_one", sort=False
        )
    if "area_nuclear" in selected and "area" in selected:
        denominator = pd.to_numeric(selected["area"], errors="coerce").replace(
            0, np.nan
        )
        selected["nc_ratio"] = (
            pd.to_numeric(selected["area_nuclear"], errors="coerce") / denominator
        )
    return selected


def _cell_index(frame: pd.DataFrame, dataset_id: str) -> pd.Index:
    names = (
        dataset_id + "_" + frame["fov"].astype(str) + "_" + frame["label"].astype(str)
    )
    names = pd.Index(names, name="cell_id")
    if names.has_duplicates:
        names = pd.Index(
            dataset_id
            + "_"
            + frame["source_ark_wdir"].astype(str)
            + "_"
            + frame["fov"].astype(str)
            + "_"
            + frame["label"].astype(str),
            name="cell_id",
        )
    if names.has_duplicates:
        raise ValueError(
            "Could not construct unique TMA cell IDs from batch, FOV, and label"
        )
    return names


def build_tma_anndata(
    plan: TMAConversionPlan,
    status: StatusCallback | None = None,
) -> tuple[ad.AnnData, dict]:
    def update(stage: str, message: str, progress: float) -> None:
        if status is not None:
            status(stage, message, progress)

    update("inspect", "Validating ARK batches, marker order, and FOV images", 0.04)
    report = inspect_tma(plan)
    pairs = discover_table_pairs(
        plan.project_root, plan.primary_filename, plan.secondary_filename
    )
    marker_order = report["marker_order"]
    primary_parts = []
    secondary_parts = []
    for index, pair in enumerate(pairs, start=1):
        update(
            "read",
            f"Reading ARK batch {index}/{len(pairs)}: {pair['batch']}",
            0.08 + 0.30 * index / len(pairs),
        )
        primary_raw = pd.read_csv(pair["primary"])
        secondary_raw = pd.read_csv(pair["secondary"])
        primary = _merge_mask_metadata(primary_raw, str(pair["batch"]), plan.mask_type)
        secondary = secondary_raw.loc[
            secondary_raw["mask_type"].astype(str).eq(plan.mask_type)
        ].copy()
        secondary["source_ark_wdir"] = pair["batch"]
        primary_parts.append(primary)
        secondary_parts.append(secondary)

    primary = pd.concat(primary_parts, ignore_index=True)
    secondary = pd.concat(secondary_parts, ignore_index=True)
    keys = ["source_ark_wdir", "fov", "label"]
    primary_key = pd.MultiIndex.from_frame(primary[keys].astype(str))
    secondary_key = pd.MultiIndex.from_frame(secondary[keys].astype(str))
    if primary_key.has_duplicates or secondary_key.has_duplicates:
        raise ValueError("Batch/FOV/label does not uniquely identify selected cells")
    secondary.index = secondary_key
    missing = primary_key.difference(secondary_key)
    extra = secondary_key.difference(primary_key)
    if len(missing) or len(extra):
        raise ValueError(
            f"Primary/secondary cell mismatch: {len(missing)} missing and {len(extra)} extra"
        )
    secondary = secondary.loc[primary_key].reset_index(drop=True)

    update("assemble", "Assembling multi-FOV AnnData", 0.50)
    # Exclude the columns the markers actually came from. Using the manifest
    # spellings would leave a case-differing column in .obs as well as .X.
    marker_columns = [report["marker_columns"][marker] for marker in marker_order]
    marker_set = set(marker_columns)
    obs_columns = [column for column in primary.columns if column not in marker_set]
    obs = primary[obs_columns].copy()
    if "cell_size" in obs and "area" in obs:
        size = pd.to_numeric(obs["cell_size"], errors="coerce").to_numpy(dtype=float)
        area = pd.to_numeric(obs["area"], errors="coerce").to_numpy(dtype=float)
        if np.allclose(size, area, equal_nan=True):
            obs = obs.drop(columns="cell_size")
    coordinate_pairs = [
        ("X_centroid", "Y_centroid"),
        ("centroid_x", "centroid_y"),
        ("centroid-1", "centroid-0"),
        ("x", "y"),
    ]
    coordinates = next(
        ((x, y) for x, y in coordinate_pairs if x in obs and y in obs), None
    )
    if coordinates is None:
        raise ValueError("No supported X/Y centroid columns were found")
    x_column, y_column = coordinates
    obs["X_centroid"] = pd.to_numeric(obs[x_column], errors="raise").to_numpy(
        dtype=float
    )
    obs["Y_centroid"] = pd.to_numeric(obs[y_column], errors="raise").to_numpy(
        dtype=float
    )
    obs["fov"] = obs["fov"].astype(str)
    obs["imageid"] = obs["fov"]
    obs["dataset_id"] = plan.dataset_id.strip()
    obs["sample_id"] = plan.dataset_id.strip()
    obs["slide_id"] = plan.dataset_id.strip()
    obs["tissue_piece"] = obs["fov"]
    obs.index = _cell_index(obs, plan.dataset_id.strip())

    marker_manifest = read_marker_manifest(plan.marker_manifest)
    var = marker_manifest.set_index("marker_name").loc[marker_order].copy()
    var.index.name = "marker"
    var["marker_order"] = np.arange(len(var), dtype=int)
    var["source_column"] = marker_columns
    x = primary[marker_columns].to_numpy(dtype=np.float32, copy=True)
    layer = secondary[marker_columns].to_numpy(dtype=np.float32, copy=True)
    adata = ad.AnnData(X=x, obs=obs, var=var)
    adata.layers[plan.layer_name] = layer
    adata.obsm["spatial"] = obs[["X_centroid", "Y_centroid"]].to_numpy(dtype=np.float32)
    adata.uns["all_markers"] = np.asarray(marker_order, dtype=str)
    adata.uns["image_manifest"] = pd.DataFrame(report["image_manifest"]).set_index(
        "imageid"
    )
    adata.uns["tma_conversion"] = {
        "created_at": now(),
        "dataset_id": plan.dataset_id.strip(),
        "project_root": str(Path(plan.project_root).expanduser().resolve()),
        "marker_manifest": str(Path(plan.marker_manifest).expanduser().resolve()),
        "image_dir": str(Path(plan.image_dir).expanduser().resolve()),
        "source_ark_wdirs": [str(pair["batch"]) for pair in pairs],
        "mask_type": plan.mask_type,
        "layer_name": plan.layer_name,
        "primary_filename": plan.primary_filename,
        "secondary_filename": plan.secondary_filename,
    }
    adata.uns["cellsam_conversion"] = {
        "created_at": now(),
        "imageid": plan.dataset_id.strip(),
        "channel_order_source": "marker manifest",
    }

    output_path = Path(plan.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    update("write", "Writing compressed multi-FOV H5AD", 0.72)
    write_h5ad_atomically(adata, output_path)
    qc_outputs = {}
    if plan.make_qc:
        update("qc", "Generating combined marker and spatial QC", 0.88)
        qc_outputs = write_qc(adata, output_path)

    counts = (
        adata.obs["imageid"]
        .value_counts()
        .reindex(sorted(adata.obs["imageid"].astype(str).unique(), key=natural_key))
    )
    counts_path = output_path.with_name(f"{output_path.stem}_cells_per_fov.csv")
    counts.rename_axis("imageid").reset_index(name="n_cells").to_csv(
        counts_path, index=False
    )
    manifest = {
        **report,
        "created_at": now(),
        "n_cells": int(adata.n_obs),
        "n_markers": int(adata.n_vars),
        "output_path": str(output_path),
        "layer_name": plan.layer_name,
        "obs_columns": list(adata.obs.columns),
        "qc_outputs": qc_outputs,
        "cells_per_fov_csv": str(counts_path),
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    update("complete", "TMA AnnData conversion complete", 1.0)
    return adata, manifest


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--project-root", type=Path, required=True)
    cli.add_argument("--image-dir", type=Path, required=True)
    cli.add_argument("--marker-manifest", type=Path, required=True)
    cli.add_argument("--dataset-id", required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--primary-filename", default="cell_table_arcsinh_transformed.csv")
    cli.add_argument("--secondary-filename", default="cell_table_size_normalized.csv")
    cli.add_argument("--layer-name", default="size_normalized")
    cli.add_argument("--mask-type", default="whole_cell")
    cli.add_argument("--no-qc", action="store_true")
    cli.add_argument("--inspect-only", action="store_true")
    cli.add_argument("--status", type=Path)
    return cli


def main() -> None:
    args = parser().parse_args()
    plan = TMAConversionPlan(
        project_root=args.project_root,
        image_dir=args.image_dir,
        marker_manifest=args.marker_manifest,
        dataset_id=args.dataset_id,
        output_path=args.output,
        primary_filename=args.primary_filename,
        secondary_filename=args.secondary_filename,
        layer_name=args.layer_name,
        mask_type=args.mask_type,
        make_qc=not args.no_qc,
    )

    def status(stage: str, message: str, progress: float) -> None:
        if args.status is not None:
            write_json(
                args.status,
                {
                    "state": "complete" if stage == "complete" else "running",
                    "stage": stage,
                    "message": message,
                    "progress": progress,
                    "updated_at": now(),
                },
            )
        print(f"[{stage}] {message}", flush=True)

    try:
        if args.inspect_only:
            print(json.dumps(inspect_tma(plan), indent=2))
            return
        _, manifest = build_tma_anndata(plan, status=status)
        if args.status is not None:
            payload = json.loads(args.status.read_text(encoding="utf-8"))
            payload["outputs"] = manifest
            write_json(args.status, payload)
        print(json.dumps(manifest, indent=2))
    except Exception as error:
        if args.status is not None:
            write_json(
                args.status,
                {
                    "state": "failed",
                    "stage": "failed",
                    "message": str(error),
                    "updated_at": now(),
                    "traceback": traceback.format_exc(),
                },
            )
        raise


if __name__ == "__main__":
    main()
