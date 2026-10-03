#!/usr/bin/env python3
"""Background worker: tumour / envelope / stroma regions and per-region summaries.

Labels every cell by its location relative to tumour nests, writes the
phenotype composition of each region and, when a fiber segmentation run is
supplied, the matrix quantity and architecture inside each region.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import traceback
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spatioev_regions_matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

from spatioev.pl.fibers import plot_tissue_regions
from spatioev.tl.ecm.fiber_stats import fiber_region_stats
from spatioev.tl.niche.regions import (
    REGION_CODES,
    define_tissue_regions,
    rasterize_tissue_regions,
    summarize_region_composition,
    write_region_boundaries,
)
from spatioev.workflows.cellsam import write_h5ad_atomically

from ._io import now, read_json, write_json
from .fiber_segmentation import (
    DEFAULT_PARAMETERS,
    OMEImage,
    label_image_path,
    measure_architecture,
    open_labels,
    physical_pixel_size,
    raster_group_reader,
    read_fiber_table,
    safe_name,
)

REGION_ORDER = ["tumour", "envelope", "stroma"]


def update_status(path: Path, state: str, message: str, *, stage: str, progress: float, **extra) -> None:
    write_json(path, {"state": state, "message": message, "stage": stage, "progress": progress, "updated_at": now(), **extra})


def fiber_manifest(manifest_path: str | None) -> dict | None:
    if not manifest_path:
        return None
    manifest = read_json(Path(manifest_path).expanduser())
    if manifest is None:
        raise FileNotFoundError(f"Fiber manifest not found or unreadable: {manifest_path}")
    return manifest


def fiber_parameters(manifest: dict | None) -> dict:
    return {**DEFAULT_PARAMETERS, **((manifest or {}).get("parameters") or {})}


def region_architecture(label_row: dict, manifest: dict, labels, raster: np.ndarray, downsample: int, pixel_size: float) -> pd.DataFrame:
    """TWOMBLI-style metrics per tissue region for one image and matrix channel."""
    from spatioev.tl.ecm.architecture import hdm_threshold

    params = fiber_parameters(manifest)
    image_path = label_image_path(label_row, manifest)
    if image_path is None or not image_path.exists():
        raise FileNotFoundError(f"source image not found ({image_path})")
    with OMEImage(image_path) as image:
        plane = image.plane(label_row["fiber_type"])
        threshold = label_row.get("hdm_threshold")
        if threshold is None:
            threshold = hdm_threshold(plane, float(params["max_display_hdm"]), float(params["hdm_saturation_pct"]))
        table = measure_architecture(
            labels, plane, tuple(labels.shape), threshold, params, pixel_size,
            raster_group_reader(raster, downsample), REGION_CODES, "all_tissue",
        )
    return table.rename(columns={"group": "region"})


def inspect_inputs(config: dict) -> dict:
    """Phenotype columns, their values and image IDs (used by the UI)."""
    adata = ad.read_h5ad(Path(config["adata_path"]).expanduser(), backed="r")
    try:
        obs = adata.obs
        candidates = {}
        for column in obs.columns:
            series = obs[column]
            if str(column).startswith("gate_") or column in {"imageid", "dataset_id", "mask_type", "sample"}:
                continue
            if isinstance(series.dtype, pd.CategoricalDtype) or series.dtype == object or pd.api.types.is_string_dtype(series):
                n_unique = series.nunique()
                if 1 < n_unique <= 80:
                    counts = series.astype(str).value_counts()
                    candidates[str(column)] = [{"value": key, "n_cells": int(value)} for key, value in counts.items()]
        return {
            "n_cells": int(adata.n_obs),
            "imageids": sorted(pd.unique(obs["imageid"].astype(str))) if "imageid" in obs else [],
            "candidate_columns": candidates,
            "has_coordinates": {"X_centroid", "Y_centroid"}.issubset(obs.columns),
        }
    finally:
        adata.file.close()


def image_shapes(config: dict, obs: pd.DataFrame, fibers: dict | None) -> dict[str, tuple[int, int]]:
    shapes: dict[str, tuple[int, int]] = {}
    if fibers:
        shapes.update({str(k): tuple(v) for k, v in fibers.get("image_shapes", {}).items()})
    for imageid, cells in obs.groupby("imageid", observed=True):
        if str(imageid) not in shapes:
            # Without the image, bound the grid by the cells plus a margin.
            shapes[str(imageid)] = (int(cells["Y_centroid"].max()) + 64, int(cells["X_centroid"].max()) + 64)
    return shapes


def save_composition_plot(composition: pd.DataFrame, phenotype_key: str, path: Path) -> Path:
    table = composition[composition["tissue_region"].isin(["all", *REGION_ORDER])]
    pivot = table.pivot_table(index="tissue_region", columns=phenotype_key, values="count", aggfunc="sum", fill_value=0)
    pivot = pivot.reindex([r for r in ["all", *REGION_ORDER] if r in pivot.index])
    shares = pivot.div(pivot.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    shares.plot(kind="barh", stacked=True, ax=ax, colormap="tab20", width=0.75)
    ax.set_xlabel("Share of cells")
    ax.set_ylabel("")
    ax.set_xlim(0, 1)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, frameon=False)
    ax.set_title("Cell composition by tissue region (all images pooled)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def run_workflow(config: dict, status_path: Path) -> dict:
    sample_id = config["sample_id"].strip()
    output_dir = Path(config["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    phenotype_key = config["phenotype_key"]
    tumour_labels = [str(v) for v in config["tumour_labels"]]
    if not tumour_labels:
        raise ValueError("Select at least one tumour phenotype")

    update_status(status_path, "running", "Loading cells", stage="load", progress=0.05)
    adata = ad.read_h5ad(Path(config["adata_path"]).expanduser())
    for column in ("imageid", "X_centroid", "Y_centroid", phenotype_key):
        if column not in adata.obs:
            raise KeyError(f"AnnData is missing obs column {column!r}")
    adata.obs["imageid"] = adata.obs["imageid"].astype(str)
    manifest = fiber_manifest(config.get("fiber_manifest"))
    fibers = manifest["outputs"] if manifest else None
    warnings: list[str] = []
    pixel_size = config.get("pixel_size_um") or (fibers or {}).get("pixel_size_um")
    if not pixel_size and config.get("image_path"):
        pixel_size = physical_pixel_size(Path(config["image_path"]))
    if not pixel_size:
        raise ValueError("Pixel size (µm per pixel) is required to convert widths to pixels")
    pixel_size = float(pixel_size)

    update_status(status_path, "running", "Finding tumour nests and envelopes", stage="regions", progress=0.15)
    regions, boundaries = define_tissue_regions(
        adata,
        phenotype_key=phenotype_key,
        tumour_labels=tumour_labels,
        envelope_width=float(config["envelope_width_um"]) / pixel_size,
        min_nest_cells=int(config.get("min_nest_cells", 30)),
        knn_k=int(config.get("knn_k", 5)),
        radius_scale=float(config.get("radius_scale", 1.0)),
        boundary_method=config.get("boundary_method", "density_mask"),
        mask_resolution=float(config["mask_resolution_um"]) / pixel_size if config.get("mask_resolution_um") else None,
        mask_sigma=float(config.get("mask_sigma", 2.0)),
        mask_threshold=float(config.get("mask_threshold", 0.1)),
        mask_closing_size=int(config.get("mask_closing_size", 5)),
    )
    regions.uns["tissue_regions"] = {
        "phenotype_key": phenotype_key,
        "tumour_labels": tumour_labels,
        "envelope_width_um": float(config["envelope_width_um"]),
        "pixel_size_um": pixel_size,
        "n_nests": int(len(boundaries)),
    }

    update_status(status_path, "running", "Summarising composition", stage="composition", progress=0.35)
    composition = summarize_region_composition(regions, phenotype_key)
    composition_path = output_dir / f"{sample_id}_region_composition.csv"
    composition.to_csv(composition_path, index=False)
    boundaries_path = write_region_boundaries(boundaries, output_dir / f"{sample_id}_region_boundaries.geojson")

    update_status(status_path, "running", "Rasterising regions", stage="raster", progress=0.45)
    shapes = image_shapes(config, regions.obs, fibers)
    downsample = int(config.get("raster_downsample", 4))
    raster_dir = output_dir / "region_rasters"
    raster_dir.mkdir(exist_ok=True)
    rasters, area_rows, plot_paths = {}, [], []
    imageids = sorted(regions.obs["imageid"].unique())
    for index, imageid in enumerate(imageids):
        raster = rasterize_tissue_regions(
            boundaries,
            regions.obs,
            shapes[imageid],
            imageid=imageid,
            tissue_radius=float(config.get("tissue_radius_um", 50.0)) / pixel_size,
            downsample=downsample,
        )
        path = raster_dir / f"{safe_name(imageid)}_regions_ds{downsample}.tiff"
        tifffile.imwrite(path, raster, compression="zlib")
        rasters[imageid] = str(path)
        counts = np.bincount(raster.ravel(), minlength=4)
        for code, name in REGION_CODES.items():
            area_rows.append({"imageid": imageid, "region": name, "area_mm2": counts[code] * downsample**2 * (pixel_size / 1000) ** 2})
        area_rows.append({"imageid": imageid, "region": "all_tissue", "area_mm2": counts[1:].sum() * downsample**2 * (pixel_size / 1000) ** 2})
        if index < int(config.get("max_region_plots", 12)):
            fig, ax = plt.subplots(figsize=(8, 8))
            cells = regions.obs[regions.obs["imageid"] == imageid]
            nests = boundaries[boundaries["imageid"].astype(str) == imageid] if len(boundaries) else boundaries
            plot_tissue_regions(cells, nests, ax=ax, title=f"{imageid}: {len(nests)} nests, envelope {config['envelope_width_um']} µm")
            plot_path = output_dir / f"{safe_name(imageid)}_tissue_regions.png"
            fig.tight_layout()
            fig.savefig(plot_path, dpi=120)
            plt.close(fig)
            plot_paths.append(str(plot_path))
    areas = pd.DataFrame(area_rows)
    areas_path = output_dir / f"{sample_id}_region_areas.csv"
    areas.to_csv(areas_path, index=False)

    fiber_stats_path = None
    if fibers:
        update_status(status_path, "running", "Measuring matrix inside each region", stage="matrix", progress=0.6)
        table = read_fiber_table(fibers["fiber_table"])
        rows = []
        for label_row in fibers["labels"]:
            imageid, fiber_type = str(label_row["imageid"]), label_row["fiber_type"]
            if imageid not in rasters:
                continue
            subset = table[(table["imageid"] == imageid) & (table["fiber_type"] == fiber_type)]
            labels = open_labels(label_row["path"]) if Path(label_row["path"]).exists() else None
            stats = fiber_region_stats(
                subset,
                tifffile.imread(rasters[imageid]),
                REGION_CODES,
                labels=labels,
                downsample=downsample,
                pixel_size_um=pixel_size,
            )
            if labels is not None and fiber_parameters(manifest).get("architecture", True):
                try:
                    stats = stats.merge(
                        region_architecture(label_row, manifest, labels, tifffile.imread(rasters[imageid]), downsample, pixel_size),
                        on="region", how="left",
                    )
                except Exception as error:  # e.g. the source image moved; keep the run's other results
                    warnings.append(f"Architecture skipped for {imageid} / {fiber_type}: {error}")
            stats.insert(0, "fiber_type", fiber_type)
            stats.insert(0, "imageid", imageid)
            rows.append(stats)
        if rows:
            fiber_stats = pd.concat(rows, ignore_index=True)
            fiber_stats_path = output_dir / f"{sample_id}_fiber_region_stats.csv"
            fiber_stats.to_csv(fiber_stats_path, index=False)

    update_status(status_path, "running", "Writing AnnData and plots", stage="write", progress=0.85)
    composition_plot = save_composition_plot(composition, phenotype_key, output_dir / f"{sample_id}_region_composition.png")
    h5ad_path = output_dir / f"{sample_id}_tissue_regions.h5ad"
    regions.obs["tissue_region"] = regions.obs["tissue_region"].astype(str)
    regions.obs["tumour_nest"] = regions.obs["tumour_nest"].astype("string").fillna("none").astype(str)
    write_h5ad_atomically(regions, h5ad_path)

    counts = regions.obs["tissue_region"].value_counts()
    outputs = {
        "sample_id": sample_id,
        "h5ad": str(h5ad_path),
        "phenotype_key": phenotype_key,
        "region_key": "tissue_region",
        "composition": str(composition_path),
        "composition_png": str(composition_plot),
        "boundaries": str(boundaries_path),
        "areas": str(areas_path),
        "fiber_region_stats": str(fiber_stats_path) if fiber_stats_path else None,
        "region_rasters": rasters,
        "raster_downsample": downsample,
        "region_plots": plot_paths,
        "pixel_size_um": pixel_size,
        "n_nests": int(len(boundaries)),
        "cells_per_region": {region: int(counts.get(region, 0)) for region in REGION_ORDER},
        "fiber_manifest": config.get("fiber_manifest"),
        "warnings": warnings,
    }
    manifest_path = output_dir / f"{sample_id}_tissue_regions_manifest.json"
    write_json(manifest_path, {"created_at": now(), "config": config, "outputs": outputs})
    outputs["manifest"] = str(manifest_path)
    update_status(status_path, "complete", f"{len(boundaries)} tumour nests; regions assigned to {regions.n_obs:,} cells", stage="complete", progress=1.0, outputs=outputs)
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
