#!/usr/bin/env python3
"""Background worker: cell-cell and cell-matrix co-localisation.

Cell-cell
    For each source -> target phenotype pair and each image: the cross-Ripley
    ``L(r) - r`` curve with a label-permutation envelope (positions fixed,
    phenotypes shuffled within the image), and, per tissue region, how many
    target cells surround each source cell relative to the image-wide
    expectation.

Cell-matrix (needs a fiber segmentation run)
    Per cell: distance to the nearest matrix pixel and the matrix area
    fraction within a radius; summarised per phenotype and region, tested
    against label permutation, and related to dense / aligned matrix tiles.

Images are analysed one at a time: TMA cores share a coordinate frame
origin, so pooling their coordinates would place cells from different cores
next to each other.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import traceback
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spatioev_coloc_matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spatioev.tl.ecm.fiber_stats import (
    cell_matrix_enrichment,
    cell_matrix_proximity,
    fiber_tile_stats,
    tile_cell_matrix_association,
)
from spatioev.tl.stats.ripley import (
    cross_ripley_local_counts,
    cross_ripley_permutation_envelope,
)

from ._io import now, read_json, write_json
from .fiber_segmentation import (
    DEFAULT_PARAMETERS,
    OMEImage,
    label_image_path,
    measure_architecture,
    open_labels,
    physical_pixel_size,
    read_fiber_table,
    tile_grid_groups,
)

REGION_ORDER = ["tumour", "envelope", "stroma"]


def update_status(path: Path, state: str, message: str, *, stage: str, progress: float, **extra) -> None:
    write_json(path, {"state": state, "message": message, "stage": stage, "progress": progress, "updated_at": now(), **extra})


def cell_cell_analysis(obs: pd.DataFrame, config: dict, pixel_size: float, region_key: str | None, report) -> tuple[pd.DataFrame, pd.DataFrame]:
    phenotype_key = config["phenotype_key"]
    radii_px = np.asarray(config["radii_um"], dtype=float) / pixel_size
    local_radius_px = float(config.get("local_radius_um", 30.0)) / pixel_size
    n_perm = int(config.get("n_permutations", 199))
    pairs = [(str(p["source"]), str(p["target"])) for p in config.get("pairs", [])]
    curves, local_rows = [], []
    imageids = sorted(obs["imageid"].unique())
    total = max(1, len(imageids) * len(pairs))
    done = 0
    for imageid in imageids:
        cells = obs[obs["imageid"] == imageid]
        image_adata = ad.AnnData(obs=cells.copy())
        present = set(cells[phenotype_key])
        for source, target in pairs:
            done += 1
            report(done / total, f"{imageid}: {source} -> {target}")
            if source not in present or target not in present:
                continue
            curve = cross_ripley_permutation_envelope(
                image_adata, phenotype_key, source, target, radii_px, n_sim=n_perm, random_state=0
            )
            if curve.empty:
                continue
            curve = curve.assign(imageid=imageid, source=source, target=target)
            curve["radius_um"] = curve["radius"] * pixel_size
            for column in ("L", "L_minus_r", "envelope_low", "envelope_high"):
                curve[f"{column}_um"] = curve[column] * pixel_size
            curve["above_envelope"] = curve["L_minus_r"] > curve["envelope_high"]
            curve["below_envelope"] = curve["L_minus_r"] < curve["envelope_low"]
            curve["excess_over_null_um"] = (curve["L_minus_r"] - (curve["envelope_low"] + curve["envelope_high"]) / 2) * pixel_size
            curves.append(curve)

            local = cross_ripley_local_counts(image_adata, phenotype_key, source, target, local_radius_px)
            if local.empty:
                continue
            local = local.set_index("cell_id")
            groups = [("all", local)]
            if region_key:
                local["region"] = cells.loc[local.index, region_key].astype(str)
                groups += [(region, frame) for region, frame in local.groupby("region")]
            for region, frame in groups:
                local_rows.append({
                    "imageid": imageid,
                    "source": source,
                    "target": target,
                    "region": region,
                    "radius_um": local_radius_px * pixel_size,
                    "n_source_cells": len(frame),
                    "mean_target_neighbours": float(frame["target_neighbor_count"].mean()),
                    "expected_target_neighbours": float(frame["expected_target_neighbor_count"].mean()),
                    "mean_neighbour_ratio": float(frame["target_neighbor_ratio"].mean()),
                    "fraction_source_with_excess": float((frame["target_neighbor_excess"] > 0).mean()),
                })
    curve_table = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    return curve_table, pd.DataFrame(local_rows)


ARCHITECTURE_TILE_METRICS = ("hdm_pct", "alignment_coherency", "length_density_mm_per_mm2", "branchpoints_per_mm2", "curvature_per_um")


def tile_architecture(manifest: dict, label_rows: list[dict], tile_px: int, pixel_size: float) -> pd.DataFrame:
    """TWOMBLI-style metrics on the association tile grid, per image and channel."""
    from spatioev.tl.ecm.architecture import hdm_threshold

    params = {**DEFAULT_PARAMETERS, **(manifest.get("parameters") or {})}
    frames = []
    for row in label_rows:
        image_path = label_image_path(row, manifest)
        if image_path is None or not image_path.exists():
            continue
        labels = open_labels(row["path"])
        reader, names = tile_grid_groups(tile_px, tuple(labels.shape))
        with OMEImage(image_path) as image:
            plane = image.plane(row["fiber_type"])
            threshold = row.get("hdm_threshold")
            if threshold is None:
                threshold = hdm_threshold(plane, float(params["max_display_hdm"]), float(params["hdm_saturation_pct"]))
            table = measure_architecture(labels, plane, tuple(labels.shape), threshold, params, pixel_size, reader, names, None)
        table[["tile_y", "tile_x"]] = table["group"].str.split(",", expand=True).astype(int)
        frames.append(table.drop(columns="group").assign(imageid=str(row["imageid"]), fiber_type=row["fiber_type"]))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def cell_matrix_analysis(obs: pd.DataFrame, fibers: dict, config: dict, pixel_size: float, region_key: str | None, report, manifest: dict | None = None):
    phenotype_key = config["phenotype_key"]
    radius_px = float(config.get("matrix_radius_um", 20.0)) / pixel_size
    contact_um = float(config.get("contact_distance_um", 5.0))
    phenotypes = [str(p) for p in config.get("matrix_phenotypes", [])] or None
    fiber_types = config.get("fiber_types") or fibers.get("channels", [])
    n_perm = int(config.get("n_permutations", 199))
    summary_rows, enrichment_rows, per_cell = [], [], []
    label_rows = [row for row in fibers["labels"] if row["fiber_type"] in fiber_types and str(row["imageid"]) in set(obs["imageid"])]
    for index, row in enumerate(label_rows):
        imageid, fiber_type = str(row["imageid"]), row["fiber_type"]
        report((index + 1) / max(1, len(label_rows)), f"{imageid}: cells vs {fiber_type}")
        cells = obs[obs["imageid"] == imageid]
        labels = open_labels(row["path"])
        downsample = int(config.get("proximity_downsample") or max(1, int(np.ceil(max(labels.shape) / 8000))))
        mask = np.asarray(labels[::downsample, ::downsample]) > 0
        proximity = cell_matrix_proximity(cells, mask, radius_px, downsample=downsample)
        proximity["distance_to_matrix_um"] = proximity["distance_to_matrix"] * pixel_size
        proximity = proximity.join(cells[[phenotype_key] + ([region_key] if region_key else [])])
        per_cell.append(proximity.assign(imageid=imageid, fiber_type=fiber_type))

        groups = [("all", proximity)]
        if region_key:
            groups += [(str(region), frame) for region, frame in proximity.groupby(region_key, observed=True)]
        for region, frame in groups:
            for phenotype, sub in frame.groupby(phenotype_key, observed=True):
                if phenotypes and str(phenotype) not in phenotypes:
                    continue
                summary_rows.append({
                    "imageid": imageid,
                    "fiber_type": fiber_type,
                    "region": region,
                    "phenotype": str(phenotype),
                    "n_cells": len(sub),
                    "median_distance_to_matrix_um": float(sub["distance_to_matrix_um"].median()),
                    "mean_matrix_fraction": float(sub["matrix_fraction"].mean()),
                    "fraction_in_contact": float((sub["distance_to_matrix_um"] <= contact_um).mean()),
                })
        for metric in ("matrix_fraction", "distance_to_matrix_um"):
            test = cell_matrix_enrichment(proximity[metric], proximity[phenotype_key], phenotypes, n_permutations=n_perm)
            if not test.empty:
                enrichment_rows.append(test.assign(imageid=imageid, fiber_type=fiber_type, metric=metric))

    summary = pd.DataFrame(summary_rows)
    enrichment = pd.concat(enrichment_rows, ignore_index=True) if enrichment_rows else pd.DataFrame()

    association = pd.DataFrame()
    if fibers.get("fiber_table") and Path(fibers["fiber_table"]).exists():
        table = read_fiber_table(fibers["fiber_table"])
        table = table[table["fiber_type"].isin(fiber_types) & table["imageid"].isin(set(obs["imageid"]))]
        tile_px = int(round(float(config.get("tile_size_um", 150.0)) / pixel_size))
        shapes = {str(k): tuple(v) for k, v in fibers.get("image_shapes", {}).items()}
        if len(table) and shapes:
            tiles = fiber_tile_stats(table, shapes, tile_size=tile_px, min_fiber_num=int(config.get("min_fiber_num", 3)), pixel_size_um=pixel_size)
            metrics = ["area_fraction_pct", "orientation_coherence", "avg_alignment_score"]
            if manifest is not None and config.get("architecture_tiles", True):
                report(0.95, "TWOMBLI-style metrics per tile")
                architecture = tile_architecture(manifest, label_rows, tile_px, pixel_size)
                if not architecture.empty:
                    keep = ["imageid", "fiber_type", "tile_y", "tile_x", *ARCHITECTURE_TILE_METRICS]
                    tiles = tiles.merge(architecture[keep], on=["imageid", "fiber_type", "tile_y", "tile_x"], how="left")
                    metrics += list(ARCHITECTURE_TILE_METRICS)
            _, association = tile_cell_matrix_association(
                tiles, obs, phenotype_key, tile_px,
                metrics=tuple(metrics),
                phenotypes=phenotypes, high_quantile=float(config.get("high_quantile", 0.75)),
                min_cells_per_tile=int(config.get("min_cells_per_tile", 5)),
            )
            association["tile_size_um"] = tile_px * pixel_size
    per_cell_table = pd.concat(per_cell) if per_cell else pd.DataFrame()
    return summary, enrichment, association, per_cell_table


def save_curve_plot(curves: pd.DataFrame, path: Path, max_images: int = 6) -> Path | None:
    if curves.empty:
        return None
    pairs = curves[["source", "target"]].drop_duplicates().values.tolist()
    images = sorted(curves["imageid"].unique())[:max_images]
    fig, axes = plt.subplots(len(pairs), 1, figsize=(7, 3.2 * len(pairs)), squeeze=False)
    for ax, (source, target) in zip(axes[:, 0], pairs):
        for imageid in images:
            sub = curves[(curves.source == source) & (curves.target == target) & (curves.imageid == imageid)]
            if sub.empty:
                continue
            line = ax.plot(sub["radius_um"], sub["L_minus_r_um"], marker="o", ms=3, label=str(imageid))[0]
            ax.fill_between(sub["radius_um"], sub["envelope_low_um"], sub["envelope_high_um"], color=line.get_color(), alpha=0.15)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_title(f"{source} → {target}: L(r) − r with 95% label-permutation envelope", fontsize=10)
        ax.set_xlabel("radius (µm)")
        ax.set_ylabel("L(r) − r (µm)")
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_matrix_plot(summary: pd.DataFrame, path: Path) -> Path | None:
    if summary.empty:
        return None
    data = summary[summary["region"] != "all"] if (summary["region"] != "all").any() else summary
    fiber_types = sorted(data["fiber_type"].unique())
    fig, axes = plt.subplots(1, len(fiber_types), figsize=(6 * len(fiber_types), 4), squeeze=False)
    for ax, fiber_type in zip(axes[0], fiber_types):
        sub = data[data["fiber_type"] == fiber_type]
        pivot = sub.pivot_table(index="phenotype", columns="region", values="mean_matrix_fraction", aggfunc="mean")
        pivot = pivot.reindex(columns=[c for c in [*REGION_ORDER, "all"] if c in pivot.columns])
        pivot.plot(kind="bar", ax=ax, width=0.8, color=["#c2410c", "#d4a017", "#3b82a0", "#6b7280"][: len(pivot.columns)])
        ax.set_title(f"Matrix fraction around cells: {fiber_type}")
        ax.set_ylabel("mean matrix area fraction")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def run_workflow(config: dict, status_path: Path) -> dict:
    sample_id = config["sample_id"].strip()
    output_dir = Path(config["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    phenotype_key = config["phenotype_key"]

    update_status(status_path, "running", "Loading cells", stage="load", progress=0.03)
    adata = ad.read_h5ad(Path(config["adata_path"]).expanduser(), backed="r")
    try:
        obs = adata.obs.copy()
    finally:
        adata.file.close()
    for column in ("imageid", "X_centroid", "Y_centroid", phenotype_key):
        if column not in obs:
            raise KeyError(f"AnnData is missing obs column {column!r}")
    obs["imageid"] = obs["imageid"].astype(str)
    obs[phenotype_key] = obs[phenotype_key].astype(str)
    region_key = config.get("region_key") or ("tissue_region" if "tissue_region" in obs else None)
    if region_key and region_key not in obs:
        raise KeyError(f"Region column {region_key!r} not found; run the tissue regions stage or clear the field")

    fibers = None
    if config.get("fiber_manifest"):
        manifest = read_json(Path(config["fiber_manifest"]).expanduser())
        if manifest is None:
            raise FileNotFoundError(f"Fiber manifest not found: {config['fiber_manifest']}")
        fibers = manifest["outputs"]
    pixel_size = config.get("pixel_size_um") or (fibers or {}).get("pixel_size_um")
    if not pixel_size and config.get("image_path"):
        pixel_size = physical_pixel_size(Path(config["image_path"]))
    if not pixel_size:
        raise ValueError("Pixel size (µm per pixel) is required")
    pixel_size = float(pixel_size)

    outputs: dict = {"sample_id": sample_id, "pixel_size_um": pixel_size, "region_key": region_key, "phenotype_key": phenotype_key}

    if config.get("pairs"):
        curves, local = cell_cell_analysis(
            obs, config, pixel_size, region_key,
            lambda f, m: update_status(status_path, "running", m, stage="cell-cell", progress=round(0.05 + 0.5 * f, 4)),
        )
        curves_path = output_dir / f"{sample_id}_cell_cell_curves.csv"
        local_path = output_dir / f"{sample_id}_cell_cell_local.csv"
        curves.to_csv(curves_path, index=False)
        local.to_csv(local_path, index=False)
        outputs.update(cell_cell_curves=str(curves_path), cell_cell_local=str(local_path))
        plot = save_curve_plot(curves, output_dir / f"{sample_id}_cell_cell_curves.png")
        outputs["cell_cell_plot"] = str(plot) if plot else None

    if fibers:
        summary, enrichment, association, per_cell = cell_matrix_analysis(
            obs, fibers, config, pixel_size, region_key,
            lambda f, m: update_status(status_path, "running", m, stage="cell-matrix", progress=round(0.55 + 0.4 * f, 4)),
            manifest=manifest,
        )
        paths = {
            "cell_matrix_summary": (summary, output_dir / f"{sample_id}_cell_matrix_summary.csv"),
            "cell_matrix_enrichment": (enrichment, output_dir / f"{sample_id}_cell_matrix_enrichment.csv"),
            "matrix_tile_association": (association, output_dir / f"{sample_id}_matrix_tile_association.csv"),
            "cell_matrix_per_cell": (per_cell, output_dir / f"{sample_id}_cell_matrix_per_cell.csv"),
        }
        for key, (frame, path) in paths.items():
            frame.to_csv(path, index=key == "cell_matrix_per_cell")
            outputs[key] = str(path)
        plot = save_matrix_plot(summary, output_dir / f"{sample_id}_cell_matrix.png")
        outputs["cell_matrix_plot"] = str(plot) if plot else None

    manifest_path = output_dir / f"{sample_id}_colocalization_manifest.json"
    write_json(manifest_path, {"created_at": now(), "config": config, "outputs": outputs})
    outputs["manifest"] = str(manifest_path)
    update_status(status_path, "complete", "Co-localisation analysis complete", stage="complete", progress=1.0, outputs=outputs)
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
