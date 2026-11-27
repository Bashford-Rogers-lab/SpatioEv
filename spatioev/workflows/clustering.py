#!/usr/bin/env python3
"""Background worker for the browser-based clustering/phenotyping workflow."""

from __future__ import annotations

import argparse
import json
import re
import traceback
import warnings
from pathlib import Path

from spatioev.workflows._io import configure_caches

configure_caches()

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from scipy.spatial import cKDTree

import spatioev as se
from spatioev.config import ClusteringConfig

from ._io import now, write_json
from .cellsam import write_h5ad_atomically


def output_folders(output_dir: Path) -> tuple[Path, Path]:
    """``qc/`` (figures, cluster summaries) and ``checkpoints/`` (clustered H5ADs).

    The results a user opens -- the annotation mappings they edit, the final
    phenotyped H5AD and annotation table -- stay at the top of the output
    folder; intermediate files go one level down.
    """
    qc_dir = Path(output_dir) / "qc"
    checkpoint_dir = Path(output_dir) / "checkpoints"
    qc_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return qc_dir, checkpoint_dir


def update_status(path: Path, state: str, message: str, **extra) -> None:
    payload = {
        "state": state,
        "message": message,
        "updated_at": now(),
        **extra,
    }
    write_json(path, payload)


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")
    return slug or "cluster"


def cluster_imageid(config: dict) -> str | None:
    value = config.get("cluster_imageid")
    return str(value).strip() if value is not None and str(value).strip() else None


def scale_groupby(config: dict) -> str | None:
    """``"imageid"`` when markers are z-scored within each FOV, else ``None``."""
    return "imageid" if bool(config.get("scale_within_image", False)) else None


def zscore(source: ad.AnnData, config: dict) -> ad.AnnData:
    groupby = scale_groupby(config)
    if groupby is not None and groupby not in source.obs:
        raise KeyError("AnnData has no 'imageid' column, so markers cannot be z-scored within each FOV")
    return se.zscore_normalize(source, groupby=groupby)


CENTROID_COLUMNS = ("X_centroid", "Y_centroid")
# Fewer cells than this per neighbourhood, on average, means the radius holds
# little more than the cell itself: usually a pixel size that is too large.
MIN_MEAN_CELLS_AVERAGED = 2.0


def neighbourhood_radius_px(config: dict) -> float | None:
    """The averaging radius in pixels, or ``None`` when markers stay per cell."""
    radius_um = float(config.get("neighbourhood_um") or 0)
    if radius_um <= 0:
        return None
    pixel_size_um = float(config.get("pixel_size_um") or 0)
    if pixel_size_um <= 0:
        raise ValueError(
            "Averaging markers over neighbouring cells needs the pixel size "
            "of the scan (µm per pixel)"
        )
    return radius_um / pixel_size_um


def neighbourhood_means(
    values: np.ndarray, coords: np.ndarray, groups: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """Mean of each cell and every cell within ``radius`` of it, per group.

    Neighbours are looked for only within a cell's own group (FOV), since
    pixel coordinates of different FOVs overlap. Missing values are skipped.
    Returns the means and, for every cell, how many cells were averaged.
    """
    values = np.asarray(values, dtype=float)
    means = np.full_like(values, np.nan)
    counts = np.zeros(len(values), dtype=int)
    for group in pd.unique(groups):
        index = np.flatnonzero(groups == group)
        pairs = cKDTree(coords[index]).query_pairs(radius, output_type="ndarray")
        diagonal = np.arange(len(index))
        rows = np.concatenate([pairs[:, 0], pairs[:, 1], diagonal])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0], diagonal])
        neighbours = sparse.csr_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(len(index), len(index))
        )
        block = values[index]
        finite = np.isfinite(block)
        sums = neighbours @ np.where(finite, block, 0.0)
        n_finite = neighbours @ finite.astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            means[index] = sums / n_finite
        counts[index] = np.diff(neighbours.indptr)
    return means, counts


def average_over_neighbours(source: ad.AnnData, config: dict) -> tuple[ad.AnnData, float | None]:
    """Replace each clustering marker by its mean over the cell's neighbourhood.

    Clusters of averaged markers follow tissue structures rather than single
    cells, which can help when a structure's own cells lack a reliable marker
    (most interstitial cells in a testis, for example). Markers listed in
    ``per_cell_markers`` keep each cell's own value: a structure one cell
    thick, such as the smooth-muscle wall around a tubule, would otherwise be
    smeared into the cells on both sides of it.

    Returns the clustering markers of ``source`` (averaged, unless averaging
    is off) and the mean number of cells averaged per cell (``None`` if off).
    """
    radius = neighbourhood_radius_px(config)
    if radius is None:
        return source, None
    markers = list(config["markers"])
    per_cell = list(config.get("per_cell_markers") or [])
    unknown = sorted(set(per_cell) - set(markers))
    if unknown:
        raise ValueError(f"Markers kept per cell are not clustering markers: {unknown}")
    missing = [column for column in CENTROID_COLUMNS if column not in source.obs]
    if missing:
        raise KeyError(f"AnnData has no {', '.join(missing)} column, so markers cannot be averaged over neighbours")

    work = source[:, markers].copy()
    groups = (
        work.obs["imageid"].astype(str).to_numpy()
        if "imageid" in work.obs
        else np.zeros(work.n_obs, dtype=int)
    )
    coords = work.obs[list(CENTROID_COLUMNS)].to_numpy(dtype=float)
    x = dense_matrix(work.X).astype(float)
    averaged = [position for position, marker in enumerate(markers) if marker not in per_cell]
    means, counts = neighbourhood_means(x[:, averaged], coords, groups, radius)
    x[:, averaged] = means
    work.X = x.astype(np.float32)
    return work, float(counts.mean())


def remove_cell_brightness(source: ad.AnnData, markers: list[str]) -> ad.AnnData:
    """Subtract each cell's mean over ``markers`` from its z-scored markers.

    Big, bright or tightly packed cells (and their neighbours' spill-over)
    raise every marker together, so clusters split cells by brightness instead
    of by which markers stand out. What is left after subtracting the mean is
    the cell's marker profile. Missing values are skipped. Returns the
    ``markers`` of ``source`` only.
    """
    if len(markers) < 2:
        raise ValueError("Removing each cell's overall brightness needs at least two markers")
    work = source[:, list(markers)].copy()
    x = dense_matrix(work.X).astype(float)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        # A cell with every marker missing has no mean; it stays missing.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        x = x - np.nanmean(x, axis=1, keepdims=True)
    work.X = x.astype(np.float32)
    return work


def subset_for_clustering(source: ad.AnnData, config: dict) -> ad.AnnData:
    imageid = cluster_imageid(config)
    if imageid is None:
        return source
    if "imageid" not in source.obs:
        raise KeyError("AnnData has no 'imageid' column for FOV-specific clustering")
    mask = source.obs["imageid"].astype(str).eq(imageid)
    if not mask.any():
        raise ValueError(f"No cells have imageid={imageid!r}")
    return source[mask].copy()


def scoped_sample_id(sample_id: str, config: dict) -> str:
    imageid = cluster_imageid(config)
    return f"{sample_id}_{slugify(imageid)}" if imageid is not None else sample_id


def dense_matrix(x) -> np.ndarray:
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def natural_key(value: object) -> tuple[int, int | str]:
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def cluster_summary(clustered: ad.AnnData, cluster_key: str = "leiden") -> pd.DataFrame:
    labels = clustered.obs[cluster_key].astype(str)
    x = dense_matrix(clustered.X).astype(float, copy=False)
    rows = []
    for cluster in sorted(labels.unique(), key=natural_key):
        mask = labels.to_numpy() == cluster
        means = np.nanmean(x[mask], axis=0)
        order = np.argsort(means)[::-1]
        top = [str(clustered.var_names[index]) for index in order[: min(4, len(order))]]
        row = {
            "cluster": cluster,
            "n_cells": int(mask.sum()),
            "fraction": float(mask.mean()),
            "top_markers": ", ".join(top),
        }
        row.update({f"mean_{marker}": float(value) for marker, value in zip(clustered.var_names, means)})
        rows.append(row)
    return pd.DataFrame(rows)


def save_umap(clustered: ad.AnnData, path: Path, *, title: str, max_points: int = 100_000) -> None:
    coords = np.asarray(clustered.obsm["X_umap"], dtype=float)
    labels = clustered.obs["leiden"].astype(str).to_numpy()
    rng = np.random.default_rng(0)
    if len(coords) > max_points:
        selected = []
        unique, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique, counts):
            indexes = np.flatnonzero(labels == label)
            target = max(100, int(round(max_points * count / len(labels))))
            selected.extend(rng.choice(indexes, size=min(target, len(indexes)), replace=False))
        selected = np.asarray(selected, dtype=int)
        coords = coords[selected]
        labels = labels[selected]

    clusters = sorted(np.unique(labels), key=natural_key)
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=160)
    for index, cluster in enumerate(clusters):
        mask = labels == cluster
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=2.0,
            alpha=0.55,
            linewidths=0,
            color=cmap(index % 20),
            label=cluster,
            rasterized=True,
        )
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, markerscale=4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# Colour limit of z-scored heatmaps. A small cluster of cells with an empty
# channel can average -7 or less; scaled to it, the differences that name the
# other clusters (around +-1) all look white. Values beyond stay printed.
HEATMAP_Z_LIMIT = 3.0


def save_heatmap(clustered: ad.AnnData, path: Path, *, title: str, scaled: bool = True) -> None:
    labels = clustered.obs["leiden"].astype(str)
    x = pd.DataFrame(dense_matrix(clustered.X), columns=clustered.var_names, index=clustered.obs_names)
    means = x.groupby(labels, observed=True).mean()
    means = means.reindex(sorted(means.index.astype(str), key=natural_key))
    height = max(3.8, 0.42 * len(means) + 1.5)
    fig, ax = plt.subplots(figsize=(max(7.0, 0.62 * len(means.columns) + 2.5), height), dpi=160)
    heatmap_options = {
        "cmap": "vlag" if scaled else "viridis",
        "ax": ax,
        "annot": True,
        "fmt": ".1f",
        "annot_kws": {"fontsize": 7},
        "cbar_kws": {
            "label": f"Mean z-score (colour capped at ±{HEATMAP_Z_LIMIT:g})" if scaled else "Mean expression"
        },
    }
    if scaled:
        heatmap_options.update(center=0, vmin=-HEATMAP_Z_LIMIT, vmax=HEATMAP_Z_LIMIT)
    sns.heatmap(means, **heatmap_options)
    ax.set_title(title)
    ax.set_xlabel("Marker")
    ax.set_ylabel("Cluster")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def clustering_config(payload: dict) -> ClusteringConfig:
    return ClusteringConfig(
        markers=list(payload["markers"]),
        resolution=float(payload.get("resolution", 0.5)),
        n_neighbors=int(payload.get("n_neighbors", 10)),
        n_pcs=int(payload.get("n_pcs", 15)),
        scale=bool(payload.get("scale", True)),
    )


def run_level0(config: dict, status_path: Path) -> dict:
    sample_id = str(config["sample_id"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    update_status(status_path, "running", "Loading AnnData", stage="load")
    source = ad.read_h5ad(config["adata_path"])
    source = subset_for_clustering(source, config)
    missing = sorted(set(config["markers"]) - set(source.var_names))
    if missing:
        raise ValueError(f"Clustering markers missing from AnnData: {missing}")

    radius_um = float(config.get("neighbourhood_um") or 0)
    if radius_um > 0:
        update_status(
            status_path,
            "running",
            f"Averaging markers over neighbouring cells within {radius_um:g} µm",
            stage="normalize",
        )
    source, mean_cells_averaged = average_over_neighbours(source, config)

    scale = bool(config.get("scale", True))
    if scale:
        within = " within each FOV" if scale_groupby(config) else ""
        update_status(status_path, "running", f"Z-score normalizing marker expression{within}", stage="normalize")
        work = zscore(source, config)
    else:
        update_status(status_path, "running", "Using unscaled marker expression", stage="normalize")
        work = source.copy()
    update_status(status_path, "running", "Running PCA, neighbors, UMAP, and Leiden", stage="cluster")
    clustered = se.cluster_cells(work, clustering_config(config))

    scoped_id = scoped_sample_id(sample_id, config)
    qc_dir, checkpoint_dir = output_folders(output_dir)
    clustered_path = checkpoint_dir / f"{scoped_id}_level0_clustered.h5ad"
    summary_path = qc_dir / f"{scoped_id}_level0_cluster_summary.csv"
    umap_path = qc_dir / f"{scoped_id}_level0_umap.png"
    heatmap_path = qc_dir / f"{scoped_id}_level0_heatmap.png"
    update_status(status_path, "running", "Writing Level-0 checkpoint and QC", stage="write")
    write_h5ad_atomically(clustered, clustered_path)
    summary = cluster_summary(clustered)
    summary.to_csv(summary_path, index=False)
    save_umap(clustered, umap_path, title=f"{scoped_id} Level-0 clusters")
    averaged_note = f" (averaged over {radius_um:g} µm)" if mean_cells_averaged is not None else ""
    save_heatmap(
        clustered,
        heatmap_path,
        title=f"{scoped_id} Level-0 marker profile{averaged_note}",
        scaled=scale,
    )
    outputs = {
        "clustered_h5ad": str(clustered_path),
        "summary_csv": str(summary_path),
        "umap_png": str(umap_path),
        "heatmap_png": str(heatmap_path),
        "n_cells": int(clustered.n_obs),
        "n_clusters": int(clustered.obs["leiden"].nunique()),
        "cluster_imageid": cluster_imageid(config),
        "scale_within_image": bool(scale and scale_groupby(config)),
        "neighbourhood_um": radius_um if mean_cells_averaged is not None else 0.0,
        "mean_cells_averaged": mean_cells_averaged,
    }
    update_status(status_path, "complete", "Level-0 clustering complete", stage="complete", outputs=outputs)
    return outputs


def attach_level0_annotations(
    source: ad.AnnData,
    level0: ad.AnnData,
    mapping_path: Path,
) -> ad.AnnData:
    missing_from_checkpoint = source.obs_names.difference(level0.obs_names)
    absent_from_source = level0.obs_names.difference(source.obs_names)
    if len(missing_from_checkpoint) or len(absent_from_source):
        raise ValueError(
            "Source AnnData and Level-0 checkpoint contain different cell IDs "
            f"({len(missing_from_checkpoint)} missing from checkpoint; "
            f"{len(absent_from_source)} absent from source)"
        )
    mapping_table = pd.read_csv(mapping_path)
    mapping = dict(zip(mapping_table["cluster"].astype(str), mapping_table["annotation"].astype(str)))
    cluster_labels = level0.obs["leiden"].astype(str)
    annotation = cluster_labels.map(mapping)
    if annotation.isna().any():
        missing = sorted(cluster_labels.loc[annotation.isna()].unique())
        raise ValueError(f"Level-0 mapping is incomplete for clusters: {missing}")
    source.obs["leiden_level0"] = cluster_labels.reindex(source.obs_names)
    source.obs["annotation"] = annotation.reindex(source.obs_names)
    return source


def run_refinement(config: dict, status_path: Path) -> dict:
    sample_id = str(config["sample_id"])
    output_dir = Path(config["output_dir"])
    update_status(status_path, "running", "Loading source and Level-0 annotations", stage="load")
    source = ad.read_h5ad(config["adata_path"])
    source = subset_for_clustering(source, config)
    level0 = ad.read_h5ad(config["level0_h5ad"])
    source = attach_level0_annotations(source, level0, Path(config["level0_mapping_path"]))
    scaled_source = None

    outputs = {}
    refinements = list(config.get("refinements", []))
    for position, refinement in enumerate(refinements, start=1):
        annotation = str(refinement["annotation"])
        update_status(
            status_path,
            "running",
            f"Refining {annotation} ({position}/{len(refinements)})",
            stage="refine",
            current_annotation=annotation,
        )
        scale = bool(refinement.get("scale", True))
        remove_brightness = bool(refinement.get("remove_brightness", False))
        if remove_brightness and not scale:
            raise ValueError(
                f"Refinement of {annotation!r}: removing each cell's overall brightness "
                "needs z-scored markers"
            )
        if scale and scaled_source is None:
            update_status(status_path, "running", "Z-score normalizing marker expression", stage="normalize")
            scaled_source = zscore(source, config)
        clustering_source = scaled_source if scale else source
        subset = clustering_source[clustering_source.obs["annotation"].astype(str) == annotation].copy()
        if subset.n_obs == 0:
            raise ValueError(f"No cells found for refinement annotation {annotation!r}")
        if remove_brightness:
            subset = remove_cell_brightness(subset, list(refinement["markers"]))
        clustered = se.cluster_cells(subset, clustering_config(refinement))
        slug = slugify(annotation)
        scoped_id = scoped_sample_id(sample_id, config)
        qc_dir, checkpoint_dir = output_folders(output_dir)
        clustered_path = checkpoint_dir / f"{scoped_id}_refine_{slug}_clustered.h5ad"
        summary_path = qc_dir / f"{scoped_id}_refine_{slug}_cluster_summary.csv"
        umap_path = qc_dir / f"{scoped_id}_refine_{slug}_umap.png"
        heatmap_path = qc_dir / f"{scoped_id}_refine_{slug}_heatmap.png"
        write_h5ad_atomically(clustered, clustered_path)
        cluster_summary(clustered).to_csv(summary_path, index=False)
        save_umap(clustered, umap_path, title=f"{scoped_id}: refine {annotation}")
        brightness_note = " (cell brightness removed)" if remove_brightness else ""
        save_heatmap(
            clustered,
            heatmap_path,
            title=f"{scoped_id}: {annotation} marker profile{brightness_note}",
            scaled=scale,
        )
        outputs[annotation] = {
            "clustered_h5ad": str(clustered_path),
            "summary_csv": str(summary_path),
            "umap_png": str(umap_path),
            "heatmap_png": str(heatmap_path),
            "n_cells": int(clustered.n_obs),
            "n_clusters": int(clustered.obs["leiden"].nunique()),
            "remove_brightness": remove_brightness,
        }
    update_status(status_path, "complete", "Refinement clustering complete", stage="complete", outputs=outputs)
    return outputs


def run_export(config: dict, status_path: Path) -> dict:
    sample_id = str(config["sample_id"])
    output_dir = Path(config["output_dir"])
    update_status(status_path, "running", "Loading source and reviewed mappings", stage="load")
    source = ad.read_h5ad(config["adata_path"])
    source = subset_for_clustering(source, config)
    level0 = ad.read_h5ad(config["level0_h5ad"])
    source = attach_level0_annotations(source, level0, Path(config["level0_mapping_path"]))
    source.obs["annotation_level2"] = source.obs["annotation"].astype("object")

    level0_umap = np.full((source.n_obs, 2), np.nan, dtype=np.float32)
    locations = source.obs_names.get_indexer(level0.obs_names)
    if (locations < 0).any():
        raise ValueError("Level-0 checkpoint contains cell IDs absent from source AnnData")
    level0_umap[locations] = np.asarray(level0.obsm["X_umap"], dtype=np.float32)
    source.obsm["X_umap_level0"] = level0_umap

    for refinement in config.get("refinements", []):
        clustered = ad.read_h5ad(refinement["clustered_h5ad"])
        mapping_table = pd.read_csv(refinement["mapping_path"])
        mapping = dict(zip(mapping_table["cluster"].astype(str), mapping_table["annotation"].astype(str)))
        labels = clustered.obs["leiden"].astype(str)
        annotation = labels.map(mapping)
        if annotation.isna().any():
            missing = sorted(labels.loc[annotation.isna()].unique())
            raise ValueError(f"Refinement mapping incomplete for {refinement['parent']}: {missing}")
        source.obs.loc[clustered.obs_names, "annotation_level2"] = annotation.astype(object)

    scoped_id = scoped_sample_id(sample_id, config)
    annotation_path = output_dir / f"{scoped_id}_phenotyping_annotations.csv"
    phenotyped_path = output_dir / f"{scoped_id}_phenotyped.h5ad"
    manifest_path = output_dir / f"{scoped_id}_clustering_manifest.json"
    update_status(status_path, "running", "Writing final full-marker AnnData", stage="write")
    source.obs[["leiden_level0", "annotation", "annotation_level2"]].to_csv(annotation_path)
    write_h5ad_atomically(source, phenotyped_path)
    manifest = {
        "sample_id": sample_id,
        "cluster_imageid": cluster_imageid(config),
        "created_at": now(),
        "source_adata": str(config["adata_path"]),
        "source_image": str(config["image_path"]),
        "level0_config": config.get("level0_config"),
        "level0_mapping": str(config["level0_mapping_path"]),
        "refinements": config.get("refinements", []),
        "annotation_csv": str(annotation_path),
        "phenotyped_h5ad": str(phenotyped_path),
    }
    write_json(manifest_path, manifest)
    outputs = {
        "annotation_csv": str(annotation_path),
        "phenotyped_h5ad": str(phenotyped_path),
        "manifest_json": str(manifest_path),
        "n_cells": int(source.n_obs),
        "n_final_labels": int(source.obs["annotation_level2"].nunique()),
    }
    update_status(status_path, "complete", "Final phenotyping export complete", stage="complete", outputs=outputs)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=["level0", "refine", "export"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    try:
        if args.action == "level0":
            run_level0(config, args.status)
        elif args.action == "refine":
            run_refinement(config, args.status)
        else:
            run_export(config, args.status)
    except Exception as exc:
        update_status(
            args.status,
            "failed",
            str(exc),
            stage="failed",
            traceback=traceback.format_exc(),
        )
        raise


if __name__ == "__main__":
    main()
