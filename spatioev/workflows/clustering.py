#!/usr/bin/env python3
"""Background worker for the browser-based clustering/phenotyping workflow."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spatioev_clustering_matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "spatioev_clustering_numba"))
if not os.environ.get("LOKY_MAX_CPU_COUNT"):
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse

import spatioev as se
from spatioev.config import ClusteringConfig


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


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
        "cbar_kws": {"label": "Mean z-score" if scaled else "Mean expression"},
    }
    if scaled:
        heatmap_options["center"] = 0
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
    missing = sorted(set(config["markers"]) - set(source.var_names))
    if missing:
        raise ValueError(f"Clustering markers missing from AnnData: {missing}")

    scale = bool(config.get("scale", True))
    if scale:
        update_status(status_path, "running", "Z-score normalizing marker expression", stage="normalize")
        work = se.zscore_normalize(source)
    else:
        update_status(status_path, "running", "Using unscaled marker expression", stage="normalize")
        work = source.copy()
    update_status(status_path, "running", "Running PCA, neighbors, UMAP, and Leiden", stage="cluster")
    clustered = se.cluster_cells(work, clustering_config(config))

    stem = output_dir / f"{sample_id}_level0"
    clustered_path = Path(f"{stem}_clustered.h5ad")
    summary_path = Path(f"{stem}_cluster_summary.csv")
    umap_path = Path(f"{stem}_umap.png")
    heatmap_path = Path(f"{stem}_heatmap.png")
    update_status(status_path, "running", "Writing Level-0 checkpoint and QC", stage="write")
    clustered.write_h5ad(clustered_path)
    summary = cluster_summary(clustered)
    summary.to_csv(summary_path, index=False)
    save_umap(clustered, umap_path, title=f"{sample_id} Level-0 clusters")
    save_heatmap(clustered, heatmap_path, title=f"{sample_id} Level-0 marker profile", scaled=scale)
    outputs = {
        "clustered_h5ad": str(clustered_path),
        "summary_csv": str(summary_path),
        "umap_png": str(umap_path),
        "heatmap_png": str(heatmap_path),
        "n_cells": int(clustered.n_obs),
        "n_clusters": int(clustered.obs["leiden"].nunique()),
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
        if scale and scaled_source is None:
            update_status(status_path, "running", "Z-score normalizing marker expression", stage="normalize")
            scaled_source = se.zscore_normalize(source)
        clustering_source = scaled_source if scale else source
        subset = clustering_source[clustering_source.obs["annotation"].astype(str) == annotation].copy()
        if subset.n_obs == 0:
            raise ValueError(f"No cells found for refinement annotation {annotation!r}")
        clustered = se.cluster_cells(subset, clustering_config(refinement))
        slug = slugify(annotation)
        stem = output_dir / f"{sample_id}_refine_{slug}"
        clustered_path = Path(f"{stem}_clustered.h5ad")
        summary_path = Path(f"{stem}_cluster_summary.csv")
        umap_path = Path(f"{stem}_umap.png")
        heatmap_path = Path(f"{stem}_heatmap.png")
        clustered.write_h5ad(clustered_path)
        cluster_summary(clustered).to_csv(summary_path, index=False)
        save_umap(clustered, umap_path, title=f"{sample_id}: refine {annotation}")
        save_heatmap(
            clustered,
            heatmap_path,
            title=f"{sample_id}: {annotation} marker profile",
            scaled=scale,
        )
        outputs[annotation] = {
            "clustered_h5ad": str(clustered_path),
            "summary_csv": str(summary_path),
            "umap_png": str(umap_path),
            "heatmap_png": str(heatmap_path),
            "n_cells": int(clustered.n_obs),
            "n_clusters": int(clustered.obs["leiden"].nunique()),
        }
    update_status(status_path, "complete", "Refinement clustering complete", stage="complete", outputs=outputs)
    return outputs


def run_export(config: dict, status_path: Path) -> dict:
    sample_id = str(config["sample_id"])
    output_dir = Path(config["output_dir"])
    update_status(status_path, "running", "Loading source and reviewed mappings", stage="load")
    source = ad.read_h5ad(config["adata_path"])
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

    annotation_path = output_dir / f"{sample_id}_phenotyping_annotations.csv"
    phenotyped_path = output_dir / f"{sample_id}_phenotyped.h5ad"
    manifest_path = output_dir / f"{sample_id}_clustering_manifest.json"
    update_status(status_path, "running", "Writing final full-marker AnnData", stage="write")
    source.obs[["leiden_level0", "annotation", "annotation_level2"]].to_csv(annotation_path)
    source.write_h5ad(phenotyped_path)
    manifest = {
        "sample_id": sample_id,
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
