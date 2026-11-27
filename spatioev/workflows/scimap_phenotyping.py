#!/usr/bin/env python3
"""Background worker for flexible SCIMAP phenotyping of selected broad populations."""

from __future__ import annotations

import argparse
import json
import math
import re
import traceback
from pathlib import Path

from spatioev.workflows._io import configure_caches

configure_caches()

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile
import zarr

from spatioev.workflows import marker_gating as mgq
from spatioev.workflows.image_collection import (
    imageids_with_images,
    natural_key,
    plane_marker_names,
    resolve_image,
    zarr_array,
)

from ._io import now, write_json

ALLOWED_RULES = {"pos", "neg", "allpos", "allneg", "anypos", "anyneg"}






def update_status(path: Path, state: str, message: str, *, stage: str, progress: float, **extra) -> None:
    write_json(
        path,
        {
            "state": state,
            "message": message,
            "stage": stage,
            "progress": progress,
            "updated_at": now(),
            **extra,
        },
    )


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()
    return slug or "subset"


def read_annotations(path_text: str | None) -> pd.DataFrame | None:
    if not path_text or not str(path_text).strip():
        return None
    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    annotations = pd.read_csv(path, index_col=0)
    annotations.index = annotations.index.astype(str)
    if not annotations.index.is_unique:
        raise ValueError("Broad annotation CSV contains duplicated cell IDs")
    return annotations


def attach_annotations(adata: ad.AnnData, annotation_path: str | None) -> ad.AnnData:
    annotations = read_annotations(annotation_path)
    if annotations is None:
        return adata
    missing = adata.obs_names.difference(annotations.index)
    extra = annotations.index.difference(adata.obs_names)
    if len(missing) or len(extra):
        raise ValueError(
            "Broad annotations and gated AnnData contain different cell IDs: "
            f"{len(missing)} cells missing annotations; {len(extra)} annotations absent from AnnData"
        )
    for column in annotations.columns:
        adata.obs[column] = annotations[column].reindex(adata.obs_names)
    return adata


#: Column names accepted for a single gate that applies to every image.
GLOBAL_GATE_COLUMNS = ("gates", "gate", "global", "selected_log1p_gate")
#: What to do for an image that has no gate for a workflow marker.
MISSING_GATE_POLICIES = ("error", "median", "gmm")


def resolve_marker_aliases(
    names: list[str], available: list[str]
) -> tuple[dict[str, str], list[str]]:
    """Map marker names onto the AnnData's marker names.

    An exact name wins; then a case/punctuation-insensitive match
    (``NAKATPASE`` -> ``NaKATPase``); then a unique match after dropping a
    trailing ``_<number>``, which pairs the antibody-lot/cycle variants panels
    accumulate (``CLDN11_1`` -> ``CLDN11_2``, ``Vimentin_1`` -> ``Vimentin_2``).
    Returns the mapping and the names that matched nothing (or more than one).
    """
    from spatioev.workflows.cellsam import marker_key

    def stem(name: str) -> str:
        return marker_key(re.sub(r"_\d+$", "", str(name).strip()))

    by_key: dict[str, list[str]] = {}
    by_stem: dict[str, list[str]] = {}
    for marker in available:
        by_key.setdefault(marker_key(marker), []).append(marker)
        by_stem.setdefault(stem(marker), []).append(marker)
    mapping: dict[str, str] = {}
    unresolved: list[str] = []
    for name in names:
        name = str(name).strip()
        if name in available:
            mapping[name] = name
            continue
        for candidates in (by_key.get(marker_key(name), []), by_stem.get(stem(name), [])):
            if len(candidates) == 1:
                mapping[name] = candidates[0]
                break
        else:
            unresolved.append(name)
    return mapping, unresolved


def read_workflow(workflow_path: Path, expression_markers: list[str]) -> tuple[pd.DataFrame, dict[str, str]]:
    """Read a phenotype workflow and rename its marker columns to AnnData names."""
    workflow = pd.read_csv(workflow_path)
    if workflow.shape[1] < 3:
        raise ValueError("Phenotype workflow must contain parent, phenotype, and at least one marker column")
    marker_columns = workflow.columns[2:].astype(str).tolist()
    mapping, unresolved = resolve_marker_aliases(marker_columns, expression_markers)
    if unresolved:
        raise ValueError(f"Workflow markers absent from AnnData: {sorted(unresolved)}")
    targets = [mapping[column] for column in marker_columns]
    duplicated = sorted({target for target in targets if targets.count(target) > 1})
    if duplicated:
        raise ValueError(f"Several workflow columns resolve to the same marker: {duplicated}")
    renamed = {column: target for column, target in mapping.items() if column != target}
    workflow = workflow.rename(columns=renamed)
    return workflow, renamed


def normalise_gate_table(
    raw: pd.DataFrame,
    expression_markers: list[str],
    imageids: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Turn a gate CSV into one gate per marker and image.

    Accepted layouts, all on the log1p scale of the gated expression values:

    - ``markers,gates`` -- one gate per marker, used for every image (also
      ``gate``/``global``, or SpatioEv's ``marker,selected_log1p_gate``);
    - ``markers,fov1,fov2,...`` -- one column per image ID (scimap's per-image
      layout); a ``gates`` column, when present too, fills the images that
      have no column of their own.

    Returns a frame indexed by marker with one column per image ID (or a
    single ``gates`` column when no image IDs are given), and a description.
    """
    frame = raw.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    if "markers" not in frame and "marker" in frame:
        frame = frame.rename(columns={"marker": "markers"})
    if "markers" not in frame:
        raise ValueError("SCIMAP gate CSV must contain a 'markers' column")
    frame["markers"] = frame["markers"].astype(str).str.strip()
    frame = frame.loc[frame["markers"].ne("") & frame["markers"].ne("nan")]
    if frame["markers"].duplicated().any():
        duplicated = frame.loc[frame["markers"].duplicated(keep=False), "markers"].unique().tolist()
        raise ValueError(f"SCIMAP gate table contains duplicated markers: {duplicated}")
    global_column = next((column for column in GLOBAL_GATE_COLUMNS if column in frame), None)
    image_set = [str(imageid) for imageid in (imageids or [])]
    image_columns = [column for column in frame.columns if column in image_set]
    if global_column is None and not image_columns:
        raise ValueError(
            "SCIMAP gate CSV must contain 'markers' and 'gates' columns, or one "
            "gate column per image ID (for example markers,fov1,fov2)"
        )
    mapping, unresolved = resolve_marker_aliases(frame["markers"].tolist(), expression_markers)
    frame = frame.loc[frame["markers"].isin(mapping)].copy()
    frame["markers"] = frame["markers"].map(mapping)
    if frame["markers"].duplicated().any():
        duplicated = frame.loc[frame["markers"].duplicated(keep=False), "markers"].unique().tolist()
        raise ValueError(f"Several gate rows resolve to the same marker: {duplicated}")
    frame = frame.set_index("markers")
    default = (
        pd.to_numeric(frame[global_column], errors="raise")
        if global_column is not None
        else pd.Series(np.nan, index=frame.index)
    )
    if image_set:
        gates = pd.DataFrame(index=frame.index, columns=image_set, dtype=float)
        for imageid in image_set:
            values = (
                pd.to_numeric(frame[imageid], errors="raise")
                if imageid in image_columns
                else pd.Series(np.nan, index=frame.index)
            )
            gates[imageid] = values.fillna(default)
    else:
        gates = pd.DataFrame({"gates": default})
    gates.index.name = "markers"
    info = {
        "layout": "per-image" if image_columns else "global",
        "global_column": global_column,
        "image_columns": image_columns,
        "renamed_markers": {name: target for name, target in mapping.items() if name != target},
        "ignored_markers": unresolved,
    }
    return gates, info


def fill_missing_gates(
    gates: pd.DataFrame, markers: list[str], policy: str = "error"
) -> tuple[pd.DataFrame, list[str]]:
    """Check every *marker* has a gate in every image; fill gaps per *policy*.

    ``error`` stops with the missing marker/image pairs; ``median`` uses the
    median of that marker's gates in the other images; ``gmm`` leaves the gap
    for scimap's Gaussian-mixture gate estimate. Returns the table and notes.
    """
    if policy not in MISSING_GATE_POLICIES:
        raise ValueError(f"Missing-gate policy must be one of {MISSING_GATE_POLICIES}")
    absent = sorted(set(markers) - set(gates.index))
    if absent:
        raise ValueError(f"Workflow markers absent from gate CSV: {absent}")
    gates = gates.copy()
    block = gates.loc[markers]
    missing = [
        (marker, str(imageid))
        for marker in markers
        for imageid in gates.columns
        if pd.isna(block.loc[marker, imageid])
    ]
    notes: list[str] = []
    if not missing:
        return gates, notes
    if policy == "error":
        shown = ", ".join(f"{marker} in {imageid}" for marker, imageid in missing[:12])
        more = f" and {len(missing) - 12} more" if len(missing) > 12 else ""
        raise ValueError(
            f"No gate for {shown}{more}. Add those gates, restrict the run to the "
            "images that have gates, or choose how to fill the gaps "
            "('median' of the other images, or 'gmm')."
        )
    if policy == "median":
        for marker in sorted({marker for marker, _ in missing}):
            row = pd.to_numeric(gates.loc[marker], errors="coerce")
            if row.notna().sum() == 0:
                raise ValueError(f"Marker {marker} has no gate in any image to take a median from")
            gates.loc[marker] = row.fillna(float(row.median()))
        notes.append(
            f"Filled {len(missing)} missing marker/image gates with the median of "
            "that marker's gates in the other images."
        )
    else:
        notes.append(
            f"{len(missing)} marker/image gates are missing; scimap estimates them "
            "with a Gaussian mixture model."
        )
    return gates, notes


def workflow_rule_notes(workflow: pd.DataFrame) -> list[str]:
    """Point out rules that scimap does not enforce the way they read.

    scimap enforces ``pos``/``neg`` only when a row lists two or more markers
    of that kind. A *single* ``pos`` or ``neg`` marker next to other rules is
    averaged into the row's score, so a cell can get the label while failing
    that marker (for example a dim SYCP3+ spermatocyte still scoring highest
    as ``MAGEA4 pos, SYCP3 neg`` spermatogonia). ``allpos``/``allneg``
    require the marker.
    """
    notes = []
    phenotype_column = workflow.columns[1]
    for _, row in workflow.iterrows():
        rules = row.iloc[2:].dropna().astype(str).str.strip().str.lower()
        if rules.nunique() < 2:
            continue
        for rule, required in (("pos", "allpos"), ("neg", "allneg")):
            markers = rules.index[rules.eq(rule)].tolist()
            if len(markers) == 1:
                notes.append(
                    f"{row[phenotype_column]}: the single '{rule}' marker {markers[0]} is "
                    f"averaged into the score, not required. Use '{required}' if cells must "
                    f"be {'positive' if rule == 'pos' else 'negative'} for {markers[0]}."
                )
    return notes


def validate_workflow(
    workflow_path: Path,
    gate_path: Path,
    expression_markers: list[str],
    imageids: list[str] | None = None,
    missing_gate_policy: str = "gmm",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Read and cross-check a phenotype workflow and its gate table.

    Workflow and gate marker names are matched to the AnnData leniently (see
    :func:`resolve_marker_aliases`). Every workflow marker needs a gate row;
    whether every *image* needs one is decided by ``missing_gate_policy``
    (see :func:`fill_missing_gates`).
    """
    workflow, renamed = read_workflow(workflow_path, expression_markers)
    gates, gate_info = normalise_gate_table(pd.read_csv(gate_path), expression_markers, imageids)
    workflow_markers = workflow.columns[2:].astype(str).tolist()
    gates, gate_notes = fill_missing_gates(gates, workflow_markers, missing_gate_policy)

    rules = workflow.iloc[:, 2:].stack().dropna().astype(str).str.strip().str.lower()
    invalid_rules = sorted(set(rules) - ALLOWED_RULES)
    if invalid_rules:
        raise ValueError(f"Unsupported phenotype-workflow rules: {invalid_rules}")
    parents = set(workflow.iloc[:, 0].dropna().astype(str))
    children = set(workflow.iloc[:, 1].dropna().astype(str))
    missing_parents = sorted(parents - children - {"all"})
    if missing_parents:
        raise ValueError(f"Workflow parents not defined as phenotypes: {missing_parents}")

    notes = [
        f"Workflow column {name!r} uses AnnData marker {target!r}"
        for name, target in renamed.items()
    ] + [
        f"Gate row {name!r} uses AnnData marker {target!r}"
        for name, target in gate_info["renamed_markers"].items()
    ] + gate_notes + workflow_rule_notes(workflow)
    if gate_info["ignored_markers"]:
        notes.append(
            f"Gate rows for markers not in the AnnData were ignored: {gate_info['ignored_markers']}"
        )
    summary = {
        "workflow_rows": int(len(workflow)),
        "workflow_markers": workflow_markers,
        "parent_column": str(workflow.columns[0]),
        "phenotype_column": str(workflow.columns[1]),
        "top_level_phenotypes": workflow.loc[
            workflow.iloc[:, 0].astype(str).eq("all"), workflow.columns[1]
        ].dropna().astype(str).tolist(),
        "gate_markers": int(len(gates)),
        "gate_layout": gate_info["layout"],
        "gate_image_columns": gate_info["image_columns"],
        "renamed_workflow_markers": renamed,
        "notes": notes,
    }
    return workflow, gates, summary


def inspect_inputs(config: dict) -> dict:
    required = {
        "gated_h5ad": Path(config["gated_h5ad"]).expanduser().resolve(),
        "image_path": Path(config["image_path"]).expanduser().resolve(),
        "gate_csv": Path(config["gate_csv"]).expanduser().resolve(),
        "workflow_csv": Path(config["workflow_csv"]).expanduser().resolve(),
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))

    backed = ad.read_h5ad(required["gated_h5ad"], backed="r")
    n_cells = int(backed.n_obs)
    markers = [str(marker) for marker in backed.var_names]
    planes = plane_marker_names(backed)
    layers = list(backed.layers.keys())
    obs = backed.obs.copy()
    obs.index = obs.index.astype(str)
    backed.file.close()

    annotations = read_annotations(config.get("broad_annotations"))
    if annotations is not None:
        missing_ids = obs.index.difference(annotations.index)
        extra_ids = annotations.index.difference(obs.index)
        if len(missing_ids) or len(extra_ids):
            raise ValueError(
                "Broad annotations and gated AnnData contain different cell IDs: "
                f"{len(missing_ids)} cells missing annotations; {len(extra_ids)} extra annotations"
            )
        for column in annotations.columns:
            obs[column] = annotations[column].reindex(obs.index)

    if "imageid" not in obs:
        raise KeyError("Gated AnnData must contain an 'imageid' observation column for SCIMAP")
    x_column, y_column = coordinate_columns(obs)
    imageids = sorted(
        obs["imageid"].dropna().astype(str).unique().tolist(), key=natural_key
    )
    if not imageids:
        raise ValueError("AnnData 'imageid' column contains no usable image IDs")
    selected_imageids = selected_images(config, imageids)
    # A TMA image folder may hold images for only some FOVs; review defaults
    # to a selected one that has an image, and no image only turns review off.
    # It never leaves the selection: the run reviews only phenotyped cells.
    with_image = imageids_with_images(required["image_path"], imageids)
    review_imageid = str(
        config.get("review_imageid")
        or next((i for i in selected_imageids if i in with_image), None)
        or selected_imageids[0]
    )
    if review_imageid not in imageids:
        raise ValueError(f"Review image ID {review_imageid!r} is absent from AnnData")
    review_image = (
        resolve_image(required["image_path"], review_imageid)
        if review_imageid in with_image
        else None
    )

    workflow, gates, workflow_summary = validate_workflow(
        required["workflow_csv"],
        required["gate_csv"],
        markers,
        imageids=selected_imageids,
        missing_gate_policy=str(config.get("missing_gate_policy") or "error"),
    )
    candidate_columns: dict[str, list[dict]] = {}
    for column in obs.columns:
        if str(column).startswith("gate_") or str(column) in {"fov", "mask_type", "imageid"}:
            continue
        series = obs[column]
        unique_count = int(series.nunique(dropna=True))
        if unique_count > 200:
            continue
        column_key = str(column).lower()
        looks_like_population = any(
            token in column_key
            for token in ["annotation", "cluster", "leiden", "phenotype", "celltype", "cell_type"]
        )
        if not (
            isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(series.dtype)
            or pd.api.types.is_bool_dtype(series.dtype)
            or looks_like_population
        ):
            continue
        counts = series.dropna().astype(str).value_counts()
        candidate_columns[str(column)] = [
            {"value": str(value), "n_cells": int(count)} for value, count in counts.items()
        ]
    if not candidate_columns:
        raise ValueError("No categorical broad-population columns were found in AnnData or annotation CSV")

    return {
        "n_cells": n_cells,
        "n_markers": len(markers),
        "markers": markers,
        "candidate_columns": candidate_columns,
        "workflow": workflow_summary,
        "workflow_preview": workflow.fillna("").astype(str).to_dict(orient="records"),
        "gate_markers": gates.index.astype(str).tolist(),
        "gate_table": gates.reset_index().to_dict(orient="records"),
        "layers": layers,
        "imageids": imageids,
        "selected_imageids": selected_imageids,
        "imageids_with_image": with_image,
        "default_review_imageid": review_imageid,
        "review_image_path": str(review_image) if review_image is not None else "",
        "image_channels": (
            mgq.canonical_channel_names(review_image, planes)
            if review_image is not None
            else list(planes)
        ),
        "coordinate_columns": {"x": x_column, "y": y_column},
    }


def selected_images(config: dict, imageids: list[str]) -> list[str]:
    """The image IDs a run covers: ``config["imageids"]`` when given, else all."""
    chosen = [str(value) for value in (config.get("imageids") or []) if str(value).strip()]
    if not chosen:
        return list(imageids)
    unknown = sorted(set(chosen) - set(imageids))
    if unknown:
        raise ValueError(f"Selected image IDs are absent from AnnData: {unknown}")
    return [imageid for imageid in imageids if imageid in set(chosen)]


def gate_values(adata: ad.AnnData, layer: str | None) -> np.ndarray:
    """The expression values the gates were drawn on: a layer, or ``.X``."""
    if layer:
        if layer not in adata.layers:
            raise KeyError(f"Gate layer {layer!r} is not in the AnnData (layers: {list(adata.layers)})")
        return dense_matrix(adata.layers[layer]).astype(float, copy=False)
    return dense_matrix(adata.X).astype(float, copy=False)


def dense_matrix(matrix) -> np.ndarray:
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def coordinate_columns(obs: pd.DataFrame) -> tuple[str, str]:
    for x_column, y_column in [
        ("X_centroid", "Y_centroid"),
        ("centroid-1", "centroid-0"),
        ("x", "y"),
    ]:
        if x_column in obs and y_column in obs:
            return x_column, y_column
    raise KeyError("No supported centroid columns found (expected X_centroid/Y_centroid or centroid-1/centroid-0)")


def plotting_labels(
    values: pd.Series, min_cells: int, rare_label: str = "Other/rare", max_labels: int = 19
) -> pd.Series:
    """Labels for a plot: rare ones, and any beyond the ``max_labels`` most
    common (the colour map has 20 colours), are grouped as *rare_label*."""
    labels = pd.Series(values, index=values.index, dtype="object").fillna("Unassigned").astype(str)
    counts = labels.value_counts()
    keep = set(counts[counts >= min_cells].index[:max_labels])
    return labels.where(labels.isin(keep), rare_label)


def save_spatial_scatter(
    adata: ad.AnnData,
    label: str,
    output_path: Path,
    title: str,
    min_cells: int,
) -> Path:
    x_column, y_column = coordinate_columns(adata.obs)
    labels = plotting_labels(adata.obs[label], min_cells)
    categories = labels.value_counts().index.tolist()
    cmap = plt.get_cmap("tab20", max(1, len(categories)))
    colors = {category: cmap(index) for index, category in enumerate(categories)}
    colors["Other/rare"] = (0.65, 0.67, 0.70, 1)
    imageids = [None]
    if "imageid" in adata.obs and adata.obs["imageid"].nunique() > 1:
        imageids = sorted(
            adata.obs["imageid"].dropna().astype(str).unique().tolist(),
            key=natural_key,
        )
    ncols = min(4, len(imageids))
    nrows = math.ceil(len(imageids) / ncols)
    # Dots sized to the cell count: visible on a small TMA core, not a solid
    # smear on a whole slide.
    cells_per_panel = adata.n_obs / len(imageids)
    point_size = float(np.clip(6000.0 / max(cells_per_panel, 1.0), 0.5, 12.0))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols + 2.0, 4.0 * nrows),
        dpi=170,
        squeeze=False,
    )
    for axis, imageid in zip(axes.ravel(), imageids):
        image_mask = np.ones(adata.n_obs, dtype=bool)
        if imageid is not None:
            image_mask = adata.obs["imageid"].astype(str).eq(imageid).to_numpy()
        for category in categories:
            mask = image_mask & labels.eq(category).to_numpy()
            axis.scatter(
                adata.obs.loc[mask, x_column],
                adata.obs.loc[mask, y_column],
                s=point_size,
                alpha=0.8,
                linewidths=0,
                color=colors[category],
                label=f"{category} ({labels.eq(category).sum():,})",
                rasterized=True,
            )
        axis.invert_yaxis()
        axis.set_aspect("equal")
        axis.set_xlabel(x_column)
        axis.set_ylabel(y_column)
        axis.set_title(str(imageid) if imageid is not None else title)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in axes.ravel()[len(imageids) :]:
        axis.axis("off")
    handles, legend_labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper left",
        bbox_to_anchor=(0.995, 0.98),
        frameon=False,
        markerscale=max(1.0, 4.0 / np.sqrt(point_size)),
        fontsize=7,
    )
    if len(imageids) > 1:
        fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_count_plot(summary: pd.DataFrame, label: str, output_path: Path, title: str) -> Path:
    plot = summary.sort_values("n_cells")
    colors = plot["review_class"].map(
        {"assigned": "#287D8E", "likely": "#D08A38", "rest": "#8C96A3", "unknown": "#B9BEC5"}
    )
    fig, axis = plt.subplots(figsize=(9, max(5, 0.30 * len(plot) + 1.5)), dpi=170)
    axis.barh(plot[label], plot["n_cells"], color=colors)
    axis.set_xlabel("Cells in selected subset")
    axis.set_title(title)
    axis.grid(axis="x", color="0.88", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_marker_heatmap(
    phenotyped: ad.AnnData,
    phenotype_label: str,
    workflow_markers: list[str],
    phenotype_summary: pd.DataFrame,
    png_path: Path,
    pdf_path: Path,
    values_path: Path,
    title: str,
) -> tuple[Path, Path, Path]:
    indexes = [phenotyped.var_names.get_loc(marker) for marker in workflow_markers]
    scaled = pd.DataFrame(
        dense_matrix(phenotyped.X)[:, indexes],
        index=phenotyped.obs_names,
        columns=workflow_markers,
    )
    positive_fraction = scaled.ge(0.5).groupby(
        phenotyped.obs[phenotype_label].astype(str), observed=True
    ).mean()
    count_lookup = phenotype_summary.set_index(phenotype_label)["n_cells"]
    positive_fraction = positive_fraction.reindex(phenotype_summary[phenotype_label])
    positive_fraction.index = [
        f"{phenotype} (n={int(count_lookup[phenotype]):,})" for phenotype in positive_fraction.index
    ]
    positive_fraction.to_csv(values_path)

    width = max(13, 0.40 * len(workflow_markers) + 5)
    height = max(7, 0.36 * len(positive_fraction) + 2.5)
    with plt.rc_context({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axis = plt.subplots(figsize=(width, height), dpi=180)
        heatmap = sns.heatmap(
            positive_fraction,
            cmap="RdBu_r",
            vmin=0,
            vmax=1,
            center=0.5,
            linewidths=0.30,
            linecolor=(1, 1, 1, 0.72),
            ax=axis,
            cbar_kws={"label": "Fraction marker-positive cells", "shrink": 0.72, "pad": 0.025},
            rasterized=True,
        )
        axis.set_xlabel("Marker")
        axis.set_ylabel("SCIMAP phenotype")
        axis.set_title(title, loc="left", pad=12, fontweight="bold")
        axis.tick_params(axis="x", labelrotation=45, labelsize=8.5, length=0)
        axis.tick_params(axis="y", labelrotation=0, labelsize=8.5, length=0)
        for tick in axis.get_xticklabels():
            tick.set_horizontalalignment("right")
        heatmap.collections[0].colorbar.ax.tick_params(labelsize=8, length=3)
        fig.tight_layout()
    fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path, values_path


def save_image_overlays(
    adata: ad.AnnData,
    phenotype_label: str,
    image_path: Path,
    output_dir: Path,
    stem: str,
    min_cells: int,
    crop_size: int,
    n_crops: int,
) -> list[Path]:
    x_column, y_column = coordinate_columns(adata.obs)
    marker_names = plane_marker_names(adata)
    windows, _ = mgq.choose_crop_windows(
        image_path,
        crop_size=crop_size,
        n_crops=n_crops,
        channel_markers=marker_names,
    )
    labels = plotting_labels(adata.obs[phenotype_label], min_cells)
    categories = labels.value_counts().index.tolist()
    cmap = plt.get_cmap("tab20", max(1, len(categories)))
    colors = {category: cmap(index) for index, category in enumerate(categories)}
    colors["Other/rare"] = (0.75, 0.75, 0.75, 1)
    channel_names = mgq.canonical_channel_names(image_path, marker_names)
    nuclear_index = next(
        (
            index
            for index, marker in enumerate(channel_names)
            if "HOECHST" in marker.upper()
            or marker.upper() in {"DAPI", "DNA_1"}
        ),
        0,
    )
    paths: list[Path] = []
    with tifffile.TiffFile(image_path) as tif:
        root = zarr.open(tif.series[0].levels[0].aszarr(), mode="r")
        image = zarr_array(root)
        for crop_number, (y_start, x_start) in enumerate(windows, start=1):
            nuclear = image[nuclear_index, y_start : y_start + crop_size, x_start : x_start + crop_size]
            background = mgq.normalize_image(nuclear, 0.5, 99.7, 0.8)
            rgb = np.zeros((*background.shape, 3), dtype=float)
            rgb[..., 2] = 0.85 * background
            rgb[..., 1] = 0.14 * background
            in_crop = (
                adata.obs[y_column].between(y_start, y_start + crop_size, inclusive="left")
                & adata.obs[x_column].between(x_start, x_start + crop_size, inclusive="left")
            )
            fig, axis = plt.subplots(figsize=(10.5, 9), dpi=180)
            axis.imshow(np.clip(rgb, 0, 1), interpolation="nearest")
            for category in categories:
                mask = in_crop & labels.eq(category)
                if not mask.any():
                    continue
                axis.scatter(
                    adata.obs.loc[mask, x_column] - x_start,
                    adata.obs.loc[mask, y_column] - y_start,
                    s=6,
                    alpha=0.78,
                    linewidths=0,
                    color=colors[category],
                    label=f"{category} ({mask.sum():,})",
                    rasterized=True,
                )
            axis.set_title(f"{stem} phenotypes on OME-TIFF, crop {crop_number}")
            axis.axis("off")
            axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, markerscale=2.5, fontsize=7)
            fig.tight_layout()
            path = output_dir / f"{stem}_scimap_image_overlay_crop{crop_number}.png"
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            paths.append(path)
    return paths


def apply_min_cells(labels: pd.Series, imageids: pd.Series, minimum: int) -> tuple[pd.Series, dict[str, int]]:
    """scimap's ``pheno_threshold_abs``, reporting what it changed.

    In each image, a phenotype with fewer than *minimum* cells is relabelled
    ``Unknown``. Returns the new labels and, per phenotype, how many cells were
    relabelled. ``minimum <= 0`` changes nothing.
    """
    labels = labels.astype(str)
    if minimum <= 0:
        return labels, {}
    sizes = labels.groupby([imageids.astype(str).to_numpy(), labels.to_numpy()]).transform("size")
    moved = sizes.lt(minimum) & labels.ne("Unknown")
    counts = labels[moved].value_counts().astype(int).to_dict()
    return labels.mask(moved, "Unknown"), counts


def run_workflow(config: dict, status_path: Path) -> dict:
    sample_id = str(config["sample_id"]).strip()
    selected = [str(value) for value in config["selected_populations"]]
    if not sample_id or not selected:
        raise ValueError("Sample ID and at least one selected population are required")
    output_dir = Path(config["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    qc_dir = output_dir / "qc"
    details_dir = output_dir / "details"
    qc_dir.mkdir(exist_ok=True)
    details_dir.mkdir(exist_ok=True)
    subset_name = str(config.get("subset_name") or "_".join(selected)).strip()
    subset_slug = slugify(subset_name)
    phenotype_label = str(config.get("phenotype_label", "scimap_phenotype")).strip()
    final_label = str(config.get("final_label", "final_hierarchical_phenotype")).strip()
    broad_column = str(config["broad_label_column"])
    gate_layer = str(config.get("gate_layer") or "").strip() or None
    missing_gate_policy = str(config.get("missing_gate_policy") or "error")
    if not phenotype_label or not final_label:
        raise ValueError("Phenotype and combined final column names cannot be empty")
    if len({broad_column, phenotype_label, final_label}) != 3:
        raise ValueError("Broad, SCIMAP phenotype, and combined final columns must have different names")
    min_cells = int(config.get("plot_min_cells", 50))
    warnings: list[str] = []

    update_status(status_path, "running", "Loading full-marker AnnData", stage="load", progress=0.05)
    full = ad.read_h5ad(config["gated_h5ad"])
    full = attach_annotations(full, config.get("broad_annotations"))
    if broad_column not in full.obs:
        raise KeyError(f"Broad label column {broad_column!r} is absent after joining annotations")
    if "imageid" not in full.obs:
        raise KeyError("Gated AnnData must contain an 'imageid' observation column for SCIMAP")
    full.obs["imageid"] = full.obs["imageid"].astype(str)
    available_populations = set(full.obs[broad_column].dropna().astype(str))
    missing_populations = sorted(set(selected) - available_populations)
    if missing_populations:
        raise ValueError(f"Selected populations absent from {broad_column!r}: {missing_populations}")
    all_imageids = sorted(full.obs["imageid"].dropna().unique().tolist(), key=natural_key)
    run_imageids = selected_images(config, all_imageids)
    mask = full.obs[broad_column].astype(str).isin(selected) & full.obs["imageid"].isin(run_imageids)
    subset = full[mask].copy()
    if subset.n_obs == 0:
        raise ValueError(f"No cells match {selected} in {broad_column!r} in the selected images")
    subset_imageids = sorted(subset.obs["imageid"].unique().tolist(), key=natural_key)

    update_status(status_path, "running", "Validating gates and phenotype workflow", stage="validate", progress=0.13)
    workflow, gates, workflow_summary = validate_workflow(
        Path(config["workflow_csv"]),
        Path(config["gate_csv"]),
        list(subset.var_names.astype(str)),
        imageids=subset_imageids,
        missing_gate_policy=missing_gate_policy,
    )
    workflow_markers = workflow_summary["workflow_markers"]

    update_status(status_path, "running", "Rescaling selected cells to the gates", stage="rescale", progress=0.23)
    from spatioev._vendor.scimap import phenotype_cells, rescale

    # Rescale only the workflow markers, on the values the gates were drawn
    # on. Rescaling every marker would fit a Gaussian mixture for each marker
    # without a gate, which is slow and can only add noise to the phenotypes.
    marker_index = {marker: index for index, marker in enumerate(subset.var_names.astype(str))}
    columns = [marker_index[marker] for marker in workflow_markers]
    raw_matrix = gate_values(subset, gate_layer)[:, columns]
    work = ad.AnnData(
        X=raw_matrix.astype(np.float32),
        obs=subset.obs.copy(),
        var=pd.DataFrame(index=pd.Index(workflow_markers, name="marker")),
    )
    gate_frame = gates.loc[workflow_markers].reset_index()
    rescaled = rescale(
        work,
        gate=gate_frame,
        log=True,
        imageid="imageid",
        method="all",
        random_state=0,
        verbose=False,
    )
    scaled_matrix = dense_matrix(rescaled.X).astype(float, copy=False)

    # Per-cell gate: the gate of the cell's own image.
    cell_images = subset.obs["imageid"].astype(str).to_numpy()
    agreement_rows = []
    for position, marker in enumerate(workflow_markers):
        gate_row = gates.loc[marker]
        if list(gates.columns) == ["gates"]:
            cell_gate = np.full(subset.n_obs, float(gate_row["gates"]))
        else:
            cell_gate = pd.Series(cell_images).map(gate_row.astype(float)).to_numpy(dtype=float)
        has_gate = np.isfinite(cell_gate)
        direct = np.log1p(np.clip(raw_matrix[:, position], 0, None)) >= np.where(has_gate, cell_gate, np.inf)
        scaled = scaled_matrix[:, position] >= 0.5
        row = {
            "marker": marker,
            "gate_min": float(np.nanmin(cell_gate)) if has_gate.any() else np.nan,
            "gate_max": float(np.nanmax(cell_gate)) if has_gate.any() else np.nan,
            "direct_positive_fraction": float(direct[has_gate].mean()) if has_gate.any() else np.nan,
            "rescaled_positive_fraction": float(scaled.mean()),
            "direct_to_rescaled_agreement": float(np.mean(direct[has_gate] == scaled[has_gate])) if has_gate.any() else np.nan,
        }
        stored_column = f"gate_{marker}_positive"
        if stored_column in subset.obs:
            stored = subset.obs[stored_column].to_numpy(dtype=bool)
            row["stored_positive_fraction"] = float(stored.mean())
            row["direct_to_stored_agreement"] = float(np.mean(direct == stored))
            if row["direct_to_stored_agreement"] < 1:
                warnings.append(
                    f"{marker}: the positive calls stored in the AnnData ({stored_column}) "
                    "differ from this gate table; the gate table is used."
                )
        agreement_rows.append(row)
    gate_agreement = pd.DataFrame(agreement_rows)

    update_status(status_path, "running", "Applying hierarchical SCIMAP phenotype rules", stage="phenotype", progress=0.38)
    rescaled = phenotype_cells(
        rescaled,
        phenotype=workflow,
        gate=float(config.get("phenotype_gate", 0.5)),
        label=phenotype_label,
        imageid="imageid",
        pheno_threshold_abs=None,
        verbose=False,
    )
    if phenotype_label not in rescaled.obs:
        raise RuntimeError("SCIMAP did not create the requested phenotype column")
    labels, below_minimum = apply_min_cells(
        rescaled.obs[phenotype_label].astype(str),
        rescaled.obs["imageid"].astype(str),
        int(config.get("pheno_threshold_abs", 10)),
    )
    rescaled.obs[phenotype_label] = labels
    if below_minimum:
        n_moved = sum(below_minimum.values())
        listed = ", ".join(f"{name} ({count})" for name, count in sorted(below_minimum.items(), key=lambda item: -item[1]))
        warnings.append(
            f"{n_moved} cells ({n_moved / rescaled.n_obs:.0%} of the subset) were set to Unknown because "
            f"their phenotype had fewer than {int(config.get('pheno_threshold_abs', 10))} cells in that image: "
            f"{listed}. Lower 'Minimum cells per phenotype per image' (0 turns it off) for small images."
        )

    stem = f"{sample_id}_{subset_slug}"
    full_stem = f"{sample_id}_broad_plus_{subset_slug}"
    # Top level: what is used next. qc/: figures and checks. details/: the
    # rescaled subset, for re-plotting or debugging.
    full_h5ad = output_dir / f"{full_stem}_phenotyped.h5ad"
    annotations_csv = output_dir / f"{full_stem}_annotations.csv"
    summary_csv = output_dir / f"{stem}_scimap_phenotype_counts.csv"
    subset_h5ad = details_dir / f"{stem}_scimap_rescaled.h5ad"
    agreement_csv = qc_dir / f"{stem}_gate_rescale_agreement.csv"
    count_png = qc_dir / f"{stem}_scimap_phenotype_counts.png"
    heatmap_png = qc_dir / f"{stem}_scimap_marker_heatmap.png"
    heatmap_pdf = qc_dir / f"{stem}_scimap_marker_heatmap.pdf"
    heatmap_values = qc_dir / f"{stem}_scimap_marker_positive_fractions.csv"
    subset_spatial = qc_dir / f"{stem}_scimap_spatial_scatter.png"
    full_spatial = qc_dir / f"{full_stem}_spatial_scatter.png"

    summary = rescaled.obs[phenotype_label].value_counts(dropna=False).rename_axis(phenotype_label).reset_index(name="n_cells")
    summary["fraction_of_subset"] = summary["n_cells"] / rescaled.n_obs
    summary["review_class"] = np.select(
        [
            summary[phenotype_label].str.startswith("likely-", na=False),
            summary[phenotype_label].str.contains("-rest", regex=False, na=False),
            summary[phenotype_label].str.lower().eq("unknown"),
        ],
        ["likely", "rest", "unknown"],
        default="assigned",
    )
    summary.to_csv(summary_csv, index=False)
    gate_agreement.to_csv(agreement_csv, index=False)

    update_status(status_path, "running", "Writing phenotyped AnnData and annotation tables", stage="write", progress=0.55)
    inputs = {
        "sample_id": sample_id,
        "broad_label_column": broad_column,
        "selected_populations": selected,
        "subset_name": subset_name,
        "imageids": subset_imageids,
        "gated_h5ad": str(Path(config["gated_h5ad"]).resolve()),
        "broad_annotations": str(config.get("broad_annotations") or ""),
        "manual_gates": str(Path(config["gate_csv"]).resolve()),
        "gate_layer": gate_layer,
        "gate_layout": workflow_summary["gate_layout"],
        "missing_gate_policy": missing_gate_policy,
        "phenotype_workflow": str(Path(config["workflow_csv"]).resolve()),
        "phenotype_gate": float(config.get("phenotype_gate", 0.5)),
        "pheno_threshold_abs": int(config.get("pheno_threshold_abs", 10)),
        "review_imageid": str(config.get("review_imageid") or ""),
    }
    rescaled.uns["scimap_phenotyping_inputs"] = inputs
    rescaled.uns["gates"] = gates.reset_index()
    from spatioev.workflows.cellsam import write_h5ad_atomically

    write_h5ad_atomically(rescaled, subset_h5ad)

    full.obs[phenotype_label] = pd.Series(pd.NA, index=full.obs_names, dtype="object")
    full.obs.loc[rescaled.obs_names, phenotype_label] = rescaled.obs[phenotype_label].astype(object)
    full.obs[final_label] = full.obs[broad_column].astype(object)
    full.obs.loc[rescaled.obs_names, final_label] = rescaled.obs[phenotype_label].astype(object)
    full.obs[[broad_column, phenotype_label, final_label]].to_csv(annotations_csv)
    if bool(config.get("write_full_h5ad", True)):
        full.uns["hierarchical_phenotyping_inputs"] = inputs
        write_h5ad_atomically(full, full_h5ad)

    update_status(status_path, "running", "Generating count, heatmap, and spatial QC", stage="qc", progress=0.72)
    save_count_plot(summary, phenotype_label, count_png, f"{sample_id} {subset_name} SCIMAP phenotype counts")
    save_marker_heatmap(
        rescaled,
        phenotype_label,
        workflow_markers,
        summary,
        heatmap_png,
        heatmap_pdf,
        heatmap_values,
        f"{sample_id} {subset_name} phenotype marker positivity",
    )
    save_spatial_scatter(rescaled, phenotype_label, subset_spatial, f"{sample_id} {subset_name} SCIMAP phenotypes", min_cells)
    save_spatial_scatter(full, final_label, full_spatial, f"{sample_id} broad tissue plus {subset_name} phenotypes", min_cells)

    overlay_paths: list[Path] = []
    with_image = imageids_with_images(config.get("image_path"), subset_imageids)
    review_imageid = str(
        config.get("review_imageid") or (with_image[0] if with_image else subset_imageids[0])
    )
    if review_imageid not in subset_imageids:
        raise ValueError(f"Review image ID {review_imageid!r} is absent from the selected subset")
    review_image = None
    if review_imageid in with_image:
        review_image = resolve_image(Path(config["image_path"]), review_imageid)
    elif bool(config.get("make_image_overlays", True)):
        warnings.append(
            f"Image overlays skipped: {review_imageid} has no image under "
            f"{config.get('image_path')}. Choose a FOV with an image to get them."
        )
    if review_image is not None and bool(config.get("make_image_overlays", True)):
        update_status(status_path, "running", "Generating original-image phenotype overlays", stage="overlay", progress=0.88)
        review_cells = subset.obs["imageid"].astype(str).eq(review_imageid).to_numpy()
        overlay_source = subset[review_cells].copy()
        overlay_source.obs[phenotype_label] = rescaled.obs.loc[overlay_source.obs_names, phenotype_label]
        overlay_paths = save_image_overlays(
            overlay_source,
            phenotype_label,
            review_image,
            qc_dir,
            stem,
            min_cells,
            int(config.get("overlay_crop_size", 1536)),
            int(config.get("overlay_n_crops", 3)),
        )

    outputs = {
        "sample_id": sample_id,
        "subset_name": subset_name,
        "selected_populations": selected,
        "broad_label_column": broad_column,
        "phenotype_label": phenotype_label,
        "final_label": final_label,
        "imageids": subset_imageids,
        "gate_layer": gate_layer,
        "n_full_cells": int(full.n_obs),
        "n_subset_cells": int(rescaled.n_obs),
        "n_phenotypes": int(rescaled.obs[phenotype_label].nunique()),
        "subset_h5ad": str(subset_h5ad),
        "full_h5ad": str(full_h5ad) if bool(config.get("write_full_h5ad", True)) else "",
        "annotations_csv": str(annotations_csv),
        "summary_csv": str(summary_csv),
        "gate_agreement_csv": str(agreement_csv),
        "count_png": str(count_png),
        "heatmap_png": str(heatmap_png),
        "heatmap_pdf": str(heatmap_pdf),
        "heatmap_values_csv": str(heatmap_values),
        "subset_spatial_png": str(subset_spatial),
        "full_spatial_png": str(full_spatial),
        "image_overlay_paths": [str(path) for path in overlay_paths],
        "review_imageid": review_imageid,
        "review_image_path": str(review_image) if review_image is not None else "",
        "workflow": workflow_summary,
        "warnings": warnings,
        "phenotyping_engine": "scimap (vendored rescale + phenotype_cells)",
    }
    manifest_path = output_dir / f"{stem}_scimap_phenotyping_manifest.json"
    outputs["manifest_path"] = str(manifest_path)
    write_json(manifest_path, {"created_at": now(), "config": config, "outputs": outputs})
    update_status(
        status_path,
        "complete",
        "SCIMAP phenotyping complete",
        stage="complete",
        progress=1.0,
        outputs=outputs,
    )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    try:
        if args.inspect_only:
            print(json.dumps(inspect_inputs(config), indent=2))
            return
        run_workflow(config, args.status)
    except Exception as error:
        update_status(
            args.status,
            "failed",
            str(error),
            stage="failed",
            progress=1.0,
            traceback=traceback.format_exc(),
        )
        raise


if __name__ == "__main__":
    main()
