"""Bridge between SpatioEv's segmentation and QuPath, keyed on ``cell_id``.

The round trip is one import and one export per core:

1. :func:`write_qupath_cells` turns the CellSAM masks and cell table of a core
   into a QuPath GeoJSON: one cell object per segmented cell, with its exact
   pixel-edge outline, its matched nucleus, ``cell_id`` as the object name, a
   stable object ID derived from ``cell_id`` and the marker measurements.
   QuPath imports it with ``spatioev_import_cells.groovy``.
2. After classifying in QuPath, ``spatioev_export_classes.groovy`` writes
   ``cell_id -> class`` and any annotations; :func:`read_qupath_classes`,
   :func:`read_qupath_annotations` and :func:`qupath_phenotyped_anndata`
   turn them into the phenotyped AnnData the spatial stages use.

Cells are never matched by position. Coordinates are pixels on the core's
OME-TIFF grid; QuPath measures outlines from pixel corners, so its centroids
sit exactly 0.5 px right of and below the table's pixel-centre centroids.

Expected core layout (as written by the SpatioEv preprocessing pipeline)::

    <core>/background/<core>.ome.tiff
    <core>/segmentation/<core>_whole_cell.tiff, <core>_nuclear.tiff
    <core>/quantification/cell_table_<kind>.csv
    <core>/qupath/            bridge files, created here
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "BRIDGE_DIRNAME",
    "QUPATH_SCRIPTS",
    "cell_uuid",
    "core_status",
    "find_cores",
    "label_polygons",
    "qupath_phenotyped_anndata",
    "read_qupath_annotations",
    "read_qupath_classes",
    "split_qupath_class",
    "write_qupath_cells",
    "write_qupath_scripts",
]

BRIDGE_DIRNAME = "qupath"
#: Groovy scripts shipped in ``spatioev/resources/qupath``.
QUPATH_SCRIPTS = ("spatioev_import_cells.groovy", "spatioev_export_classes.groovy")
#: Annotation classes whose cells are dropped from the analysis (case-insensitive prefix).
EXCLUDE_CLASSES = ("exclude", "ignore", "artefact", "artifact")

# Fixed namespace so a cell's QuPath object ID is the same on every import.
_UUID_NAMESPACE = uuid.UUID("6f1c3c1e-6c55-4a8e-9c2b-5d1f0b7a9e21")
_TABLE_COLUMNS = {
    "label", "centroid_x", "centroid_y", "eccentricity", "major_axis_length_px", "minor_axis_length_px",
    "perimeter_px", "solidity", "cell_size", "cell_area_px2", "passes_size_qc", "matched_nuclear_label",
    "nuclear_size", "nuclear_to_cell_area_ratio", "sample", "mask_type", "cell_id",
}


def cell_uuid(cell_id: str) -> str:
    """Deterministic QuPath object ID for a cell."""
    return str(uuid.uuid5(_UUID_NAMESPACE, str(cell_id)))


# --------------------------------------------------------------------------- #
# Core folders
# --------------------------------------------------------------------------- #
def _is_core(path: Path) -> bool:
    return (path / "segmentation").is_dir() and (path / "quantification").is_dir()


def find_cores(roots: str | Path | Iterable[str | Path]) -> list[Path]:
    """Core folders under ``roots`` (a core folder itself, or folders of cores)."""
    if isinstance(roots, (str, Path)):
        roots = [roots]
    found: dict[Path, None] = {}
    for root in roots:
        root = Path(root).expanduser().resolve()
        if _is_core(root):
            found[root] = None
            continue
        for child in sorted(root.iterdir()) if root.is_dir() else []:
            if child.is_dir() and not child.name.startswith(".") and _is_core(child):
                found[child] = None
    return list(found)


def _core_files(core: Path, table: str = "arcsinh_transformed") -> dict[str, Path]:
    name = core.name
    bridge = core / BRIDGE_DIRNAME
    return {
        "image": core / "background" / f"{name}.ome.tiff",
        "cell_mask": core / "segmentation" / f"{name}_whole_cell.tiff",
        "nuclear_mask": core / "segmentation" / f"{name}_nuclear.tiff",
        "table": core / "quantification" / f"cell_table_{table}.csv",
        "cells_geojson": bridge / f"{name}_cells.geojson",
        "classes": bridge / f"{name}_cell_classes.csv",
        "annotations": bridge / f"{name}_annotations.geojson",
        "phenotyped": bridge / f"{name}_phenotyped.h5ad",
    }


def core_status(core: str | Path, table: str = "arcsinh_transformed") -> dict:
    """Which inputs and bridge files exist for a core."""
    core = Path(core)
    paths = _core_files(core, table)
    status = {"core": core.name, "path": str(core)}
    status.update({key: path.exists() for key, path in paths.items()})
    return status


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def _runs_polygon(binary: np.ndarray, row0: int, col0: int):
    """Exact outline of a binary crop as the union of its pixel runs."""
    import shapely

    padded = np.pad(binary.astype(np.int8), ((0, 0), (1, 1)))
    step = np.diff(padded, axis=1)
    starts = np.argwhere(step == 1)
    ends = np.argwhere(step == -1)
    boxes = shapely.box(col0 + starts[:, 1], row0 + starts[:, 0], col0 + ends[:, 1], row0 + ends[:, 0] + 1)
    return shapely.union_all(boxes).simplify(0)


def label_polygons(mask: np.ndarray, labels: Sequence[int] | None = None) -> dict[int, object]:
    """Pixel-edge outline of every label in ``mask``.

    Vertices sit on pixel corners (pixel ``(r, c)`` spans ``[c, c+1) x [r, r+1)``),
    QuPath's convention, so a polygon's area equals the label's pixel count.
    """
    import scipy.ndimage as ndi

    mask = np.asarray(mask)
    wanted = None if labels is None else {int(v) for v in labels}
    out = {}
    for index, box in enumerate(ndi.find_objects(mask), start=1):
        if box is None or (wanted is not None and index not in wanted):
            continue
        out[index] = _runs_polygon(mask[box] == index, box[0].start, box[1].start)
    return out


def _polygonal(geometry):
    """Polygon or MultiPolygon parts only; clipping can leave lines QuPath rejects."""
    import shapely

    parts = [g for g in shapely.get_parts(geometry) if g.geom_type == "Polygon" and g.area > 0]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else shapely.MultiPolygon(parts)


def _largest_polygon(geometry):
    import shapely

    parts = [g for g in shapely.get_parts(geometry) if g.geom_type == "Polygon" and g.area > 0]
    return max(parts, key=lambda g: g.area) if parts else None


# --------------------------------------------------------------------------- #
# SpatioEv -> QuPath
# --------------------------------------------------------------------------- #
def _markers(table: pd.DataFrame) -> list[str]:
    return [c for c in table.columns if c not in _TABLE_COLUMNS and not c.endswith("_nuclear")]


def _sample_name(core: Path, table: pd.DataFrame) -> str:
    if "sample" in table and table["sample"].nunique() == 1:
        return str(table["sample"].iloc[0])
    return core.name


def write_qupath_cells(
    core: str | Path,
    out_path: str | Path | None = None,
    table: str = "arcsinh_transformed",
    include_nuclear: bool = True,
    decimals: int = 4,
    pixel_size_um: float | None = None,
) -> dict:
    """Write a core's cells as QuPath GeoJSON.

    Parameters
    ----------
    core : path
        Core folder (see the module docstring for the layout).
    out_path : path, optional
        Defaults to ``<core>/qupath/<core>_cells.geojson``.
    table : str
        Which ``cell_table_<table>.csv`` supplies the measurements:
        ``arcsinh_transformed`` (default, what the original import script
        used), ``size_normalized`` or ``raw``.
    include_nuclear : bool
        Also add each marker's nuclear mean as ``"<marker> nucleus"``.
    decimals : int
        Rounding of measurements; keeps files small without affecting
        classification.
    pixel_size_um : float, optional
        For the ``Area µm^2`` measurement; read from the OME-TIFF if omitted.

    Returns
    -------
    dict with the output path and counts (cells, nuclei attached, cells
    without a nucleus, nuclei shared by two cells).

    Measurement names: the bare marker name is the whole-cell mean (as in the
    original ``import_quan.groovy``), so existing QuPath classifiers keep
    working.
    """
    import shapely
    import tifffile

    core = Path(core).expanduser().resolve()
    paths = _core_files(core, table)
    for key in ("cell_mask", "table"):
        if not paths[key].exists():
            raise FileNotFoundError(f"{core.name}: missing {paths[key]}")
    cells_table = pd.read_csv(paths["table"])
    sample = _sample_name(core, cells_table)
    markers = _markers(cells_table)
    if pixel_size_um is None and paths["image"].exists():
        from spatioev.workflows.fiber_segmentation import physical_pixel_size

        pixel_size_um = physical_pixel_size(paths["image"])

    cell_mask = tifffile.imread(paths["cell_mask"])
    cell_geoms = label_polygons(cell_mask, cells_table["label"].astype(int))
    nucleus_geoms = {}
    matched = cells_table.get("matched_nuclear_label")
    if paths["nuclear_mask"].exists() and matched is not None:
        nuclear_labels = matched.fillna(0).astype(int)
        nucleus_geoms = label_polygons(tifffile.imread(paths["nuclear_mask"]), nuclear_labels[nuclear_labels > 0])

    features, attached, missing = [], 0, 0
    for row in cells_table.to_dict("records"):
        label = int(row["label"])
        geometry = cell_geoms.get(label)
        geometry = _polygonal(geometry) if geometry is not None else None
        if geometry is None:
            continue
        cell_id = f"{sample}_{label}"
        measurements = {marker: round(float(row[marker]), decimals) for marker in markers if pd.notna(row[marker])}
        if include_nuclear:
            measurements.update({
                f"{marker} nucleus": round(float(row[f"{marker}_nuclear"]), decimals)
                for marker in markers if f"{marker}_nuclear" in row and pd.notna(row[f"{marker}_nuclear"])
            })
        if "cell_area_px2" in row:
            measurements["Area px^2"] = float(row["cell_area_px2"])
            if pixel_size_um:
                measurements["Area µm^2"] = round(float(row["cell_area_px2"]) * pixel_size_um**2, decimals)
        for column, name in (("eccentricity", "Eccentricity"), ("solidity", "Solidity"),
                             ("nuclear_to_cell_area_ratio", "Nucleus/Cell area ratio"), ("passes_size_qc", "Passes size QC")):
            if column in row and pd.notna(row[column]):
                measurements[name] = round(float(row[column]), decimals)
        measurements["Label"] = label
        feature = {
            "type": "Feature",
            "id": cell_uuid(cell_id),
            "geometry": shapely.geometry.mapping(geometry),
            "properties": {"objectType": "cell", "name": cell_id, "measurements": measurements},
        }
        nucleus_label = int(row.get("matched_nuclear_label") or 0) if pd.notna(row.get("matched_nuclear_label")) else 0
        nucleus = nucleus_geoms.get(nucleus_label) if nucleus_label > 0 else None
        if nucleus is not None:
            nucleus = nucleus if geometry.contains(nucleus) else _largest_polygon(nucleus.intersection(geometry))
        if nucleus is not None:
            feature["nucleusGeometry"] = shapely.geometry.mapping(nucleus)
            attached += 1
        else:
            missing += 1
        features.append(feature)

    out_path = Path(out_path) if out_path else paths["cells_geojson"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump({"type": "FeatureCollection", "features": features}, handle, separators=(",", ":"))
    temporary.replace(out_path)
    shared = int(matched[matched.fillna(0) > 0].duplicated().sum()) if matched is not None else 0
    return {
        "core": core.name,
        "sample": sample,
        "path": str(out_path),
        "cells": len(features),
        "nuclei_attached": attached,
        "cells_without_nucleus": missing,
        "nuclei_shared_by_two_cells": shared,
        "table": table,
        "markers": markers,
    }


def write_qupath_scripts(destination: str | Path) -> list[Path]:
    """Copy the SpatioEv QuPath scripts into ``destination`` (e.g. QuPath's scripts folder)."""
    destination = Path(destination).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    written = []
    for name in QUPATH_SCRIPTS:
        target = destination / name
        target.write_text(files("spatioev.resources").joinpath("qupath", name).read_text(encoding="utf-8"), encoding="utf-8")
        written.append(target)
    return written


# --------------------------------------------------------------------------- #
# QuPath -> SpatioEv
# --------------------------------------------------------------------------- #
def split_qupath_class(name: str | None) -> tuple[str, str]:
    """``"Fibroblast: WNT5A+"`` -> ``("Fibroblast", "WNT5A+")``; empty -> ``("Unclassified", "")``."""
    if name is None or (isinstance(name, float) and np.isnan(name)) or not str(name).strip():
        return "Unclassified", ""
    parts = [part.strip() for part in str(name).split(":")]
    return parts[0] or "Unclassified", ": ".join(parts[1:])


def read_qupath_classes(path: str | Path) -> pd.DataFrame:
    """Read a ``*_cell_classes.csv`` written by ``spatioev_export_classes.groovy``.

    Rows without a ``cell_id`` (objects created in QuPath rather than
    imported) are dropped; duplicated IDs raise, since they would mean the
    cells were imported twice.
    """
    classes = pd.read_csv(path, dtype={"cell_id": str, "qupath_class": str})
    classes = classes[classes["cell_id"].notna() & (classes["cell_id"].str.len() > 0)].copy()
    duplicated = classes["cell_id"][classes["cell_id"].duplicated()].unique()
    if len(duplicated):
        raise ValueError(
            f"{Path(path).name}: {len(duplicated)} cell IDs appear more than once (e.g. {duplicated[0]}). "
            "The cells were probably imported into QuPath twice."
        )
    classes["qupath_class"] = classes["qupath_class"].fillna("Unclassified").replace("", "Unclassified")
    split = classes["qupath_class"].map(split_qupath_class)
    classes["phenotype"] = [base for base, _ in split]
    classes["qupath_subclass"] = [sub for _, sub in split]
    return classes.set_index("cell_id", drop=False)


def read_qupath_annotations(path: str | Path) -> pd.DataFrame:
    """Annotations exported from QuPath, one row per object, pixel coordinates."""
    from shapely.geometry import shape

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    features = payload.get("features", []) if isinstance(payload, dict) else payload
    rows = []
    for feature in features:
        properties = feature.get("properties", {}) or {}
        classification = properties.get("classification") or {}
        name = classification.get("name") if isinstance(classification, dict) else classification
        geometry = feature.get("geometry")
        if not geometry:
            continue
        rows.append({
            "name": properties.get("name"),
            "classification": name or "Unclassified",
            "geometry": shape(geometry),
        })
    table = pd.DataFrame(rows, columns=["name", "classification", "geometry"])
    if len(table):
        table["area_px"] = [g.area for g in table["geometry"]]
    return table


def _annotation_labels(obs: pd.DataFrame, annotations: pd.DataFrame, exclude: Sequence[str]) -> tuple[pd.Series, pd.Series]:
    """Per cell: excluded by an exclusion annotation, and the smallest other annotation holding it."""
    import shapely

    # Table centroids are pixel centres; annotation vertices are pixel corners.
    x = obs["X_centroid"].to_numpy(float) + 0.5
    y = obs["Y_centroid"].to_numpy(float) + 0.5
    excluded = np.zeros(len(obs), dtype=bool)
    region = np.full(len(obs), None, dtype=object)
    best_area = np.full(len(obs), np.inf)
    for annotation in annotations.itertuples(index=False):
        inside = shapely.contains_xy(annotation.geometry, x, y)
        if not inside.any():
            continue
        if str(annotation.classification).lower().startswith(tuple(exclude)):
            excluded |= inside
        elif annotation.classification != "Unclassified":
            better = inside & (annotation.area_px < best_area)
            region[better] = annotation.classification
            best_area[better] = annotation.area_px
    return pd.Series(excluded, index=obs.index), pd.Series(region, index=obs.index)


def qupath_phenotyped_anndata(
    core: str | Path,
    table: str = "arcsinh_transformed",
    drop_excluded: bool = True,
    exclude_classes: Sequence[str] = EXCLUDE_CLASSES,
):
    """Build the phenotyped AnnData for one core from its QuPath export.

    ``X`` holds the whole-cell marker means from ``cell_table_<table>.csv``
    (nuclear means in ``layers["nuclear"]``). ``obs`` is indexed by
    ``cell_id`` and carries ``imageid`` (the core), pixel centroids,
    morphology, ``qupath_class`` (QuPath's full class), ``phenotype`` (its
    base class) and ``qupath_subclass``. With an annotation export, cells in
    "Exclude"/"Ignore"/"Artefact" annotations are dropped (or flagged with
    ``drop_excluded=False``) and ``qupath_annotation`` records the smallest
    other classified annotation containing each cell.
    """
    import anndata as ad

    core = Path(core).expanduser().resolve()
    paths = _core_files(core, table)
    if not paths["classes"].exists():
        raise FileNotFoundError(f"{core.name}: no QuPath export at {paths['classes']}; run spatioev_export_classes.groovy")
    cells_table = pd.read_csv(paths["table"])
    sample = _sample_name(core, cells_table)
    markers = _markers(cells_table)
    cells_table["cell_id"] = sample + "_" + cells_table["label"].astype(int).astype(str)
    cells_table = cells_table.set_index("cell_id", drop=False)

    classes = read_qupath_classes(paths["classes"])
    unknown = classes.index.difference(cells_table.index)
    if len(unknown):
        raise ValueError(f"{core.name}: {len(unknown)} QuPath cell IDs are not in the cell table (e.g. {unknown[0]})")
    obs = pd.DataFrame(index=cells_table.index)
    obs["cell_id"] = cells_table["cell_id"]
    obs["imageid"] = sample
    obs["sample"] = sample
    obs["label"] = cells_table["label"].astype(int)
    obs["X_centroid"] = cells_table["centroid_x"].astype(float)
    obs["Y_centroid"] = cells_table["centroid_y"].astype(float)
    for column in ("cell_area_px2", "eccentricity", "solidity", "nuclear_to_cell_area_ratio", "matched_nuclear_label"):
        if column in cells_table:
            obs[column] = cells_table[column].to_numpy()
    obs["qupath_class"] = classes["qupath_class"].reindex(obs.index).fillna("Not in QuPath export")
    obs["phenotype"] = classes["phenotype"].reindex(obs.index).fillna("Not in QuPath export")
    obs["qupath_subclass"] = classes["qupath_subclass"].reindex(obs.index).fillna("")

    n_excluded = 0
    if paths["annotations"].exists():
        annotations = read_qupath_annotations(paths["annotations"])
        if len(annotations):
            excluded, region = _annotation_labels(obs, annotations, [c.lower() for c in exclude_classes])
            obs["excluded_by_annotation"] = excluded
            obs["qupath_annotation"] = region.fillna("none").astype(str)
            n_excluded = int(excluded.sum())

    X = cells_table[markers].to_numpy(dtype=np.float32)
    nuclear = [f"{m}_nuclear" for m in markers]
    layers = {"nuclear": cells_table[nuclear].to_numpy(dtype=np.float32)} if all(c in cells_table for c in nuclear) else {}
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=markers), layers=layers)
    for column in ("qupath_class", "phenotype", "qupath_subclass", "imageid", "sample"):
        adata.obs[column] = adata.obs[column].astype("category")
    if drop_excluded and "excluded_by_annotation" in adata.obs:
        adata = adata[~adata.obs["excluded_by_annotation"].to_numpy()].copy()
    adata.uns["qupath_bridge"] = {
        "core": core.name,
        "classes_file": str(paths["classes"]),
        "annotations_file": str(paths["annotations"]) if paths["annotations"].exists() else "",
        "table": table,
        "n_cells_table": int(len(cells_table)),
        "n_cells_classified": int(len(classes)),
        "n_excluded": n_excluded,
        "excluded_dropped": bool(drop_excluded),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    return adata

