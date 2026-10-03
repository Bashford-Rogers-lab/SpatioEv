"""Tumour / envelope / stroma tissue regions built from tumour-cell nests.

A thin, opinionated layer over the boundary functions in
:mod:`spatioev.tl.niche.boundaries`:

1. tumour cells are grouped into nests by spatial connected components;
2. each nest gets a boundary polygon;
3. the polygon is expanded by the envelope width;
4. every cell is labelled by where it sits: inside a nest (``tumour``), in
   the expanded ring (``envelope``) or anywhere else in the tissue
   (``stroma``).

The region describes a cell's *location*, not its phenotype: a tumour cell
lying alone in the stroma is in the ``stroma`` region.

Distances are in the coordinate units of ``X_centroid``/``Y_centroid``,
normally pixels; convert micrometres with the image pixel size.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from .boundaries import (
    add_niche_regions_to_obs,
    assign_cells_to_niche_regions,
    buffer_niche_boundaries,
    build_niche_boundaries,
    cluster_spatial_components,
)

__all__ = [
    "REGION_CODES",
    "define_tissue_regions",
    "median_cell_spacing",
    "read_region_boundaries",
    "rasterize_tissue_regions",
    "summarize_region_composition",
    "write_region_boundaries",
]

#: Integer codes used in region rasters; 0 is outside tissue.
REGION_CODES = {1: "tumour", 2: "envelope", 3: "stroma"}

# A cell inside one nest and inside another nest's envelope is tumour.
_PRIORITY = {"component": 0, "core": 0, "inner_border": 0, "outer_border": 1}
_TO_REGION = {"component": "tumour", "core": "tumour", "inner_border": "tumour", "outer_border": "envelope"}


def define_tissue_regions(
    adata: ad.AnnData,
    phenotype_key: str,
    tumour_labels: str | Sequence[str],
    envelope_width: float,
    image_key: str = "imageid",
    x_key: str = "X_centroid",
    y_key: str = "Y_centroid",
    region_key: str = "tissue_region",
    nest_key: str = "tumour_nest",
    min_nest_cells: int = 30,
    knn_k: int = 5,
    radius: float | None = None,
    radius_quantile: float = 0.9,
    radius_scale: float = 1.0,
    boundary_method: str = "density_mask",
    concavity: float = 0.3,
    mask_resolution: float | None = None,
    mask_sigma: float = 2.0,
    mask_threshold: float = 0.1,
    mask_closing_size: int = 5,
) -> tuple[ad.AnnData, pd.DataFrame]:
    """Label every cell as tumour, envelope or stroma.

    Parameters
    ----------
    phenotype_key : str
        ``adata.obs`` column holding phenotypes.
    tumour_labels : str or list of str
        Phenotype value(s) that define tumour nests.
    envelope_width : float
        Width of the envelope ring outside each nest, in coordinate units.
    min_nest_cells : int
        Tumour components smaller than this do not form a nest.
    knn_k, radius, radius_quantile, radius_scale
        Control which tumour cells are connected. With ``radius=None`` the
        radius is the ``radius_quantile`` of each image's distance to the
        ``knn_k``-th nearest tumour cell, times ``radius_scale``. ``knn_k=1``
        (the lower-level default) shatters loosely packed nests into
        singletons, hence 5 here.
    boundary_method, concavity, mask_*
        Passed to :func:`~spatioev.tl.niche.build_niche_boundaries`.
        ``mask_resolution`` is the boundary grid spacing in coordinate units;
        when omitted it is half the median nearest-neighbour distance between
        cells, which keeps nests solid whether coordinates are pixels or
        micrometres. A grid much finer than the cell spacing outlines every
        small cluster separately instead of the nest.

    Returns
    -------
    adata : AnnData
        Copy with ``obs[region_key]`` (categorical: tumour / envelope /
        stroma) and ``obs[nest_key]`` (nest id, or NA outside nests and
        envelopes).
    boundaries : DataFrame
        One row per nest with ``geometry`` and ``expanded_geometry``.
    """
    if isinstance(tumour_labels, str):
        tumour_labels = [tumour_labels]
    tumour_labels = [str(label) for label in tumour_labels]
    work = adata.copy()
    work.obs[phenotype_key] = work.obs[phenotype_key].astype(str)

    out = adata.copy()

    def all_stroma():
        out.obs[region_key] = pd.Categorical(["stroma"] * out.n_obs, categories=list(REGION_CODES.values()))
        out.obs[nest_key] = pd.NA
        return out, pd.DataFrame(columns=[nest_key, image_key, "n_cells", "geometry", "expanded_geometry"])

    if not work.obs[phenotype_key].isin(tumour_labels).any():
        # e.g. a TMA core with no tumour; the component search cannot run.
        return all_stroma()
    if mask_resolution is None:
        mask_resolution = 0.5 * median_cell_spacing(work.obs, image_key, x_key, y_key)

    component_key = f"{nest_key}__component"
    work = cluster_spatial_components(
        work,
        label_key=phenotype_key,
        label_value=tumour_labels,
        image_key=image_key,
        x_key=x_key,
        y_key=y_key,
        component_key=component_key,
        target_key=f"{nest_key}__is_tumour",
        radius=radius,
        knn_k=knn_k,
        radius_quantile=radius_quantile,
        radius_scale=radius_scale,
        min_component_size=min_nest_cells,
        assign_singletons=False,
    )
    boundaries = build_niche_boundaries(
        work,
        component_key=component_key,
        image_key=image_key,
        x_key=x_key,
        y_key=y_key,
        min_cluster_size=min_nest_cells,
        method=boundary_method,
        concavity=concavity,
        mask_resolution=mask_resolution,
        mask_sigma=mask_sigma,
        mask_threshold=mask_threshold,
        mask_closing_size=mask_closing_size,
    )

    if boundaries.empty:
        return all_stroma()

    boundaries = buffer_niche_boundaries(boundaries, component_key, expand_by=float(envelope_width))
    assignments = assign_cells_to_niche_regions(
        work,
        boundaries,
        component_key=component_key,
        image_key=image_key,
        x_key=x_key,
        y_key=y_key,
        region_key="__region",
        mode="buffer",
    )
    labelled = add_niche_regions_to_obs(
        work,
        assignments,
        region_key="__region",
        component_key=component_key,
        out_region_key="__region",
        out_component_key="__nest",
        outside_label="outside",
        region_priority=_PRIORITY,
    )
    region = labelled.obs["__region"].map(_TO_REGION).fillna("stroma")
    out.obs[region_key] = pd.Categorical(region.reindex(out.obs_names), categories=list(REGION_CODES.values()))
    out.obs[nest_key] = labelled.obs["__nest"].reindex(out.obs_names)

    boundaries = boundaries.rename(columns={component_key: nest_key})
    boundaries["envelope_width"] = float(envelope_width)
    return out, boundaries.reset_index(drop=True)


def median_cell_spacing(obs: pd.DataFrame, image_key: str = "imageid", x_key: str = "X_centroid", y_key: str = "Y_centroid") -> float:
    """Median distance from each cell to its nearest neighbour, pooled over images."""
    from scipy.spatial import cKDTree

    distances = []
    for _, cells in obs.groupby(image_key, observed=True):
        coords = cells[[x_key, y_key]].dropna().to_numpy(float)
        if len(coords) > 1:
            distances.append(cKDTree(coords).query(coords, k=2)[0][:, 1])
    if not distances:
        return 1.0
    return float(np.median(np.concatenate(distances)))


def summarize_region_composition(
    adata: ad.AnnData,
    phenotype_key: str,
    region_key: str = "tissue_region",
    image_key: str = "imageid",
    include_all: bool = True,
) -> pd.DataFrame:
    """Phenotype counts and proportions per image and region.

    Unlike :func:`~spatioev.tl.niche.summarize_niche_composition`, this covers
    every cell, stroma included, and is aggregated per region rather than per
    nest. With ``include_all`` an ``all`` region gives the whole-image
    composition. Proportions are within each image x region.
    """
    obs = adata.obs[[image_key, region_key, phenotype_key]].copy()
    obs[region_key] = obs[region_key].astype(str)
    obs[phenotype_key] = obs[phenotype_key].astype(str)
    frames = [obs]
    if include_all:
        frames.append(obs.assign(**{region_key: "all"}))
    stacked = pd.concat(frames, ignore_index=True)
    counts = stacked.groupby([image_key, region_key, phenotype_key]).size().rename("count").reset_index()
    phenotypes = sorted(obs[phenotype_key].unique())
    full = (
        counts.set_index([image_key, region_key, phenotype_key])["count"]
        .unstack(fill_value=0)
        .reindex(columns=phenotypes, fill_value=0)
        .stack()
        .rename("count")
        .reset_index()
    )
    full["region_total"] = full.groupby([image_key, region_key])["count"].transform("sum")
    full["proportion"] = np.where(full["region_total"] > 0, full["count"] / full["region_total"], np.nan)
    return full


def _fill_polygon(raster, polygon, value, downsample, only_where=None):
    from skimage.draw import polygon as draw_polygon

    def rows_cols(ring):
        coords = np.asarray(ring.coords)
        return coords[:, 1] / downsample, coords[:, 0] / downsample

    rr, cc = draw_polygon(*rows_cols(polygon.exterior), shape=raster.shape)
    mask = np.zeros(raster.shape, dtype=bool)
    mask[rr, cc] = True
    for interior in polygon.interiors:
        rr, cc = draw_polygon(*rows_cols(interior), shape=raster.shape)
        mask[rr, cc] = False
    if only_where is not None:
        mask &= only_where
    raster[mask] = value


def _polygons(geometry):
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    return [geom for geom in getattr(geometry, "geoms", []) if geom.geom_type == "Polygon"]


def rasterize_tissue_regions(
    boundaries: pd.DataFrame,
    obs: pd.DataFrame,
    image_shape: tuple[int, int],
    imageid: str | None = None,
    tissue_radius: float = 100.0,
    downsample: int = 4,
    image_key: str = "imageid",
    x_key: str = "X_centroid",
    y_key: str = "Y_centroid",
) -> np.ndarray:
    """Paint tumour / envelope / stroma onto a pixel grid for one image.

    Tissue is every pixel within ``tissue_radius`` of a cell. Nests are
    tumour (code 1), the expanded ring is envelope (2) where it lies in
    tissue, the rest of the tissue is stroma (3), and 0 is outside tissue.
    The grid is ``image_shape`` divided by ``downsample`` (rounded up).

    Used to measure matrix inside each region with
    :func:`spatioev.tl.ecm.fiber_region_stats`.
    """
    import scipy.ndimage as ndi

    ds = max(1, int(downsample))
    grid = (int(np.ceil(image_shape[0] / ds)), int(np.ceil(image_shape[1] / ds)))
    cells = obs if imageid is None else obs[obs[image_key].astype(str) == str(imageid)]
    seeds = np.ones(grid, dtype=bool)
    rows = np.clip((cells[y_key].to_numpy(float) // ds).astype(int), 0, grid[0] - 1)
    cols = np.clip((cells[x_key].to_numpy(float) // ds).astype(int), 0, grid[1] - 1)
    seeds[rows, cols] = False
    tissue = ndi.distance_transform_edt(seeds) * ds <= tissue_radius

    raster = np.zeros(grid, dtype=np.uint8)
    raster[tissue] = 3
    nests = boundaries if imageid is None else boundaries[boundaries[image_key].astype(str) == str(imageid)]
    for geometry in nests.get("expanded_geometry", pd.Series(dtype=object)):
        for polygon in _polygons(geometry):
            _fill_polygon(raster, polygon, 2, ds, only_where=tissue)
    for geometry in nests.get("geometry", pd.Series(dtype=object)):
        for polygon in _polygons(geometry):
            _fill_polygon(raster, polygon, 1, ds)
    return raster


def write_region_boundaries(boundaries: pd.DataFrame, path: str | Path, nest_key: str = "tumour_nest") -> Path:
    """Write nest and envelope polygons as GeoJSON (pixel coordinates)."""
    from shapely.geometry import mapping

    features = []
    for _, row in boundaries.iterrows():
        properties = {
            key: (None if pd.isna(value) else value.item() if hasattr(value, "item") else value)
            for key, value in row.items()
            if key not in {"geometry", "expanded_geometry", "shrunk_geometry", "bounds"}
        }
        for kind in ("geometry", "expanded_geometry"):
            geometry = row.get(kind)
            if geometry is None or geometry.is_empty:
                continue
            features.append({
                "type": "Feature",
                "geometry": mapping(geometry),
                "properties": {**properties, "kind": "nest" if kind == "geometry" else "envelope_outer"},
            })
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")
    temporary.replace(path)
    return path


def read_region_boundaries(path: str | Path, nest_key: str = "tumour_nest", image_key: str = "imageid") -> pd.DataFrame:
    """Read polygons written by :func:`write_region_boundaries`."""
    from shapely.geometry import shape

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows: dict[tuple, dict] = {}
    for feature in payload.get("features", []):
        properties = dict(feature.get("properties", {}))
        kind = properties.pop("kind", "nest")
        key = (str(properties.get(image_key)), str(properties.get(nest_key)))
        row = rows.setdefault(key, dict(properties))
        row["geometry" if kind == "nest" else "expanded_geometry"] = shape(feature["geometry"])
    return pd.DataFrame(list(rows.values()))
