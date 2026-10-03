"""Fiber quantity and architecture summaries per image, tile and tissue region.

The density and tile summaries are adapted from ark-analysis
``ark.segmentation.fiber_segmentation`` (MIT License, Copyright (c) 2023
Angelo Lab; see ``spatioev/pp/LICENSE.ark-analysis``):

- :func:`calculate_fiber_density` is ark's ``calculate_density``.
- :func:`fiber_tile_stats` follows ark's ``generate_tile_stats``, including
  its rule that a tile with fewer than ``min_fiber_num`` fibers gets NaN
  statistics. Unlike ark it accepts any image size and tile size, keeps the
  partial tiles at the right and bottom edges, and divides by each tile's real
  area.
- :func:`fiber_image_stats` is ark's per-FOV summary, using the true image
  area instead of assuming a square image.

Added here: ``area_fraction_pct`` (fiber pixels per tissue pixel, defined even
where ark's density is NaN), axial orientation coherence, per-region
summaries for tumour / envelope / stroma, and a tile-level association between
matrix metrics and cell phenotypes.

All coordinates are pixels: ``X_centroid`` is the column, ``Y_centroid`` the
row, as written by :func:`spatioev.pp.fibers.fiber_table_from_labels`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "FIBER_SUMMARY_PROPERTIES",
    "calculate_fiber_density",
    "cell_matrix_enrichment",
    "cell_matrix_proximity",
    "fiber_image_stats",
    "fiber_region_stats",
    "fiber_tile_stats",
    "orientation_coherence",
    "tile_cell_matrix_association",
]

#: Per-fiber properties averaged in every summary (ark's list, minus the raw
#: orientation, whose arithmetic mean is meaningless for axial angles).
FIBER_SUMMARY_PROPERTIES = (
    "major_axis_length",
    "minor_axis_length",
    "area",
    "eccentricity",
    "euler_number",
)


def calculate_fiber_density(fiber_df: pd.DataFrame, total_pixels: float) -> tuple[float, float]:
    """Pixel-area and fiber-count density, both multiplied by 100 (ark).

    ``pixel_density = 100 * sum(area) / total_pixels`` and
    ``fiber_density = 100 * n_fibers / total_pixels``.
    """
    fiber_num = len(np.unique(fiber_df["label"]))
    pixel_sum = float(np.sum(fiber_df["area"].to_numpy()))
    return pixel_sum / total_pixels * 100, fiber_num / total_pixels * 100


def orientation_coherence(orientation_deg) -> tuple[float, float]:
    """Axial circular statistics of fiber orientations in degrees.

    Returns ``(coherence, mean_orientation_deg)``. Coherence is the length of
    the mean doubled-angle vector: 1 when every fiber is parallel, near 0 when
    orientations are uniform. Doubling the angle makes 0 and 180 degrees the
    same direction, as they are for an undirected fiber.
    """
    angles = np.radians(np.asarray(orientation_deg, dtype=float))
    angles = angles[np.isfinite(angles)]
    if angles.size == 0:
        return np.nan, np.nan
    vector = np.exp(2j * angles).mean()
    return float(np.abs(vector)), float(np.mod(np.degrees(np.angle(vector) / 2), 180.0))


def _shape_for(image_shapes, imageid) -> tuple[int, int]:
    if isinstance(image_shapes, Mapping):
        if imageid not in image_shapes:
            raise KeyError(f"No image shape supplied for imageid {imageid!r}")
        shape = image_shapes[imageid]
    else:
        shape = image_shapes
    return int(shape[0]), int(shape[1])


def _summarise(group: pd.DataFrame, area_px: float, min_fiber_num: int, pixel_size_um: float | None) -> dict:
    # Area counts all matrix. Counts, shape and alignment use the ridge fibers
    # only: objects added by include_bright are cross-sections and patches,
    # with no meaningful length or orientation.
    fibers = group[group["object_type"] != "bright_matrix"] if "object_type" in group else group
    n = len(fibers)
    fiber_area = float(group["area"].sum()) if len(group) else 0.0
    row = {
        "n_fibers": n,
        "n_bright_matrix": len(group) - n,
        "fiber_area": fiber_area,
        "area_px": float(area_px),
        "area_fraction_pct": fiber_area / area_px * 100 if area_px > 0 else np.nan,
        "pixel_density": np.nan,
        "fiber_density": np.nan,
        "avg_alignment_score": np.nan,
        "orientation_coherence": np.nan,
        "mean_orientation": np.nan,
        **{f"avg_{prop}": np.nan for prop in FIBER_SUMMARY_PROPERTIES},
    }
    if pixel_size_um:
        area_mm2 = area_px * (pixel_size_um / 1000.0) ** 2
        row["fibers_per_mm2"] = n / area_mm2 if area_mm2 > 0 else np.nan
    if n >= min_fiber_num and n > 0 and area_px > 0:
        row["pixel_density"] = calculate_fiber_density(group, area_px)[0]
        row["fiber_density"] = calculate_fiber_density(fibers, area_px)[1]
        for prop in FIBER_SUMMARY_PROPERTIES:
            if prop in fibers:
                row[f"avg_{prop}"] = float(fibers[prop].mean())
        if "alignment_score" in fibers:
            scores = fibers["alignment_score"].to_numpy(dtype=float)
            scores = scores[np.isfinite(scores)]
            row["avg_alignment_score"] = float(scores.mean()) if len(scores) >= min_fiber_num else np.nan
        if "orientation" in fibers:
            row["orientation_coherence"], row["mean_orientation"] = orientation_coherence(fibers["orientation"])
    return row


def fiber_image_stats(
    fiber_df: pd.DataFrame,
    image_shapes,
    image_key: str = "imageid",
    fiber_type_key: str = "fiber_type",
    pixel_size_um: float | None = None,
) -> pd.DataFrame:
    """Whole-image fiber summary per image and fiber type (ark's FOV stats).

    ``image_shapes`` is one ``(rows, columns)`` tuple or a mapping from
    imageid to its shape.
    """
    rows = []
    keys = [image_key, fiber_type_key]
    for (imageid, fiber_type), group in fiber_df.groupby(keys, sort=True):
        height, width = _shape_for(image_shapes, imageid)
        rows.append({
            image_key: imageid,
            fiber_type_key: fiber_type,
            **_summarise(group, height * width, 1, pixel_size_um),
        })
    return pd.DataFrame(rows)


def fiber_tile_stats(
    fiber_df: pd.DataFrame,
    image_shapes,
    tile_size: int = 512,
    min_fiber_num: int = 5,
    image_key: str = "imageid",
    fiber_type_key: str = "fiber_type",
    x_key: str = "X_centroid",
    y_key: str = "Y_centroid",
    pixel_size_um: float | None = None,
) -> pd.DataFrame:
    """Fiber density, alignment and shape per square tile (ark's tile stats).

    Fibers are assigned to the tile containing their centroid. Every tile of
    every image is reported, including empty ones and the partial tiles at
    the right and bottom edges; ``tile_y``/``tile_x`` are the tile's top-left
    pixel. Statistics are NaN in tiles with fewer than ``min_fiber_num``
    fibers, as in ark, except ``n_fibers``, ``fiber_area`` and
    ``area_fraction_pct``, which are always defined.
    """
    tile_size = int(tile_size)
    if tile_size <= 0:
        raise ValueError("tile_size must be a positive number of pixels")
    rows = []
    for (imageid, fiber_type), group in fiber_df.groupby([image_key, fiber_type_key], sort=True):
        height, width = _shape_for(image_shapes, imageid)
        tile_row = np.floor(group[y_key].to_numpy(float) / tile_size).astype(int)
        tile_col = np.floor(group[x_key].to_numpy(float) / tile_size).astype(int)
        members = group.assign(_tr=tile_row, _tc=tile_col).groupby(["_tr", "_tc"])
        lookup = {key: frame for key, frame in members}
        for row_index in range(int(np.ceil(height / tile_size))):
            for col_index in range(int(np.ceil(width / tile_size))):
                y0, x0 = row_index * tile_size, col_index * tile_size
                tile_h, tile_w = min(tile_size, height - y0), min(tile_size, width - x0)
                frame = lookup.get((row_index, col_index), group.iloc[0:0])
                rows.append({
                    image_key: imageid,
                    fiber_type_key: fiber_type,
                    "tile_y": y0,
                    "tile_x": x0,
                    "tile_height": tile_h,
                    "tile_width": tile_w,
                    **_summarise(frame, tile_h * tile_w, min_fiber_num, pixel_size_um),
                })
    return pd.DataFrame(rows)


def fiber_region_stats(
    fiber_df: pd.DataFrame,
    region_raster: np.ndarray,
    region_names: Mapping[int, str],
    labels: np.ndarray | None = None,
    downsample: int = 1,
    min_fiber_num: int = 1,
    x_key: str = "X_centroid",
    y_key: str = "Y_centroid",
    pixel_size_um: float | None = None,
    chunk_rows: int = 2048,
) -> pd.DataFrame:
    """Fiber quantity and architecture inside each tissue region of one image.

    Parameters
    ----------
    fiber_df : DataFrame
        Fibers of a single image and fiber type.
    region_raster : ndarray of int
        Region code per pixel, possibly downsampled by ``downsample``; 0 means
        outside tissue. Built by
        :func:`spatioev.tl.niche.rasterize_tissue_regions`.
    region_names : mapping
        Region code to name, e.g. ``{1: "tumour", 2: "envelope", 3: "stroma"}``.
    labels : ndarray, optional
        Full-resolution fiber label image. When given, ``fiber_area`` counts
        the fiber pixels that actually fall inside each region; otherwise each
        fiber's whole area is credited to the region holding its centroid.

    Returns
    -------
    DataFrame
        One row per region plus an ``all_tissue`` row.
    """
    region_raster = np.asarray(region_raster)
    ds = max(1, int(downsample))
    codes = sorted(int(code) for code in region_names)
    pixel_area = np.bincount(region_raster.ravel(), minlength=max(codes) + 1).astype(float) * ds * ds

    rows_y = np.clip((fiber_df[y_key].to_numpy(float) // ds).astype(int), 0, region_raster.shape[0] - 1)
    cols_x = np.clip((fiber_df[x_key].to_numpy(float) // ds).astype(int), 0, region_raster.shape[1] - 1)
    fiber_region = region_raster[rows_y, cols_x] if len(fiber_df) else np.array([], dtype=int)

    fiber_pixels = None
    if labels is not None:
        fiber_pixels = np.zeros(max(codes) + 1, dtype=float)
        for start in range(0, labels.shape[0], chunk_rows):
            block = np.asarray(labels[start : start + chunk_rows])
            ys, xs = np.nonzero(block)
            if ys.size:
                codes_here = region_raster[
                    np.minimum((ys + start) // ds, region_raster.shape[0] - 1),
                    np.minimum(xs // ds, region_raster.shape[1] - 1),
                ]
                fiber_pixels += np.bincount(codes_here, minlength=len(fiber_pixels))[: len(fiber_pixels)]

    out = []
    tissue_codes = [code for code in codes if code != 0]
    for name, selected in [(region_names[code], [code]) for code in tissue_codes] + [("all_tissue", tissue_codes)]:
        mask = np.isin(fiber_region, selected)
        area = float(pixel_area[selected].sum())
        row = {"region": name, **_summarise(fiber_df.loc[mask], area, min_fiber_num, pixel_size_um)}
        if fiber_pixels is not None:
            row["fiber_area"] = float(fiber_pixels[selected].sum())
            row["area_fraction_pct"] = row["fiber_area"] / area * 100 if area > 0 else np.nan
        out.append(row)
    return pd.DataFrame(out)


def tile_cell_matrix_association(
    tile_stats: pd.DataFrame,
    obs: pd.DataFrame,
    phenotype_key: str,
    tile_size: int,
    metrics: Sequence[str] = ("area_fraction_pct", "orientation_coherence"),
    phenotypes: Sequence[str] | None = None,
    high_quantile: float = 0.75,
    min_cells_per_tile: int = 1,
    image_key: str = "imageid",
    fiber_type_key: str = "fiber_type",
    x_key: str = "X_centroid",
    y_key: str = "Y_centroid",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Relate matrix density/alignment per tile to the cells in that tile.

    Cells are binned on the same grid as ``tile_stats`` (which must come from
    :func:`fiber_tile_stats` with the same ``tile_size``). Only tiles holding
    at least ``min_cells_per_tile`` cells are used, so empty glass does not
    count as "low matrix".

    For each image, fiber type, matrix metric and phenotype the summary gives:

    - Spearman correlation across tiles between the metric and the
      phenotype's share of cells in the tile;
    - the phenotype's share in "high-matrix" tiles (metric at or above its
      ``high_quantile`` within the image) versus all other tissue tiles, and
      their ratio (>1 = enriched where matrix is dense / aligned).

    Returns
    -------
    per_tile : DataFrame
        Tile metrics joined with per-phenotype cell counts and shares.
    summary : DataFrame
        One row per image x fiber type x metric x phenotype.
    """
    from scipy.stats import spearmanr

    cells = obs[[image_key, x_key, y_key, phenotype_key]].dropna().copy()
    cells[phenotype_key] = cells[phenotype_key].astype(str)
    if phenotypes is not None:
        phenotypes = [str(p) for p in phenotypes]
    cells["tile_y"] = (np.floor(cells[y_key] / tile_size) * tile_size).astype(int)
    cells["tile_x"] = (np.floor(cells[x_key] / tile_size) * tile_size).astype(int)
    counts = cells.groupby([image_key, "tile_y", "tile_x", phenotype_key]).size().unstack(fill_value=0)
    total = counts.sum(axis=1).rename("n_cells")
    shares = counts.div(total, axis=0).add_prefix("share_")
    counts = counts.add_prefix("count_")
    cell_table = pd.concat([total, counts, shares], axis=1).reset_index()
    cell_table[image_key] = cell_table[image_key].astype(str)

    tiles = tile_stats.copy()
    tiles[image_key] = tiles[image_key].astype(str)
    per_tile = tiles.merge(cell_table, on=[image_key, "tile_y", "tile_x"], how="inner")
    per_tile = per_tile[per_tile["n_cells"] >= min_cells_per_tile]
    all_phenotypes = sorted(c.removeprefix("share_") for c in per_tile.columns if c.startswith("share_"))
    targets = phenotypes or all_phenotypes

    rows = []
    for (imageid, fiber_type), frame in per_tile.groupby([image_key, fiber_type_key], sort=True):
        for metric in metrics:
            if metric not in frame:
                continue
            values = frame[metric].to_numpy(float)
            finite = np.isfinite(values)
            if finite.sum() < 3:
                continue
            cutoff = np.nanquantile(values, high_quantile)
            high = finite & (values >= cutoff)
            other = finite & ~high
            for phenotype in targets:
                count_col = f"count_{phenotype}"
                share = frame.get(f"share_{phenotype}", pd.Series(0.0, index=frame.index)).to_numpy(float)
                count = frame.get(count_col, pd.Series(0, index=frame.index)).to_numpy(float)
                n_cells = frame["n_cells"].to_numpy(float)
                if np.nanstd(share[finite]) > 0 and np.nanstd(values[finite]) > 0:
                    rho, p_value = spearmanr(values[finite], share[finite])
                else:
                    rho, p_value = np.nan, np.nan
                share_high = count[high].sum() / n_cells[high].sum() if n_cells[high].sum() else np.nan
                share_other = count[other].sum() / n_cells[other].sum() if n_cells[other].sum() else np.nan
                rows.append({
                    image_key: imageid,
                    fiber_type_key: fiber_type,
                    "metric": metric,
                    "phenotype": phenotype,
                    "n_tiles": int(finite.sum()),
                    "n_high_tiles": int(high.sum()),
                    "high_cutoff": float(cutoff),
                    "spearman_rho": float(rho),
                    "spearman_p": float(p_value),
                    "share_in_high_tiles": float(share_high),
                    "share_in_other_tiles": float(share_other),
                    "enrichment_ratio": float(share_high / share_other) if share_other else np.nan,
                })
    return per_tile, pd.DataFrame(rows)


def cell_matrix_proximity(
    obs: pd.DataFrame,
    fiber_mask: np.ndarray,
    radius: float,
    downsample: int = 1,
    x_key: str = "X_centroid",
    y_key: str = "Y_centroid",
) -> pd.DataFrame:
    """Distance to matrix and local matrix fraction for every cell of one image.

    Measured on the fiber pixels themselves rather than fiber centroids, so a
    cell lying beside a long fiber is close to it even when the fiber's
    centroid is far away.

    Parameters
    ----------
    obs : DataFrame
        Cells of a single image, coordinates in full-resolution pixels.
    fiber_mask : ndarray
        Fiber label image or boolean mask (nonzero = matrix). May already be
        downsampled by ``downsample``.
    radius : float
        Neighbourhood radius in full-resolution pixels.
    downsample : int
        Factor by which ``fiber_mask`` is smaller than the image. Whole slides
        are usually measured at 2-4 to keep the distance transform in memory.

    Returns
    -------
    DataFrame indexed like ``obs`` with ``distance_to_matrix`` (pixels, 0 when
    the centroid is on a fiber) and ``matrix_fraction`` (share of pixels
    within ``radius`` that are matrix).
    """
    import scipy.ndimage as ndi
    from scipy.signal import oaconvolve

    ds = max(1, int(downsample))
    mask = np.asarray(fiber_mask) > 0
    distance = ndi.distance_transform_edt(~mask) * ds if mask.any() else np.full(mask.shape, np.inf)
    r = max(1, int(round(radius / ds)))
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    disk = (yy**2 + xx**2) <= r * r
    # Overlap-add FFT convolution: the same zero-padded result as
    # ndi.convolve, but a 20 um disk over a TMA core takes ~0.2 s instead of
    # minutes, and memory stays bounded on whole slides.
    fraction = oaconvolve(mask.astype(np.float32), disk.astype(np.float32) / disk.sum(), mode="same")
    # FFT round-off leaves ~1e-9 where there is no matrix at all; a pixel
    # inside the disk contributes at least 1 / disk.sum(), far above this.
    fraction = np.where(fraction < 1e-6, 0.0, np.minimum(fraction, 1.0))
    rows = np.clip((obs[y_key].to_numpy(float) // ds).astype(int), 0, mask.shape[0] - 1)
    cols = np.clip((obs[x_key].to_numpy(float) // ds).astype(int), 0, mask.shape[1] - 1)
    return pd.DataFrame(
        {"distance_to_matrix": distance[rows, cols], "matrix_fraction": fraction[rows, cols]},
        index=obs.index,
    )


def cell_matrix_enrichment(
    values: pd.Series,
    labels: pd.Series,
    phenotypes: Sequence[str] | None = None,
    n_permutations: int = 999,
    random_state: int | None = 0,
) -> pd.DataFrame:
    """Is a phenotype nearer to (or surrounded by more) matrix than chance?

    Compares the mean of ``values`` (e.g. ``matrix_fraction`` or
    ``distance_to_matrix``) over cells of each phenotype with the same mean
    after shuffling phenotype labels among all cells of the image. Positions
    stay fixed, so the null keeps the tissue's own layout and only asks
    whether *this* phenotype sits in matrix-rich places more than cells in
    general. Use one image at a time.

    Returns one row per phenotype with the observed mean, the null mean and
    SD, a z-score and a two-sided permutation p-value.
    """
    frame = pd.DataFrame({"value": values, "label": labels.astype(str)}).dropna()
    if frame.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(random_state)
    vals = frame["value"].to_numpy(float)
    labs = frame["label"].to_numpy()
    targets = [str(p) for p in phenotypes] if phenotypes is not None else sorted(set(labs))
    rows = []
    for phenotype in targets:
        mask = labs == phenotype
        n = int(mask.sum())
        if n == 0 or n == len(vals):
            continue
        observed = float(vals[mask].mean())
        null = np.array([vals[rng.permutation(len(vals))[:n]].mean() for _ in range(n_permutations)])
        sd = float(null.std(ddof=1))
        centre = float(null.mean())
        extreme = np.sum(np.abs(null - centre) >= abs(observed - centre))
        rows.append({
            "phenotype": phenotype,
            "n_cells": n,
            "observed_mean": observed,
            "null_mean": centre,
            "null_sd": sd,
            "z_score": (observed - centre) / sd if sd > 0 else np.nan,
            "p_value": float((extreme + 1) / (n_permutations + 1)),
        })
    return pd.DataFrame(rows)
