"""TWOMBLI-style matrix architecture metrics.

TWOMBLI (Wershof et al., *Life Sci Alliance* 2021) measures matrix pattern in
Fiji by chaining Ridge Detection, AnaMorf (Barry et al.) and OrientationJ.
This module re-implements its metric definitions in Python on SpatioEv's
fiber masks; no TWOMBLI code is used (its repository carries no licence).
The definitions follow AnaMorf v2 and IAClassLibrary (GPL-3.0, the same
licence as SpatioEv) and the TWOMBLI v1 macro:

========================  ====================================================
total length              skeleton length after pruning short terminal branches
endpoints / branchpoints  skeleton pixels with one / three or more neighbours;
                          adjacent branch pixels count as one branchpoint
curvature                 mean Menger curvature (1 / circumradius) of the
                          points ``window`` steps before and after each
                          skeleton point, over paths longer than 2 x window
fractal dimension         box counting: 10 box sizes from 2 px to 1/4.5 of the
                          area's longest side, grid centred, slope of
                          log(occupied boxes) vs log(size); NaN when R^2 < 0.9
lacunarity (AnaMorf)      ``|var / mean^2 - 1|`` of the binary mask's pixels
alignment                 coherency of the summed structure tensor (OrientationJ
                          "Dominant Direction"), 0 = isotropic, 1 = parallel
% HDM                     share of pixels brighter than ``lo + (1 - D/255) *
                          (hi - lo)`` with ``D`` = TWOMBLI's "Maximum Display
                          HDM" (default 200) and ``lo``/``hi`` the intensity range
========================  ====================================================

Where SpatioEv deliberately differs, and why:

- The mask is SpatioEv's fiber segmentation (ark-analysis method), skeletonised.
  TWOMBLI's Ridge Detection mask is a different detector, so values are
  "TWOMBLI-style" and comparable within SpatioEv, not to Fiji output.
- Curvature uses every skeleton path. AnaMorf's whole-image mode uses only the
  single longest path through the skeleton.
- AnaMorf's lacunarity depends only on the share ``p`` of mask pixels that are
  matrix: it equals ``|1/p - 2|``. It is reported for comparability, alongside
  a gliding-box lacunarity at a set box size, which measures how unevenly
  matrix and gaps are distributed at that scale (1 = perfectly even).
- ``hi``/``lo`` for % HDM are robust percentiles, not the raw image min/max,
  so a few hot pixels cannot move the threshold. The threshold is set once per
  image and applied to every region and tile of it.
- Metrics are summed per region or tile from one skeleton of the whole image,
  so cutting a region does not invent fiber ends at its boundary.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd
import scipy.ndimage as ndi

__all__ = [
    "anamorf_lacunarity",
    "architecture_by_group",
    "box_counting_dimension",
    "gliding_box_lacunarity",
    "hdm_threshold",
    "matrix_architecture",
    "matrix_architecture_tiled",
    "orientation_coherency",
    "skeleton_features",
]

SQRT2 = math.sqrt(2.0)
_ORTH = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIAG = ((-1, -1), (-1, 1), (1, -1), (1, 1))


# --------------------------------------------------------------------------- #
# Skeleton
# --------------------------------------------------------------------------- #
def _shift(array: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """``out[r, c] = array[r + dr, c + dc]``, zero outside."""
    out = np.zeros_like(array)
    rows, cols = array.shape
    out[max(0, -dr) : rows - max(0, dr), max(0, -dc) : cols - max(0, dc)] = array[
        max(0, dr) : rows - max(0, -dr), max(0, dc) : cols - max(0, -dc)
    ]
    return out


def _neighbours(skeleton: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Neighbour count and length weight per skeleton pixel.

    A diagonal neighbour only counts when neither pixel of the corner it cuts
    is set; otherwise the step is already made by the two orthogonal ones.
    Without this a one-pixel staircase looks like a chain of junctions and
    its length is overestimated. Each pixel carries half of every step it is
    part of, so the weights sum to the skeleton's length (a straight line of
    N pixels measures N - 1).
    """
    s = skeleton.astype(np.uint8)
    orth = sum(_shift(s, dr, dc) for dr, dc in _ORTH)
    diag = np.zeros_like(s)
    for dr, dc in _DIAG:
        diag += _shift(s, dr, dc) & (1 - _shift(s, dr, 0)) & (1 - _shift(s, 0, dc))
    count = (orth + diag) * s
    weight = (orth + SQRT2 * diag) / 2.0 * s
    return count, weight


def _label_branches(skeleton: np.ndarray, branches: np.ndarray) -> tuple[np.ndarray, int]:
    """Connected branch pieces, linked by the same rule as :func:`_neighbours`.

    Plain 8-connected labelling lets a branch reach round a junction pixel
    through a diagonal, so a short spur stays fused to the fiber it sprouts
    from and is never pruned.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    index = -np.ones(branches.shape, dtype=np.int64)
    rows, cols = np.nonzero(branches)
    index[rows, cols] = np.arange(rows.size)
    if rows.size == 0:
        return np.zeros(branches.shape, dtype=np.int64), 0
    sources, targets = [], []
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        neighbour = np.where(_shift(branches, dr, dc), _shift(index, dr, dc), -1)
        linked = branches & (neighbour >= 0)
        if dr and dc:  # diagonal: only when the corner it cuts is empty
            linked &= ~_shift(skeleton, dr, 0) & ~_shift(skeleton, 0, dc)
        sources.append(index[linked])
        targets.append(neighbour[linked])
    sources, targets = np.concatenate(sources), np.concatenate(targets)
    graph = coo_matrix((np.ones(sources.size), (sources, targets)), shape=(rows.size, rows.size))
    n, component = connected_components(graph, directed=False)
    labels = np.zeros(branches.shape, dtype=np.int64)
    labels[rows, cols] = component + 1
    return labels, int(n)


def _walk(component: np.ndarray, offset: tuple[int, int]) -> np.ndarray:
    """Order the pixels of a thin path, starting at an end, orthogonal steps first."""
    rows, cols = np.nonzero(component)
    points = set(zip(rows.tolist(), cols.tolist()))
    def degree(p):
        return sum((p[0] + dr, p[1] + dc) in points for dr, dc in _ORTH + _DIAG)
    start = min(points, key=degree)
    path, seen, current = [start], {start}, start
    while True:
        step = next(
            ((current[0] + dr, current[1] + dc) for dr, dc in _ORTH + _DIAG
             if (current[0] + dr, current[1] + dc) in points and (current[0] + dr, current[1] + dc) not in seen),
            None,
        )
        if step is None:
            break
        path.append(step)
        seen.add(step)
        current = step
    return np.asarray(path, dtype=float) + np.asarray(offset, dtype=float)


def _menger(path: np.ndarray, window: int) -> np.ndarray:
    """Menger curvature 1/R of (p[j-w], p[j], p[j+w]) for every interior j."""
    centre, before, after = path[window:-window], path[: -2 * window], path[2 * window :]
    a = np.linalg.norm(centre - before, axis=1)
    b = np.linalg.norm(centre - after, axis=1)
    c = np.linalg.norm(before - after, axis=1)
    cross = (before[:, 0] - centre[:, 0]) * (after[:, 1] - centre[:, 1]) - (before[:, 1] - centre[:, 1]) * (after[:, 0] - centre[:, 0])
    product = a * b * c
    # 1/R = 4 * area / (abc), area = |cross| / 2; collinear points give 0.
    return np.where(product > 0, 2.0 * np.abs(cross) / np.where(product > 0, product, 1.0), 0.0)


def skeleton_features(mask: np.ndarray, min_branch_length: float = 10.0, curvature_window: int = 30) -> dict:
    """Skeletonise a fiber mask and locate every skeleton-derived quantity.

    Parameters
    ----------
    mask : ndarray of bool or int
        Fiber mask or label image (nonzero = fiber).
    min_branch_length : float
        Terminal branches and isolated pieces shorter than this (pixels) are
        pruned, as AnaMorf's "Minimum Branch Length" and Ridge Detection's
        minimum line length do in TWOMBLI.
    curvature_window : int
        Steps along the path on each side of a point used for its curvature.

    Returns
    -------
    dict with ``skeleton`` (bool image), ``length_weight`` (per-pixel length,
    px), ``endpoints`` (bool image), ``branchpoints`` (N x 2 row/col
    centroids), ``curvature`` (N x 3 row, col, 1/px), ``segments`` (N x 3
    midpoint row, col, length px).
    """
    from skimage.morphology import skeletonize

    skeleton = skeletonize(np.asarray(mask) > 0)
    for _ in range(2):  # prune, then re-thin the stubs pruning can leave
        count, weight = _neighbours(skeleton)
        junction = skeleton & (count >= 3)
        branches = skeleton & ~junction
        labels, n = _label_branches(skeleton, branches)
        if n == 0 or min_branch_length <= 0:
            break
        index = np.arange(1, n + 1)
        length = ndi.sum(weight, labels, index)
        terminal = ndi.maximum(skeleton & (count <= 1), labels, index) > 0
        short = index[(length < min_branch_length) & terminal]
        if short.size == 0:
            break
        skeleton = skeleton & ~np.isin(labels, short)
        skeleton = skeletonize(skeleton)

    count, weight = _neighbours(skeleton)
    endpoints = skeleton & (count == 1)
    junction = skeleton & (count >= 3)
    junction_labels, n_junctions = ndi.label(junction, structure=np.ones((3, 3)))
    branchpoints = (
        np.asarray(ndi.center_of_mass(junction, junction_labels, np.arange(1, n_junctions + 1)), dtype=float).reshape(-1, 2)
        if n_junctions else np.empty((0, 2))
    )

    branches = skeleton & ~junction
    labels, n = _label_branches(skeleton, branches)
    segments, curvature = [], []
    if n:
        index = np.arange(1, n + 1)
        lengths = ndi.sum(weight, labels, index)
        middles = np.asarray(ndi.center_of_mass(branches, labels, index), dtype=float).reshape(-1, 2)
        segments = np.column_stack([middles, lengths])
        sizes = ndi.sum(branches, labels, index)
        window = max(1, int(curvature_window))
        for label, box in zip(index, ndi.find_objects(labels)):
            if sizes[label - 1] <= 2 * window:
                continue
            path = _walk(labels[box] == label, (box[0].start, box[1].start))
            if len(path) > 2 * window:
                values = _menger(path, window)
                curvature.append(np.column_stack([path[window:-window], values]))
    return {
        "skeleton": skeleton,
        "length_weight": weight,
        "endpoints": endpoints,
        "branchpoints": branchpoints,
        "curvature": np.vstack(curvature) if curvature else np.empty((0, 3)),
        "segments": np.asarray(segments).reshape(-1, 3) if len(segments) else np.empty((0, 3)),
    }


# --------------------------------------------------------------------------- #
# Pattern metrics
# --------------------------------------------------------------------------- #
def _crop_to(valid: np.ndarray | None, *arrays):
    if valid is None:
        return (None, *arrays)
    rows, cols = np.nonzero(valid)
    if rows.size == 0:
        return (None, *[None for _ in arrays])
    box = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
    return (valid[box], *[a[box] for a in arrays])


def box_counting_dimension(binary: np.ndarray, valid: np.ndarray | None = None, n_scales: int = 10, min_box: int = 2) -> float:
    """Box-counting fractal dimension as AnaMorf / IAClassLibrary compute it.

    Box sizes run from ``min_box`` to ``round(longest side / 4.5)`` in
    ``n_scales`` steps (integer step, as in the Java original), on a grid
    centred on the area. Returns NaN when the area is too small for the scale
    range, when a scale finds no matrix, or when the log-log fit has
    R^2 < 0.9.

    Like AnaMorf's estimator it reads slightly low in absolute terms -- about
    0.96 for a straight line and 1.92 for a filled square -- because the
    largest boxes reach a quarter of the image and overhang its edge. Values
    are comparable between images measured the same way.
    """
    valid, binary = _crop_to(valid, np.asarray(binary) > 0)
    if binary is None:
        return np.nan
    if valid is not None:
        binary = binary & valid
    height, width = binary.shape
    eps_max = int(round(max(height, width) / 4.5))
    if eps_max < min_box + n_scales:
        return np.nan
    step = (eps_max - min_box) // (n_scales - 1)
    sizes, counts = [], []
    for p in range(n_scales):
        eps = int(round(min_box + p * step))
        # AnaMorf starts the grid at round((side - eps) / 2) and steps back
        # by eps while it is positive, ending in (-eps, 0].
        off_y = int(round((height - eps) / 2.0))
        off_x = int(round((width - eps) / 2.0))
        off_y = -((-off_y) % eps) if off_y > 0 else off_y
        off_x = -((-off_x) % eps) if off_x > 0 else off_x
        padded = np.pad(binary, ((-off_y, 0), (-off_x, 0)))
        ph, pw = padded.shape
        padded = np.pad(padded, ((0, (-ph) % eps), (0, (-pw) % eps)))
        blocks = padded.reshape(padded.shape[0] // eps, eps, padded.shape[1] // eps, eps).any(axis=(1, 3))
        sizes.append(eps)
        counts.append(int(blocks.sum()))
    counts = np.asarray(counts, dtype=float)
    if (counts <= 0).any():
        return np.nan
    x, y = np.log(sizes), np.log(counts)
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    total = ((y - y.mean()) ** 2).sum()
    r_squared = 1 - (residual**2).sum() / total if total > 0 else 0.0
    return float(-slope) if r_squared >= 0.9 else np.nan


def anamorf_lacunarity(binary: np.ndarray, valid: np.ndarray | None = None) -> float:
    """AnaMorf's lacunarity ``|var / mean^2 - 1|`` over the area's pixels.

    For a binary mask with matrix share ``p`` this equals ``|1/p - 2|``: it is
    a transform of coverage. Kept so results can be set beside TWOMBLI's.
    """
    values = (np.asarray(binary) > 0)[valid] if valid is not None else (np.asarray(binary) > 0).ravel()
    p = values.mean() if values.size else np.nan
    return float(abs(1.0 / p - 2.0)) if p and np.isfinite(p) else np.nan


def _box_sums(binary: np.ndarray, valid: np.ndarray, box: int, stride: int = 1) -> np.ndarray:
    """Matrix pixel counts of every ``box`` x ``box`` window fully inside ``valid``."""
    def integral(a):
        out = np.zeros((a.shape[0] + 1, a.shape[1] + 1))
        out[1:, 1:] = np.cumsum(np.cumsum(a, axis=0, dtype=float), axis=1)
        return out
    if binary.shape[0] < box or binary.shape[1] < box:
        return np.empty(0)
    s, v = integral(binary.astype(float)), integral(valid.astype(float))
    def window(i):
        return i[box::stride, box::stride] - i[:-box:stride, box::stride] - i[box::stride, :-box:stride] + i[:-box:stride, :-box:stride]
    inside = window(v) >= box * box - 0.5
    return window(s)[inside]


def gliding_box_lacunarity(binary: np.ndarray, box: int, valid: np.ndarray | None = None, stride: int = 1) -> float:
    """Gliding-box lacunarity ``E[M^2] / E[M]^2`` at one box size.

    ``M`` is the number of matrix pixels in a ``box`` x ``box`` window, slid
    across every position whose window lies entirely inside ``valid``
    (Allain & Cloitre 1991). 1 means matrix is spread evenly at this scale;
    larger values mean it is clumped with gaps between.
    """
    valid, binary = _crop_to(valid, np.asarray(binary) > 0)
    if binary is None:
        return np.nan
    if valid is None:
        valid = np.ones(binary.shape, dtype=bool)
    sums = _box_sums(binary & valid, valid, int(box), stride)
    if sums.size == 0 or sums.mean() == 0:
        return np.nan
    return float((sums**2).mean() / sums.mean() ** 2)


def _structure_tensor_terms(binary: np.ndarray, sigma: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    smooth = ndi.gaussian_filter(np.asarray(binary, dtype=float), sigma)
    fy, fx = np.gradient(smooth)
    return fx * fx, fy * fy, fx * fy


def _coherency(jxx: float, jyy: float, jxy: float) -> tuple[float, float]:
    trace = jxx + jyy
    if trace <= 0:
        return np.nan, np.nan
    coherency = math.sqrt((jyy - jxx) ** 2 + 4 * jxy**2) / trace
    gradient_angle = 0.5 * math.degrees(math.atan2(2 * jxy, jxx - jyy))
    return float(coherency), float(np.mod(gradient_angle + 90.0, 180.0))  # fibers run across the gradient


def orientation_coherency(binary: np.ndarray, valid: np.ndarray | None = None, sigma: float = 1.0) -> tuple[float, float]:
    """OrientationJ-style dominant direction of a mask.

    Returns ``(coherency, orientation_deg)``: coherency of the structure
    tensor summed over the area (0 isotropic, 1 all parallel) and the
    dominant fiber direction in degrees from +x (``[0, 180)``, image rows
    pointing down, as in the fiber table).
    """
    jxx, jyy, jxy = _structure_tensor_terms(binary, sigma)
    if valid is not None:
        jxx, jyy, jxy = jxx[valid], jyy[valid], jxy[valid]
    return _coherency(float(jxx.sum()), float(jyy.sum()), float(jxy.sum()))


def hdm_threshold(intensity: np.ndarray, max_display_hdm: float = 200.0, saturation_pct: float = 0.35, max_samples: int = 4_000_000) -> float:
    """Intensity above which a pixel counts as high-density matrix.

    TWOMBLI scales the image to 8 bits, displays 0..``max_display_hdm`` of
    the inverted image and counts the pixels left non-black, i.e. those above
    ``lo + (1 - max_display_hdm / 255) * (hi - lo)``. Here ``lo``/``hi`` are
    the ``saturation_pct / 2`` and ``100 - saturation_pct / 2`` percentiles
    (TWOMBLI's contrast-saturation default, 0.35 %). Large images are
    subsampled evenly for the percentiles.
    """
    n_pixels = int(np.prod(intensity.shape))
    stride = max(1, int(math.ceil(math.sqrt(n_pixels / max_samples))))
    sample = np.asarray(intensity[::stride, ::stride], dtype=float)
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return np.nan
    lo, hi = np.percentile(sample, [saturation_pct / 2, 100 - saturation_pct / 2])
    return float(lo + (1 - max_display_hdm / 255.0) * (hi - lo))


# --------------------------------------------------------------------------- #
# Per-region / per-tile aggregation
# --------------------------------------------------------------------------- #
_SUM_KEYS = (
    "area_px", "mask_px", "skeleton_px", "length_px", "endpoints", "branchpoints",
    "curvature_sum", "curvature_n", "segments", "segment_length_px",
    "jxx", "jyy", "jxy", "hdm_px", "intensity_px",
    "lac_sum", "lac_sq_sum", "lac_n", "fd_weighted", "fd_weight",
)


def _group_sums(features: dict, mask: np.ndarray, groups: np.ndarray, n_groups: int, intensity, threshold,
                lacunarity_box: int, sigma: float, keep: np.ndarray, fd_min_px: int) -> dict[str, np.ndarray]:
    """Additive per-group totals for one image or tile. ``keep`` marks the tile core."""
    g = np.where(keep, groups, 0).astype(np.int64)
    flat = g.ravel()
    size = n_groups + 1
    def total(weights):
        return np.bincount(flat, weights=np.asarray(weights, dtype=float).ravel(), minlength=size)[:size]
    def at(points, weights=None):
        if len(points) == 0:
            return np.zeros(size)
        rows = np.clip(np.round(points[:, 0]).astype(int), 0, groups.shape[0] - 1)
        cols = np.clip(np.round(points[:, 1]).astype(int), 0, groups.shape[1] - 1)
        return np.bincount(g[rows, cols], weights=weights, minlength=size)[:size]

    skeleton = features["skeleton"]
    sums = {
        "area_px": np.bincount(flat, minlength=size)[:size].astype(float),
        "mask_px": total(mask),
        "skeleton_px": total(skeleton),
        "length_px": total(features["length_weight"]),
        "endpoints": total(features["endpoints"]),
        "branchpoints": at(features["branchpoints"]),
        "curvature_sum": at(features["curvature"][:, :2], features["curvature"][:, 2]),
        "curvature_n": at(features["curvature"][:, :2]),
        "segments": at(features["segments"][:, :2]),
        "segment_length_px": at(features["segments"][:, :2], features["segments"][:, 2]),
    }
    jxx, jyy, jxy = _structure_tensor_terms(skeleton, sigma)
    sums.update(jxx=total(jxx), jyy=total(jyy), jxy=total(jxy))
    if intensity is not None and np.isfinite(threshold):
        sums["hdm_px"] = total(np.asarray(intensity) > threshold)
        sums["intensity_px"] = sums["area_px"].copy()
    else:
        sums["hdm_px"] = np.zeros(size)
        sums["intensity_px"] = np.zeros(size)

    # Scale-dependent pattern metrics: computed per group on its own pixels.
    lac_sum, lac_sq, lac_n = np.zeros(size), np.zeros(size), np.zeros(size)
    fd_weighted, fd_weight = np.zeros(size), np.zeros(size)
    boxes = ndi.find_objects(g)
    for label, box in enumerate(boxes, start=1):
        if box is None or label > n_groups:
            continue
        valid = g[box] == label
        sub = skeleton[box]
        if lacunarity_box > 0:
            sums_box = _box_sums(sub & valid, valid, lacunarity_box)
            lac_sum[label], lac_sq[label], lac_n[label] = sums_box.sum(), (sums_box**2).sum(), sums_box.size
        area = valid.sum()
        if area >= fd_min_px:
            fd = box_counting_dimension(sub, valid)
            if np.isfinite(fd):
                fd_weighted[label], fd_weight[label] = fd * area, area
    sums.update(lac_sum=lac_sum, lac_sq_sum=lac_sq, lac_n=lac_n, fd_weighted=fd_weighted, fd_weight=fd_weight)
    return sums


def _finalise(sums: Mapping[str, float], pixel_size_um: float) -> dict:
    px_mm = pixel_size_um / 1000.0
    area_mm2 = sums["area_px"] * px_mm**2
    length_mm = sums["length_px"] * px_mm
    def ratio(a, b):
        return float(a / b) if b > 0 else np.nan
    coherency, orientation = _coherency(sums["jxx"], sums["jyy"], sums["jxy"])
    lac_mean = ratio(sums["lac_sum"], sums["lac_n"])
    lac_sq_mean = ratio(sums["lac_sq_sum"], sums["lac_n"])
    coverage = ratio(sums["skeleton_px"], sums["area_px"])
    return {
        "tissue_area_mm2": float(area_mm2),
        "fiber_area_fraction_pct": 100 * ratio(sums["mask_px"], sums["area_px"]),
        "total_length_mm": float(length_mm),
        "length_density_mm_per_mm2": ratio(length_mm, area_mm2),
        "endpoints": int(sums["endpoints"]),
        "branchpoints": int(sums["branchpoints"]),
        "endpoints_per_mm2": ratio(sums["endpoints"], area_mm2),
        "branchpoints_per_mm2": ratio(sums["branchpoints"], area_mm2),
        "branchpoints_per_mm_length": ratio(sums["branchpoints"], length_mm),
        "mean_branch_length_um": ratio(sums["segment_length_px"], sums["segments"]) * pixel_size_um,
        "curvature_per_um": ratio(sums["curvature_sum"], sums["curvature_n"]) / pixel_size_um,
        "curvature_points": int(sums["curvature_n"]),
        "fractal_dimension": ratio(sums["fd_weighted"], sums["fd_weight"]),
        "lacunarity_anamorf": float(abs(1.0 / coverage - 2.0)) if coverage and np.isfinite(coverage) else np.nan,
        "lacunarity_gliding": ratio(lac_sq_mean, lac_mean**2) if lac_mean and np.isfinite(lac_mean) else np.nan,
        "alignment_coherency": coherency,
        "dominant_orientation": orientation,
        "hdm_pct": 100 * ratio(sums["hdm_px"], sums["intensity_px"]),
        "fiber_thickness_um": ratio(sums["mask_px"], sums["length_px"]) * pixel_size_um,
    }


def architecture_by_group(
    mask: np.ndarray,
    groups: np.ndarray,
    group_names: Mapping[int, str],
    intensity: np.ndarray | None = None,
    threshold: float | None = None,
    pixel_size_um: float = 1.0,
    min_branch_length: float = 10.0,
    curvature_window: int = 30,
    lacunarity_box: int = 30,
    sigma: float = 1.0,
    all_name: str | None = "all_tissue",
    fd_min_px: int = 2500,
) -> pd.DataFrame:
    """TWOMBLI-style metrics for every group (region or tile) of one image.

    Parameters
    ----------
    mask : ndarray
        Fiber mask or label image (nonzero = fiber), full resolution.
    groups : ndarray of int
        Same shape as ``mask``; the group each pixel belongs to, 0 = ignore.
    group_names : mapping
        Group code to name.
    intensity, threshold
        Matrix channel and the % HDM cut-off from :func:`hdm_threshold`;
        omit both to skip % HDM.
    pixel_size_um : float
        Converts lengths and areas to micrometres / mm.
    min_branch_length, curvature_window, lacunarity_box
        In pixels; see :func:`skeleton_features` and
        :func:`gliding_box_lacunarity`.
    all_name : str, optional
        Name of an extra row pooling every nonzero group.
    fd_min_px : int
        Groups smaller than this get NaN fractal dimension: box counting
        needs a scale range.
    """
    groups = np.asarray(groups)
    features = skeleton_features(mask, min_branch_length, curvature_window)
    return _rows_from_sums(
        _group_sums(features, np.asarray(mask) > 0, groups, int(max(group_names)), intensity,
                    np.nan if threshold is None else threshold, int(lacunarity_box), sigma,
                    np.ones(groups.shape, dtype=bool), fd_min_px),
        group_names, pixel_size_um, all_name,
        pooled=(features, np.asarray(mask) > 0, groups > 0, int(lacunarity_box), fd_min_px),
    )


def _rows_from_sums(sums, group_names, pixel_size_um, all_name, pooled=None, pooled_scalars=None):
    rows = []
    for code, name in sorted(group_names.items()):
        if code >= len(sums["area_px"]) or sums["area_px"][code] == 0:
            continue
        rows.append({"group": name, **_finalise({k: v[code] for k, v in sums.items()}, pixel_size_um)})
    if all_name is not None:
        codes = [c for c in group_names if c < len(sums["area_px"])]
        pooled_sums = {k: float(np.asarray(v)[codes].sum()) for k, v in sums.items()}
        row = {"group": all_name, **_finalise(pooled_sums, pixel_size_um)}
        if pooled is not None:  # scale-dependent metrics on the pooled area itself
            features, mask, valid, box, fd_min_px = pooled
            row["fractal_dimension"] = box_counting_dimension(features["skeleton"], valid) if valid.sum() >= fd_min_px else np.nan
            row["lacunarity_gliding"] = gliding_box_lacunarity(features["skeleton"], box, valid) if box > 0 else np.nan
        elif pooled_scalars is not None:
            row.update(pooled_scalars)
        rows.append(row)
    return pd.DataFrame(rows)


def matrix_architecture(
    mask: np.ndarray,
    intensity: np.ndarray | None = None,
    valid: np.ndarray | None = None,
    pixel_size_um: float = 1.0,
    min_branch_length: float = 10.0,
    curvature_window: int = 30,
    lacunarity_box: int = 30,
    max_display_hdm: float = 200.0,
    saturation_pct: float = 0.35,
) -> dict:
    """All TWOMBLI-style metrics for one area of one image.

    ``valid`` restricts the measurement to an area (e.g. tissue); the whole
    image is used when omitted, as TWOMBLI does. The % HDM threshold is taken
    from the whole ``intensity`` image.
    """
    mask = np.asarray(mask)
    groups = np.ones(mask.shape, dtype=np.int64) if valid is None else np.asarray(valid).astype(np.int64)
    threshold = hdm_threshold(intensity, max_display_hdm, saturation_pct) if intensity is not None else None
    table = architecture_by_group(
        mask, groups, {1: "area"}, intensity, threshold, pixel_size_um,
        min_branch_length, curvature_window, lacunarity_box, all_name=None,
    )
    return table.iloc[0].drop("group").to_dict() if len(table) else {}


def matrix_architecture_tiled(
    mask_reader,
    shape: tuple[int, int],
    group_reader: Callable[[int, int, int, int], np.ndarray],
    group_names: Mapping[int, str],
    intensity_reader=None,
    threshold: float | None = None,
    pixel_size_um: float = 1.0,
    min_branch_length: float = 10.0,
    curvature_window: int = 30,
    lacunarity_box: int = 30,
    tile_size: int = 4096,
    all_name: str | None = "all_tissue",
    fd_min_px: int = 2500,
    max_coarse_side: int = 8192,
    coarse_factor: int | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> pd.DataFrame:
    """:func:`architecture_by_group` for images too large to skeletonise at once.

    ``mask_reader`` / ``intensity_reader`` support 2-D slicing (NumPy, zarr);
    ``group_reader(y0, y1, x0, x1)`` returns the group codes for a window.
    Each tile is skeletonised with a margin of context so paths and
    curvature are continuous across seams; only the tile core is counted.
    Length, counts, curvature, alignment, % HDM and the AnaMorf lacunarity
    are sums and so exact up to seam effects; gliding-box lacunarity loses
    the boxes that straddle a seam.

    Fractal dimension depends on the size of the area measured, so it is not
    averaged over tiles. The skeleton's box occupancy is kept on a grid
    ``k = ceil(longest side / max_coarse_side)`` pixels square and box
    counting runs once per region on it (smallest box ``2k`` pixels). For an
    image up to ``max_coarse_side`` that grid is the image itself, so results
    equal :func:`architecture_by_group`. Pass ``coarse_factor`` to fix ``k``
    so that slides of different sizes are box-counted on the same scales.
    """
    rows, cols = int(shape[0]), int(shape[1])
    report = progress or (lambda fraction, message: None)
    if rows <= tile_size and cols <= tile_size:
        mask = np.asarray(mask_reader[0:rows, 0:cols])
        intensity = None if intensity_reader is None else np.asarray(intensity_reader[0:rows, 0:cols])
        report(0.5, "Measuring architecture")
        return architecture_by_group(
            mask, group_reader(0, rows, 0, cols), group_names, intensity, threshold, pixel_size_um,
            min_branch_length, curvature_window, lacunarity_box, all_name=all_name, fd_min_px=fd_min_px,
        )
    k = int(coarse_factor) if coarse_factor else max(1, int(math.ceil(max(rows, cols) / max_coarse_side)))
    tile_size = max(k, (int(tile_size) // k) * k)
    margin = int(max(4 * curvature_window, 2 * min_branch_length, lacunarity_box, 64))
    n_groups = int(max(group_names))
    totals = {key: np.zeros(n_groups + 1) for key in _SUM_KEYS}
    coarse_shape = (int(math.ceil(rows / k)), int(math.ceil(cols / k)))
    occupancy = np.zeros(coarse_shape, dtype=bool)
    coarse_groups = np.zeros(coarse_shape, dtype=np.int64)
    tiles = [(y0, min(y0 + tile_size, rows), x0, min(x0 + tile_size, cols)) for y0 in range(0, rows, tile_size) for x0 in range(0, cols, tile_size)]
    for index, (y0, y1, x0, x1) in enumerate(tiles):
        py0, py1, px0, px1 = max(0, y0 - margin), min(rows, y1 + margin), max(0, x0 - margin), min(cols, x1 + margin)
        mask = np.asarray(mask_reader[py0:py1, px0:px1]) > 0
        groups = np.asarray(group_reader(py0, py1, px0, px1))
        keep = np.zeros(mask.shape, dtype=bool)
        keep[y0 - py0 : y1 - py0, x0 - px0 : x1 - px0] = True
        intensity = None if intensity_reader is None else np.asarray(intensity_reader[py0:py1, px0:px1])
        features = skeleton_features(mask, min_branch_length, curvature_window)
        sums = _group_sums(features, mask, groups, n_groups, intensity, np.nan if threshold is None else threshold,
                           int(lacunarity_box), 1.0, keep, fd_min_px=np.iinfo(np.int64).max)
        for key in _SUM_KEYS:
            totals[key] += sums[key]
        core = features["skeleton"][y0 - py0 : y1 - py0, x0 - px0 : x1 - px0]
        core_groups = groups[y0 - py0 : y1 - py0, x0 - px0 : x1 - px0]
        ch, cw = int(math.ceil(core.shape[0] / k)), int(math.ceil(core.shape[1] / k))
        padded = np.pad(core, ((0, ch * k - core.shape[0]), (0, cw * k - core.shape[1])))
        occupancy[y0 // k : y0 // k + ch, x0 // k : x0 // k + cw] = padded.reshape(ch, k, cw, k).any(axis=(1, 3))
        coarse_groups[y0 // k : y0 // k + ch, x0 // k : x0 // k + cw] = core_groups[::k, ::k][:ch, :cw]
        report((index + 1) / len(tiles), f"Architecture, tile {index + 1}/{len(tiles)}")

    coarse_min = max(1, fd_min_px // (k * k))
    for code, box in enumerate(ndi.find_objects(coarse_groups), start=1):
        if box is None or code not in group_names:
            continue
        valid = coarse_groups[box] == code
        if valid.sum() >= coarse_min:
            fd = box_counting_dimension(occupancy[box], valid)
            if np.isfinite(fd):
                totals["fd_weighted"][code], totals["fd_weight"][code] = fd, 1.0
    pooled = {}
    if all_name is not None:
        valid = coarse_groups > 0
        pooled["fractal_dimension"] = box_counting_dimension(occupancy, valid) if valid.sum() >= coarse_min else np.nan
    return _rows_from_sums(totals, group_names, pixel_size_um, all_name, pooled_scalars=pooled)
