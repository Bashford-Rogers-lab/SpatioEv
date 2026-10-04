"""Synthetic multiplexed-immunofluorescence TMA cores for practising SpatioEv.

:func:`make_demo_tma` writes a small cohort of artificial TMA cores in the
layout the SpatioEv preprocessing pipeline produces (see
:mod:`spatioev.io.qupath`), so every stage can be tried without real data::

    <destination>/README.txt
    <destination>/demo_sample_sheet.csv
    <destination>/DEMO_01/background/DEMO_01.ome.tiff
    <destination>/DEMO_01/segmentation/DEMO_01_whole_cell.tiff, DEMO_01_nuclear.tiff
    <destination>/DEMO_01/quantification/cell_table_<kind>.csv, channel_map.csv
    <destination>/DEMO_01/qupath/   only with ``with_classes=True``

Each core is a round piece of tissue with tumour nests (densely packed PCK+
epithelium) in a stroma of fibroblasts, macrophages, T cells and small
vessels, stained for 17 markers including five matrix proteins (COL1, COL12,
COL4, COL6, FN). The cell tables are measured from the image and masks that
were written, so every reader sees the same numbers.

Group B cores differ from group A by design, so the cohort comparison has
something to find: their collagen fibers are more aligned, they have more
WNT5A+ fibroblasts, and their macrophages sit next to fibroblasts.

Nothing in the data comes from a patient. Geometry is set for 0.325 µm
pixels; ``size`` changes the field of view, not the size of cells.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage as ndi

__all__ = ["CELL_CLASSES", "PANEL", "PIXEL_SIZE_UM", "make_demo_tma"]

#: Channel order of the demo OME-TIFFs (and of the cell-table marker columns).
PANEL = (
    "DAPI_INIT", "DAPI_FINAL", "CD3", "CD31", "CD45", "CD68", "CD90", "COL1", "COL12",
    "COL4", "COL6", "FN", "PCK", "PDPN", "SMA", "VIM", "WNT5A",
)
PIXEL_SIZE_UM = 0.325
#: QuPath classes written with ``with_classes=True`` (derived class after the colon).
CELL_CLASSES = ("Tumour", "Fibroblast", "Fibroblast: WNT5A+", "Macrophage", "T cell", "Endothelial", "Other")

_MATRIX = ("COL1", "COL12", "COL4", "COL6", "FN")
_MORPHOLOGY_COLUMNS = (
    "label", "centroid_y", "centroid_x", "eccentricity", "major_axis_length_px", "minor_axis_length_px",
    "perimeter_px", "solidity", "cell_size", "cell_area_px2", "passes_size_qc", "matched_nuclear_label",
    "nuclear_size", "nuclear_to_cell_area_ratio", "sample", "mask_type",
)
_TABLE_KINDS = ("raw", "size_normalized", "arcsinh_transformed")
_ARCSINH_COFACTOR = 5.0
_SIZE_QC_RANGE_PX = (100, 8000)
_SEPTUM_PX = 90  # minimum stroma between two tumour nests (~30 µm)

# Cell types (index into _TYPE_NAMES; "Fibroblast: WNT5A+" is a flag on fibroblasts).
TUMOUR, FIBROBLAST, MACROPHAGE, TCELL, ENDOTHELIAL, OTHER = range(6)
_TYPE_NAMES = ("Tumour", "Fibroblast", "Macrophage", "T cell", "Endothelial", "Other")

# Per cell type: nucleus semi-axes (long range, short/long ratio range, px),
# boundary irregularity, territory growth (long, short; px) and its irregularity.
_SHAPES = {
    TUMOUR: dict(a=(10.0, 12.5), ratio=(0.72, 0.95), wobble=0.07, grow=((8.0, 11.0), (8.0, 11.0)), cell_wobble=0.08),
    FIBROBLAST: dict(a=(13.0, 17.5), ratio=(0.28, 0.42), wobble=0.05, grow=((12.0, 22.0), (2.5, 4.5)), cell_wobble=0.06),
    MACROPHAGE: dict(a=(9.5, 12.0), ratio=(0.70, 0.92), wobble=0.12, grow=((7.0, 11.0), (7.0, 11.0)), cell_wobble=0.14),
    TCELL: dict(a=(7.5, 9.5), ratio=(0.85, 1.00), wobble=0.04, grow=((2.0, 3.5), (2.0, 3.5)), cell_wobble=0.05),
    ENDOTHELIAL: dict(a=(12.0, 15.0), ratio=(0.28, 0.38), wobble=0.04, grow=((6.0, 10.0), (2.0, 3.5)), cell_wobble=0.05),
    OTHER: dict(a=(9.0, 12.0), ratio=(0.60, 0.90), wobble=0.08, grow=((4.0, 7.0), (4.0, 7.0)), cell_wobble=0.08),
}

# Per cell type and marker: (probability positive, median level when positive).
_EXPRESSION = {
    TUMOUR: {"PCK": (0.97, 9000), "VIM": (0.05, 1500), "PDPN": (0.06, 1500), "WNT5A": (0.04, 1200)},
    FIBROBLAST: {"CD90": (0.80, 5000), "PDPN": (0.65, 4200), "SMA": (0.55, 6000), "VIM": (0.95, 7000)},
    MACROPHAGE: {"CD68": (0.95, 8000), "CD45": (0.85, 4000), "VIM": (0.50, 3000), "PDPN": (0.10, 1500), "WNT5A": (0.08, 1500)},
    TCELL: {"CD3": (0.95, 7000), "CD45": (0.95, 6000), "VIM": (0.30, 1500)},
    ENDOTHELIAL: {"CD31": (0.95, 7000), "VIM": (0.60, 3000), "CD90": (0.30, 2000)},
    OTHER: {"VIM": (0.40, 1500), "CD45": (0.10, 1200), "CD90": (0.08, 1200)},
}
_CELL_MARKERS = ("CD3", "CD31", "CD45", "CD68", "CD90", "PCK", "PDPN", "SMA", "VIM", "WNT5A")
_MEMBRANE_MARKERS = ("CD3", "CD31", "CD45", "CD90", "PDPN")
_GRANULAR_MARKERS = ("CD68", "WNT5A")

# Group differences (A, B).
_GROUP_SETTINGS = {
    "A": dict(orientation_spread=1.3, orientation_scale=90.0, fiber_spread=0.55, wrap=0.6, wnt5a_fraction=0.15, sma_fraction=0.50, macrophage_pull=None),
    "B": dict(orientation_spread=0.16, orientation_scale=160.0, fiber_spread=0.08, wrap=0.25, wnt5a_fraction=0.40, sma_fraction=0.70, macrophage_pull=15.0),
}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def make_demo_tma(
    destination: str | Path,
    n_cores: int = 6,
    size: int = 1500,
    seed: int = 0,
    with_classes: bool = False,
    overwrite: bool = False,
) -> list[Path]:
    """Write a synthetic multiplexed-IF TMA cohort for practising SpatioEv.

    Parameters
    ----------
    destination : path
        Folder that receives one folder per core, ``demo_sample_sheet.csv``
        and ``README.txt``. Created if missing.
    n_cores : int
        Number of cores, named ``DEMO_01``, ``DEMO_02``, ... The first half
        (rounded up) is group ``A``, the rest group ``B``.
    size : int
        Image width and height in pixels (0.325 µm per pixel). 1500 px gives
        about 1,000 cells per core.
    seed : int
        Random seed; the same seed always gives the same data.
    with_classes : bool
        Also write what QuPath's ``spatioev_export_classes.groovy`` would
        write (``<core>/qupath/<core>_cell_classes.csv`` with the true cell
        types and an empty ``<core>_annotations.geojson``), so the QuPath step
        can be skipped.
    overwrite : bool
        Replace core folders that already exist. Without it, existing core
        folders raise :class:`FileExistsError` and nothing is written.

    Returns
    -------
    list of Path
        The core folders, in order.

    Notes
    -----
    Each core holds a 17-channel uint16 OME-TIFF (channels as in
    :data:`PANEL`, a three-level pyramid), whole-cell and nuclear label
    images (nuclei are numbered differently from cells; the cell table's
    ``matched_nuclear_label`` links them) and the cell tables:
    ``cell_table_raw.csv`` holds the sum of each channel over each cell
    (``<marker>_nuclear``: over its nucleus), ``cell_table_size_normalized.csv``
    those sums divided by cell (nucleus) area, and
    ``cell_table_arcsinh_transformed.csv`` ``arcsinh(size_normalized / 5)``.
    """
    destination = Path(destination).expanduser()
    n_cores, size, seed = int(n_cores), int(size), int(seed)
    if n_cores < 1:
        raise ValueError("n_cores must be at least 1")
    if size < 256:
        raise ValueError("size must be at least 256 pixels")
    if seed < 0:
        raise ValueError("seed must be zero or positive")
    width = max(2, len(str(n_cores)))
    names = [f"DEMO_{index:0{width}d}" for index in range(1, n_cores + 1)]
    n_group_a = (n_cores + 1) // 2
    groups = ["A" if index < n_group_a else "B" for index in range(n_cores)]

    existing = [name for name in names if (destination / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"{destination}: core folder(s) {', '.join(existing)} already exist; "
            "pass overwrite=True (or --overwrite) to replace them"
        )
    destination.mkdir(parents=True, exist_ok=True)

    cores, summaries = [], []
    for index, (name, group) in enumerate(zip(names, groups)):
        rng = np.random.default_rng([seed, index])
        summary = _write_core(destination, name, group, size, rng, with_classes)
        cores.append(destination / name)
        summaries.append(summary)

    pd.DataFrame({
        "sample_id": names,
        "group": groups,
        "imageid": [""] * n_cores,
        "patient_id": [f"P{index:0{width}d}" for index in range(1, n_cores + 1)],
        "project_root": [""] * n_cores,
    }).to_csv(destination / "demo_sample_sheet.csv", index=False)
    _write_readme(destination, summaries, size, seed, with_classes)
    return cores


# --------------------------------------------------------------------------- #
# Random fields and shapes
# --------------------------------------------------------------------------- #
def _standardise(field: np.ndarray) -> np.ndarray:
    field = field - field.mean()
    spread = field.std()
    return (field / spread if spread > 0 else field).astype(np.float32)


def _smooth_noise(rng: np.random.Generator, shape: tuple[int, int], scale: float) -> np.ndarray:
    """Zero-mean, unit-variance noise with features about ``scale`` pixels across."""
    height, width = shape
    scale = max(float(scale), 2.0)
    step = max(1, int(scale // 4))
    coarse = rng.standard_normal((height // step + 3, width // step + 3)).astype(np.float32)
    coarse = ndi.gaussian_filter(coarse, 0.5 * scale / step, mode="wrap")
    fine = ndi.zoom(coarse, step, order=1) if step > 1 else coarse
    return _standardise(fine[:height, :width])


def _fine_noise(rng: np.random.Generator, shape: tuple[int, int], sigma: float) -> np.ndarray:
    return _standardise(ndi.gaussian_filter(rng.standard_normal(shape, dtype=np.float32), sigma))


def _harmonics(rng: np.random.Generator, amplitude: float, orders=(2, 3, 4, 5)) -> np.ndarray:
    """Random Fourier coefficients ``(order, amplitude, phase)`` for an irregular outline."""
    orders = np.asarray(orders, dtype=np.float32)
    amplitudes = rng.normal(0, amplitude, len(orders)) / np.sqrt(orders - 1)
    return np.column_stack([orders, amplitudes, rng.uniform(0, 2 * np.pi, len(orders))]).astype(np.float32)


def _outline(angle: np.ndarray, harmonics: np.ndarray) -> np.ndarray:
    radius = np.ones_like(angle)
    for order, amplitude, phase in harmonics:
        radius += amplitude * np.cos(order * angle + phase)
    return np.maximum(radius, 0.35)


def _poisson_points(
    rng: np.random.Generator,
    allowed: np.ndarray,
    min_dist: float,
    n_max: int,
    n_candidates: int,
    density: np.ndarray | None = None,
    existing: np.ndarray | None = None,
) -> np.ndarray:
    """Dart throwing: random points in ``allowed`` at least ``min_dist`` apart."""
    rows, cols = np.nonzero(allowed)
    if not len(rows) or n_max <= 0:
        return np.zeros((0, 2), dtype=float)
    pick = rng.integers(0, len(rows), n_candidates)
    if density is not None:
        pick = pick[rng.random(n_candidates) < density[rows[pick], cols[pick]]]
    candidates = np.column_stack([rows[pick], cols[pick]]).astype(float) + rng.uniform(-0.5, 0.5, (len(pick), 2))
    cell = min_dist
    grid: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for y, x in (existing if existing is not None else np.zeros((0, 2))):
        grid.setdefault((int(y // cell), int(x // cell)), []).append((y, x))
    limit = min_dist * min_dist
    accepted = []
    for y, x in candidates:
        gy, gx = int(y // cell), int(x // cell)
        clear = True
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for py, px in grid.get((gy + dy, gx + dx), ()):
                    if (py - y) ** 2 + (px - x) ** 2 < limit:
                        clear = False
                        break
                if not clear:
                    break
            if not clear:
                break
        if clear:
            accepted.append((y, x))
            grid.setdefault((gy, gx), []).append((y, x))
            if len(accepted) >= n_max:
                break
    return np.asarray(accepted, dtype=float).reshape(-1, 2)


def _ellipse_mask(shape, centre, semi_axes, angle, harmonics=None) -> np.ndarray:
    """Filled (optionally irregular) ellipse as a boolean image."""
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    dy, dx = rows - centre[0], cols - centre[1]
    cos, sin = np.cos(angle), np.sin(angle)
    u = (dx * cos + dy * sin) / semi_axes[0]
    v = (-dx * sin + dy * cos) / semi_axes[1]
    rho = np.sqrt(u * u + v * v)
    if harmonics is not None:
        rho = rho / _outline(np.arctan2(v, u), harmonics)
    return rho < 1


# --------------------------------------------------------------------------- #
# Tissue layout
# --------------------------------------------------------------------------- #
@dataclass
class _Layout:
    tissue: np.ndarray
    nest: np.ndarray
    lumen: np.ndarray
    d_nest: np.ndarray
    d_lumen: np.ndarray
    theta: np.ndarray
    vessels: list
    aggregates: np.ndarray
    centre: tuple[float, float]
    radius: float


def _layout(rng: np.random.Generator, size: int, group: str) -> _Layout:
    shape = (size, size)
    area_scale = (size / 1500.0) ** 2
    centre = (size / 2 + rng.normal(0, size * 0.01), size / 2 + rng.normal(0, size * 0.01))
    radius = 0.44 * size
    rows, cols = np.ogrid[:size, :size]
    dist = np.hypot(rows - centre[0], cols - centre[1]).astype(np.float32)
    angle = np.arctan2(rows - centre[0], cols - centre[1]).astype(np.float32)
    edge = radius * (_outline(angle, _harmonics(rng, 0.03, (2, 3, 4, 5, 6))) + 0.012 * _smooth_noise(rng, shape, 25))
    tissue = dist < edge
    tissue_area = float(tissue.sum())

    # Tumour nests: irregular blobs on a warped grid, ~30-40% of the tissue,
    # laid out at half resolution.
    n_nests = int(rng.integers(3, 7))
    n_nests = max(2, int(round(n_nests * area_scale))) if area_scale < 1 else n_nests
    target = rng.uniform(0.32, 0.40)
    weights = rng.uniform(0.75, 1.3, n_nests)
    radii = np.sqrt(target * tissue_area * weights / weights.sum() / np.pi)
    centres = []
    for index in np.argsort(-radii):
        r = radii[index]
        for attempt in range(400):
            room = max(radius - r - 0.06 * radius, 1.0)
            offset_r, offset_a = room * np.sqrt(rng.random()), rng.uniform(0, 2 * np.pi)
            candidate = (centre[0] + offset_r * np.sin(offset_a), centre[1] + offset_r * np.cos(offset_a))
            if all(np.hypot(candidate[0] - c[0], candidate[1] - c[1]) > (r + radii[j]) * 1.02 + 0.05 * size for j, c in centres):
                break
            if attempt % 20 == 19:
                r *= 0.97
        radii[index] = r
        centres.append((index, candidate))
    blobs = [[c, radii[j], rng.uniform(0, np.pi), rng.uniform(0.65, 1.0), _harmonics(rng, 0.20)] for j, c in centres]
    half = np.arange(0, size, 2, dtype=np.float32) + 0.5
    half_shape = (len(half), len(half))
    wy = half[:, None] + 30 * _smooth_noise(rng, half_shape, 50) + 7 * _smooth_noise(rng, half_shape, 12)
    wx = half[None, :] + 30 * _smooth_noise(rng, half_shape, 50) + 7 * _smooth_noise(rng, half_shape, 12)

    def depth_of(blob):
        (cy, cx), r, rot, ratio, harmonics = blob
        dy, dx = wy - cy, wx - cx
        u = (dx * np.cos(rot) + dy * np.sin(rot)) / r
        v = (-dx * np.sin(rot) + dy * np.cos(rot)) / (r * ratio)
        return np.sqrt(u * u + v * v) / _outline(np.arctan2(v, u), harmonics)

    inner = ndi.binary_erosion(tissue, iterations=12)
    inner_half = inner[::2, ::2]
    min_area = 65000 * area_scale
    for _ in range(6):
        # Every pixel belongs to the blob it is deepest inside; a band of stroma
        # (_SEPTUM_PX wide) along the borders between blobs keeps nests apart.
        best = np.full(half_shape, np.inf, dtype=np.float32)
        owner = np.zeros(half_shape, dtype=np.int32)
        for number, blob in enumerate(blobs, start=1):
            depth = depth_of(blob)
            closer = depth < best
            best[closer], owner[closer] = depth[closer], number
        border = (ndi.maximum_filter(owner, 3) != owner) | (ndi.minimum_filter(owner, 3) != owner)
        nest_half = (best < 1) & (ndi.distance_transform_edt(~border) >= _SEPTUM_PX / 4) & inner_half
        areas = 4 * np.bincount(owner[nest_half], minlength=len(blobs) + 1)[1:]
        fraction = areas.sum() / tissue_area
        small = [k for k in range(n_nests) if areas[k] < min_area]
        if 0.28 <= fraction <= 0.42 and not small:
            break
        scale = float(np.clip(np.sqrt(target / max(fraction, 0.05)), 0.85, 1.2))
        for k, blob in enumerate(blobs):
            blob[1] *= scale * (1.15 if k in small else 1.0)
    # Up to two small tumour clusters (too small to count as nests), clear of the nests.
    for _ in range(int(rng.integers(0, 3)) if area_scale >= 0.5 else 0):
        r = rng.uniform(24, 36)
        clear = 2 * ndi.distance_transform_edt(~nest_half)
        spots = np.argwhere((clear > _SEPTUM_PX / 2 + 1.3 * r + 10) & ndi.binary_erosion(inner_half, iterations=int(r / 2) + 5))
        if not len(spots):
            break
        cy, cx = half[spots[rng.integers(len(spots))]]
        blob = [(cy, cx), r, rng.uniform(0, np.pi), rng.uniform(0.7, 1.0), _harmonics(rng, 0.15)]
        nest_half |= (depth_of(blob) < 1) & inner_half
    nest = ndi.zoom(nest_half.astype(np.float32), 2, order=1, grid_mode=True, mode="nearest")[:size, :size] > 0.5
    nest = ndi.binary_opening(nest & inner, iterations=3)
    nest = ndi.binary_fill_holes(nest)
    labels, n_labels = ndi.label(nest)
    if n_labels:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        nest = (sizes >= 1200)[labels]

    d_nest = ndi.distance_transform_edt(~nest).astype(np.float32)

    # Vessels: elliptical lumens in the stroma, kept clear of nests and each other.
    stroma_core = tissue & (d_nest > 45) & ndi.binary_erosion(tissue, iterations=45)
    n_vessels = int(rng.integers(3, 7))
    n_vessels = max(1, int(round(n_vessels * area_scale))) if area_scale < 1 else n_vessels
    vessels, lumen = [], np.zeros(shape, dtype=bool)
    candidates = np.argwhere(stroma_core)
    for _ in range(n_vessels):
        for _attempt in range(60):
            if not len(candidates):
                break
            y, x = candidates[rng.integers(len(candidates))]
            r = rng.uniform(9, 26)
            if d_nest[y, x] > r + 35 and all(np.hypot(y - v["centre"][0], x - v["centre"][1]) > 110 for v in vessels):
                axes = (r, r * rng.uniform(0.45, 1.0))
                rot = rng.uniform(0, np.pi)
                vessels.append({"centre": (float(y), float(x)), "axes": axes, "angle": rot})
                lumen |= _ellipse_mask(shape, (y, x), axes, rot)
                break
    d_lumen = ndi.distance_transform_edt(~lumen).astype(np.float32) if lumen.any() else np.full(shape, np.inf, np.float32)

    # Lymphoid aggregates (T-cell clusters) in the stroma.
    free = tissue & (d_nest > 70) & (d_lumen > 60) & ndi.binary_erosion(tissue, iterations=60)
    spots = np.argwhere(free)
    n_aggregates = int(rng.integers(1, 3)) if len(spots) and area_scale >= 0.5 else 0
    aggregates = spots[rng.integers(len(spots), size=n_aggregates)].astype(float) if n_aggregates else np.zeros((0, 2))

    theta = _orientation_field(rng, shape, d_nest, d_lumen, group)
    return _Layout(tissue, nest, lumen, d_nest, d_lumen, theta, vessels, aggregates, centre, radius)


def _orientation_field(rng, shape, d_nest, d_lumen, group) -> np.ndarray:
    """Local fiber direction (radians): a group-specific field that wraps around nests and vessels."""
    settings = _GROUP_SETTINGS[group]
    base = rng.uniform(0, np.pi) + settings["orientation_spread"] * _smooth_noise(rng, shape, settings["orientation_scale"])
    c, s = np.cos(2 * base), np.sin(2 * base)
    for distance, reach, strength in ((d_nest, 45.0, settings["wrap"]), (d_lumen, 25.0, 0.8)):
        finite = np.where(np.isfinite(distance), distance, 1e4)
        smooth = ndi.gaussian_filter(np.minimum(finite, 400), 4)
        gy, gx = np.gradient(smooth)
        tangent = np.arctan2(gx, -gy)
        weight = strength * np.exp(-finite / reach)
        c = (1 - weight) * c + weight * np.cos(2 * tangent)
        s = (1 - weight) * s + weight * np.sin(2 * tangent)
    return (0.5 * np.arctan2(s, c)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Cells
# --------------------------------------------------------------------------- #
@dataclass
class _Cells:
    y: np.ndarray
    x: np.ndarray
    kind: np.ndarray
    wnt5a: np.ndarray
    angle: np.ndarray
    in_nest: np.ndarray


def _place_cells(rng: np.random.Generator, layout: _Layout, group: str) -> _Cells:
    from scipy.spatial import cKDTree

    settings = _GROUP_SETTINGS[group]
    shape = layout.tissue.shape
    inner = ndi.binary_erosion(layout.tissue, iterations=10)

    # Tumour nests: densely packed.
    nest_core = ndi.binary_erosion(layout.nest, iterations=9)
    nest_area = float(nest_core.sum())
    tumour = _poisson_points(rng, nest_core, 26.0, int(nest_area / 500) + 10, int(nest_area / 50) + 50)

    # Stroma: lymphoid aggregates first, then sparser cells with patchy density.
    stroma_allowed = inner & (layout.d_nest > 16) & (layout.d_lumen > 24)
    stroma_area = float(stroma_allowed.sum())
    aggregate_points = []
    for cy, cx in layout.aggregates:
        disc = stroma_allowed & _ellipse_mask(shape, (cy, cx), (rng.uniform(55, 85),) * 2, 0.0, _harmonics(rng, 0.15))
        aggregate_points.append(_poisson_points(rng, disc, 21.0, 400, int(disc.sum() / 40) + 10))
    aggregate_points = np.concatenate(aggregate_points) if aggregate_points else np.zeros((0, 2))
    density = 0.35 + 0.65 / (1 + np.exp(-1.6 * _smooth_noise(rng, shape, 90)))
    density *= 1 + 0.7 * np.exp(-layout.d_nest / 60)
    density /= density.max()
    stroma = _poisson_points(
        rng, stroma_allowed, 26.0, int(stroma_area / 1500), int(stroma_area / 90) + 50, density=density, existing=np.concatenate([tumour, aggregate_points]),
    )

    # Endothelium: flattened cells lining each lumen.
    endo_y, endo_x, endo_angle = [], [], []
    for vessel in layout.vessels:
        (cy, cx), (a, b), rot = vessel["centre"], vessel["axes"], vessel["angle"]
        perimeter = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))
        n = max(3, int(round(perimeter / 34)))
        t = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.uniform(0, 2 * np.pi) + rng.normal(0, 0.12, n)
        u, v = (a + 5) * np.cos(t), (b + 5) * np.sin(t)
        endo_y.extend(cy + u * np.sin(rot) + v * np.cos(rot))
        endo_x.extend(cx + u * np.cos(rot) - v * np.sin(rot))
        du, dv = -(a + 5) * np.sin(t), (b + 5) * np.cos(t)
        endo_angle.extend(np.arctan2(du * np.sin(rot) + dv * np.cos(rot), du * np.cos(rot) - dv * np.sin(rot)))

    n_t, n_a, n_s, n_e = len(tumour), len(aggregate_points), len(stroma), len(endo_y)
    kind = np.concatenate([
        np.full(n_t, TUMOUR), np.full(n_a, TCELL), np.full(n_s, OTHER), np.full(n_e, ENDOTHELIAL)
    ]).astype(np.int8)
    points = np.concatenate([tumour, aggregate_points, stroma, np.column_stack([endo_y, endo_x]).reshape(-1, 2)])

    # A few lymphocytes and macrophages inside nests.
    roll = rng.random(n_t)
    kind[:n_t][roll < 0.03] = TCELL
    kind[:n_t][(roll >= 0.03) & (roll < 0.04)] = MACROPHAGE

    # Stroma: fibroblasts, then macrophages, T cells, others. In group A
    # fibroblasts and macrophages fill complementary patches; in group B
    # macrophages sit next to fibroblasts.
    stroma_index = np.arange(n_t + n_a, n_t + n_a + n_s)
    zone = _smooth_noise(rng, shape, 120)[_pixel(points[stroma_index, 0], shape[0]), _pixel(points[stroma_index, 1], shape[1])]
    pull = settings["macrophage_pull"]
    weights = np.exp((0.0 if pull else 1.3) * zone)
    fibroblasts = rng.choice(stroma_index, int(round(0.55 * n_s)), replace=False, p=weights / weights.sum()) if n_s else np.zeros(0, int)
    kind[fibroblasts] = FIBROBLAST
    candidates = kind[stroma_index] == OTHER
    rest = stroma_index[candidates]
    if len(rest) and len(fibroblasts):
        distance, _ = cKDTree(points[fibroblasts]).query(points[rest])
        if pull:
            weights = np.exp(-distance / pull) + 1e-3
        else:
            weights = np.exp(-1.3 * zone[candidates]) * (1 - np.exp(-distance / 30)) + 1e-3
        chosen = rng.choice(rest, int(round(0.45 * len(rest))), replace=False, p=weights / weights.sum())
        kind[chosen] = MACROPHAGE
    rest = stroma_index[kind[stroma_index] == OTHER]
    if len(rest):
        near_nest = layout.d_nest[_pixel(points[rest, 0], shape[0]), _pixel(points[rest, 1], shape[1])]
        weights = 1 + 1.5 * np.exp(-near_nest / 80)
        chosen = rng.choice(rest, int(round(0.40 * len(rest))), replace=False, p=weights / weights.sum())
        kind[chosen] = TCELL

    # WNT5A+ fibroblasts, concentrated near nest edges.
    wnt5a = np.zeros(len(kind), dtype=bool)
    fibro = np.flatnonzero(kind == FIBROBLAST)
    if len(fibro):
        near_nest = layout.d_nest[_pixel(points[fibro, 0], shape[0]), _pixel(points[fibro, 1], shape[1])]
        weights = np.exp(-near_nest / 100)
        chosen = rng.choice(fibro, int(round(settings["wnt5a_fraction"] * len(fibro))), replace=False, p=weights / weights.sum())
        wnt5a[chosen] = True

    angle = rng.uniform(0, np.pi, len(kind))
    along = (kind == FIBROBLAST)
    angle[along] = layout.theta[_pixel(points[along, 0], shape[0]), _pixel(points[along, 1], shape[1])] + rng.normal(0, 0.12, along.sum())
    angle[n_t + n_a + n_s :] = endo_angle
    in_nest = np.arange(len(kind)) < n_t
    return _Cells(points[:, 0], points[:, 1], kind, wnt5a, angle, in_nest)


def _pixel(values: np.ndarray, limit: int) -> np.ndarray:
    return np.clip(np.round(values).astype(int), 0, limit - 1)


def _rasterize(rng: np.random.Generator, cells: _Cells, layout: _Layout) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nuclear and whole-cell label images (both numbered by cell index + 1)."""
    shape = layout.tissue.shape
    stroma = layout.tissue & ~layout.nest & ~layout.lumen
    allowed = {ENDOTHELIAL: stroma & (layout.d_lumen <= 13)}
    roughness = 0.05 * _smooth_noise(rng, shape, 5)
    nuclear = np.zeros(shape, dtype=np.int32)
    owner = np.zeros(shape, dtype=np.int32)
    best = np.full(shape, 1.0, dtype=np.float32)
    n = len(cells.kind)
    for index in range(n):
        kind = int(cells.kind[index])
        spec = _SHAPES[kind]
        a = rng.uniform(*spec["a"])
        b = a * rng.uniform(*spec["ratio"])
        grow_long, grow_short = rng.uniform(*spec["grow"][0]), rng.uniform(*spec["grow"][1])
        ta, tb = a + grow_long, b + grow_short
        nucleus_h, cell_h = _harmonics(rng, spec["wobble"]), _harmonics(rng, spec["cell_wobble"])
        cy, cx, rot = cells.y[index], cells.x[index], cells.angle[index]
        reach = int(np.ceil(max(ta, tb) * 1.4)) + 2
        y0, y1 = max(0, int(cy) - reach), min(shape[0], int(cy) + reach + 1)
        x0, x1 = max(0, int(cx) - reach), min(shape[1], int(cx) + reach + 1)
        if y0 >= y1 or x0 >= x1:
            continue
        dy = (np.arange(y0, y1, dtype=np.float32) - cy)[:, None]
        dx = (np.arange(x0, x1, dtype=np.float32) - cx)[None, :]
        cos, sin = np.cos(rot), np.sin(rot)
        u, v = dx * cos + dy * sin, -dx * sin + dy * cos
        un, vn = u / a, v / b
        inside = np.sqrt(un * un + vn * vn) < _outline(np.arctan2(vn, un), nucleus_h)
        ut, vt = u / ta, v / tb
        if kind in (FIBROBLAST, ENDOTHELIAL):
            # Spindle: tapers to points along the long axis.
            score = np.sqrt(ut * ut + vt * vt / np.sqrt(np.clip(1 - ut * ut, 0.03, None)))
        else:
            score = np.sqrt(ut * ut + vt * vt)
        score = score / _outline(np.arctan2(vt, ut), cell_h) + roughness[y0:y1, x0:x1]
        region = (layout.nest if cells.in_nest[index] else allowed.get(kind, stroma))[y0:y1, x0:x1]
        window_best = best[y0:y1, x0:x1]
        claim = (score < window_best) & region
        window_best[claim] = score[claim]
        owner[y0:y1, x0:x1][claim] = index + 1
        free = inside & (nuclear[y0:y1, x0:x1] == 0) & region
        nuclear[y0:y1, x0:x1][free] = index + 1

    # A nucleus always belongs to its own cell.
    has_nucleus = nuclear > 0
    owner[has_nucleus] = nuclear[has_nucleus]
    owner = _keep_largest_parts(owner)
    nuclear[owner != nuclear] = 0
    nucleus_px = np.bincount(nuclear.ravel(), minlength=n + 1)
    cell_px = np.bincount(owner.ravel(), minlength=n + 1)
    keep = (nucleus_px >= 25) & (cell_px >= 60)
    keep[0] = False
    owner[~keep[owner]] = 0
    nuclear[~keep[nuclear]] = 0
    return nuclear, owner, np.flatnonzero(keep)


def _keep_largest_parts(labels: np.ndarray) -> np.ndarray:
    """Keep each label's largest 4-connected piece."""
    for index, box in enumerate(ndi.find_objects(labels), start=1):
        if box is None:
            continue
        window = labels[box]
        pieces, n_pieces = ndi.label(window == index)
        if n_pieces > 1:
            sizes = np.bincount(pieces.ravel())
            sizes[0] = 0
            window[(pieces > 0) & (pieces != sizes.argmax())] = 0
    return labels


# --------------------------------------------------------------------------- #
# Matrix channels
# --------------------------------------------------------------------------- #
def _trace(rng, theta, allowed, seeds, lengths, step=0.75, wave=(0.25, 40.0), wander=0.03, spread=0.0):
    """Trace fibers along ``theta`` from ``seeds`` in both directions.

    Each fiber deviates from the field by its own angle (sd ``spread``) and
    crimps with its own amplitude and wavelength around ``wave``. Returns
    flat pixel indices of the centre lines, the fiber of every point and the
    arc position of every point.
    """
    shape = theta.shape
    n = len(seeds)
    bias = rng.normal(0, spread, n) if spread > 0 else np.zeros(n)
    amplitude = wave[0] * rng.uniform(0.5, 1.4, n)
    wavelength = wave[1] * rng.uniform(0.6, 1.5, n)
    indices, owners, positions = [], [], []
    for direction in (1.0, -1.0):
        y, x = seeds[:, 0].copy(), seeds[:, 1].copy()
        alive = np.ones(n, dtype=bool)
        heading = None
        offset = np.zeros(n)
        phase = rng.uniform(0, 2 * np.pi, n)
        half = lengths / 2
        for step_index in range(int(np.ceil(half.max() / step)) + 1):
            iy, ix = _pixel(y, shape[0]), _pixel(x, shape[1])
            inside = (y >= 0) & (y < shape[0]) & (x >= 0) & (x < shape[1])
            alive &= inside & allowed[iy, ix] & (step_index * step <= half)
            if not alive.any():
                break
            live = np.flatnonzero(alive)
            if direction > 0 or step_index > 0:
                indices.append(iy[live] * shape[1] + ix[live])
                owners.append(live)
                positions.append(np.full(len(live), direction * step_index * step))
            offset += -0.05 * offset + rng.normal(0, wander, n)
            angle = theta[iy, ix] + bias + offset + amplitude * np.sin(2 * np.pi * step_index * step / wavelength + phase)
            dy, dx = np.sin(angle), np.cos(angle)
            if heading is None:
                dy, dx = direction * dy, direction * dx
            else:
                flip = dy * heading[0] + dx * heading[1] < 0
                dy, dx = np.where(flip, -dy, dy), np.where(flip, -dx, dx)
            heading = (dy, dx)
            y = y + step * dy
            x = x + step * dx
    if not indices:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    return np.concatenate(indices), np.concatenate(owners), np.concatenate(positions)


def _render(shape, traced, weights, sigmas, step=0.75, beads=None, rng=None, gaps=0.0) -> np.ndarray:
    """Draw traced fibers: per-fiber peak intensity ``weights`` and blur ``sigmas``.

    With ``rng``, brightness also drifts along each fiber, and where it
    drops below ``gaps`` the fiber breaks.
    """
    flat, owner, position = traced
    out = np.zeros(shape, dtype=np.float32)
    if not len(flat):
        return out
    point_weight = weights[owner]
    if rng is not None:
        n = len(weights)
        drift = 0.0
        for wavelength in (rng.uniform(50, 110, n), rng.uniform(140, 300, n)):
            drift = drift + np.sin(2 * np.pi * position / wavelength[owner] + rng.uniform(0, 2 * np.pi, n)[owner])
        along = np.clip(0.62 + 0.30 * drift, 0, None)
        along[along < gaps] = 0
        point_weight = point_weight * along
    if beads is not None:
        point_weight = point_weight * (0.25 + 0.75 * np.sin(2 * np.pi * position / beads) ** 2)
    for sigma in np.unique(sigmas):
        sel = sigmas[owner] == sigma
        if not sel.any():
            continue
        acc = np.bincount(flat[sel], weights=point_weight[sel] * np.sqrt(2 * np.pi) * sigma * step, minlength=shape[0] * shape[1])
        out += ndi.gaussian_filter(acc.reshape(shape).astype(np.float32), sigma)
    return out


def _fiber_seeds(rng, allowed, n, theta=None, bundle_fraction=0.0):
    rows, cols = np.nonzero(allowed)
    if not len(rows) or n <= 0:
        return np.zeros((0, 2))
    pick = rng.integers(0, len(rows), n)
    seeds = np.column_stack([rows[pick], cols[pick]]).astype(float) + rng.uniform(-0.5, 0.5, (n, 2))
    if theta is None or bundle_fraction <= 0:
        return seeds
    leaders = seeds[rng.random(n) < bundle_fraction]
    companions = []
    for y, x in leaders:
        angle = theta[int(y), int(x)]
        normal = np.array([np.cos(angle), -np.sin(angle)])
        for k in range(int(rng.integers(1, 4))):
            companions.append((y, x) + normal * rng.choice([-1, 1]) * (k + 1) * rng.uniform(3.5, 7.0))
    if companions:
        seeds = np.concatenate([seeds, np.asarray(companions)])
    return seeds


def _blobs(rng, shape, allowed, n, radius_range, texture) -> np.ndarray:
    """Filled, irregular, mottled patches (collagen bundles cut across)."""
    out = np.zeros(shape, dtype=np.float32)
    rows, cols = np.nonzero(allowed)
    if not len(rows):
        return out
    for _ in range(n):
        k = rng.integers(len(rows))
        r = rng.uniform(*radius_range)
        cy, cx = rows[k], cols[k]
        reach = int(r * 1.6) + 3
        y0, y1 = max(0, cy - reach), min(shape[0], cy + reach + 1)
        x0, x1 = max(0, cx - reach), min(shape[1], cx + reach + 1)
        local = _ellipse_mask((y1 - y0, x1 - x0), (cy - y0, cx - x0), (r, r * rng.uniform(0.6, 1.0)), rng.uniform(0, np.pi), _harmonics(rng, 0.22))
        out[y0:y1, x0:x1] = np.maximum(out[y0:y1, x0:x1], local * rng.lognormal(np.log(1.0), 0.25))
    out = ndi.gaussian_filter(out, 1.6)
    return out * np.clip(0.75 + 0.35 * texture, 0.2, 1.4)


def _matrix_channels(rng: np.random.Generator, layout: _Layout, group: str, fibroblast_cells: np.ndarray) -> dict[str, np.ndarray]:
    shape = layout.tissue.shape
    area_scale = (shape[0] / 1500.0) ** 2
    fiber_space = layout.tissue & (layout.d_nest > 3) & (layout.d_lumen > 15)
    soft = ndi.gaussian_filter(fiber_space.astype(np.float32), 1.5)
    stroma_area = float(fiber_space.sum())
    theta = layout.theta

    # Collagen fibers shared (in part) by COL1, COL12, COL6 and FN.
    spread = _GROUP_SETTINGS[group]["fiber_spread"]
    seeds = _fiber_seeds(rng, fiber_space & (layout.d_nest > 6), int(stroma_area / 1500), theta, bundle_fraction=0.4)
    n = len(seeds)
    lengths = np.clip(rng.lognormal(np.log(170), 0.55, n), 40, 650)
    traced = _trace(rng, theta, fiber_space, seeds, lengths, spread=spread)
    sigmas = rng.choice(np.array([1.0, 1.5, 2.2]), n, p=[0.35, 0.40, 0.25])
    intensity = rng.lognormal(np.log(6000), 0.5, n)
    in_col1 = rng.random(n) < 0.9
    in_col12 = rng.random(n) < 0.55
    in_col6 = rng.random(n) < 0.35
    in_fn = rng.random(n) < 0.25
    texture = _smooth_noise(rng, shape, 3)
    haze = np.clip(_smooth_noise(rng, shape, 22), 0, None)
    n_patches = max(1, int(round(rng.integers(3, 8) * area_scale)))
    patches = _blobs(rng, shape, fiber_space & (layout.d_nest > 40) & (layout.d_lumen > 40), n_patches, (12, 30), texture)
    patch_level = rng.lognormal(np.log(9000), 0.2)

    col1 = _render(shape, traced, intensity * in_col1, sigmas, rng=rng, gaps=0.12)
    col1 += 0.22 * ndi.gaussian_filter(col1, 7) + patch_level * patches + 450 * haze
    col12 = _render(shape, traced, 0.7 * intensity * in_col12, sigmas, rng=rng, gaps=0.2) + 0.5 * patch_level * patches * (rng.random() < 0.6)

    own = _fiber_seeds(rng, fiber_space, int(stroma_area / 3500), theta, bundle_fraction=0.2)
    traced_own = _trace(rng, theta, fiber_space, own, np.clip(rng.lognormal(np.log(110), 0.5, len(own)), 30, 400), spread=spread + 0.2)
    col12 += _render(shape, traced_own, rng.lognormal(np.log(3500), 0.4, len(own)), rng.choice(np.array([0.9, 1.3]), len(own)), rng=rng, gaps=0.2)
    col12 = (col12 + 300 * np.clip(_smooth_noise(rng, shape, 30), 0, None)) * (1 + 0.8 * np.exp(-layout.d_nest / 50))

    # COL6: beaded microfibrils, part of the collagen bundles, pericellular rings round fibroblasts.
    meshy = _orientation_jitter(rng, theta, 0.9)
    mesh = _fiber_seeds(rng, fiber_space, int(stroma_area / 1300))
    traced_mesh = _trace(rng, meshy, fiber_space, mesh, np.clip(rng.lognormal(np.log(55), 0.5, len(mesh)), 15, 200), wave=(0.4, 25.0), wander=0.06, spread=0.9)
    col6 = _render(shape, traced, 0.4 * intensity * in_col6, sigmas, rng=rng, gaps=0.3)
    col6 += _render(shape, traced_mesh, rng.lognormal(np.log(2600), 0.4, len(mesh)), np.full(len(mesh), 0.85), beads=6.0)
    if fibroblast_cells.any():
        ring = ndi.binary_dilation(fibroblast_cells, iterations=4) & ~fibroblast_cells
        col6 += 1400 * ndi.gaussian_filter(ring.astype(np.float32), 1.2) * np.clip(1 + 0.5 * texture, 0, None)
    col6 += 0.25 * patch_level * patches + 250 * np.clip(_smooth_noise(rng, shape, 15), 0, None)

    # FN: fine fibrils, puncta, some co-distribution with collagen.
    finer = _orientation_jitter(rng, theta, 0.6)
    fibrils = _fiber_seeds(rng, fiber_space, int(stroma_area / 650))
    traced_fn = _trace(rng, finer, fiber_space, fibrils, np.clip(rng.lognormal(np.log(60), 0.5, len(fibrils)), 15, 250), wave=(0.35, 22.0), wander=0.05, spread=0.5)
    fn = _render(shape, traced_fn, rng.lognormal(np.log(2600), 0.45, len(fibrils)), rng.choice(np.array([0.7, 0.9]), len(fibrils)), rng=rng, gaps=0.25)
    fn += _render(shape, traced, 0.35 * intensity * in_fn, sigmas, rng=rng, gaps=0.3)
    dots = _fiber_seeds(rng, fiber_space, int(stroma_area / 450))
    acc = np.zeros(shape[0] * shape[1])
    np.add.at(acc, _pixel(dots[:, 0], shape[0]) * shape[1] + _pixel(dots[:, 1], shape[1]), rng.lognormal(np.log(3500), 0.5, len(dots)) * 2 * np.pi * 1.1**2)
    fn += ndi.gaussian_filter(acc.reshape(shape).astype(np.float32), 1.1)
    fn += 350 * np.clip(_smooth_noise(rng, shape, 18), 0, None)

    # COL4: basement membrane round nests and vessels, patchy.
    edge = layout.nest & ~ndi.binary_erosion(layout.nest, iterations=2)
    vessel_wall = (layout.d_lumen >= 11) & (layout.d_lumen <= 14) & layout.tissue
    patchy = np.clip(0.55 + 0.45 * _smooth_noise(rng, shape, 30), 0.05, 1.2)
    col4 = 6500 * ndi.gaussian_filter((edge * patchy).astype(np.float32), 0.9)
    col4 += 8000 * ndi.gaussian_filter((vessel_wall * np.clip(patchy + 0.3, 0.2, 1.3)).astype(np.float32), 0.9)
    col4 += 60 * np.clip(_smooth_noise(rng, shape, 40), 0, None) * layout.tissue

    return {
        "COL1": col1 * soft,
        "COL12": col12 * soft,
        "COL4": col4 * (layout.tissue & ~layout.lumen),
        "COL6": col6 * soft,
        "FN": fn * soft,
    }


def _orientation_jitter(rng, theta, spread) -> np.ndarray:
    return (theta + spread * _smooth_noise(rng, theta.shape, 40)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Cell channels and imaging
# --------------------------------------------------------------------------- #
def _cell_levels(rng, kinds, wnt5a, nest_factor, group) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Per-cell level of each cell marker and DAPI, and which cells are positive (index 0 = background)."""
    n = len(kinds)
    levels, positives = {}, {}
    sma_fraction = _GROUP_SETTINGS[group]["sma_fraction"]
    for marker in _CELL_MARKERS:
        value = rng.lognormal(np.log(50), 0.5, n)
        is_positive = np.zeros(n, dtype=bool)
        for kind, table in _EXPRESSION.items():
            if marker not in table:
                continue
            probability, median = table[marker]
            if kind == FIBROBLAST and marker == "SMA":
                probability = sma_fraction
            members = np.flatnonzero(kinds == kind)
            positive = members[rng.random(len(members)) < probability]
            value[positive] = rng.lognormal(np.log(median), 0.4, len(positive))
            is_positive[positive] = True
        levels[marker], positives[marker] = value, is_positive
    levels["PCK"][kinds == TUMOUR] *= nest_factor[kinds == TUMOUR]
    positive = np.flatnonzero(wnt5a)
    levels["WNT5A"][positive] = rng.lognormal(np.log(5000), 0.35, len(positive))
    positives["WNT5A"][positive] = True
    dapi = rng.lognormal(np.log(9000), 0.25, n)
    dapi[kinds == TUMOUR] *= 1.25
    dapi[kinds == TCELL] *= 1.3
    levels["DAPI_INIT"] = dapi
    levels["DAPI_FINAL"] = dapi * rng.normal(0.76, 0.04, n)
    levels = {marker: np.concatenate([[0.0], value]).astype(np.float32) for marker, value in levels.items()}
    positives = {marker: np.concatenate([[False], value]) for marker, value in positives.items()}
    return levels, positives


def _cell_channels(rng, cell_mask, nuclear_mask, levels, positives, layout) -> dict[str, np.ndarray]:
    from skimage.segmentation import find_boundaries

    shape = cell_mask.shape
    inside = cell_mask > 0
    nucleus = nuclear_mask > 0
    cyto = inside & ~nucleus
    edges = find_boundaries(cell_mask, mode="inner")
    membrane = (np.exp(-(ndi.distance_transform_edt(~edges) ** 2) / (2 * 1.3**2)) * inside).astype(np.float32)
    textures = [np.clip(1 + 0.25 * _fine_noise(rng, shape, 1.0), 0.2, None) for _ in range(3)]
    granular = np.clip(1 + 0.9 * _fine_noise(rng, shape, 0.8), 0.05, None)
    profiles = {
        "membrane": 0.95 * membrane + 0.30 * cyto + 0.06 * nucleus,
        "cytoplasm": cyto * (0.85 + 0.20 * membrane) + 0.10 * nucleus,
        "keratin": cyto * (0.72 + 0.40 * membrane) + 0.07 * nucleus,
    }
    channels = {}
    for index, marker in enumerate(_CELL_MARKERS):
        if marker in _MEMBRANE_MARKERS:
            profile = profiles["membrane"]
        elif marker == "PCK":
            profile = profiles["keratin"]
        else:
            profile = profiles["cytoplasm"]
        # Specific stain follows the membrane or cytoplasm; non-specific background is flat.
        profile = np.where(positives[marker][cell_mask], profile, inside)
        signal = levels[marker][cell_mask] * profile * textures[index % 3]
        if marker in _GRANULAR_MARKERS:
            signal = signal * granular
        channels[marker] = signal.astype(np.float32)
    # Secreted WNT5A leaves a faint halo; larger vessels have a smooth-muscle (SMA) coat.
    channels["WNT5A"] += 0.18 * ndi.gaussian_filter(channels["WNT5A"], 5)
    coat = np.zeros(shape, dtype=np.float32)
    for vessel in layout.vessels:
        if vessel["axes"][0] >= 17:
            ring = _ellipse_mask(shape, vessel["centre"], (vessel["axes"][0] + 22, vessel["axes"][1] + 22), vessel["angle"])
            coat += ring & (layout.d_lumen >= 14)
    if coat.any():
        channels["SMA"] += 4500 * ndi.gaussian_filter(coat * np.clip(1 + 0.4 * textures[0], 0, None), 1.2) * ~layout.lumen

    chromatin = np.clip(1 + 0.32 * _fine_noise(rng, shape, 1.1), 0.2, None)
    rim = ndi.distance_transform_edt(nucleus)
    dome = np.clip(0.75 + 0.06 * np.minimum(rim, 4), 0, 1.0)
    dapi_profile = (nucleus * dome * chromatin).astype(np.float32)
    channels["DAPI_INIT"] = levels["DAPI_INIT"][cell_mask] * dapi_profile
    channels["DAPI_FINAL"] = levels["DAPI_FINAL"][cell_mask] * dapi_profile * np.clip(1 + 0.08 * textures[1], 0.5, None)
    return channels


def _image(rng, signals: dict[str, np.ndarray], layout: _Layout) -> np.ndarray:
    """Add autofluorescence, optical blur and noise; return the CYX uint16 stack."""
    shape = layout.tissue.shape
    tissue = ndi.gaussian_filter(layout.tissue.astype(np.float32), 2.0)
    variation = np.clip(1 + 0.3 * _smooth_noise(rng, shape, 80), 0.4, None)
    stack = np.zeros((len(PANEL), *shape), dtype=np.uint16)
    collagen_af = signals["COL1"]
    background = tissue * variation * np.where(layout.lumen, np.float32(0.4), np.float32(1.0))
    for index, marker in enumerate(PANEL):
        offset = rng.uniform(25, 60)
        autofluorescence = np.float32(rng.uniform(70, 220))
        signal = signals[marker] + autofluorescence * background
        if marker.startswith("DAPI"):
            signal = signal + 0.03 * collagen_af
        signal = ndi.gaussian_filter(signal, 1.0) + offset
        noisy = signal + rng.standard_normal(shape, dtype=np.float32) * np.sqrt(1.5 * signal + 16.0)
        stack[index] = np.clip(np.rint(noisy), 0, 65535).astype(np.uint16)
    return stack


# --------------------------------------------------------------------------- #
# Measurement and writing
# --------------------------------------------------------------------------- #
def _measure(stack, cell_mask, nuclear_mask, matched, sample) -> dict[str, pd.DataFrame]:
    """Cell tables measured from the written image and masks."""
    from skimage.measure import regionprops_table

    props = pd.DataFrame(regionprops_table(
        cell_mask,
        properties=("label", "centroid", "eccentricity", "axis_major_length", "axis_minor_length", "perimeter", "solidity", "area"),
    ))
    labels = props["label"].to_numpy()
    nuclei = matched[labels]
    cell_size = props["area"].to_numpy(dtype=float)
    nuclear_size = np.bincount(nuclear_mask.ravel(), minlength=int(nuclear_mask.max()) + 1)[nuclei].astype(float)
    cell_flat, nuclear_flat = cell_mask.ravel(), nuclear_mask.ravel()
    n_cells, n_nuclei = int(cell_mask.max()) + 1, int(nuclear_mask.max()) + 1

    raw = {}
    for index, marker in enumerate(PANEL):
        plane = stack[index].ravel()
        raw[marker] = np.bincount(cell_flat, weights=plane, minlength=n_cells)[labels]
        raw[f"{marker}_nuclear"] = np.bincount(nuclear_flat, weights=plane, minlength=n_nuclei)[nuclei]
    morphology = pd.DataFrame({
        "label": labels,
        "centroid_y": props["centroid-0"],
        "centroid_x": props["centroid-1"],
        "eccentricity": props["eccentricity"],
        "major_axis_length_px": props["axis_major_length"],
        "minor_axis_length_px": props["axis_minor_length"],
        "perimeter_px": props["perimeter"],
        "solidity": props["solidity"],
        "cell_size": cell_size,
        "cell_area_px2": cell_size,
        "passes_size_qc": (cell_size >= _SIZE_QC_RANGE_PX[0]) & (cell_size <= _SIZE_QC_RANGE_PX[1]),
        "matched_nuclear_label": nuclei,
        "nuclear_size": nuclear_size,
        "nuclear_to_cell_area_ratio": nuclear_size / cell_size,
        "sample": sample,
        "mask_type": "whole_cell",
    })
    tables = {}
    for kind in _TABLE_KINDS:
        values = {}
        for marker in PANEL:
            whole, nuclear = raw[marker], raw[f"{marker}_nuclear"]
            if kind != "raw":
                whole, nuclear = whole / cell_size, nuclear / nuclear_size
            if kind == "arcsinh_transformed":
                whole, nuclear = np.arcsinh(whole / _ARCSINH_COFACTOR), np.arcsinh(nuclear / _ARCSINH_COFACTOR)
            values[marker], values[f"{marker}_nuclear"] = whole, nuclear
        tables[kind] = pd.concat([pd.DataFrame(values), morphology], axis=1)[_table_columns()]
    return tables


def _table_columns() -> list[str]:
    columns = []
    for marker in PANEL:
        columns += [marker, f"{marker}_nuclear"]
    return columns + list(_MORPHOLOGY_COLUMNS)


def _downsample(stack: np.ndarray) -> np.ndarray:
    channels, height, width = stack.shape
    height, width = height // 2 * 2, width // 2 * 2
    blocks = stack[:, :height, :width].reshape(channels, height // 2, 2, width // 2, 2)
    return np.rint(blocks.mean(axis=(2, 4))).astype(stack.dtype)


def _write_ome(path: Path, stack: np.ndarray, name: str) -> None:
    import tifffile

    metadata = {
        "axes": "CYX",
        "Name": name,
        "Channel": {"Name": list(PANEL)},
        "PhysicalSizeX": PIXEL_SIZE_UM,
        "PhysicalSizeXUnit": "µm",
        "PhysicalSizeY": PIXEL_SIZE_UM,
        "PhysicalSizeYUnit": "µm",
    }
    pixels_per_cm = 1e4 / PIXEL_SIZE_UM
    levels = [stack]
    for _ in range(2):
        levels.append(_downsample(levels[-1]))
    options = {"compression": "zlib", "predictor": True, "photometric": "minisblack", "resolutionunit": "CENTIMETER"}
    with tifffile.TiffWriter(path, ome=True, bigtiff=stack.nbytes > 2**31) as tif:
        tif.write(stack, subifds=len(levels) - 1, tile=(512, 512), metadata=metadata, resolution=(pixels_per_cm, pixels_per_cm), **options)
        for level, data in enumerate(levels[1:], start=1):
            scale = pixels_per_cm / 2**level
            tif.write(data, subfiletype=1, tile=(256, 256), resolution=(scale, scale), **options)


def _write_core(destination: Path, name: str, group: str, size: int, rng: np.random.Generator, with_classes: bool) -> dict:
    import tifffile

    layout = _layout(rng, size, group)
    cells = _place_cells(rng, layout, group)
    nuclear, owner, kept = _rasterize(rng, cells, layout)

    # Final numbering: cells 1..N in reading order, nuclei a random permutation.
    order = kept[np.lexsort((cells.x[kept - 1], cells.y[kept - 1]))]
    n = len(order)
    cell_label = np.zeros(len(cells.kind) + 1, dtype=np.uint32)
    cell_label[order] = np.arange(1, n + 1, dtype=np.uint32)
    nuclear_label = np.zeros_like(cell_label)
    nuclear_label[order] = rng.permutation(n).astype(np.uint32) + 1
    cell_mask, nuclear_mask = cell_label[owner], nuclear_label[nuclear]
    matched = np.zeros(n + 1, dtype=np.int64)
    matched[cell_label[order]] = nuclear_label[order]
    kinds = cells.kind[order - 1]
    wnt5a = cells.wnt5a[order - 1]

    nest_labels, n_nests = ndi.label(layout.nest)
    nest_of_cell = nest_labels[_pixel(cells.y[order - 1], size), _pixel(cells.x[order - 1], size)]
    nest_factor = rng.uniform(0.7, 1.3, n_nests + 1)[nest_of_cell]
    levels, positives = _cell_levels(rng, kinds, wnt5a, nest_factor, group)
    signals = _cell_channels(rng, cell_mask, nuclear_mask, levels, positives, layout)
    signals.update(_matrix_channels(rng, layout, group, np.isin(cell_mask, np.flatnonzero(kinds == FIBROBLAST) + 1)))
    stack = _image(rng, signals, layout)
    del signals
    tables = _measure(stack, cell_mask, nuclear_mask, matched, name)

    final = destination / name
    work = destination / f".{name}.partial"
    if work.exists():
        shutil.rmtree(work)
    try:
        for folder in ("background", "segmentation", "quantification"):
            (work / folder).mkdir(parents=True)
        _write_ome(work / "background" / f"{name}.ome.tiff", stack, name)
        tifffile.imwrite(work / "segmentation" / f"{name}_whole_cell.tiff", cell_mask, compression="zlib")
        tifffile.imwrite(work / "segmentation" / f"{name}_nuclear.tiff", nuclear_mask, compression="zlib")
        for kind, table in tables.items():
            table.to_csv(work / "quantification" / f"cell_table_{kind}.csv", index=False)
        pd.DataFrame({
            "channel_index": np.arange(len(PANEL)), "marker_name": PANEL, "quantification_column": PANEL,
        }).to_csv(work / "quantification" / "channel_map.csv", index=False)
        classes = np.array(_TYPE_NAMES, dtype=object)[kinds]
        classes[wnt5a & (kinds == FIBROBLAST)] = "Fibroblast: WNT5A+"
        if with_classes:
            raw = tables["raw"]
            (work / "qupath").mkdir()
            pd.DataFrame({
                "image": name,
                "cell_id": [f"{name}_{label}" for label in raw["label"]],
                "qupath_class": classes[raw["label"].to_numpy() - 1],
                "centroid_x_px": raw["centroid_x"] + 0.5,
                "centroid_y_px": raw["centroid_y"] + 0.5,
            }).to_csv(work / "qupath" / f"{name}_cell_classes.csv", index=False)
            (work / "qupath" / f"{name}_annotations.geojson").write_text(
                json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8"
            )
        if final.exists():
            shutil.rmtree(final)
        work.rename(final)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    counts = pd.Series(classes).value_counts()
    return {"name": name, "group": group, "cells": n, "classes": {key: int(counts.get(key, 0)) for key in CELL_CLASSES}}


def _write_readme(destination: Path, summaries: list[dict], size: int, seed: int, with_classes: bool) -> None:
    lines = [
        "SpatioEv demo data",
        "==================",
        "",
        "This folder was generated by `spatioev demo`. Every image, mask and table in it is",
        "synthetic: it was drawn by a computer program and contains no real patient data.",
        "",
        f"It imitates a small tissue microarray (TMA) stained with 17 markers ({', '.join(PANEL)}),",
        f"imaged at {PIXEL_SIZE_UM} micrometres per pixel ({size} x {size} pixels per core).",
        "Each core has tumour nests (PCK) in a stroma with fibroblasts (CD90, PDPN, SMA, VIM,",
        "some WNT5A), macrophages (CD68, CD45), T cells (CD3, CD45), small vessels (CD31) and",
        "five matrix stains (COL1, COL12, COL4, COL6, FN).",
        "",
        "Cores and groups (also in demo_sample_sheet.csv):",
    ]
    for summary in summaries:
        lines.append(f"  {summary['name']}  group {summary['group']}  {summary['cells']:,} cells")
    lines += [
        "",
        "Group B was made different from group A on purpose, so the cohort comparison has",
        "something to find: its collagen fibers are more aligned, it has more WNT5A+",
        "fibroblasts, and its macrophages sit closer to fibroblasts.",
        "",
        "Each core folder holds:",
        "  background/<core>.ome.tiff          the multi-channel image",
        "  segmentation/<core>_whole_cell.tiff  cell outlines (one number per cell)",
        "  segmentation/<core>_nuclear.tiff     nucleus outlines",
        "  quantification/cell_table_*.csv      one row per cell with its marker levels",
    ]
    if with_classes:
        lines.append("  qupath/<core>_cell_classes.csv      cell types, as QuPath would export them")
    lines += ["", f"Regenerate the same data with seed {seed}.", ""]
    (destination / "README.txt").write_text("\n".join(lines), encoding="utf-8")
