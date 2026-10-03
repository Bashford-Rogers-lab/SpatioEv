"""ECM fiber segmentation and per-fiber measurements.

Adapted from ark-analysis ``ark.segmentation.fiber_segmentation``
(https://github.com/angelolab/ark-analysis), MIT License,
Copyright (c) 2023 Angelo Lab. The licence text ships beside this module as
``LICENSE.ark-analysis``.

The segmentation steps and their defaults are ark's: Gaussian blur, local
contrast enhancement (CLAHE), a Frangi ridge filter, a distance transform of
the ridge mask, three-class multi-Otsu watershed markers on a Sobel elevation
map, then removal of small objects. On an image that fits in one tile,
:func:`segment_fibers` reproduces ark's label image exactly.

What differs from upstream:

- Inputs are arrays rather than ark's folder-per-FOV layout, so there is no
  ``alpineer``/``xarray`` dependency.
- :func:`segment_fibers_tiled` handles whole-slide images. ark normalises
  intensity, sets the Frangi ``gamma`` and picks the Otsu thresholds from the
  whole image; computed per tile those would drift (an empty tile gets a tiny
  ``gamma`` and turns noise into ridges), so the tiled path computes all three
  once over the full image and applies them to every tile.
- :func:`calculate_fiber_alignment` uses a KD-tree instead of an all-pairs
  distance matrix, which does not fit in memory for a whole slide.
- ``include_bright`` (off by default, so the ark path stays exact) adds the
  bright matrix the ridge filter leaves out: cross-cut bundles and dense
  patches, which Frangi suppresses as blob-like.
- Orientation differences are axial by default. ark subtracts raw
  ``regionprops`` angles, so two nearly horizontal fibers at +89 and -89
  degrees score as maximally misaligned. ``axial=False`` restores ark's score.
- The fiber table follows SpatioEv conventions: ``X_centroid`` is the column,
  ``Y_centroid`` the row, ``orientation`` is in degrees from the +x axis in
  ``[0, 180)``, and each row is indexed by a unique ``fiber_id`` so it can be
  passed straight to :mod:`spatioev.tl.ecm`.
"""

from __future__ import annotations

import math
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage as ndi

__all__ = [
    "FIBER_PROPERTIES",
    "bright_matrix_cuts",
    "bright_matrix_mask",
    "calculate_fiber_alignment",
    "fiber_table_from_labels",
    "resolve_clahe_kernel_size",
    "segment_fibers",
    "segment_fibers_tiled",
]

#: ``regionprops`` properties measured for every fiber (ark's ``FIBER_OBJECT_PROPS``).
FIBER_PROPERTIES = (
    "label",
    "centroid",
    "major_axis_length",
    "minor_axis_length",
    "orientation",
    "area",
    "eccentricity",
    "euler_number",
)

_DEFAULT_WIDTHS = (1, 3, 5, 7, 9)  # ark: range(1, 10, 2)


def resolve_clahe_kernel_size(
    image_shape: tuple[int, int],
    contrast_scaling_divisor: float = 128,
    clahe_kernel_size: int | None = None,
) -> int:
    """Return the CLAHE window in pixels.

    ark sets the window to ``image rows / contrast_scaling_divisor``. That ties
    the window to the image size, so a whole slide would get a far larger
    window than a TMA core at the same resolution. Pass ``clahe_kernel_size``
    explicitly to keep the window fixed in pixels across images.
    """
    if clahe_kernel_size is not None:
        return max(1, int(clahe_kernel_size))
    return max(1, int(image_shape[0] / contrast_scaling_divisor))


def _remove_small_labels(labeled: np.ndarray, min_size: int) -> np.ndarray:
    """Zero labels smaller than ``min_size`` pixels.

    Equivalent to ark's ``remove_small_objects(labeled, min_size=...)`` but
    independent of scikit-image's deprecation of ``min_size`` in 0.26.
    """
    if min_size <= 1 or labeled.max() == 0:
        return labeled
    sizes = np.bincount(labeled.ravel())
    small = sizes < min_size
    small[0] = False
    if small.any():
        labeled = labeled.copy()
        labeled[small[labeled]] = 0
    return labeled


def _empty_steps(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    zeros = np.zeros(shape, dtype=float)
    return {
        "raw": zeros,
        "blurred": zeros,
        "contrast_adjusted": zeros,
        "ridges": zeros,
        "distance_transformed": zeros,
        "thresholded": zeros,
        "elevation_map": zeros,
        "unfiltered_labels": np.zeros(shape, dtype=np.int32),
        "labels": np.zeros(shape, dtype=np.int32),
    }


def segment_fibers(
    image: np.ndarray,
    blur: float = 2,
    contrast_scaling_divisor: float = 128,
    fiber_widths: Iterable[float] = _DEFAULT_WIDTHS,
    ridge_cutoff: float = 0.1,
    sobel_blur: float = 1,
    min_fiber_size: int = 15,
    clahe_kernel_size: int | None = None,
    include_bright: bool = False,
    bright_local_window: int = 0,
    bright_floor: float = 0.25,
    bright_cuts: tuple[float, float] | None = None,
    return_steps: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    """Segment fiber objects from one 2-D matrix channel.

    Parameters
    ----------
    image : ndarray
        One channel (rows x columns), for example COL1.
    blur : float
        Gaussian blur sigma applied before contrast enhancement.
    contrast_scaling_divisor : float
        ark's CLAHE control: the window is ``rows / contrast_scaling_divisor``
        pixels unless ``clahe_kernel_size`` is given.
    fiber_widths : iterable of float
        Frangi scales in pixels. Larger widths can merge close, narrow
        branches into one thicker fiber.
    ridge_cutoff : float
        Ridge inclusion threshold after Frangi filtering (Frangi output is
        multiplied by 10,000 first, as in ark).
    sobel_blur : float
        Gaussian sigma for the Sobel elevation map used by the watershed.
    min_fiber_size : int
        Minimum fiber area in pixels.
    clahe_kernel_size : int, optional
        CLAHE window in pixels, overriding ``contrast_scaling_divisor``.
    include_bright : bool
        Also count matrix the ridge-based steps leave out (not in ark). The
        Frangi filter suppresses blob-like structures by design, so cross-cut
        bundles and dense patches are only outlined, and ark's watershed
        occasionally drops whole bright fibers. With this on, pixels no fiber
        covers are added as extra objects when they are in the image's
        brightest class (upper cut of a three-class multi-Otsu threshold of
        the blurred channel) or, with ``bright_local_window``, brighter than
        their neighbourhood.
    bright_local_window : int
        Side of the neighbourhood (pixels, made odd) for the local test; 0
        uses only the brightest class. Large patches of medium intensity next
        to dark gaps are only caught locally; a global cut low enough to catch
        them floods the haze between fibers elsewhere. About 65 um suited
        COL1 at 0.325 um/px.
    bright_floor : float
        For the local test, pixels must also exceed ``tissue cut + floor x
        (bright cut - tissue cut)``, so dim tissue next to empty glass is not
        counted.
    bright_cuts : (float, float), optional
        ``(tissue cut, bright cut)`` to use instead of computing them from this
        image (the tiled path passes the whole-slide cuts).
    return_steps : bool
        Also return every intermediate image (for parameter tuning plots).

    Returns
    -------
    labels : ndarray of int32
        Fiber label image, 0 = background. Labels above
        ``steps["n_ridge_objects"]`` are the added bright matrix.
    steps : dict, optional
        Intermediate images when ``return_steps`` is true.
    """
    labels, steps, _ = _segment_fibers(
        image, blur, contrast_scaling_divisor, fiber_widths, ridge_cutoff, sobel_blur, min_fiber_size,
        clahe_kernel_size, include_bright, bright_local_window, bright_floor, bright_cuts, return_steps,
    )
    return (labels, steps) if return_steps else labels


def _segment_fibers(
    image: np.ndarray,
    blur: float = 2,
    contrast_scaling_divisor: float = 128,
    fiber_widths: Iterable[float] = _DEFAULT_WIDTHS,
    ridge_cutoff: float = 0.1,
    sobel_blur: float = 1,
    min_fiber_size: int = 15,
    clahe_kernel_size: int | None = None,
    include_bright: bool = False,
    bright_local_window: int = 0,
    bright_floor: float = 0.25,
    bright_cuts: tuple[float, float] | None = None,
    return_steps: bool = False,
) -> tuple[np.ndarray, dict | None, int]:
    """Implementation of :func:`segment_fibers`; also returns the number of ridge objects."""
    from skimage.exposure import equalize_adapthist
    from skimage.filters import frangi, sobel, threshold_multiotsu
    from skimage.segmentation import watershed

    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(f"segment_fibers expects a 2-D channel, got shape {image.shape}")
    widths = list(fiber_widths)
    kernel = resolve_clahe_kernel_size(image.shape, contrast_scaling_divisor, clahe_kernel_size)

    raw = image.astype(float)
    blurred = ndi.gaussian_filter(raw, sigma=blur)
    peak = float(np.max(blurred)) if blurred.size else 0.0
    if not np.isfinite(peak) or peak <= 0:
        steps = _empty_steps(image.shape)
        return steps["labels"], (steps if return_steps else None), 0

    contrast_adjusted = equalize_adapthist(blurred / peak, kernel_size=kernel)
    ridges = frangi(contrast_adjusted, sigmas=widths, black_ridges=False) * 10000
    distance_transformed = ndi.gaussian_filter(
        ndi.distance_transform_edt(ridges > ridge_cutoff), sigma=1
    )

    try:
        thresholds = threshold_multiotsu(distance_transformed, classes=3)
    except ValueError:
        # Too few distinct values (no ridges at all); nothing to segment.
        steps = _empty_steps(image.shape)
        steps.update(raw=raw, blurred=blurred, contrast_adjusted=contrast_adjusted, ridges=ridges)
        return steps["labels"], (steps if return_steps else None), 0

    labels, threshed, elevation_map, unfiltered = _watershed_fibers(
        distance_transformed, thresholds, sobel_blur, min_fiber_size, sobel, watershed
    )
    n_ridge = int(labels.max())
    bright = None
    if include_bright:
        cuts = bright_cuts if bright_cuts is not None else bright_matrix_cuts(blurred)
        bright = bright_matrix_mask(blurred, cuts, bright_local_window, bright_floor)
        labels = _add_bright_matrix(labels, bright, min_fiber_size)
    if not return_steps:
        return labels, None, n_ridge
    return labels, {
        "raw": raw,
        "blurred": blurred,
        "contrast_adjusted": contrast_adjusted,
        "ridges": ridges,
        "distance_transformed": distance_transformed,
        "thresholded": threshed,
        "elevation_map": elevation_map,
        "unfiltered_labels": unfiltered,
        "bright": bright if bright is not None else np.zeros(image.shape, dtype=bool),
        "labels": labels,
        "n_ridge_objects": n_ridge,
    }, n_ridge


def bright_matrix_cuts(blurred: np.ndarray, sample_stride: int = 4) -> tuple[float, float]:
    """Three-class multi-Otsu cuts of the blurred channel: (background|tissue, tissue|bright).

    A few saturated spots can stretch the histogram so far that the
    brightest class holds only them. When that class is under 0.1% of the
    sample, the cuts are recomputed with the sample clipped at its 99.9th
    percentile.
    """
    from skimage.filters import threshold_multiotsu

    sample = np.asarray(blurred, dtype=float)[::sample_stride, ::sample_stride]
    try:
        low, high = threshold_multiotsu(sample, classes=3)
        ceiling = np.percentile(sample, 99.9)
        if high > ceiling:
            low, high = threshold_multiotsu(np.minimum(sample, ceiling), classes=3)
        return float(low), float(high)
    except ValueError:  # flat image
        return float(np.inf), float(np.inf)


def bright_matrix_mask(blurred: np.ndarray, cuts: tuple[float, float], local_window: int = 0, floor: float = 0.25) -> np.ndarray:
    """Pixels in the brightest class, or brighter than their neighbourhood and above the floor."""
    low, high = cuts
    mask = blurred > high
    if local_window and np.isfinite(low) and np.isfinite(high):
        from skimage.filters import threshold_local

        window = int(local_window) // 2 * 2 + 1
        local = threshold_local(np.asarray(blurred, dtype=float), block_size=window, method="gaussian")
        mask |= (blurred > local) & (blurred > low + floor * (high - low))
    return mask


def _add_bright_matrix(labels: np.ndarray, bright: np.ndarray, min_size: int) -> np.ndarray:
    """Add bright pixels no fiber covers as new objects, numbered after the fibers."""
    extra, n = ndi.label(bright & (labels == 0), structure=np.ones((3, 3)))
    if n == 0:
        return labels
    extra = _remove_small_labels(extra, min_size)
    out = labels.copy()
    added = extra > 0
    out[added] = extra[added] + int(labels.max())
    return out.astype(np.int32)


def _watershed_fibers(distance_transformed, thresholds, sobel_blur, min_fiber_size, sobel, watershed):
    threshed = np.zeros_like(distance_transformed)
    threshed[distance_transformed < thresholds[0]] = 1
    threshed[distance_transformed > thresholds[1]] = 2
    elevation_map = sobel(ndi.gaussian_filter(distance_transformed, sigma=sobel_blur))
    # ark casts the elevation map to int32 before the watershed; kept for parity.
    segmentation = watershed(elevation_map.astype(np.int32), threshed.astype(np.int32)) - 1
    unfiltered, _ = ndi.label(segmentation)
    labels = (_remove_small_labels(unfiltered, min_fiber_size) * segmentation).astype(np.int32)
    return labels, threshed, elevation_map, unfiltered.astype(np.int32)


# --------------------------------------------------------------------------- #
# Whole-slide (tiled) segmentation
# --------------------------------------------------------------------------- #
def _tile_grid(shape: tuple[int, int], core: int) -> list[tuple[int, int, int, int]]:
    rows, cols = shape
    return [
        (y0, min(y0 + core, rows), x0, min(x0 + core, cols))
        for y0 in range(0, rows, core)
        for x0 in range(0, cols, core)
    ]


def _padded(bounds, shape, overlap):
    y0, y1, x0, x1 = bounds
    return (
        max(0, y0 - overlap),
        min(shape[0], y1 + overlap),
        max(0, x0 - overlap),
        min(shape[1], x1 + overlap),
    )


def _hessian_norm_max(image: np.ndarray, sigma: float) -> float:
    """Largest Hessian Frobenius norm at ``sigma`` -- what Frangi uses for gamma."""
    from skimage.feature import hessian_matrix, hessian_matrix_eigvals

    eigvals = hessian_matrix_eigvals(
        hessian_matrix(-image, sigma, mode="reflect", use_gaussian_derivatives=True)
    )
    return float(np.sqrt((eigvals**2).sum(0)).max())


def segment_fibers_tiled(
    image,
    blur: float = 2,
    contrast_scaling_divisor: float = 128,
    fiber_widths: Iterable[float] = _DEFAULT_WIDTHS,
    ridge_cutoff: float = 0.1,
    sobel_blur: float = 1,
    min_fiber_size: int = 15,
    clahe_kernel_size: int | None = None,
    include_bright: bool = False,
    bright_local_window: int = 0,
    bright_floor: float = 0.25,
    tile_size: int = 2048,
    overlap: int = 128,
    imageid: str = "image",
    fiber_type: str = "fiber",
    labels_out=None,
    work_dir: str | Path | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Segment fibers in an image too large to filter in one pass.

    Images no larger than ``tile_size`` in both dimensions go through
    :func:`segment_fibers` unchanged, so their labels match ark exactly. Larger
    images are processed in tiles with ``overlap`` pixels of context on every
    side. Intensity normalisation, the Frangi ``gamma`` and the multi-Otsu
    thresholds are computed once over the whole image so every tile is
    segmented on the same scale. A fiber belongs to the tile that contains its
    centroid, so fibers crossing a seam are kept once and whole, provided they
    are shorter than ``overlap``.

    Parameters
    ----------
    image : array-like
        2-D channel supporting slicing, for example a NumPy array, a
        ``tifffile`` memmap or a zarr array; tiles are read on demand.
    tile_size : int
        Core tile edge in pixels. Rounded down to a multiple of the CLAHE
        window so the contrast grid lines up across tiles.
    overlap : int
        Context added around each tile, rounded up to a multiple of the CLAHE
        window.
    imageid, fiber_type : str
        Written into the returned fiber table.
    labels_out : array-like, optional
        Writable 2-D integer array (e.g. a ``numpy.memmap``) that receives the
        label image. Allocated in memory when omitted.
    work_dir : path, optional
        Folder for temporary float32 memmaps of intermediate images. A
        temporary directory is used when omitted.
    progress : callable, optional
        ``progress(fraction, message)`` callback.

    Returns
    -------
    labels : array-like
        Fiber labels, unique across the image.
    fiber_table : DataFrame
        One row per fiber; see :func:`fiber_table_from_labels`.
    """
    from skimage.exposure import equalize_adapthist
    from skimage.filters import frangi, sobel, threshold_multiotsu
    from skimage.measure import regionprops_table
    from skimage.segmentation import watershed

    shape = tuple(int(s) for s in image.shape)
    if len(shape) != 2:
        raise ValueError(f"segment_fibers_tiled expects a 2-D channel, got shape {shape}")
    widths = list(fiber_widths)
    kernel = resolve_clahe_kernel_size(shape, contrast_scaling_divisor, clahe_kernel_size)
    report = progress or (lambda fraction, message: None)

    if shape[0] <= tile_size and shape[1] <= tile_size:
        report(0.1, "Segmenting whole image")
        labels, _, n_ridge = _segment_fibers(
            np.asarray(image[:, :]), blur, contrast_scaling_divisor, widths, ridge_cutoff, sobel_blur,
            min_fiber_size, kernel, include_bright, bright_local_window, bright_floor, None, False,
        )
        if labels_out is not None:
            labels_out[:, :] = labels
            labels = labels_out
        report(1.0, "Segmentation complete")
        return labels, fiber_table_from_labels(np.asarray(labels), imageid=imageid, fiber_type=fiber_type, ridge_label_max=n_ridge)

    core = max(kernel, (int(tile_size) // kernel) * kernel)
    overlap = int(math.ceil(max(int(overlap), 3 * max(widths), 4 * blur, bright_local_window // 2 if include_bright else 0) / kernel) * kernel)
    tiles = _tile_grid(shape, core)
    n_tiles = len(tiles)

    with tempfile.TemporaryDirectory(dir=work_dir) as scratch:
        contrast = np.lib.format.open_memmap(Path(scratch) / "contrast.npy", "w+", np.float32, shape)
        distance = np.lib.format.open_memmap(Path(scratch) / "distance.npy", "w+", np.float32, shape)
        if labels_out is None:
            labels_out = np.zeros(shape, dtype=np.int32)

        # Pass 1: global intensity range of the blurred image (ark divides by its max),
        # and a sample of it for the bright-matrix threshold.
        low, high = np.inf, -np.inf
        samples = []
        for index, bounds in enumerate(tiles):
            py0, py1, px0, px1 = _padded(bounds, shape, overlap)
            y0, y1, x0, x1 = bounds
            blurred = ndi.gaussian_filter(np.asarray(image[py0:py1, px0:px1], dtype=float), sigma=blur)
            inner = blurred[y0 - py0 : y1 - py0, x0 - px0 : x1 - px0]
            low, high = min(low, float(inner.min())), max(high, float(inner.max()))
            if include_bright:
                samples.append(inner[::4, ::4].ravel())
            report(0.15 * (index + 1) / n_tiles, f"Intensity range, tile {index + 1}/{n_tiles}")
        if not np.isfinite(high) or high <= 0:
            labels_out[:, :] = 0
            return labels_out, fiber_table_from_labels(np.zeros((1, 1), np.int32), imageid, fiber_type)
        low = max(low, 0.0) / high
        cuts = bright_matrix_cuts(np.concatenate(samples)[None, :], sample_stride=1) if include_bright else None

        # Pass 2: CLAHE on a common intensity scale, and the global Frangi gamma.
        # equalize_adapthist rescales every call to its input's own range, so
        # a sentinel row holding the global minimum and maximum pins that range.
        gamma = 0.0
        for index, bounds in enumerate(tiles):
            py0, py1, px0, px1 = _padded(bounds, shape, overlap)
            y0, y1, x0, x1 = bounds
            blurred = ndi.gaussian_filter(np.asarray(image[py0:py1, px0:px1], dtype=float), sigma=blur)
            scaled = np.clip(blurred / high, low, 1.0)
            sentinel = np.full((1, scaled.shape[1]), low)
            sentinel[0, -1] = 1.0
            adjusted = equalize_adapthist(np.vstack([scaled, sentinel]), kernel_size=kernel)[:-1]
            inner = adjusted[y0 - py0 : y1 - py0, x0 - px0 : x1 - px0]
            contrast[y0:y1, x0:x1] = inner
            gamma = max(gamma, _hessian_norm_max(inner, widths[0]))
            report(0.15 + 0.25 * (index + 1) / n_tiles, f"Contrast, tile {index + 1}/{n_tiles}")
        gamma = gamma / 2 if gamma > 0 else 1.0

        # Pass 3: ridges and the distance map.
        d_low, d_high = np.inf, -np.inf
        for index, bounds in enumerate(tiles):
            py0, py1, px0, px1 = _padded(bounds, shape, overlap)
            y0, y1, x0, x1 = bounds
            block = np.asarray(contrast[py0:py1, px0:px1], dtype=float)
            ridges = frangi(block, sigmas=widths, black_ridges=False, gamma=gamma) * 10000
            dist = ndi.gaussian_filter(ndi.distance_transform_edt(ridges > ridge_cutoff), sigma=1)
            inner = dist[y0 - py0 : y1 - py0, x0 - px0 : x1 - px0]
            distance[y0:y1, x0:x1] = inner
            d_low, d_high = min(d_low, float(inner.min())), max(d_high, float(inner.max()))
            report(0.40 + 0.30 * (index + 1) / n_tiles, f"Ridges, tile {index + 1}/{n_tiles}")

        # Otsu thresholds from the same histogram scikit-image would build on
        # the whole image: 256 bins spanning the image's own min to max. Any
        # other binning shifts the thresholds and with them every fiber edge.
        if not d_high > d_low:
            labels_out[:, :] = 0
            return labels_out, fiber_table_from_labels(np.zeros((1, 1), np.int32), imageid, fiber_type)
        edges = np.histogram_bin_edges([d_low, d_high], bins=256, range=(d_low, d_high))
        histogram = np.zeros(256, dtype=np.int64)
        for y0, y1, x0, x1 in tiles:
            histogram += np.histogram(np.asarray(distance[y0:y1, x0:x1]), bins=edges)[0]
        centres = (edges[:-1] + edges[1:]) / 2
        try:
            thresholds = threshold_multiotsu(classes=3, hist=(histogram, centres))
        except ValueError:
            labels_out[:, :] = 0
            return labels_out, fiber_table_from_labels(np.zeros((1, 1), np.int32), imageid, fiber_type)

        # Pass 4: watershed per tile; keep fibers whose centroid is in the core.
        tables = []
        next_label = 1
        for index, bounds in enumerate(tiles):
            py0, py1, px0, px1 = _padded(bounds, shape, overlap)
            y0, y1, x0, x1 = bounds
            block = np.asarray(distance[py0:py1, px0:px1], dtype=float)
            local, *_ = _watershed_fibers(block, thresholds, sobel_blur, min_fiber_size, sobel, watershed)
            n_local_ridge = int(local.max())
            if include_bright:
                tile_blurred = ndi.gaussian_filter(np.asarray(image[py0:py1, px0:px1], dtype=float), sigma=blur)
                bright = bright_matrix_mask(tile_blurred, cuts, bright_local_window, bright_floor)
                local = _add_bright_matrix(local, bright, min_fiber_size)
            if local.max() > 0:
                props = pd.DataFrame(regionprops_table(local, properties=FIBER_PROPERTIES))
                props["object_type"] = np.where(props["label"] > n_local_ridge, "bright_matrix", "fiber")
                cy = props["centroid-0"] + py0
                cx = props["centroid-1"] + px0
                keep = (cy >= y0) & (cy < y1) & (cx >= x0) & (cx < x1)
                props = props.loc[keep].copy()
                if len(props):
                    mapping = np.zeros(int(local.max()) + 1, dtype=np.int64)
                    new_ids = np.arange(next_label, next_label + len(props))
                    mapping[props["label"].to_numpy()] = new_ids
                    relabelled = mapping[local]
                    region = labels_out[py0:py1, px0:px1]
                    painted = relabelled > 0
                    region[painted] = relabelled[painted]
                    labels_out[py0:py1, px0:px1] = region
                    props["label"] = new_ids
                    props["centroid-0"] += py0
                    props["centroid-1"] += px0
                    tables.append(props)
                    next_label += len(props)
            report(0.70 + 0.30 * (index + 1) / n_tiles, f"Watershed, tile {index + 1}/{n_tiles}")

    raw_table = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    return labels_out, _format_fiber_table(raw_table, imageid, fiber_type)


# --------------------------------------------------------------------------- #
# Fiber table
# --------------------------------------------------------------------------- #
def _format_fiber_table(props: pd.DataFrame, imageid: str, fiber_type: str) -> pd.DataFrame:
    columns = [
        "fiber_id", "imageid", "fiber_type", "object_type", "label", "X_centroid", "Y_centroid", "area",
        "major_axis_length", "minor_axis_length", "eccentricity", "euler_number",
        "orientation", "orientation_rowaxis_rad",
    ]
    if props.empty:
        return pd.DataFrame(columns=columns).set_index("fiber_id", drop=False)
    theta = props["orientation"].to_numpy(dtype=float)
    out = pd.DataFrame({
        "imageid": str(imageid),
        "fiber_type": str(fiber_type),
        "object_type": props["object_type"].to_numpy() if "object_type" in props else "fiber",
        "label": props["label"].to_numpy(dtype=np.int64),
        "X_centroid": props["centroid-1"].to_numpy(dtype=float),
        "Y_centroid": props["centroid-0"].to_numpy(dtype=float),
        "area": props["area"].to_numpy(dtype=float),
        "major_axis_length": props["major_axis_length"].to_numpy(dtype=float),
        "minor_axis_length": props["minor_axis_length"].to_numpy(dtype=float),
        "eccentricity": props["eccentricity"].to_numpy(dtype=float),
        "euler_number": props["euler_number"].to_numpy(dtype=float),
        # regionprops measures from the row axis; the major axis points along
        # (d_row, d_col) = (cos t, sin t), i.e. (dx, dy) = (sin t, cos t), whose
        # angle from +x is pi/2 - t. Axial, so fold into [0, 180).
        "orientation": np.mod(np.degrees(np.pi / 2 - theta), 180.0),
        "orientation_rowaxis_rad": theta,
    })
    out.insert(0, "fiber_id", [f"{imageid}__{fiber_type}__{label}" for label in out["label"]])
    return out.set_index("fiber_id", drop=False)


def fiber_table_from_labels(
    labels: np.ndarray,
    imageid: str = "image",
    fiber_type: str = "fiber",
    offset: tuple[float, float] = (0.0, 0.0),
    ridge_label_max: int | None = None,
) -> pd.DataFrame:
    """Measure every fiber in a label image.

    Columns: ``fiber_id`` (also the index), ``imageid``, ``fiber_type``,
    ``label``, ``X_centroid`` (column), ``Y_centroid`` (row), ``area``,
    ``major_axis_length``, ``minor_axis_length``, ``eccentricity``,
    ``euler_number``, ``orientation`` (degrees from +x, ``[0, 180)``) and
    ``orientation_rowaxis_rad`` (the raw ``regionprops`` angle ark uses).
    ``offset`` is ``(row, column)`` added to the centroids. ``object_type``
    is ``fiber``, or ``bright_matrix`` for labels above ``ridge_label_max``
    (the objects ``include_bright`` added).
    """
    from skimage.measure import regionprops_table

    labels = np.asarray(labels)
    if labels.max() == 0:
        return _format_fiber_table(pd.DataFrame(), imageid, fiber_type)
    props = pd.DataFrame(regionprops_table(labels, properties=FIBER_PROPERTIES))
    props["centroid-0"] += offset[0]
    props["centroid-1"] += offset[1]
    if ridge_label_max is not None:
        props["object_type"] = np.where(props["label"] > ridge_label_max, "bright_matrix", "fiber")
    return _format_fiber_table(props, imageid, fiber_type)


def calculate_fiber_alignment(
    fiber_df: pd.DataFrame,
    k: int = 4,
    axis_thresh: float = 2,
    axial: bool = True,
    image_key: str = "imageid",
    fiber_type_key: str | None = "fiber_type",
    angle_key: str = "orientation_rowaxis_rad",
    x_key: str = "X_centroid",
    y_key: str = "Y_centroid",
) -> pd.DataFrame:
    """Score how differently each fiber is oriented from its k nearest fibers.

    ark's score: ``sqrt(sum((neighbour_angle - angle)**2)) / k`` in radians,
    over the ``k`` nearest fibers whose length-to-width ratio is at least
    ``axis_thresh``. **Lower means more aligned.** Fibers below the ratio get
    NaN, as do objects added by ``include_bright`` (``object_type ==
    "bright_matrix"``), which are neither scored nor used as neighbours.
    Computed separately for each image and fiber type.

    Parameters
    ----------
    axial : bool
        Use the axial angle difference ``min(d, pi - d)``. Orientations are
        undirected, so +89 and -89 degrees are 2 degrees apart; ark's raw
        difference calls them 178 apart, which makes horizontal fibers look
        misaligned. ``False`` reproduces ark.
    """
    from scipy.spatial import cKDTree

    out = fiber_df.copy()
    out["alignment_score"] = np.nan
    if out.empty:
        return out
    keys = [image_key] + ([fiber_type_key] if fiber_type_key and fiber_type_key in out else [])
    ratio = out["major_axis_length"].to_numpy(float) / out["minor_axis_length"].to_numpy(float)
    eligible = pd.Series(ratio >= axis_thresh, index=out.index)
    if "object_type" in out:
        eligible &= out["object_type"] != "bright_matrix"

    for _, group in out.loc[eligible].groupby(keys, sort=False):
        coords = group[[y_key, x_key]].to_numpy(float)
        angles = group[angle_key].to_numpy(float)
        n_query = min(k + 1, len(group))
        _, neighbours = cKDTree(coords).query(coords, k=n_query)
        neighbours = np.asarray(neighbours).reshape(len(group), -1)[:, 1:]
        diff = np.abs(angles[neighbours] - angles[:, None])
        if axial:
            diff = np.mod(diff, np.pi)
            diff = np.minimum(diff, np.pi - diff)
        out.loc[group.index, "alignment_score"] = np.sqrt((diff**2).sum(axis=1)) / k
    return out
