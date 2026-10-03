"""Plots for fiber segmentation and tissue regions."""

from __future__ import annotations

import numpy as np

__all__ = ["pick_qc_crops", "plot_fiber_overlay", "plot_fiber_qc", "plot_fiber_segmentation_steps", "plot_tissue_regions"]

_STEP_PANELS = (
    ("raw", "Raw channel"),
    ("blurred", "Gaussian blur"),
    ("contrast_adjusted", "Contrast adjusted (CLAHE)"),
    ("ridges", "Frangi ridges"),
    ("distance_transformed", "Ridge distance map"),
    ("thresholded", "Watershed markers"),
    ("elevation_map", "Sobel elevation map"),
    ("unfiltered_labels", "Unfiltered fibers"),
    ("labels", "Final fibers"),
)


def _display_range(image: np.ndarray) -> tuple[float, float]:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(finite, [1, 99.5])
    return float(low), float(max(high, low + 1e-9))


def _label_colours(labels: np.ndarray, seed: int = 0):
    from matplotlib.colors import ListedColormap

    rng = np.random.default_rng(seed)
    colours = rng.uniform(0.25, 1.0, size=(int(labels.max()) + 1, 4))
    colours[:, 3] = 1.0
    colours[0] = (0, 0, 0, 0)
    return ListedColormap(colours)


def plot_fiber_segmentation_steps(steps: dict, title: str | None = None, figsize=(12, 12)):
    """3 x 3 panel of every segmentation step, as in ark's tuning plot.

    ``steps`` is the dictionary returned by
    ``spatioev.pp.segment_fibers(..., return_steps=True)``.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=figsize)
    for ax, (key, name) in zip(axes.ravel(), _STEP_PANELS):
        image = steps.get(key)
        if image is None:
            ax.axis("off")
            continue
        if key.endswith("labels"):
            ax.imshow(np.zeros_like(image), cmap="gray")
            ax.imshow(image, cmap=_label_colours(image), interpolation="nearest")
            name = f"{name} ({len(np.unique(image)) - 1})"
        else:
            low, high = _display_range(np.asarray(image, dtype=float))
            ax.imshow(image, cmap="bone", vmin=low, vmax=high)
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_fiber_overlay(image: np.ndarray, labels: np.ndarray, ax=None, title: str | None = None, max_dimension: int = 1600):
    """Show a matrix channel with segmented fibers outlined.

    Large images are strided down to ``max_dimension`` pixels on the long
    side for display only.
    """
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries

    stride = max(1, int(np.ceil(max(image.shape) / max_dimension)))
    small = np.asarray(image[::stride, ::stride], dtype=float)
    small_labels = np.asarray(labels[::stride, ::stride])
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    low, high = _display_range(small)
    ax.imshow(small, cmap="gray", vmin=low, vmax=high)
    edges = find_boundaries(small_labels, mode="inner") if stride == 1 else small_labels > 0
    overlay = np.zeros((*small.shape, 4))
    overlay[edges] = (1.0, 0.35, 0.1, 0.9)
    ax.imshow(overlay, interpolation="nearest")
    ax.set_title(title or "Segmented fibers")
    ax.axis("off")
    return ax


def plot_tissue_regions(
    obs,
    boundaries=None,
    region_key: str = "tissue_region",
    x_key: str = "X_centroid",
    y_key: str = "Y_centroid",
    ax=None,
    title: str | None = None,
    point_size: float | None = None,
):
    """Scatter cells coloured by tumour / envelope / stroma, with nest outlines."""
    import matplotlib.pyplot as plt

    colours = {"tumour": "#c2410c", "envelope": "#d4a017", "stroma": "#3b82a0"}
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    size = point_size or max(0.5, min(6.0, 20000 / max(len(obs), 1)))
    for region, colour in colours.items():
        sub = obs[obs[region_key].astype(str) == region]
        ax.scatter(sub[x_key], sub[y_key], s=size, c=colour, label=f"{region} ({len(sub):,})", linewidths=0)
    if boundaries is not None and len(boundaries):
        for column, style in (("geometry", "-"), ("expanded_geometry", "--")):
            for geometry in boundaries.get(column, []):
                if geometry is None or geometry.is_empty:
                    continue
                parts = [geometry] if geometry.geom_type == "Polygon" else list(getattr(geometry, "geoms", []))
                for part in parts:
                    if part.geom_type != "Polygon":
                        continue
                    xs, ys = part.exterior.xy
                    ax.plot(xs, ys, style, color="black", linewidth=0.8)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.legend(loc="upper right", markerscale=4, fontsize=8, frameon=True)
    ax.set_title(title or "Tissue regions")
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def _stretch(image: np.ndarray) -> np.ndarray:
    low, high = _display_range(np.asarray(image, dtype=float))
    return np.clip((np.asarray(image, dtype=float) - low) / (high - low), 0, 1)


def _qc_overlay(gray: np.ndarray, labels: np.ndarray, bright_lookup: np.ndarray | None, outlines: bool = True) -> np.ndarray:
    """RGB: teal = ridge fibers, magenta = added bright matrix, orange outlines."""
    from skimage.segmentation import find_boundaries

    rgb = np.dstack([gray] * 3)
    inside = labels > 0
    bright = np.zeros(labels.shape, dtype=bool)
    if bright_lookup is not None and inside.any():
        bright = bright_lookup[np.clip(labels, 0, len(bright_lookup) - 1)] & inside
    fiber = inside & ~bright
    rgb[fiber] = rgb[fiber] * 0.5 + np.array([0.0, 0.55, 0.55]) * 0.5
    rgb[bright] = rgb[bright] * 0.35 + np.array([1.0, 0.15, 0.8]) * 0.65
    if outlines:
        rgb[find_boundaries(labels, mode="inner")] = (1.0, 0.45, 0.0)
    return rgb


def pick_qc_crops(mask_small: np.ndarray, stride: int, crop_size: int = 400, n_crops: int = 3) -> list[tuple[int, int]]:
    """Top-left corners (full-resolution pixels) of crops with dense, median and sparse matrix.

    ``mask_small`` is the matrix mask strided by ``stride``. Windows are
    ranked by matrix area; only windows holding some matrix are considered.
    """
    window = max(1, crop_size // stride)
    rows, cols = mask_small.shape
    if rows < window or cols < window:
        return [(0, 0)]
    integral = np.zeros((rows + 1, cols + 1))
    integral[1:, 1:] = np.cumsum(np.cumsum(mask_small, axis=0, dtype=float), axis=1)
    step = max(1, window // 2)
    candidates = []
    for r in range(0, rows - window + 1, step):
        for c in range(0, cols - window + 1, step):
            area = integral[r + window, c + window] - integral[r, c + window] - integral[r + window, c] + integral[r, c]
            candidates.append((area / window**2, r, c))
    candidates = [cand for cand in candidates if cand[0] > 0.02]
    if not candidates:
        return [(0, 0)]
    candidates.sort()
    picks = [candidates[-1], candidates[len(candidates) // 2], candidates[len(candidates) // 6]][:n_crops]
    seen, corners = set(), []
    for _, r, c in picks:
        if (r, c) not in seen:
            seen.add((r, c))
            corners.append((r * stride, c * stride))
    return corners


def plot_fiber_qc(
    image,
    labels,
    bright_labels=None,
    title: str | None = None,
    crop_size: int = 400,
    n_crops: int = 3,
    max_dimension: int = 1400,
):
    """QC figure for one image and matrix channel.

    Top: the whole image, raw and with the segmentation overlaid. Below: up
    to ``n_crops`` full-resolution crops picked automatically (densest,
    median and sparse matrix). Teal = ridge fibers, magenta = matrix added by
    ``include_bright`` (labels listed in ``bright_labels``), orange =
    outlines. ``image`` and ``labels`` may be lazy (zarr / memmap); only the
    strided overview and the crops are read.
    """
    import matplotlib.pyplot as plt

    rows_full, cols_full = int(image.shape[0]), int(image.shape[1])
    stride = max(1, int(np.ceil(max(rows_full, cols_full) / max_dimension)))
    small = np.asarray(image[::stride, ::stride], dtype=float)
    small_labels = np.asarray(labels[::stride, ::stride])
    lookup = None
    if bright_labels is not None and len(bright_labels):
        top = int(max(int(np.max(bright_labels)), int(small_labels.max())))
        lookup = np.zeros(top + 2, dtype=bool)
        lookup[np.asarray(bright_labels, dtype=np.int64)] = True
    corners = pick_qc_crops(small_labels > 0, stride, crop_size, n_crops)

    fig = plt.figure(figsize=(14, 7 + 4.8 * len(corners)))
    grid = fig.add_gridspec(1 + len(corners), 2, height_ratios=[7] + [4.8] * len(corners))
    gray = _stretch(small)
    ax = fig.add_subplot(grid[0, 0])
    ax.imshow(gray, cmap="gray")
    ax.set_title("raw (contrast-stretched)", fontsize=10)
    for (y, x) in corners:
        ax.add_patch(plt.Rectangle((x / stride, y / stride), crop_size / stride, crop_size / stride, fill=False, edgecolor="yellow", linewidth=1))
    ax.axis("off")
    ax = fig.add_subplot(grid[0, 1])
    ax.imshow(_qc_overlay(gray, small_labels, lookup, outlines=stride <= 2))
    area = 100 * (small_labels > 0).mean()
    ax.set_title(f"segmentation: {area:.1f}% of image" + ("  (teal = ridge fibers, magenta = bright matrix)" if lookup is not None else ""), fontsize=10)
    ax.axis("off")
    for row, (y, x) in enumerate(corners, start=1):
        crop = np.asarray(image[y : y + crop_size, x : x + crop_size], dtype=float)
        crop_labels = np.asarray(labels[y : y + crop_size, x : x + crop_size])
        crop_gray = _stretch(crop)
        ax = fig.add_subplot(grid[row, 0])
        ax.imshow(crop_gray, cmap="gray")
        ax.set_title(f"crop at row {y}, col {x}", fontsize=10)
        ax.axis("off")
        ax = fig.add_subplot(grid[row, 1])
        ax.imshow(_qc_overlay(crop_gray, crop_labels, lookup))
        ax.set_title(f"{len(np.unique(crop_labels)) - 1} objects, {100 * (crop_labels > 0).mean():.0f}% of crop", fontsize=10)
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / fig.get_figheight()))
    else:
        fig.tight_layout()
    return fig
