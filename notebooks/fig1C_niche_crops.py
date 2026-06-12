"""
Figure 1C — Ductal niche identification from segmentation mask
==============================================================
Shows two ductal niches (one organized/low-pseudotime, one irregular/high-pseudotime)
with their DAPI · CK19 · NaKATPase channel crops and coloured segmentation mask overlay.

Run this script on your Mac (where the external drive and h5ad are accessible):
    python notebooks/fig1C_niche_crops.py

Output: notebooks/results/pseudotime_exp2/fig1C_niche_crops.pdf  (and .png)
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import tifffile
from skimage.segmentation import find_boundaries

# ── Configuration ─────────────────────────────────────────────────────────────
ROOT = Path("/Users/shihongwu/SpatioEv")
DATA_DIR = ROOT / "data" / "exp_2"
RESULT_DIR = ROOT / "notebooks" / "results" / "pseudotime_exp2"

ADATA_PATH       = DATA_DIR / "34434_1_adata.h5ad"
ANNOTATION_PATH  = DATA_DIR / "34434_1_annotation.csv"
PIXEL_FEAT_PATH  = DATA_DIR / "pixel_features.csv"
SEG_DIR          = DATA_DIR / "segmentation"
ARK_WDIR         = Path("/Volumes/Shihong_5/pancreatic_image_analysis/34434_1/ark_wdir")

DUCT_NICHE_KEY   = "pancreatic ductal epithelium_mask_component"
DUCT_LABEL       = "pancreatic ductal epithelium"

# Target niches — selected for visual contrast in niche morphology
# Low grade: large, well-organised round duct
NICHE_LOW_PT     = "global__component_8795"   # 70 cells, circularity=0.943
# High grade: irregular, branching/fragmented structure
NICHE_HIGH_PT    = "global__component_10397"  # 39 cells, circularity=0.143

# Padding around each niche crop (pixels)
CROP_PAD         = 120

# Channel display names and colours for composite
CHANNELS = {
    "DAPI":       {"cmap": "Blues",  "composite_rgb": np.array([0.20, 0.45, 0.85])},
    "CK19":       {"cmap": "Greens", "composite_rgb": np.array([0.15, 0.80, 0.25])},
    "NaKATPase":  {"cmap": "Reds",   "composite_rgb": np.array([0.90, 0.15, 0.15])},
}
CHANNEL_NAMES = list(CHANNELS.keys())

# Niche mask overlay colours
NICHE_COLORS = {
    NICHE_LOW_PT:  "#4393c3",   # blue  – low-grade organised duct
    NICHE_HIGH_PT: "#d6604d",   # red   – high-grade irregular duct
}

# Publication style
matplotlib.rcParams.update({
    "font.family": "Arial",
    "font.size": 6,
    "axes.labelsize": 6,
    "xtick.labelsize": 5.5,
    "ytick.labelsize": 5.5,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


# ── Helper: discover ARK image directory structure ────────────────────────────
def find_channel_dir(ark_wdir: Path, fov: str) -> Path:
    """
    Try common Pixie/ARK directory layouts:
      1. ark_wdir/image_data/<fov>/
    Returns the first existing match, or raises FileNotFoundError.
    """
    candidates = [
        ark_wdir / "image_data" / fov,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Cannot find channel images for {fov} in {ark_wdir}.\n"
        f"Tried: {[str(c) for c in candidates]}\n"
        f"Please update ARK_WDIR or the find_channel_dir() function."
    )


def find_channel_file(channel_dir: Path, channel: str) -> Path:
    """
    Find a channel TIFF, trying common naming variants case-insensitively.
    """
    name_variants = {
        "DAPI":      ["DNA_1",],
        "CK19":      ["CK19", ],
        "NaKATPase": ["NaKATPase",],
    }
    for name in name_variants.get(channel, [channel]):
        for ext in [".tiff", ".tif"]:
            p = channel_dir / (name + ext)
            if p.exists():
                return p
    # Fall back to case-insensitive glob
    for p in channel_dir.glob("*.tif*"):
        if channel.lower() in p.stem.lower():
            return p
    raise FileNotFoundError(
        f"Cannot find channel '{channel}' in {channel_dir}.\n"
        f"Available files: {sorted(channel_dir.glob('*.tif*'))}"
    )


# ── Helper: normalise image to [0, 1] with percentile clipping ───────────────
def norm_percentile(img: np.ndarray, lo: float = 1.0, hi: float = 99.5) -> np.ndarray:
    vlo, vhi = np.percentile(img, lo), np.percentile(img, hi)
    if vhi == vlo:
        return np.zeros_like(img, dtype=float)
    return np.clip((img.astype(float) - vlo) / (vhi - vlo), 0, 1)


# ── Helper: build composite RGB from individual channel images ────────────────
def build_composite(channel_imgs: dict) -> np.ndarray:
    """channel_imgs: {channel_name: normalised 2-D float array}"""
    h, w = next(iter(channel_imgs.values())).shape
    composite = np.zeros((h, w, 3), dtype=float)
    for ch, img in channel_imgs.items():
        rgb = CHANNELS[ch]["composite_rgb"]
        composite += img[:, :, None] * rgb[None, None, :]
    return np.clip(composite, 0, 1)


# ── Helper: build coloured mask overlay ──────────────────────────────────────
def build_mask_overlay(composite: np.ndarray,
                       mask_crop: np.ndarray,
                       niche_cells: dict,
                       alpha: float = 0.45) -> np.ndarray:
    """
    Overlay coloured niche fills on composite.
    niche_cells: {niche_id: set(label_ids_in_this_fov)}
    """
    overlay = composite.copy()
    for niche_id, labels in niche_cells.items():
        color = np.array(matplotlib.colors.to_rgb(NICHE_COLORS[niche_id]))
        for lbl in labels:
            cell_mask = (mask_crop == lbl)
            if cell_mask.any():
                overlay[cell_mask] = (1 - alpha) * composite[cell_mask] + alpha * color
    return np.clip(overlay, 0, 1)


# ── Load adata and extract niche cell info ────────────────────────────────────
def load_niche_cells(niche_ids: list) -> dict:
    """
    Returns:
        {niche_id: {'fov': str, 'labels': list[int],
                    'x_local': np.ndarray, 'y_local': np.ndarray}}
    """
    print("Loading adata … (this may take a moment)")
    import anndata as ad
    import spatioev as sv

    adata = ad.read_h5ad(ADATA_PATH)

    # Merge annotations (Tier_A needed for sv.cluster_spatial_components_from_mask)
    annotations = pd.read_csv(ANNOTATION_PATH, index_col=0)
    annotation_cols = [col for col in ["Tier_A", "Tier_B"] if col in annotations.columns]
    adata.obs = adata.obs.join(annotations[annotation_cols], how="left")

    # Compute ductal niche component assignments — same call as in the notebook
    # (deterministic; matches the niche IDs in niche_pseudotime_results.csv)
    print("Computing ductal niche components …")
    adata = sv.cluster_spatial_components_from_mask(
        adata,
        seg_dir=SEG_DIR,
        label_key="Tier_A",
        label_value=DUCT_LABEL,
        fov_key="fov",
        cell_label_key="label",
        connection_mode="label_adjacency",
        gap_tolerance=5,
        stitch_across_fovs=True,
        fov_grid_cols=2,
        stitch_gap_tolerance=5,
        connectivity=2,
        min_component_size=3,
        assign_singletons=True,
    )
    print(f"  → {adata.obs[DUCT_NICHE_KEY].nunique()} components found")

    # NOTE on coordinate systems:
    #   adata.obs['X_centroid'] / ['Y_centroid']  → whole-slide pixel space
    #       (corrected back from per-FOV to original WSI coordinates; used for
    #        spatial plots across all FOVs but NOT for indexing per-FOV images)
    #   pixel_features['geometric_centroid_x/y']  → local per-FOV pixel space
    #       (needed to index into per-FOV segmentation masks and channel TIFFs)
    #
    # We re-merge pixel_features here to obtain the local FOV coordinates.
    pf = pd.read_csv(PIXEL_FEAT_PATH)
    obs_index_name = adata.obs.index.name or "cell_id"
    obs = (
        adata.obs.reset_index(names=obs_index_name)
        .merge(pf[["label", "fov", "geometric_centroid_x", "geometric_centroid_y"]],
               on=["label", "fov"], how="left")
    )

    result = {}
    for niche_id in niche_ids:
        cells = obs[obs[DUCT_NICHE_KEY] == niche_id]
        if len(cells) == 0:
            raise ValueError(f"No cells found for niche '{niche_id}'.")

        fov_counts = cells["fov"].value_counts()
        primary_fov = fov_counts.index[0]  # FOV with most cells
        cells_in_fov = cells[cells["fov"] == primary_fov]

        # NOTE: scikit-image regionprops centroid ordering — centroid[0]=row, centroid[1]=col
        # pixel_features exports these as geometric_centroid_x=row, geometric_centroid_y=col
        # So: row_coord = geometric_centroid_x, col_coord = geometric_centroid_y
        result[niche_id] = {
            "fov": primary_fov,
            "labels": cells_in_fov["label"].astype(int).tolist(),
            "row_local": cells_in_fov["geometric_centroid_x"].values,  # axis-0 (rows)
            "col_local": cells_in_fov["geometric_centroid_y"].values,  # axis-1 (cols)
            "n_cells_total": len(cells),
            "n_cells_fov": len(cells_in_fov),
        }
        print(f"  {niche_id}: FOV={primary_fov}, {len(cells_in_fov)} cells in FOV "
              f"(total={len(cells)})")
    return result


# ── Load a single channel image (full FOV) ───────────────────────────────────
def load_channel(fov: str, channel: str) -> np.ndarray:
    ch_dir = find_channel_dir(ARK_WDIR, fov)
    ch_file = find_channel_file(ch_dir, channel)
    img = tifffile.imread(str(ch_file))
    if img.ndim == 3:
        img = img[0]  # take first frame if multi-page
    return img.astype(float)


# ── Compute padded bounding box in local FOV pixel space ─────────────────────
def bounding_box(row_arr, col_arr, pad=CROP_PAD, img_shape=None):
    """Returns (r0, r1, c0, c1) clipped to img_shape."""
    r0 = max(0, int(row_arr.min()) - pad)
    r1 = int(row_arr.max()) + pad
    c0 = max(0, int(col_arr.min()) - pad)
    c1 = int(col_arr.max()) + pad
    if img_shape is not None:
        r1 = min(r1, img_shape[0])
        c1 = min(c1, img_shape[1])
    return r0, r1, c0, c1


# ── Main figure generation ────────────────────────────────────────────────────
def make_figure(niche_data: dict):
    """
    Layout: 2 rows (one per niche) × 4 columns
      Col 0: DAPI (greyscale, niche outline)
      Col 1: CK19 (green tones)
      Col 2: NaKATPase (red tones)
      Col 3: RGB composite + filled niche mask overlay
    """
    niche_ids = [NICHE_LOW_PT, NICHE_HIGH_PT]
    n_rows = len(niche_ids)
    n_cols = 4  # DAPI | CK19 | NaKATPase | composite+mask

    col_labels  = ["DAPI", "CK19", "NaKATPase", "Composite + niche mask"]
    row_labels  = ["Ductal niche 1", "Ductal niche 2"]
    scale_bar_um = 50  # µm
    pixel_size   = 0.325  # µm / pixel

    # Publication sizing: 82 mm wide (fits row 2 of 6.69" canvas), 6 pt type
    mm2in = 1 / 25.4
    fig_w = 82 * mm2in
    panel_size = (fig_w - 0.25) / n_cols
    fig_h = panel_size * n_rows + 0.35    # square panels + header + legend space

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        gridspec_kw={"hspace": 0.06, "wspace": 0.04},
    )
    if n_rows == 1:
        axes = axes[None, :]

    for row_idx, niche_id in enumerate(niche_ids):
        info = niche_data[niche_id]
        fov    = info["fov"]
        r_arr  = info["row_local"]   # axis-0 coordinates
        c_arr  = info["col_local"]   # axis-1 coordinates
        labels = set(info["labels"])

        # ── Load and crop segmentation mask ──────────────────────────────────
        seg_file = SEG_DIR / f"{fov}_whole_cell.tiff"
        if not seg_file.exists():
            raise FileNotFoundError(f"Segmentation mask not found: {seg_file}")
        seg_mask_full = tifffile.imread(str(seg_file))
        if seg_mask_full.ndim == 3:
            seg_mask_full = seg_mask_full[0]
        img_h, img_w = seg_mask_full.shape

        # Load one channel first to check its shape
        probe_img = load_channel(fov, CHANNEL_NAMES[0])
        ch_h, ch_w = probe_img.shape[:2]

        print(f"  [{niche_id}] seg mask shape: ({img_h}, {img_w})  "
              f"channel image shape: ({ch_h}, {ch_w})")
        print(f"  [{niche_id}] cell row range: {r_arr.min():.0f}–{r_arr.max():.0f}  "
              f"col range: {c_arr.min():.0f}–{c_arr.max():.0f}")

        # Use the stricter of seg mask and channel image as the clip boundary
        clip_h = min(img_h, ch_h)
        clip_w = min(img_w, ch_w)
        r0, r1, c0, c1 = bounding_box(r_arr, c_arr, pad=CROP_PAD,
                                       img_shape=(clip_h, clip_w))
        print(f"  [{niche_id}] crop box: rows {r0}–{r1}, cols {c0}–{c1}")

        if r1 <= r0 or c1 <= c0:
            raise ValueError(
                f"Empty crop for {niche_id} in {fov}: rows [{r0},{r1}), cols [{c0},{c1}).\n"
                f"Cell row range {r_arr.min():.0f}–{r_arr.max():.0f}, "
                f"col range {c_arr.min():.0f}–{c_arr.max():.0f}, "
                f"image ({ch_h}×{ch_w})."
            )

        seg_crop = seg_mask_full[r0:r1, c0:c1]

        # ── Load and crop channels ────────────────────────────────────────────
        ch_crops = {}
        ch_crops[CHANNEL_NAMES[0]] = norm_percentile(probe_img[r0:r1, c0:c1])
        for ch in CHANNEL_NAMES[1:]:
            full_img = load_channel(fov, ch)
            ch_crops[ch] = norm_percentile(full_img[r0:r1, c0:c1])
        del probe_img, full_img  # free memory

        composite = build_composite(ch_crops)

        # ── Build mask overlay on composite ──────────────────────────────────
        niche_labels_in_crop = {niche_id: labels}
        overlay = build_mask_overlay(composite, seg_crop, niche_labels_in_crop)

        # Also draw a thin outline for the other niche if present in this crop
        # (only if the other niche happens to appear in the same FOV crop)
        other_id = NICHE_HIGH_PT if niche_id == NICHE_LOW_PT else NICHE_LOW_PT
        other_info = niche_data.get(other_id, {})
        if other_info.get("fov") == fov:
            crop_labels = set(np.unique(seg_crop).tolist())
            other_labels_in_box = set(other_info.get("labels", [])) & crop_labels
            if other_labels_in_box:
                niche_labels_all = {niche_id: labels, other_id: other_labels_in_box}
                overlay = build_mask_overlay(composite, seg_crop, niche_labels_all)

        # ── Draw panels ───────────────────────────────────────────────────────
        panel_data = [
            (ch_crops["DAPI"],   CHANNELS["DAPI"]["cmap"],   "DAPI"),
            (ch_crops["CK19"],   CHANNELS["CK19"]["cmap"],   "CK19"),
            (ch_crops["NaKATPase"], CHANNELS["NaKATPase"]["cmap"], "NaKATPase"),
            (overlay,            None,                        "Composite + mask"),
        ]

        for col_idx, (img_data, cmap, _) in enumerate(panel_data):
            ax = axes[row_idx, col_idx]
            if cmap is not None:
                ax.imshow(img_data, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
            else:
                ax.imshow(img_data, interpolation="nearest")

            # Draw niche cell outlines on all panels using contour of mask
            for nid, nlabels in {niche_id: labels}.items():
                niche_binary = np.isin(seg_crop, list(nlabels)).astype(np.uint8)
                if niche_binary.any():
                    boundary = find_boundaries(niche_binary, mode="outer")
                    boundary_r, boundary_c = np.where(boundary)
                    ax.scatter(boundary_c, boundary_r, s=0.3,
                               c=NICHE_COLORS[nid], linewidths=0,
                               rasterized=True, zorder=5)

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Column headers (top row only)
            if row_idx == 0:
                ax.set_title(col_labels[col_idx], fontsize=6, pad=2)

            # Row labels (leftmost column only)
            if col_idx == 0:
                ax.set_ylabel(row_labels[row_idx], fontsize=6, labelpad=3,
                              rotation=90, va="center")

        # ── Scale bar on rightmost panel ──────────────────────────────────────
        ax_sb = axes[row_idx, -1]
        sb_px = scale_bar_um / pixel_size
        crop_w = c1 - c0
        crop_h = r1 - r0
        sb_x0 = crop_w * 0.07
        sb_y0 = crop_h * 0.93
        ax_sb.plot([sb_x0, sb_x0 + sb_px], [sb_y0, sb_y0],
                   color="white", lw=1.5, solid_capstyle="butt", zorder=10)
        ax_sb.text(sb_x0 + sb_px / 2, sb_y0 - crop_h * 0.03,
                   f"{scale_bar_um} µm",
                   ha="center", va="bottom", fontsize=5,
                   color="white", zorder=10)

    # ── Legend for niche colours ──────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(fc=NICHE_COLORS[NICHE_LOW_PT],  ec="none",
                       label="Ductal niche 1"),
        mpatches.Patch(fc=NICHE_COLORS[NICHE_HIGH_PT], ec="none",
                       label="Ductal niche 2"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=2,
        fontsize=6,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.savefig(RESULT_DIR / "fig1C_niche_crops.pdf",
                dpi=300, bbox_inches="tight")
    fig.savefig(RESULT_DIR / "fig1C_niche_crops.png",
                dpi=300, bbox_inches="tight")
    print(f"\nSaved to:\n  {RESULT_DIR / 'fig1C_niche_crops.pdf'}")
    print(f"  {RESULT_DIR / 'fig1C_niche_crops.png'}")
    plt.show()


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Check external drive is accessible
    if not ARK_WDIR.exists():
        print(f"ERROR: External drive not mounted or path wrong:\n  {ARK_WDIR}")
        print("Please mount the drive and re-run.")
        sys.exit(1)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Figure 1C: Ductal niche identification ===")
    print(f"Niche A (low-grade organised): {NICHE_LOW_PT}")
    print(f"Niche B (high-grade irregular): {NICHE_HIGH_PT}")
    print()

    niche_data = load_niche_cells([NICHE_LOW_PT, NICHE_HIGH_PT])
    make_figure(niche_data)
