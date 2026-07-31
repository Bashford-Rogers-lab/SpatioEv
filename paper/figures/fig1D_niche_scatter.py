"""
Figure 1D — Spatial map of ductal niche connected components
=============================================================
Every cell coloured by its ductal niche identity.
Non-ductal cells shown as a subtle grey background.

Run from the SpatioEv root:
    python notebooks/fig1D_niche_scatter.py

Output: paper/notebooks/results/pseudotime_exp2/fig1D_niche_scatter.pdf  (and .png)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# ── Configuration ─────────────────────────────────────────────────────────────
ROOT          = Path("/Users/shihongwu/SpatioEv")
DATA_DIR      = ROOT / "data" / "exp_2"
RESULT_DIR    = ROOT / "notebooks" / "results" / "pseudotime_exp2"

ADATA_PATH    = DATA_DIR / "34434_1_adata.h5ad"
ANNOTATION_PATH = DATA_DIR / "34434_1_annotation.csv"
SEG_DIR       = DATA_DIR / "segmentation"

DUCT_NICHE_KEY = "pancreatic ductal epithelium_mask_component"
DUCT_LABEL     = "pancreatic ductal epithelium"

PIXEL_SIZE_UM  = 0.325   # µm per pixel

# ── Publication style ─────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":   "Arial",
    "font.size":      6,
    "axes.labelsize": 6,
    "pdf.fonttype":  42,
    "svg.fonttype":  "none",
})

# ── Colour palette for niche components ───────────────────────────────────────
# 40 visually distinct colours — cycling for 40 k+ components.
# Randomised order (fixed seed) so sequential component IDs ≠ similar colour.
_BASE_PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
    "#3182bd", "#e6550d", "#31a354", "#756bb1", "#636363",
]
_rng = np.random.default_rng(42)
PALETTE = [_BASE_PALETTE[i] for i in _rng.permutation(len(_BASE_PALETTE))]
N_COLORS = len(PALETTE)


def comp_color(component_name: str) -> str:
    """Deterministic colour from component name via integer hash."""
    try:
        idx = int(component_name.split("_")[-1])
    except (ValueError, AttributeError):
        idx = hash(component_name)
    return PALETTE[abs(idx) % N_COLORS]


# ── Load adata and compute niche components ───────────────────────────────────
def load_adata():
    import anndata as ad
    import spatioev as sv

    print("Loading adata …")
    adata = ad.read_h5ad(ADATA_PATH)

    annotations = pd.read_csv(ANNOTATION_PATH, index_col=0)
    for col in ["Tier_A", "Tier_B"]:
        if col in annotations.columns:
            adata.obs[col] = annotations[col].reindex(adata.obs_names)

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
    n = adata.obs[DUCT_NICHE_KEY].nunique()
    print(f"  → {n} components found")
    return adata


# ── Build the figure ──────────────────────────────────────────────────────────
def make_figure(adata):
    obs = adata.obs.copy()

    # Use Tier_A annotation — reliable ductal/non-ductal split regardless of
    # whether assign_singletons fills DUCT_NICHE_KEY for non-ductal cells too
    duct_mask    = obs["Tier_A"] == DUCT_LABEL
    non_duct_obs = obs[~duct_mask]
    duct_obs     = obs[duct_mask & obs[DUCT_NICHE_KEY].notna()].copy()

    print(f"  non-ductal: {len(non_duct_obs):,}  |  ductal with niche ID: {len(duct_obs):,}")

    # Per-cell colour from component id
    duct_obs["_color"] = duct_obs[DUCT_NICHE_KEY].map(comp_color)

    # ── Figure sizing (single Nature column = 89 mm; auto-height from data) ──
    x_range = obs["X_centroid"].max() - obs["X_centroid"].min()
    y_range = obs["Y_centroid"].max() - obs["Y_centroid"].min()
    aspect  = y_range / x_range          # >1 → portrait tissue

    target_w_mm = 51.0
    fig_w = target_w_mm / 25.4
    fig_h = fig_w * aspect + 0.15

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # ── Background: non-ductal cells (downsampled for speed) ─────────────────
    # Pass color as explicit RGBA tuple — unambiguous, no matplotlib string parsing
    gray_rgba = (0.78, 0.78, 0.78, 0.4)
    bg = non_duct_obs.sample(frac=0.4, random_state=1) if len(non_duct_obs) > 300_000 else non_duct_obs
    ax.scatter(
        bg["X_centroid"].values,
        bg["Y_centroid"].values,
        color=gray_rgba,
        s=0.07,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )

    # ── Ductal niche cells coloured by component ──────────────────────────────
    duct_shuffled = duct_obs.sample(frac=1, random_state=0)
    ax.scatter(
        duct_shuffled["X_centroid"].values,
        duct_shuffled["Y_centroid"].values,
        c=duct_shuffled["_color"].tolist(),
        s=0.25,
        alpha=0.92,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )

    # ── Scale bar ─────────────────────────────────────────────────────────────
    scale_um   = 1000           # 1 mm
    scale_px   = scale_um / PIXEL_SIZE_UM
    x_min      = obs["X_centroid"].min()
    x_max      = obs["X_centroid"].max()
    y_min      = obs["Y_centroid"].min()
    y_max      = obs["Y_centroid"].max()
    margin_x   = (x_max - x_min) * 0.03
    margin_y   = (y_max - y_min) * 0.03
    sb_x0      = x_max - margin_x - scale_px
    sb_x1      = x_max - margin_x
    sb_y       = y_min + margin_y

    ax.plot([sb_x0, sb_x1], [sb_y, sb_y],
            color="black", lw=1.5, solid_capstyle="butt", zorder=10)
    ax.text((sb_x0 + sb_x1) / 2, sb_y + (y_max - y_min) * 0.012,
            "1 mm", ha="center", va="bottom", fontsize=5,
            color="black", zorder=10)

    # ── Axes cleanup ─────────────────────────────────────────────────────────
    ax.set_xlim(x_min - margin_x * 2, x_max + margin_x * 2)
    ax.set_ylim(y_min - margin_y * 2, y_max + margin_y * 2)
    ax.invert_yaxis()          # image convention: y=0 at top
    ax.set_aspect("equal")
    ax.axis("off")

    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # ── Save ─────────────────────────────────────────────────────────────────
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = RESULT_DIR / "fig1D_niche_scatter.pdf"
    out_png = RESULT_DIR / "fig1D_niche_scatter.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nSaved:\n  {out_pdf}\n  {out_png}")
    plt.show()


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    adata = load_adata()
    make_figure(adata)
