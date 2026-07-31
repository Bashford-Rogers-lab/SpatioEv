"""
Supplementary Figure 8 — Multi-scale cell density analysis (PDAC IMC, exp_1)
=============================================================================
Panel A : Workflow schematic — tile-based general, KNN local, radius interaction
Panel B : General density — cell count per tile (all cells)
Panel C : General density — pixel area coverage per tile (all cells)
Panel D : Phenotype tile density — tumour cells
Panel E : Cross-phenotype tile density Pearson correlation matrix
Panel F : KDE density landscape — CD4 T cells
Panel G : KNN local density — all cells spatial scatter
Panel H : KNN local density — CD4 T cells
Panel I : Radius interaction density — tumour ↔ CD4 T cells spatial map

Data:
  data/exp_1/exp_1.h5ad
  results/svm_phenotyping_results.csv

Run:
    python notebooks/suppfig8_density.py

Output: notebooks/results/suppfig8/suppfig8_*.pdf (.png)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as mpe
import matplotlib.colors as mcolors
import seaborn as sns
import spatioev as sv

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "exp_1" / "exp_1.h5ad"
ANN_PATH  = ROOT / "results" / "svm_phenotyping_results.csv"
OUT_DIR   = Path(__file__).parent / "results" / "suppfig8"

MM2IN = 1 / 25.4

# ── Constants ──────────────────────────────────────────────────────────────────
PHENOTYPE_KEY = "annotated_clusters_update3"
IMAGE_ID      = "exp"
SOURCE_PHENO  = "tumour"
TARGET_PHENO  = "CD4 T cells"
RADIUS        = 30
K_NEIGHBORS   = 5

PHENOTYPE_LABELS = {
    "CD4 T cells":     "CD4 T cells",
    "CD90+ CAFs":      "CD90+ CAFs",
    "Dendritic cells": "Dendritic cells",
    "Macrophages":     "Macrophages",
    "Monocytes":       "Monocytes",
    "PDPN+ CAFs":      "PDPN+ CAFs",
    "T cells":         "T cells",
    "endothelial":     "Endothelial",
    "myCAFs":          "myCAFs",
    "tumour":          "Tumour",
    "Unknown":         "Unknown",
}


# ── Publication RC ─────────────────────────────────────────────────────────────
def _set_pub_rc():
    plt.rcParams.update({
        "font.family":         "Arial",
        "font.size":           6,
        "axes.titlesize":      6.5,
        "axes.labelsize":      6,
        "xtick.labelsize":     5.5,
        "ytick.labelsize":     5.5,
        "axes.linewidth":      0.5,
        "xtick.major.width":   0.5,
        "ytick.major.width":   0.5,
        "xtick.major.size":    2.0,
        "ytick.major.size":    2.0,
        "lines.linewidth":     0.8,
        "legend.fontsize":     5.0,
        "legend.handlelength": 0.8,
        "pdf.fonttype":        42,
        "svg.fonttype":        "none",
    })


def _despine(ax):
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)


# ── Data loading ───────────────────────────────────────────────────────────────
def load_data():
    print("Loading exp_1.h5ad …")
    adata = sv.load_h5ad(str(DATA_PATH))
    ann   = pd.read_csv(ANN_PATH, index_col=0)[["svm_refined_phenotype"]]
    ann   = ann.rename(columns={"svm_refined_phenotype": "annotated_clusters_update3"})
    adata.obs = adata.obs.join(ann, how="left")
    print(f"  {adata.n_obs:,} cells loaded")
    return adata


def compute_all(adata):
    """Run all density computations and return results dict."""
    print("Computing tile-based density …")
    df            = sv.assign_tiles(adata)
    density       = sv.compute_general_density(df)
    pheno_density = sv.compute_phenotype_density(df, phenotype_key=PHENOTYPE_KEY)
    corr          = sv.phenotype_density_correlation(
                        pheno_density, phenotype_key=PHENOTYPE_KEY)

    print("Computing KDE density (CD4 T cells) …")
    Xg, Yg, Z = sv.compute_kde_density(
        adata, phenotype_key=PHENOTYPE_KEY, phenotype=TARGET_PHENO)

    print("Computing KNN local density …")
    adata = sv.compute_local_density_all_cells(adata, k_neighbors=K_NEIGHBORS)
    adata = sv.compute_local_density_by_phenotype(
        adata, phenotype_key=PHENOTYPE_KEY, k_neighbors=K_NEIGHBORS)

    print("Computing radius interaction density …")
    sv.compute_tissue_areas(adata)
    adata = sv.detect_edge_cells(adata, radius=RADIUS)
    adata = sv.phenotype_interaction_density(
        adata,
        phenotype_key=PHENOTYPE_KEY,
        source_pheno=SOURCE_PHENO,
        target_pheno=TARGET_PHENO,
        radius=RADIUS,
        exclude_edge=True,
        edge_key="edge_cell",
    )
    print("All computations done.")

    return dict(
        density=density,
        pheno_density=pheno_density,
        corr=corr,
        kde=(Xg, Yg, Z),
    ), adata


# ══════════════════════════════════════════════════════════════════════════════
# Panel drawing functions
# ══════════════════════════════════════════════════════════════════════════════

def draw_schematic(ax, show_label=True):
    """Panel A — 3-part conceptual schematic of density approaches."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    rng = np.random.default_rng(0)

    # ── Column positions (3 equal sections) ──────────────────────────────────
    col_x    = [0.06, 0.38, 0.70]   # left edge of each column
    col_w    = 0.28
    box_top  = 0.88
    diag_y   = 0.28   # top of illustration area
    diag_h   = 0.54   # height of illustration area

    titles   = ["General density\n(tile-based)",
                "KNN local density\n(k-nearest cells)",
                "Radius interaction\n(source ↔ target)"]
    colors   = ["#4a90d9", "#27ae60", "#e07b39"]

    for col, (cx, title, col_color) in enumerate(zip(col_x, titles, colors)):
        cx_mid = cx + col_w / 2

        # ── Column title ─────────────────────────────────────────────────────
        ax.text(cx_mid, box_top, title, ha="center", va="top",
                fontsize=5.2, fontweight="semibold", color=col_color,
                multialignment="center")

        # ── Separator line under title ────────────────────────────────────────
        ax.plot([cx + 0.01, cx + col_w - 0.01], [box_top - 0.08, box_top - 0.08],
                color=col_color, lw=0.6, alpha=0.4)

        # ── Illustration ──────────────────────────────────────────────────────
        iy_top  = box_top - 0.11   # top of illustration
        iy_bot  = diag_y           # bottom of illustration

        if col == 0:
            # Tile grid — occupies left 58% of column; colorbar in right 18%
            n_cols_t, n_rows_t = 5, 5
            tile_area_w = col_w * 0.58
            tw = tile_area_w / n_cols_t
            th = (iy_top - iy_bot) / (n_rows_t + 0.5)
            vals = rng.uniform(0, 1, (n_rows_t, n_cols_t))
            cmap = plt.cm.viridis
            for ri in range(n_rows_t):
                for ci in range(n_cols_t):
                    tx = cx + 0.01 + ci * tw
                    ty = iy_bot + ri * th + 0.02
                    fc = cmap(vals[ri, ci])
                    rect = mpatches.Rectangle((tx, ty), tw * 0.88, th * 0.85,
                                              facecolor=fc, edgecolor="white",
                                              linewidth=0.2)
                    ax.add_patch(rect)
            # Colorbar-like strip — placed in rightmost 18% of column
            cb_x = cx + col_w * 0.80
            cb_h = (iy_top - iy_bot) * 0.65
            cb_y = iy_bot + (iy_top - iy_bot) * 0.18
            for vi, v in enumerate(np.linspace(0, 1, 20)):
                seg_h = cb_h / 20
                rect = mpatches.Rectangle((cb_x, cb_y + vi * seg_h), 0.018, seg_h,
                                          facecolor=cmap(v), linewidth=0)
                ax.add_patch(rect)
            ax.text(cb_x + 0.009, cb_y + cb_h + 0.01, "High",
                    ha="center", va="bottom", fontsize=3.8, color="#444444")
            ax.text(cb_x + 0.009, cb_y - 0.01, "Low",
                    ha="center", va="top", fontsize=3.8, color="#444444")

        elif col == 1:
            # KNN diagram: cells as circles, lines to k nearest
            n_pts = 12
            pts_x = rng.uniform(cx + 0.025, cx + col_w - 0.045, n_pts)
            pts_y = rng.uniform(iy_bot + 0.04, iy_top - 0.04, n_pts)

            # Pick central cell as closest to centroid
            cx_m = pts_x.mean(); cy_m = pts_y.mean()
            dists = np.hypot(pts_x - cx_m, pts_y - cy_m)
            center_idx = int(np.argmin(dists))

            # k nearest to center
            all_dists = np.hypot(pts_x - pts_x[center_idx],
                                  pts_y - pts_y[center_idx])
            all_dists[center_idx] = np.inf
            nn_idx = np.argsort(all_dists)[:K_NEIGHBORS]
            max_nn_dist = all_dists[nn_idx].max()

            # Background cells
            for i in range(n_pts):
                if i == center_idx:
                    continue
                fc = "#4a90d9" if i in nn_idx else "#aaaaaa"
                circle = mpatches.Circle((pts_x[i], pts_y[i]), 0.012,
                                          facecolor=fc, edgecolor="white",
                                          linewidth=0.3, zorder=3)
                ax.add_patch(circle)

            # Lines from center to k nearest
            for i in nn_idx:
                ax.plot([pts_x[center_idx], pts_x[i]],
                        [pts_y[center_idx], pts_y[i]],
                        color="#4a90d9", lw=0.7, ls="--", alpha=0.7, zorder=2)

            # Mean distance circle
            circ = mpatches.Circle((pts_x[center_idx], pts_y[center_idx]),
                                    max_nn_dist * 0.7,
                                    facecolor="none", edgecolor="#4a90d9",
                                    linewidth=0.7, ls=":", zorder=2)
            ax.add_patch(circ)

            # Central cell
            c_circ = mpatches.Circle((pts_x[center_idx], pts_y[center_idx]),
                                      0.016, facecolor="#e07b39",
                                      edgecolor="white", linewidth=0.4, zorder=4)
            ax.add_patch(c_circ)

            # Legend text
            ax.text(cx_mid, iy_bot + 0.005, "density = 1 / mean_dist(k)",
                    ha="center", va="bottom", fontsize=3.8, color="#555555",
                    style="italic")

        else:  # col == 2 — radius interaction
            n_bg  = 14
            n_src = 1
            bg_x  = rng.uniform(cx + 0.02, cx + col_w - 0.03, n_bg)
            bg_y  = rng.uniform(iy_bot + 0.03, iy_top - 0.03, n_bg)

            # Background cells
            for i in range(n_bg):
                ax.add_patch(mpatches.Circle((bg_x[i], bg_y[i]), 0.010,
                                              facecolor="#cccccc", edgecolor="white",
                                              linewidth=0.2, zorder=2))

            # Source cell in center
            src_x, src_y = cx_mid - 0.02, (iy_bot + iy_top) / 2 + 0.02
            r_draw = 0.10   # visual radius

            radius_circ = mpatches.Circle((src_x, src_y), r_draw,
                                           facecolor="#e07b3915",
                                           edgecolor="#e07b39", linewidth=0.8,
                                           ls="--", zorder=3)
            ax.add_patch(radius_circ)

            # Mark target cells inside radius
            inside = np.hypot(bg_x - src_x, bg_y - src_y) < r_draw
            for i, ins in enumerate(inside):
                if ins:
                    ax.add_patch(mpatches.Circle((bg_x[i], bg_y[i]), 0.012,
                                                  facecolor="#27ae60",
                                                  edgecolor="white",
                                                  linewidth=0.3, zorder=4))

            # Source cell on top
            ax.add_patch(mpatches.Circle((src_x, src_y), 0.016,
                                          facecolor="#e07b39", edgecolor="white",
                                          linewidth=0.4, zorder=5))

            # Radius label
            ax.annotate("", xy=(src_x + r_draw, src_y),
                        xytext=(src_x, src_y),
                        arrowprops=dict(arrowstyle="-|>", color="#e07b39",
                                        lw=0.6), zorder=6)
            ax.text(src_x + r_draw / 2, src_y + 0.012, f"r = {RADIUS} px",
                    ha="center", va="bottom", fontsize=3.8, color="#e07b39")

            # Legend
            leg_y = iy_bot + 0.005
            for lx, lc, lt in [(cx + 0.02, "#e07b39", "Tumour (source)"),
                                (cx + 0.12, "#27ae60", "CD4 T (target)"),
                                (cx + 0.22, "#cccccc", "Other")]:
                ax.add_patch(mpatches.Circle((lx, leg_y + 0.008), 0.008,
                                              facecolor=lc, edgecolor="none"))
                ax.text(lx + 0.014, leg_y + 0.008, lt,
                        va="center", fontsize=3.5, color="#555555")

    prefix = "A   " if show_label else ""
    ax.set_title(f"{prefix}Density analysis framework",
                 fontsize=6.5, fontweight="semibold", pad=3, loc="left")


def _tile_heatmap(ax, density_df, value, cbar_label, title, imageid=IMAGE_ID):
    """Shared helper: pivot density_df and draw seaborn heatmap."""
    df  = density_df[density_df["imageid"] == imageid]
    hm  = df.pivot(index="tile_y", columns="tile_x", values=value)
    sns.heatmap(
        hm, ax=ax, cmap="viridis", square=True,
        xticklabels=False, yticklabels=False,
        cbar_kws={"label": cbar_label, "shrink": 0.75, "pad": 0.02,
                  "aspect": 25},
    )
    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.ax.tick_params(labelsize=4.5, length=2, width=0.4)
        cbar.set_label(cbar_label, fontsize=5.0, labelpad=2)
        cbar.outline.set_linewidth(0.3)
    ax.set_xlabel("Tile X", fontsize=5.5, labelpad=2)
    ax.set_ylabel("Tile Y", fontsize=5.5, labelpad=2)
    ax.tick_params(length=0)
    ax.set_title(title, fontsize=6.5, fontweight="semibold", pad=3, loc="left")


def draw_cell_count_density(ax, density, show_label=True):
    """Panel B — tile-based cell-count density (all cells)."""
    prefix = "B   " if show_label else ""
    _tile_heatmap(ax, density, "object_density",
                  "Cells / tile area (%)",
                  f"{prefix}Cell count density (all cells)")


def draw_pixel_density(ax, density, show_label=True):
    """Panel C — tile-based pixel-area coverage density (all cells)."""
    prefix = "C   " if show_label else ""
    _tile_heatmap(ax, density, "pixel_density",
                  "Cell area / tile area (%)",
                  f"{prefix}Pixel area coverage (all cells)")


def draw_tumour_density(ax, pheno_density, show_label=True):
    """Panel D — tile-based cell-count density for tumour cells."""
    prefix = "D   " if show_label else ""
    tdf = pheno_density[pheno_density[PHENOTYPE_KEY] == SOURCE_PHENO].copy()
    _tile_heatmap(ax, tdf, "object_density",
                  "Cells / tile area (%)",
                  f"{prefix}Tumour cell density")


def draw_phenotype_correlation(ax, corr, show_label=True):
    """Panel E — cross-phenotype Pearson density correlation matrix."""
    # Rename labels
    idx    = [PHENOTYPE_LABELS.get(c, c) for c in corr.index]
    corr_r = corr.copy()
    corr_r.index   = idx
    corr_r.columns = idx

    mask = np.triu(np.ones_like(corr_r, dtype=bool), k=1)
    sns.heatmap(
        corr_r, ax=ax, mask=mask,
        cmap="RdBu_r", vmin=-1, vmax=1, center=0,
        square=True, annot=False, linewidths=0.3, linecolor="#eeeeee",
        xticklabels=True, yticklabels=True,
        cbar_kws={"label": "Pearson r", "shrink": 0.75, "pad": 0.02,
                  "aspect": 25},
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right",
                       fontsize=4.5)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=4.5, rotation=0)
    ax.tick_params(length=0, pad=1)

    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.ax.tick_params(labelsize=4.5, length=2, width=0.4)
        cbar.set_label("Pearson r", fontsize=5.0, labelpad=2)
        cbar.outline.set_linewidth(0.3)

    prefix = "E   " if show_label else ""
    ax.set_title(f"{prefix}Phenotype density correlation",
                 fontsize=6.5, fontweight="semibold", pad=3, loc="left")


def draw_kde_density(ax, Xg, Yg, Z, show_label=True):
    """Panel F — KDE density landscape for CD4 T cells."""
    im = ax.imshow(
        Z, cmap="viridis", origin="lower",
        extent=[Xg.min(), Xg.max(), Yg.min(), Yg.max()],
        aspect="auto",
    )
    ax.invert_yaxis()
    ax.set_xlabel("X centroid (px)", fontsize=5.5, labelpad=2)
    ax.set_ylabel("Y centroid (px)", fontsize=5.5, labelpad=2)
    ax.tick_params(labelsize=4.5, length=2, pad=1)
    cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02, aspect=25)
    cbar.ax.tick_params(labelsize=4.5, length=2, width=0.4)
    cbar.set_label("Density", fontsize=5.0, labelpad=2)
    cbar.outline.set_linewidth(0.3)

    prefix = "F   " if show_label else ""
    ax.set_title(f"{prefix}KDE density — CD4 T cells",
                 fontsize=6.5, fontweight="semibold", pad=3, loc="left")


def _scatter_density(ax, adata, density_key, phenotype=None,
                     title="", cbar_label="Local density",
                     point_size=0.3, cmap="viridis"):
    """Shared helper: scatter coloured by a density column."""
    mask = adata.obs["imageid"] == IMAGE_ID
    if phenotype is not None:
        mask = mask & (adata.obs[PHENOTYPE_KEY] == phenotype)
    data = adata[mask]

    x   = data.obs["X_centroid"].values
    y   = data.obs["Y_centroid"].values
    val = data.obs[density_key].values

    sc = ax.scatter(x, y, c=val, cmap=cmap, s=point_size, edgecolors="none",
                    rasterized=True)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("X centroid (px)", fontsize=5.5, labelpad=2)
    ax.set_ylabel("Y centroid (px)", fontsize=5.5, labelpad=2)
    ax.tick_params(labelsize=4.5, length=2, pad=1)
    _despine(ax)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02, aspect=25)
    cbar.ax.tick_params(labelsize=4.5, length=2, width=0.4)
    cbar.set_label(cbar_label, fontsize=5.0, labelpad=2)
    cbar.outline.set_linewidth(0.3)

    ax.set_title(title, fontsize=6.5, fontweight="semibold", pad=3, loc="left")


def draw_knn_all_density(ax, adata, show_label=True):
    """Panel G — KNN local density, all cells."""
    prefix = "G   " if show_label else ""
    _scatter_density(ax, adata, "density_all",
                     phenotype=None,
                     title=f"{prefix}KNN local density — all cells",
                     cbar_label="Local density (1/px)",
                     point_size=0.2)


def draw_knn_cd4_density(ax, adata, show_label=True):
    """Panel H — KNN local density, CD4 T cells only."""
    prefix = "H   " if show_label else ""
    _scatter_density(ax, adata, "density_pheno",
                     phenotype=TARGET_PHENO,
                     title=f"{prefix}KNN local density — CD4 T cells",
                     cbar_label="Local density (1/px)",
                     point_size=1.0)


def draw_interaction_density(ax, adata, show_label=True):
    """Panel I — radius-based tumour ↔ CD4 T cell interaction density."""
    feature  = f"interaction_density__{SOURCE_PHENO}__to__{TARGET_PHENO}"
    img_mask = adata.obs["imageid"] == IMAGE_ID
    src_mask = img_mask & (adata.obs[PHENOTYPE_KEY] == SOURCE_PHENO)

    img_data = adata[img_mask]
    src_data = adata[src_mask]

    # Background: all cells, grey
    ax.scatter(img_data.obs["X_centroid"], img_data.obs["Y_centroid"],
               color="#cccccc", s=0.2, alpha=0.3, edgecolors="none",
               rasterized=True)

    # Source (tumour) cells coloured by interaction density
    sc = ax.scatter(src_data.obs["X_centroid"], src_data.obs["Y_centroid"],
                    c=src_data.obs[feature], cmap="viridis",
                    s=0.6, edgecolors="none", rasterized=True)

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("X centroid (px)", fontsize=5.5, labelpad=2)
    ax.set_ylabel("Y centroid (px)", fontsize=5.5, labelpad=2)
    ax.tick_params(labelsize=4.5, length=2, pad=1)
    _despine(ax)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02, aspect=25)
    cbar.ax.tick_params(labelsize=4.5, length=2, width=0.4)
    cbar.set_label(f"CD4 T cells / area (r={RADIUS}px)", fontsize=4.5, labelpad=2)
    cbar.outline.set_linewidth(0.3)

    prefix = "I   " if show_label else ""
    ax.set_title(f"{prefix}Tumour ↔ CD4 T cell interaction density",
                 fontsize=6.5, fontweight="semibold", pad=3, loc="left")


# ══════════════════════════════════════════════════════════════════════════════
# Standalone panel export
# ══════════════════════════════════════════════════════════════════════════════

_PANEL_SIZES = {
    "A": (87,  55),  # 3-part schematic
    "B": (45,  47),  # cell count heatmap
    "C": (45,  47),  # pixel area heatmap
    "D": (45,  47),  # tumour density heatmap
    "E": (82,  78),  # correlation matrix
    "F": (68,  68),  # KDE
    "G": (72,  70),  # KNN all-cell
    "H": (72,  70),  # KNN CD4 T
    "I": (72,  70),  # interaction density
}


def _save_panel(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"suppfig8_{name}"
    pdf  = OUT_DIR / f"{stem}.pdf"
    png  = OUT_DIR / f"{stem}.png"
    fig.savefig(pdf, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {pdf.name}")


def make_standalone_panels():
    """Save each panel A–I as individual PDF + PNG."""
    _set_pub_rc()

    adata        = load_data()
    computed, adata = compute_all(adata)
    density      = computed["density"]
    pheno_density = computed["pheno_density"]
    corr         = computed["corr"]
    Xg, Yg, Z   = computed["kde"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\nSaving standalone panels:")

    def _fig(panel, axes_rect=None):
        w, h = _PANEL_SIZES[panel]
        fig  = plt.figure(figsize=(w * MM2IN, h * MM2IN), facecolor="white")
        if axes_rect is None:
            ax = fig.add_axes([0.14, 0.14, 0.72, 0.74])
        else:
            ax = fig.add_axes(axes_rect)
        return fig, ax

    # Panel A — schematic (no standard axes box)
    w, h = _PANEL_SIZES["A"]
    fig  = plt.figure(figsize=(w * MM2IN, h * MM2IN), facecolor="white")
    ax   = fig.add_axes([0.01, 0.06, 0.98, 0.88])
    draw_schematic(ax, show_label=False)
    _save_panel(fig, "A")

    # Panel B — cell count density
    fig, ax = _fig("B")
    draw_cell_count_density(ax, density, show_label=False)
    _save_panel(fig, "B")

    # Panel C — pixel area density
    fig, ax = _fig("C")
    draw_pixel_density(ax, density, show_label=False)
    _save_panel(fig, "C")

    # Panel D — tumour density
    fig, ax = _fig("D")
    draw_tumour_density(ax, pheno_density, show_label=False)
    _save_panel(fig, "D")

    # Panel E — correlation matrix (needs more bottom/left margin for labels)
    fig, ax = _fig("E", axes_rect=[0.22, 0.22, 0.62, 0.66])
    draw_phenotype_correlation(ax, corr, show_label=False)
    _save_panel(fig, "E")

    # Panel F — KDE
    fig, ax = _fig("F")
    draw_kde_density(ax, Xg, Yg, Z, show_label=False)
    _save_panel(fig, "F")

    # Panel G — KNN all-cell
    fig, ax = _fig("G")
    draw_knn_all_density(ax, adata, show_label=False)
    _save_panel(fig, "G")

    # Panel H — KNN CD4 T cells
    fig, ax = _fig("H")
    draw_knn_cd4_density(ax, adata, show_label=False)
    _save_panel(fig, "H")

    # Panel I — interaction density
    fig, ax = _fig("I")
    draw_interaction_density(ax, adata, show_label=False)
    _save_panel(fig, "I")

    print(f"\nAll panels saved to {OUT_DIR}/")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    make_standalone_panels()
