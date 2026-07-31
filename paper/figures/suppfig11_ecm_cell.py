"""
Supplementary Figure 11 — ECM-cell spatial analysis (RA/OA synovial tissue)
=============================================================================
Panels A–B (original Supp Fig 5A–B: fiber UMAPs) are held by the user.
This script generates Panels C–N from pre-computed CSV tables.

Panel layout
------------
Block 2 — COL6 dark-zone biology
  C  Spatial overlay: cells coloured by COL6-dark-zone region (JRP112)
  D  Cell fraction by dark-zone region, RA vs OA
  E  Area-normalised cell density by dark-zone region, RA vs OA
  E2 Fisher's exact test enrichment heatmap (cell types × dark-zone region)

Block 3 — Cell-ECM spatial relationships
  F  Cell-to-fiber nearest distance (phenotype × fiber type)
  G  Fiber density near cells (COL6A1 and CHP, by dark-zone region)
  H  Cross Ripley's K curves (cell-fiber co-localisation, 8 pairs)

Block 4 — ECM spatial organisation and coupling  [emphasis on Ripley/Moran]
  I  Fiber Moran's I bars (RA vs OA, faceted by morphological feature)
  J  RA-minus-OA Moran's I delta heatmap (fiber type × feature)
  K  Local Moran's I spatial map (COL6A1 alignment hotspots, JRP112)
  L  Global cross Moran's I barplot (ECM feature × cell-type coupling)
  M  Local cross Moran's I spatial map (CHP area × monocytes, JRP112)
  N  ECM-cell neighbourhood profiling heatmap (8 cluster types)

All pre-computed results live in:
  paper/notebooks/results/ra_oa_ecm_cell/spatioev_module_paper_applications/tables/
  paper/notebooks/results/ra_oa_ecm_cell/chp_density_micro_holes_col6_dark_segmentation/outputs/

Run from SpatioEv root:
    python notebooks/suppfig11_ecm_cell.py
"""

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR",  "/private/tmp/numba")

ROOT = Path(__file__).parent.parent
os.chdir(ROOT)

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import anndata as ad

# ── Paths ──────────────────────────────────────────────────────────────────────
RESULTS_DIR  = ROOT / "notebooks" / "results" / "ra_oa_ecm_cell"
DARK_DIR     = RESULTS_DIR / "chp_density_micro_holes_col6_dark_segmentation"
DARK_OUT     = DARK_DIR / "outputs"
TABLES_DIR   = RESULTS_DIR / "spatioev_module_paper_applications" / "tables"
OUT_DIR      = ROOT / "notebooks" / "results" / "suppfig11"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MM2IN = 1 / 25.4

# ── Publication RC ─────────────────────────────────────────────────────────────
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
    "pdf.fonttype":        42,
    "svg.fonttype":        "none",
})

# ── Shared palettes and constants ─────────────────────────────────────────────
PIXEL_SIZE_UM = 0.325
IMAGE_KEY     = "imageid"
GROUP_KEY     = "pathology"
PHENOTYPE_KEY = "phenotype"
FIBER_TYPE_KEY = "fiber_type"
X_KEY = "X_centroid"
Y_KEY = "Y_centroid"
QC_IMAGE = "JRP112"
GROUP_ORDER = ["RA", "OA"]
GROUP_PALETTE = {"RA": "#d62728", "OA": "#1f77b4"}

REGION_ORDER = ["core", "inner_edge", "outer_edge", "background", "outside_tissue"]
REGION_PALETTE = {
    "core":           "#2b6cb0",
    "inner_edge":     "#f6ad55",
    "outer_edge":     "#c53030",
    "background":     "#cbd5e0",
    "outside_tissue": "#edf2f7",
}

PAPER_CELL_TYPES = [
    "B cells", "CD4 T cells", "CD8 T cells", "T cells",
    "Dendritic cells", "Macrophages", "MERTK+ Macrophages",
    "Monocytes", "Neutrophils", "Vascular cells",
]
PAPER_FIBER_TYPES = ["COL6A1", "CHP", "COL4A1", "FN", "COL1A1bis"]

FIBER_PALETTE = {
    "COL6A1":    "#4e9ab4",
    "CHP":       "#e07b39",
    "COL4A1":    "#6aab6e",
    "FN":        "#9b6bbf",
    "COL1A1bis": "#c94f6d",
}


def _despine(ax):
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)


def _save(fig, name):
    stem = OUT_DIR / f"suppfig11_{name}"
    fig.savefig(f"{stem}.pdf", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: suppfig11_{name}.pdf")


def _placeholder_ax(ax, msg):
    ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center",
            fontsize=5.5, color="#aaaaaa",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f7f7f7",
                      edgecolor="#cccccc", linewidth=0.4))
    ax.axis("off")


# ══════════════════════════════════════════════════════════════════════════════
# Block 2 — COL6 dark-zone biology
# ══════════════════════════════════════════════════════════════════════════════

def make_panel_C():
    """Spatial overlay: cells coloured by COL6-dark-zone region (JRP112)."""
    # Load cell coordinates (obs-only, memory-efficient)
    adata_obs = ad.read_h5ad(ROOT / "data" / "RA_OA" / "tmp6.h5ad", backed="r").obs.copy()

    # Load dark-zone cell assignments
    cell_dark = pd.read_csv(DARK_OUT / "04_cell_chp_micro_hole_region_assignments.csv")
    cell_dark = cell_dark[["cell_id", "COL6_dark_region"]].drop_duplicates("cell_id")
    adata_obs = adata_obs.join(
        cell_dark.set_index("cell_id")["COL6_dark_region"], how="left")
    adata_obs["COL6_dark_region"] = adata_obs["COL6_dark_region"].fillna("outside_tissue")

    sub = adata_obs[adata_obs[IMAGE_KEY] == QC_IMAGE].copy()

    fig, ax = plt.subplots(figsize=(48 * MM2IN, 47 * MM2IN), facecolor="white")
    for region in REGION_ORDER:
        m = sub["COL6_dark_region"] == region
        if m.any():
            ax.scatter(sub.loc[m, X_KEY], sub.loc[m, Y_KEY],
                       c=REGION_PALETTE[region],
                       s=0.15,
                       alpha=0.25 if region == "outside_tissue" else 0.8,
                       edgecolors="none", rasterized=True,
                       label=region.replace("_", " "))

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("X centroid (px)", fontsize=5.5, labelpad=2)
    ax.set_ylabel("Y centroid (px)", fontsize=5.5, labelpad=2)
    ax.tick_params(labelsize=4.5, length=2, pad=1)
    handles = [mpatches.Patch(facecolor=REGION_PALETTE[r], label=r.replace("_", " "))
               for r in REGION_ORDER if r in sub["COL6_dark_region"].values]
    ax.legend(handles=handles, fontsize=3.8, loc="upper right",
              frameon=True, facecolor="white", edgecolor="none",
              framealpha=0.7,
              title="COL6 region", title_fontsize=4.0,
              borderpad=0.4, handlelength=1.0, handletextpad=0.4)
    ax.set_title(f"COL6 dark-zone regions ({QC_IMAGE})",
                 fontsize=6.5, fontweight="semibold", pad=3, loc="left")
    _despine(ax)
    fig.tight_layout()
    _save(fig, "C")


def make_panel_D():
    """Cell fraction by COL6-dark-zone region, RA vs OA."""
    df = pd.read_csv(TABLES_DIR / "01_col6_dark_region_cell_fraction_by_phenotype.csv")
    df = df[df[PHENOTYPE_KEY].isin(PAPER_CELL_TYPES)].copy()
    df["COL6_dark_region"] = pd.Categorical(
        df["COL6_dark_region"], categories=REGION_ORDER[:4], ordered=True)
    df = df[df["COL6_dark_region"].notna()]

    col = "cell_fraction" if "cell_fraction" in df.columns else df.columns[-1]

    g = sns.FacetGrid(
        df, col=PHENOTYPE_KEY, col_wrap=5,
        height=28 * MM2IN, aspect=1.0,
        sharey=False,
    )
    g.map_dataframe(
        sns.barplot, x="COL6_dark_region", y=col,
        hue=GROUP_KEY, hue_order=GROUP_ORDER,
        palette=GROUP_PALETTE, linewidth=0.4,
        errorbar="se", order=REGION_ORDER[:4],
    )
    region_labels = [r.replace("_", " ") for r in REGION_ORDER[:4]]
    for ax in g.axes.flat:
        ax.set_xticks(range(len(region_labels)))
        ax.set_xticklabels(region_labels, rotation=30, ha="right", fontsize=4.0)
        ax.tick_params(labelbottom=True, labelsize=4.0, length=2, pad=1)
        ax.axhline(0, color="black", lw=0.3)
        _despine(ax)
    g.set_axis_labels("Region", "Cell fraction", fontsize=5.5)
    g.set_titles("{col_name}", size=5.0)
    g.fig.suptitle("Cell fraction by COL6 dark-zone region",
                   fontsize=6.5, fontweight="semibold", x=0.02, ha="left")
    g.fig.subplots_adjust(top=0.88, hspace=0.55, wspace=0.35)

    # Add shared legend
    handles = [mpatches.Patch(facecolor=GROUP_PALETTE[g_], label=g_)
               for g_ in GROUP_ORDER]
    g.fig.legend(handles=handles, fontsize=4.5, frameon=False,
                 loc="upper right", bbox_to_anchor=(1.0, 1.0),
                 title="Pathology", title_fontsize=5.0)
    _save(g.fig, "D")


def make_panel_E():
    """Area-normalised cell density by dark-zone region, RA vs OA."""
    df = pd.read_csv(TABLES_DIR / "02_col6_dark_region_cell_density_per_area.csv")
    df = df[df[PHENOTYPE_KEY].isin(PAPER_CELL_TYPES)].copy()
    dens_col = next((c for c in df.columns if "density" in c.lower() or "per" in c.lower()),
                    df.columns[-1])
    df["COL6_dark_region"] = pd.Categorical(
        df.get("COL6_dark_region", df.get("region", "background")),
        categories=REGION_ORDER[:4], ordered=True)
    df = df[df["COL6_dark_region"].notna()]

    g = sns.FacetGrid(
        df, col=PHENOTYPE_KEY, col_wrap=5,
        height=28 * MM2IN, aspect=1.0,
        sharey=False,
    )
    g.map_dataframe(
        sns.barplot, x="COL6_dark_region", y=dens_col,
        hue=GROUP_KEY, hue_order=GROUP_ORDER,
        palette=GROUP_PALETTE, linewidth=0.4,
        errorbar="se", order=REGION_ORDER[:4],
    )
    region_labels = [r.replace("_", " ") for r in REGION_ORDER[:4]]
    for ax in g.axes.flat:
        ax.set_xticks(range(len(region_labels)))
        ax.set_xticklabels(region_labels, rotation=30, ha="right", fontsize=4.0)
        ax.tick_params(labelbottom=True, labelsize=4.0, length=2, pad=1)
        _despine(ax)
    g.set_axis_labels("Region", "Cells / mm²", fontsize=5.5)
    g.set_titles("{col_name}", size=5.0)
    g.fig.suptitle("Area-normalised cell density by COL6 dark-zone region",
                   fontsize=6.5, fontweight="semibold", x=0.02, ha="left")
    g.fig.subplots_adjust(top=0.88, hspace=0.55, wspace=0.35)
    handles = [mpatches.Patch(facecolor=GROUP_PALETTE[g_], label=g_)
               for g_ in GROUP_ORDER]
    g.fig.legend(handles=handles, fontsize=4.5, frameon=False,
                 loc="upper right", bbox_to_anchor=(1.0, 1.0),
                 title="Pathology", title_fontsize=5.0)
    _save(g.fig, "E")


def make_panel_E2():
    """Fisher's exact test enrichment heatmap: cell types in COL6 dark-zone regions."""
    from statsmodels.stats.multitest import multipletests  # already installed in spatioev_env

    df = pd.read_csv(TABLES_DIR / "07_fisher_enrichment_dark_zones.csv")

    region_order = ["core", "inner_edge", "outer_edge"]
    ct_order = [c for c in PAPER_CELL_TYPES if c in df[PHENOTYPE_KEY].values]
    rlabels  = ["Core", "Inner edge", "Outer edge"]

    # Mean log2(OR) and fraction-significant per cell type × region
    pivot_OR  = (
        df.groupby([PHENOTYPE_KEY, "region"], observed=True)["log2_OR"]
        .mean()
        .unstack("region")
        .reindex(index=ct_order, columns=region_order)
    )
    pivot_sig = (
        df.groupby([PHENOTYPE_KEY, "region"], observed=True)["significant"]
        .mean()
        .unstack("region")
        .reindex(index=ct_order, columns=region_order)
    )

    vmax = max(float(np.nanpercentile(np.abs(pivot_OR.values[~np.isnan(pivot_OR.values)]), 95)), 1.0)

    fig, ax = plt.subplots(figsize=(42 * MM2IN, 55 * MM2IN), facecolor="white")
    im = ax.imshow(pivot_OR.values, cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, aspect="auto", interpolation="none")

    # Grid lines
    n_ct, n_reg = pivot_OR.shape
    for x in np.arange(-0.5, n_reg, 1):
        ax.axvline(x, color="white", linewidth=0.5)
    for y in np.arange(-0.5, n_ct, 1):
        ax.axhline(y, color="white", linewidth=0.5)

    # Significance markers (* where majority of samples are significant)
    for i, ct in enumerate(ct_order):
        for j, region in enumerate(region_order):
            try:
                frac = float(pivot_sig.at[ct, region])
            except (KeyError, ValueError):
                frac = 0.0
            if frac > 0.5:
                ax.text(j, i, "*", ha="center", va="center",
                        fontsize=7, color="black", fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.65, aspect=20, pad=0.03)
    cbar.ax.tick_params(labelsize=4.5, length=2, width=0.4)
    cbar.set_label("Mean log2(OR) vs background", fontsize=5.0, labelpad=2)
    cbar.outline.set_linewidth(0.3)

    ax.set_xticks(range(n_reg))
    ax.set_xticklabels(rlabels, rotation=30, ha="right", fontsize=5.0)
    ax.set_yticks(range(n_ct))
    ax.set_yticklabels(ct_order, fontsize=5.0)
    ax.tick_params(length=0, pad=2)
    ax.set_xlabel("Dark zone region", fontsize=5.5, labelpad=3)
    ax.set_title("Cell-type enrichment in COL6 dark zones\n(Fisher's exact test, * = FDR < 0.05 in >50% of samples)",
                 fontsize=6.5, fontweight="semibold", pad=3, loc="left")

    fig.tight_layout()
    _save(fig, "E2")


# ══════════════════════════════════════════════════════════════════════════════
# Block 3 — Cell-ECM spatial relationships
# ══════════════════════════════════════════════════════════════════════════════

def make_panel_F():
    """Cell-to-fiber nearest distance (phenotype × fiber type)."""
    df = pd.read_csv(TABLES_DIR / "03b_cell_to_fiber_nearest_distance_by_sample.csv")
    dist_col = next((c for c in df.columns if "dist" in c.lower() or "distance" in c.lower()
                     and "mean" in c.lower()), None)
    if dist_col is None:
        # Fall back to first numeric column that isn't obvious metadata
        numeric = df.select_dtypes("number").columns
        dist_col = [c for c in numeric if c not in [IMAGE_KEY, GROUP_KEY, PHENOTYPE_KEY,
                    FIBER_TYPE_KEY, "n_cells", "n_fibers"]]
        dist_col = dist_col[0] if dist_col else numeric[0]

    df = df[df[PHENOTYPE_KEY].isin(PAPER_CELL_TYPES)].copy()
    df["dist_um"] = df[dist_col] * PIXEL_SIZE_UM

    fiber_order = [f for f in PAPER_FIBER_TYPES if f in df[FIBER_TYPE_KEY].unique()]

    fig, axes = plt.subplots(1, len(fiber_order),
                             figsize=(len(fiber_order) * 30 * MM2IN, 60 * MM2IN),
                             sharey=False, facecolor="white")
    if len(fiber_order) == 1:
        axes = [axes]

    for ax, ftype in zip(axes, fiber_order):
        sub = df[df[FIBER_TYPE_KEY] == ftype]
        sns.boxplot(
            data=sub, x=PHENOTYPE_KEY, y="dist_um",
            hue=GROUP_KEY, hue_order=[g for g in GROUP_ORDER if g in sub[GROUP_KEY].values],
            palette=GROUP_PALETTE, linewidth=0.4, fliersize=1.0, ax=ax,
            order=[p for p in PAPER_CELL_TYPES if p in sub[PHENOTYPE_KEY].values],
        )
        ax.set_title(ftype, fontsize=5.5, pad=2,
                     color=FIBER_PALETTE.get(ftype, "#333333"), fontweight="semibold")
        ax.set_xlabel("")
        ax.set_ylabel("Nearest distance (µm)" if ax == axes[0] else "",
                      fontsize=5.5, labelpad=2)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=4.0)
        ax.tick_params(labelsize=4.5, length=2, pad=1)
        ax.get_legend().remove() if ax.get_legend() else None
        _despine(ax)

    axes[0].figure.suptitle("Cell-to-fiber nearest distance",
                             fontsize=6.5, fontweight="semibold", x=0.02, ha="left", y=1.02)
    handles = [mpatches.Patch(facecolor=GROUP_PALETTE[g_], label=g_) for g_ in GROUP_ORDER]
    axes[-1].legend(handles=handles, fontsize=4.0, frameon=False,
                    loc="upper right", title="Pathology", title_fontsize=4.5)
    fig.tight_layout()
    _save(fig, "F")


def make_panel_G():
    """Fiber density near cells (COL6A1 + CHP, by dark-zone region)."""
    df = pd.read_csv(TABLES_DIR / "04_fiber_density_near_cells_by_region.csv")
    df = df[df[PHENOTYPE_KEY].isin(PAPER_CELL_TYPES)].copy()

    dens_col = next((c for c in df.columns if "density" in c.lower()), None)
    if dens_col is None:
        numeric = [c for c in df.select_dtypes("number").columns
                   if c not in [IMAGE_KEY, GROUP_KEY, PHENOTYPE_KEY, FIBER_TYPE_KEY]]
        dens_col = numeric[0] if numeric else df.columns[-1]

    fiber_show = [f for f in ["COL6A1", "CHP"] if f in df[FIBER_TYPE_KEY].unique()]
    region_show = [r for r in REGION_ORDER[:4] if r in df.get("COL6_dark_region",
                   df.get("region", pd.Series())).values]

    g = sns.FacetGrid(
        df[df[FIBER_TYPE_KEY].isin(fiber_show)],
        col=FIBER_TYPE_KEY, col_order=fiber_show,
        height=40 * MM2IN, aspect=1.75,  # 4/5 height (50→40), aspect scaled to keep width constant
        sharey=False,
    )

    def _strip_box(data, x, y, hue, **kwargs):
        sns.boxplot(data=data, x=x, y=y, hue=hue,
                    hue_order=[g_ for g_ in GROUP_ORDER if g_ in data[hue].values],
                    palette=GROUP_PALETTE, linewidth=0.4, fliersize=1.0,
                    order=[p for p in PAPER_CELL_TYPES if p in data[x].values],
                    **{k: v for k, v in kwargs.items()
                       if k in ["ax", "order", "hue_order"]})

    g.map_dataframe(_strip_box, x=PHENOTYPE_KEY, y=dens_col, hue=GROUP_KEY)
    for ax in g.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=4.0)
        ax.tick_params(labelsize=4.5, length=2, pad=1)
        _despine(ax)
    g.set_axis_labels(PHENOTYPE_KEY, "Fiber density (a.u.)", fontsize=5.5)
    g.set_titles("{col_name}", size=5.5)
    g.fig.suptitle("Fiber density near cells",
                   fontsize=6.5, fontweight="semibold", x=0.02, ha="left")
    g.fig.subplots_adjust(top=0.88)
    handles = [mpatches.Patch(facecolor=GROUP_PALETTE[g_], label=g_) for g_ in GROUP_ORDER]
    g.fig.legend(handles=handles, fontsize=4.5, frameon=False,
                 loc="upper right", bbox_to_anchor=(1.0, 1.0),
                 title="Pathology", title_fontsize=5.0)
    _save(g.fig, "G")


def make_panel_H():
    """Cross Ripley's K curves — embed pre-computed figure (CSV was not persisted)."""
    import matplotlib.image as mpimg

    FIGS_DIR = RESULTS_DIR / "spatioev_module_paper_applications" / "figures"
    png_path = FIGS_DIR / "05_cross_ripleys_k_cell_fiber_curves.png"

    img = mpimg.imread(str(png_path))
    h_px, w_px = img.shape[:2]
    target_w = 80 * MM2IN            # 1/2 of original 160 mm
    target_h = target_w * h_px / w_px

    fig, ax = plt.subplots(figsize=(target_w, target_h), facecolor="white")
    ax.imshow(img, aspect="equal", interpolation="lanczos")
    ax.axis("off")
    ax.set_title("Cross Ripley's K: cell–fiber co-localisation",
                 fontsize=6.5, fontweight="semibold", pad=3, loc="left")
    fig.tight_layout(pad=0.2)
    _save(fig, "H")


# ══════════════════════════════════════════════════════════════════════════════
# Block 4 — ECM spatial organisation and coupling
# ══════════════════════════════════════════════════════════════════════════════

def make_panel_I():
    """Fiber Moran's I bars — spatial autocorrelation, RA vs OA, faceted by feature."""
    df = pd.read_csv(TABLES_DIR / "08_fiber_morans_i_by_sample_type_feature.csv")
    df = df[df[FIBER_TYPE_KEY].isin(PAPER_FIBER_TYPES)].copy()
    features = [f for f in ["area", "major_axis_length", "eccentricity", "alignment_score"]
                if f in df["feature"].values]
    df = df[df["feature"].isin(features)]
    group_vals = [g for g in GROUP_ORDER if g in df[GROUP_KEY].values]

    fiber_order = [f for f in PAPER_FIBER_TYPES if f in df[FIBER_TYPE_KEY].values]
    g = sns.FacetGrid(
        df, col="feature", col_order=features, col_wrap=2,
        height=28 * MM2IN, aspect=0.87, sharey=False,   # 2/3 width and height (42→28, aspect 1.3×2/3≈0.87)
    )
    g.map_dataframe(
        sns.barplot, x=FIBER_TYPE_KEY, y="morans_i",
        hue=GROUP_KEY, hue_order=group_vals,
        palette=GROUP_PALETTE, linewidth=0.4, errorbar="se",
        order=fiber_order,
    )
    for i, ax in enumerate(g.axes.flat):
        ax.axhline(0, color="#333333", linestyle="--", linewidth=0.5)
        ax.set_xticks(range(len(fiber_order)))
        ax.set_xticklabels(fiber_order, rotation=30, ha="right", fontsize=4.0)
        ax.tick_params(labelbottom=True, labelsize=4.5, length=2, pad=1)
        ax.set_xlabel("")
        # Only left column (even indices in col_wrap=2 grid) gets y-axis label
        ax.set_ylabel("Moran's I" if i % 2 == 0 else "", fontsize=5.5, labelpad=2)
        _despine(ax)
    g.set_titles("{col_name}", size=5.5)
    g.fig.subplots_adjust(top=0.92, hspace=0.65, wspace=0.4)
    handles = [mpatches.Patch(facecolor=GROUP_PALETTE.get(g_, "#888"), label=g_)
               for g_ in group_vals]
    # Place legend bottom-right to avoid clashing with top-right panel title
    g.fig.legend(handles=handles, fontsize=4.5, frameon=False,
                 loc="lower right", bbox_to_anchor=(1.0, 0.02),
                 title="Pathology", title_fontsize=5.0)
    _save(g.fig, "I")


def make_panel_J():
    """RA-minus-OA Moran's I delta heatmap (fiber type × morphological feature)."""
    df = pd.read_csv(TABLES_DIR / "08b_fiber_morans_i_group_delta.csv")
    delta_col = next((c for c in df.columns if "minus" in c.lower() or "delta" in c.lower()), None)
    if delta_col is None:
        # Try to construct from group columns
        gcols = [c for c in df.columns if c not in [FIBER_TYPE_KEY, "feature"]]
        if len(gcols) >= 2:
            df["delta"] = df[gcols[0]] - df[gcols[1]]
            delta_col = "delta"
        else:
            delta_col = gcols[0] if gcols else df.columns[-1]

    df = df[df.get(FIBER_TYPE_KEY, df.columns[0]).isin(PAPER_FIBER_TYPES)
            if FIBER_TYPE_KEY in df.columns else slice(None)].copy()
    heat = df.pivot(index=FIBER_TYPE_KEY, columns="feature", values=delta_col) \
        if FIBER_TYPE_KEY in df.columns and "feature" in df.columns else df

    fig, ax = plt.subplots(figsize=(47 * MM2IN, 45 * MM2IN), facecolor="white")  # 2/3 width (70→47)
    sns.heatmap(
        heat, cmap="coolwarm", center=0, annot=True, fmt=".2f",
        linewidths=0.3, linecolor="#dddddd",
        cbar_kws={"label": "ΔMoran's I (RA − OA)", "shrink": 0.8, "aspect": 20},
        ax=ax,
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=4.5, length=2, width=0.4)
    cbar.set_label("ΔMoran's I (RA − OA)", fontsize=5.0, labelpad=2)
    cbar.outline.set_linewidth(0.3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=4.5)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=4.5)
    ax.tick_params(length=0, pad=1)
    ax.set_xlabel("Fiber morphological feature", fontsize=5.5, labelpad=2)
    ax.set_ylabel("Fiber type", fontsize=5.5, labelpad=2)
    ax.set_title("RA − OA Moran's I delta",
                 fontsize=6.5, fontweight="semibold", pad=3, loc="left")
    fig.tight_layout()
    _save(fig, "J")


def make_panel_K():
    """Local Moran's I spatial map — COL6A1 alignment hotspots in JRP112."""
    df = pd.read_csv(TABLES_DIR / "09_local_moran_col6_alignment_example.csv")

    local_col = next((c for c in df.columns if "local_moran" in c.lower()), None)
    if local_col is None:
        local_col = next((c for c in df.columns if "moran" in c.lower()), None)
    if local_col is None:
        numeric = [c for c in df.select_dtypes("number").columns
                   if c not in [X_KEY, Y_KEY, "label", "k"]]
        local_col = numeric[0] if numeric else df.columns[-1]

    x_col = X_KEY if X_KEY in df.columns else df.select_dtypes("number").columns[0]
    y_col = Y_KEY if Y_KEY in df.columns else df.select_dtypes("number").columns[1]

    vals = df[local_col].values
    vmax = np.nanpercentile(np.abs(vals), 97)

    fig, ax = plt.subplots(figsize=(52 * MM2IN, 52 * MM2IN), facecolor="white")
    sc = ax.scatter(df[x_col], df[y_col],
                    c=vals, cmap="coolwarm",
                    vmin=-vmax, vmax=vmax,
                    s=1.0, edgecolors="none", rasterized=True)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7, aspect=20, pad=0.02)
    cbar.ax.tick_params(labelsize=4.5, length=2, width=0.4)
    cbar.set_label("Local Moran's I", fontsize=5.0, labelpad=2)
    cbar.outline.set_linewidth(0.3)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("X centroid (px)", fontsize=5.5, labelpad=2)
    ax.set_ylabel("Y centroid (px)", fontsize=5.5, labelpad=2)
    ax.tick_params(labelsize=4.5, length=2, pad=1)
    ax.set_title(f"Local Moran's I —\nCOL6A1 alignment ({QC_IMAGE})",
                 fontsize=6.5, fontweight="semibold", pad=3, loc="left")
    _despine(ax)
    fig.tight_layout()
    _save(fig, "K")


def make_panel_L():
    """Global cross Moran's I — ECM feature × cell-type spatial coupling."""
    df = pd.read_csv(TABLES_DIR / "10_cross_morans_i_ecm_cell_coupling.csv")
    cm_col = next((c for c in df.columns if "cross_moran" in c.lower()), None)
    if cm_col is None:
        numeric = [c for c in df.select_dtypes("number").columns
                   if "n_fiber" not in c.lower()]
        cm_col = numeric[0] if numeric else df.columns[-1]

    if "comparison" not in df.columns:
        df["comparison"] = (df.get(FIBER_TYPE_KEY, "") + " / "
                            + df.get(PHENOTYPE_KEY, "") + " / "
                            + df.get("fiber_feature", ""))

    group_vals = [g for g in GROUP_ORDER if g in df[GROUP_KEY].values]
    n_pairs = df["comparison"].nunique()
    fig_w = max(45, n_pairs * 7) * MM2IN   # 1/2 of original width

    fig, ax = plt.subplots(figsize=(fig_w, 55 * MM2IN), facecolor="white")
    sns.barplot(
        data=df, x="comparison", y=cm_col,
        hue=GROUP_KEY, hue_order=group_vals,
        palette=GROUP_PALETTE, linewidth=0.4, errorbar="se", ax=ax,
    )
    ax.axhline(0, color="#333333", linestyle="--", linewidth=0.5)
    ax.set_xlabel("ECM–cell comparison", fontsize=5.5, labelpad=2)
    ax.set_ylabel("Cross Moran's I", fontsize=5.5, labelpad=2)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=4.0)
    ax.tick_params(labelsize=4.5, length=2, pad=1)
    ax.legend(fontsize=4.0, frameon=False, title="Pathology", title_fontsize=4.5)
    ax.set_title("Cross Moran's I: ECM–cell spatial coupling",
                 fontsize=6.5, fontweight="semibold", pad=3, loc="left")
    _despine(ax)
    fig.tight_layout()
    _save(fig, "L")


def make_panel_M():
    """Local cross Moran's I spatial map — CHP area × monocytes (JRP112)."""
    df = pd.read_csv(TABLES_DIR / "11_local_cross_moran_chp_monocyte_example.csv")

    local_col = next((c for c in df.columns if "local_cross" in c.lower()
                      or "cross_moran" in c.lower()), None)
    if local_col is None:
        numeric = [c for c in df.select_dtypes("number").columns
                   if c not in [X_KEY, Y_KEY, "label", "k", "n_neighbors"]]
        local_col = numeric[0] if numeric else df.columns[-1]

    x_col = X_KEY if X_KEY in df.columns else df.select_dtypes("number").columns[0]
    y_col = Y_KEY if Y_KEY in df.columns else df.select_dtypes("number").columns[1]

    vals = df[local_col].values
    vmax = np.nanpercentile(np.abs(vals), 97)

    fig, ax = plt.subplots(figsize=(52 * MM2IN, 52 * MM2IN), facecolor="white")
    sc = ax.scatter(df[x_col], df[y_col],
                    c=vals, cmap="coolwarm",
                    vmin=-vmax, vmax=vmax,
                    s=1.0, edgecolors="none", rasterized=True)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7, aspect=20, pad=0.02)
    cbar.ax.tick_params(labelsize=4.5, length=2, width=0.4)
    cbar.set_label("Local cross Moran's I", fontsize=5.0, labelpad=2)
    cbar.outline.set_linewidth(0.3)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("X centroid (px)", fontsize=5.5, labelpad=2)
    ax.set_ylabel("Y centroid (px)", fontsize=5.5, labelpad=2)
    ax.tick_params(labelsize=4.5, length=2, pad=1)
    ax.set_title(f"Local cross Moran's I —\nCHP area × monocytes ({QC_IMAGE})",
                 fontsize=6.5, fontweight="semibold", pad=3, loc="left")
    _despine(ax)
    fig.tight_layout()
    _save(fig, "M")


def make_panel_N():
    """ECM-cell neighbourhood profiling heatmap (8 cluster types)."""
    df = pd.read_csv(TABLES_DIR / "11_ecm_cell_neighborhood_profiles.csv")
    label_col = "ecm_cell_neighborhood" if "ecm_cell_neighborhood" in df.columns else \
        next((c for c in df.columns if "neighborhood" in c.lower() or "cluster" in c.lower()),
             df.columns[0])

    cell_cols = [c for c in df.columns if c in PAPER_CELL_TYPES
                 or any(p.replace(" ", "_") in c for p in PAPER_CELL_TYPES)]
    if not cell_cols:
        # Fall back to any fraction/proportion columns
        cell_cols = [c for c in df.select_dtypes("number").columns
                     if c not in [label_col, "ecm_cell_neighborhood",
                                  "fiber_density_COL6A1", "fiber_density_CHP",
                                  "col6_dark_score", "n_cells"]]

    if not cell_cols:
        fig, ax = plt.subplots(figsize=(80 * MM2IN, 55 * MM2IN), facecolor="white")
        _placeholder_ax(ax, "ECM-cell neighbourhood heatmap\nNo cell-type columns found in CSV")
        _save(fig, "N")
        return

    agg = df.groupby(label_col)[cell_cols].mean()
    agg = agg.rename(columns={c: c.replace("frac_", "").replace("_", " ") for c in agg.columns})

    n_clusters, n_phenos = agg.shape
    fig, ax = plt.subplots(figsize=(max(53, n_phenos * 5) * MM2IN,   # 2/3 width
                                    max(55, n_clusters * 8) * MM2IN),
                           facecolor="white")
    sns.heatmap(
        agg, cmap="Blues", linewidths=0.2, linecolor="#eeeeee",
        annot=False,
        cbar_kws={"label": "Cell-type fraction", "shrink": 0.6, "aspect": 20},
        ax=ax,
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=4.5, length=2, width=0.4)
    cbar.set_label("Cell-type fraction", fontsize=5.0, labelpad=2)
    cbar.outline.set_linewidth(0.3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=4.5)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=4.5)
    ax.tick_params(length=0, pad=1)
    ax.set_xlabel("Cell type", fontsize=5.5, labelpad=2)
    ax.set_ylabel("ECM-cell neighbourhood cluster", fontsize=5.5, labelpad=2)
    ax.set_title("ECM-cell neighbourhood profiling",
                 fontsize=6.5, fontweight="semibold", pad=3, loc="left")
    fig.tight_layout()
    _save(fig, "N")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Output directory: {OUT_DIR}\n")

    print("Block 2 — COL6 dark-zone biology")
    make_panel_C()
    make_panel_D()
    make_panel_E()
    make_panel_E2()

    print("\nBlock 3 — Cell-ECM spatial relationships")
    make_panel_F()
    make_panel_G()
    make_panel_H()

    print("\nBlock 4 — ECM spatial organisation and coupling")
    make_panel_I()
    make_panel_J()
    make_panel_K()
    make_panel_L()
    make_panel_M()
    make_panel_N()

    print(f"\nDone. All panels saved to {OUT_DIR}/")
