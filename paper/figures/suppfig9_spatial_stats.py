"""
Supplementary Figure 9 — Spatial statistics of the PDAC tumour microenvironment
================================================================================
Panel A : Workflow schematic — three spatial statistics frameworks
Panel B : Ripley's K multi-scale heatmap (all phenotypes × 5 radii)
Panel C : Ripley's K line plot — selected phenotypes across scales
Panel D : Local Ripley B lineage hotspot map (TLS identification)
Panel E : Cross-Ripley all-pairs co-localisation matrix at 50 µm
Panel F : Cross-Ripley curve — ductal epithelium ↔ fibroblasts
Panel G : Cross-Ripley permutation envelope — ductal epithelium ↔ T cells
Panel H : Local Moran's I FAP spatial quadrant map (CAF activation hotspots)
Panel I : Cross-Moran's I matrix + invasion-front spatial map (combined)

Data:
  data/exp_2/34434_1_adata.h5ad
  data/exp_2/34434_1_annotation.csv
  data/exp_2/pixel_features.csv

Run:
    python notebooks/suppfig9_spatial_stats.py

Output: paper/notebooks/results/suppfig9/suppfig9_*.pdf (.png)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import seaborn as sns
import spatioev as sv

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
DATA_PATH  = ROOT / "data" / "exp_2" / "34434_1_adata.h5ad"
ANN_PATH   = ROOT / "data" / "exp_2" / "34434_1_annotation.csv"
PIXEL_PATH = ROOT / "data" / "exp_2" / "pixel_features.csv"
OUT_DIR    = Path(__file__).parent / "results" / "suppfig9"

MM2IN = 1 / 25.4

# ── Analysis constants ─────────────────────────────────────────────────────────
PHENOTYPE_KEY   = "Tier_A"
IMAGE_ID        = "34434_1"
PIXEL_SIZE_UM   = 0.325           # µm per pixel
RADIUS_UM       = 50
RADIUS_PX       = RADIUS_UM / PIXEL_SIZE_UM
RADII_UM        = [20, 50, 100, 150, 200]
RADII_PX        = [r / PIXEL_SIZE_UM for r in RADII_UM]
K_NEIGHBORS     = 8
N_PERM          = 99              # permutation envelope simulations
SOURCE_PHENO    = "pancreatic ductal epithelium"
TARGET_PHENO    = "Fibroblasts"
TARGET_IMMUNE   = "T cells"
B_PHENO         = "B lineage"
FAP_KEY         = "FAP_expr_z"
SRC_FEAT_KEY    = "entropy_z"     # ductal source feature for cross-Moran's I
TGT_FEAT_KEY    = "FAP_expr_z"   # fibroblast target feature for cross-Moran's I
FIB_MARKERS     = ["FAP", "aSMA", "PDPN", "Thy1"]
DUCT_MARKERS    = ["Ki67", "CK19", "NaKATPase"]
MORPH_FEATURES  = ["nc_ratio", "polarity_score", "entropy", "inertia"]
FIB_FEAT_KEYS   = [f"{m}_expr_z" for m in FIB_MARKERS]
DUCT_FEAT_KEYS  = [f"{m}_expr_z" for m in DUCT_MARKERS] + \
                  [f"{f}_z" for f in MORPH_FEATURES]
SELECTED_PHENOS = [SOURCE_PHENO, "Fibroblasts", TARGET_IMMUNE, B_PHENO]

PHENOTYPE_LABELS = {
    "pancreatic ductal epithelium": "Ductal epithelium",
    "Fibroblasts":       "Fibroblasts",
    "T cells":           "T cells",
    "B lineage":         "B lineage",
    "Macrophages":       "Macrophages",
    "Endothelial cells": "Endothelial",
}

LISA_PALETTE = {
    "high-high":    "#b2182b",
    "low-low":      "#2166ac",
    "high-low":     "#f4a582",
    "low-high":     "#92c5de",
    "unclassified": "#dddddd",
}

CROSS_PALETTE = {
    "high-high":    "#b2182b",   # high entropy duct + high FAP stroma = invasion front
    "low-low":      "#2166ac",   # quiescent / normal
    "high-low":     "#fdae61",   # morphologically complex duct, low-FAP stroma
    "low-high":     "#74add1",   # ordered duct, activated neighbouring stroma
    "unclassified": "#dddddd",
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
    print("Loading 34434_1_adata.h5ad ...")
    adata = sv.load_h5ad(str(DATA_PATH))

    ann = pd.read_csv(ANN_PATH, index_col=0)
    ann = ann[[c for c in ["Tier_A", "Tier_B"] if c in ann.columns]]
    adata.obs = adata.obs.join(ann, how="left")

    try:
        pix = pd.read_csv(PIXEL_PATH, index_col=0)
        adata.obs = adata.obs.merge(pix, on=["label", "fov"], how="left")
        pix_cols = pix.columns.tolist()
        mask = adata.obs[pix_cols].notna().all(axis=1)
        print(f"  Pixel features joined: {mask.sum():,} / {len(mask):,} cells retained")
        adata = adata[mask].copy()
    except (FileNotFoundError, OSError, KeyError) as exc:
        print(f"  Pixel features not available ({exc}) — morphological features skipped.")

    print(f"  {adata.n_obs:,} cells loaded")
    return adata


# ── Computation ────────────────────────────────────────────────────────────────
def compute_all(adata):
    """Run all spatial-statistics computations and return (results_dict, adata)."""
    results = {}

    # ── Marker z-scores ────────────────────────────────────────────────────────
    print("Adding marker z-scores ...")
    var_names = list(adata.var_names)
    markers_to_add = [m for m in FIB_MARKERS + DUCT_MARKERS if m in var_names]
    if markers_to_add:
        adata = sv.add_obs_from_var(adata, markers=markers_to_add, zscore=True)
    morph_present = [k for k in MORPH_FEATURES if k in adata.obs.columns]
    if morph_present:
        adata = sv.add_zscore_obs_features(adata, feature_keys=morph_present)

    # ── Ripley's K multi-scale ─────────────────────────────────────────────────
    print("Computing Ripley K (multi-scale) ...")
    ripley_list = []
    for r_px, r_um in zip(RADII_PX, RADII_UM):
        df = sv.tl.ripleys_k_by_phenotype(
            adata, phenotype_key=PHENOTYPE_KEY, radius=r_px,
            x_key="X_centroid", y_key="Y_centroid", image_key="imageid",
        )
        df["radius_um"]    = r_um
        df["radius_label"] = f"{r_um} µm"
        ripley_list.append(df)
    results["ripley_all"] = pd.concat(ripley_list, ignore_index=True)

    # ── Local Ripley (B lineage TLS) ───────────────────────────────────────────
    print("Computing local Ripley (B lineage) ...")
    sv.tl.ripley_local_counts_by_phenotype(
        adata, phenotype_key=PHENOTYPE_KEY, radius=RADIUS_PX,
        x_key="X_centroid", y_key="Y_centroid", image_key="imageid",
        add_to_obs=True,
    )
    hotspot_cols = [c for c in adata.obs.columns if "ripley_local_hotspot" in c]
    results["hotspot_col"] = hotspot_cols[0] if hotspot_cols else None

    # ── Cross-Ripley all-pairs ─────────────────────────────────────────────────
    print("Computing cross-Ripley all-pairs ...")
    results["cross_all"] = sv.tl.cross_ripleys_k_all_pairs(
        adata, phenotype_key=PHENOTYPE_KEY, radius=RADIUS_PX,
        x_key="X_centroid", y_key="Y_centroid", image_key="imageid",
        include_self_pairs=False,
    )

    # ── Cross-Ripley curve: ductal ↔ fibroblasts ───────────────────────────────
    print("Computing cross-Ripley curve (ductal <-> fibroblasts) ...")
    radii_curve = np.linspace(10, 800, 30)
    results["curve_df"] = sv.tl.cross_ripleys_curve_by_phenotype(
        adata, phenotype_key=PHENOTYPE_KEY,
        source_phenotype=SOURCE_PHENO, target_phenotype=TARGET_PHENO,
        radii=radii_curve,
        x_key="X_centroid", y_key="Y_centroid", image_key="imageid",
    )

    # ── Permutation envelope: ductal ↔ T cells ────────────────────────────────
    print(f"Computing permutation envelope (ductal <-> T cells, n={N_PERM}) ...")
    results["perm_df"] = sv.tl.cross_ripley_permutation_envelope(
        adata, phenotype_key=PHENOTYPE_KEY,
        source_phenotype=SOURCE_PHENO, target_phenotype=TARGET_IMMUNE,
        radii=np.array(RADII_PX), n_sim=N_PERM,
        x_key="X_centroid", y_key="Y_centroid", image_key="imageid",
        random_state=42,
    )

    # ── Local Moran's I: FAP on fibroblasts ────────────────────────────────────
    print("Computing local Moran's I (FAP, fibroblasts) ...")
    adata_fib = adata[adata.obs[PHENOTYPE_KEY] == "Fibroblasts"].copy()
    if FAP_KEY in adata_fib.obs.columns:
        adata_fib = sv.tl.add_local_morans_i(
            adata_fib, value_key=FAP_KEY,
            x_key="X_centroid", y_key="Y_centroid", image_key="imageid",
            k=K_NEIGHBORS,
        )
        adata_fib = sv.tl.add_local_morans_i_quadrants(adata_fib, value_key=FAP_KEY)
    results["adata_fib"]   = adata_fib
    results["fap_quad_col"] = f"local_morans_quadrant__{FAP_KEY}"

    # ── Cross-Moran's I: fibroblast markers × ductal features ─────────────────
    print("Computing cross-Moran's I feature matrix ...")
    fib_feats  = [f for f in FIB_FEAT_KEYS  if f in adata.obs.columns]
    duct_feats = [f for f in DUCT_FEAT_KEYS if f in adata.obs.columns]
    if fib_feats and duct_feats:
        raw_matrix = sv.tl.cross_morans_i_feature_matrix(
            adata, phenotype_key=PHENOTYPE_KEY,
            source_phenotype="Fibroblasts",
            target_phenotype=SOURCE_PHENO,
            source_feature_keys=fib_feats,
            target_feature_keys=duct_feats,
            radius=RADIUS_PX, agg="mean",
            x_key="X_centroid", y_key="Y_centroid", image_key="imageid",
            k=K_NEIGHBORS,
        )
        # Average across images if imageid column present
        grp_cols = [c for c in ["source_feature", "target_feature"] if c in raw_matrix.columns]
        results["matrix_df"] = (
            raw_matrix.groupby(grp_cols)["cross_morans_i"].mean().reset_index()
            if grp_cols else raw_matrix
        )
    else:
        print("  Feature columns not found — cross-Moran's I matrix skipped.")
        results["matrix_df"] = None

    # ── Local cross-Moran's I: invasion front ─────────────────────────────────
    src_feat = SRC_FEAT_KEY if SRC_FEAT_KEY in adata.obs.columns else (duct_feats[0] if duct_feats else None)
    tgt_feat = TGT_FEAT_KEY if TGT_FEAT_KEY in adata.obs.columns else (fib_feats[0] if fib_feats else None)
    results["src_feat"] = src_feat
    results["tgt_feat"] = tgt_feat

    if src_feat and tgt_feat:
        print(f"Computing local cross-Moran's I ({src_feat} x {tgt_feat}) ...")
        adata_duct_local = sv.tl.add_local_cross_morans_i_between_phenotypes(
            adata, phenotype_key=PHENOTYPE_KEY,
            source_phenotype=SOURCE_PHENO,
            target_phenotype="Fibroblasts",
            source_feature_key=src_feat,
            target_feature_key=tgt_feat,
            radius=RADIUS_PX, agg="mean",
            x_key="X_centroid", y_key="Y_centroid", image_key="imageid",
            k=K_NEIGHBORS,
        )
        neighbor_col = f"neighbor_mean__{tgt_feat}"
        local_i_col  = next(
            (c for c in adata_duct_local.obs.columns
             if "local_cross_morans_i" in c and src_feat in c),
            None,
        )
        if local_i_col:
            adata_duct_local = sv.tl.add_local_cross_morans_i_quadrants(
                adata_duct_local,
                source_value_key=src_feat,
                target_neighbor_value_key=neighbor_col,
                local_i_key=local_i_col,
            )
        cross_quad_col = next(
            (c for c in adata_duct_local.obs.columns if "cross_morans_quadrant" in c),
            None,
        )
        results["adata_duct_local"] = adata_duct_local
        results["cross_quad_col"]   = cross_quad_col
    else:
        print("  Source/target feature not available — invasion front map skipped.")
        results["adata_duct_local"] = None
        results["cross_quad_col"]   = None

    print("All computations done.")
    return results, adata


# ══════════════════════════════════════════════════════════════════════════════
# Panel drawing functions
# ══════════════════════════════════════════════════════════════════════════════

def draw_schematic(ax, show_label=True):
    """Panel A — 3-column conceptual schematic of spatial statistics approaches."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    rng = np.random.default_rng(7)

    col_x   = [0.04, 0.37, 0.70]
    col_w   = 0.28
    box_top = 0.92
    iy_top  = 0.80
    iy_bot  = 0.22

    titles = ["Self-clustering\n(Ripley's K)",
              "Co-localisation\n(Cross-Ripley K)",
              "Feature coupling\n(Cross-Moran's I)"]
    colors = ["#4a90d9", "#27ae60", "#e07b39"]

    for col, (cx, title, col_color) in enumerate(zip(col_x, titles, colors)):
        cx_mid = cx + col_w / 2

        ax.text(cx_mid, box_top, title, ha="center", va="top",
                fontsize=5.2, fontweight="semibold", color=col_color,
                multialignment="center")
        ax.plot([cx + 0.01, cx + col_w - 0.01],
                [box_top - 0.09, box_top - 0.09],
                color=col_color, lw=0.6, alpha=0.4)

        if col == 0:
            # Random background cells
            bx = rng.uniform(cx + 0.02, cx + col_w - 0.02, 18)
            by = rng.uniform(iy_bot + 0.06, iy_top - 0.06, 18)
            for i in range(18):
                ax.add_patch(mpatches.Circle((bx[i], by[i]), 0.009,
                                              facecolor="#cccccc", edgecolor="none", zorder=2))
            # Cluster of blue cells
            qx, qy = cx_mid - 0.01, (iy_bot + iy_top) / 2 + 0.02
            cl_x = qx + rng.uniform(-0.055, 0.055, 8)
            cl_y = qy + rng.uniform(-0.09, 0.09, 8)
            for i in range(8):
                ax.add_patch(mpatches.Circle((cl_x[i], cl_y[i]), 0.011,
                                              facecolor="#4a90d9", edgecolor="none", zorder=3))
            # Query cell + radius circle
            r_draw = 0.085
            ax.add_patch(mpatches.Circle((qx, qy), 0.014, facecolor="#e07b39",
                                          edgecolor="white", linewidth=0.4, zorder=5))
            ax.add_patch(mpatches.Circle((qx, qy), r_draw,
                                          facecolor="none", edgecolor="#4a90d9",
                                          linewidth=0.8, ls=":", zorder=4))
            ax.text(cx_mid, iy_bot + 0.02, "L(r) - r > 0 = clustered",
                    ha="center", va="bottom", fontsize=3.6, color="#555555", style="italic")

        elif col == 1:
            # Background
            for _ in range(8):
                ax.add_patch(mpatches.Circle(
                    (rng.uniform(cx + 0.02, cx + col_w - 0.02),
                     rng.uniform(iy_bot + 0.06, iy_top - 0.06)),
                    0.009, facecolor="#cccccc", edgecolor="none", zorder=2))
            # Source cells (orange)
            src_x = rng.uniform(cx + 0.04, cx + col_w * 0.55, 5)
            src_y = rng.uniform(iy_bot + 0.10, iy_top - 0.10, 5)
            for i in range(5):
                ax.add_patch(mpatches.Circle((src_x[i], src_y[i]), 0.012,
                                              facecolor="#e07b39", edgecolor="white",
                                              linewidth=0.3, zorder=3))
            # Target cells (green)
            tgt_x = rng.uniform(cx + 0.06, cx + col_w - 0.02, 10)
            tgt_y = rng.uniform(iy_bot + 0.06, iy_top - 0.06, 10)
            for i in range(10):
                ax.add_patch(mpatches.Circle((tgt_x[i], tgt_y[i]), 0.010,
                                              facecolor="#27ae60", edgecolor="white",
                                              linewidth=0.3, zorder=3))
            # Highlighted source + radius
            hx, hy = src_x[2], src_y[2]
            r_d2   = 0.09
            ax.add_patch(mpatches.Circle((hx, hy), 0.015, facecolor="#e07b39",
                                          edgecolor="white", linewidth=0.5, zorder=5))
            ax.add_patch(mpatches.Circle((hx, hy), r_d2,
                                          facecolor="#27ae6012",
                                          edgecolor="#27ae60",
                                          linewidth=0.8, ls="--", zorder=4))
            # Legend
            for lx, lc, lt in [(cx + 0.01, "#e07b39", "Source"),
                                (cx + 0.11, "#27ae60", "Target")]:
                ax.add_patch(mpatches.Circle((lx, iy_bot + 0.015), 0.008,
                                              facecolor=lc, edgecolor="none"))
                ax.text(lx + 0.014, iy_bot + 0.015, lt,
                        va="center", fontsize=3.5, color="#555555")
            ax.text(cx_mid, iy_bot + 0.055, "excess neighbors > 0 = co-enriched",
                    ha="center", va="bottom", fontsize=3.6, color="#555555", style="italic")

        else:   # col == 2: Cross-Moran's I
            # Left half: fibroblast circles with FAP gradient (higher y = higher FAP)
            n_fib = 9
            fib_x = rng.uniform(cx + 0.02, cx + col_w * 0.44, n_fib)
            fib_y = rng.uniform(iy_bot + 0.08, iy_top - 0.06, n_fib)
            fap_norm = (fib_y - fib_y.min()) / (fib_y.ptp() + 1e-9)
            cmap_r = plt.cm.RdBu_r
            for i in range(n_fib):
                ax.add_patch(mpatches.Circle((fib_x[i], fib_y[i]), 0.012,
                                              facecolor=cmap_r(fap_norm[i]),
                                              edgecolor="white", linewidth=0.3, zorder=3))

            # Right half: ductal circles with Ki67 gradient (higher y = higher Ki67)
            n_duct = 9
            duct_x = rng.uniform(cx + col_w * 0.57, cx + col_w - 0.02, n_duct)
            duct_y = rng.uniform(iy_bot + 0.08, iy_top - 0.06, n_duct)
            ki67_norm = (duct_y - duct_y.min()) / (duct_y.ptp() + 1e-9)
            for i in range(n_duct):
                ax.add_patch(mpatches.Circle((duct_x[i], duct_y[i]), 0.012,
                                              facecolor=cmap_r(ki67_norm[i]),
                                              edgecolor="white", linewidth=0.3, zorder=3))

            # Divider line
            ax.plot([cx + col_w * 0.50, cx + col_w * 0.50],
                    [iy_bot + 0.06, iy_top - 0.04],
                    color="#bbbbbb", lw=0.5, ls="--", zorder=1)

            # Bracket indicating coupling
            mid_y = (iy_bot + iy_top) / 2
            ax.annotate("", xy=(cx + col_w * 0.56, mid_y),
                         xytext=(cx + col_w * 0.44, mid_y),
                         arrowprops=dict(arrowstyle="<->", color="#e07b39",
                                         lw=0.7), zorder=6)

            # Labels
            ax.text(cx + col_w * 0.22, iy_top - 0.02, "Fibroblasts\n(FAP)",
                    ha="center", va="top", fontsize=3.5, color="#555555")
            ax.text(cx + col_w * 0.78, iy_top - 0.02, "Ductal\n(Ki67)",
                    ha="center", va="top", fontsize=3.5, color="#555555")
            ax.text(cx_mid, iy_bot + 0.02,
                    "cross-Moran's I > 0 = spatially coupled",
                    ha="center", va="bottom", fontsize=3.6, color="#555555", style="italic")

    prefix = "A   " if show_label else ""
    ax.set_title(f"{prefix}Spatial statistics framework",
                 fontsize=6.5, fontweight="semibold", pad=3, loc="left")


def draw_ripley_heatmap(ax, ripley_all, show_label=True):
    """Panel B — Ripley L(r)-r multi-scale heatmap across all phenotypes."""
    heat_df = (
        ripley_all
        .groupby(["phenotype", "radius_label"])["L_minus_r"]
        .mean()
        .unstack()
        .reindex(columns=[f"{r} µm" for r in RADII_UM])
    )
    heat_df.index = [PHENOTYPE_LABELS.get(p, p) for p in heat_df.index]

    sns.heatmap(
        heat_df, ax=ax, cmap="coolwarm", center=0,
        linewidths=0.5, annot=False,
        cbar_kws={"label": "L(r)−r", "shrink": 0.70, "pad": 0.02, "aspect": 20},
    )
    cbar = ax.collections[0].colorbar
    if cbar:
        cbar.ax.tick_params(labelsize=4.5, length=2, width=0.4)
        cbar.set_label("L(r)−r", fontsize=5.0, labelpad=2)
        cbar.outline.set_linewidth(0.3)
    ax.set_xlabel("Neighbourhood radius", fontsize=5.5, labelpad=2)
    ax.set_ylabel("", labelpad=2)
    ax.tick_params(length=0, pad=1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=4.5)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=4.5)

    prefix = "B   " if show_label else ""
    ax.set_title(f"{prefix}Self-clustering by phenotype and scale",
                 fontsize=6.5, fontweight="semibold", pad=3, loc="left")


def draw_ripley_lineplot(ax, ripley_all, show_label=True):
    """Panel C — L(r)-r line plot for selected phenotypes."""
    plot_df = ripley_all[ripley_all["phenotype"].isin(SELECTED_PHENOS)].copy()
    plot_df["pheno_label"] = plot_df["phenotype"].map(
        lambda p: PHENOTYPE_LABELS.get(p, p))

    pheno_colors = {
        PHENOTYPE_LABELS.get(SOURCE_PHENO, SOURCE_PHENO): "#9467bd",
        "Fibroblasts": "#ff7f0e",
        "T cells":     "#2ca02c",
        "B lineage":   "#1f77b4",
    }

    radius_order = [f"{r} µm" for r in RADII_UM]
    grp = (plot_df
           .groupby(["pheno_label", "radius_label"])["L_minus_r"]
           .mean()
           .reset_index())
    grp["radius_label"] = pd.Categorical(grp["radius_label"],
                                          categories=radius_order, ordered=True)

    for pheno, sub in grp.groupby("pheno_label"):
        sub = sub.sort_values("radius_label")
        color = pheno_colors.get(pheno, "#888888")
        ax.plot(range(len(sub)), sub["L_minus_r"].values,
                "o-", color=color, markersize=2.5, lw=0.9, label=pheno)

    ax.axhline(0, color="black", ls="--", lw=0.7)
    ax.set_xticks(range(len(RADII_UM)))
    ax.set_xticklabels([f"{r} µm" for r in RADII_UM], fontsize=4.5)
    ax.set_xlabel("Neighbourhood radius", fontsize=5.5, labelpad=2)
    ax.set_ylabel("L(r)−r", fontsize=5.5, labelpad=2)
    ax.tick_params(labelsize=4.5, length=2, pad=1)
    ax.legend(fontsize=4.2, loc="upper left", frameon=False)
    _despine(ax)

    prefix = "C   " if show_label else ""
    ax.set_title(f"{prefix}Self-clustering across spatial scales",
                 fontsize=6.5, fontweight="semibold", pad=3, loc="left")


def draw_b_lineage_hotspot(ax, adata, hotspot_col, show_label=True):
    """Panel D — B lineage local Ripley hotspot map (TLS identification)."""
    img_mask = adata.obs["imageid"] == IMAGE_ID
    all_data = adata[img_mask]
    b_mask   = all_data.obs[PHENOTYPE_KEY] == B_PHENO
    b_data   = all_data[b_mask]
    non_b    = all_data[~b_mask]

    ax.scatter(non_b.obs["X_centroid"], non_b.obs["Y_centroid"],
               c="#e0e0e0", s=0.18, alpha=0.4, edgecolors="none", rasterized=True)

    if hotspot_col and hotspot_col in b_data.obs.columns:
        hot = b_data.obs[hotspot_col].fillna(False).astype(bool)
        non_hot = b_data[~hot]
        hot_sub = b_data[hot]
        ax.scatter(non_hot.obs["X_centroid"], non_hot.obs["Y_centroid"],
                   c="#92c5de", s=0.5, edgecolors="none", rasterized=True,
                   label="B lineage (non-hotspot)", zorder=3)
        ax.scatter(hot_sub.obs["X_centroid"], hot_sub.obs["Y_centroid"],
                   c="#b2182b", s=0.5, edgecolors="none", rasterized=True,
                   label=f"B lineage (TLS hotspot, n={hot.sum()})", zorder=4)
    else:
        ax.scatter(b_data.obs["X_centroid"], b_data.obs["Y_centroid"],
                   c="#1f77b4", s=0.5, edgecolors="none", rasterized=True, label="B lineage")

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("X centroid (px)", fontsize=5.5, labelpad=2)
    ax.set_ylabel("Y centroid (px)", fontsize=5.5, labelpad=2)
    ax.tick_params(labelsize=4.5, length=2, pad=1)
    ax.legend(fontsize=4.0, loc="lower right", frameon=False, markerscale=2)
    _despine(ax)

    prefix = "D   " if show_label else ""
    ax.set_title(f"{prefix}B lineage local clustering hotspots (TLS)",
                 fontsize=6.5, fontweight="semibold", pad=3, loc="left")


def draw_cross_ripley_pairs(ax, cross_all, show_label=True):
    """Panel E — Cross-Ripley all-pairs co-localisation matrix at 50 µm."""
    excess_col = next(
        (c for c in ["source_neighbor_excess", "neighbor_excess", "L_minus_r"]
         if c in cross_all.columns), cross_all.columns[-1])

    heat_df = (
        cross_all
        .groupby(["source", "target"])[excess_col]
        .mean()
        .unstack()
    )
    heat_df.index   = [PHENOTYPE_LABELS.get(p, p) for p in heat_df.index]
    heat_df.columns = [PHENOTYPE_LABELS.get(p, p) for p in heat_df.columns]

    sns.heatmap(
        heat_df, ax=ax, cmap="RdBu_r", center=0,
        linewidths=0.4, linecolor="#eeeeee",
        annot=False,
        cbar_kws={"label": "Neighbour excess", "shrink": 0.75,
                  "pad": 0.02, "aspect": 25},
    )
    cbar = ax.collections[0].colorbar
    if cbar:
        cbar.ax.tick_params(labelsize=4.5, length=2, width=0.4)
        cbar.set_label("Neighbour excess", fontsize=5.0, labelpad=2)
        cbar.outline.set_linewidth(0.3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=4.5)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=4.5)
    ax.tick_params(length=0, pad=1)
    ax.set_xlabel("Target phenotype", fontsize=5.5, labelpad=2)
    ax.set_ylabel("Source phenotype", fontsize=5.5, labelpad=2)

    prefix = "E   " if show_label else ""
    ax.set_title(f"{prefix}All-pairs co-localisation at {RADIUS_UM} µm",
                 fontsize=6.5, fontweight="semibold", pad=3, loc="left")


def draw_cross_ripley_curve(ax, curve_df, show_label=True):
    """Panel F — Cross-Ripley curve: ductal epithelium <-> fibroblasts."""
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    if "imageid" in curve_df.columns:
        for i, (img, grp) in enumerate(curve_df.groupby("imageid")):
            grp = grp.sort_values("radius")
            ax.plot(grp["radius"] * PIXEL_SIZE_UM, grp["L_minus_r"],
                    lw=0.9, color=prop_cycle[i % len(prop_cycle)], label=str(img))
    else:
        curve_df = curve_df.sort_values("radius")
        ax.plot(curve_df["radius"] * PIXEL_SIZE_UM, curve_df["L_minus_r"],
                lw=0.9, color="#4a90d9", label="observed")

    ax.axhline(0, color="black", ls="--", lw=0.7)
    ax.axvline(RADIUS_UM, color="#aaaaaa", ls=":", lw=0.6)
    ymin, ymax = ax.get_ylim()
    ax.text(RADIUS_UM + 2, ymin + (ymax - ymin) * 0.04, f"{RADIUS_UM} µm",
            fontsize=4.0, color="#888888")
    ax.set_xlabel("Radius (µm)", fontsize=5.5, labelpad=2)
    ax.set_ylabel("L(r)−r", fontsize=5.5, labelpad=2)
    ax.tick_params(labelsize=4.5, length=2, pad=1)
    ax.legend(fontsize=4.0, loc="upper left", frameon=False)
    _despine(ax)

    src_lbl = PHENOTYPE_LABELS.get(SOURCE_PHENO, SOURCE_PHENO)
    tgt_lbl = PHENOTYPE_LABELS.get(TARGET_PHENO, TARGET_PHENO)
    prefix  = "F   " if show_label else ""
    ax.set_title(f"{prefix}{src_lbl} ↔ {tgt_lbl} co-localisation",
                 fontsize=5.0, fontweight="semibold", pad=3, loc="left")


def draw_permutation_envelope(ax, perm_df, show_label=True):
    """Panel G — Cross-Ripley permutation envelope: ductal <-> T cells."""
    obs_col = next((c for c in ["L_minus_r", "observed", "L_r"]
                    if c in perm_df.columns), None)
    lo_col  = next((c for c in ["lower", "lo", "q025", "env_lo", "env_low"]
                    if c in perm_df.columns), None)
    hi_col  = next((c for c in ["upper", "hi", "q975", "env_hi", "env_high"]
                    if c in perm_df.columns), None)

    if obs_col is None:
        ax.text(0.5, 0.5, "Permutation envelope data unavailable",
                transform=ax.transAxes, ha="center", va="center", fontsize=5)
        return

    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    def _plot_one(grp, color, label):
        grp = grp.sort_values("radius_um")
        ax.plot(grp["radius_um"], grp[obs_col], lw=0.9, color=color, label=label)
        if lo_col and hi_col:
            ax.fill_between(grp["radius_um"], grp[lo_col], grp[hi_col],
                            color=color, alpha=0.15)

    perm = perm_df.copy()
    perm["radius_um"] = perm["radius"] * PIXEL_SIZE_UM

    if "imageid" in perm.columns:
        for i, (img, grp) in enumerate(perm.groupby("imageid")):
            _plot_one(grp, prop_cycle[i % len(prop_cycle)], str(img))
    else:
        _plot_one(perm, "#4a90d9", "Observed")
        if lo_col and hi_col:
            ax.plot([], [], color="#4a90d9", alpha=0.3, lw=5,
                    label=f"Permutation env. (n={N_PERM})")

    ax.axhline(0, color="black", ls="--", lw=0.7)
    ax.set_xlabel("Radius (µm)", fontsize=5.5, labelpad=2)
    ax.set_ylabel("L(r)−r", fontsize=5.5, labelpad=2)
    ax.tick_params(labelsize=4.5, length=2, pad=1)
    ax.legend(fontsize=4.0, loc="upper left", frameon=False)
    _despine(ax)

    src_lbl = PHENOTYPE_LABELS.get(SOURCE_PHENO, SOURCE_PHENO)
    imm_lbl = PHENOTYPE_LABELS.get(TARGET_IMMUNE, TARGET_IMMUNE)
    prefix  = "G   " if show_label else ""
    ax.set_title(f"{prefix}{src_lbl} ↔ {imm_lbl}: permutation test",
                 fontsize=5.0, fontweight="semibold", pad=3, loc="left")


def draw_local_morans_fap(ax, adata, adata_fib, fap_quad_col, show_label=True):
    """Panel H — Local Moran's I FAP quadrant map (CAF activation hotspots)."""
    img_mask = adata.obs["imageid"] == IMAGE_ID
    all_data = adata[img_mask]
    fib_ids  = adata_fib.obs.index
    non_fib  = all_data[~all_data.obs.index.isin(fib_ids)]

    ax.scatter(non_fib.obs["X_centroid"], non_fib.obs["Y_centroid"],
               c="#e8e8e8", s=0.15, alpha=0.4, edgecolors="none", rasterized=True)

    fib_img = adata_fib[adata_fib.obs["imageid"] == IMAGE_ID]
    if fap_quad_col in fib_img.obs.columns:
        quad_vals = fib_img.obs[fap_quad_col].fillna("unclassified")
        for quad, fc in LISA_PALETTE.items():
            mask = quad_vals == quad
            if mask.any():
                sub = fib_img[mask]
                ax.scatter(sub.obs["X_centroid"], sub.obs["Y_centroid"],
                           c=fc, s=0.5, edgecolors="none", rasterized=True,
                           label=quad, zorder=3)
        handles = [mpatches.Patch(facecolor=LISA_PALETTE[q], label=q)
                   for q in ["high-high", "low-low", "high-low", "low-high"]]
        ax.legend(handles=handles, fontsize=3.8,
                  loc="center left", bbox_to_anchor=(1.02, 0.5),
                  bbox_transform=ax.transAxes,
                  frameon=False, title="FAP LISA", title_fontsize=4.0,
                  borderaxespad=0)
    else:
        ax.scatter(fib_img.obs["X_centroid"], fib_img.obs["Y_centroid"],
                   c="#ff7f0e", s=0.5, edgecolors="none", rasterized=True,
                   label="Fibroblasts")
        ax.legend(fontsize=4.0, frameon=False)

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("X centroid (px)", fontsize=5.5, labelpad=2)
    ax.set_ylabel("Y centroid (px)", fontsize=5.5, labelpad=2)
    ax.tick_params(labelsize=4.5, length=2, pad=1)
    _despine(ax)

    prefix = "H   " if show_label else ""
    ax.set_title(f"{prefix}FAP spatial autocorrelation — fibroblast hotspots",
                 fontsize=6.5, fontweight="semibold", pad=3, loc="left")


def draw_cross_morans_combined(ax_matrix, ax_map,
                                matrix_df, adata, adata_duct_local, cross_quad_col,
                                show_label=True):
    """Panel I — Cross-Moran's I matrix (left) + invasion-front spatial map (right)."""
    prefix = "I   " if show_label else ""

    # ── Left: cross-Moran's I heatmap ─────────────────────────────────────────
    if matrix_df is not None and not matrix_df.empty:
        heat_df = matrix_df.pivot(
            index="source_feature", columns="target_feature", values="cross_morans_i")
        heat_df.index   = [i.replace("_expr_z", "").replace("_z", "")
                            for i in heat_df.index]
        heat_df.columns = [c.replace("_expr_z", "").replace("_z", "")
                            for c in heat_df.columns]

        sns.heatmap(
            heat_df, ax=ax_matrix, cmap="RdBu_r", center=0,
            linewidths=0.4, linecolor="#eeeeee",
            annot=False,
            cbar_kws={"label": "Cross-Moran's I", "shrink": 0.75,
                      "pad": 0.02, "aspect": 20},
        )
        cbar = ax_matrix.collections[0].colorbar
        if cbar:
            cbar.ax.tick_params(labelsize=4.5, length=2, width=0.4)
            cbar.set_label("Cross-Moran's I", fontsize=4.5, labelpad=2)
            cbar.outline.set_linewidth(0.3)
        ax_matrix.set_xticklabels(
            ax_matrix.get_xticklabels(), rotation=40, ha="right", fontsize=4.5)
        ax_matrix.set_yticklabels(
            ax_matrix.get_yticklabels(), rotation=0, fontsize=4.5)
        ax_matrix.tick_params(length=0, pad=1)
        ax_matrix.set_xlabel("Ductal epithelium features", fontsize=5.0, labelpad=2)
        ax_matrix.set_ylabel("Fibroblast markers", fontsize=5.0, labelpad=2)
        ax_matrix.set_title(f"{prefix}Fibroblast–ductal feature coupling",
                             fontsize=6.5, fontweight="semibold", pad=3, loc="left")
    else:
        ax_matrix.text(0.5, 0.5, "Cross-Moran's I\n(feature data unavailable)",
                       transform=ax_matrix.transAxes,
                       ha="center", va="center", fontsize=5.5, color="#888888")
        ax_matrix.axis("off")

    # ── Right: invasion front spatial map ─────────────────────────────────────
    if adata_duct_local is not None and cross_quad_col is not None:
        img_mask = adata.obs["imageid"] == IMAGE_ID
        all_data = adata[img_mask]
        duct_ids = adata_duct_local.obs.index
        non_duct = all_data[~all_data.obs.index.isin(duct_ids)]

        ax_map.scatter(non_duct.obs["X_centroid"], non_duct.obs["Y_centroid"],
                       c="#e8e8e8", s=0.15, alpha=0.35, edgecolors="none", rasterized=True)

        duct_img  = adata_duct_local[adata_duct_local.obs["imageid"] == IMAGE_ID]
        quad_vals = duct_img.obs[cross_quad_col].fillna("unclassified")
        for quad, fc in CROSS_PALETTE.items():
            mask = quad_vals == quad
            if mask.any():
                sub = duct_img[mask]
                ax_map.scatter(sub.obs["X_centroid"], sub.obs["Y_centroid"],
                               c=fc, s=0.5, edgecolors="none", rasterized=True,
                               label=quad, zorder=3)

        ax_map.set_aspect("equal")
        ax_map.invert_yaxis()
        ax_map.set_xlabel("X centroid (px)", fontsize=5.5, labelpad=2)
        ax_map.set_ylabel("Y centroid (px)", fontsize=5.5, labelpad=2)
        ax_map.tick_params(labelsize=4.5, length=2, pad=1)
        handles = [
            mpatches.Patch(facecolor="#b2182b", label="HH: invasion front"),
            mpatches.Patch(facecolor="#2166ac", label="LL: quiescent/normal"),
            mpatches.Patch(facecolor="#fdae61", label="HL: disorganised, low-FAP stroma"),
            mpatches.Patch(facecolor="#74add1", label="LH: ordered, activated stroma"),
        ]
        ax_map.legend(handles=handles, fontsize=3.5,
                      loc="upper left", bbox_to_anchor=(1.03, 1),
                      bbox_transform=ax_map.transAxes,
                      frameon=False, title="Entropy \xd7 FAP-stroma",
                      title_fontsize=3.8, borderaxespad=0)
        _despine(ax_map)
        ax_map.set_title("Invasion front (ductal entropy \xd7 FAP co-localisation)",
                         fontsize=6.5, fontweight="semibold", pad=3, loc="left")
    else:
        ax_map.text(0.5, 0.5, "Invasion front map\n(feature data unavailable)",
                    transform=ax_map.transAxes,
                    ha="center", va="center", fontsize=5.5, color="#888888")
        ax_map.axis("off")


# ══════════════════════════════════════════════════════════════════════════════
# Standalone panel export
# ══════════════════════════════════════════════════════════════════════════════

_PANEL_SIZES = {
    "A": (87,  55),   # 3-column schematic
    "B": (64,  72),   # Ripley heatmap
    "C": (51,  58),   # Ripley line plot
    "D": (72,  70),   # B lineage hotspot map
    "E": (84,  66),   # cross-Ripley all-pairs
    "F": (51,  58),   # cross-Ripley curve
    "G": (51,  58),   # permutation envelope
    "H": (72,  70),   # local Moran's I FAP map
    "I": (75,  130),  # combined: cross-Moran's I matrix (top) + invasion front (bottom)
}


def _save_panel(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"suppfig9_{name}"
    pdf  = OUT_DIR / f"{stem}.pdf"
    png  = OUT_DIR / f"{stem}.png"
    fig.savefig(pdf, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {pdf.name}")


def make_standalone_panels():
    """Save each panel A–I as individual PDF + PNG."""
    _set_pub_rc()

    adata = load_data()
    results, adata = compute_all(adata)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\nSaving standalone panels:")

    def _fig(panel, axes_rect=None):
        w, h = _PANEL_SIZES[panel]
        fig  = plt.figure(figsize=(w * MM2IN, h * MM2IN), facecolor="white")
        if axes_rect is None:
            ax = fig.add_axes([0.16, 0.16, 0.68, 0.70])
        else:
            ax = fig.add_axes(axes_rect)
        return fig, ax

    # Panel A — schematic
    w, h = _PANEL_SIZES["A"]
    fig  = plt.figure(figsize=(w * MM2IN, h * MM2IN), facecolor="white")
    ax   = fig.add_axes([0.01, 0.06, 0.98, 0.88])
    draw_schematic(ax, show_label=False)
    _save_panel(fig, "A")

    # Panel B — Ripley heatmap (extra left margin for phenotype labels)
    fig, ax = _fig("B", axes_rect=[0.24, 0.16, 0.58, 0.68])
    draw_ripley_heatmap(ax, results["ripley_all"], show_label=False)
    _save_panel(fig, "B")

    # Panel C — Ripley line plot
    fig, ax = _fig("C")
    draw_ripley_lineplot(ax, results["ripley_all"], show_label=False)
    _save_panel(fig, "C")

    # Panel D — B lineage hotspot map
    fig, ax = _fig("D")
    draw_b_lineage_hotspot(ax, adata, results["hotspot_col"], show_label=False)
    _save_panel(fig, "D")

    # Panel E — cross-Ripley all-pairs (extra margins for axis labels)
    fig, ax = _fig("E", axes_rect=[0.22, 0.22, 0.58, 0.62])
    draw_cross_ripley_pairs(ax, results["cross_all"], show_label=False)
    _save_panel(fig, "E")

    # Panel F — cross-Ripley curve
    fig, ax = _fig("F")
    draw_cross_ripley_curve(ax, results["curve_df"], show_label=False)
    _save_panel(fig, "F")

    # Panel G — permutation envelope
    fig, ax = _fig("G")
    draw_permutation_envelope(ax, results["perm_df"], show_label=False)
    _save_panel(fig, "G")

    # Panel H — local Moran's I FAP map
    fig, ax = _fig("H")
    draw_local_morans_fap(ax, adata, results["adata_fib"],
                          results["fap_quad_col"], show_label=False)
    _save_panel(fig, "H")

    # Panel I — combined: cross-Moran's I matrix (top) + invasion front map (bottom)
    # heatmap row gets 1/3 of content height (~40mm ≈ 2/3 of original 68mm)
    # map row gets 2/3 of content height (~80mm, square-ish for equal-aspect scatter)
    w, h = _PANEL_SIZES["I"]
    fig  = plt.figure(figsize=(w * MM2IN, h * MM2IN), facecolor="white")
    gs   = gridspec.GridSpec(2, 1, figure=fig,
                              height_ratios=[1, 3],
                              hspace=0.45,
                              left=0.14, right=0.88, bottom=0.07, top=0.96)
    ax_matrix = fig.add_subplot(gs[0])
    ax_map    = fig.add_subplot(gs[1])
    draw_cross_morans_combined(
        ax_matrix, ax_map,
        results["matrix_df"], adata,
        results["adata_duct_local"], results["cross_quad_col"],
        show_label=False,
    )
    _save_panel(fig, "I")


if __name__ == "__main__":
    make_standalone_panels()
