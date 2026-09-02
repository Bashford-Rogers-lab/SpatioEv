"""
Supplementary Figure 6 — Segmentation QC and unsupervised clustering-based phenotyping
========================================================================================
Panel A : Schematic workflow (raw segmentation → area/NC filter → clustering → annotation)
Panel B : Cell area (µm²) distribution with debris / merged thresholds
Panel C : Nuclear-to-cell ratio distribution with abnormal NC threshold
Panel D : QC category summary — cell counts per flag category
Panel E : Spatial scatter of cells coloured by QC status (retained vs. flagged)
Panel F : Unsupervised Leiden cluster heatmap — mean z-scored expression per cluster
Panel G : Spatial scatter of cells coloured by Leiden cluster identity

Data  : data/exp_1/exp_1.h5ad  (single PDAC sample, 51,932 cells × 25 markers)
Scripts mirrored from:
  notebooks/00_dev_seg_qc_testing.ipynb          (Panels A–E)
  notebooks/01_dev_clustering_based_phenotyping_test.ipynb  (Panels F–G)

Run:
    python notebooks/suppfig6_qc_clustering.py

Output: paper/notebooks/results/suppfig6/suppfig6_qc_clustering.pdf (.png)

──────────────────────────────────────────────────────────────────────────────
USER CONFIGURATION
──────────────────────────────────────────────────────────────────────────────
After running this script once and reviewing Panel F (cluster heatmap) and
Panel G (spatial scatter), update LEVEL0_ANNOTATION_MAP below with meaningful
cell-type labels.  Assign the label "artifact" to any cluster you identify as
a tissue/staining artefact — it will be highlighted with a hatched border in
Panel G.
──────────────────────────────────────────────────────────────────────────────
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as mpe
import seaborn as sns

import spatioev as sv
from spatioev.config import QCConfig, ClusteringConfig


# ── File paths ─────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "exp_1" / "exp_1.h5ad"
OUT_DIR   = Path(__file__).parent / "results" / "suppfig6"
CACHE_PATH = OUT_DIR / "_cluster_cache.h5ad"   # cached clustering for reproducibility

MM2IN = 1 / 25.4


# ── QC parameters ──────────────────────────────────────────────────────────────
QC_CONFIG = QCConfig(
    pixel_size=0.325,
    min_area_um2=10,
    max_area_um2=1000,
    max_nc_ratio=1.0,
)

# ── Clustering parameters ──────────────────────────────────────────────────────
LEVEL0_MARKERS = ["DNA_1", "PCK", "COL4A1", "Vimentin", "CD45", "CD68", "SMA", "PDPN"]

LEVEL0_CONFIG = ClusteringConfig(
    markers=LEVEL0_MARKERS,
    resolution=0.2,
    n_neighbors=10,
    n_pcs=7,
    scale=True,
)

# Display names for markers in heatmap x-axis
MARKER_LABELS = {
    "DNA_1":    "DAPI",
    "PCK":      "Pan-CK",
    "COL4A1":   "COL4A1",
    "Vimentin": "Vimentin",
    "CD45":     "CD45",
    "CD68":     "CD68",
    "SMA":      "α-SMA",
    "PDPN":     "PDPN",
}

# ── Annotation map — EDIT AFTER REVIEWING PANELS F & G ────────────────────────
# Keys = Leiden cluster IDs (as strings: "0", "1", …).
# Use "artifact" for any tissue/staining artefact cluster.
# The script will still run with the default placeholder labels.
LEVEL0_ANNOTATION_MAP = {
    "0": "immune",
    "1": "tumour",
    "2": "tumour",
    "3": "fibroblast",
    "4": "artifact",
    "5": "tumour",
    "6": "tumour",
    "7": "artifact",
    "8": "artifact",
}

# ── Colour schemes ─────────────────────────────────────────────────────────────
QC_PALETTE = {
    "retained":          "#bbbbbb",
    "debris_fragment":   "#e07b39",
    "merged_cell":       "#9b59b6",
    "abnormal_nc_ratio": "#e74c3c",
}

QC_LABELS = {
    "retained":          "Retained",
    "debris_fragment":   "Debris /\nfragment",
    "merged_cell":       "Merged\ncell",
    "abnormal_nc_ratio": "Abnormal\nNC ratio",
}

CLUSTER_CMAP = "tab10"
ARTIFACT_HATCH = "////"   # hatching for artefact cluster in spatial scatter


# ── Publication RC ─────────────────────────────────────────────────────────────
def _set_pub_rc():
    plt.rcParams.update({
        "font.family":        "Arial",
        "font.size":          6,
        "axes.titlesize":     6.5,
        "axes.labelsize":     6,
        "xtick.labelsize":    5.5,
        "ytick.labelsize":    5.5,
        "axes.linewidth":     0.5,
        "xtick.major.width":  0.5,
        "ytick.major.width":  0.5,
        "xtick.major.size":   2.0,
        "ytick.major.size":   2.0,
        "lines.linewidth":    0.8,
        "legend.fontsize":    5.0,
        "legend.handlelength": 0.8,
        "pdf.fonttype":       42,
        "svg.fonttype":       "none",
    })


def _panel_label(ax, letter, dx=-0.12, dy=1.06):
    ax.text(dx, dy, letter,
            transform=ax.transAxes,
            fontsize=8, fontweight="bold",
            va="top", ha="left")


def _despine(ax):
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)


# ── Data loading & QC ──────────────────────────────────────────────────────────
def load_and_qc():
    import anndata as ad
    adata = sv.io.load_h5ad(str(DATA_PATH))
    adata_qc = sv.pp.run_segmentation_qc(adata.copy(), QC_CONFIG)

    # Combined QC status column (priority: area > NC ratio)
    status = pd.Series("retained", index=adata_qc.obs_names)
    status[adata_qc.obs["area_category"] == "debris_fragment"] = "debris_fragment"
    status[adata_qc.obs["area_category"] == "merged_cell"]     = "merged_cell"
    # Only flag NC ratio if area is OK
    nc_flag = (
        (adata_qc.obs["nc_ratio_category"] == "abnormal_nc_ratio") &
        (adata_qc.obs["area_category"] == "normal_area")
    )
    status[nc_flag] = "abnormal_nc_ratio"
    adata_qc.obs["qc_status"] = status.values
    return adata_qc


# ── Clustering (with caching) ──────────────────────────────────────────────────
def get_clustering(adata_qc):
    import anndata as ad

    # Work on QC-retained cells only
    adata_clean = sv.pp.filter_segmentation_errors(adata_qc)

    if CACHE_PATH.exists():
        print(f"Loading cached clustering from {CACHE_PATH}")
        adata_clustered = ad.read_h5ad(CACHE_PATH)
        # Re-attach cluster labels to clean adata
        adata_clean.obs["cluster_level0"] = (
            adata_clustered.obs["leiden"]
            .reindex(adata_clean.obs_names)
            .astype(str)
        )
        return adata_clean, adata_clustered

    print("Running unsupervised clustering (will be cached for future runs)...")
    import scanpy as sc
    sc.settings.seed = 42
    np.random.seed(42)

    adata_norm = sv.pp.zscore_normalize(adata_clean.copy())
    adata_clustered = sv.tl.cluster_cells(adata_norm, LEVEL0_CONFIG)

    # Save cluster labels back to clean adata
    adata_clean.obs["cluster_level0"] = (
        adata_clustered.obs["leiden"]
        .reindex(adata_clean.obs_names)
        .astype(str)
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    adata_clustered.write_h5ad(CACHE_PATH)
    print(f"Cached to {CACHE_PATH}")
    return adata_clean, adata_clustered


# ══════════════════════════════════════════════════════════════════════════════
# Panel drawing functions
# ══════════════════════════════════════════════════════════════════════════════

def draw_schematic(ax, show_label=True):
    """Panel A — workflow schematic with 4 steps (informative redesign)."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    steps = [
        {
            "title":   "Cell\nsegmentation",
            "face":    "#cce5ff",
            "edge":    "#4a90d9",
            "details": [
                ("Whole-cell &\nnuclear masks",  "#444444"),
                ("Mesmer + Pixie\nfeature extraction", "#555555"),
                ("51,932 cells input",           "#4a90d9"),
            ],
        },
        {
            "title":   "Morphological\nQC filter",
            "face":    "#fde8d0",
            "edge":    "#e07b39",
            "details": [
                ("area < 10 µm²\n→ debris/fragment", "#e07b39"),
                ("area > 1000 µm²\n→ merged cell",   "#9b59b6"),
                ("NC ratio > 1.0\n→ abnormal",        "#e74c3c"),
            ],
        },
        {
            "title":   "Unsupervised\nclustering",
            "face":    "#d4edda",
            "edge":    "#27ae60",
            "details": [
                ("8 lineage markers",              "#444444"),
                ("Leiden (k=10, res=0.2)",         "#444444"),
                ("Artifacts identified\nspatially", "#27ae60"),
            ],
        },
        {
            "title":   "Cell type\nannotation",
            "face":    "#ede7f6",
            "edge":    "#7b5ea7",
            "details": [
                ("Heatmap &\nspatial review",        "#444444"),
                ("Artifact clusters\nexcluded",       "#7b5ea7"),
                ("Final cell type\nlabels assigned",  "#444444"),
            ],
        },
    ]

    n        = len(steps)
    box_w    = 0.20
    box_h    = 0.66
    title_h  = 0.17          # coloured header band height
    gap      = (1.0 - n * box_w) / (n + 1)
    y_center = 0.52

    for i, step in enumerate(steps):
        x0 = gap + i * (box_w + gap)
        y0 = y_center - box_h / 2

        # ── outer box ──────────────────────────────────────────────────────
        rect = mpatches.FancyBboxPatch(
            (x0, y0), box_w, box_h,
            boxstyle="round,pad=0.012",
            facecolor=step["face"], edgecolor=step["edge"],
            linewidth=0.8, transform=ax.transData, clip_on=False,
            zorder=2,
        )
        ax.add_patch(rect)

        # ── coloured header band ────────────────────────────────────────────
        hdr = mpatches.FancyBboxPatch(
            (x0, y0 + box_h - title_h), box_w, title_h,
            boxstyle="round,pad=0.012",
            facecolor=step["edge"], edgecolor="none", alpha=0.30,
            transform=ax.transData, clip_on=False,
            zorder=3,
        )
        ax.add_patch(hdr)

        # ── step title ──────────────────────────────────────────────────────
        ax.text(
            x0 + box_w / 2, y0 + box_h - title_h / 2,
            step["title"],
            ha="center", va="center", fontsize=5.0,
            fontweight="bold", color="#222222",
            transform=ax.transData, zorder=4, linespacing=1.25,
        )

        # ── detail lines ────────────────────────────────────────────────────
        detail_area_h = box_h - title_h - 0.04
        n_det = len(step["details"])
        slot_h = detail_area_h / n_det
        for j, (text, color) in enumerate(step["details"]):
            y_det = y0 + box_h - title_h - 0.02 - (j + 0.5) * slot_h
            ax.text(
                x0 + box_w / 2, y_det, text,
                ha="center", va="center", fontsize=3.8,
                color=color, transform=ax.transData,
                zorder=4, linespacing=1.2,
            )

        # ── arrow to next ───────────────────────────────────────────────────
        if i < n - 1:
            x_start = x0 + box_w + 0.004
            x_end   = x_start + gap - 0.010
            ax.annotate(
                "",
                xy=(x_end, y_center), xytext=(x_start, y_center),
                xycoords="data", textcoords="data",
                arrowprops=dict(arrowstyle="-|>", color="#666666",
                                lw=0.9, mutation_scale=8),
                zorder=5,
            )

    prefix = "A   " if show_label else ""
    ax.set_title(f"{prefix}QC and phenotyping workflow", fontsize=6.5,
                 fontweight="semibold", pad=3, loc="left")


def draw_area_histogram(ax, adata_qc, show_label=True):
    """Panel B — cell area distribution."""
    areas = adata_qc.obs["area_um2"].dropna().values
    ax.hist(areas, bins=80, color="#4a90d9", edgecolor="none", alpha=0.85,
            linewidth=0, rasterized=True)
    ax.axvline(QC_CONFIG.min_area_um2, color="#e07b39", lw=1.0, ls="--",
               label=f"min {QC_CONFIG.min_area_um2} µm²")
    ax.axvline(QC_CONFIG.max_area_um2, color="#9b59b6", lw=1.0, ls="--",
               label=f"max {QC_CONFIG.max_area_um2} µm²")
    ax.set_xlabel("Cell area (µm²)", labelpad=2)
    ax.set_ylabel("Cell count", labelpad=2)
    prefix = "B   " if show_label else ""
    ax.set_title(f"{prefix}Cell area distribution", fontsize=6.5,
                 fontweight="semibold", pad=3, loc="left")
    ax.legend(fontsize=4.5, frameon=False, handlelength=1.2,
              loc="upper right", borderpad=0.2)
    ax.tick_params(length=2, pad=1)
    _despine(ax)


def draw_nc_histogram(ax, adata_qc, show_label=True):
    """Panel C — nucleus-to-cell ratio distribution, with flagged region highlighted."""
    nc        = adata_qc.obs["nc_ratio"].dropna().values
    threshold = QC_CONFIG.max_nc_ratio
    # Clip x-axis to 99.5th percentile to remove long sparse tail
    xlim_max  = float(np.percentile(nc, 99.5))
    nc_plot   = nc[nc <= xlim_max]

    # Draw histogram; colour each bar by QC status
    n_arr, bins, patches = ax.hist(
        nc_plot, bins=70,
        color="#4a90d9", edgecolor="none", alpha=0.80,
        linewidth=0, rasterized=True,
    )
    for patch, left in zip(patches, bins[:-1]):
        if left >= threshold:
            patch.set_facecolor("#e74c3c")
            patch.set_alpha(0.90)

    # Shaded region above threshold
    ax.axvspan(threshold, xlim_max, alpha=0.07, color="#e74c3c", linewidth=0, zorder=0)
    ax.axvline(threshold, color="#e74c3c", lw=1.0, ls="--", zorder=3)

    # Count annotation
    n_abn  = int((nc > threshold).sum())
    n_tot  = len(nc)
    ax.text(0.97, 0.96,
            f"NC > {threshold:.1f}:\n{n_abn:,} cells ({n_abn / n_tot * 100:.1f}%)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=4.5, color="#e74c3c",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="none", alpha=0.85))

    ax.set_xlim(0, xlim_max)
    ax.set_xlabel("Nuclear-to-cell ratio", labelpad=2)
    ax.set_ylabel("Cell count", labelpad=2)
    prefix = "C   " if show_label else ""
    ax.set_title(f"{prefix}NC ratio distribution", fontsize=6.5,
                 fontweight="semibold", pad=3, loc="left")
    ax.tick_params(length=2, pad=1)
    _despine(ax)


def draw_qc_summary_bar(ax, adata_qc, show_label=True):
    """Panel D — horizontal bar chart of cell counts per QC category."""
    cats_order = ["retained", "abnormal_nc_ratio", "merged_cell", "debris_fragment"]
    counts = {c: (adata_qc.obs["qc_status"] == c).sum() for c in cats_order}
    total  = adata_qc.n_obs

    labels  = [QC_LABELS[c] for c in cats_order]
    values  = [counts[c] for c in cats_order]
    colors  = [QC_PALETTE[c] for c in cats_order]
    y_pos   = np.arange(len(cats_order))

    bars = ax.barh(y_pos, values, color=colors, height=0.6,
                   linewidth=0.4, edgecolor="white")

    # Percentage annotations
    for bar, val in zip(bars, values):
        pct = val / total * 100
        ax.text(bar.get_width() + total * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,}  ({pct:.1f}%)",
                va="center", ha="left", fontsize=5.0, color="#444444")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=5.5)
    ax.set_xlabel("Cell count", labelpad=2)
    prefix = "D   " if show_label else ""
    ax.set_title(f"{prefix}QC category summary", fontsize=6.5,
                 fontweight="semibold", pad=3, loc="left")
    ax.set_xlim(0, total * 1.35)
    ax.tick_params(length=2, pad=1)
    _despine(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def draw_spatial_qc(ax, adata_qc, show_label=True):
    """Panel E — tissue scatter coloured by QC status."""
    rng = np.random.default_rng(42)
    cats_order = ["retained", "debris_fragment", "merged_cell", "abnormal_nc_ratio"]

    for cat in cats_order:
        mask = (adata_qc.obs["qc_status"] == cat).values
        if not mask.any():
            continue
        x = adata_qc.obs.loc[mask, "X_centroid"].values
        y = adata_qc.obs.loc[mask, "Y_centroid"].values

        # Downsample retained for speed; show all flagged
        if cat == "retained":
            idx = rng.choice(len(x), min(8000, len(x)), replace=False)
            x, y = x[idx], y[idx]

        ax.scatter(x, y,
                   c=QC_PALETTE[cat],
                   s=0.8,                                   # uniform dot size
                   alpha=(0.30 if cat == "retained" else 0.90),
                   linewidths=0, rasterized=True,
                   label=QC_LABELS[cat].replace("\n", " "),
                   zorder=(1 if cat == "retained" else 3))

    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    prefix = "E   " if show_label else ""
    ax.set_title(f"{prefix}Spatial QC status", fontsize=6.5,
                 fontweight="semibold", pad=3, loc="left")

    handles = [mpatches.Patch(facecolor=QC_PALETTE[c],
                              label=QC_LABELS[c].replace("\n", " "))
               for c in cats_order]
    ax.legend(handles=handles, fontsize=4.5, frameon=False,
              loc="upper right", borderpad=0.2,
              handlelength=0.8, handleheight=0.7,
              labelspacing=0.25)


def draw_cluster_heatmap(ax, adata_clustered, show_label=True):
    """Panel F — mean z-scored expression per Leiden cluster."""
    import scipy.sparse as sp_sparse

    X = adata_clustered.X
    if sp_sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=float)

    markers_present = list(adata_clustered.var_names)
    X_df = pd.DataFrame(X, index=adata_clustered.obs_names,
                        columns=markers_present)
    X_df["cluster"] = adata_clustered.obs["leiden"].values

    cluster_mean = X_df.groupby("cluster")[markers_present].mean()
    cluster_order = sorted(cluster_mean.index.tolist(),
                           key=lambda c: int(c) if c.isdigit() else ord(c[0]))
    cluster_mean = cluster_mean.loc[cluster_order]

    # Annotate cluster rows using the annotation map
    row_labels = [
        f"{cid}  {LEVEL0_ANNOTATION_MAP.get(cid, '')}"
        if LEVEL0_ANNOTATION_MAP.get(cid, "") not in ("", f"cluster {cid}")
        else f"Cluster {cid}"
        for cid in cluster_order
    ]
    col_labels = [MARKER_LABELS.get(m, m) for m in markers_present]

    # Clip extreme values for display
    vmax = 2.0
    sns.heatmap(
        cluster_mean,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-vmax, vmax=vmax,
        linewidths=0.25,
        linecolor="#eeeeee",
        xticklabels=col_labels,
        yticklabels=row_labels,
        square=True,                    # equal aspect per cell
        cbar_kws={"label": "Mean z-score", "shrink": 0.55,
                  "pad": 0.03, "aspect": 15},
        rasterized=True,
    )
    ax.set_xticklabels(ax.get_xticklabels(),
                       rotation=35, ha="right", fontsize=5.0, va="top")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=5.0, rotation=0)
    ax.tick_params(length=0, pad=2)
    prefix = "F   " if show_label else ""
    ax.set_title(f"{prefix}Cluster marker profile", fontsize=6.5,
                 fontweight="semibold", pad=3, loc="left")
    ax.set_ylabel("Leiden cluster", fontsize=5.5, labelpad=2)
    ax.set_xlabel("")

    # Colorbar font sizes
    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.ax.tick_params(labelsize=4.5, length=2, width=0.4)
        cbar.set_label("Mean z-score", fontsize=5.0, labelpad=3)
        cbar.outline.set_linewidth(0.3)


def draw_spatial_clusters(ax, adata_clean, adata_clustered, show_label=True):
    """Panel G — tissue scatter coloured by Leiden cluster."""
    rng = np.random.default_rng(42)

    cluster_ids = adata_clean.obs["cluster_level0"].values.astype(str)
    unique_ids  = sorted(set(cluster_ids), key=lambda c: int(c) if c.isdigit() else 999)
    cmap        = plt.get_cmap(CLUSTER_CMAP, len(unique_ids))
    cluster_colors = {c: cmap(i) for i, c in enumerate(unique_ids)}

    x_all = adata_clean.obs["X_centroid"].values
    y_all = adata_clean.obs["Y_centroid"].values

    # Identify artefact cluster(s)
    artifact_ids = {k for k, v in LEVEL0_ANNOTATION_MAP.items()
                    if v.lower() == "artifact"}

    for cid in unique_ids:
        mask = cluster_ids == cid
        xi, yi = x_all[mask], y_all[mask]
        n_samp = min(3000, mask.sum())
        idx    = rng.choice(mask.sum(), n_samp, replace=False)

        is_artifact = cid in artifact_ids
        label = LEVEL0_ANNOTATION_MAP.get(cid, f"Cluster {cid}")
        ax.scatter(xi[idx], yi[idx],
                   c=[cluster_colors[cid]],
                   s=(1.5 if is_artifact else 0.5),
                   alpha=(0.95 if is_artifact else 0.65),
                   linewidths=0, rasterized=True,
                   label=f"C{cid}: {label}",
                   marker=("D" if is_artifact else "o"),
                   zorder=(4 if is_artifact else 2))

    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    prefix = "G   " if show_label else ""
    ax.set_title(f"{prefix}Spatial Leiden clusters", fontsize=6.5,
                 fontweight="semibold", pad=3, loc="left")

    handles = [
        mpatches.Patch(facecolor=cluster_colors[cid],
                       label=f"C{cid}: {LEVEL0_ANNOTATION_MAP.get(cid, cid)}")
        for cid in unique_ids
    ]
    ax.legend(handles=handles, fontsize=3.8, frameon=False,
              loc="upper right", borderpad=0.2,
              handlelength=0.7, handleheight=0.6,
              labelspacing=0.18, ncol=2)


# ══════════════════════════════════════════════════════════════════════════════
# Main figure assembly
# ══════════════════════════════════════════════════════════════════════════════

def make_figure():
    _set_pub_rc()

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"Loading {DATA_PATH} ...")
    adata_qc = load_and_qc()
    print(f"  {adata_qc.n_obs:,} cells loaded; "
          f"{(adata_qc.obs['qc_status'] != 'retained').sum():,} flagged by QC")

    adata_clean, adata_clustered = get_clustering(adata_qc)
    n_clusters = adata_clustered.obs["leiden"].nunique()
    print(f"  {adata_clean.n_obs:,} retained cells → {n_clusters} Leiden clusters")

    # ── Figure layout: 2 rows ──────────────────────────────────────────────────
    # Row 1 (4 panels): A schematic, B area hist, C NC ratio hist, D QC bar
    # Row 2 (3 panels): E spatial QC, F cluster heatmap, G spatial clusters
    fig_w = 170 * MM2IN
    fig_h = 145 * MM2IN
    fig   = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    # Row 1 — top: 4 panels
    gs1 = mgridspec.GridSpec(
        1, 4,
        width_ratios=[1.35, 1, 1, 1],
        left=0.03, right=0.99, top=0.98, bottom=0.60,
        wspace=0.42,
    )
    ax_a = fig.add_subplot(gs1[0, 0])
    ax_b = fig.add_subplot(gs1[0, 1])
    ax_c = fig.add_subplot(gs1[0, 2])
    ax_d = fig.add_subplot(gs1[0, 3])

    # Row 2 — bottom: 3 panels
    gs2 = mgridspec.GridSpec(
        1, 3,
        width_ratios=[1, 1.1, 1],
        left=0.04, right=0.99, top=0.55, bottom=0.04,
        wspace=0.40,
    )
    ax_e = fig.add_subplot(gs2[0, 0])
    ax_f = fig.add_subplot(gs2[0, 1])
    ax_g = fig.add_subplot(gs2[0, 2])

    # ── Draw panels ────────────────────────────────────────────────────────────
    draw_schematic(ax_a)
    draw_area_histogram(ax_b, adata_qc)
    draw_nc_histogram(ax_c, adata_qc)
    draw_qc_summary_bar(ax_d, adata_qc)
    draw_spatial_qc(ax_e, adata_qc)
    draw_cluster_heatmap(ax_f, adata_clustered)
    draw_spatial_clusters(ax_g, adata_clean, adata_clustered)

    # ── Save ──────────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = OUT_DIR / "suppfig6_qc_clustering.pdf"
    out_png = OUT_DIR / "suppfig6_qc_clustering.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nSaved:\n  {out_pdf}\n  {out_png}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Standalone panel export
# ══════════════════════════════════════════════════════════════════════════════

# Per-panel figure sizes (width mm, height mm)
_PANEL_SIZES = {
    "A": (95,  45),   # 2/3 height of original
    "B": (58,  52),
    "C": (58,  52),
    "D": (58,  52),
    "E": (58,  78),
    "F": (68,  78),
    "G": (58,  78),
}


def _save_panel(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"suppfig6_{name}"
    pdf = OUT_DIR / f"{stem}.pdf"
    png = OUT_DIR / f"{stem}.png"
    fig.savefig(pdf, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {pdf.name}")


def make_standalone_panels():
    """Save each panel A–G as an individual PDF + PNG."""
    _set_pub_rc()

    print(f"\nLoading {DATA_PATH} ...")
    adata_qc = load_and_qc()
    adata_clean, adata_clustered = get_clustering(adata_qc)
    print(f"  {adata_qc.n_obs:,} cells; "
          f"{adata_clustered.obs['leiden'].nunique()} Leiden clusters")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\nSaving standalone panels:")

    def _fig(panel):
        w, h = _PANEL_SIZES[panel]
        return plt.figure(figsize=(w * MM2IN, h * MM2IN), facecolor="white")

    # Panel A — schematic (no letter prefix for standalone)
    fig = _fig("A")
    ax  = fig.add_axes([0.02, 0.06, 0.96, 0.88])
    draw_schematic(ax, show_label=False)
    _save_panel(fig, "A")

    # Panel B — area histogram
    fig = _fig("B")
    ax  = fig.add_axes([0.16, 0.16, 0.80, 0.76])
    draw_area_histogram(ax, adata_qc, show_label=False)
    _save_panel(fig, "B")

    # Panel C — NC ratio histogram
    fig = _fig("C")
    ax  = fig.add_axes([0.16, 0.16, 0.80, 0.76])
    draw_nc_histogram(ax, adata_qc, show_label=False)
    _save_panel(fig, "C")

    # Panel D — QC summary bar chart
    fig = _fig("D")
    ax  = fig.add_axes([0.22, 0.14, 0.70, 0.78])
    draw_qc_summary_bar(ax, adata_qc, show_label=False)
    _save_panel(fig, "D")

    # Panel E — spatial QC scatter
    fig = _fig("E")
    ax  = fig.add_axes([0.02, 0.02, 0.96, 0.94])
    draw_spatial_qc(ax, adata_qc, show_label=False)
    _save_panel(fig, "E")

    # Panel F — cluster heatmap
    fig = _fig("F")
    ax  = fig.add_axes([0.18, 0.12, 0.68, 0.82])
    draw_cluster_heatmap(ax, adata_clustered, show_label=False)
    _save_panel(fig, "F")

    # Panel G — spatial cluster scatter
    fig = _fig("G")
    ax  = fig.add_axes([0.02, 0.02, 0.96, 0.94])
    draw_spatial_clusters(ax, adata_clean, adata_clustered, show_label=False)
    _save_panel(fig, "G")

    print(f"\nAll panels saved to {OUT_DIR}/")


if __name__ == "__main__":
    make_figure()
    make_standalone_panels()
