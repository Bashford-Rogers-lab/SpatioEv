"""
Supplementary Figure 10 — Identification and characterization of tumor nests in
cancer metastasis in liver
==============================================================================
Panel lettering follows the published figure legend verbatim (this figure was
carried over unchanged from Figure 4 of the original manuscript):

A  : RCN cell-type composition stacked bar
B  : Spatial scatter coloured by RCN
C  : Spatial scatter coloured by connected ROIs
D  : [Placeholder] Representative liver metastasis image (DAPI + CK19)
E  : UMAP of liver metastasis cells coloured by phenotype
F  : Overlay of tumour-cell and other-cell clusters
G  : Tumour nests defined by DBSCAN, with zoomed insets
       (individual nest / defined boundary / shrunken + expanded boundaries)
H  : Tumour nest boundaries across the entire liver metastasis image
I  : UMAP embedding of boundary cell neighbourhoods
J  : Stacked bar of the cellular composition of boundary cell neighbourhoods
K  : Proportions of cells in different boundary clusters

Sources:
  - 05_dev_spatial_niche_boundaries.ipynb   (liver metastasis boundary panels)
  - 09_RA_OA_ECM_cell_spatioev_module_paper_applications.ipynb (RCN panels A–C)

Run from anywhere:
    python paper/figures/suppfig10_niche_boundary.py

Outputs
-------
Figures : paper/notebooks/results/suppfig10/suppfig10_*.pdf (.png)
Tables  : paper/notebooks/results/suppfig10/tables/*.csv

Panels beyond the published lettering (kept for reference, not part of the
figure) are saved with an ``extra_`` prefix.
"""

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR",  "/private/tmp/numba")

# Resolve the repository root by walking up to the directory containing
# pyproject.toml, so the script works from any working directory.
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = next(
    (p for p in (_HERE, *_HERE.parents) if (p / "pyproject.toml").is_file()),
    _HERE,
)
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
ROOT = PROJECT_ROOT

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.neighbors import BallTree
from sklearn.cluster import KMeans
import spatioev as sv

# ── Output directories ─────────────────────────────────────────────────────────
OUT_DIR    = ROOT / "paper" / "notebooks" / "results" / "suppfig10"
TABLES_DIR = OUT_DIR / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

MM2IN = 1 / 25.4

# ── Analysis constants ─────────────────────────────────────────────────────────
PIXEL_SIZE_UM        = 0.325
TUMOUR_LABEL         = "tumour"
TUMOUR_DBSCAN_EPS    = 45
TUMOUR_DBSCAN_MINPTS = 5
ROI_DBSCAN_EPS       = 100
ROI_DBSCAN_MINPTS    = 3
BOUNDARY_BUFFER_UM   = 30.0
BOUNDARY_BUFFER_PX   = BOUNDARY_BUFFER_UM / PIXEL_SIZE_UM
MASK_RESOLUTION      = 6.0
NEIGHBOURHOOD_RADIUS_UM = 30.0
NEIGHBOURHOOD_RADIUS_PX = NEIGHBOURHOOD_RADIUS_UM / PIXEL_SIZE_UM
N_BOUNDARY_CLUSTERS  = 5      # five tumour boundary types, per the manuscript
RANDOM_STATE         = 42


def _save_table(df, name):
    """Write a statistics table to TABLES_DIR."""
    if df is None or (hasattr(df, "empty") and df.empty):
        print(f"  [skip] {name} — no data")
        return
    path = TABLES_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"  table: {path.name}  ({len(df)} rows)")

# ── Publication RC ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "Arial",
    "font.size":         6,
    "axes.titlesize":    6.5,
    "axes.labelsize":    6,
    "xtick.labelsize":   5.5,
    "ytick.labelsize":   5.5,
    "axes.linewidth":    0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size":  2.0,
    "ytick.major.size":  2.0,
    "lines.linewidth":   0.8,
    "legend.fontsize":   5.0,
    "pdf.fonttype":      42,
    "svg.fonttype":      "none",
})


def _save(fig, name, title=None):
    """Apply title label and save as PDF + PNG."""
    if title:
        fig.axes[0].set_title(title, fontsize=6.5, fontweight="semibold",
                              pad=3, loc="left")
    stem = OUT_DIR / f"suppfig10_{name}"
    fig.savefig(f"{stem}.pdf", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: suppfig10_{name}.pdf")


def _placeholder(name, msg):
    fig, ax = plt.subplots(figsize=(65 * MM2IN, 55 * MM2IN), facecolor="white")
    ax.text(0.5, 0.5, msg, transform=ax.transAxes,
            ha="center", va="center", fontsize=6, color="#aaaaaa",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5",
                      edgecolor="#cccccc", linewidth=0.5))
    ax.axis("off")
    _save(fig, name)


# ══════════════════════════════════════════════════════════════════════════════
# RA/OA section — Panels A–D
# ══════════════════════════════════════════════════════════════════════════════

print("Loading RA/OA data ...")
adata_ra = sv.load_h5ad("data/RA_OA/tmp6.h5ad")

IMAGE_KEY    = "imageid"
PHENOTYPE_KEY = "phenotype"
GROUP_KEY    = "pathology"
X_KEY        = "X_centroid"
Y_KEY        = "Y_centroid"

# Detect pre-computed neighbourhood / RCN column
RCN_CANDIDATES = ["neighborhood", "RCN", "rcn", "spatial_lda_cluster",
                   "kmeans", "topic", "scimap_neighborhood"]
rcn_col = next((c for c in RCN_CANDIDATES if c in adata_ra.obs.columns), None)

if rcn_col:
    print(f"  Using RCN column: '{rcn_col}'")
else:
    print("  No pre-computed RCN column found — Panels A–D will be placeholders.")
    print(f"  Available obs columns: {list(adata_ra.obs.columns)}")

# ── Panel A — RCN cell-type composition ───────────────────────────────────────
if rcn_col is not None:
    comp = (adata_ra.obs
            .groupby([rcn_col, PHENOTYPE_KEY]).size()
            .reset_index(name="count"))
    comp["proportion"] = comp["count"] / comp.groupby(rcn_col)["count"].transform("sum")
    pivot = comp.pivot(index=rcn_col, columns=PHENOTYPE_KEY,
                       values="proportion").fillna(0)
    phenos = pivot.columns.tolist()
    cmap20 = plt.cm.get_cmap("tab20", len(phenos))
    colors = [cmap20(i) for i in range(len(phenos))]

    fig, ax = plt.subplots(figsize=(80 * MM2IN, 65 * MM2IN), facecolor="white")
    bottom = np.zeros(len(pivot))
    for col, c in zip(phenos, colors):
        ax.bar(range(len(pivot)), pivot[col].values,
               bottom=bottom, color=c, width=0.8, label=col)
        bottom += pivot[col].values
    ax.set_xlim(-0.5, len(pivot) - 0.5)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(len(pivot)))
    ax.set_xticklabels([str(r) for r in pivot.index],
                       rotation=45, ha="right", fontsize=4.5)
    ax.set_ylabel("Proportion", fontsize=5.5)
    ax.legend(fontsize=3.5, loc="upper left", bbox_to_anchor=(1.02, 1),
              bbox_transform=ax.transAxes, frameon=False,
              title="Cell type", title_fontsize=4.0, borderaxespad=0)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    _save(fig, "A", title="A   RCN cell-type composition")
    _save_table(comp, "suppfig10_A_rcn_celltype_composition")
    _save_table(
        adata_ra.obs.groupby([GROUP_KEY, rcn_col], observed=True)
        .size().reset_index(name="n_cells"),
        "suppfig10_A_rcn_counts_by_pathology")
else:
    _placeholder("A", "Panel A — RCN composition\n(requires pre-computed RCN column in obs)")

# ── Panel B — Spatial scatter coloured by RCN ─────────────────────────────────
if rcn_col is not None:
    ra_samples = (adata_ra.obs[adata_ra.obs[GROUP_KEY].str.upper() == "RA"][IMAGE_KEY]
                  .unique().tolist())
    sample_id = ra_samples[0] if ra_samples else adata_ra.obs[IMAGE_KEY].unique()[0]
    print(f"  Panel B sample: {sample_id}")

    sub = adata_ra[adata_ra.obs[IMAGE_KEY] == sample_id]
    vals = sub.obs[rcn_col].astype(str)
    uniq = sorted(vals.unique())
    cmap = plt.cm.get_cmap("tab20", len(uniq))
    col_map = {v: cmap(i) for i, v in enumerate(uniq)}

    fig, ax = plt.subplots(figsize=(72 * MM2IN, 70 * MM2IN), facecolor="white")
    for v in uniq:
        m = vals == v
        ax.scatter(sub.obs.loc[m, X_KEY], sub.obs.loc[m, Y_KEY],
                   c=[col_map[v]], s=0.5, edgecolors="none",
                   rasterized=True, label=str(v))
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("X centroid (px)", fontsize=5.5)
    ax.set_ylabel("Y centroid (px)", fontsize=5.5)
    ax.legend(fontsize=3.5, loc="upper left", bbox_to_anchor=(1.02, 1),
              bbox_transform=ax.transAxes, frameon=False,
              title="RCN", title_fontsize=4.0, borderaxespad=0)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    _save(fig, "B", title="B   RCN spatial distribution")
    _save_table(
        vals.value_counts().rename_axis("RCN")
        .reset_index(name="n_cells").assign(sample=str(sample_id)),
        "suppfig10_B_rcn_cell_counts")
else:
    _placeholder("B", "Panel B — RCN spatial map\n(requires pre-computed RCN column in obs)")

# ── Panel C — Connected ROI scatter ───────────────────────────────────────────
if rcn_col is not None:
    sub_img = adata_ra[adata_ra.obs[IMAGE_KEY] == sample_id].copy()
    # Assign connected components per-RCN using cluster_spatial_niches
    sub_img.obs["connected_roi"] = "unassigned"
    for rcn_val in sub_img.obs[rcn_col].dropna().unique():
        safe = str(rcn_val).replace(" ", "_").replace("/", "_")
        comp_col = f"{safe}_component"
        try:
            sub_img = sv.tl.cluster_spatial_niches(
                sub_img, label_key=rcn_col, label_value=rcn_val,
                eps=ROI_DBSCAN_EPS, min_samples=ROI_DBSCAN_MINPTS)
            if comp_col in sub_img.obs.columns:
                mask = sub_img.obs[comp_col].astype(str) != "-1"
                sub_img.obs.loc[mask[mask].index, "connected_roi"] = (
                    str(rcn_val) + "_" + sub_img.obs.loc[mask[mask].index, comp_col].astype(str))
        except Exception:
            pass

    vals = sub_img.obs["connected_roi"].astype(str)
    uniq = sorted(vals.unique())
    cmap2 = plt.cm.get_cmap("tab20b", min(len(uniq), 40))
    col_map2 = {v: ("#dddddd" if v == "unassigned" else cmap2(i % 40))
                for i, v in enumerate(uniq)}

    fig, ax = plt.subplots(figsize=(72 * MM2IN, 70 * MM2IN), facecolor="white")
    for v in ["unassigned"] + [u for u in uniq if u != "unassigned"]:
        m = vals == v
        if m.any():
            ax.scatter(sub_img.obs.loc[m, X_KEY], sub_img.obs.loc[m, Y_KEY],
                       c=[col_map2[v]], s=0.4 if v == "unassigned" else 0.6,
                       alpha=0.4 if v == "unassigned" else 0.85,
                       edgecolors="none", rasterized=True)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("X centroid (px)", fontsize=5.5)
    ax.set_ylabel("Y centroid (px)", fontsize=5.5)
    n_rois = sum(1 for v in uniq if v != "unassigned")
    ax.text(0.02, 0.97, f"n = {n_rois} ROIs", transform=ax.transAxes,
            fontsize=4.5, va="top", color="#555555")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    _save(fig, "C", title="C   Connected ROI spatial map")
    _save_table(
        vals.value_counts().rename_axis("connected_roi")
        .reset_index(name="n_cells").assign(sample=str(sample_id)),
        "suppfig10_C_connected_roi_sizes")
    print(f"    {n_rois} connected ROIs in {sample_id}")
else:
    _placeholder("C", "Panel C — Connected ROIs\n(requires pre-computed RCN column in obs)")

# ── Panel D — RCN distribution RA vs OA ───────────────────────────────────────
if rcn_col is not None:
    counts = (adata_ra.obs
              .groupby([IMAGE_KEY, GROUP_KEY, rcn_col]).size()
              .reset_index(name="count"))
    counts["proportion"] = (counts["count"] /
                            counts.groupby(IMAGE_KEY)["count"].transform("sum").clip(lower=1))

    group_order = ["RA", "OA"] if all(
        g in counts[GROUP_KEY].values for g in ["RA", "OA"]
    ) else sorted(counts[GROUP_KEY].unique())
    palette = {"RA": "#d62728", "OA": "#1f77b4"}

    fig, ax = plt.subplots(figsize=(80 * MM2IN, 65 * MM2IN), facecolor="white")
    sns.boxplot(data=counts, x=rcn_col, y="proportion",
                hue=GROUP_KEY, hue_order=group_order,
                palette=palette, linewidth=0.5, fliersize=1.5, ax=ax)
    ax.set_xlabel("RCN", fontsize=5.5)
    ax.set_ylabel("Proportion of cells", fontsize=5.5)
    ax.tick_params(labelsize=4.5, length=2)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=4.5)
    handles = [mpatches.Patch(facecolor=palette.get(g, "#888888"), label=g)
               for g in group_order]
    ax.legend(handles=handles, fontsize=4.0, frameon=False,
              title="Pathology", title_fontsize=4.5)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    # Not part of the published figure lettering — kept for reference only.
    _save(fig, "extra_rcn_abundance", title="RCN abundance across RA and OA")
else:
    print("  [skip] extra RCN abundance panel — no RCN column")


# ══════════════════════════════════════════════════════════════════════════════
# Liver metastasis section — Panels E–K
# Extracted directly from 05_dev_spatial_niche_boundaries.ipynb
# ══════════════════════════════════════════════════════════════════════════════

print("\nLoading liver metastasis data (exp_1) ...")
# NOTE: this previously read "data/exp_1.h5ad", which does not exist.
adata = sv.load_h5ad(str(ROOT / "data" / "exp_1" / "exp_1.h5ad"))

# --- from notebook cell 3 ---
ann = pd.read_csv(ROOT / "results" / "svm_phenotyping_results.csv", index_col=0)
ann = ann[["annotated_clusters_update3", "svm_prediction"]]
adata.obs = adata.obs.join(ann, how="left")
print(f"  {adata.n_obs:,} cells annotated")

# ── Panel E — Phenotype annotation heatmap ────────────────────────────────────
try:
    import scipy.sparse as sp_sparse
    X = adata.X
    if sp_sparse.issparse(X):
        X = X.toarray()
    var_names = list(adata.var_names)
    pheno_col = "annotated_clusters_update3"
    obs = adata.obs[pheno_col].dropna()
    common = obs.index.intersection(adata.obs_names)
    pheno_arr = obs.loc[common].values
    X_sub = X[adata.obs_names.isin(common), :]
    phenos = sorted(set(pheno_arr))
    mean_mat = np.array([X_sub[pheno_arr == p, :].mean(axis=0) for p in phenos])
    mean_df = pd.DataFrame(mean_mat, index=phenos, columns=var_names)
    mean_df = ((mean_df - mean_df.mean()) / mean_df.std().clip(lower=1e-6)).clip(-3, 3)

    fig, ax = plt.subplots(figsize=(90 * MM2IN, 72 * MM2IN), facecolor="white")
    sns.heatmap(mean_df, ax=ax, cmap="RdBu_r", center=0,
                linewidths=0.2, annot=False,
                cbar_kws={"label": "z-score", "shrink": 0.7, "aspect": 20})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=4.5)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=4.5)
    ax.tick_params(length=0, pad=1)
    ax.set_xlabel("Marker", fontsize=5.5)
    ax.set_ylabel("")
    fig.tight_layout()
    # Not part of the published figure lettering — kept for reference only.
    _save(fig, "extra_annotation_heatmap", title="Liver metastasis cell annotation")
    _save_table(mean_df.reset_index().rename(columns={"index": "phenotype"}),
                "suppfig10_extra_annotation_marker_zscores")
except Exception as exc:
    print(f"  Annotation heatmap error: {exc}")

# ── Panel D — Representative tissue image (placeholder) ───────────────────────
_placeholder("D", "D   Representative liver metastasis image\nDAPI + CK19 fluorescence\n(source image file required)")

# ── Panel E — UMAP coloured by phenotype ─────────────────────────────────────
if "X_umap" not in adata.obsm:
    print("  Computing UMAP for Panel E ...")
    try:
        import umap as umap_lib
        X_raw = adata.X
        import scipy.sparse as sp_sparse
        if sp_sparse.issparse(X_raw):
            X_raw = X_raw.toarray()
        adata.obsm["X_umap"] = umap_lib.UMAP(
            n_components=2, random_state=42, min_dist=0.3, n_neighbors=15
        ).fit_transform(X_raw)
    except ImportError:
        print("  umap-learn not found — Panel G will be a placeholder")
        adata.obsm["X_umap"] = None

if adata.obsm.get("X_umap") is not None:
    emb = adata.obsm["X_umap"]
    phenos = adata.obs["annotated_clusters_update3"].fillna("Unknown").values
    cmap20 = plt.cm.get_cmap("tab20", len(set(phenos)))
    col_map = {p: cmap20(i) for i, p in enumerate(sorted(set(phenos)))}

    fig, ax = plt.subplots(figsize=(65 * MM2IN, 65 * MM2IN), facecolor="white")
    for p in sorted(set(phenos)):
        m = phenos == p
        ax.scatter(emb[m, 0], emb[m, 1], c=[col_map[p]], s=0.4,
                   edgecolors="none", rasterized=True, label=p)
    ax.set_xlabel("UMAP 1", fontsize=5.5)
    ax.set_ylabel("UMAP 2", fontsize=5.5)
    ax.legend(fontsize=3.5, loc="upper left", bbox_to_anchor=(1.02, 1),
              bbox_transform=ax.transAxes, frameon=False,
              title="Phenotype", title_fontsize=4.0, borderaxespad=0)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    _save(fig, "E", title="E   Phenotype UMAP")
    _save_table(
        adata.obs["annotated_clusters_update3"].fillna("Unknown")
        .value_counts().rename_axis("phenotype").reset_index(name="n_cells"),
        "suppfig10_E_phenotype_counts")
else:
    _placeholder("E", "Panel E — UMAP\n(umap-learn not available)")

# ── Panel F — Tumour / other overlay ─────────────────────────────────────────
# equivalent to the initial spatial view before boundary computation
print("  Panel F — tumour/other spatial overlay ...")
phenos_h = adata.obs["annotated_clusters_update3"].fillna("other").values
is_tumour = phenos_h == TUMOUR_LABEL

fig, ax = plt.subplots(figsize=(72 * MM2IN, 70 * MM2IN), facecolor="white")
ax.scatter(adata.obs.loc[~is_tumour, "X_centroid"],
           adata.obs.loc[~is_tumour, "Y_centroid"],
           c="#d0d0d0", s=0.2, alpha=0.4, edgecolors="none",
           rasterized=True, label="Other cells")
ax.scatter(adata.obs.loc[is_tumour, "X_centroid"],
           adata.obs.loc[is_tumour, "Y_centroid"],
           c="#d62728", s=0.5, edgecolors="none",
           rasterized=True, label="Tumour")
ax.set_aspect("equal")
ax.invert_yaxis()
ax.set_xlabel("X centroid (px)", fontsize=5.5)
ax.set_ylabel("Y centroid (px)", fontsize=5.5)
handles = [mpatches.Patch(facecolor="#d62728", label="Tumour"),
           mpatches.Patch(facecolor="#d0d0d0", label="Other cells")]
ax.legend(handles=handles, fontsize=4.0, loc="lower right", frameon=False)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
fig.tight_layout()
_save(fig, "F", title="F   Tumour and other cells")
_save_table(pd.DataFrame([
    {"group": "tumour", "n_cells": int(is_tumour.sum())},
    {"group": "other",  "n_cells": int((~is_tumour).sum())},
]), "suppfig10_F_tumour_other_counts")

# ── Pipeline: cluster niches → boundaries → assign regions ────────────────────
# (directly from notebook cells 4–9)
print("  Running tumour niche clustering (DBSCAN) ...")
adata = sv.tl.cluster_spatial_niches(
    adata,
    label_key="annotated_clusters_update3",
    label_value=TUMOUR_LABEL,
    eps=TUMOUR_DBSCAN_EPS,
    min_samples=TUMOUR_DBSCAN_MINPTS,
)

# ── Panel G — DBSCAN tumour nests ─────────────────────────────────────────────
# (from notebook cell 6: sv.pl.plot_spatial_category with tumour_component)
print("  Panel G — DBSCAN tumour nests ...")
fig, ax = plt.subplots(figsize=(72 * MM2IN, 70 * MM2IN), facecolor="white")
sv.pl.plot_spatial_category(
    adata,
    feature="tumour_component",
    image_id=None,
    image_key="imageid",
    x_key="X_centroid",
    y_key="Y_centroid",
    point_size=0.5,
    alpha=0.85,
    palette="tab20",
    show_legend=False,
    figsize=(72 * MM2IN, 70 * MM2IN),
    ax=ax,
)
ax.set_xlabel("X centroid (px)", fontsize=5.5)
ax.set_ylabel("Y centroid (px)", fontsize=5.5)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
fig.tight_layout()
_save(fig, "G", title="G   Tumour nests (DBSCAN)")

_nest_counts = (adata.obs["tumour_component"].dropna()
                .value_counts().rename_axis("tumour_component")
                .reset_index(name="n_cells"))
_save_table(_nest_counts, "suppfig10_G_tumour_nest_sizes")
print(f"    {len(_nest_counts)} tumour nests detected")

print("  Building niche boundaries ...")
boundary_df = sv.tl.build_niche_boundaries(
    adata,
    component_key="tumour_component",
    min_cluster_size=20,
    method="density_mask",
    mask_resolution=MASK_RESOLUTION,
    mask_sigma=2.0,
    mask_threshold=0.08,
    mask_closing_size=9,
)

buffered_boundary_df = sv.tl.buffer_niche_boundaries(
    boundary_df,
    component_key="tumour_component",
    expand_by=BOUNDARY_BUFFER_PX,
    shrink_by=BOUNDARY_BUFFER_PX,
)

assignments_df = sv.tl.assign_cells_to_niche_regions(
    adata,
    buffered_boundary_df,
    component_key="tumour_component",
    image_key="imageid",
    x_key="X_centroid",
    y_key="Y_centroid",
    region_key="tumour_region",
    mode="distance_to_edge",
    boundary_width=BOUNDARY_BUFFER_PX,
)

# ── Panel H — Niche boundaries ────────────────────────────────────────────────
# (from notebook cell 8: sv.pl.plot_niche_boundaries)
print("  Panel H — niche boundaries ...")
fig, ax = plt.subplots(figsize=(72 * MM2IN, 70 * MM2IN), facecolor="white")
sv.pl.plot_niche_boundaries(
    adata,
    buffered_boundary_df,
    image_id="exp",
    image_key="imageid",
    x_key="X_centroid",
    y_key="Y_centroid",
    point_size=0.4,
    point_color="lightgray",
    point_alpha=0.5,
    boundary_color="black",
    boundary_linewidth=0.6,
    expanded_color="red",
    expanded_linewidth=0.5,
    shrunk_color="blue",
    shrunk_linewidth=0.5,
    figsize=(72 * MM2IN, 70 * MM2IN),
    ax=ax,
)
ax.set_xlabel("X centroid (px)", fontsize=5.5)
ax.set_ylabel("Y centroid (px)", fontsize=5.5)
handles_j = [
    plt.Line2D([0], [0], color="black",  lw=0.8, label="Boundary"),
    plt.Line2D([0], [0], color="red",    lw=0.6, ls="--", label="Expanded"),
    plt.Line2D([0], [0], color="blue",   lw=0.6, ls=":",  label="Shrunk"),
]
ax.legend(handles=handles_j, fontsize=4.0, loc="lower right", frameon=False)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
fig.tight_layout()
_save(fig, "H", title="H   Tumour-nest boundaries")

_save_table(boundary_df, "suppfig10_H_niche_boundaries")

# ── Region assignment (input to panels I–K) ───────────────────────────────────
print("  Assigning cells to tumour regions ...")
adata = sv.add_niche_regions_to_obs(
    adata,
    assignments_df,
    region_key="tumour_region",
    component_key="tumour_component",
)

_region_counts = (adata.obs["tumour_region"].fillna("background")
                  .value_counts().rename_axis("tumour_region")
                  .reset_index(name="n_cells"))
_region_counts["fraction"] = _region_counts["n_cells"] / _region_counts["n_cells"].sum()
_save_table(_region_counts, "suppfig10_region_assignment_counts")

# Reference panel (region assignments) — not part of the published lettering.
fig, ax = plt.subplots(figsize=(72 * MM2IN, 70 * MM2IN), facecolor="white")
sv.pl.plot_spatial_category(
    adata,
    feature="tumour_region",
    image_id="exp",
    image_key="imageid",
    x_key="X_centroid",
    y_key="Y_centroid",
    point_size=0.5,
    alpha=0.9,
    palette="tab20",
    show_legend=True,
    figsize=(72 * MM2IN, 70 * MM2IN),
    ax=ax,
)
ax.set_xlabel("X centroid (px)", fontsize=5.5)
ax.set_ylabel("Y centroid (px)", fontsize=5.5)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
fig.tight_layout()
_save(fig, "extra_region_assignments", title="Tumour region assignments")


# ══════════════════════════════════════════════════════════════════════════════
# Boundary cell neighbourhood profiling — Panels I, J, K
# ══════════════════════════════════════════════════════════════════════════════
# Cells lying on the tumour-nest boundary are profiled by the phenotype
# composition of their local neighbourhood (radius NEIGHBOURHOOD_RADIUS_UM),
# then embedded with UMAP and partitioned by k-means into N_BOUNDARY_CLUSTERS
# distinct tumour boundary types.

print("\n  Profiling boundary cell neighbourhoods ...")

region_series = adata.obs["tumour_region"].fillna("background").astype(str)
boundary_mask = region_series.str.contains("border|boundary|edge", case=False,
                                           regex=True, na=False).to_numpy()

if boundary_mask.sum() < N_BOUNDARY_CLUSTERS * 10:
    print(f"    Only {int(boundary_mask.sum())} boundary cells found "
          f"(regions present: {sorted(region_series.unique())}).")
    print("    Falling back to all cells assigned to a tumour component.")
    boundary_mask = adata.obs["tumour_component"].notna().to_numpy()

n_boundary = int(boundary_mask.sum())
print(f"    {n_boundary:,} boundary cells")

if n_boundary >= N_BOUNDARY_CLUSTERS * 10:
    coords_all = adata.obs[["X_centroid", "Y_centroid"]].to_numpy(dtype=float)
    phen_all   = adata.obs["annotated_clusters_update3"].fillna("Unknown").astype(str).to_numpy()
    categories = sorted(pd.unique(phen_all))
    cat_index  = {c: i for i, c in enumerate(categories)}

    finite = np.isfinite(coords_all).all(axis=1)
    tree   = BallTree(coords_all[finite])
    phen_finite = phen_all[finite]

    b_idx    = np.where(boundary_mask & finite)[0]
    b_coords = coords_all[b_idx]

    print(f"    Counting neighbours within {NEIGHBOURHOOD_RADIUS_UM:.0f} µm "
          f"({NEIGHBOURHOOD_RADIUS_PX:.0f} px) ...")
    neighbours = tree.query_radius(b_coords, r=NEIGHBOURHOOD_RADIUS_PX)

    profile = np.zeros((len(b_idx), len(categories)), dtype=float)
    for i, nb in enumerate(neighbours):
        if len(nb) == 0:
            continue
        vals, counts = np.unique(phen_finite[nb], return_counts=True)
        for v, c in zip(vals, counts):
            profile[i, cat_index[v]] = c

    totals = profile.sum(axis=1, keepdims=True)
    profile_frac = np.divide(profile, totals, out=np.zeros_like(profile),
                             where=totals > 0)

    profile_df = pd.DataFrame(profile_frac, columns=categories)
    profile_df.insert(0, "cell_id", adata.obs_names[b_idx])
    profile_df["n_neighbours"] = totals.ravel().astype(int)

    # ── k-means into boundary types ───────────────────────────────────────────
    print(f"    k-means into {N_BOUNDARY_CLUSTERS} boundary types ...")
    km = KMeans(n_clusters=N_BOUNDARY_CLUSTERS, random_state=RANDOM_STATE,
                n_init=10)
    boundary_cluster = km.fit_predict(profile_frac)
    profile_df["boundary_cluster"] = boundary_cluster
    _save_table(profile_df, "suppfig10_IJK_boundary_neighbourhood_profiles")

    # ── UMAP of the neighbourhood profiles ────────────────────────────────────
    emb_b = None
    try:
        import umap as umap_lib
        print("    UMAP of boundary neighbourhood profiles ...")
        emb_b = umap_lib.UMAP(n_components=2, random_state=RANDOM_STATE,
                              min_dist=0.3, n_neighbors=15).fit_transform(profile_frac)
    except ImportError:
        print("    umap-learn not found — Panel I will be a placeholder")

    # ── Panel I — UMAP of boundary cell neighbourhoods ────────────────────────
    if emb_b is not None:
        cmap_b  = plt.get_cmap("tab10", N_BOUNDARY_CLUSTERS)
        fig, ax = plt.subplots(figsize=(65 * MM2IN, 65 * MM2IN), facecolor="white")
        for c in range(N_BOUNDARY_CLUSTERS):
            m = boundary_cluster == c
            ax.scatter(emb_b[m, 0], emb_b[m, 1], c=[cmap_b(c)], s=0.6,
                       edgecolors="none", rasterized=True,
                       label=f"Cluster {c} (n={int(m.sum()):,})")
        ax.set_xlabel("UMAP 1", fontsize=5.5)
        ax.set_ylabel("UMAP 2", fontsize=5.5)
        ax.legend(fontsize=3.5, loc="upper left", bbox_to_anchor=(1.02, 1),
                  bbox_transform=ax.transAxes, frameon=False,
                  title="Boundary cluster", title_fontsize=4.0, borderaxespad=0)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        _save(fig, "I", title="I   Boundary cell neighbourhood UMAP")

        umap_df = pd.DataFrame({
            "cell_id": adata.obs_names[b_idx],
            "umap_1":  emb_b[:, 0],
            "umap_2":  emb_b[:, 1],
            "boundary_cluster": boundary_cluster,
        })
        _save_table(umap_df, "suppfig10_I_boundary_umap_coordinates")
    else:
        _placeholder("I", "Panel I — Boundary neighbourhood UMAP\n(umap-learn not available)")

    # ── Panel J — Composition of each boundary cluster ────────────────────────
    print("  Panel J — boundary cluster composition ...")
    comp_b = (profile_df.groupby("boundary_cluster")[categories]
              .mean()
              .loc[:, lambda d: d.sum(axis=0) > 0])
    comp_b = comp_b.div(comp_b.sum(axis=1), axis=0)

    cmap_p  = plt.get_cmap("tab20", comp_b.shape[1])
    fig, ax = plt.subplots(figsize=(75 * MM2IN, 55 * MM2IN), facecolor="white")
    bottom = np.zeros(len(comp_b))
    for i, ph in enumerate(comp_b.columns):
        ax.bar(comp_b.index.astype(str), comp_b[ph].to_numpy(), bottom=bottom,
               color=cmap_p(i), width=0.75, label=ph, linewidth=0)
        bottom += comp_b[ph].to_numpy()
    ax.set_xlabel("Boundary cluster", fontsize=5.5)
    ax.set_ylabel("Mean neighbourhood proportion", fontsize=5.5)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=4.5, length=2)
    ax.legend(fontsize=3.5, loc="upper left", bbox_to_anchor=(1.02, 1),
              bbox_transform=ax.transAxes, frameon=False,
              title="Phenotype", title_fontsize=4.0, borderaxespad=0)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    _save(fig, "J", title="J   Boundary neighbourhood composition")

    _save_table(comp_b.reset_index(), "suppfig10_J_boundary_cluster_composition")

    # ── Panel K — Proportion of cells per boundary cluster ────────────────────
    print("  Panel K — boundary cluster proportions ...")
    prop_b = (pd.Series(boundary_cluster).value_counts().sort_index()
              .rename_axis("boundary_cluster").reset_index(name="n_cells"))
    prop_b["proportion"] = prop_b["n_cells"] / prop_b["n_cells"].sum()

    cmap_b  = plt.get_cmap("tab10", N_BOUNDARY_CLUSTERS)
    fig, ax = plt.subplots(figsize=(60 * MM2IN, 52 * MM2IN), facecolor="white")
    ax.bar(prop_b["boundary_cluster"].astype(str), prop_b["proportion"],
           color=[cmap_b(c) for c in prop_b["boundary_cluster"]],
           width=0.7, linewidth=0)
    for _, r in prop_b.iterrows():
        ax.text(str(r["boundary_cluster"]), r["proportion"] + 0.008,
                f"{r['proportion']*100:.1f}%", ha="center", va="bottom",
                fontsize=4.0, color="#444444")
    ax.set_xlabel("Boundary cluster", fontsize=5.5)
    ax.set_ylabel("Proportion of boundary cells", fontsize=5.5)
    ax.set_ylim(0, min(1.0, prop_b["proportion"].max() * 1.25))
    ax.tick_params(labelsize=4.5, length=2)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    _save(fig, "K", title="K   Boundary cluster proportions")

    _save_table(prop_b, "suppfig10_K_boundary_cluster_proportions")

else:
    print("    Too few boundary cells — Panels I, J, K will be placeholders.")
    for p in ["I", "J", "K"]:
        _placeholder(p, f"Panel {p} — boundary neighbourhood analysis\n"
                        f"(insufficient boundary cells: {n_boundary})")


# ── Analysis parameters, for the Methods section ──────────────────────────────
_save_table(pd.DataFrame([
    ("pixel_size_um",            PIXEL_SIZE_UM,           "µm per pixel"),
    ("tumour_label",             TUMOUR_LABEL,            "phenotype treated as tumour"),
    ("tumour_dbscan_eps",        TUMOUR_DBSCAN_EPS,       "DBSCAN eps for tumour nests (px)"),
    ("tumour_dbscan_min_samples", TUMOUR_DBSCAN_MINPTS,   "DBSCAN min_samples for tumour nests"),
    ("roi_dbscan_eps",           ROI_DBSCAN_EPS,          "DBSCAN eps for connected ROIs (px)"),
    ("roi_dbscan_min_samples",   ROI_DBSCAN_MINPTS,       "DBSCAN min_samples for connected ROIs"),
    ("boundary_buffer_um",       BOUNDARY_BUFFER_UM,      "expansion / shrink distance for boundaries"),
    ("mask_resolution",          MASK_RESOLUTION,         "density-mask resolution for boundary extraction"),
    ("neighbourhood_radius_um",  NEIGHBOURHOOD_RADIUS_UM, "radius for boundary neighbourhood profiling"),
    ("n_boundary_clusters",      N_BOUNDARY_CLUSTERS,     "k-means clusters (tumour boundary types)"),
    ("random_state",             RANDOM_STATE,            "random seed"),
    ("boundary_method",          "density_mask",          "sv.tl.build_niche_boundaries method"),
], columns=["parameter", "value", "description"]), "suppfig10_analysis_parameters")

print(f"\nAll panels saved to {OUT_DIR}/")
print(f"Tables saved to     {TABLES_DIR}/")
