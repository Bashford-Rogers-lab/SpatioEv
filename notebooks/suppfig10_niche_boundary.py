"""
Supplementary Figure 10 — Neighbourhood identification, ROI selection, and boundary analysis
============================================================================================
Extracted directly from:
  - 05_dev_spatial_niche_boundaries.ipynb   (liver metastasis / exp_1 boundary panels)
  - 09_RA_OA_ECM_cell_spatioev_module_paper_applications.ipynb (RA/OA neighbourhood panels)

Each cell from those notebooks becomes a panel here, with:
  - Publication RC (Arial 6 pt)
  - savefig() call added after each plot

Panels
------
A  : RCN cell-type composition stacked bar (RA/OA)
B  : Spatial scatter coloured by RCN — representative RA sample
C  : Spatial scatter coloured by connected ROI components
D  : RCN abundance distribution across RA vs OA
E  : Liver metastasis phenotype annotation heatmap
F  : [Placeholder] Representative tissue image (DAPI + CK19)
G  : UMAP of liver metastasis cells coloured by phenotype
H  : Tumour / other cells spatial overlay
I  : Tumour nests (DBSCAN component labels)
J  : Tumour niche boundaries with expanded / shrunk contours
K  : Tumour region assignments (core / border / background)

Run from SpatioEv root:
    python notebooks/suppfig10_niche_boundary.py
"""

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR",  "/private/tmp/numba")

# Run from project root
ROOT = Path(__file__).parent.parent
os.chdir(ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import spatioev as sv

# ── Output directory ───────────────────────────────────────────────────────────
OUT_DIR = Path("notebooks/results/suppfig10")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MM2IN = 1 / 25.4

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
                eps=100, min_samples=3)
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
    _save(fig, "D", title="D   RCN abundance across RA and OA")
else:
    _placeholder("D", "Panel D — RCN distribution\n(requires pre-computed RCN column in obs)")


# ══════════════════════════════════════════════════════════════════════════════
# Liver metastasis section — Panels E–K
# Extracted directly from 05_dev_spatial_niche_boundaries.ipynb
# ══════════════════════════════════════════════════════════════════════════════

print("\nLoading liver metastasis data (exp_1) ...")
adata = sv.load_h5ad("data/exp_1.h5ad")

# --- from notebook cell 3 ---
ann = pd.read_csv("results/svm_phenotyping_results.csv", index_col=0)
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
    _save(fig, "E", title="E   Liver metastasis cell annotation")
except Exception as exc:
    print(f"  Panel E error: {exc}")
    _placeholder("E", f"Panel E — Annotation heatmap\n({str(exc)[:60]})")

# ── Panel F — Tissue image placeholder ────────────────────────────────────────
_placeholder("F", "F   Representative tissue image\nDAP + CK19 fluorescence\n(source image file required)")

# ── Panel G — UMAP coloured by phenotype ─────────────────────────────────────
if "X_umap" not in adata.obsm:
    print("  Computing UMAP for Panel G ...")
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
    _save(fig, "G", title="G   Phenotype UMAP")
else:
    _placeholder("G", "Panel G — UMAP\n(umap-learn not available)")

# ── Panel H — Tumour / other overlay ─────────────────────────────────────────
# equivalent to the initial spatial view before boundary computation
print("  Panel H — tumour/other spatial overlay ...")
phenos_h = adata.obs["annotated_clusters_update3"].fillna("other").values
is_tumour = phenos_h == "tumour"

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
_save(fig, "H", title="H   Tumour and other cells")

# ── Pipeline: cluster niches → boundaries → assign regions ────────────────────
# (directly from notebook cells 4–9)
print("  Running tumour niche clustering (DBSCAN) ...")
adata = sv.tl.cluster_spatial_niches(
    adata,
    label_key="annotated_clusters_update3",
    label_value="tumour",
    eps=45,
    min_samples=5,
)

# ── Panel I — DBSCAN tumour nests ─────────────────────────────────────────────
# (from notebook cell 6: sv.pl.plot_spatial_category with tumour_component)
print("  Panel I — DBSCAN tumour nests ...")
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
_save(fig, "I", title="I   Tumour nests (DBSCAN)")

print("  Building niche boundaries ...")
boundary_df = sv.tl.build_niche_boundaries(
    adata,
    component_key="tumour_component",
    min_cluster_size=20,
    method="density_mask",
    mask_resolution=6.0,
    mask_sigma=2.0,
    mask_threshold=0.08,
    mask_closing_size=9,
)

buffered_boundary_df = sv.tl.buffer_niche_boundaries(
    boundary_df,
    component_key="tumour_component",
    expand_by=30 / 0.325,
    shrink_by=30 / 0.325,
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
    boundary_width=30 / 0.325,
)

# ── Panel J — Niche boundaries ────────────────────────────────────────────────
# (from notebook cell 8: sv.pl.plot_niche_boundaries)
print("  Panel J — niche boundaries ...")
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
_save(fig, "J", title="J   Tumour-nest boundaries")

# ── Panel K — Tumour region assignments ───────────────────────────────────────
# (from notebook cell 9–10: add_niche_regions_to_obs + plot_spatial_category)
print("  Panel K — tumour region assignments ...")
adata = sv.add_niche_regions_to_obs(
    adata,
    assignments_df,
    region_key="tumour_region",
    component_key="tumour_component",
)

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
_save(fig, "K", title="K   Tumour region assignments")

print(f"\nAll panels saved to {OUT_DIR}/")
