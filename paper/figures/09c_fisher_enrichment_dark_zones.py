#!/usr/bin/env python3
"""
09c_fisher_enrichment_dark_zones.py
────────────────────────────────────────────────────────────────────────────────
Per-sample Fisher's exact test: cell-type enrichment in COL6 dark zone regions.

Reproduces the approach from Richard et al. (EMBO Mol Med, 2025)
[DOI 10.1038/s44320-025-00149-7], which computed per-section Fisher's exact
tests comparing cell-type frequencies in COL6A1-rich vs COL6A1-poor regions:

    "Cell subtype frequencies were then measured in each of the regions
    and the frequencies were exported into R, where a Fisher's exact test
    was run."  — Richard et al., Methods

Here the test is applied to the CHP alpha-shape hole regions produced by
09b_spatialcells_col6_dark_segmentation.py.  Each region (core, inner_edge,
outer_edge) is compared to background using a per-sample two-sided Fisher's
exact test, and p-values are adjusted with Benjamini–Hochberg FDR across all
tests.

Inputs (from 09b_spatialcells_col6_dark_segmentation.py):
    DARK_DIR/04_cell_chp_micro_hole_region_assignments.csv

Outputs:
    TABLES_DIR/07_fisher_enrichment_dark_zones.csv         — full results table
    FIGURES_DIR/11_fisher_enrichment_heatmap.png           — summary heatmap
    FIGURES_DIR/11b_fisher_enrichment_per_sample.png       — per-sample heatmaps

Run:
    conda activate spatial-cells-env   # or any env with scipy + statsmodels
    cd /Users/shihongwu/SpatioEv
    python notebooks/09c_fisher_enrichment_dark_zones.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent if _HERE.name == "notebooks" else _HERE

RESULTS_DIR = PROJECT_ROOT / "notebooks" / "results" / "ra_oa_ecm_cell"
DARK_DIR    = RESULTS_DIR / "chp_density_micro_holes_col6_dark_segmentation" / "outputs"
APP_DIR     = RESULTS_DIR / "spatioev_module_paper_applications"
FIGURES_DIR = APP_DIR / "figures"
TABLES_DIR  = APP_DIR / "tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# ── Column keys ────────────────────────────────────────────────────────────────
IMAGE_KEY     = "imageid"
GROUP_KEY     = "pathology"
PHENOTYPE_KEY = "phenotype"

# ── Cell types reported in Richard et al. ─────────────────────────────────────
PAPER_CELL_TYPES = [
    "B cells",
    "CD4 T cells",
    "CD8 T cells",
    "T cells",
    "Dendritic cells",
    "Macrophages",
    "MERTK+ Macrophages",
    "Monocytes",
    "Neutrophils",
    "Vascular cells",
]

# Regions to test against background (matching paper: COL6-poor vs COL6-rich)
TEST_REGIONS  = ["core", "inner_edge", "outer_edge"]
REF_REGION    = "background"
# Exclude outside_tissue cells from all denominators
EXCLUDE_REGIONS = {"outside_tissue"}

# ── Publication figure style ───────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})

REGION_LABELS = {
    "core":       "Core",
    "inner_edge": "Inner edge",
    "outer_edge": "Outer edge",
}


def savefig(fig: plt.Figure, filename: str) -> None:
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=250, bbox_inches="tight")
    print(f"  Saved: {path}")


# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading cell-region assignments…")
cell_dark = pd.read_csv(DARK_DIR / "04_cell_chp_micro_hole_region_assignments.csv")
print(f"  Total cells: {len(cell_dark):,}")
print(f"  Region breakdown:\n{cell_dark['COL6_dark_region'].value_counts().to_string()}")

# Keep only paper cell types, exclude outside_tissue rows
cell_sub = cell_dark[
    cell_dark[PHENOTYPE_KEY].isin(PAPER_CELL_TYPES)
    & ~cell_dark["COL6_dark_region"].isin(EXCLUDE_REGIONS)
].copy()
print(f"  Cells after filtering (paper types, in-tissue): {len(cell_sub):,}")

# ── Per-sample Fisher's exact test ────────────────────────────────────────────
print("\nRunning per-sample Fisher's exact tests…")
rows: list[dict] = []

for (image_id, pathology), grp in cell_sub.groupby([IMAGE_KEY, GROUP_KEY], observed=True):
    region_totals = grp["COL6_dark_region"].value_counts()
    ref_total = int(region_totals.get(REF_REGION, 0))

    if ref_total == 0:
        print(f"  ⚠  {image_id}: no background cells, skipping")
        continue

    for phenotype in PAPER_CELL_TYPES:
        in_ref = grp["COL6_dark_region"].eq(REF_REGION) & grp[PHENOTYPE_KEY].eq(phenotype)
        n_ref_type  = int(in_ref.sum())
        n_ref_other = ref_total - n_ref_type

        for region in TEST_REGIONS:
            test_total = int(region_totals.get(region, 0))
            if test_total == 0:
                continue

            in_test = grp["COL6_dark_region"].eq(region) & grp[PHENOTYPE_KEY].eq(phenotype)
            n_test_type  = int(in_test.sum())
            n_test_other = test_total - n_test_type

            # 2×2 contingency table
            # rows = [test region, background]
            # cols = [this cell type, all other cell types]
            table = [
                [n_test_type,  n_test_other],
                [n_ref_type,   n_ref_other],
            ]

            try:
                odds_ratio, p_val = fisher_exact(table, alternative="two-sided")
            except Exception as exc:
                print(f"  ⚠  Fisher error {image_id} {phenotype} {region}: {exc}")
                odds_ratio, p_val = np.nan, np.nan

            rows.append({
                IMAGE_KEY:               image_id,
                GROUP_KEY:               pathology,
                PHENOTYPE_KEY:           phenotype,
                "region":                region,
                "n_type_in_region":      n_test_type,
                "n_other_in_region":     n_test_other,
                "total_in_region":       test_total,
                "n_type_in_background":  n_ref_type,
                "n_other_in_background": n_ref_other,
                "total_in_background":   ref_total,
                "odds_ratio":            odds_ratio,
                "p_value":               p_val,
            })

fisher_df = pd.DataFrame(rows)
print(f"  Total tests run: {len(fisher_df)}")

if fisher_df.empty:
    print("  ⚠  No tests were generated. Check that the CSV has the expected columns.")
    raise SystemExit(1)

# ── Benjamini–Hochberg FDR correction ─────────────────────────────────────────
valid = fisher_df["p_value"].notna()
_, p_adj, _, _ = multipletests(fisher_df.loc[valid, "p_value"].values, method="fdr_bh")
fisher_df["p_adj"] = np.nan
fisher_df.loc[valid, "p_adj"] = p_adj

fisher_df["log2_OR"]       = np.log2(fisher_df["odds_ratio"].clip(1e-6, 1e6))
fisher_df["neg_log10_padj"] = -np.log10(fisher_df["p_adj"].clip(1e-300))
fisher_df["significant"]    = fisher_df["p_adj"] < 0.05
fisher_df["direction"]      = np.where(
    ~fisher_df["significant"], "ns",
    np.where(fisher_df["odds_ratio"] > 1, "enriched", "depleted"),
)

# ── Print summary ──────────────────────────────────────────────────────────────
total_sig   = fisher_df["significant"].sum()
n_enriched  = (fisher_df["direction"] == "enriched").sum()
n_depleted  = (fisher_df["direction"] == "depleted").sum()
print(f"\n  Significant (FDR < 0.05): {total_sig} / {len(fisher_df)}")
print(f"    Enriched in region:  {n_enriched}")
print(f"    Depleted from region:{n_depleted}")

print("\n  Summary by cell type and region (median OR, fraction of samples significant):")
summ = (
    fisher_df.groupby([PHENOTYPE_KEY, "region"], observed=True)
    .agg(
        median_OR=("odds_ratio", "median"),
        frac_sig=("significant", "mean"),
        n_samples=("odds_ratio", "count"),
    )
    .reset_index()
)
print(summ.to_string(index=False))

# ── Save results table ─────────────────────────────────────────────────────────
out_csv = TABLES_DIR / "07_fisher_enrichment_dark_zones.csv"
fisher_df.to_csv(out_csv, index=False)
print(f"\n  Results saved → {out_csv}")

# ── Figure 1: Summary heatmap (mean log2 OR across samples) ───────────────────
print("\nPlotting Figure 11: summary heatmap…")

ct_order     = [c for c in PAPER_CELL_TYPES if c in fisher_df[PHENOTYPE_KEY].values]
region_order = [r for r in TEST_REGIONS if r in fisher_df["region"].values]
rlabels      = [REGION_LABELS.get(r, r) for r in region_order]

# Mean log2(OR) per cell type × region, averaged across samples
pivot_OR  = (
    fisher_df.groupby([PHENOTYPE_KEY, "region"], observed=True)["log2_OR"]
    .mean()
    .unstack("region")
    .reindex(index=ct_order, columns=region_order)
)
# Fraction of samples where the test is significant
pivot_sig = (
    fisher_df.groupby([PHENOTYPE_KEY, "region"], observed=True)["significant"]
    .mean()
    .unstack("region")
    .reindex(index=ct_order, columns=region_order)
)

vmax = max(np.nanpercentile(np.abs(pivot_OR.values[~np.isnan(pivot_OR.values)]), 95), 1.0)

fig, ax = plt.subplots(figsize=(3.2, 3.8))
sns.heatmap(
    pivot_OR,
    ax=ax,
    cmap="RdBu_r",
    center=0,
    vmin=-vmax,
    vmax=vmax,
    linewidths=0.4,
    linecolor="white",
    cbar_kws={"label": "Mean log2(OR) vs background", "shrink": 0.65},
)
ax.set_xticklabels(rlabels, rotation=35, ha="right")
ax.set_xlabel("")
ax.set_ylabel("")

# Overlay a star where the majority of samples are significant
for i, ct in enumerate(ct_order):
    for j, region in enumerate(region_order):
        try:
            frac = pivot_sig.at[ct, region]
        except KeyError:
            continue
        if frac > 0.5:
            ax.text(j + 0.5, i + 0.5, "*", ha="center", va="center",
                    fontsize=8, color="black", fontweight="bold")

ax.set_title(
    "Cell-type enrichment in COL6 dark zones\nvs background  (* = majority of samples FDR < 0.05)",
    pad=4, fontsize=7,
)
plt.tight_layout()
savefig(fig, "11_fisher_enrichment_heatmap.png")
plt.close()

# ── Figure 2: Per-sample heatmaps ─────────────────────────────────────────────
print("Plotting Figure 11b: per-sample heatmaps…")

images   = sorted(fisher_df[IMAGE_KEY].unique())
n_images = len(images)

fig, axes = plt.subplots(
    1, n_images,
    figsize=(2.5 * n_images, 4.2),
    sharey=True,
)
if n_images == 1:
    axes = [axes]

for ax, image_id in zip(axes, images):
    sub = fisher_df[fisher_df[IMAGE_KEY].eq(image_id)]
    pathology = sub[GROUP_KEY].iloc[0] if not sub.empty else ""

    pivot = (
        sub.pivot(index=PHENOTYPE_KEY, columns="region", values="log2_OR")
        .reindex(index=ct_order, columns=region_order)
    )
    sig_p = (
        sub.pivot(index=PHENOTYPE_KEY, columns="region", values="significant")
        .reindex(index=ct_order, columns=region_order)
    )

    sns.heatmap(
        pivot,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-3,
        vmax=3,
        linewidths=0.3,
        linecolor="white",
        cbar=False,
        xticklabels=rlabels,
        yticklabels=(ax == axes[0]),
    )
    ax.set_xticklabels(rlabels, rotation=40, ha="right", fontsize=5)
    ax.set_title(f"{image_id}\n({pathology})", fontsize=6)
    ax.set_xlabel("")
    ax.set_ylabel("")

    for i, ct in enumerate(ct_order):
        for j, region in enumerate(region_order):
            try:
                is_sig = bool(sig_p.at[ct, region])
            except (KeyError, ValueError):
                is_sig = False
            if is_sig:
                ax.text(j + 0.5, i + 0.5, "*", ha="center", va="center",
                        fontsize=7, color="black", fontweight="bold")

axes[0].set_ylabel("Cell type")

# Shared colorbar
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

sm = ScalarMappable(cmap="RdBu_r", norm=Normalize(vmin=-3, vmax=3))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes[-1], shrink=0.65, pad=0.03)
cbar.set_label("log2(OR) vs background", fontsize=6)

fig.suptitle(
    "Per-sample Fisher's exact test — COL6 dark zone enrichment\n(* = FDR < 0.05)",
    fontsize=7, y=1.01,
)
plt.tight_layout()
savefig(fig, "11b_fisher_enrichment_per_sample.png")
plt.close()

print("\n✓ Done.")
print(f"\nOutputs:")
print(f"  {TABLES_DIR / '07_fisher_enrichment_dark_zones.csv'}")
print(f"  {FIGURES_DIR / '11_fisher_enrichment_heatmap.png'}")
print(f"  {FIGURES_DIR / '11b_fisher_enrichment_per_sample.png'}")
