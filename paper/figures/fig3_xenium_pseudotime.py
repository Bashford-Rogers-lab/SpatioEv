"""
Figure 3 — Xenium spatial trajectory: ductal reprogramming and niche dynamics
=============================================================================
Five-panel figure:

  A  Ductal cell gene expression along pseudotime (2 rows × 3 cols LOWESS,
     per-branch colored lines + faint scatter + Spearman ρ)
     Genes: PROX1, CFTR, KRT7 | TFF2, MUC5AC, MKI67

  B  Surrounding cell-type composition along pseudotime (2 rows × 2 cols,
     per-sample LOWESS)
     Cell types: Fibroblasts, Myeloid_cells, T_cells, pancreatic_acinar_epithelium

  C  Branch-specific gene expression divergence (1 row × 4 subpanels,
     per-branch LOWESS; genes: CFTR, AMY2A, KRT7, PROX1)

  D  Cell-cell interaction dynamics (line plot: N active L-R pairs per
     pseudotime bin × sample; CellChat lr_interaction_summary.csv)

  E  Spatial pseudotime map — pdac_pancreas_v1, gray background + ductal
     cells colored by xenium_pseudotime_norm (viridis)

Width: 170 mm   Height: ~230 mm

Run:
    python notebooks/fig3_xenium_pseudotime.py

Output: paper/notebooks/results/fig3/fig3_xenium_pseudotime.pdf (.png)
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import statsmodels.api as sm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from fig2_shared_config import MM2IN, set_pub_rc, make_branch_palette

set_pub_rc()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).parent.parent
DATA_XE  = BASE / "data" / "xenium_pancreas_10x"
PT_PATH  = DATA_XE / "pseudotime" / "xenium_pseudotime_result_df.pkl"
NF_PATH  = DATA_XE / "pooled_xenium_niche_feature_df.pkl"
CC_PATH  = DATA_XE / "spatialcellchat" / "lr_interaction_summary.csv"
SPATIAL_META = DATA_XE / "spatialcellchat" / "pdac_pancreas_v1_cell_meta.csv"

OUT_DIR = BASE  / "paper" / "notebooks" / "results" / "fig3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KEY = "xenium_ductal_epithelium_component"

# ── Panel C — 9-gene common-trend (3×3, single pooled LOWESS) ─────────────────
# One LOWESS line per subplot = trend common across all branches and samples.
PANEL_C_POOLED = [
    ("state__EPCAM_expr_z__mean",  "EPCAM",  "epithelial identity"),
    ("state__KRT7_expr_z__mean",   "KRT7",   "pan-ductal"),
    ("state__CFTR_expr_z__mean",   "CFTR",   "duct function ↓"),
    ("state__AGR3_expr_z__mean",   "AGR3",   "secretory duct"),
    ("state__TFF2_expr_z__mean",   "TFF2",   "gastric meta ↑"),
    ("state__MUC5AC_expr_z__mean", "MUC5AC", "gastric meta ↑"),
    ("state__PROX1_expr_z__mean",  "PROX1",  "normal ductal ↓"),
    ("state__MKI67_expr_z__mean",  "MKI67",  "proliferation ↑"),
    ("state__UBE2C_expr_z__mean",  "UBE2C",  "cell cycle ↑"),
]

# ── Branches / colours ────────────────────────────────────────────────────────
XE_BRANCHES = ["trunk", "branch 1", "branch 2", "branch 3",
                "branch 4", "branch 5", "other"]
BRANCH_PAL  = make_branch_palette(XE_BRANCHES)

SAMPLE_COLORS = {
    "normal_nondiseased_v1": "#2ca02c",
    "pdac_addon_v1":         "#ff7f0e",
    "pdac_pancreas_v1":      "#d62728",
    "pdac_io_v1":            "#9467bd",
}
SAMPLE_LABELS = {
    "normal_nondiseased_v1": "Normal",
    "pdac_addon_v1":         "PDAC (add-on)",
    "pdac_pancreas_v1":      "PDAC (primary)",
    "pdac_io_v1":            "PDAC (IO)",
}

# ── Panel A genes (global pseudotime trend) ───────────────────────────────────
PANEL_A_GENES = [
    ("state__PROX1_expr_z__mean",  "PROX1",   "↓ along PT"),
    ("state__CFTR_expr_z__mean",   "CFTR",    "variable by branch"),
    ("state__KRT7_expr_z__mean",   "KRT7",    "↑ branches 1 & 5"),
    ("state__TFF2_expr_z__mean",   "TFF2",    "↑ along PT"),
    ("state__MUC5AC_expr_z__mean", "MUC5AC",  "↑ along PT"),
    ("state__MKI67_expr_z__mean",  "MKI67",   "↑ along PT"),
]

# ── Panel C genes (branch discriminators) ────────────────────────────────────
PANEL_C_GENES = [
    ("state__CFTR_expr_z__mean",  "CFTR"),
    ("state__AMY2A_expr_z__mean", "AMY2A"),
    ("state__KRT7_expr_z__mean",  "KRT7"),
    ("state__PROX1_expr_z__mean", "PROX1"),
]

# ── LOWESS / Spearman helpers ──────────────────────────────────────────────────
def _lowess(x, y, frac=0.40, delta=0.05):
    """LOWESS; delta in normalised [0–1] pseudotime units."""
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 20:
        return None, None
    order = np.argsort(x[valid])
    res = sm.nonparametric.lowess(
        y[valid][order], x[valid][order],
        frac=frac, delta=delta, return_sorted=True,
    )
    return res[:, 0], res[:, 1]


def _spearman(x, y):
    """Numpy-only Spearman ρ + approximate p-value."""
    from math import erfc, sqrt
    n = len(x)
    if n < 5:
        return 0.0, 1.0
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    d2 = np.sum((rx - ry) ** 2)
    rho = 1 - 6 * d2 / max(n * (n ** 2 - 1), 1)
    t   = rho * np.sqrt((n - 2) / max(1 - rho ** 2, 1e-12))
    p   = erfc(abs(t) / sqrt(2))
    return rho, p


# ── Data loading ──────────────────────────────────────────────────────────────
def load_merged():
    with open(PT_PATH, "rb") as f:
        pt = pickle.load(f)
    with open(NF_PATH, "rb") as f:
        nf = pickle.load(f)

    surround_cols = [
        "surround__Fibroblasts__n_cells",
        "surround__Myeloid_cells__n_cells",
        "surround__T_cells__n_cells",
        "surround__pancreatic_acinar_epithelium__n_cells",
    ]
    gene_cols = list(dict.fromkeys(
        [c for c, *_ in PANEL_A_GENES]
        + [c for c, _ in PANEL_C_GENES]
        + [c for c, *_ in PANEL_C_POOLED]
    ))

    merged = (
        pt[[KEY, "sample_id", "major_branch", "xenium_pseudotime_norm"]]
        .merge(nf[[KEY] + surround_cols + gene_cols], on=KEY, how="inner")
    )

    # Proportions from absolute counts
    total = merged[surround_cols].sum(axis=1).replace(0, np.nan)
    for col in surround_cols:
        ct = col.replace("surround__", "").replace("__n_cells", "")
        merged[f"prop__{ct}"] = merged[col] / total

    return merged


# ── Panel A — ductal gene expression per branch ───────────────────────────────
def draw_panel_A(gs_outer, merged, branches_present):
    nrows, ncols = 2, 3
    gs = gridspec.GridSpecFromSubplotSpec(
        nrows, ncols, subplot_spec=gs_outer,
        hspace=0.90, wspace=0.42,
    )
    rng = np.random.default_rng(0)

    for idx, (col, gene, direction) in enumerate(PANEL_A_GENES):
        r, c = divmod(idx, ncols)
        ax = plt.gcf().add_subplot(gs[r, c])

        pt_v = merged["xenium_pseudotime_norm"].values.astype(float)
        y_v  = merged[col].values.astype(float)
        bc   = merged["major_branch"].values

        ax.axhline(np.nanmean(y_v), color="#cccccc", lw=0.6, ls="--", zorder=0)

        for branch in branches_present:
            if branch == "other":
                continue
            color = BRANCH_PAL.get(branch, "#888")
            mask  = (bc == branch) & np.isfinite(pt_v) & np.isfinite(y_v)
            if mask.sum() < 20:
                continue
            bx, by = pt_v[mask], y_v[mask]
            lo, hi = np.percentile(bx, 5), np.percentile(bx, 95)

            samp = rng.choice(mask.sum(), size=min(120, mask.sum()), replace=False)
            ax.scatter(bx[samp], by[samp], s=0.6, color=color, alpha=0.12,
                       linewidths=0, rasterized=True, zorder=1)

            lx, ly = _lowess(bx, by)
            if lx is None:
                continue
            inr = (lx >= lo) & (lx <= hi)
            if inr.sum() > 5:
                ax.plot(lx[inr], ly[inr], color=color, lw=1.2, alpha=0.90,
                        solid_capstyle="round", zorder=3)

        valid = np.isfinite(pt_v) & np.isfinite(y_v)
        if valid.sum() > 30:
            rho, p = _spearman(pt_v[valid], y_v[valid])
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
            ax.text(0.97, 0.05, f"ρ = {rho:+.2f}{sig}",
                    transform=ax.transAxes, fontsize=4.0,
                    ha="right", va="bottom", color="#444")

        ax.set_title(f"{gene}  ({direction})", fontsize=5.0, pad=2,
                     fontweight="semibold", loc="left")
        ax.set_xlim(-0.02, 1.02)
        ax.tick_params(labelsize=4.0, length=2, pad=1)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        if r == nrows - 1:
            ax.set_xlabel("Pseudotime", fontsize=4.5, labelpad=1)
        else:
            ax.set_xticklabels([])
        if c == 0:
            ax.set_ylabel("z-score", fontsize=4.5, labelpad=2)


# ── Panel B — surrounding cell composition ────────────────────────────────────
SURROUND_PANELS = [
    ("prop__Fibroblasts",                 "Fibroblasts",  "desmoplasia ↑"),
    ("prop__Myeloid_cells",               "Myeloid cells","immune niche"),
    ("prop__T_cells",                     "T cells",      "immune infiltration"),
    ("prop__pancreatic_acinar_epithelium","Acinar epi.",  "normal tissue ↓"),
]

def draw_panel_B(gs_outer, merged):
    gs = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs_outer,
        hspace=0.90, wspace=0.42,
    )
    samples_present = [s for s in SAMPLE_COLORS
                       if s in merged["sample_id"].unique()]
    rng = np.random.default_rng(1)

    for idx, (col, label, note) in enumerate(SURROUND_PANELS):
        r, c = divmod(idx, 2)
        ax = plt.gcf().add_subplot(gs[r, c])

        pt_v = merged["xenium_pseudotime_norm"].values.astype(float)
        y_v  = merged[col].values.astype(float)
        sv   = merged["sample_id"].values

        ax.axhline(np.nanmean(y_v), color="#cccccc", lw=0.6, ls="--", zorder=0)

        for sid in samples_present:
            color = SAMPLE_COLORS[sid]
            mask  = (sv == sid) & np.isfinite(pt_v) & np.isfinite(y_v)
            if mask.sum() < 15:
                continue
            bx, by = pt_v[mask], y_v[mask]
            samp = rng.choice(mask.sum(), size=min(80, mask.sum()), replace=False)
            ax.scatter(bx[samp], by[samp], s=0.6, color=color, alpha=0.10,
                       linewidths=0, rasterized=True, zorder=1)
            lx, ly = _lowess(bx, by)
            if lx is not None:
                ax.plot(lx, ly, color=color, lw=1.2, alpha=0.90,
                        solid_capstyle="round", zorder=3)

        ax.set_title(f"{label}  ({note})", fontsize=5.0, pad=2,
                     fontweight="semibold", loc="left")
        ax.set_xlim(-0.02, 1.02)
        ax.tick_params(labelsize=4.0, length=2, pad=1)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        if r == 1:
            ax.set_xlabel("Pseudotime", fontsize=4.5, labelpad=1)
        else:
            ax.set_xticklabels([])
        if c == 0:
            ax.set_ylabel("Proportion", fontsize=4.5, labelpad=2)


# ── Panel C — branch-specific gene expression divergence ─────────────────────
def draw_panel_C(gs_outer, merged, branches_present):
    gs = gridspec.GridSpecFromSubplotSpec(
        1, len(PANEL_C_GENES), subplot_spec=gs_outer,
        hspace=0.0, wspace=0.42,
    )
    rng = np.random.default_rng(2)

    for idx, (col, gene) in enumerate(PANEL_C_GENES):
        ax = plt.gcf().add_subplot(gs[0, idx])

        pt_v = merged["xenium_pseudotime_norm"].values.astype(float)
        y_v  = merged[col].values.astype(float)
        bc   = merged["major_branch"].values

        ax.axhline(np.nanmean(y_v), color="#cccccc", lw=0.6, ls="--", zorder=0)

        for branch in branches_present:
            if branch == "other":
                continue
            color = BRANCH_PAL.get(branch, "#888")
            mask  = (bc == branch) & np.isfinite(pt_v) & np.isfinite(y_v)
            if mask.sum() < 15:
                continue
            bx, by = pt_v[mask], y_v[mask]
            lo, hi = np.percentile(bx, 5), np.percentile(bx, 95)

            samp = rng.choice(mask.sum(), size=min(80, mask.sum()), replace=False)
            ax.scatter(bx[samp], by[samp], s=0.6, color=color, alpha=0.10,
                       linewidths=0, rasterized=True, zorder=1)

            lx, ly = _lowess(bx, by)
            if lx is None:
                continue
            inr = (lx >= lo) & (lx <= hi)
            if inr.sum() > 5:
                ax.plot(lx[inr], ly[inr], color=color, lw=1.4, alpha=0.90,
                        solid_capstyle="round", zorder=3)

        ax.set_title(gene, fontsize=6.0, pad=3, fontweight="bold", loc="center")
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel("Pseudotime", fontsize=4.5, labelpad=1)
        ax.tick_params(labelsize=4.0, length=2, pad=1)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        if idx == 0:
            ax.set_ylabel("z-score", fontsize=4.5, labelpad=2)


# ── Panel D — CellChat LR dynamics ────────────────────────────────────────────
BIN_ORDER = ["bin1", "bin2", "bin3", "bin4", "bin5"]
BIN_XLAB  = ["1\n(early)", "2", "3", "4", "5\n(late)"]

def draw_panel_D(ax, lr_df):
    sig = lr_df[lr_df["prob"] > 1e-7]
    n_per = (sig.groupby(["sample_id", "pt_bin"]).size()
               .reset_index(name="n_lr"))

    for sid in SAMPLE_COLORS:
        sub = n_per[n_per["sample_id"] == sid].set_index("pt_bin")
        y = [sub.at[b, "n_lr"] if b in sub.index else np.nan
             for b in BIN_ORDER]
        ax.plot(range(5), y, "-o",
                color=SAMPLE_COLORS[sid], lw=1.4, ms=3.5,
                label=SAMPLE_LABELS[sid], zorder=3)

    ax.set_xticks(range(5))
    ax.set_xticklabels(BIN_XLAB, fontsize=4.0)
    ax.set_xlabel("Pseudotime bin", fontsize=4.5, labelpad=1)
    ax.set_ylabel("N active L-R pairs", fontsize=4.5, labelpad=2)
    ax.tick_params(labelsize=4.0, length=2, pad=1)
    ax.legend(fontsize=4.0, frameon=False, loc="upper left",
              handlelength=1.2, labelspacing=0.3)
    ax.set_title("Cell-cell interaction activity", fontsize=5.0, pad=3,
                 fontweight="semibold", loc="left")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)


# ── Panel E — spatial pseudotime map ─────────────────────────────────────────
def draw_panel_E(ax):
    meta = pd.read_csv(SPATIAL_META)
    cells = meta.dropna(subset=["x", "y"])

    non_d = cells[cells["Tier_A"] != "pancreatic ductal epithelium"]
    ax.scatter(non_d["x"], non_d["y"],
               s=0.05, c="#d0d0d0", linewidths=0, rasterized=True, zorder=1)

    ductal = cells[
        (cells["Tier_A"] == "pancreatic ductal epithelium") &
        cells["xenium_pseudotime_norm"].notna()
    ]
    sc = ax.scatter(ductal["x"], ductal["y"],
                    c=ductal["xenium_pseudotime_norm"],
                    cmap="viridis", s=0.3, linewidths=0,
                    vmin=0, vmax=1, rasterized=True, zorder=2)

    cb = plt.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("Pseudotime", fontsize=4.0, labelpad=2)
    cb.ax.tick_params(labelsize=3.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("pdac_pancreas_v1 — spatial pseudotime", fontsize=5.0, pad=3,
                 fontweight="semibold", loc="left")


# ── Figure assembly ────────────────────────────────────────────────────────────
def make_figure():
    merged = load_merged()
    lr_df  = pd.read_csv(CC_PATH)

    branches_present = [b for b in XE_BRANCHES
                        if b in merged["major_branch"].unique()]

    fig = plt.figure(figsize=(170 * MM2IN, 230 * MM2IN), facecolor="white")

    # Outer grid: 4 rows
    # 0: A + B (tall — 2-row gene/surround grids)
    # 1: C (medium — 1-row branch divergence)
    # 2: D + E (tall — line chart + spatial)
    # 3: legend strip (slim)
    outer = gridspec.GridSpec(
        4, 1, figure=fig,
        left=0.07, right=0.97,
        top=0.96, bottom=0.06,
        hspace=0.52,
        height_ratios=[2.4, 1.15, 1.85, 0.20],
    )

    # Row 0 — A (3/5) + B (2/5)
    row0 = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[0],
        wspace=0.30, width_ratios=[3, 2],
    )
    draw_panel_A(row0[0], merged, branches_present)
    draw_panel_B(row0[1], merged)

    # Row 1 — C (full width)
    draw_panel_C(outer[1], merged, branches_present)

    # Row 2 — D (2/5) + E (3/5)
    row2 = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[2],
        wspace=0.30, width_ratios=[2, 3],
    )
    ax_D = fig.add_subplot(row2[0])
    draw_panel_D(ax_D, lr_df)
    ax_E = fig.add_subplot(row2[1])
    draw_panel_E(ax_E)

    # Panel labels (positions need canvas to be rendered first)
    fig.canvas.draw()
    label_kw = dict(fontsize=7, fontweight="bold", transform=fig.transFigure,
                    va="top")
    # Approximate y-positions based on height_ratios
    top_a  = 0.960
    top_c  = 1 - (2.4 / (2.4 + 1.15 + 1.85 + 0.20)) * (0.96 - 0.06) - 0.04
    top_de = 1 - ((2.4 + 1.15) / (2.4 + 1.15 + 1.85 + 0.20)) * (0.96 - 0.06) - 0.04

    fig.text(0.07, top_a,  "A", **label_kw)
    fig.text(0.64, top_a,  "B", **label_kw)
    fig.text(0.07, top_c,  "C", **label_kw)
    fig.text(0.07, top_de, "D", **label_kw)
    fig.text(0.44, top_de, "E", **label_kw)

    # ── Shared legends ────────────────────────────────────────────────────────
    # Branch legend (panels A and C)
    branch_handles = [
        mlines.Line2D([], [], color=BRANCH_PAL[b], lw=1.4,
                      label=("Trunk" if b == "trunk" else b.replace("branch ", "Branch ")))
        for b in branches_present if b != "other"
    ]
    fig.legend(handles=branch_handles,
               fontsize=3.8, ncol=min(6, len(branch_handles)),
               loc="lower left", bbox_to_anchor=(0.02, 0.01),
               frameon=False, handlelength=1.2, columnspacing=0.8,
               labelspacing=0.25,
               title="Branches (A, C)", title_fontsize=3.8)

    # Sample legend (panel B)
    sample_handles = [
        mlines.Line2D([], [], color=SAMPLE_COLORS[s], lw=1.4,
                      label=SAMPLE_LABELS[s])
        for s in SAMPLE_COLORS
    ]
    fig.legend(handles=sample_handles,
               fontsize=3.8, ncol=2,
               loc="lower right", bbox_to_anchor=(0.98, 0.01),
               frameon=False, handlelength=1.2, columnspacing=0.8,
               labelspacing=0.25,
               title="Samples (B, D)", title_fontsize=3.8)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_pdf = OUT_DIR / "fig3_xenium_pseudotime.pdf"
    out_png = OUT_DIR / "fig3_xenium_pseudotime.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved:\n  {out_pdf}\n  {out_png}")
    plt.show()


# ── Standalone panel helpers ──────────────────────────────────────────────────
def make_panel_C_standalone():
    """Generate the 9-gene common-trend LOWESS (3×3) as its own PDF.

    ── PUBLISHED AS **FIGURE 3B** ──────────────────────────────────────────────
    Output file: fig3_panelC_standalone.pdf
    Invoke with: python paper/figures/fig3_xenium_pseudotime.py C

    NOTE the code collision: `C` here produces Figure 3B, whereas `C` in
    fig3_panels.py produces Figure 3E (per-sample niche composition).
    See the panel map in the fig3_panels.py module docstring.

    Each subplot shows ONE LOWESS line (frac = 0.40) computed on all ductal
    cells pooled, ignoring branch and sample, representing the common
    pseudotime trend. Genes: EPCAM, KRT7, CFTR, AGR3, TFF2, MUC5AC, PROX1,
    MKI67, UBE2C. Plotted against `xenium_pseudotime_norm` (un-centred).
    """
    merged = load_merged()

    nrows, ncols = 3, 3
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(85 * MM2IN, 65 * MM2IN),
                             facecolor="white")
    fig.subplots_adjust(left=0.12, right=0.97, top=0.94, bottom=0.10,
                        hspace=0.45, wspace=0.30)

    pt_v = merged["xenium_pseudotime_norm"].values.astype(float)

    for idx, (col, gene, note) in enumerate(PANEL_C_POOLED):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]

        y_v   = merged[col].values.astype(float)
        valid = np.isfinite(pt_v) & np.isfinite(y_v)

        ax.axhline(np.nanmean(y_v), color="#cccccc", lw=0.6, ls="--", zorder=0)

        if valid.sum() >= 20:
            rng  = np.random.default_rng(idx)
            samp = rng.choice(valid.sum(), size=min(200, valid.sum()), replace=False)
            ax.scatter(pt_v[valid][samp], y_v[valid][samp],
                       s=0.5, c="#aaaaaa", alpha=0.10,
                       linewidths=0, rasterized=True, zorder=1)

            from statsmodels.nonparametric.smoothers_lowess import lowess as _sm_lowess
            order = np.argsort(pt_v[valid])
            res   = sm.nonparametric.lowess(
                y_v[valid][order], pt_v[valid][order],
                frac=0.40, delta=0.05, return_sorted=True,
            )
            ax.plot(res[:, 0], res[:, 1],
                    color="#1f4e79", lw=1.5, alpha=0.92,
                    solid_capstyle="round", zorder=3)

            rho, p = _spearman(pt_v[valid], y_v[valid])
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
            ax.text(0.97, 0.05, f"ρ = {rho:+.2f}{sig}",
                    transform=ax.transAxes, fontsize=4.0,
                    ha="right", va="bottom", color="#444")

        ax.set_title(f"{gene}  ({note})", fontsize=5.0, pad=2,
                     fontweight="semibold", loc="left")
        ax.set_xlim(-0.02, 1.02)
        ax.tick_params(labelsize=4.0, length=2, pad=1)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        if r == nrows - 1:
            ax.set_xlabel("Pseudotime", fontsize=4.5, labelpad=1)
        else:
            ax.set_xticklabels([])
        if c == 0:
            ax.set_ylabel("z-score", fontsize=4.5, labelpad=2)

    fig.text(0.01, 0.97, "C", fontsize=8, fontweight="bold",
             va="top", transform=fig.transFigure)

    out_pdf = OUT_DIR / "fig3_panelC_standalone.pdf"
    out_png = OUT_DIR / "fig3_panelC_standalone.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved:\n  {out_pdf}\n  {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    # Usage:
    #   python notebooks/fig3_xenium_pseudotime.py         → full composite figure
    #   python notebooks/fig3_xenium_pseudotime.py C       → Panel C standalone PDF
    #                                                         (9-gene pooled-trend 3×3)
    args = sys.argv[1:]
    if "C" in [a.upper() for a in args]:
        make_panel_C_standalone()
    else:
        make_figure()
