#!/usr/bin/env python3
"""
fig3_panels.py
==============
Generates Figure 3 panels as SEPARATE PDF files for free arrangement
on a 6.69" × 8.86" (170 × 225 mm) canvas.

Panels
------
A  Spatial pseudotime maps — all 4 Xenium samples (2 × 2 grid)
   Cells coloured by xenium_pseudotime_norm (viridis); non-ductal cells in gray.

B  Ductal gene expression along pseudotime (2 × 3 LOWESS, per-branch)
   Genes: PROX1, CFTR, KRT7 | TFF2, MUC5AC, MKI67

C  Surrounding niche composition along pseudotime (2 × 2 LOWESS, per-sample)
   Cell types: Fibroblasts, Myeloid cells, T cells, Acinar epithelium

D  Branch-discriminating gene expression (1 × 4 LOWESS, per-branch)
   Genes: CFTR, AMY2A, KRT7, PROX1

Outputs (all in paper/notebooks/results/fig3/)
-----------------------------------------
fig3_panelA.pdf / .png
fig3_panelB.pdf / .png
fig3_panelC.pdf / .png
fig3_panelD.pdf / .png

Usage (from project root):
    python notebooks/fig3_panels.py          # all panels
    python notebooks/fig3_panels.py A C      # specific panels only
"""

from __future__ import annotations

import pickle
import sys
from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

try:
    import statsmodels.api as sm
    _HAS_SM = True
except ImportError:
    _HAS_SM = False

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[1]   # SpatioEv/
DATA_XE = ROOT / "data" / "xenium_pancreas_10x"
PT_PATH = DATA_XE / "pseudotime" / "xenium_pseudotime_result_df.pkl"
NF_PATH = DATA_XE / "pooled_xenium_niche_feature_df.pkl"
OUT_DIR = ROOT / "notebooks" / "results" / "fig3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "notebooks"))
from fig2_shared_config import MM2IN, set_pub_rc, make_branch_palette
set_pub_rc()

# ── Constants ─────────────────────────────────────────────────────────────────
SAMPLE_ORDER = ["normal_nondiseased_v1", "pdac_addon_v1",
                "pdac_pancreas_v1",      "pdac_io_v1"]
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

XE_BRANCHES = ["trunk", "branch 1", "branch 2", "branch 3",
                "branch 4", "branch 5", "other"]
BRANCH_PAL  = make_branch_palette(XE_BRANCHES)

KEY = "xenium_ductal_epithelium_component"

PANEL_B_GENES = [
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

SURROUND_COLS = [
    "surround__Fibroblasts__n_cells",
    "surround__Myeloid_cells__n_cells",
    "surround__T_cells__n_cells",
    "surround__pancreatic_acinar_epithelium__n_cells",
]
SURROUND_PANELS = [
    ("prop__Fibroblasts",                  "Fibroblasts",  "desmoplasia ↑"),
    ("prop__Myeloid_cells",                "Myeloid cells","immune niche"),
    ("prop__T_cells",                      "T cells",      "immune infiltration"),
    ("prop__pancreatic_acinar_epithelium", "Acinar epi.",  "normal tissue ↓"),
]

PANEL_D_GENES = [
    ("state__CFTR_expr_z__mean",  "CFTR"),
    ("state__AMY2A_expr_z__mean", "AMY2A"),
    ("state__KRT7_expr_z__mean",  "KRT7"),
    ("state__PROX1_expr_z__mean", "PROX1"),
]

# Panel C (pooled) — 3×3 grid, one LOWESS per subplot (all cells pooled)
# Gene order matches the original fig3_xenium_pseudotime.pdf Panel C.
PANEL_C_POOLED_GENES = [
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


# ── Shared helpers ─────────────────────────────────────────────────────────────
def _lowess(x, y, frac=0.40, delta=0.05):
    """LOWESS smooth; returns (xs, ys) or (None, None) if too few points."""
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 20:
        return None, None
    order = np.argsort(x[valid])
    xv, yv = x[valid][order], y[valid][order]
    if _HAS_SM:
        res = sm.nonparametric.lowess(yv, xv, frac=frac, delta=delta,
                                      return_sorted=True)
        return res[:, 0], res[:, 1]
    # Fallback: edge-padded running-mean smoother (covers full x range)
    w = max(int(frac * len(xv)), 3)
    half = w // 2
    yv_pad = np.pad(yv, (half, w - 1 - half), mode="edge")
    ys = np.convolve(yv_pad, np.ones(w) / w, mode="valid")
    return xv, ys


def _spearman(x, y):
    """Numpy-only Spearman ρ + approximate p-value."""
    n = len(x)
    if n < 5:
        return 0.0, 1.0
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    d2 = np.sum((rx - ry) ** 2)
    rho = 1 - 6 * d2 / max(n * (n ** 2 - 1), 1)
    t   = rho * sqrt((n - 2) / max(1 - rho ** 2, 1e-12))
    p   = erfc(abs(t) / sqrt(2))
    return rho, p


def _save(fig, stem: str):
    """Save PDF + PNG and close figure."""
    pdf = OUT_DIR / f"{stem}.pdf"
    png = OUT_DIR / f"{stem}.png"
    fig.savefig(pdf, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {pdf.name}")


def load_merged() -> pd.DataFrame:
    """Load and merge pseudotime + niche-feature tables."""
    with open(PT_PATH, "rb") as f:
        pt = pickle.load(f)
    with open(NF_PATH, "rb") as f:
        nf = pickle.load(f)

    gene_cols = list(dict.fromkeys(
        [c for c, *_ in PANEL_B_GENES]
        + [c for c, _ in PANEL_D_GENES]
        + [c for c, *_ in PANEL_C_POOLED_GENES]
    ))

    merged = (
        pt[[KEY, "sample_id", "major_branch", "xenium_pseudotime_norm"]]
        .merge(nf[[KEY] + SURROUND_COLS + gene_cols], on=KEY, how="inner")
    )

    # Surround proportions from raw counts
    total = merged[SURROUND_COLS].sum(axis=1).replace(0, np.nan)
    for col in SURROUND_COLS:
        ct = col.replace("surround__", "").replace("__n_cells", "")
        merged[f"prop__{ct}"] = merged[col] / total

    return merged


# ── Panel A — Spatial pseudotime maps (2×2, all 4 samples) ───────────────────
def make_panel_A():
    print("Panel A — spatial pseudotime maps")
    fig, axes = plt.subplots(2, 2, figsize=(6.69, 4.2), facecolor="white")
    fig.subplots_adjust(left=0.01, right=0.89, top=0.95, bottom=0.01,
                        hspace=0.08, wspace=0.05)

    last_sc = None
    for i, sid in enumerate(SAMPLE_ORDER):
        ax = axes[i // 2, i % 2]
        meta_path = DATA_XE / "spatialcellchat" / f"{sid}_cell_meta.csv"

        if not meta_path.exists():
            ax.text(0.5, 0.5, f"{sid}\n(missing)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=5)
            ax.axis("off")
            continue

        meta = pd.read_csv(meta_path,
                           usecols=["x", "y", "Tier_A", "xenium_pseudotime_norm"])
        non_d  = meta[meta["Tier_A"] != "pancreatic ductal epithelium"]
        ductal = meta[
            (meta["Tier_A"] == "pancreatic ductal epithelium") &
            meta["xenium_pseudotime_norm"].notna()
        ]
        if len(non_d) > 30_000:
            non_d = non_d.sample(30_000, random_state=42)

        ax.scatter(non_d["x"], non_d["y"],
                   s=0.03, c="#d8d8d8", linewidths=0,
                   rasterized=True, zorder=1)
        last_sc = ax.scatter(ductal["x"], ductal["y"],
                             c=ductal["xenium_pseudotime_norm"],
                             cmap="viridis", s=0.3, linewidths=0,
                             vmin=0, vmax=1, rasterized=True, zorder=2)

        ax.set_aspect("equal", adjustable="datalim")
        ax.margins(0.02)
        ax.axis("off")
        ax.set_title(SAMPLE_LABELS[sid], fontsize=5.5, pad=2,
                     fontweight="semibold")

    # Shared colorbar on the right
    if last_sc is not None:
        cax = fig.add_axes([0.91, 0.18, 0.014, 0.60])
        cb  = fig.colorbar(last_sc, cax=cax)
        cb.ax.tick_params(labelsize=4)
        cb.set_label("Pseudotime", fontsize=5, labelpad=2)
        cb.set_ticks([0, 0.5, 1])

    fig.text(0.005, 0.985, "A", fontsize=9, fontweight="bold",
             va="top", transform=fig.transFigure)

    _save(fig, "fig3_panelA")


# ── Panel B — Ductal gene LOWESS (2×3, per-branch) ────────────────────────────
def make_panel_B(merged: pd.DataFrame):
    print("Panel B — ductal gene expression LOWESS")
    branches_present = [b for b in XE_BRANCHES
                        if b in merged["major_branch"].unique()]

    nrows, ncols = 3, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.35, 2.5), facecolor="white")
    fig.subplots_adjust(left=0.12, right=0.97, top=0.94, bottom=0.14,
                        hspace=0.45, wspace=0.30)
    rng = np.random.default_rng(0)

    for idx, (col, gene, direction) in enumerate(PANEL_B_GENES):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]

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

            samp = rng.choice(mask.sum(), size=min(120, mask.sum()), replace=False)
            ax.scatter(bx[samp], by[samp], s=0.6, color=color, alpha=0.12,
                       linewidths=0, rasterized=True, zorder=1)

            lx, ly = _lowess(bx, by)
            if lx is None:
                continue
            ax.plot(lx, ly, color=color, lw=1.2, alpha=0.90,
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
        if r == nrows - 1:
            ax.set_xlabel("Pseudotime", fontsize=4.5, labelpad=1)
        else:
            ax.set_xticklabels([])
        if c == 0:
            ax.set_ylabel("z-score", fontsize=4.5, labelpad=2)

    # Branch legend at bottom
    handles = [
        mlines.Line2D([], [], color=BRANCH_PAL[b], lw=1.4,
                      label="Trunk" if b == "trunk"
                      else b.replace("branch ", "Branch "))
        for b in branches_present if b != "other"
    ]
    fig.legend(handles=handles, fontsize=4, ncol=len(handles),
               loc="lower center", bbox_to_anchor=(0.54, 0.00),
               frameon=False, handlelength=1.2, columnspacing=0.8,
               labelspacing=0.25, title="Branch", title_fontsize=4)

    fig.text(0.005, 0.985, "B", fontsize=9, fontweight="bold",
             va="top", transform=fig.transFigure)

    _save(fig, "fig3_panelB")


# ── Panel B (sample) — Ductal gene LOWESS (2×3, per-sample, no branches) ─────
def make_panel_B_sample(merged: pd.DataFrame):
    """Same 9-gene grid as Panel B, but LOWESS lines coloured by sample (not branch)."""
    print("Panel B (sample) — ductal gene expression LOWESS, per-sample")
    samples_present = [s for s in SAMPLE_ORDER
                       if s in merged["sample_id"].unique()]

    nrows, ncols = 3, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.35, 2.5), facecolor="white")
    fig.subplots_adjust(left=0.12, right=0.97, top=0.94, bottom=0.14,
                        hspace=0.45, wspace=0.30)
    rng = np.random.default_rng(0)

    for idx, (col, gene, direction) in enumerate(PANEL_B_GENES):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]

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
            samp = rng.choice(mask.sum(), size=min(100, mask.sum()), replace=False)
            ax.scatter(bx[samp], by[samp], s=0.6, color=color, alpha=0.10,
                       linewidths=0, rasterized=True, zorder=1)
            lx, ly = _lowess(bx, by)
            if lx is not None:
                ax.plot(lx, ly, color=color, lw=1.2, alpha=0.90,
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
        if r == nrows - 1:
            ax.set_xlabel("Pseudotime", fontsize=4.5, labelpad=1)
        else:
            ax.set_xticklabels([])
        if c == 0:
            ax.set_ylabel("z-score", fontsize=4.5, labelpad=2)

    # Sample legend at bottom
    handles = [
        mlines.Line2D([], [], color=SAMPLE_COLORS[s], lw=1.4,
                      label=SAMPLE_LABELS[s])
        for s in samples_present
    ]
    fig.legend(handles=handles, fontsize=4, ncol=len(handles),
               loc="lower center", bbox_to_anchor=(0.54, 0.00),
               frameon=False, handlelength=1.2, columnspacing=0.8,
               labelspacing=0.25, title="Sample", title_fontsize=4)

    fig.text(0.005, 0.985, "B", fontsize=9, fontweight="bold",
             va="top", transform=fig.transFigure)

    _save(fig, "fig3_panelB_sample")


# ── Panel C (pooled) — 9-gene LOWESS, single pooled trend (3×3) ───────────────
def make_panel_C_pooled(merged: pd.DataFrame):
    """9 genes × 3×3 grid. Each subplot: ONE LOWESS line computed on ALL ductal
    cells regardless of branch or sample — the common pseudotime trend."""
    print("Panel C (pooled) — 9-gene common-trend LOWESS")

    nrows, ncols = 3, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.69, 4.2), facecolor="white")
    fig.subplots_adjust(left=0.10, right=0.97, top=0.94, bottom=0.08,
                        hspace=1.0, wspace=0.45)

    pt_v = merged["xenium_pseudotime_norm"].values.astype(float)

    for idx, (col, gene, note) in enumerate(PANEL_C_POOLED_GENES):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]

        y_v   = merged[col].values.astype(float)
        valid = np.isfinite(pt_v) & np.isfinite(y_v)

        ax.axhline(np.nanmean(y_v), color="#cccccc", lw=0.6, ls="--", zorder=0)

        if valid.sum() >= 20:
            # Scatter a faint cloud so density is visible
            rng  = np.random.default_rng(idx)
            samp = rng.choice(valid.sum(), size=min(200, valid.sum()), replace=False)
            ax.scatter(pt_v[valid][samp], y_v[valid][samp],
                       s=0.5, c="#aaaaaa", alpha=0.10,
                       linewidths=0, rasterized=True, zorder=1)

            # Single pooled LOWESS
            lx, ly = _lowess(pt_v[valid], y_v[valid])
            if lx is not None:
                ax.plot(lx, ly, color="#1f4e79", lw=1.5, alpha=0.92,
                        solid_capstyle="round", zorder=3)

            # Spearman ρ
            rho, p = _spearman(pt_v[valid], y_v[valid])
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
            ax.text(0.97, 0.05, f"ρ = {rho:+.2f}{sig}",
                    transform=ax.transAxes, fontsize=4.0,
                    ha="right", va="bottom", color="#444")

        ax.set_title(f"{gene}  ({note})", fontsize=5.0, pad=2,
                     fontweight="semibold", loc="left")
        ax.set_xlim(-0.02, 1.02)
        ax.tick_params(labelsize=4.0, length=2, pad=1)
        if r == nrows - 1:
            ax.set_xlabel("Pseudotime", fontsize=4.5, labelpad=1)
        else:
            ax.set_xticklabels([])
        if c == 0:
            ax.set_ylabel("z-score", fontsize=4.5, labelpad=2)

    fig.text(0.005, 0.985, "C", fontsize=9, fontweight="bold",
             va="top", transform=fig.transFigure)

    _save(fig, "fig3_panelC_pooled")


# ── Panel C — Niche composition LOWESS (2×2, per-sample) ─────────────────────
def make_panel_C(merged: pd.DataFrame):
    print("Panel C — surrounding niche composition LOWESS")
    samples_present = [s for s in SAMPLE_ORDER
                       if s in merged["sample_id"].unique()]

    fig, axes = plt.subplots(2, 2, figsize=(3.35, 2.56), facecolor="white")
    fig.subplots_adjust(left=0.13, right=0.97, top=0.91, bottom=0.20,
                        hspace=0.45, wspace=0.40)
    rng = np.random.default_rng(1)

    for idx, (col, label, note) in enumerate(SURROUND_PANELS):
        r, c = divmod(idx, 2)
        ax = axes[r, c]

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
        if r == 1:
            ax.set_xlabel("Pseudotime", fontsize=4.5, labelpad=1)
        else:
            ax.set_xticklabels([])
        if c == 0:
            ax.set_ylabel("Proportion", fontsize=4.5, labelpad=2)

    # Sample legend at bottom
    handles = [
        mlines.Line2D([], [], color=SAMPLE_COLORS[s], lw=1.4,
                      label=SAMPLE_LABELS[s])
        for s in samples_present
    ]
    fig.legend(handles=handles, fontsize=4, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, 0.00),
               frameon=False, handlelength=1.2, columnspacing=1.0,
               labelspacing=0.25, title="Sample", title_fontsize=4)

    fig.text(0.005, 0.985, "C", fontsize=9, fontweight="bold",
             va="top", transform=fig.transFigure)

    _save(fig, "fig3_panelC")


# ── Panel C (pooled niche) — niche composition LOWESS, single pooled trend ────
def make_panel_C_pooled_niche(merged: pd.DataFrame):
    """Same 2×2 niche-composition grid as Panel C, but ONE pooled LOWESS line
    (all cells together, dark blue) instead of per-sample coloured lines."""
    print("Panel C (pooled niche) — niche composition LOWESS, pooled trend")

    fig, axes = plt.subplots(2, 2, figsize=(3.35, 2.56), facecolor="white")
    fig.subplots_adjust(left=0.13, right=0.97, top=0.91, bottom=0.12,
                        hspace=0.45, wspace=0.40)
    rng = np.random.default_rng(1)

    pt_v = merged["xenium_pseudotime_norm"].values.astype(float)

    for idx, (col, label, note) in enumerate(SURROUND_PANELS):
        r, c = divmod(idx, 2)
        ax = axes[r, c]

        y_v   = merged[col].values.astype(float)
        valid = np.isfinite(pt_v) & np.isfinite(y_v)

        ax.axhline(np.nanmean(y_v[valid]), color="#cccccc", lw=0.6, ls="--", zorder=0)

        if valid.sum() >= 20:
            samp = rng.choice(valid.sum(), size=min(200, valid.sum()), replace=False)
            ax.scatter(pt_v[valid][samp], y_v[valid][samp],
                       s=0.5, c="#aaaaaa", alpha=0.10,
                       linewidths=0, rasterized=True, zorder=1)

            lx, ly = _lowess(pt_v[valid], y_v[valid])
            if lx is not None:
                ax.plot(lx, ly, color="#1f4e79", lw=1.5, alpha=0.92,
                        solid_capstyle="round", zorder=3)

            rho, p = _spearman(pt_v[valid], y_v[valid])
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
            ax.text(0.97, 0.05, f"ρ = {rho:+.2f}{sig}",
                    transform=ax.transAxes, fontsize=4.0,
                    ha="right", va="bottom", color="#444")

        ax.set_title(f"{label}  ({note})", fontsize=5.0, pad=2,
                     fontweight="semibold", loc="left")
        ax.set_xlim(-0.02, 1.02)
        ax.tick_params(labelsize=4.0, length=2, pad=1)
        if r == 1:
            ax.set_xlabel("Pseudotime", fontsize=4.5, labelpad=1)
        else:
            ax.set_xticklabels([])
        if c == 0:
            ax.set_ylabel("Proportion", fontsize=4.5, labelpad=2)

    fig.text(0.005, 0.985, "C", fontsize=9, fontweight="bold",
             va="top", transform=fig.transFigure)

    _save(fig, "fig3_panelC_pooled_niche")


# ── Panel C (branch) — Niche composition LOWESS (2×2, per-branch) ────────────
def make_panel_C_branch(merged: pd.DataFrame):
    """Same 2×2 niche-composition grid as Panel C, but coloured by branch
    (BRANCH_PAL) instead of per-sample."""
    print("Panel C (branch) — niche composition LOWESS, per-branch")
    branches_present = [b for b in XE_BRANCHES
                        if b in merged["major_branch"].unique() and b != "other"]

    fig, axes = plt.subplots(2, 2, figsize=(3.35, 2.56), facecolor="white")
    fig.subplots_adjust(left=0.13, right=0.97, top=0.91, bottom=0.20,
                        hspace=0.45, wspace=0.40)
    rng = np.random.default_rng(3)

    for idx, (col, label, note) in enumerate(SURROUND_PANELS):
        r, c = divmod(idx, 2)
        ax = axes[r, c]

        pt_v = merged["xenium_pseudotime_norm"].values.astype(float)
        y_v  = merged[col].values.astype(float)
        bc   = merged["major_branch"].values

        ax.axhline(np.nanmean(y_v[np.isfinite(y_v)]), color="#cccccc",
                   lw=0.6, ls="--", zorder=0)

        for branch in branches_present:
            color = BRANCH_PAL.get(branch, "#888")
            mask  = (bc == branch) & np.isfinite(pt_v) & np.isfinite(y_v)
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
        if r == 1:
            ax.set_xlabel("Pseudotime", fontsize=4.5, labelpad=1)
        else:
            ax.set_xticklabels([])
        if c == 0:
            ax.set_ylabel("Proportion", fontsize=4.5, labelpad=2)

    # Branch legend at bottom
    handles = [
        mlines.Line2D([], [], color=BRANCH_PAL[b], lw=1.4,
                      label="Trunk" if b == "trunk"
                      else b.replace("branch ", "Branch "))
        for b in branches_present
    ]
    fig.legend(handles=handles, fontsize=4, ncol=len(handles),
               loc="lower center", bbox_to_anchor=(0.5, 0.00),
               frameon=False, handlelength=1.2, columnspacing=0.8,
               labelspacing=0.25, title="Branch", title_fontsize=4)

    fig.text(0.005, 0.985, "C", fontsize=9, fontweight="bold",
             va="top", transform=fig.transFigure)

    _save(fig, "fig3_panelC_branch")


# ── Panel D — Branch-discriminating genes (1×4, per-branch) ──────────────────
def make_panel_D(merged: pd.DataFrame):
    print("Panel D — branch-discriminating gene LOWESS")
    branches_present = [b for b in XE_BRANCHES
                        if b in merged["major_branch"].unique()]

    fig, axes = plt.subplots(1, len(PANEL_D_GENES), figsize=(6.69, 2.5),
                              facecolor="white")
    fig.subplots_adjust(left=0.10, right=0.97, top=0.85, bottom=0.30,
                        wspace=0.45)
    rng = np.random.default_rng(2)

    for idx, (col, gene) in enumerate(PANEL_D_GENES):
        ax = axes[idx]

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
        if idx == 0:
            ax.set_ylabel("z-score", fontsize=4.5, labelpad=2)

    # Branch legend at bottom
    handles = [
        mlines.Line2D([], [], color=BRANCH_PAL[b], lw=1.4,
                      label="Trunk" if b == "trunk"
                      else b.replace("branch ", "Branch "))
        for b in branches_present if b != "other"
    ]
    fig.legend(handles=handles, fontsize=4, ncol=len(handles),
               loc="lower center", bbox_to_anchor=(0.54, 0.00),
               frameon=False, handlelength=1.2, columnspacing=0.8,
               labelspacing=0.25, title="Branch", title_fontsize=4)

    fig.text(0.005, 0.975, "D", fontsize=9, fontweight="bold",
             va="top", transform=fig.transFigure)

    _save(fig, "fig3_panelD")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Panel codes (case-insensitive except Bs/Cp/Cb/Cn):
    #   A    spatial pseudotime maps (2×2)
    #   B    ductal gene LOWESS, per-branch (2×3)
    #   Bs   ductal gene LOWESS, per-sample (2×3)
    #   Cp   common-trend gene LOWESS (3×3, single pooled line)
    #   C    niche composition LOWESS, per-sample (2×2)
    #   Cn   niche composition LOWESS, pooled single line (2×2)
    #   Cb   niche composition LOWESS, per-branch (2×2)
    #   D    branch-diverging gene LOWESS (1×4)
    panels_raw = sys.argv[1:] or ["A", "B", "Bs", "Cp", "C", "Cn", "Cb", "D"]
    panels = [p for p in panels_raw]          # preserve case (Bs, Cp)
    panels_up = [p.upper() for p in panels]

    merged = None
    if any(p in panels_up for p in ["B", "BS", "CP", "C", "CN", "CB", "D"]):
        print("Loading pseudotime + niche-feature data …")
        merged = load_merged()
        print(f"  {len(merged):,} ductal niches loaded")

    if "A" in panels_up:
        make_panel_A()
    if "B" in panels_up:
        make_panel_B(merged)
    if "BS" in panels_up:
        make_panel_B_sample(merged)
    if "CP" in panels_up:
        make_panel_C_pooled(merged)
    if "C" in panels_up:
        make_panel_C(merged)
    if "CN" in panels_up:
        make_panel_C_pooled_niche(merged)
    if "CB" in panels_up:
        make_panel_C_branch(merged)
    if "D" in panels_up:
        make_panel_D(merged)

    print(f"\nDone. Outputs in: {OUT_DIR}")
