"""
Supplementary Figure — Xenium spatial trajectory characterization
=================================================================
Five sections:

  A  Niche-level UMAP (4 sub-panels)
       pseudotime (viridis) | branch (branch palette) |
       disease group | sample

  B  Spatial pseudotime maps — all 4 samples (2 × 2 grid)
       Gray non-ductal background; ductal cells colored by
       xenium_pseudotime_norm (viridis)

  C  Branch composition per sample (normalized stacked bar)
       Shows how ductal niches distribute across branches
       in each sample

  D  Xenium trajectory module scores vs pseudotime
       (per-branch LOWESS + Spearman ρ, 1 × 4)
       epithelial identity | PanIN-like | desmoplastic | proliferation

  E  Pathology / histology score validation vs pseudotime
       (per-branch LOWESS + Spearman ρ, 1 × 4)
       normal duct-like | ADM/PanIN | desmoplastic tumor | immune inflamed

Width: 170 mm   Height: ~310 mm

Run:
    python notebooks/suppfig_xenium_trajectory.py

Output: paper/notebooks/results/suppfig_xenium/suppfig_xenium_trajectory.pdf (.png)
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import statsmodels.api as sm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from fig2_shared_config import MM2IN, set_pub_rc, make_branch_palette

set_pub_rc()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = Path(__file__).parent.parent
DATA_XE = BASE / "data" / "xenium_pancreas_10x"
PT_PATH = DATA_XE / "pseudotime" / "xenium_pseudotime_result_df.pkl"

OUT_DIR = BASE / "notebooks" / "results" / "suppfig_xenium"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour palettes ───────────────────────────────────────────────────────────
XE_BRANCHES = ["trunk", "branch 1", "branch 2", "branch 3",
                "branch 4", "branch 5", "other"]
BRANCH_PAL  = make_branch_palette(XE_BRANCHES)

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

DISEASE_COLORS = {"NormalPancreas": "#2ca02c", "PDAC": "#c0392b"}
STAGE_COLORS   = {
    "Normal":       "#2ca02c",
    "Stage IIB":    "#ff7f0e",
    "Stage III":    "#d62728",
    "Not provided": "#aaaaaa",
}

# ── Module-score panels ────────────────────────────────────────────────────────
PANEL_D_SCORES = [
    ("xenium_epithelial_identity_score",   "Epithelial\nidentity",   "↓ along PT"),
    ("xenium_panin_like_remodeling_score", "PanIN-like\nremodeling", "↑ along PT"),
    ("xenium_desmoplastic_context_score",  "Desmoplastic\ncontext",  "↑ along PT"),
    ("xenium_proliferation_score",         "Proliferation",          "↑ along PT"),
]
PANEL_E_SCORES = [
    ("histology__normal_duct_like_score",  "Normal\nduct-like",      "↓ along PT"),
    ("histology__adm_panin_like_score",    "ADM / PanIN",            "↑ along PT"),
    ("histology__desmoplastic_tumor_score","Desmoplastic\ntumor",    "↑ along PT"),
    ("histology__immune_inflamed_score",   "Immune\ninflamed",       "variable"),
]

# ── LOWESS / Spearman helpers ──────────────────────────────────────────────────
def _lowess(x, y, frac=0.40, delta=0.05):
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


def _score_lowess_panel(ax, pt, col, label, note, branches_present, rng):
    """Per-branch LOWESS on a score column vs pseudotime_norm."""
    pt_v  = pt["xenium_pseudotime_norm"].values.astype(float)
    y_v   = pt[col].values.astype(float)
    bc    = pt["major_branch"].values

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

        samp = rng.choice(mask.sum(), size=min(100, mask.sum()), replace=False)
        ax.scatter(bx[samp], by[samp], s=0.8, color=color, alpha=0.12,
                   linewidths=0, rasterized=True, zorder=1)

        lx, ly = _lowess(bx, by)
        if lx is None:
            continue
        inr = (lx >= lo) & (lx <= hi)
        if inr.sum() > 5:
            ax.plot(lx[inr], ly[inr], color=color, lw=1.3, alpha=0.90,
                    solid_capstyle="round", zorder=3)

    valid = np.isfinite(pt_v) & np.isfinite(y_v)
    if valid.sum() > 30:
        rho, p = _spearman(pt_v[valid], y_v[valid])
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
        ax.text(0.97, 0.05, f"ρ = {rho:+.2f}{sig}",
                transform=ax.transAxes, fontsize=3.8,
                ha="right", va="bottom", color="#444")

    ax.set_title(f"{label}  ({note})", fontsize=4.8, pad=2,
                 fontweight="semibold", loc="left")
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Pseudotime", fontsize=4.5, labelpad=1)
    ax.tick_params(labelsize=4.0, length=2, pad=1)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)


# ── Panel A — UMAP (4 colourings) ────────────────────────────────────────────
def draw_panel_A(gs, pt):
    sub = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=gs, wspace=0.35)
    fig = plt.gcf()

    # A1 — pseudotime
    ax = fig.add_subplot(sub[0])
    sc = ax.scatter(pt["UMAP1"], pt["UMAP2"],
                    c=pt["xenium_pseudotime_norm"], cmap="viridis",
                    s=0.6, linewidths=0, rasterized=True, alpha=0.8)
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Pseudotime", fontsize=3.5, labelpad=1)
    cb.ax.tick_params(labelsize=3.0)
    ax.set_title("Pseudotime", fontsize=5.0, pad=2, fontweight="semibold")
    ax.axis("off")

    # A2 — branch
    ax = fig.add_subplot(sub[1])
    branch_order = [b for b in XE_BRANCHES if b in pt["major_branch"].unique()]
    for branch in branch_order:
        mask = pt["major_branch"] == branch
        ax.scatter(pt.loc[mask, "UMAP1"], pt.loc[mask, "UMAP2"],
                   c=BRANCH_PAL.get(branch, "#ccc"),
                   s=0.6, linewidths=0, rasterized=True, alpha=0.8,
                   label=branch.replace("branch ", "Branch ").replace("trunk", "Trunk"))
    ax.set_title("Branch", fontsize=5.0, pad=2, fontweight="semibold")
    ax.axis("off")
    ax.legend(fontsize=3.2, frameon=False, loc="lower left",
              markerscale=2.5, handletextpad=0.3, labelspacing=0.25,
              borderaxespad=0)

    # A3 — disease group
    ax = fig.add_subplot(sub[2])
    for dg, color in DISEASE_COLORS.items():
        mask = pt["disease_group"] == dg
        label = "Normal" if dg == "NormalPancreas" else "PDAC"
        ax.scatter(pt.loc[mask, "UMAP1"], pt.loc[mask, "UMAP2"],
                   c=color, s=0.6, linewidths=0, rasterized=True,
                   alpha=0.8, label=label)
    ax.set_title("Disease group", fontsize=5.0, pad=2, fontweight="semibold")
    ax.axis("off")
    ax.legend(fontsize=3.8, frameon=False, loc="lower left",
              markerscale=2.5, handletextpad=0.3, labelspacing=0.3)

    # A4 — sample
    ax = fig.add_subplot(sub[3])
    for sid in SAMPLE_ORDER:
        mask = pt["sample_id"] == sid
        ax.scatter(pt.loc[mask, "UMAP1"], pt.loc[mask, "UMAP2"],
                   c=SAMPLE_COLORS[sid], s=0.6, linewidths=0,
                   rasterized=True, alpha=0.8, label=SAMPLE_LABELS[sid])
    ax.set_title("Sample", fontsize=5.0, pad=2, fontweight="semibold")
    ax.axis("off")
    ax.legend(fontsize=3.2, frameon=False, loc="lower left",
              markerscale=2.5, handletextpad=0.3, labelspacing=0.25)


# ── Panel B — Spatial pseudotime maps (2×2) ───────────────────────────────────
def draw_panel_B(gs):
    sub = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs, hspace=0.22, wspace=0.08)
    fig = plt.gcf()
    rng = np.random.default_rng(99)

    for i, sid in enumerate(SAMPLE_ORDER):
        r, c = divmod(i, 2)
        ax = fig.add_subplot(sub[r, c])

        meta_path = DATA_XE / "spatialcellchat" / f"{sid}_cell_meta.csv"
        if not meta_path.exists():
            ax.text(0.5, 0.5, f"{sid}\n(file missing)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=4)
            ax.axis("off")
            continue

        meta = pd.read_csv(meta_path, usecols=["x", "y", "Tier_A",
                                                "xenium_pseudotime_norm"])

        non_d = meta[meta["Tier_A"] != "pancreatic ductal epithelium"]
        ductal = meta[
            (meta["Tier_A"] == "pancreatic ductal epithelium") &
            meta["xenium_pseudotime_norm"].notna()
        ]

        # Subsample non-ductal for speed (keep all ductal)
        if len(non_d) > 30_000:
            non_d = non_d.sample(30_000, random_state=42)

        ax.scatter(non_d["x"], non_d["y"],
                   s=0.03, c="#d8d8d8", linewidths=0,
                   rasterized=True, zorder=1)
        sc = ax.scatter(ductal["x"], ductal["y"],
                        c=ductal["xenium_pseudotime_norm"],
                        cmap="viridis", s=0.3, linewidths=0,
                        vmin=0, vmax=1, rasterized=True, zorder=2)

        cb = fig.colorbar(sc, ax=ax, fraction=0.028, pad=0.01)
        cb.ax.tick_params(labelsize=3.0)
        if c == 1:
            cb.set_label("PT", fontsize=3.5, labelpad=1)
        else:
            cb.remove()

        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(SAMPLE_LABELS[sid], fontsize=5.0, pad=2,
                     fontweight="semibold")


# ── Panel C — Branch composition stacked bar ─────────────────────────────────
def draw_panel_C(gs, pt):
    sub = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs, width_ratios=[2, 1], wspace=0.35)
    fig = plt.gcf()

    # Left: stacked bar
    ax = fig.add_subplot(sub[0])

    branch_order = [b for b in XE_BRANCHES if b != "other"]
    branch_order_with_other = branch_order + ["other"]

    counts = (pt.groupby(["sample_id", "major_branch"])
                .size()
                .reset_index(name="n"))
    totals = counts.groupby("sample_id")["n"].sum()
    counts["frac"] = counts.apply(
        lambda r: r["n"] / totals[r["sample_id"]], axis=1)

    bar_data = {}
    for b in branch_order_with_other:
        bar_data[b] = []
        for sid in SAMPLE_ORDER:
            sub_row = counts[(counts["sample_id"] == sid) &
                             (counts["major_branch"] == b)]
            bar_data[b].append(sub_row["frac"].values[0]
                               if len(sub_row) > 0 else 0.0)

    x = np.arange(len(SAMPLE_ORDER))
    bottoms = np.zeros(len(SAMPLE_ORDER))
    for b in branch_order_with_other:
        vals = np.array(bar_data[b])
        ax.bar(x, vals, bottom=bottoms, color=BRANCH_PAL.get(b, "#ccc"),
               width=0.65, edgecolor="white", linewidth=0.4,
               label=b.replace("branch ", "Branch ").replace("trunk", "Trunk"))
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels([SAMPLE_LABELS[s] for s in SAMPLE_ORDER],
                       fontsize=4.2, rotation=25, ha="right")
    ax.set_ylabel("Fraction of niches", fontsize=4.5, labelpad=2)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=4.0, length=2)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.set_title("Branch composition per sample", fontsize=5.0, pad=3,
                 fontweight="semibold", loc="left")

    # Legend on right (empty axis)
    ax_leg = fig.add_subplot(sub[1])
    ax_leg.axis("off")
    handles = [
        mpatches.Patch(facecolor=BRANCH_PAL.get(b, "#ccc"),
                       label=b.replace("branch ", "Branch ").replace("trunk", "Trunk"))
        for b in branch_order_with_other
    ]
    ax_leg.legend(handles=handles, fontsize=4.0, frameon=False,
                  loc="center left", labelspacing=0.40,
                  handlelength=1.0, handletextpad=0.4)


# ── Panels D & E — Score LOWESS rows ─────────────────────────────────────────
def draw_score_row(gs, pt, score_list, row_label, branches_present):
    sub = gridspec.GridSpecFromSubplotSpec(
        1, len(score_list), subplot_spec=gs, wspace=0.45)
    fig = plt.gcf()
    rng = np.random.default_rng(7)

    for i, (col, label, note) in enumerate(score_list):
        ax = fig.add_subplot(sub[i])
        _score_lowess_panel(ax, pt, col, label, note, branches_present, rng)
        if i == 0:
            ax.set_ylabel(f"{row_label}\n(z-score)", fontsize=4.5, labelpad=2)


# ── Main figure ───────────────────────────────────────────────────────────────
def make_figure():
    with open(PT_PATH, "rb") as f:
        pt = pickle.load(f)

    branches_present = [b for b in XE_BRANCHES
                        if b in pt["major_branch"].unique()]

    fig = plt.figure(figsize=(170 * MM2IN, 310 * MM2IN), facecolor="white")

    # Outer layout: 5 rows
    # A (UMAP)   — short
    # B (maps)   — tall
    # C (bar)    — short-medium
    # D (scores) — medium
    # E (histol) — medium
    outer = gridspec.GridSpec(
        5, 1, figure=fig,
        left=0.08, right=0.97,
        top=0.97, bottom=0.04,
        hspace=0.58,
        height_ratios=[1.0, 2.0, 0.80, 1.05, 1.05],
    )

    draw_panel_A(outer[0], pt)
    draw_panel_B(outer[1])
    draw_panel_C(outer[2], pt)
    draw_score_row(outer[3], pt, PANEL_D_SCORES,
                   "Xenium module score", branches_present)
    draw_score_row(outer[4], pt, PANEL_E_SCORES,
                   "Histology score", branches_present)

    # ── Panel labels ──────────────────────────────────────────────────────────
    label_kw = dict(fontsize=7.5, fontweight="bold",
                    transform=fig.transFigure, va="top")
    total_h = 1.0 + 2.0 + 0.80 + 1.05 + 1.05  # sum of height_ratios
    margins_h = (0.97 - 0.04)   # top - bottom
    hspace_total = 0.58 * 4     # hspace × (nrows-1) — approximate

    def row_top(row_idx):
        # Approximate normalized y coordinate of the top of each row
        above = sum([1.0, 2.0, 0.80, 1.05, 1.05][:row_idx])
        return 0.97 - margins_h * above / (total_h + hspace_total * 0.5)

    fig.text(0.08, row_top(0), "A", **label_kw)
    fig.text(0.08, row_top(1) - 0.01, "B", **label_kw)
    fig.text(0.08, row_top(2) - 0.01, "C", **label_kw)
    fig.text(0.08, row_top(3) - 0.01, "D", **label_kw)
    fig.text(0.08, row_top(4) - 0.01, "E", **label_kw)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_pdf = OUT_DIR / "suppfig_xenium_trajectory.pdf"
    out_png = OUT_DIR / "suppfig_xenium_trajectory.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved:\n  {out_pdf}\n  {out_png}")
    plt.show()


if __name__ == "__main__":
    make_figure()
