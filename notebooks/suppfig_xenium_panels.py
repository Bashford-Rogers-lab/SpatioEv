"""
suppfig_xenium_panels.py
========================
Outputs one PDF + PNG per panel at 6.69″ × 8.86″ (170 × 225 mm).

  A  Cell-type annotation dotplot  (mean z-score, fraction expressing)
  B  Niche UMAP — 4 colourings
  C  Branch composition per sample (normalized stacked bar)
  D  Xenium module score LOWESS (4 scores)
  E  Histology score LOWESS (4 scores)

Note: spatial pseudotime maps (formerly Panel C) have been moved to fig3_panels.py Panel A.

Run all panels:
    python notebooks/suppfig_xenium_panels.py

Run specific panels (e.g. D and E only):
    python notebooks/suppfig_xenium_panels.py D E
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

try:
    import statsmodels.api as sm
    _HAS_STATSMODELS = True
except ModuleNotFoundError:
    _HAS_STATSMODELS = False

sys.path.insert(0, str(Path(__file__).parent))
from fig2_shared_config import MM2IN, set_pub_rc, make_branch_palette

set_pub_rc()

# ── Canvas ────────────────────────────────────────────────────────────────────
W_IN  = 6.69   # 170 mm
H_IN  = 8.86   # 225 mm

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = Path(__file__).parent.parent
DATA_XE = BASE / "data" / "xenium_pancreas_10x"
PT_PATH = DATA_XE / "pseudotime" / "xenium_pseudotime_result_df.pkl"
OUT_DIR = BASE / "notebooks" / "results" / "suppfig_xenium"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Palettes ──────────────────────────────────────────────────────────────────
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

# ── Cell-type display order & labels ─────────────────────────────────────────
CELLTYPE_ORDER = [
    "pancreatic ductal epithelium",
    "pancreatic acinar epithelium",
    "Fibroblasts",
    "Endothelial cells",
    "Islets",
    "Myeloid cells",
    "T cells",
    "Mast cells",
    "B lineage",
]
CELLTYPE_LABELS = {
    "pancreatic ductal epithelium": "Ductal epi.",
    "pancreatic acinar epithelium": "Acinar epi.",
    "Fibroblasts":                  "Fibroblasts",
    "Endothelial cells":            "Endothelial",
    "Islets":                       "Islets",
    "Myeloid cells":                "Myeloid",
    "T cells":                      "T cells",
    "Mast cells":                   "Mast cells",
    "B lineage":                    "B lineage",
}

# ── Dotplot marker genes (from 98-gene Xenium panel) ─────────────────────────
DOTPLOT_GENES = [
    # Epithelial / structural
    "EPCAM",  "EGFR",   "DMBT1", "MET",
    # Fibroblast
    "PDGFRA", "ACTA2",
    # Endothelial
    "SOX17",
    # Pan-immune
    "PTPRC",
    # Myeloid
    "AIF1",   "CD68",   "FCGR1A","CD14",  "CD163", "TREM2",
    # T / NK
    "CD3D",   "CD3E",   "CD8A",  "CD4",   "FOXP3",
    "NKG7",   "GNLY",   "KLRD1",
    # B / Plasma
    "MS4A1",  "CD79A",  "MZB1",
    # Mast
    "KIT",    "CPA3",   "GATA2",
    # Proliferation
    "MKI67",
]

# ── Score panels ──────────────────────────────────────────────────────────────
PANEL_E_SCORES = [
    "xenium_epithelial_identity_score",
    "xenium_panin_like_remodeling_score",
    "xenium_desmoplastic_context_score",
    "xenium_proliferation_score",
]
PANEL_E_LABELS = [
    "Epithelial\nidentity",
    "PanIN-like\nremodeling",
    "Desmoplastic\ncontext",
    "Proliferation",
]
PANEL_F_SCORES = [
    "histology__normal_duct_like_score",
    "histology__adm_panin_like_score",
    "histology__desmoplastic_tumor_score",
    "histology__immune_inflamed_score",
]
PANEL_F_LABELS = [
    "Normal\nduct-like",
    "ADM / PanIN",
    "Desmoplastic\ntumor",
    "Immune\ninflamed",
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _new_fig(panel_letter: str):
    fig = plt.figure(figsize=(W_IN, H_IN), facecolor="white")
    fig.text(0.015, 0.985, panel_letter,
             fontsize=9, fontweight="bold", va="top", ha="left",
             transform=fig.transFigure)
    return fig


def _save(fig, name: str):
    pdf = OUT_DIR / f"suppfig_xenium_{name}.pdf"
    png = OUT_DIR / f"suppfig_xenium_{name}.png"
    fig.savefig(pdf, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {pdf.name}")


def _lowess(x, y, frac=0.25, delta=0.005):
    """
    LOWESS via local linear regression with tricube weights.
    Matches statsmodels behaviour when it is unavailable.
    frac  : fraction of data used per local fit
    delta : minimum x-spacing between full fits (points in between are interpolated)
    """
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 20:
        return None, None
    if _HAS_STATSMODELS:
        order = np.argsort(x[valid])
        res = sm.nonparametric.lowess(
            y[valid][order], x[valid][order],
            frac=frac, delta=delta, return_sorted=True,
        )
        return res[:, 0], res[:, 1]

    # Pure-numpy LOWESS: local linear regression + tricube kernel
    order = np.argsort(x[valid])
    xv    = x[valid][order]
    yv    = y[valid][order]
    n     = len(xv)
    k     = max(int(np.ceil(frac * n)), 3)

    # Sparse evaluation grid spaced >= delta apart (then interpolate)
    eval_mask = np.zeros(n, dtype=bool)
    eval_mask[0] = True
    last = xv[0]
    for i in range(1, n):
        if xv[i] - last >= delta:
            eval_mask[i] = True
            last = xv[i]
    eval_mask[-1] = True
    eval_idx = np.where(eval_mask)[0]

    ys_eval = np.empty(len(eval_idx))
    for j, i in enumerate(eval_idx):
        xi    = xv[i]
        dists = np.abs(xv - xi)
        nn    = np.argpartition(dists, min(k, n) - 1)[:k] if k < n else np.arange(n)
        max_d = dists[nn].max()
        if max_d < 1e-12:
            ys_eval[j] = yv[nn].mean()
            continue
        u  = np.clip(dists[nn] / max_d, 0.0, 1.0)
        w  = (1.0 - u ** 3) ** 3          # tricube
        wx, wy = xv[nn], yv[nn]
        ws = w.sum()
        if ws < 1e-12:
            ys_eval[j] = wy.mean()
            continue
        xbar = np.dot(w, wx) / ws
        ybar = np.dot(w, wy) / ws
        sxx  = np.dot(w, (wx - xbar) ** 2)
        sxy  = np.dot(w, (wx - xbar) * (wy - ybar))
        beta = sxy / sxx if abs(sxx) > 1e-12 else 0.0
        ys_eval[j] = ybar + beta * (xi - xbar)

    ys = np.interp(xv, xv[eval_idx], ys_eval)
    return xv, ys


def _spearman(x, y):
    from math import erfc, sqrt
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


def _score_lowess_ax(ax, pt, col, label, branches_present, rng, ylabel=None):
    """LOWESS per branch + global ρ for one score vs pseudotime."""
    pt_v = pt["xenium_pseudotime_norm"].values.astype(float)
    y_v  = pt[col].values.astype(float)
    bc   = pt["major_branch"].values

    ax.axhline(np.nanmean(y_v), color="#cccccc", lw=0.6, ls="--", zorder=0)

    for branch in branches_present:
        if branch == "other":
            continue
        color = BRANCH_PAL.get(branch, "#888")
        mask  = (bc == branch) & np.isfinite(pt_v) & np.isfinite(y_v)
        if mask.sum() < 15:
            continue
        bx, by = pt_v[mask], y_v[mask]

        samp = rng.choice(mask.sum(), size=min(300, mask.sum()), replace=False)
        ax.scatter(bx[samp], by[samp], s=0.8, color=color, alpha=0.15,
                   linewidths=0, rasterized=True, zorder=1)

        lx, ly = _lowess(bx, by)
        if lx is not None:
            ax.plot(lx, ly, color=color, lw=1.3, alpha=0.90,
                    solid_capstyle="round", zorder=3)

    valid = np.isfinite(pt_v) & np.isfinite(y_v)
    rho, p = (0.0, 1.0)
    if valid.sum() > 30:
        rho, p = _spearman(pt_v[valid], y_v[valid])
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
        ax.text(0.97, 0.05, f"ρ = {rho:+.2f}{sig}",
                transform=ax.transAxes, fontsize=4.0,
                ha="right", va="bottom", color="#444")

    ax.set_title(label, fontsize=5.2, pad=2.5,
                 fontweight="semibold", loc="left")
    ax.set_xlim(-0.02, 1.02)
    # Y: let matplotlib autoscale (same as original trajectory figure)
    ax.set_xlabel("Pseudotime", fontsize=4.5, labelpad=1)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=4.5, labelpad=2)
    ax.tick_params(labelsize=4.0, length=2, pad=1)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    return rho


# ─────────────────────────────────────────────────────────────────────────────
# Panel A — Cell-type annotation dotplot
# ─────────────────────────────────────────────────────────────────────────────

def _load_dotplot_data():
    """Pool ≤2000 cells/type from all samples; return (genes, mean_expr, frac_pos)."""
    MAX_PER_TYPE = 2000
    all_frames = []

    for sid in SAMPLE_ORDER:
        meta_path   = DATA_XE / "spatialcellchat" / f"{sid}_cell_meta.csv"
        counts_path = DATA_XE / "spatialcellchat" / f"{sid}_counts_common_genes.csv.gz"
        if not meta_path.exists() or not counts_path.exists():
            continue

        meta = pd.read_csv(meta_path, usecols=["cell_id", "Tier_A"])
        meta = meta[meta["Tier_A"].isin(CELLTYPE_ORDER)].copy()
        meta["cell_id"] = meta["cell_id"].astype(str)

        counts = pd.read_csv(counts_path, index_col=0)
        counts.index = counts.index.astype(str)

        merged = meta.set_index("cell_id").join(counts, how="inner")

        parts = []
        for ct in merged["Tier_A"].unique():
            sub = merged[merged["Tier_A"] == ct]
            if len(sub) > MAX_PER_TYPE:
                sub = sub.sample(MAX_PER_TYPE, random_state=42)
            parts.append(sub)
        if parts:
            all_frames.append(pd.concat(parts))

    df = pd.concat(all_frames, ignore_index=False)

    genes = [g for g in DOTPLOT_GENES if g in df.columns]
    df_e  = df[["Tier_A"] + genes].copy()

    mean_expr = df_e.groupby("Tier_A")[genes].mean()
    frac_pos  = df_e.groupby("Tier_A")[genes].apply(lambda x: (x > 0).mean())

    return genes, mean_expr, frac_pos


def make_panel_A():
    print("Panel A — cell-type annotation dotplot")
    genes, mean_expr, frac_pos = _load_dotplot_data()

    # Reindex to desired order
    ct_order = [ct for ct in CELLTYPE_ORDER if ct in mean_expr.index]
    mean_expr = mean_expr.loc[ct_order, genes]
    frac_pos  = frac_pos.loc[ct_order, genes]

    n_ct  = len(ct_order)
    n_g   = len(genes)

    fig = _new_fig("A")

    # Square-cell layout: row spacing = column spacing so dots nearly fill cells.
    # This gives a compact dotplot. Place it high on the canvas so blank space
    # below gets cropped by bbox_inches="tight".
    ax_l  = 0.20;  ax_w = 0.70
    col_sp = ax_w * W_IN / max(n_g - 1, 1)   # inches per gene column
    row_sp = col_sp                            # equal spacing in both directions
    ax_h   = row_sp * (n_ct - 1) / H_IN      # axes height (figure fraction)
    ax_b   = 0.97 - ax_h                      # top of axes at ~97%, just below panel letter
    ax = fig.add_axes([ax_l, ax_b, ax_w, ax_h])

    # Dot size: 78% of cell width → dots nearly touch in tightest direction
    s_max = np.pi * (col_sp * 0.39 * 72) ** 2
    s_max = max(s_max, 12)

    vmax = np.nanpercentile(np.abs(mean_expr.values), 98)
    vmax = max(vmax, 0.5)

    # Draw dots
    for ci, ct in enumerate(ct_order):
        for gi, gene in enumerate(genes):
            z  = mean_expr.loc[ct, gene]
            fp = frac_pos.loc[ct, gene]
            size = max(fp * s_max, 0.8)
            color_val = np.clip(z / vmax, -1, 1)
            cmap = plt.cm.RdBu_r
            c = cmap((color_val + 1) / 2)
            ax.scatter(gi, ci, s=size, c=[c], linewidths=0.3,
                       edgecolors="#555", zorder=2)

    # Grid lines
    for gi in range(n_g):
        ax.axvline(gi, color="#eeeeee", lw=0.4, zorder=0)
    for ci in range(n_ct):
        ax.axhline(ci, color="#eeeeee", lw=0.4, zorder=0)

    # Gene separator lines between groups
    gene_group_ends = [3, 5, 6, 7, 13, 18, 21, 24, 25]  # after structural groups
    for gi in gene_group_ends:
        if gi < n_g:
            ax.axvline(gi - 0.5, color="#aaaaaa", lw=0.8, ls="--", zorder=1)

    ax.set_xlim(-0.5, n_g - 0.5)
    ax.set_ylim(-0.5, n_ct - 0.5)
    ax.set_xticks(range(n_g))
    ax.set_xticklabels(genes, rotation=55, ha="right", fontsize=4.2)
    ax.set_yticks(range(n_ct))
    ax.set_yticklabels([CELLTYPE_LABELS.get(ct, ct) for ct in ct_order],
                       fontsize=4.8)
    ax.set_title("Cell-type annotation markers", fontsize=6.0,
                 fontweight="semibold", pad=22, loc="left")
    for sp in ["top", "right", "bottom", "left"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)

    # Group labels above gene axis
    group_info = [
        ("Epithelial /\nstructural",  0, 6),
        ("Myeloid",                   8, 13),
        ("T / NK",                   13, 21),
        ("B / Plasma",               21, 24),
        ("Mast",                     24, 27),
        ("Prolif.",                  27, n_g),
    ]
    from matplotlib.transforms import blended_transform_factory
    # x in data coords, y just above the axes top (axes coords)
    blend = blended_transform_factory(ax.transData, ax.transAxes)
    for glabel, g0, g1 in group_info:
        if g0 >= n_g:
            continue
        g1 = min(g1, n_g)
        mid = (g0 + g1 - 1) / 2
        ax.text(mid, 1.01, glabel,
                ha="center", va="bottom", fontsize=3.6, color="#333",
                transform=blend)

    # Colorbar — right of axes, same height
    sm_ = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r,
                                 norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    sm_.set_array([])
    cax = fig.add_axes([ax_l + ax_w + 0.02, ax_b, 0.018, ax_h])
    cb = fig.colorbar(sm_, cax=cax)
    cb.set_label("Mean z-score", fontsize=4.2, labelpad=2)
    cb.ax.tick_params(labelsize=3.8)

    # Size legend — directly below x-tick labels (ax_b - tick_label_height - gap)
    tick_h = 0.05  # figure-fraction height of rotated tick labels (4.2pt font, ~55°)
    leg_l = ax_l;  leg_b = ax_b - tick_h - 0.02;  leg_w = 0.42;  leg_h = 0.07
    leg_ax = fig.add_axes([leg_l, leg_b, leg_w, leg_h])
    leg_ax.axis("off")
    fracs  = [0.25, 0.50, 0.75, 1.00]
    labels = ["25%", "50%", "75%", "100%"]
    xs     = np.linspace(0.20, 0.85, len(fracs))
    for xi, fp, lbl in zip(xs, fracs, labels):
        leg_ax.scatter(xi, 0.60, s=fp * s_max, c="gray",
                       edgecolors="#555", linewidths=0.35,
                       transform=leg_ax.transData)
        leg_ax.text(xi, 0.05, lbl, ha="center", va="top", fontsize=3.8,
                    color="#444", transform=leg_ax.transData)
    leg_ax.set_xlim(0.0, 1.0)
    leg_ax.set_ylim(0.0, 1.0)
    leg_ax.text(0.02, 0.98, "Fraction z > 0:", ha="left", va="top",
                fontsize=4.2, color="#555",
                transform=leg_ax.transAxes)

    _save(fig, "A")


# ─────────────────────────────────────────────────────────────────────────────
# Panel B — UMAP (4 colourings)
# ─────────────────────────────────────────────────────────────────────────────

def make_panel_B(pt):
    print("Panel B — UMAP")
    fig = _new_fig("B")

    # bottom=0.74 leaves room below the UMAPs for category legends.
    # right=0.91 leaves room on the right for the pseudotime colorbar.
    gs = gridspec.GridSpec(
        1, 4, figure=fig,
        left=0.03, right=0.91, top=0.97, bottom=0.74,
        wspace=0.40,
    )

    # B1 — pseudotime (colorbar added separately after all axes)
    ax = fig.add_subplot(gs[0])
    sc_pt = ax.scatter(pt["UMAP1"], pt["UMAP2"],
                       c=pt["xenium_pseudotime_norm"], cmap="viridis",
                       s=0.8, linewidths=0, rasterized=True, alpha=0.8)
    ax.set_title("Pseudotime", fontsize=5.5, pad=2.5, fontweight="semibold")
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # B2 — branch
    ax = fig.add_subplot(gs[1])
    branch_order = [b for b in XE_BRANCHES if b in pt["major_branch"].unique()]
    for branch in branch_order:
        mask = pt["major_branch"] == branch
        ax.scatter(pt.loc[mask, "UMAP1"], pt.loc[mask, "UMAP2"],
                   c=BRANCH_PAL.get(branch, "#ccc"),
                   s=0.8, linewidths=0, rasterized=True, alpha=0.8,
                   label=branch.capitalize().replace("branch ", "Branch "))
    ax.set_title("Branch", fontsize=5.5, pad=2.5, fontweight="semibold")
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.legend(fontsize=3.5, frameon=False, loc="upper left",
              bbox_to_anchor=(0, -0.03), bbox_transform=ax.transAxes,
              markerscale=2.5, handletextpad=0.3, labelspacing=0.2,
              borderaxespad=0, ncol=2)

    # B3 — disease group
    ax = fig.add_subplot(gs[2])
    for dg, color in DISEASE_COLORS.items():
        mask  = pt["disease_group"] == dg
        label = "Normal" if dg == "NormalPancreas" else "PDAC"
        ax.scatter(pt.loc[mask, "UMAP1"], pt.loc[mask, "UMAP2"],
                   c=color, s=0.8, linewidths=0, rasterized=True,
                   alpha=0.8, label=label)
    ax.set_title("Disease group", fontsize=5.5, pad=2.5, fontweight="semibold")
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.legend(fontsize=3.5, frameon=False, loc="upper left",
              bbox_to_anchor=(0, -0.03), bbox_transform=ax.transAxes,
              markerscale=2.5, handletextpad=0.3, labelspacing=0.2,
              borderaxespad=0, ncol=2)

    # B4 — sample
    ax = fig.add_subplot(gs[3])
    for sid in SAMPLE_ORDER:
        mask = pt["sample_id"] == sid
        ax.scatter(pt.loc[mask, "UMAP1"], pt.loc[mask, "UMAP2"],
                   c=SAMPLE_COLORS[sid], s=0.8, linewidths=0,
                   rasterized=True, alpha=0.8, label=SAMPLE_LABELS[sid])
    ax.set_title("Sample", fontsize=5.5, pad=2.5, fontweight="semibold")
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.legend(fontsize=3.5, frameon=False, loc="upper left",
              bbox_to_anchor=(0, -0.03), bbox_transform=ax.transAxes,
              markerscale=2.5, handletextpad=0.3, labelspacing=0.2,
              borderaxespad=0, ncol=2)

    # Pseudotime colorbar — standalone axes to the right of all 4 UMAPs
    cax = fig.add_axes([0.925, 0.78, 0.010, 0.16])
    cb = fig.colorbar(sc_pt, cax=cax)
    cb.set_label("Pseudotime", fontsize=3.8, labelpad=2)
    cb.ax.tick_params(labelsize=3.2)

    _save(fig, "B")


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Panel C — Branch composition stacked bar
# (formerly Panel D; Panel C spatial maps moved to fig3_panels.py)
# ─────────────────────────────────────────────────────────────────────────────

def make_panel_C(pt):
    print("Panel C — branch composition")
    fig = _new_fig("C")

    gs = gridspec.GridSpec(
        1, 2, figure=fig,
        left=0.10, right=0.65, top=0.82, bottom=0.67,
        width_ratios=[2.2, 0.8], wspace=0.30,
    )

    branch_order = [b for b in XE_BRANCHES if b != "other"] + ["other"]

    counts = (pt.groupby(["sample_id", "major_branch"])
               .size()
               .reset_index(name="n"))
    totals = counts.groupby("sample_id")["n"].sum()
    counts["frac"] = counts.apply(
        lambda r: r["n"] / totals[r["sample_id"]], axis=1)

    ax = fig.add_subplot(gs[0])
    x       = np.arange(len(SAMPLE_ORDER))
    bottoms = np.zeros(len(SAMPLE_ORDER))

    for b in branch_order:
        vals = np.array([
            counts.loc[(counts["sample_id"] == sid) & (counts["major_branch"] == b),
                       "frac"].values[0]
            if len(counts[(counts["sample_id"] == sid) & (counts["major_branch"] == b)]) > 0
            else 0.0
            for sid in SAMPLE_ORDER
        ])
        ax.bar(x, vals, bottom=bottoms,
               color=BRANCH_PAL.get(b, "#ccc"),
               width=0.65, edgecolor="white", linewidth=0.4)
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels([SAMPLE_LABELS[s] for s in SAMPLE_ORDER],
                       fontsize=4.5, rotation=25, ha="right")
    ax.set_ylabel("Fraction of niches", fontsize=5.0, labelpad=2)
    ax.set_ylim(0, 1)
    ax.set_title("Branch composition per sample",
                 fontsize=5.5, pad=3, fontweight="semibold", loc="left")
    ax.tick_params(labelsize=4.2, length=2)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    # Legend
    ax_leg = fig.add_subplot(gs[1])
    ax_leg.axis("off")
    handles = [
        mpatches.Patch(
            facecolor=BRANCH_PAL.get(b, "#ccc"),
            label=b.capitalize().replace("branch ", "Branch "))
        for b in branch_order
    ]
    ax_leg.legend(handles=handles, fontsize=4.2, frameon=False,
                  loc="center left", labelspacing=0.45,
                  handlelength=1.0, handletextpad=0.4)

    _save(fig, "C")


# ─────────────────────────────────────────────────────────────────────────────
# Panel D — Xenium module score LOWESS
# (formerly Panel E)
# ─────────────────────────────────────────────────────────────────────────────

def make_panel_D(pt):
    print("Panel D — Xenium module scores")
    branches_present = [b for b in XE_BRANCHES
                        if b in pt["major_branch"].unique()]
    rng = np.random.default_rng(7)

    fig = _new_fig("D")
    gs  = gridspec.GridSpec(
        1, 4, figure=fig,
        left=0.10, right=0.535, top=0.93, bottom=0.85,
        wspace=0.45,
    )
    for i, (col, label) in enumerate(zip(PANEL_E_SCORES, PANEL_E_LABELS)):
        ax = fig.add_subplot(gs[i])
        ylabel = "Module score (z)" if i == 0 else None
        _score_lowess_ax(ax, pt, col, label, branches_present, rng, ylabel=ylabel)

    # Branch colour legend — centred below the (now narrower) gridspec
    handles = [
        mpatches.Patch(facecolor=BRANCH_PAL[b],
                       label=b.capitalize().replace("branch ", "Branch "))
        for b in branches_present if b != "other"
    ]
    fig.legend(handles=handles,
               fontsize=3.8, frameon=False,
               loc="upper center", ncol=3,
               bbox_to_anchor=(0.32, 0.82),
               handlelength=0.9, handletextpad=0.3, columnspacing=0.8)

    _save(fig, "D")


# ─────────────────────────────────────────────────────────────────────────────
# Panel E — Histology score LOWESS
# (formerly Panel F)
# ─────────────────────────────────────────────────────────────────────────────

def make_panel_E(pt):
    print("Panel E — Histology scores")
    branches_present = [b for b in XE_BRANCHES
                        if b in pt["major_branch"].unique()]
    rng = np.random.default_rng(7)

    fig = _new_fig("E")
    gs  = gridspec.GridSpec(
        1, 4, figure=fig,
        left=0.10, right=0.535, top=0.93, bottom=0.85,
        wspace=0.45,
    )
    for i, (col, label) in enumerate(zip(PANEL_F_SCORES, PANEL_F_LABELS)):
        ax = fig.add_subplot(gs[i])
        ylabel = "Histology score (z)" if i == 0 else None
        _score_lowess_ax(ax, pt, col, label, branches_present, rng, ylabel=ylabel)

    handles = [
        mpatches.Patch(facecolor=BRANCH_PAL[b],
                       label=b.capitalize().replace("branch ", "Branch "))
        for b in branches_present if b != "other"
    ]
    fig.legend(handles=handles,
               fontsize=3.8, frameon=False,
               loc="upper center", ncol=3,
               bbox_to_anchor=(0.32, 0.82),
               handlelength=0.9, handletextpad=0.3, columnspacing=0.8)

    _save(fig, "E")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    requested = set(sys.argv[1:]) if len(sys.argv) > 1 else set("ABCDE")
    requested = {p.upper() for p in requested}

    # Load pseudotime data once (needed by B, C, D, E)
    pt = None
    if requested & {"B", "C", "D", "E"}:
        print("Loading pseudotime data …")
        with open(PT_PATH, "rb") as f:
            pt = pickle.load(f)

    if "A" in requested:
        make_panel_A()
    if "B" in requested:
        make_panel_B(pt)
    if "C" in requested:
        make_panel_C(pt)
    if "D" in requested:
        make_panel_D(pt)
    if "E" in requested:
        make_panel_E(pt)

    print(f"\nAll outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
