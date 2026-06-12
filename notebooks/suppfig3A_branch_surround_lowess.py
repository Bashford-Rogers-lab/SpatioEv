"""
Supplementary Figure 3A — Branch-composition Muller plot + surround LOWESS
===========================================================================
Two-part layout per figure:

  TOP (shared):  Muller/stream plot — stacked area showing the proportion of
                 cells from each branch at every pseudotime bin.  This traces
                 the tree topology: trunk fills the left, branches diverge
                 rightward.  (Same tree for all 4 features — shown once.)

  BOTTOM:  1 × 4 LOWESS panels (one per surround feature).
           Each branch's LOWESS is clipped to its actual pseudotime range
           (5th–95th percentile) and shaded relative to the overall mean,
           so diverging trends are immediately visible.

Width: 170 mm  Height: 85 mm

Run:
    python notebooks/suppfig3A_branch_surround_lowess.py

Output: notebooks/results/suppfig3/suppfig3A_branch_surround_lowess.pdf (.png)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.gridspec as mgridspec
import statsmodels.api as sm
from scipy.ndimage import uniform_filter1d
from scipy.stats import spearmanr

from fig2_shared_config import (
    CACHE_DIR, MM2IN, set_pub_rc,
    BRANCH_CONTEXT_TREND_FEATURES, BRANCH_CONTEXT_LABELS,
    make_branch_palette, assign_branch_bio_names, MODULE_COLS,
)

set_pub_rc()

OUT_SUPPFIG3 = Path("/Users/shihongwu/SpatioEv/notebooks/results/suppfig3")
N_TOP_BRANCHES = 6
N_BINS         = 80    # pseudotime bins for Muller plot


FIBRO_FEAT       = "surround_prop__Fibroblasts"
FIBRO_BRANCH     = "branch 12"          # PanIN-like branch with sample heterogeneity
SAMPLE_COLORS    = ["#e07b3a", "#5b8db8", "#6aab6a", "#b06ab0"]  # 4 distinct samples


def load_data():
    with open(CACHE_DIR / "pooled_niche_result_df.pkl", "rb") as f:
        df = pickle.load(f)
    with open(CACHE_DIR / "pooled_pathology_feature_df.pkl", "rb") as f:
        pf = pickle.load(f)
    niche_key = "pancreatic ductal epithelium_mask_component"
    avail_ctx = [c for c in BRANCH_CONTEXT_TREND_FEATURES if c in pf.columns]
    # Include sample_id in merged df for per-sample stratification
    sid_col = next((c for c in df.columns if "sample" in c.lower()), None)
    merge_cols = [niche_key, "image_id"] + avail_ctx
    merged = df.merge(
        pf[merge_cols].drop_duplicates(subset=[niche_key, "image_id"]),
        on=[niche_key, "image_id"], how="left",
    )
    # Robust percentile scaling (2nd–98th) so one outlier branch doesn't
    # compress all others into a narrow region of the x-axis.
    mask = merged["pooled_pseudotime"].notna()
    pt_raw = merged.loc[mask, "pooled_pseudotime"]
    pt_lo  = np.percentile(pt_raw, 2)
    pt_hi  = np.percentile(pt_raw, 98)
    scaled = (pt_raw - pt_lo) / max(pt_hi - pt_lo, 1e-9)
    merged.loc[mask, "pseudotime_scaled"] = scaled.clip(0, 1)
    # Carry sample_id through
    merged["_sid_col"] = sid_col
    return merged, avail_ctx, sid_col


def _lowess(x, y, frac=0.65):
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 30:
        return None, None
    o = np.argsort(x[valid])
    r = sm.nonparametric.lowess(y[valid][o], x[valid][o], frac=frac, return_sorted=True)
    return r[:, 0], r[:, 1]


def make_figure():
    df, avail_ctx, sid_col = load_data()

    # ── Branch ordering ───────────────────────────────────────────────────────
    bc = df["major_branch"].value_counts()
    excl = {"unassigned"}
    has_trunk = "trunk" in bc.index
    non_trunk = [b for b in bc.index if b not in excl | {"trunk"}][:N_TOP_BRANCHES]
    branch_order = (["trunk"] if has_trunk else []) + non_trunk

    branch_palette = make_branch_palette(branch_order)
    if has_trunk:
        branch_palette["trunk"] = "#444444"

    avail_mod = [c for c in MODULE_COLS if c in df.columns]
    bio_names = assign_branch_bio_names(df, avail_mod)

    pt_col = "pseudotime_scaled"

    # ── Branching point (90th pct of trunk) ───────────────────────────────────
    trunk_pt = df.loc[df["major_branch"] == "trunk", pt_col].dropna()
    branch_point = float(np.percentile(trunk_pt, 90)) if len(trunk_pt) > 0 else 0.35

    # ── Build Muller data ─────────────────────────────────────────────────────
    bins        = np.linspace(0, 1, N_BINS + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    raw_counts = {}
    for b in branch_order:
        vals = df.loc[df["major_branch"] == b, pt_col].dropna().values
        hist, _ = np.histogram(vals, bins=bins)
        raw_counts[b] = uniform_filter1d(hist.astype(float), size=6)

    total = np.maximum(sum(raw_counts.values()), 1e-9)
    proportions = {b: raw_counts[b] / total for b in branch_order}

    # Re-order for Muller plot: trunk at bottom, then branches sorted by their
    # mean pseudotime (early-diverging branches closest to trunk visually).
    non_trunk_sorted = sorted(
        [b for b in branch_order if b != "trunk"],
        key=lambda b: df.loc[df["major_branch"] == b, pt_col].dropna().mean()
    )
    muller_order = (["trunk"] if has_trunk else []) + non_trunk_sorted

    # ── Layout ────────────────────────────────────────────────────────────────
    n_cols = len(avail_ctx)
    fig_w  = 170 * MM2IN
    fig_h  = 90  * MM2IN    # +5 mm for ruler strip

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs_main = mgridspec.GridSpec(
        2, 1, height_ratios=[1, 3.0],
        left=0.08, right=0.98, top=0.93, bottom=0.14,
        hspace=0.48,
    )

    # ── TOP: Muller stream plot ───────────────────────────────────────────────
    ax_m = fig.add_subplot(gs_main[0, 0])

    bottom = np.zeros(N_BINS)
    for b in muller_order:
        color = branch_palette.get(b, "#888")
        prop  = proportions[b]
        ax_m.fill_between(bin_centers, bottom, bottom + prop,
                          color=color, alpha=0.88, linewidth=0, zorder=2)
        ax_m.plot(bin_centers, bottom + prop,
                  color="white", lw=0.25, alpha=0.6, zorder=3)
        bottom += prop

    # Branch-point annotation: arrow below the plot to avoid title collision
    ax_m.axvline(branch_point, color="#888888", lw=0.8, ls="--", zorder=4)
    ax_m.annotate(
        "branch\npoint",
        xy=(branch_point, 0), xycoords="data",
        xytext=(branch_point + 0.04, -0.28), textcoords="data",
        fontsize=3.8, color="#888888", va="top", ha="left",
        arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.5),
    )

    ax_m.set_xlim(0, 1)
    ax_m.set_ylim(0, 1)
    ax_m.set_yticks([0, 0.5, 1])
    ax_m.set_yticklabels(["0", "0.5", "1"], fontsize=4.5)
    ax_m.set_xticklabels([])
    ax_m.tick_params(length=2, pad=1.5)
    ax_m.set_ylabel("Branch\nproportion", fontsize=5.0, labelpad=2)
    ax_m.set_title("Branch composition along pseudotime",
                   fontsize=5.5, pad=2, loc="left")
    for sp in ["top", "right"]:
        ax_m.spines[sp].set_visible(False)

    # ── Pre-compute per-branch pseudotime bounds (global scale) ─────────────
    # Used for (1) branch-normalised x, (2) the pseudotime-position ruler strip.
    branch_pt_bounds = {}
    for b in branch_order:
        vals = df.loc[df["major_branch"] == b, pt_col].dropna().values
        if len(vals) >= 30:
            branch_pt_bounds[b] = (np.percentile(vals, 10), np.percentile(vals, 90))

    # ── BOTTOM: 4 feature LOWESS panels (branch-normalised pseudotime) ───────
    # Each branch's pseudotime is normalised to [0,1] within its own p10–p90
    # range.  This prevents parallel branches from crowding the same x-position
    # while still showing within-branch temporal trends.
    gs_bot = mgridspec.GridSpecFromSubplotSpec(
        2, n_cols,
        subplot_spec=gs_main[1, 0],
        wspace=0.28, hspace=0.0,
        height_ratios=[0.12, 1.0],   # top strip = pseudotime ruler
    )

    # ── Pseudotime-position ruler (one per feature column, but shared info) ──
    for col_idx in range(n_cols):
        ax_ruler = fig.add_subplot(gs_bot[0, col_idx])
        for b in branch_order:
            if b not in branch_pt_bounds:
                continue
            lo, hi = branch_pt_bounds[b]
            color = branch_palette.get(b, "#888")
            lw = 2.5 if b == "trunk" else 1.6
            ax_ruler.plot([lo, hi], [0, 0], color=color, lw=lw,
                          solid_capstyle="butt", alpha=0.85)
        ax_ruler.axvline(branch_point, color="#aaaaaa", lw=0.7, ls="--", zorder=4)
        ax_ruler.set_xlim(0, 1)
        ax_ruler.set_ylim(-0.5, 0.5)
        ax_ruler.axis("off")
        if col_idx == 0:
            ax_ruler.text(-0.01, 0, "Global\npseudotime", transform=ax_ruler.transAxes,
                          fontsize=3.5, ha="right", va="center", color="#888888")

    for col_idx, feat in enumerate(avail_ctx):
        ax = fig.add_subplot(gs_bot[1, col_idx])

        feat_vals = df[feat].dropna()
        overall_mean = feat_vals.mean() if len(feat_vals) else 0

        # Mean reference
        ax.axhline(overall_mean, color="#cccccc", lw=0.7, ls="--", zorder=1)
        # x=0 is each branch's own start; x=1 is each branch's own end
        ax.axvline(0.0, color="#dddddd", lw=0.6, ls=":", zorder=1)

        rho_annotations = []
        all_lowess_y = []   # collect LOWESS y-values to set tight y-axis

        for branch in branch_order:
            if branch not in branch_pt_bounds:
                continue
            pt_lo_b, pt_hi_b = branch_pt_bounds[branch]
            sub = df[df["major_branch"] == branch][[pt_col, feat]].dropna()
            if len(sub) < 60:
                continue
            x_raw = sub[pt_col].to_numpy()
            y     = sub[feat].to_numpy()
            color    = branch_palette.get(branch, "#888")
            is_trunk = (branch == "trunk")

            # Normalise pseudotime to [0,1] within this branch's p10–p90
            span = max(pt_hi_b - pt_lo_b, 1e-9)
            x_norm = ((x_raw - pt_lo_b) / span).clip(0, 1)

            # Restrict to inner 80% of normalised range to avoid edge artefacts
            mask = (x_norm >= 0.10) & (x_norm <= 0.90)
            xc, yc = x_norm[mask], y[mask]
            if len(xc) < 60:
                continue

            # Faint scatter (light, to give density cue without crowding)
            rng  = np.random.default_rng(abs(hash(branch)) % (2 ** 31))
            sidx = rng.choice(len(xc), size=min(120, len(xc)), replace=False)
            ax.scatter(xc[sidx], yc[sidx], s=0.3, color=color,
                       alpha=0.06, linewidths=0, rasterized=True, zorder=2)

            # LOWESS on normalised x; clip display to inner 80% to avoid edge artefacts
            lx, ly = _lowess(xc, yc)
            if lx is not None:
                keep = (lx >= 0.10) & (lx <= 0.90)
                lx_plot, ly_plot = lx[keep], ly[keep]
                if len(lx_plot) < 3:
                    lx_plot, ly_plot = lx, ly
                lw = 1.8 if is_trunk else 1.0
                zo = 6 if is_trunk else 4
                ax.plot(lx_plot, ly_plot, color=color, lw=lw, alpha=0.92,
                        solid_capstyle="round", zorder=zo)
                all_lowess_y.extend(ly_plot.tolist())

            # Spearman ρ on raw (global) pseudotime for honest reporting
            rho, p = spearmanr(x_raw, y)
            sig   = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            short = "Trunk" if is_trunk else bio_names.get(branch, branch)
            rho_annotations.append((rho, sig, color, short, is_trunk))

        # ── y-axis from LOWESS range (not scatter outliers) ──────────────────
        if all_lowess_y:
            ylo_l = min(all_lowess_y)
            yhi_l = max(all_lowess_y)
            pad   = max((yhi_l - ylo_l) * 0.25, 0.01)
            ax.set_ylim(max(0, ylo_l - pad), yhi_l + pad)
        else:
            ax.autoscale(axis="y", tight=False)

        # ρ annotations: trunk (bold) first, then branches by |ρ|
        trunk_ann   = [a for a in rho_annotations if a[4]]
        branch_anns = sorted([a for a in rho_annotations if not a[4]],
                              key=lambda t: -abs(t[0]))
        for k, (rho, sig, color, short, it) in enumerate((trunk_ann + branch_anns)[:N_TOP_BRANCHES + 1]):
            ax.text(0.97, 0.03 + k * 0.085,
                    f"{short} ρ={rho:+.2f}{sig}",
                    transform=ax.transAxes, fontsize=3.5,
                    ha="right", va="bottom", color=color,
                    fontweight="bold" if it else "normal")

        ax.set_xlim(0.08, 0.92)
        ax.tick_params(length=2, pad=1.5, labelsize=4.8)
        ax.set_xlabel("Branch-relative pseudotime", fontsize=5.0, labelpad=1)
        ax.set_title(BRANCH_CONTEXT_LABELS.get(feat, feat), fontsize=5.2, pad=2)
        if col_idx == 0:
            ax.set_ylabel("Surround proportion", fontsize=5.2, labelpad=2)
        else:
            ax.set_yticklabels([])
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

    # ── Legend (line handles) ─────────────────────────────────────────────────
    handles = []
    for b in branch_order:
        if b not in branch_palette:
            continue
        label = "Trunk" if b == "trunk" else f"{bio_names.get(b, b)}  ({b})"
        lw    = 2.2 if b == "trunk" else 1.2
        handles.append(mlines.Line2D([], [], color=branch_palette[b],
                                     lw=lw, label=label))
    fig.legend(
        handles=handles, fontsize=4.2, ncol=4,
        loc="lower center", bbox_to_anchor=(0.5, 0.0),
        frameon=False, handlelength=1.4, handleheight=0.8,
        borderpad=0.2, labelspacing=0.2, columnspacing=1.2,
    )

    OUT_SUPPFIG3.mkdir(parents=True, exist_ok=True)
    out_pdf = OUT_SUPPFIG3 / "suppfig3A_branch_surround_lowess.pdf"
    out_png = OUT_SUPPFIG3 / "suppfig3A_branch_surround_lowess.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved:\n  {out_pdf}\n  {out_png}")
    plt.show()


if __name__ == "__main__":
    make_figure()
