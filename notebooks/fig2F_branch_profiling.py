"""
Figure 2F — Branch profiling: module heatmap + branch-transition dotplot
=========================================================================
Two-panel column (90 mm wide):
  Top   : Heatmap — rows = major branches, cols = PDAC modules.
           Values = mean module z-score; right strip = disease fraction.
  Bottom: Transition dotplot — branch × time-bin transitions.
           Dot size ∝ |Δ z-score|, colour = direction (RdBu_r, red=↑).
           Data from all_branch_time_transition_feature_deltas.csv
           (filtered to dataset == "multiplexed imaging").

Width: 90 mm  Height: 130 mm

Run:
    python notebooks/fig2F_branch_profiling.py

Output: notebooks/results/fig2/fig2F_branch_profiling.pdf (.png)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from fig2_shared_config import (
    CACHE_DIR, OUT_DIR, MM2IN, set_pub_rc,
    MODULE_COLS, MODULE_LABELS,
    DISEASE_PALETTE,
    make_branch_palette, assign_branch_bio_names,
)

set_pub_rc()

TRANSITION_CSV = (
    Path(__file__).parent
    / "results/trajectory_microenvironment_interactions/tables"
    / "all_branch_time_transition_feature_deltas.csv"
)

# Top features to show in dotplot (tune as needed)
DOTPLOT_CATEGORIES = {"composition", "fibroblast status", "spatial co-localization"}
DOTPLOT_MAX_FEATURES = 12   # max y-labels
DOTPLOT_MAX_BRANCHES = 6    # top branches only
DOT_SCALE = 45              # pt² per unit abs_delta_z=1


def load_data():
    with open(CACHE_DIR / "pooled_niche_result_df.pkl", "rb") as f:
        df = pickle.load(f)
    return df


def load_transition_data(branch_order):
    """Load and filter transition delta CSV."""
    if not TRANSITION_CSV.exists():
        return None
    td = pd.read_csv(TRANSITION_CSV)
    # Keep multiplexed imaging rows only
    if "dataset" in td.columns:
        td = td[td["dataset"] == "multiplexed imaging"].copy()
    # Exclude unassigned and trunk-only
    excl_branches = {"unassigned", ""}
    if "branch" in td.columns:
        td = td[~td["branch"].isin(excl_branches)]
        # Keep only top branches from heatmap
        td = td[td["branch"].isin(set(branch_order))]
    # Filter categories
    if "category" in td.columns:
        td = td[td["category"].isin(DOTPLOT_CATEGORIES)]
    # Use delta_z for size/color; fallback to delta
    if "delta_z" not in td.columns and "delta" in td.columns:
        td["delta_z"] = td["delta"]
    if "abs_delta_z" not in td.columns:
        td["abs_delta_z"] = td["delta_z"].abs()
    return td


def shorten_transition(t):
    """'branch 23: early → mid' → 'B23: e→m'"""
    t = str(t)
    import re
    m = re.match(r"branch\s*([\w.]+):\s*(\w+)\s*(?:→|->)\s*(\w+)", t, re.IGNORECASE)
    if m:
        br, s, e = m.group(1), m.group(2)[:1].lower(), m.group(3)[:1].lower()
        return f"B{br}: {s}→{e}"
    return t[:14]


def make_figure():
    df = load_data()

    # ── Branch ordering ───────────────────────────────────────────────────────
    bc = df["major_branch"].value_counts()
    excl = {"unassigned"}
    branch_order = (
        (["trunk"] if "trunk" in bc.index else []) +
        [b for b in bc.index if b not in excl | {"trunk"}]
    )[:8]

    branch_palette = make_branch_palette(branch_order)
    avail_mod = [c for c in MODULE_COLS if c in df.columns]
    bio_names = assign_branch_bio_names(df, avail_mod)

    # ── Build heatmap matrix ──────────────────────────────────────────────────
    sub = df[df["major_branch"].isin(branch_order)].copy()
    bm = sub.groupby("major_branch", observed=True)[avail_mod].mean().reindex(branch_order)
    bz = bm.apply(lambda col: (col - col.mean()) / max(col.std(ddof=0), 1e-8), axis=0)

    # Disease fraction strip
    bd = (
        sub.groupby(["major_branch", "disease_group"], observed=True)
        .size().rename("n").reset_index()
    )
    bd["frac"] = bd["n"] / bd.groupby("major_branch", observed=True)["n"].transform("sum")
    bd_pivot = bd.pivot(index="major_branch", columns="disease_group", values="frac").reindex(branch_order).fillna(0)

    # ── Load transition data ──────────────────────────────────────────────────
    td = load_transition_data(branch_order)
    has_dotplot = td is not None and len(td) > 0

    # ── Layout ────────────────────────────────────────────────────────────────
    fig_w = 90  * MM2IN
    fig_h = 110 * MM2IN   # shorter — top panel is 2/3 original height

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    if has_dotplot:
        gs_outer = mgridspec.GridSpec(
            2, 1, height_ratios=[0.67, 1.2],
            left=0.30, right=0.97, top=0.97, bottom=0.10,
            hspace=0.50,
        )
        # Top: heatmap + disease strip + cbar — narrowed to 2/3 of available width
        top_bbox      = gs_outer[0].get_position(fig)
        top_right_2_3 = top_bbox.x0 + (top_bbox.x1 - top_bbox.x0) * (2 / 3)
        gs_left = mgridspec.GridSpec(
            1, 3,
            left=top_bbox.x0, right=top_right_2_3,
            bottom=top_bbox.y0, top=top_bbox.y1,
            width_ratios=[len(avail_mod) * 0.7, len(bd_pivot.columns) * 1.2, 0.30],
            wspace=0.14,
        )
        ax_hm   = fig.add_subplot(gs_left[0, 0])
        ax_dis  = fig.add_subplot(gs_left[0, 1])
        ax_cbar = fig.add_subplot(gs_left[0, 2])
        ax_dot  = fig.add_subplot(gs_outer[1])
    else:
        gs_left = mgridspec.GridSpec(
            1, 3,
            width_ratios=[len(avail_mod) * 0.7, len(bd_pivot.columns) * 1.2, 0.25],
            left=0.30, right=0.97, top=0.97, bottom=0.55,
            wspace=0.14,
        )
        ax_hm   = fig.add_subplot(gs_left[0, 0])
        ax_dis  = fig.add_subplot(gs_left[0, 1])
        ax_cbar = fig.add_subplot(gs_left[0, 2])
        ax_dot  = None

    # ── Heatmap ───────────────────────────────────────────────────────────────
    mat  = bz.values
    vmax = max(abs(mat[np.isfinite(mat)]).max(), 1e-3)

    im = ax_hm.imshow(mat, aspect="auto", cmap="RdBu_r",
                      vmin=-vmax, vmax=vmax, interpolation="nearest")

    row_labels = [f"{bio_names.get(b, b)}  ({b})" for b in branch_order]
    ax_hm.set_yticks(range(len(branch_order)))
    ax_hm.set_yticklabels(row_labels, fontsize=4.5)
    ax_hm.yaxis.set_tick_params(length=0, pad=2)

    col_labels = [MODULE_LABELS.get(c, c).replace("\n", " ") for c in avail_mod]
    ax_hm.set_xticks(range(len(avail_mod)))
    ax_hm.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=5.5, va="top")
    ax_hm.xaxis.set_tick_params(length=0, pad=1)

    for i in range(len(branch_order) + 1):
        ax_hm.axhline(i - 0.5, color="white", lw=0.4)
    for j in range(len(avail_mod) + 1):
        ax_hm.axvline(j - 0.5, color="white", lw=0.4)
    for sp in ax_hm.spines.values():
        sp.set_visible(False)
    ax_hm.set_title("Module z-score", fontsize=5.0, pad=3, loc="left")

    # ── Disease strip ─────────────────────────────────────────────────────────
    disease_order = [d for d in DISEASE_PALETTE if d in bd_pivot.columns]
    left = np.zeros(len(branch_order))
    for dis in disease_order:
        vals = bd_pivot.get(dis, pd.Series(0, index=branch_order)).reindex(branch_order).fillna(0).values
        ax_dis.barh(range(len(branch_order)), vals, left=left,
                    color=DISEASE_PALETTE[dis], height=0.7, linewidth=0)
        left += vals
    ax_dis.set_xlim(0, 1); ax_dis.set_ylim(-0.5, len(branch_order) - 0.5)
    ax_dis.set_yticks([]); ax_dis.invert_yaxis()
    ax_dis.set_xticks([0, 1]); ax_dis.set_xticklabels(["0", "1"], fontsize=4.2)
    ax_dis.set_xlabel("Disease\nfrac.", fontsize=4.5, labelpad=1)
    for sp in ax_dis.spines.values(): sp.set_visible(False)
    dhandles = [mpatches.Patch(facecolor=DISEASE_PALETTE[d],
                               label=d.replace("NormalPancreas", "Normal"))
                for d in disease_order]
    ax_dis.legend(handles=dhandles, fontsize=4.0, loc="upper right",
                  frameon=False, handlelength=0.7, handleheight=0.6,
                  borderpad=0.1, labelspacing=0.15)

    # ── Colorbar ─────────────────────────────────────────────────────────────
    cbar = plt.colorbar(im, cax=ax_cbar, orientation="vertical")
    cbar.set_label("z-score", fontsize=4.2, labelpad=2)
    cbar.ax.tick_params(labelsize=4.0, length=2, width=0.4)
    cbar.outline.set_linewidth(0.3)

    # ── Transition dotplot ────────────────────────────────────────────────────
    if has_dotplot and ax_dot is not None:
        # Select top features by mean abs_delta_z
        feat_importance = (
            td.groupby("label", observed=True)["abs_delta_z"].mean()
            .sort_values(ascending=False)
        )
        top_features = feat_importance.index[:DOTPLOT_MAX_FEATURES].tolist()
        top_branches = [b for b in branch_order if b in td["branch"].unique()][:DOTPLOT_MAX_BRANCHES]

        td_plot = td[td["label"].isin(top_features) & td["branch"].isin(top_branches)].copy()
        td_plot["short_transition"] = td_plot["transition"].apply(shorten_transition)

        # Build ordered x-axis: group by branch, then transitions within branch
        trans_order = []
        for b in top_branches:
            sub_t = td_plot[td_plot["branch"] == b]["short_transition"].unique().tolist()
            sub_t = sorted(sub_t)  # early→mid before mid→late
            trans_order.extend(sub_t)
        trans_order = list(dict.fromkeys(trans_order))  # dedup, preserve order

        y_order = list(reversed(top_features))  # most important at top

        x_pos = {t: i for i, t in enumerate(trans_order)}
        y_pos = {f: i for i, f in enumerate(y_order)}

        vmax_dot = max(td_plot["delta_z"].abs().max(), 1.0)
        norm_dot  = Normalize(vmin=-vmax_dot, vmax=vmax_dot)
        cmap_dot  = plt.get_cmap("RdBu_r")

        for _, row in td_plot.iterrows():
            xi = x_pos.get(row["short_transition"])
            yi = y_pos.get(row["label"])
            if xi is None or yi is None:
                continue
            s  = max(row["abs_delta_z"] * DOT_SCALE, 2.0)
            c  = cmap_dot(norm_dot(row["delta_z"]))
            ax_dot.scatter(xi, yi, s=s, c=[c], linewidths=0.3,
                           edgecolors="#555555", zorder=3)

        # Branch separators (vertical dashed lines)
        xc = 0
        for b in top_branches:
            n_t = sum(1 for t in trans_order
                      if td_plot[(td_plot["branch"] == b) & (td_plot["short_transition"] == t)].shape[0] > 0)
            xc_new = xc + n_t
            if xc > 0:
                ax_dot.axvline(xc - 0.5, color="#bbbbbb", lw=0.5, ls="--", zorder=1)
            # Branch label at midpoint
            if n_t > 0:
                label = bio_names.get(b, b)
                mid   = xc + (n_t - 1) / 2.0
                ax_dot.text(mid, len(y_order) + 0.1, label,
                            ha="center", va="bottom", fontsize=3.8,
                            color=branch_palette.get(b, "#555"),
                            rotation=15, clip_on=False)
            xc = xc_new

        ax_dot.set_xlim(-0.6, len(trans_order) - 0.4)
        ax_dot.set_ylim(-0.6, len(y_order) + 0.5)
        ax_dot.set_xticks(range(len(trans_order)))
        ax_dot.set_xticklabels(trans_order, rotation=40, ha="right",
                               fontsize=3.8, va="top")
        import textwrap
        wrapped_y = [textwrap.fill(lbl, width=22) for lbl in y_order]
        ax_dot.set_yticks(range(len(y_order)))
        ax_dot.set_yticklabels(wrapped_y, fontsize=4.2, linespacing=1.2)
        ax_dot.tick_params(length=0, pad=2)
        ax_dot.set_title("Feature changes at branch transitions\n(dot size = |Δz|, red=↑ blue=↓)",
                         fontsize=4.8, pad=3, loc="left")
        for sp in ["top", "right"]:
            ax_dot.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]:
            ax_dot.spines[sp].set_linewidth(0.4)

        # Colorbar for dotplot
        sm = ScalarMappable(norm=norm_dot, cmap=cmap_dot)
        cbar_dot = plt.colorbar(sm, ax=ax_dot, orientation="vertical",
                                fraction=0.04, pad=0.03, aspect=20)
        cbar_dot.set_label("Δ z-score", fontsize=4.0, labelpad=2)
        cbar_dot.ax.tick_params(labelsize=3.8, length=2, width=0.3)
        cbar_dot.outline.set_linewidth(0.3)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = OUT_DIR / "fig2F_branch_profiling.pdf"
    out_png = OUT_DIR / "fig2F_branch_profiling.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved:\n  {out_pdf}\n  {out_png}")
    plt.show()


if __name__ == "__main__":
    make_figure()
