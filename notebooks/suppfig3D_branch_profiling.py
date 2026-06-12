"""
Supplementary Figure 3D — Branch microenvironment profiling
============================================================
Per-branch characterisation in suppfig2C/2D dot+line style:

  A (top row, 3 panels):    Morphological scores — pseudotime-binned mean per branch.
                             One panel per metric; one coloured line per branch.

  B (middle row, 4 panels): Fibroblast activation markers — same style.

  C (bottom, faceted):      T/B cell subtype composition near each branch
                             (suppfig2C/2D style: one panel per subtype,
                             x = branches in pseudotime order, y = fraction).

Width: 170 mm   Height: ~165 mm

Run (requires spatioev_env for scipy):
    python notebooks/suppfig3D_branch_profiling.py

Output: notebooks/results/suppfig3/suppfig3D_branch_profiling.pdf (.png)
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
from scipy.stats import spearmanr

from fig2_shared_config import (
    CACHE_DIR, MM2IN, set_pub_rc,
    make_branch_palette, assign_branch_bio_names, MODULE_COLS, MODULE_LABELS,
)

set_pub_rc()

OUT_SUPPFIG3 = Path("/Users/shihongwu/SpatioEv/notebooks/results/suppfig3")

BRANCH_ORDER = ["trunk", "branch 12", "branch 20", "branch 15",
                "branch 17", "branch 14.b", "branch 23.a"]

# Use the 6 canonical MODULE_COLS — consistent with fig2 and suppfig3C
# Short panel titles (no newlines needed at this font size)
MORPH_COLS = MODULE_COLS   # imported from fig2_shared_config
MORPH_LABELS_SHORT = {
    "pdac_early_duct_anchor_score":      "Early-duct anchor",
    "pdac_panin_like_dysplasia_score":   "PanIN-like dysplasia",
    "pdac_invasive_gland_forming_score": "Invasive gland-forming",
    "pdac_invasion_desmoplasia_axis":    "Invasion–desmoplasia",
    "pdac_proliferation_axis":           "Proliferation",
    "pdac_dedifferentiation_axis":       "Dedifferentiation",
}

FIB_COLS = [
    "surround__Fibroblasts__FAP_expr_z__mean",
    "surround__Fibroblasts__aSMA_expr_z__mean",
    "surround__Fibroblasts__PDPN_expr_z__mean",
    "surround__Fibroblasts__Thy1_expr_z__mean",
]
FIB_LABELS = {
    "surround__Fibroblasts__FAP_expr_z__mean":  "FAP",
    "surround__Fibroblasts__aSMA_expr_z__mean": "αSMA",
    "surround__Fibroblasts__PDPN_expr_z__mean": "PDPN",
    "surround__Fibroblasts__Thy1_expr_z__mean": "Thy1",
}

TCELL_SUBTYPES_KEEP = [
    "CD8 T cells", "activated CD8 T cells", "cytotoxic CD8 T cells",
    "CD4 T cells", "activated CD4 T cells", "Tregs", "Th2-like cells",
    "Tfh-like cells",
]
BCELL_SUBTYPES_KEEP = [
    "plasmablasts-like", "naive B cells", "memory B cells",
    "APC-like B cells", "GZMB+ B cells", "PDL1+ B cells",
]
SUBTYPE_PALETTE = {
    "CD8 T cells":             "#1565C0",
    "activated CD8 T cells":   "#42A5F5",
    "cytotoxic CD8 T cells":   "#0D47A1",
    "CD4 T cells":             "#E65100",
    "activated CD4 T cells":   "#FF8F00",
    "Tregs":                   "#6A1B9A",
    "Th2-like cells":          "#CE93D8",
    "Tfh-like cells":          "#AB47BC",
    "plasmablasts-like":       "#2E7D32",
    "naive B cells":           "#66BB6A",
    "memory B cells":          "#1B5E20",
    "APC-like B cells":        "#A5D6A7",
    "GZMB+ B cells":           "#004D40",
    "PDL1+ B cells":           "#80CBC4",
    "Other":                   "#BDBDBD",
}

EXP_MAP  = {"35559_1": "exp_4", "34434_1": "exp_2",
             "33694_1": "exp_3", "40331_1": "exp_5"}
DATA_DIR = Path("/Users/shihongwu/SpatioEv/data/combined_exp_2_3_4_5")
EXP_BASE = Path("/Users/shihongwu/SpatioEv/data")
RADIUS_UM = 100.0
NK = "pancreatic ductal epithelium_mask_component"
N_BINS = 6   # pseudotime bins for panels A+B (fewer = smoother)
MIN_NICHE_PER_BIN = 15  # skip bin if fewer niches than this


# ── Data loading ────────────────────────────────────────────────────────────────

def load_niche_data():
    """Niche result (has MORPH_COLS + pseudotime) + FIB_COLS from pf."""
    with open(DATA_DIR / "pooled_niche_result_df.pkl", "rb") as f:
        nr = pickle.load(f)
    with open(DATA_DIR / "pooled_pathology_feature_df.pkl", "rb") as f:
        pf = pickle.load(f)
    extra = [c for c in FIB_COLS if c in pf.columns and c not in nr.columns]
    merged = nr.merge(pf[[NK, "image_id"] + extra], on=[NK, "image_id"], how="left")
    return merged


def bin_by_pseudotime(merged, pt_col, metric_col, branch_present,
                      n_bins=N_BINS, min_count=MIN_NICHE_PER_BIN):
    """Compute per-branch mean ± SE in pseudotime bins. Returns dict branch→(x, y, se)."""
    edges = np.linspace(0, 1, n_bins + 1)
    mids  = 0.5 * (edges[:-1] + edges[1:])
    sub   = merged[[pt_col, metric_col, "major_branch"]].dropna()

    # Clip outliers (1–99th percentile) to keep y-axis clean
    lo, hi = np.nanpercentile(sub[metric_col], [1, 99])
    sub = sub[(sub[metric_col] >= lo) & (sub[metric_col] <= hi)].copy()
    sub["bin"] = pd.cut(sub[pt_col], bins=edges, labels=False, include_lowest=True)

    result = {}
    for branch in branch_present:
        bsub = sub[sub["major_branch"] == branch]
        xs, ys, ses = [], [], []
        for bi, mid in enumerate(mids):
            bbin = bsub[bsub["bin"] == bi][metric_col]
            if len(bbin) >= min_count:
                xs.append(mid)
                ys.append(bbin.mean())
                ses.append(bbin.sem())
        if len(xs) >= 2:
            result[branch] = (np.array(xs), np.array(ys), np.array(ses))
    return result


def compute_tb_subtype_per_branch(branch_order, nr):
    """T/B subtype fractions per branch via spatial proximity (100 µm)."""
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        print("WARNING: scipy not found — skipping T/B subtype panel.")
        return None

    all_records = []

    for sid, exp in EXP_MAP.items():
        ann_path = EXP_BASE / exp / f"{sid}_annotation.csv"
        sc_path  = DATA_DIR / f"spatial_cells_structural_{sid}.pkl"
        if not ann_path.exists() or not sc_path.exists():
            print(f"  Missing files for {sid}, skipping.")
            continue

        nr_sid = (nr[nr["sample_id"] == sid][[NK, "image_id", "major_branch"]]
                  .drop_duplicates())

        ann = pd.read_csv(ann_path, index_col=0)

        with open(sc_path, "rb") as f:
            sc = pickle.load(f)
        sc["cell_id_str"] = sc["cell_id"].astype(str)

        # Auto-detect ID format (some samples store full "sid_N", others bare numeric)
        tb_probe = sc.loc[sc["Tier_A"].isin(["T cells", "B lineage"]), "cell_id_str"]
        if tb_probe.isin(ann.index).mean() > 0.05:
            ann_idx = ann
        else:
            ann_idx = ann.copy()
            ann_idx.index = ann_idx.index.str.replace(f"{sid}_", "", regex=False)

        sc = sc.merge(
            ann_idx[["Tier_B"]].rename_axis("cell_id_str"),
            left_on="cell_id_str", right_index=True, how="left",
        )

        tb = sc[
            sc["Tier_A"].isin(["T cells", "B lineage"])
            & sc["Tier_B"].notna()
            & sc["x"].notna()
        ].copy()
        if len(tb) == 0:
            continue
        tb_xy = tb[["x", "y"]].values

        # Ductal cells with correct branch from niche result
        duct = sc[sc["Tier_A"] == "pancreatic ductal epithelium"].copy()
        duct = duct.merge(
            nr_sid.rename(columns={"major_branch": "nr_branch"}),
            on=[NK, "image_id"], how="inner",
        )
        duct = duct[duct["nr_branch"].isin(branch_order) & duct["x"].notna()]

        for branch in branch_order:
            b_xy = duct.loc[duct["nr_branch"] == branch, ["x", "y"]].values
            if len(b_xy) < 10:
                continue
            tree = cKDTree(b_xy)
            dists, _ = tree.query(tb_xy, k=1, workers=-1)
            nearby   = tb[dists <= RADIUS_UM]["Tier_B"].value_counts()
            n_total  = nearby.sum()
            if n_total < 5:
                continue
            for subtype, count in nearby.items():
                all_records.append({"branch": branch, "subtype": subtype,
                                    "count": count})

    if not all_records:
        return None

    rec_df = pd.DataFrame(all_records)
    agg    = rec_df.groupby(["branch", "subtype"])["count"].sum().reset_index()
    agg["frac"] = agg["count"] / agg.groupby("branch")["count"].transform("sum")
    pivot  = agg.pivot(index="branch", columns="subtype", values="frac").fillna(0)
    return pivot.reindex(branch_order).dropna(how="all")


# ── Panel drawing helpers ────────────────────────────────────────────────────────

def _draw_binned_panel(ax, merged, pt_col, metric_col, metric_label,
                       branch_present, branch_palette, bio_names,
                       ylabel=None, ylim=None):
    """Binned dot+line per branch — suppfig2C/2D style with branch colours.
    ylim: shared y-axis limits tuple (lo, hi); auto if None."""
    binned = bin_by_pseudotime(merged, pt_col, metric_col, branch_present)

    # Mean reference (global, clipped)
    all_vals = merged[metric_col].dropna()
    lo, hi   = np.nanpercentile(all_vals, [1, 99])
    ax.axhline(all_vals.clip(lo, hi).mean(), color="#cccccc", lw=0.5,
               ls="--", zorder=0)

    for branch in branch_present:
        if branch not in binned:
            continue
        x, y, se = binned[branch]
        color = branch_palette.get(branch, "#888")
        ax.fill_between(x, y - se, y + se, color=color, alpha=0.12, zorder=1)
        ax.plot(x, y, color=color, lw=1.3, alpha=0.9,
                solid_capstyle="round", zorder=2)
        ax.scatter(x, y, s=7, color=color, edgecolors="white",
                   linewidths=0.4, zorder=3)

    ax.set_title(metric_label, fontsize=5.2, pad=2, fontweight="semibold")
    ax.set_xlim(-0.03, 1.03)
    ax.set_xticks([0, 0.5, 1.0])
    ax.tick_params(length=2, pad=1.5, labelsize=4.8)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=5.0, labelpad=2)
    else:
        ax.set_yticklabels([])
    ax.set_xlabel("Pseudotime (q)", fontsize=5.0, labelpad=1)


def _draw_subtype_panel(ax, tb_pivot, subtype, branch_present,
                        branch_palette, bio_labels, color):
    """One T/B subtype panel — branch identity on x, fraction on y.

    Dots are colored by branch palette (same as panels A/B legend) so the
    reader can immediately link each point to its branch color.  A thin gray
    line connects the points only to guide the eye across the categorical axis.
    Alternating vertical shading helps separate adjacent branches.
    """
    branches_in_tb = [b for b in branch_present if b in tb_pivot.index]
    if subtype not in tb_pivot.columns or len(branches_in_tb) < 2:
        ax.set_visible(False)
        return

    x = np.arange(len(branches_in_tb))
    y = tb_pivot.loc[branches_in_tb, subtype].values

    # Alternating vertical shading (even branches get a faint background)
    for i in range(0, len(branches_in_tb), 2):
        ax.axvspan(i - 0.5, i + 0.5, color="#f2f2f2", zorder=0, lw=0)

    # Mean reference
    ax.axhline(np.nanmean(y), color="#cccccc", lw=0.5, ls="--", zorder=1)

    # Thin gray connecting line — just guides the eye, no biological trend implied
    ax.plot(x, y, color="#aaaaaa", lw=0.8, alpha=0.7,
            solid_capstyle="round", zorder=2)

    # Branch-colored dots — each dot uses the same color as that branch in panels A/B
    for xi, yi, branch in zip(x, y, branches_in_tb):
        dot_color = branch_palette.get(branch, "#888888")
        ax.scatter(xi, yi, s=18, color=dot_color,
                   edgecolors="white", linewidths=0.5, zorder=3)

    # Spearman ρ across branch order
    if len(x) >= 4 and np.std(y) > 0:
        rho, p = spearmanr(x, y)
        sig    = "**" if p < 0.01 else ("*" if p < 0.05 else "")
        ax.text(0.97, 0.05, f"ρ = {rho:+.2f}{sig}",
                transform=ax.transAxes, fontsize=4.0,
                ha="right", va="bottom", color="#555555")

    # X-axis: short branch labels, colored by branch palette
    ax.set_xticks(x)
    ax.set_xticklabels(bio_labels, fontsize=4.0, rotation=40,
                       ha="right", rotation_mode="anchor")
    for tick_lbl, branch in zip(ax.get_xticklabels(), branches_in_tb):
        tick_lbl.set_color(branch_palette.get(branch, "#333333"))

    ax.set_xlim(-0.7, len(branches_in_tb) - 0.3)
    ax.tick_params(length=2, pad=1, labelsize=4.5)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)


# ── Main figure ─────────────────────────────────────────────────────────────────

def make_figure():
    merged = load_niche_data()

    avail_mod      = [c for c in MODULE_COLS if c in merged.columns]
    branch_palette = make_branch_palette(BRANCH_ORDER)
    bio_names      = assign_branch_bio_names(merged, avail_mod)
    branch_present = [b for b in BRANCH_ORDER if b in merged["major_branch"].unique()]

    # Very short branch labels for panel C x-axis (avoid overlap)
    SHORT_LABELS = {
        "trunk":       "Trunk",
        "branch 12":   "PanIN-1",
        "branch 20":   "Normal",
        "branch 15":   "Dediff.",
        "branch 17":   "InvGF-1",
        "branch 14.b": "InvGF-2",
        "branch 23.a": "PanIN-2",
    }
    bio_labels = [SHORT_LABELS.get(b, b) for b in branch_present]

    avail_morph = [c for c in MORPH_COLS if c in merged.columns]
    avail_fib   = [c for c in FIB_COLS   if c in merged.columns]
    pt_col      = "pooled_pseudotime_q"

    # ── T/B computation ────────────────────────────────────────────────────────
    print("Computing T/B cell subtype proximities …")
    tb_pivot = compute_tb_subtype_per_branch(branch_present, merged)
    has_tb   = tb_pivot is not None and len(tb_pivot) > 0
    if has_tb:
        print(f"  Branches: {list(tb_pivot.index)}")
        # Subtypes to show: ≥1% in any branch, ordered T cells first then B cells
        # "Other" excluded — not biologically meaningful
        keep = tb_pivot.columns[tb_pivot.max(axis=0) >= 0.01].tolist()
        ordered_subtypes = (
            [s for s in TCELL_SUBTYPES_KEEP if s in keep] +
            [s for s in BCELL_SUBTYPES_KEEP  if s in keep]
        )
        n_st = len(ordered_subtypes)
    else:
        ordered_subtypes = []
        n_st = 0

    # ── Compute shared y-limits for rows A and B ──────────────────────────────
    def _shared_ylim(cols):
        all_y = []
        for col in cols:
            for branch in branch_present:
                binned = bin_by_pseudotime(merged, pt_col, col, [branch])
                if branch in binned:
                    all_y.extend(binned[branch][1])
        if not all_y:
            return None
        lo, hi = min(all_y), max(all_y)
        pad = (hi - lo) * 0.18
        return (lo - pad, hi + pad)

    ylim_morph = _shared_ylim(avail_morph)
    ylim_fib   = _shared_ylim(avail_fib)

    # ── Layout ────────────────────────────────────────────────────────────────
    # Row A: 6 module scores in 2 rows × 3 cols
    n_morph   = len(avail_morph)       # up to 6
    A_COLS    = 3
    A_ROWS    = int(np.ceil(n_morph / A_COLS))   # 2
    n_fib     = len(avail_fib)         # 4
    tb_ncols  = 5                           # 5 cols → ceil(13/5)=3 rows (was 4 cols, 4 rows)
    tb_nrows  = int(np.ceil(n_st / tb_ncols)) if n_st > 0 else 0

    fig_w  = 170 * MM2IN
    rA_h   = 34 * A_ROWS  * MM2IN     # 68 mm for 2-row block (was 36→72)
    rB_h   = 34 * MM2IN               # fib row (was 38)
    rC_h   = (36 * tb_nrows) * MM2IN if has_tb else 0   # 36 mm/row (was 48)
    fig_h  = rA_h + rB_h + rC_h + 10 * MM2IN

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    n_main  = 3 if has_tb else 2
    hratios = ([rA_h, rB_h, rC_h] if has_tb else [rA_h, rB_h])
    gs_main = mgridspec.GridSpec(
        n_main, 1,
        left=0.08, right=0.86, top=0.97, bottom=0.04,
        hspace=0.50, height_ratios=hratios,
    )

    gs_a = mgridspec.GridSpecFromSubplotSpec(
        A_ROWS, A_COLS, subplot_spec=gs_main[0],
        hspace=0.72, wspace=0.22,
    )
    gs_b = mgridspec.GridSpecFromSubplotSpec(
        1, n_fib, subplot_spec=gs_main[1], wspace=0.22,
    )
    if has_tb and tb_nrows > 0:
        gs_c = mgridspec.GridSpecFromSubplotSpec(
            tb_nrows, tb_ncols, subplot_spec=gs_main[2],
            hspace=1.10, wspace=0.30,
        )

    # Branch legend (right of figure, spans rows A+B)
    leg_patches = [
        mpatches.Patch(
            facecolor=branch_palette.get(b, "#888"), linewidth=0,
            label="Trunk" if b == "trunk" else bio_names.get(b, b),
        )
        for b in branch_present
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # A — Disease-progression module scores (6 panels, 2×3 grid)
    # ═══════════════════════════════════════════════════════════════════════════
    for idx, col in enumerate(avail_morph):
        r, c = divmod(idx, A_COLS)
        ax   = fig.add_subplot(gs_a[r, c])
        is_left  = (c == 0)
        is_bot   = (r == A_ROWS - 1) or (idx >= n_morph - A_COLS)
        _draw_binned_panel(
            ax, merged, pt_col, col,
            MORPH_LABELS_SHORT.get(col, col),
            branch_present, branch_palette, bio_names,
            ylabel=("Score" if is_left else None),
            ylim=ylim_morph,
        )
        if not is_bot:
            ax.set_xticklabels([])
            ax.set_xlabel("")

    # Hide unused cells if n_morph not a multiple of A_COLS
    for idx in range(n_morph, A_ROWS * A_COLS):
        r, c = divmod(idx, A_COLS)
        fig.add_subplot(gs_a[r, c]).set_visible(False)

    pos_a = gs_main[0].get_position(fig)
    fig.text(
        (pos_a.x0 + pos_a.x1) / 2, pos_a.y1 + 0.014,
        "Disease-progression module scores along pseudotime  (per branch)",
        ha="center", va="bottom", fontsize=5.5, fontweight="bold",
        transform=fig.transFigure,
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # B — Fibroblast markers (binned dot+line per branch)
    # ═══════════════════════════════════════════════════════════════════════════
    for idx, col in enumerate(avail_fib):
        ax = fig.add_subplot(gs_b[0, idx])
        _draw_binned_panel(
            ax, merged, pt_col, col,
            FIB_LABELS.get(col, col),
            branch_present, branch_palette, bio_names,
            ylabel=("Mean expr.\n(z-scored)" if idx == 0 else None),
            ylim=ylim_fib,
        )

    pos_b = gs_main[1].get_position(fig)
    fig.text(
        (pos_b.x0 + pos_b.x1) / 2, pos_b.y1 + 0.018,
        "Fibroblast activation markers along pseudotime  (per branch)",
        ha="center", va="bottom", fontsize=5.5, fontweight="bold",
        transform=fig.transFigure,
    )

    # Single right-side legend spanning rows A+B
    fig.legend(
        handles=leg_patches, fontsize=4.0, ncol=1,
        loc="center left",
        bbox_to_anchor=(0.875, (pos_a.y0 + pos_b.y1) / 2),
        frameon=False, handlelength=0.9, handleheight=0.8,
        borderpad=0.2, labelspacing=0.30,
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # C — T/B subtype fractions per branch (suppfig2C/2D style)
    # ═══════════════════════════════════════════════════════════════════════════
    if has_tb and n_st > 0:
        pos_c = gs_main[2].get_position(fig)
        fig.text(
            (pos_c.x0 + pos_c.x1) / 2, pos_c.y1 + 0.018,
            "T/B cell subtype proximity near each branch  (≤100 µm)",
            ha="center", va="bottom", fontsize=5.5, fontweight="bold",
            transform=fig.transFigure,
        )

        for idx, subtype in enumerate(ordered_subtypes):
            r, c  = divmod(idx, tb_ncols)
            ax    = fig.add_subplot(gs_c[r, c])
            color = SUBTYPE_PALETTE.get(subtype, "#888888")

            _draw_subtype_panel(
                ax, tb_pivot, subtype, branch_present,
                branch_palette, bio_labels, color,
            )
            ax.set_title(subtype, fontsize=4.8, pad=2,
                         color=color, fontweight="semibold")
            if c == 0:
                ax.set_ylabel("Fraction of proximal\nT/B cells (≤100 µm)", fontsize=4.5, labelpad=2)
            else:
                ax.set_yticklabels([])
            ax.set_xlabel("")

        # Hide unused panels
        for idx in range(n_st, tb_nrows * tb_ncols):
            r, c = divmod(idx, tb_ncols)
            fig.add_subplot(gs_c[r, c]).set_visible(False)

        # Footnote
        fig.text(0.5, 0.005,
                 "Dot color matches branch identity in panels A and B.  "
                 "y = fraction of all T/B cells within ≤100 µm of each branch's ductal niches.",
                 ha="center", va="bottom", fontsize=4.0, color="#777777",
                 style="italic", transform=fig.transFigure)

    # ── Save ──────────────────────────────────────────────────────────────────
    OUT_SUPPFIG3.mkdir(parents=True, exist_ok=True)
    out_pdf = OUT_SUPPFIG3 / "suppfig3D_branch_profiling.pdf"
    out_png = OUT_SUPPFIG3 / "suppfig3D_branch_profiling.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved:\n  {out_pdf}\n  {out_png}")
    plt.show()


if __name__ == "__main__":
    make_figure()
