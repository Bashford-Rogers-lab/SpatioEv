#!/usr/bin/env python3
"""
suppfig_cellchat.py
====================
Supplementary figure: SpatialCellChat — cell-cell communication across
pseudotime bins in Xenium PDAC samples.

Panels
------
A  Total interaction strength across pseudotime bins (line plot, 3 samples)
B  Active pathway count per bin (grouped bar chart)
C  Pathway-level communication heatmap (pathways x bins, grouped by sample)
D  Top LR-pair bubble plot: x=pseudotime bin, y=LR pair, size=prob, color=pathway

Usage (from notebooks/):
    python suppfig_cellchat.py

Outputs:
    results/suppfig_cellchat/suppfig_cellchat.pdf
    results/suppfig_cellchat/suppfig_cellchat.png
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker

# ---- Paths ------------------------------------------------------------------
ROOT     = Path(__file__).resolve().parents[1]   # SpatioEv/
DATA_DIR = ROOT / "data" / "xenium_pancreas_10x" / "spatialcellchat"
OUT_DIR  = ROOT  / "paper" / "notebooks" / "results" / "suppfig_cellchat"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT  / "paper" / "notebooks"))
from fig2_shared_config import MM2IN, set_pub_rc
set_pub_rc()

# ---- Data -------------------------------------------------------------------
# Prefer the _improved run if it exists, fall back to original May-31 run.
lr_path = (DATA_DIR / "lr_interaction_summary_improved.csv"
           if (DATA_DIR / "lr_interaction_summary_improved.csv").exists()
           else DATA_DIR / "lr_interaction_summary.csv")
pw_path = (DATA_DIR / "pathway_summary_improved.csv"
           if (DATA_DIR / "pathway_summary_improved.csv").exists()
           else DATA_DIR / "pathway_summary.csv")

lr = pd.read_csv(lr_path)
pw = pd.read_csv(pw_path)

# ---- Constants --------------------------------------------------------------
SAMPLE_ORDER  = ["normal_nondiseased_v1", "pdac_pancreas_v1", "pdac_addon_v1", "pdac_io_v1"]
SAMPLE_LABELS = {
    "normal_nondiseased_v1": "Normal",
    "pdac_pancreas_v1":      "PDAC (primary)",
    "pdac_addon_v1":         "PDAC (add-on)",
    "pdac_io_v1":            "PDAC (IO)",
}
SAMPLE_COLORS = {
    "normal_nondiseased_v1": "#2ca02c",
    "pdac_pancreas_v1":      "#d62728",
    "pdac_addon_v1":         "#ff7f0e",
    "pdac_io_v1":            "#9467bd",
}
BIN_ORDER  = ["bin1", "bin2", "bin3", "bin4", "bin5"]
BIN_LABELS = {"bin1": "B1", "bin2": "B2", "bin3": "B3", "bin4": "B4", "bin5": "B5"}
BIN_X      = {b: i for i, b in enumerate(BIN_ORDER)}

present_samples = [s for s in SAMPLE_ORDER if s in lr["sample_id"].unique()]

PW_PALETTE = {
    "CXCL":     "#1f77b4",
    "EDN":      "#d62728",
    "CCL":      "#ff7f0e",
    "IL1":      "#9467bd",
    "RESISTIN": "#8c564b",
    "CD70":     "#17becf",
    "MHC-II":   "#bcbd22",
    "GRN":      "#e377c2",
}

W_IN = 183 * MM2IN
H_IN = 200 * MM2IN
fig  = plt.figure(figsize=(W_IN, H_IN), facecolor="white")

# ---- Panel A: Total interaction strength ------------------------------------
gs_a = gridspec.GridSpec(1, 1, figure=fig, left=0.10, right=0.50, top=0.97, bottom=0.76)
ax_a = fig.add_subplot(gs_a[0, 0])
fig.text(0.03, 0.97, "A", fontsize=7, fontweight="bold", va="top")

agg = (lr.groupby(["sample_id", "pt_bin"])["prob"]
         .sum().reset_index().rename(columns={"prob": "total_prob"}))
for sid in present_samples:
    sub = agg[agg["sample_id"] == sid].copy()
    sub["bin_x"] = sub["pt_bin"].map(BIN_X)
    sub = sub.sort_values("bin_x")
    ax_a.plot(sub["bin_x"], sub["total_prob"] * 1e3,
              color=SAMPLE_COLORS[sid], lw=1.2, marker="o", markersize=2.5,
              label=SAMPLE_LABELS[sid], zorder=3)
ax_a.set_xticks(range(len(BIN_ORDER)))
ax_a.set_xticklabels([BIN_LABELS[b] for b in BIN_ORDER], fontsize=5)
ax_a.set_xlabel("Pseudotime bin", fontsize=5.5, labelpad=2)
ax_a.set_ylabel("Total signal (prob x 10^3)", fontsize=5.5, labelpad=3)
ax_a.set_title("Interaction strength", fontsize=5.5, pad=3)
ax_a.tick_params(labelsize=4.5)
ax_a.legend(fontsize=4, frameon=False, handlelength=1.2, handletextpad=0.4,
            labelspacing=0.3, loc="upper right", borderaxespad=0.3)
ax_a.spines[["top", "right"]].set_visible(False)
ax_a.yaxis.set_major_locator(mticker.MaxNLocator(4))

# ---- Panel B: Pathway count per bin -----------------------------------------
gs_b = gridspec.GridSpec(1, 1, figure=fig, left=0.57, right=0.97, top=0.97, bottom=0.76)
ax_b = fig.add_subplot(gs_b[0, 0])
fig.text(0.53, 0.97, "B", fontsize=7, fontweight="bold", va="top")

n_pw = (lr.groupby(["sample_id", "pt_bin"])["pathway_name"]
          .nunique().reset_index()
          .rename(columns={"pathway_name": "n_pathways"}))
bar_w = 0.22
offsets = {s: (i - (len(present_samples) - 1) / 2) * bar_w
           for i, s in enumerate(present_samples)}
for sid in present_samples:
    sub = n_pw[n_pw["sample_id"] == sid].copy()
    sub["bin_x"] = sub["pt_bin"].map(BIN_X)
    sub = sub.sort_values("bin_x")
    xs = [BIN_X[b] + offsets[sid] for b in sub["pt_bin"]]
    ax_b.bar(xs, sub["n_pathways"], width=bar_w * 0.85,
             color=SAMPLE_COLORS[sid], alpha=0.85, zorder=2,
             label=SAMPLE_LABELS[sid])
ax_b.set_xticks(range(len(BIN_ORDER)))
ax_b.set_xticklabels([BIN_LABELS[b] for b in BIN_ORDER], fontsize=5)
ax_b.set_xlabel("Pseudotime bin", fontsize=5.5, labelpad=2)
ax_b.set_ylabel("Active pathways (n)", fontsize=5.5, labelpad=3)
ax_b.set_title("Pathway count per bin", fontsize=5.5, pad=3)
ax_b.tick_params(labelsize=4.5)
ax_b.legend(fontsize=4, frameon=False, handlelength=0.8, handletextpad=0.3,
            labelspacing=0.3, loc="upper left", borderaxespad=0.3)
ax_b.spines[["top", "right"]].set_visible(False)
ax_b.yaxis.set_major_locator(mticker.MaxNLocator(5, integer=True))

# ---- Panel C: Pathway heatmap -----------------------------------------------
gs_c = gridspec.GridSpec(1, 1, figure=fig, left=0.10, right=0.88, top=0.72, bottom=0.44)
ax_c = fig.add_subplot(gs_c[0, 0])
fig.text(0.03, 0.72, "C", fontsize=7, fontweight="bold", va="top")

agg_pw = (pw.groupby(["pathway_name", "sample_id", "pt_bin"])["prob"]
            .sum().reset_index())
agg_pw["col_key"] = (agg_pw["sample_id"].map(SAMPLE_LABELS) + "\n"
                     + agg_pw["pt_bin"].map(BIN_LABELS))
pivot = agg_pw.pivot_table(index="pathway_name", columns="col_key",
                            values="prob", aggfunc="sum")
col_order, sample_boundaries = [], []
for sid in present_samples:
    if sid not in agg_pw["sample_id"].unique():
        continue
    label = SAMPLE_LABELS[sid]
    for b in BIN_ORDER:
        key = f"{label}\n{BIN_LABELS[b]}"
        if key in pivot.columns:
            col_order.append(key)
    sample_boundaries.append((len(col_order), sid))
pivot = pivot.reindex(columns=[c for c in col_order if c in pivot.columns], fill_value=0)
row_order = pivot.sum(axis=1).sort_values(ascending=False).index.tolist()
pivot = pivot.loc[row_order]
data = pivot.values.copy()
row_max = data.max(axis=1, keepdims=True); row_max[row_max == 0] = 1
data_norm = data / row_max

im = ax_c.imshow(data_norm, aspect="auto", cmap="Blues", vmin=0, vmax=1)
ax_c.set_yticks(range(len(row_order)))
ax_c.set_yticklabels(row_order, fontsize=5)
ax_c.set_xticks(range(len(col_order)))
ax_c.set_xticklabels(
    [c.replace("\n", " ").replace("Normal ", "Norm ") for c in col_order],
    fontsize=3.8, rotation=45, ha="right")
ax_c.set_title("Pathway communication (row-normalized)", fontsize=5.5, pad=3)
ax_c.tick_params(left=False, bottom=False, labelsize=4.5)
prev = 0
for end_col, sid in sample_boundaries:
    n_cols = end_col - prev
    if prev > 0:
        ax_c.axvline(prev - 0.5, color="white", lw=2, zorder=3)
    ax_c.text((prev + end_col - 1) / 2, -1.6, SAMPLE_LABELS[sid],
              fontsize=4.5, ha="center", va="top",
              color=SAMPLE_COLORS[sid], fontweight="bold")
    prev = end_col
cax = fig.add_axes([0.90, 0.45, 0.008, 0.20])
cb  = fig.colorbar(im, cax=cax)
cb.set_label("Norm. prob.", fontsize=4, labelpad=2)
cb.ax.tick_params(labelsize=3.5)
cb.set_ticks([0, 0.5, 1])

# ---- Panel D: LR bubble plot ------------------------------------------------
n_samp = len(present_samples)
gs_d = gridspec.GridSpec(1, n_samp, figure=fig,
                          left=0.18, right=0.97, top=0.38, bottom=0.06,
                          wspace=0.15)
fig.text(0.03, 0.38, "D", fontsize=7, fontweight="bold", va="top")

top_lr = (lr.groupby("interaction_name_2")["prob"]
            .sum().sort_values(ascending=False).head(12).index.tolist())
lr_sub = lr[lr["interaction_name_2"].isin(top_lr)].copy()
y_order = (lr_sub.groupby("interaction_name_2")["prob"]
                 .sum().sort_values(ascending=False).index.tolist())
prob_max   = lr_sub["prob"].max()
SIZE_SCALE = 60 / prob_max if prob_max > 0 else 1
all_pathways = sorted(lr_sub["pathway_name"].dropna().unique())
pw_colors = {p: PW_PALETTE.get(p, f"#{abs(hash(p)) & 0xFFFFFF:06x}") for p in all_pathways}

for col_i, sid in enumerate(present_samples):
    ax = fig.add_subplot(gs_d[0, col_i])
    sub = lr_sub[lr_sub["sample_id"] == sid]
    for _, row in sub.iterrows():
        if row["interaction_name_2"] not in y_order:
            continue
        xi = BIN_X.get(row["pt_bin"])
        if xi is None: continue
        yi  = len(y_order) - 1 - y_order.index(row["interaction_name_2"])
        s   = max(row["prob"] * SIZE_SCALE, 0.3)
        ax.scatter(xi, yi, s=s,
                   color=pw_colors.get(str(row["pathway_name"]).strip(), "#888"),
                   alpha=0.80, linewidths=0.3, edgecolors="none", zorder=3)
    ax.set_xlim(-0.6, 4.6)
    ax.set_ylim(-0.8, len(y_order) - 0.2)
    ax.set_xticks(range(len(BIN_ORDER)))
    ax.set_xticklabels([BIN_LABELS[b] for b in BIN_ORDER], fontsize=4)
    ax.set_xlabel("Pseudotime bin", fontsize=4.5, labelpad=2)
    if col_i == 0:
        ax.set_yticks(range(len(y_order)))
        ax.set_yticklabels(list(reversed(y_order)), fontsize=3.8)
    else:
        ax.set_yticks([])
    ax.set_title(SAMPLE_LABELS.get(sid, sid), fontsize=5,
                 color=SAMPLE_COLORS.get(sid, "#333"), pad=3, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=4)
    for yi in range(len(y_order)):
        ax.axhline(yi, color="#eeeeee", lw=0.4, zorder=0)
    for xi in range(len(BIN_ORDER)):
        ax.axvline(xi, color="#eeeeee", lw=0.4, zorder=0)

# Pathway legend
fig.legend(
    handles=[mpatches.Patch(facecolor=pw_colors[p], label=p, linewidth=0)
             for p in all_pathways],
    fontsize=4.5, frameon=False, ncol=len(all_pathways),
    loc="lower center", bbox_to_anchor=(0.58, 0.005),
    handlelength=0.8, handletextpad=0.3, columnspacing=0.8,
    title="Pathway", title_fontsize=4.5)

# Size legend
size_vals = [prob_max * f for f in [0.25, 0.5, 1.0]]
fig.legend(
    handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor="#888",
                    markersize=np.sqrt(max(v * SIZE_SCALE, 0.3)),
                    label=f"{v*1e6:.1f}e-6") for v in size_vals],
    fontsize=4, frameon=False, ncol=1,
    loc="lower left", bbox_to_anchor=(0.03, 0.02),
    handletextpad=0.3, labelspacing=0.4,
    title="Prob.", title_fontsize=4)

# ---- Save -------------------------------------------------------------------
pdf = OUT_DIR / "suppfig_cellchat.pdf"
png = OUT_DIR / "suppfig_cellchat.png"
fig.savefig(pdf, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {pdf}")
print(f"Saved: {png}")
plt.close(fig)
