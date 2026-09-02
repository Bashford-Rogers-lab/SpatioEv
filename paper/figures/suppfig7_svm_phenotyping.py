"""
Supplementary Figure 7 — SVM-based cell phenotype refinement
=============================================================
Panel A : Workflow schematic (SCIMAP → features → RBF SVM → probabilities → refined phenotypes)
Panel B : Normalised confusion matrix (row-normalised = per-class recall)
Panel C : Per-class precision / recall / F1 horizontal bar chart
Panel D : Maximum SVM probability histogram by training status
Panel E : Per-class SVM probability boxplots
Panel F : Refinement outcome summary bar chart
Panel G : [Placeholder] GMM marker distributions      (colleague's autogating work)
Panel H : [Placeholder] Autogating accuracy / metrics (colleague's autogating work)
Panel I : [Placeholder] Autogating confusion matrix   (colleague's autogating work)

Data sources (pre-computed by 02_dev_SVM_phenotype_probility.ipynb):
  results/svm_phenotyping_test_confusion_matrix.csv
  results/svm_phenotyping_test_report.csv
  results/svm_phenotyping_results.csv

Run:
    python notebooks/suppfig7_svm_phenotyping.py

Output: paper/notebooks/results/suppfig7/suppfig7_*.pdf (.png)

──────────────────────────────────────────────────────────────────────────────
Key numbers from the pre-computed SVM run
──────────────────────────────────────────────────────────────────────────────
  51,932 total cells  |  3,824 artifact (excluded)
  48,108 modelled cells — 10 phenotype classes + Unknown
  Train 29,092 / Test 9,698 / Unknown (predict-only) 9,318
  Overall test-set accuracy: 95.1 %
  Confidence threshold: P ≥ 0.60
  Refinement: 6,467 Unknown resolved + 1,422 disagreements corrected
──────────────────────────────────────────────────────────────────────────────
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import matplotlib.patches as mpatches
import seaborn as sns


# ── File paths ─────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
RESULT_DIR = ROOT / "results"
OUT_DIR    = Path(__file__).parent / "results" / "suppfig7"

MM2IN = 1 / 25.4

# ── Constants ──────────────────────────────────────────────────────────────────
PROBABILITY_THRESHOLD = 0.60

PHENOTYPE_LABELS = {
    "CD4 T cells":     "CD4 T cells",
    "CD90+ CAFs":      "CD90+ CAFs",
    "Dendritic cells": "Dendritic cells",
    "Macrophages":     "Macrophages",
    "Monocytes":       "Monocytes",
    "PDPN+ CAFs":      "PDPN+ CAFs",
    "T cells":         "T cells",
    "endothelial":     "Endothelial",
    "myCAFs":          "myCAFs",
    "tumour":          "Tumour",
    "Unknown":         "Unknown",
}

STATUS_PALETTE = {
    "train":        "#4a90d9",
    "test":         "#e07b39",
    "predict_only": "#27ae60",
}
STATUS_LABELS = {
    "train":        "Train (n = 29,092)",
    "test":         "Test (n = 9,698)",
    "predict_only": "Unknown / predict-only (n = 9,318)",
}

REFINEMENT_ORDER = [
    "kept_original_label",
    "refined_unknown_from_prediction",
    "refined_disagreement_from_prediction",
    "kept_prediction_only_low_confidence",
]
REFINEMENT_DISPLAY = {
    "kept_original_label":                  "Kept original label",
    "refined_unknown_from_prediction":      "Unknown resolved",
    "refined_disagreement_from_prediction": "Disagreement corrected",
    "kept_prediction_only_low_confidence":  "Low-conf. Unknown kept",
}
REFINEMENT_COLORS = {
    "kept_original_label":                  "#bbbbbb",
    "refined_unknown_from_prediction":      "#27ae60",
    "refined_disagreement_from_prediction": "#4a90d9",
    "kept_prediction_only_low_confidence":  "#e07b39",
}


# ── Publication RC ─────────────────────────────────────────────────────────────
def _set_pub_rc():
    plt.rcParams.update({
        "font.family":         "Arial",
        "font.size":           6,
        "axes.titlesize":      6.5,
        "axes.labelsize":      6,
        "xtick.labelsize":     5.5,
        "ytick.labelsize":     5.5,
        "axes.linewidth":      0.5,
        "xtick.major.width":   0.5,
        "ytick.major.width":   0.5,
        "xtick.major.size":    2.0,
        "ytick.major.size":    2.0,
        "lines.linewidth":     0.8,
        "legend.fontsize":     5.0,
        "legend.handlelength": 0.8,
        "pdf.fonttype":        42,
        "svg.fonttype":        "none",
    })


def _despine(ax):
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)


# ── Data loading ───────────────────────────────────────────────────────────────
def load_data():
    cm  = pd.read_csv(RESULT_DIR / "svm_phenotyping_test_confusion_matrix.csv",
                      index_col=0)
    rpt = pd.read_csv(RESULT_DIR / "svm_phenotyping_test_report.csv",
                      index_col=0)
    res = pd.read_csv(RESULT_DIR / "svm_phenotyping_results.csv",
                      index_col=0)
    return cm, rpt, res


# ══════════════════════════════════════════════════════════════════════════════
# Panel drawing functions
# ══════════════════════════════════════════════════════════════════════════════

def draw_schematic(ax, show_label=True):
    """Panel A — SVM workflow schematic (5 steps)."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    steps = [
        {
            "title":   "Clustering +\nSCIMAP annotation",
            "face":    "#cce5ff",
            "edge":    "#4a90d9",
            "details": [
                ("Unsupervised Leiden\nclustering", "#444444"),
                ("SCIMAP prior-knowledge\nphenotyping", "#555555"),
                ("51,932 cells input", "#4a90d9"),
            ],
        },
        {
            "title":   "Feature\nconstruction",
            "face":    "#d4edda",
            "edge":    "#27ae60",
            "details": [
                ("25 expression markers\n(weight = 1.0)", "#444444"),
                ("19 morphology features\n(weight = 0.4)", "#444444"),
                ("44 features total", "#27ae60"),
            ],
        },
        {
            "title":   "RBF SVM\ntraining",
            "face":    "#fde8d0",
            "edge":    "#e07b39",
            "details": [
                ("Train: 29,092 cells", "#444444"),
                ("Test: 9,698 cells", "#444444"),
                ("10 classes · 95.1%\ntest accuracy", "#e07b39"),
            ],
        },
        {
            "title":   "Probability\nscoring",
            "face":    "#ede7f6",
            "edge":    "#7b5ea7",
            "details": [
                ("P(class) per cell", "#444444"),
                ("Confidence threshold\nP ≥ 0.60", "#7b5ea7"),
                ("Unknown cells\nresolved or flagged", "#444444"),
            ],
        },
        {
            "title":   "Refined\nphenotypes",
            "face":    "#d1ecf1",
            "edge":    "#17a2b8",
            "details": [
                ("6,467 Unknown resolved", "#27ae60"),
                ("1,422 labels corrected", "#4a90d9"),
                ("7,889 cells improved", "#17a2b8"),
            ],
        },
    ]

    n        = len(steps)
    box_w    = 0.155
    box_h    = 0.66
    title_h  = 0.17
    gap      = (1.0 - n * box_w) / (n + 1)
    y_center = 0.52

    for i, step in enumerate(steps):
        x0 = gap + i * (box_w + gap)
        y0 = y_center - box_h / 2

        ax.add_patch(mpatches.FancyBboxPatch(
            (x0, y0), box_w, box_h,
            boxstyle="round,pad=0.012",
            facecolor=step["face"], edgecolor=step["edge"],
            linewidth=0.8, transform=ax.transData, clip_on=False, zorder=2,
        ))
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0, y0 + box_h - title_h), box_w, title_h,
            boxstyle="round,pad=0.012",
            facecolor=step["edge"], edgecolor="none", alpha=0.28,
            transform=ax.transData, clip_on=False, zorder=3,
        ))
        ax.text(
            x0 + box_w / 2, y0 + box_h - title_h / 2,
            step["title"],
            ha="center", va="center", fontsize=4.8, fontweight="bold",
            color="#222222", transform=ax.transData, zorder=4, linespacing=1.25,
        )

        detail_area_h = box_h - title_h - 0.04
        slot_h = detail_area_h / len(step["details"])
        for j, (text, color) in enumerate(step["details"]):
            y_det = y0 + box_h - title_h - 0.02 - (j + 0.5) * slot_h
            ax.text(
                x0 + box_w / 2, y_det, text,
                ha="center", va="center", fontsize=3.6, color=color,
                transform=ax.transData, zorder=4, linespacing=1.2,
            )

        if i < n - 1:
            x_s = x0 + box_w + 0.003
            x_e = x_s + gap - 0.008
            ax.annotate(
                "", xy=(x_e, y_center), xytext=(x_s, y_center),
                xycoords="data", textcoords="data",
                arrowprops=dict(arrowstyle="-|>", color="#666666",
                                lw=0.9, mutation_scale=8),
                zorder=5,
            )

    prefix = "A   " if show_label else ""
    ax.set_title(f"{prefix}SVM phenotyping workflow", fontsize=6.5,
                 fontweight="semibold", pad=3, loc="left")


def draw_confusion_matrix(ax, cm, show_label=True):
    """Panel B — row-normalised confusion matrix (diagonal = recall)."""
    cm_norm = cm.div(cm.sum(axis=1), axis=0)
    labels  = [PHENOTYPE_LABELS.get(c, c) for c in cm.index]

    # Annotate cells ≥ 5 % only to avoid clutter
    annot = cm_norm.map(lambda x: f"{x:.2f}" if x >= 0.05 else "")

    sns.heatmap(
        cm_norm,
        ax=ax,
        cmap="Blues",
        vmin=0, vmax=1,
        linewidths=0.3, linecolor="#dddddd",
        xticklabels=labels, yticklabels=labels,
        annot=annot, fmt="",
        annot_kws={"size": 3.8},
        square=True,
        cbar_kws={"label": "Recall", "shrink": 0.70,
                  "pad": 0.02, "aspect": 20},
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right",
                       fontsize=4.8, va="top")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=4.8, rotation=0)
    ax.tick_params(length=0, pad=2)
    ax.set_xlabel("Predicted phenotype", fontsize=5.5, labelpad=3)
    ax.set_ylabel("True phenotype", fontsize=5.5, labelpad=3)

    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.ax.tick_params(labelsize=4.5, length=2, width=0.4)
        cbar.set_label("Recall", fontsize=5.0, labelpad=3)
        cbar.outline.set_linewidth(0.3)

    prefix = "B   " if show_label else ""
    ax.set_title(f"{prefix}Confusion matrix (recall-normalised)",
                 fontsize=6.5, fontweight="semibold", pad=3, loc="left")


def draw_per_class_metrics(ax, rpt, show_label=True):
    """Panel C — per-class precision / recall / F1 grouped horizontal bars."""
    skip = {"accuracy", "macro avg", "weighted avg"}
    classes = [c for c in rpt.index if c not in skip]
    df = rpt.loc[classes].sort_values("f1-score", ascending=True)  # bottom = lowest F1
    labels   = [PHENOTYPE_LABELS.get(c, c) for c in df.index]
    supports = df["support"].astype(int).values
    y = np.arange(len(df))
    h = 0.22

    ax.barh(y + h, df["precision"], height=h, color="#4a90d9",
            alpha=0.85, label="Precision", linewidth=0)
    ax.barh(y,     df["recall"],    height=h, color="#e07b39",
            alpha=0.85, label="Recall",    linewidth=0)
    ax.barh(y - h, df["f1-score"],  height=h, color="#27ae60",
            alpha=0.85, label="F1-score",  linewidth=0)

    total_test = supports.sum()
    for i, (cls, sup) in enumerate(zip(df.index, supports)):
        low = sup < 50
        txt = f"n = {sup}" + (" †" if low else "")
        ax.text(1.01, y[i], txt, va="center", ha="left", fontsize=3.8,
                color="#e74c3c" if low else "#777777")

    acc = float(rpt.loc["accuracy", "f1-score"])
    ax.axvline(acc, color="#333333", lw=0.8, ls="--",
               label=f"Overall acc. {acc:.1%}")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=5.0)
    ax.set_xlabel("Score", labelpad=2)
    ax.set_xlim(0, 1.20)
    ax.tick_params(length=2, pad=1)
    ax.legend(fontsize=4.5, frameon=False, loc="lower right",
              handlelength=0.8, borderpad=0.2, labelspacing=0.2)
    ax.text(0.0, -0.12, "† n < 50 test cells (low prevalence)",
            transform=ax.transAxes, fontsize=4.0, color="#e74c3c")
    _despine(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    prefix = "C   " if show_label else ""
    ax.set_title(f"{prefix}Per-class classification metrics",
                 fontsize=6.5, fontweight="semibold", pad=3, loc="left")


def draw_probability_histogram(ax, res, show_label=True):
    """Panel D — max SVM probability histogram by training status."""
    bins = np.linspace(0, 1, 51)

    for status in ["train", "test", "predict_only"]:
        vals = res.loc[res["svm_training_status"] == status,
                       "svm_max_probability"].values
        ax.hist(vals, bins=bins, color=STATUS_PALETTE[status], alpha=0.55,
                label=STATUS_LABELS[status], linewidth=0,
                density=True, rasterized=True)

    ax.axvline(PROBABILITY_THRESHOLD, color="#333333", lw=0.9, ls="--",
               label=f"Threshold = {PROBABILITY_THRESHOLD:.2f}")

    ax.set_xlabel("Maximum SVM probability", labelpad=2)
    ax.set_ylabel("Density", labelpad=2)
    ax.legend(fontsize=4.2, frameon=False, handlelength=0.9,
              loc="upper left", borderpad=0.2, labelspacing=0.28)
    ax.tick_params(length=2, pad=1)
    _despine(ax)

    prefix = "D   " if show_label else ""
    ax.set_title(f"{prefix}SVM prediction confidence",
                 fontsize=6.5, fontweight="semibold", pad=3, loc="left")


def draw_probability_boxplots(ax, res, show_label=True):
    """Panel E — per-class max-probability boxplots (train + test cells only)."""
    df = res[res["svm_training_status"].isin(["train", "test"])].copy()

    # Sort classes by median probability descending (highest confidence at left)
    order_keys = (
        df.groupby("scimap_phenotype")["svm_max_probability"]
          .median()
          .sort_values(ascending=False)
          .index.tolist()
    )
    order_labels = [PHENOTYPE_LABELS.get(k, k) for k in order_keys]
    cmap = plt.get_cmap("tab10", len(order_labels))
    box_data = [df[df["scimap_phenotype"] == k]["svm_max_probability"].values
                for k in order_keys]

    bplot = ax.boxplot(
        box_data,
        vert=True,
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops=dict(color="#222222", linewidth=1.0),
        whiskerprops=dict(linewidth=0.6, color="#555555"),
        capprops=dict(linewidth=0.6, color="#555555"),
        boxprops=dict(linewidth=0.5),
    )
    for i, patch in enumerate(bplot["boxes"]):
        patch.set_facecolor(cmap(i))
        patch.set_alpha(0.75)

    ax.axhline(PROBABILITY_THRESHOLD, color="#333333", lw=0.9, ls="--",
               label=f"Threshold = {PROBABILITY_THRESHOLD:.2f}")

    ax.set_xticks(range(1, len(order_labels) + 1))
    ax.set_xticklabels(order_labels, rotation=40, ha="right",
                       fontsize=4.8, va="top")
    ax.set_ylabel("Maximum SVM probability", labelpad=2)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=4.5, frameon=False, handlelength=0.9,
              loc="lower right", borderpad=0.2)
    ax.tick_params(length=2, pad=1)
    _despine(ax)

    prefix = "E   " if show_label else ""
    ax.set_title(f"{prefix}SVM confidence by phenotype class",
                 fontsize=6.5, fontweight="semibold", pad=3, loc="left")


def draw_refinement_summary(ax, res, show_label=True):
    """Panel F — refinement outcome horizontal bar chart."""
    # Merge the 4 'kept_original_label_model_disagrees' cells (n=4) into kept_original
    merge_map = {
        "kept_original_label":                  "kept_original_label",
        "kept_original_label_model_disagrees":  "kept_original_label",
        "refined_unknown_from_prediction":       "refined_unknown_from_prediction",
        "refined_disagreement_from_prediction":  "refined_disagreement_from_prediction",
        "kept_prediction_only_low_confidence":   "kept_prediction_only_low_confidence",
    }
    counts = res["svm_refinement_status"].map(merge_map).value_counts()
    total  = counts.sum()

    values = [int(counts.get(k, 0)) for k in REFINEMENT_ORDER]
    colors = [REFINEMENT_COLORS[k]  for k in REFINEMENT_ORDER]
    labels = [REFINEMENT_DISPLAY[k] for k in REFINEMENT_ORDER]
    y_pos  = np.arange(len(REFINEMENT_ORDER))

    bars = ax.barh(y_pos, values, color=colors, height=0.6,
                   linewidth=0.4, edgecolor="white")

    for bar, val in zip(bars, values):
        pct = val / total * 100
        ax.text(bar.get_width() + total * 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,}  ({pct:.1f}%)",
                va="center", ha="left", fontsize=5.0, color="#444444")

    n_improved = (counts.get("refined_unknown_from_prediction", 0) +
                  counts.get("refined_disagreement_from_prediction", 0))
    pct_improved = n_improved / total * 100

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=5.5)
    ax.set_xlabel("Cell count", labelpad=2)
    ax.set_xlim(0, total * 1.38)
    ax.tick_params(length=2, pad=1)
    _despine(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    prefix = "F   " if show_label else ""
    ax.set_title(
        f"{prefix}Refinement outcome  "
        f"[{n_improved:,} cells updated · {pct_improved:.1f}%]",
        fontsize=6.5, fontweight="semibold", pad=3, loc="left",
    )


def draw_placeholder(ax, letter, description, show_label=True):
    """Generic grey placeholder for colleague's autogating panels."""
    ax.set_facecolor("#f7f7f7")
    for sp in ax.spines.values():
        sp.set_color("#cccccc")
        sp.set_linewidth(0.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5, 0.56, "[ Placeholder ]",
            ha="center", va="center", fontsize=8,
            color="#c0c0c0", fontstyle="italic",
            transform=ax.transAxes)
    ax.text(0.5, 0.38, description,
            ha="center", va="center", fontsize=5.5,
            color="#c8c8c8", fontstyle="italic",
            transform=ax.transAxes)
    ax.text(0.5, 0.22, "(colleague's autogating work)",
            ha="center", va="center", fontsize=4.5,
            color="#d0d0d0", fontstyle="italic",
            transform=ax.transAxes)
    prefix = f"{letter}   " if show_label else ""
    ax.set_title(f"{prefix}{description}", fontsize=6.5,
                 fontweight="semibold", pad=3, loc="left")


# ══════════════════════════════════════════════════════════════════════════════
# Combined figure
# ══════════════════════════════════════════════════════════════════════════════

def make_figure():
    _set_pub_rc()

    print("Loading SVM result CSVs …")
    cm, rpt, res = load_data()

    fig_w = 170 * MM2IN
    fig_h = 210 * MM2IN
    fig   = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    # Row 1 — A (schematic, wide), B (confusion), C (metrics)
    gs1 = mgridspec.GridSpec(
        1, 3,
        width_ratios=[1.7, 1.1, 1.0],
        left=0.03, right=0.99, top=0.98, bottom=0.70,
        wspace=0.42,
    )
    ax_a = fig.add_subplot(gs1[0, 0])
    ax_b = fig.add_subplot(gs1[0, 1])
    ax_c = fig.add_subplot(gs1[0, 2])

    # Row 2 — D (histogram), E (boxplots, wide), F (refinement bar)
    gs2 = mgridspec.GridSpec(
        1, 3,
        width_ratios=[0.85, 1.35, 0.90],
        left=0.06, right=0.99, top=0.64, bottom=0.36,
        wspace=0.48,
    )
    ax_d = fig.add_subplot(gs2[0, 0])
    ax_e = fig.add_subplot(gs2[0, 1])
    ax_f = fig.add_subplot(gs2[0, 2])

    # Row 3 — G, H, I (placeholders, equal width)
    gs3 = mgridspec.GridSpec(
        1, 3,
        left=0.05, right=0.99, top=0.30, bottom=0.04,
        wspace=0.35,
    )
    ax_g = fig.add_subplot(gs3[0, 0])
    ax_h = fig.add_subplot(gs3[0, 1])
    ax_i = fig.add_subplot(gs3[0, 2])

    draw_schematic(ax_a)
    draw_confusion_matrix(ax_b, cm)
    draw_per_class_metrics(ax_c, rpt)
    draw_probability_histogram(ax_d, res)
    draw_probability_boxplots(ax_e, res)
    draw_refinement_summary(ax_f, res)
    draw_placeholder(ax_g, "G", "Spatial overlay — disagreement corrections")
    draw_placeholder(ax_h, "H", "Autogating accuracy / metrics")
    draw_placeholder(ax_i, "I", "Autogating confusion matrix")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = OUT_DIR / "suppfig7_svm_phenotyping.pdf"
    out_png = OUT_DIR / "suppfig7_svm_phenotyping.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nSaved:\n  {out_pdf}\n  {out_png}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Standalone panel export
# ══════════════════════════════════════════════════════════════════════════════

_PANEL_SIZES = {
    "A": (96,  50),   # 5-box schematic, wide
    "B": (72,  78),   # 10×10 square heatmap
    "C": (80,  68),   # horizontal bars, 10 classes
    "D": (45,  58),   # histogram
    "E": (46,  68),   # boxplots, 10 classes
    "F": (58,  44),   # refinement bar chart
    "G": (58,  55),   # placeholder
    "H": (58,  55),   # placeholder
    "I": (58,  55),   # placeholder
}


def _save_panel(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"suppfig7_{name}"
    pdf  = OUT_DIR / f"{stem}.pdf"
    png  = OUT_DIR / f"{stem}.png"
    fig.savefig(pdf, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {pdf.name}")


def make_standalone_panels():
    """Save each panel A–I as an individual PDF + PNG."""
    _set_pub_rc()

    print(f"\nLoading SVM result CSVs …")
    cm, rpt, res = load_data()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\nSaving standalone panels:")

    def _fig(panel):
        w, h = _PANEL_SIZES[panel]
        return plt.figure(figsize=(w * MM2IN, h * MM2IN), facecolor="white")

    # Panel A — schematic
    fig = _fig("A")
    ax  = fig.add_axes([0.02, 0.06, 0.96, 0.88])
    draw_schematic(ax, show_label=False)
    _save_panel(fig, "A")

    # Panel B — confusion matrix
    fig = _fig("B")
    ax  = fig.add_axes([0.22, 0.22, 0.65, 0.70])
    draw_confusion_matrix(ax, cm, show_label=False)
    _save_panel(fig, "B")

    # Panel C — per-class metrics
    fig = _fig("C")
    ax  = fig.add_axes([0.22, 0.16, 0.72, 0.76])
    draw_per_class_metrics(ax, rpt, show_label=False)
    _save_panel(fig, "C")

    # Panel D — probability histogram
    fig = _fig("D")
    ax  = fig.add_axes([0.16, 0.18, 0.80, 0.74])
    draw_probability_histogram(ax, res, show_label=False)
    _save_panel(fig, "D")

    # Panel E — probability boxplots
    fig = _fig("E")
    ax  = fig.add_axes([0.12, 0.22, 0.84, 0.70])
    draw_probability_boxplots(ax, res, show_label=False)
    _save_panel(fig, "E")

    # Panel F — refinement summary
    fig = _fig("F")
    ax  = fig.add_axes([0.28, 0.14, 0.64, 0.78])
    draw_refinement_summary(ax, res, show_label=False)
    _save_panel(fig, "F")

    # Panel G — placeholder
    fig = _fig("G")
    ax  = fig.add_axes([0.05, 0.05, 0.90, 0.88])
    draw_placeholder(ax, "G", "Spatial overlay — disagreement corrections", show_label=False)
    _save_panel(fig, "G")

    # Panel H — placeholder
    fig = _fig("H")
    ax  = fig.add_axes([0.05, 0.05, 0.90, 0.88])
    draw_placeholder(ax, "H", "Autogating accuracy / metrics", show_label=False)
    _save_panel(fig, "H")

    # Panel I — placeholder
    fig = _fig("I")
    ax  = fig.add_axes([0.05, 0.05, 0.90, 0.88])
    draw_placeholder(ax, "I", "Autogating confusion matrix", show_label=False)
    _save_panel(fig, "I")

    print(f"\nAll panels saved to {OUT_DIR}/")


if __name__ == "__main__":
    make_figure()
    make_standalone_panels()
