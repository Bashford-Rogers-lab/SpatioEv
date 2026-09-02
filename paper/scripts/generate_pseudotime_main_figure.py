"""Generate the detailed pseudotime-focused main Figure 1.

The layout follows the earlier manuscript's detailed trajectory figure logic,
but updates the analysis narrative to the current SpatioEv workflow:
representative PDAC sample first, then pooled four-sample trajectory analysis.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import fill

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpecFromSubplotSpec


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "manuscript" / "figures"
TABLE_DIR = ROOT / "manuscript" / "analysis_tables"

REP_SAMPLE = "34434_1"
REP_SAMPLE_LABEL = "Representative PDAC sample"

PALETTE = {
    "blue": "#3178b7",
    "teal": "#2a9d8f",
    "green": "#59a14f",
    "orange": "#f28e2b",
    "red": "#e15759",
    "purple": "#8e6bbf",
    "gold": "#edc948",
    "gray": "#6f6f6f",
    "navy": "#273c75",
}

PHENOTYPE_COLORS = {
    "Fibroblasts": "#2a9d8f",
    "pancreatic ductal epithelium": "#4e79a7",
    "Vimentin only mesenchyme": "#b07aa1",
    "Endothelial cells": "#59a14f",
    "pancreatic acinar epithelium": "#f28e2b",
    "T cells": "#e15759",
    "B lineage": "#edc948",
    "Vascular smooth muscle": "#76b7b2",
    "Muscularis externa": "#9c755f",
    "noise": "#bab0ab",
}

SAMPLE_COLORS = {
    "40331_1": "#273c75",
    "34434_1": "#3178b7",
    "33694_1": "#e15759",
    "35559_1": "#8e6bbf",
}


def configure_style() -> None:
    sns.set_theme(context="paper", style="white")
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_letter(ax: plt.Axes, letter: str, x: float = -0.08, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="right",
    )


def zscore_frame(df: pd.DataFrame, axis: int = 0) -> pd.DataFrame:
    values = df.astype(float)
    if axis == 0:
        return (values - values.mean()) / values.std(ddof=0).replace(0, np.nan)
    return values.sub(values.mean(axis=1), axis=0).div(
        values.std(axis=1, ddof=0).replace(0, np.nan), axis=0
    )


def shorten(text: str, width: int = 26) -> str:
    text = str(text).replace("__", " ").replace("_", " ")
    if len(text) <= width:
        return text
    return text[: width - 1] + "..."


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = ROOT / "data" / "combined_exp_2_3_4_5"
    result = pd.read_pickle(base / "pooled_niche_result_df.pkl")
    spatial = pd.read_pickle(
        base / f"spatial_cells_auto_branch_n24_v3_with_epithelial_{REP_SAMPLE}.pkl"
    )
    panin = pd.read_pickle(base / "pooled_pathology_with_panin_validation_scores.pkl")
    trends = pd.read_csv(
        ROOT
         / "paper" / "notebooks"
        / "results"
        / "trajectory_microenvironment_interactions"
        / "tables"
        / "multiplexed_microenvironment_trends_contextual.csv"
    )
    merged = result.copy()
    for col in [
        "panin_validation__normal_duct_like_score",
        "panin_validation__lg_panin_like_score",
        "panin_validation__hg_panin_like_score",
        "panin_validation__invasive_desmoplastic_context_score",
        "panin_validation__panin_grade_like_axis",
    ]:
        if col in panin.columns and col not in merged.columns:
            merged[col] = panin[col].to_numpy()
    return merged, spatial, panin, trends


def plot_workflow(ax: plt.Axes) -> None:
    ax.axis("off")
    panel_letter(ax, "A", x=-0.015, y=1.02)
    ax.set_title(
        "Updated SpatioEv pseudotime workflow: representative sample first, then pooled atlas",
        loc="left",
        pad=6,
    )
    stages = [
        ("Segmented cells", "phenotypes\nmarker state\ncoordinates"),
        ("Ductal niches", "connected epithelial\ncomponents"),
        ("Feature blocks", "morphology\npixel texture\ntopology\nsurroundings"),
        ("Pathology modules", "duct organization\nPanIN-like dysplasia\ninvasion/desmoplasia"),
        ("Principal tree", "root at early\norganized duct-like\nniches"),
        ("Tissue readout", "pseudotime maps\nbranch programs\nniche ecology"),
        ("Pooled atlas", "1 normal pancreas\n3 PDAC samples\n46,574 niches"),
    ]
    xs = np.linspace(0.055, 0.945, len(stages))
    colors = [
        PALETTE["gray"],
        PALETTE["blue"],
        PALETTE["teal"],
        PALETTE["purple"],
        PALETTE["orange"],
        PALETTE["red"],
        PALETTE["navy"],
    ]
    for i, ((title, body), x, color) in enumerate(zip(stages, xs, colors)):
        rect = mpl.patches.FancyBboxPatch(
            (x - 0.055, 0.36),
            0.11,
            0.40,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            facecolor=color,
            edgecolor="none",
            alpha=0.95,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        ax.text(
            x,
            0.66,
            title,
            color="white",
            weight="bold",
            ha="center",
            va="center",
            fontsize=8,
            transform=ax.transAxes,
        )
        ax.text(
            x,
            0.49,
            body,
            color="white",
            ha="center",
            va="center",
            fontsize=7,
            linespacing=1.15,
            transform=ax.transAxes,
        )
        if i < len(stages) - 1:
            ax.annotate(
                "",
                xy=(xs[i + 1] - 0.071, 0.56),
                xytext=(x + 0.066, 0.56),
                xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="-|>", color="#5f6b7a", lw=1.1),
            )
    ax.text(
        0.055,
        0.18,
        "New inference upgrade: graph/topology, boundary exposure, pixel morphology, and graph-defined surrounding context are integrated before trajectory fitting.",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=8,
        color="#253858",
    )


def plot_spatial_phenotypes(ax: plt.Axes, spatial: pd.DataFrame) -> None:
    panel_letter(ax, "B")
    n = min(140_000, len(spatial))
    show = spatial.sample(n=n, random_state=11)
    colors = show["Tier_A"].map(PHENOTYPE_COLORS).fillna("#c9c9c9")
    ax.scatter(show["x"], show["y"], c=colors, s=0.12, linewidths=0, alpha=0.72, rasterized=True)
    ax.set_title(f"{REP_SAMPLE_LABEL}\ncell phenotype scaffold")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    handles = []
    for label in [
        "pancreatic ductal epithelium",
        "Fibroblasts",
        "Vimentin only mesenchyme",
        "Endothelial cells",
        "T cells",
        "B lineage",
    ]:
        handles.append(
            mpl.lines.Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=PHENOTYPE_COLORS[label],
                markersize=4,
                label=shorten(label, 18),
            )
        )
    ax.legend(handles=handles, loc="lower left", frameon=False, ncol=1, handletextpad=0.3)


def plot_ductal_niche_map(ax: plt.Axes, spatial: pd.DataFrame) -> None:
    panel_letter(ax, "C")
    bg = spatial.sample(n=min(90_000, len(spatial)), random_state=12)
    ax.scatter(bg["x"], bg["y"], c="#dddddd", s=0.08, linewidths=0, alpha=0.28, rasterized=True)
    niche = spatial.loc[spatial["has_pooled_niche"].fillna(False)].copy()
    niche = niche.sample(n=min(100_000, len(niche)), random_state=13)
    branches = niche["major_branch"].fillna("unassigned").astype(str)
    top_branches = branches.value_counts().head(8).index.tolist()
    branch_show = branches.where(branches.isin(top_branches), "other")
    palette = dict(
        zip(
            top_branches + ["other"],
            sns.color_palette("tab20", n_colors=len(top_branches) + 1),
        )
    )
    ax.scatter(
        niche["x"],
        niche["y"],
        c=branch_show.map(palette),
        s=0.25,
        linewidths=0,
        alpha=0.85,
        rasterized=True,
    )
    ax.set_title("ductal niche branch\nassignments in tissue")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=palette[label],
            markersize=3.5,
            label=label,
        )
        for label in top_branches[:5]
    ]
    ax.legend(handles=handles, frameon=False, loc="lower left", handletextpad=0.25)


def plot_feature_blocks(ax: plt.Axes) -> None:
    panel_letter(ax, "D")
    ax.axis("off")
    ax.set_title("feature blocks entering\nnew trajectory inference", loc="left")
    rows = [
        ("Morph.", "N:C ratio, area, shape,\nconcavities, polarity"),
        ("Pixel", "entropy, lacunarity,\nHaralick, CK19/NaKATPase"),
        ("Graph", "degree, clustering,\nbridges, skeleton branches"),
        ("Edge", "boundary fraction,\nexternal degree, cross-edges"),
        ("Context", "fibroblasts, immune,\nendothelium, stromal markers"),
        ("Modules", "early duct, PanIN-like,\ninvasion/desmoplasia"),
    ]
    y0 = 0.88
    for i, (name, desc) in enumerate(rows):
        y = y0 - i * 0.145
        ax.add_patch(
            mpl.patches.Rectangle(
                (0.02, y - 0.065),
                0.035,
                0.095,
                transform=ax.transAxes,
                color=list(PALETTE.values())[i % len(PALETTE)],
                alpha=0.92,
            )
        )
        ax.text(
            0.075,
            y - 0.033,
            name,
            transform=ax.transAxes,
            color="#222222",
            weight="bold",
            fontsize=7,
            ha="left",
            va="center",
        )
        ax.text(0.43, y - 0.033, desc, transform=ax.transAxes, color="#222222", fontsize=6.5, va="center")


def plot_umap_pseudotime(ax: plt.Axes, df: pd.DataFrame, title: str, letter: str) -> None:
    panel_letter(ax, letter)
    show = df.sample(n=min(len(df), 18_000), random_state=15)
    sc = ax.scatter(
        show["UMAP1"],
        show["UMAP2"],
        c=show["pooled_pseudotime"],
        cmap="turbo",
        s=1.3,
        linewidths=0,
        alpha=0.72,
        rasterized=True,
    )
    ax.set_title(title)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.01, label="pseudotime")


def plot_tissue_pseudotime(ax: plt.Axes, spatial: pd.DataFrame) -> None:
    panel_letter(ax, "F")
    bg = spatial.sample(n=min(90_000, len(spatial)), random_state=16)
    ax.scatter(bg["x"], bg["y"], c="#d7d7d7", s=0.08, linewidths=0, alpha=0.22, rasterized=True)
    niche = spatial.loc[spatial["has_pooled_niche"].fillna(False)].copy()
    sc = ax.scatter(
        niche.sample(n=min(120_000, len(niche)), random_state=17)["x"],
        niche.sample(n=min(120_000, len(niche)), random_state=17)["y"],
        c=niche.sample(n=min(120_000, len(niche)), random_state=17)["pooled_pseudotime"],
        cmap="turbo",
        s=0.23,
        linewidths=0,
        alpha=0.86,
        rasterized=True,
    )
    ax.set_title("pseudotime projected\nback to tissue space")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.01, label="pseudotime")


def select_representative_components(sample_df: pd.DataFrame) -> list[tuple[str, float, str]]:
    candidates = sample_df.loc[sample_df["n_cells"] >= 12].copy()
    quantiles = [0.1, 0.5, 0.9]
    labels = ["early", "intermediate", "late"]
    selected = []
    for q, label in zip(quantiles, labels):
        target = candidates["pooled_pseudotime"].quantile(q)
        idx = (candidates["pooled_pseudotime"] - target).abs().idxmin()
        row = candidates.loc[idx]
        selected.append(
            (
                str(row["pancreatic ductal epithelium_mask_component"]),
                float(row["pooled_pseudotime"]),
                label,
            )
        )
    return selected


def plot_representative_niches(fig: plt.Figure, subspec, spatial: pd.DataFrame, sample_df: pd.DataFrame) -> None:
    sub = GridSpecFromSubplotSpec(1, 3, subplot_spec=subspec, wspace=0.08)
    reps = select_representative_components(sample_df)
    for i, (component, pseudotime, label) in enumerate(reps):
        ax = fig.add_subplot(sub[0, i])
        if i == 0:
            panel_letter(ax, "G")
        cells = spatial[
            spatial["pancreatic ductal epithelium_mask_component"].astype(str) == component
        ]
        if cells.empty:
            ax.axis("off")
            continue
        pad = 180
        xmin, xmax = cells["x"].min() - pad, cells["x"].max() + pad
        ymin, ymax = cells["y"].min() - pad, cells["y"].max() + pad
        local = spatial[
            spatial["x"].between(xmin, xmax) & spatial["y"].between(ymin, ymax)
        ].copy()
        local = local.sample(n=min(2_000, len(local)), random_state=31 + i)
        local_colors = local["Tier_A"].map(PHENOTYPE_COLORS).fillna("#c9c9c9")
        ax.scatter(local["x"], local["y"], c=local_colors, s=1.3, alpha=0.45, linewidths=0)
        ax.scatter(cells["x"], cells["y"], c="#111111", s=6, alpha=0.85, linewidths=0)
        ax.set_title(f"{label}\npt={pseudotime:.1f}", fontsize=8)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_color("#d0d0d0")


def plot_pooled_umap_by_sample(ax: plt.Axes, merged: pd.DataFrame) -> None:
    panel_letter(ax, "H")
    show = merged.sample(n=min(len(merged), 24_000), random_state=19)
    for sample_id, sdf in show.groupby("sample_id"):
        ax.scatter(
            sdf["UMAP1"],
            sdf["UMAP2"],
            s=1.3,
            linewidths=0,
            alpha=0.60,
            color=SAMPLE_COLORS.get(sample_id, "#999999"),
            label=sample_id,
            rasterized=True,
        )
    ax.set_title("pooled four-sample\nniche embedding")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.legend(frameon=False, markerscale=3, loc="best")


def plot_branch_occupancy(ax: plt.Axes, merged: pd.DataFrame) -> None:
    panel_letter(ax, "I")
    counts = merged.groupby(["major_branch", "sample_id"]).size().unstack(fill_value=0)
    counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).head(10).index]
    frac = counts.div(counts.sum(axis=1), axis=0)
    frac.plot(
        kind="barh",
        stacked=True,
        ax=ax,
        color=[SAMPLE_COLORS.get(c, "#999999") for c in frac.columns],
        width=0.8,
    )
    ax.set_title("branch occupancy\nacross four samples")
    ax.set_xlabel("fraction of branch niches")
    ax.set_ylabel("")
    ax.legend(frameon=False, title="", loc="lower right", ncol=1)


def plot_module_heatmap(ax: plt.Axes, merged: pd.DataFrame) -> None:
    panel_letter(ax, "J")
    score_cols = {
        "pdac_early_duct_anchor_score": "early duct",
        "pdac_panin_like_dysplasia_score": "PanIN-like",
        "pdac_invasion_desmoplasia_axis": "invasion/desmoplasia",
        "pdac_proliferation_axis": "proliferation",
        "pdac_dedifferentiation_axis": "dedifferentiation",
        "panin_validation__panin_grade_like_axis": "PanIN grade-like",
    }
    available = {k: v for k, v in score_cols.items() if k in merged.columns}
    tmp = merged.copy()
    tmp["pseudotime_decile"] = pd.qcut(
        tmp["pooled_pseudotime"], q=10, labels=False, duplicates="drop"
    )
    deciles = tmp.groupby("pseudotime_decile")[list(available)].median().rename(columns=available)
    deciles_z = zscore_frame(deciles, axis=0)
    sns.heatmap(
        deciles_z.T,
        ax=ax,
        cmap="vlag",
        center=0,
        cbar_kws={"label": "z-scored median"},
        linewidths=0.2,
    )
    ax.set_title("pathology modules\nacross pseudotime")
    ax.set_xlabel("pseudotime decile")
    ax.set_ylabel("")


def plot_microenvironment_trends(ax: plt.Axes, trends: pd.DataFrame) -> None:
    panel_letter(ax, "K")
    show = trends.sort_values("spearman_r", key=lambda s: s.abs(), ascending=False).head(8).copy()
    show["short_label"] = [shorten(x, 33) for x in show["label"]]
    colors = [PALETTE["teal"] if x >= 0 else PALETTE["orange"] for x in show["spearman_r"]]
    sns.barplot(
        data=show,
        x="spearman_r",
        y="short_label",
        hue="short_label",
        ax=ax,
        palette=dict(zip(show["short_label"], colors)),
        legend=False,
    )
    ax.axvline(0, color="#333333", lw=0.7)
    ax.set_title("microenvironment dynamics\nalong contextual pseudotime")
    ax.set_xlabel("Spearman r")
    ax.set_ylabel("")
    for i, row in enumerate(show.itertuples()):
        ax.text(
            row.spearman_r + (0.012 if row.spearman_r >= 0 else -0.012),
            i,
            f"{row.spearman_r:.2f}",
            ha="left" if row.spearman_r >= 0 else "right",
            va="center",
            fontsize=6.5,
        )


def write_summary_tables(merged: pd.DataFrame) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    sample_summary = (
        merged.groupby(["sample_id", "disease_group"])
        .size()
        .reset_index(name="n_niches")
        .sort_values(["disease_group", "sample_id"])
    )
    sample_summary.to_csv(TABLE_DIR / "figure_1_pseudotime_sample_summary.csv", index=False)
    branch_summary = (
        merged.groupby(["sample_id", "major_branch"])
        .size()
        .reset_index(name="n_niches")
        .sort_values(["sample_id", "n_niches"], ascending=[True, False])
    )
    branch_summary.to_csv(TABLE_DIR / "figure_1_pseudotime_branch_occupancy.csv", index=False)


def main() -> None:
    configure_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    merged, spatial, _panin, trends = load_inputs()
    sample_df = merged.loc[merged["sample_id"] == REP_SAMPLE].copy()
    write_summary_tables(merged)

    fig = plt.figure(figsize=(15.5, 15.2), constrained_layout=True)
    gs = fig.add_gridspec(
        4,
        4,
        height_ratios=[0.72, 1.18, 1.08, 1.15],
        width_ratios=[1.0, 1.0, 1.0, 1.0],
    )

    plot_workflow(fig.add_subplot(gs[0, :]))
    plot_spatial_phenotypes(fig.add_subplot(gs[1, 0]), spatial)
    plot_ductal_niche_map(fig.add_subplot(gs[1, 1]), spatial)
    plot_feature_blocks(fig.add_subplot(gs[1, 2]))
    plot_umap_pseudotime(
        fig.add_subplot(gs[1, 3]),
        sample_df,
        f"{REP_SAMPLE} niche embedding\n({len(sample_df):,} niches)",
        "E",
    )
    plot_tissue_pseudotime(fig.add_subplot(gs[2, 0]), spatial)
    plot_representative_niches(fig, gs[2, 1:3], spatial, sample_df)
    plot_pooled_umap_by_sample(fig.add_subplot(gs[2, 3]), merged)
    plot_branch_occupancy(fig.add_subplot(gs[3, 0]), merged)
    plot_module_heatmap(fig.add_subplot(gs[3, 1]), merged)
    plot_microenvironment_trends(fig.add_subplot(gs[3, 2:]), trends)

    fig.suptitle(
        "SpatioEv morphology/topology pseudotime reconstructs epithelial niche evolution",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    out = FIG_DIR / "figure_1_pseudotime_single_to_pooled.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
