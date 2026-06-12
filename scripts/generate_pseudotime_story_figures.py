"""Generate manuscript figures for the revised SpatioEv pseudotime story.

The figures follow the requested logic:

1. A worked single-sample PDAC example (34434_1) that introduces phenotypes,
   ductal niche identification, graph/niche features, pathology modules, PCA,
   UMAP, pseudotime, surrounding ecology, and cell-cell interaction dynamics.
2. A pooled four-sample multiplexed imaging atlas.
3. A Xenium transfer analysis using modality-adapted feature families.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpecFromSubplotSpec
from scipy.spatial import cKDTree
from statsmodels.nonparametric.smoothers_lowess import lowess


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "manuscript" / "figures"
TABLE_DIR = ROOT / "manuscript" / "analysis_tables"

REP_SAMPLE = "34434_1"
COMPONENT_KEY = "pancreatic ductal epithelium_mask_component"
XENIUM_COMPONENT_KEY = "xenium_ductal_epithelium_component"

SAMPLE_LABELS = {
    "40331_1": "normal pancreas",
    "34434_1": "PDAC 34434",
    "33694_1": "PDAC 33694",
    "35559_1": "PDAC 35559",
}

XENIUM_LABELS = {
    "normal_nondiseased_v1": "normal pancreas",
    "pdac_pancreas_v1": "PDAC pancreas",
    "pdac_addon_v1": "PDAC add-on",
    "pdac_io_v1": "PDAC invasion",
}

PHENOTYPE_COLORS = {
    "pancreatic ductal epithelium": "#2f6fbd",
    "Fibroblasts": "#2a9d8f",
    "Vimentin only mesenchyme": "#b07aa1",
    "Endothelial cells": "#59a14f",
    "pancreatic acinar epithelium": "#f28e2b",
    "T cells": "#d64f4f",
    "B lineage": "#edc948",
    "Muscularis externa": "#9c755f",
    "Vascular smooth muscle": "#76b7b2",
    "Nerves": "#8e6bbf",
    "noise": "#bab0ab",
}

SAMPLE_COLORS = {
    "40331_1": "#2f4858",
    "34434_1": "#33658a",
    "33694_1": "#f26419",
    "35559_1": "#7a5195",
}

XENIUM_SAMPLE_COLORS = {
    "normal_nondiseased_v1": "#2f4858",
    "pdac_pancreas_v1": "#33658a",
    "pdac_addon_v1": "#f26419",
    "pdac_io_v1": "#7a5195",
}

MODULE_COLUMNS = {
    "pdac_early_duct_anchor_score": "early duct",
    "pdac_panin_like_dysplasia_score": "PanIN-like",
    "pdac_invasion_desmoplasia_axis": "invasion/desmoplasia",
    "pdac_proliferation_axis": "proliferation",
    "pdac_dedifferentiation_axis": "dedifferentiation",
}

XENIUM_SCORE_COLUMNS = {
    "histology__normal_duct_like_score": "normal duct-like",
    "histology__adm_panin_like_score": "ADM/PanIN-like",
    "histology__desmoplastic_tumor_score": "desmoplastic tumor",
    "histology__immune_inflamed_score": "immune inflamed",
    "histology__immune_exclusion_score": "immune exclusion",
    "histology__gland_poor_undifferentiated_score": "gland-poor",
}


def configure_style() -> None:
    sns.set_theme(context="paper", style="white")
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "font.size": 7.4,
            "axes.titlesize": 8.1,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.0,
            "axes.linewidth": 0.62,
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
        fontsize=11,
        fontweight="bold",
        ha="right",
        va="top",
    )


def clean_axis(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)


def shorten(text: str, width: int = 28) -> str:
    value = str(text).replace("__", " ").replace("_", " ")
    return value if len(value) <= width else value[: width - 1] + "..."


def zscore(df: pd.DataFrame, axis: int = 0) -> pd.DataFrame:
    values = df.astype(float)
    if axis == 1:
        mean = values.mean(axis=1)
        std = values.std(axis=1, ddof=0).replace(0, np.nan)
        return values.sub(mean, axis=0).div(std, axis=0)
    return (values - values.mean()) / values.std(ddof=0).replace(0, np.nan)


def robust_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=seed)


def load_multiplexed() -> dict[str, pd.DataFrame]:
    base = ROOT / "data" / "combined_exp_2_3_4_5"
    pooled = pd.read_pickle(base / "pooled_niche_result_df.pkl")
    pooled_features = pd.read_pickle(base / "pooled_pathology_feature_df.pkl")
    spatial = pd.read_pickle(base / f"spatial_cells_auto_branch_n24_v3_with_epithelial_{REP_SAMPLE}.pkl")
    exp2_features = pd.read_pickle(base / "per_sample" / "exp_2_pathology_feature_table.pkl")
    exp2 = pooled.loc[pooled["sample_id"] == REP_SAMPLE].copy()
    feature_cols = [
        c
        for c in exp2_features.columns
        if c not in exp2.columns or c in {COMPONENT_KEY, "image_id"}
    ]
    exp2 = exp2.merge(exp2_features[feature_cols], on=[COMPONENT_KEY, "image_id"], how="left")
    interactions = pd.read_pickle(
        ROOT
        / "notebooks"
        / "results"
        / "trajectory_microenvironment_interactions"
        / "multiplexed_epithelial_niche_local_colocalization.pkl"
    )
    trends = pd.read_csv(
        ROOT
        / "notebooks"
        / "results"
        / "trajectory_microenvironment_interactions"
        / "tables"
        / "multiplexed_microenvironment_trends_contextual.csv"
    )
    interaction_trends = pd.read_csv(
        ROOT
        / "notebooks"
        / "results"
        / "trajectory_microenvironment_interactions"
        / "tables"
        / "multiplexed_epithelial_niche_colocalization_trends.csv"
    )
    return {
        "pooled": pooled,
        "pooled_features": pooled_features,
        "spatial": spatial,
        "exp2": exp2,
        "exp2_features": exp2_features,
        "interactions": interactions,
        "trends": trends,
        "interaction_trends": interaction_trends,
    }


def load_tier_annotations() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "exp_2" / "34434_1_annotation.csv", usecols=["Tier_A", "Tier_B"])


def load_xenium() -> dict[str, pd.DataFrame]:
    xenium = pd.read_pickle(ROOT / "data" / "xenium_pancreas_10x" / "pseudotime" / "xenium_pseudotime_result_df.pkl")
    feature_blocks = pd.read_csv(ROOT / "data" / "xenium_pancreas_10x" / "pseudotime" / "xenium_pseudotime_feature_blocks.csv")
    audit = pd.read_csv(ROOT / "data" / "xenium_pancreas_10x" / "xenium_dataset_audit.csv")
    tier_counts = pd.read_csv(ROOT / "data" / "xenium_pancreas_10x" / "xenium_tier_a_counts.csv")
    branch_biology = pd.read_csv(ROOT / "data" / "xenium_pancreas_10x" / "pseudotime" / "xenium_branch_biology_summary.csv")
    micro = pd.read_csv(
        ROOT
        / "notebooks"
        / "results"
        / "trajectory_microenvironment_interactions"
        / "tables"
        / "xenium_microenvironment_trends_sample_centered.csv"
    )
    lr = pd.read_csv(
        ROOT
        / "notebooks"
        / "results"
        / "trajectory_microenvironment_interactions"
        / "tables"
        / "xenium_lr_potential_trends.csv"
    )
    return {
        "xenium": xenium,
        "feature_blocks": feature_blocks,
        "audit": audit,
        "tier_counts": tier_counts,
        "branch_biology": branch_biology,
        "micro": micro,
        "lr": lr,
    }


def feature_block_summary(df: pd.DataFrame) -> pd.DataFrame:
    groups = [
        (
            "topology",
            lambda c: c.startswith("topology__"),
            "degree, clustering, bridges, skeleton leaves/branchpoints",
            "ductal connectedness, branching, fragmentation, tortuosity",
        ),
        (
            "geometry",
            lambda c: c.startswith("geometry__"),
            "area, perimeter, circularity, hull metrics, orientation",
            "gland compactness and architectural distortion",
        ),
        (
            "cell-graph state",
            lambda c: c.startswith("features__"),
            "marker/morphology summaries over graph-linked epithelial cells",
            "local epithelial phenotype and cytologic state",
        ),
        (
            "graph surroundings",
            lambda c: c.startswith("graph_surround__"),
            "hop counts, phenotype entropy, surround-minus-niche contrasts",
            "boundary exposure and microenvironment reach",
        ),
        (
            "surround composition",
            lambda c: c.startswith("surround_prop__") or c.startswith("surround__"),
            "fibroblast, immune, endothelial, mesenchymal proportions and markers",
            "desmoplasia, immune context, vascular and stromal remodeling",
        ),
        (
            "cell-state summaries",
            lambda c: c.startswith("state__"),
            "mean, median, dispersion, quantiles of cell features",
            "heterogeneity inside ductal epithelial niches",
        ),
        (
            "PDAC modules",
            lambda c: c.startswith("pdac_"),
            "early duct, PanIN-like, invasion/desmoplasia, proliferation",
            "pathology-inspired axes for trajectory interpretation",
        ),
    ]
    rows = []
    for name, matcher, examples, biology in groups:
        cols = [c for c in df.columns if matcher(c)]
        rows.append(
            {
                "feature_family": name,
                "n_features": len(cols),
                "examples": examples,
                "biological_readout": biology,
            }
        )
    return pd.DataFrame(rows)


def write_supporting_tables(mpx: dict[str, pd.DataFrame], xen: dict[str, pd.DataFrame]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    feature_block_summary(mpx["exp2_features"]).to_csv(
        TABLE_DIR / "supplementary_table_pseudotime_feature_blocks_34434.csv",
        index=False,
    )
    workflow_rows = [
        ("1", "Tier_A/Tier_B phenotype scaffold", "Broad tissue compartments and marker-refined subtypes anchor spatial interpretation."),
        ("2", "Connected ductal epithelial components", "Duct fragments, glands, and epithelial aggregates become the local niche unit."),
        ("3", "30 um cell graph", "Physical adjacency, boundary exposure, and local mixing are encoded as graph structure."),
        ("4", "Niche graph and surrounding hops", "Five-hop graph surroundings summarize stromal, immune, endothelial, and mesenchymal context."),
        ("5", "Morphology, pixel, geometry, topology features", "Cytologic atypia, polarity loss, texture heterogeneity, ductal shape, and branch complexity are quantified."),
        ("6", "PDAC pathology modules", "Signed feature groups become early duct, PanIN-like, invasion/desmoplasia, proliferation, and dedifferentiation scores."),
        ("7", "PCA/UMAP diagnostics and principal tree", "The state space is inspected and ordered with a rooted principal tree."),
        ("8", "Tissue back-projection and dynamics", "Pseudotime, branch, module, surrounding, and interaction changes are mapped back to real niches."),
    ]
    pd.DataFrame(workflow_rows, columns=["step", "operation", "biological_purpose"]).to_csv(
        TABLE_DIR / "supplementary_table_pseudotime_workflow_steps.csv",
        index=False,
    )
    xen["feature_blocks"].groupby("feature_block").agg(
        n_features=("feature", "size"),
        example_features=("feature", lambda s: "; ".join(s.head(4))),
    ).reset_index().to_csv(
        TABLE_DIR / "supplementary_table_xenium_feature_blocks.csv",
        index=False,
    )


def plot_spatial_tier_a(ax: plt.Axes, spatial: pd.DataFrame) -> None:
    panel_letter(ax, "A")
    show = robust_sample(spatial, 165_000, 101)
    colors = show["Tier_A"].map(PHENOTYPE_COLORS).fillna("#d0d0d0")
    ax.scatter(show["x"], show["y"], c=colors, s=0.10, alpha=0.70, linewidths=0, rasterized=True)
    ax.set_title("34434 phenotype scaffold\nTier_A spatial map")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    handles = []
    for label in [
        "pancreatic ductal epithelium",
        "Fibroblasts",
        "T cells",
        "B lineage",
        "Endothelial cells",
        "Vimentin only mesenchyme",
    ]:
        handles.append(
            mpl.lines.Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=PHENOTYPE_COLORS[label],
                markersize=3.2,
                label=shorten(label, 18),
            )
        )
    ax.legend(handles=handles, loc="lower left", frameon=False, ncol=1, handletextpad=0.2)


def plot_tier_b_heatmap(ax: plt.Axes, ann: pd.DataFrame) -> None:
    panel_letter(ax, "B")
    top_a = ann["Tier_A"].value_counts().head(8).index
    top_b = ann["Tier_B"].value_counts().head(18).index
    table = pd.crosstab(ann.loc[ann["Tier_B"].isin(top_b), "Tier_B"], ann.loc[ann["Tier_B"].isin(top_b), "Tier_A"])
    table = table.reindex(index=top_b, columns=top_a).fillna(0)
    heat = zscore(np.log1p(table), axis=1)
    heat.index = [shorten(x, 34) for x in heat.index]
    heat.columns = [shorten(x, 16) for x in heat.columns]
    sns.heatmap(
        heat,
        ax=ax,
        cmap="vlag",
        center=0,
        linewidths=0.1,
        cbar_kws={"label": "row-z log count"},
    )
    ax.set_title("Tier_B refinement within\nmajor tissue compartments")
    ax.set_xlabel("Tier_A")
    ax.set_ylabel("Tier_B")


def select_components(exp2: pd.DataFrame) -> list[tuple[str, str, float]]:
    candidates = exp2.loc[exp2["n_cells"] >= 35].copy()
    selected = []
    for label, quantile in [("early organized duct", 0.12), ("late desmoplastic niche", 0.88)]:
        target = candidates["pooled_pseudotime"].quantile(quantile)
        row = candidates.loc[(candidates["pooled_pseudotime"] - target).abs().idxmin()]
        selected.append((str(row[COMPONENT_KEY]), label, float(row["pooled_pseudotime"])))
    return selected


def draw_component_example(ax: plt.Axes, spatial: pd.DataFrame, component: str, title: str, pseudotime: float, letter: str | None = None) -> None:
    if letter:
        panel_letter(ax, letter)
    comp = spatial.loc[spatial[COMPONENT_KEY].astype(str) == component].copy()
    if comp.empty:
        ax.axis("off")
        return
    pad = 230
    xmin, xmax = comp["x"].min() - pad, comp["x"].max() + pad
    ymin, ymax = comp["y"].min() - pad, comp["y"].max() + pad
    local = spatial.loc[spatial["x"].between(xmin, xmax) & spatial["y"].between(ymin, ymax)].copy()
    local = robust_sample(local, 4500, 211)
    ax.scatter(
        local["x"],
        local["y"],
        c=local["Tier_A"].map(PHENOTYPE_COLORS).fillna("#c9c9c9"),
        s=1.1,
        alpha=0.38,
        linewidths=0,
        rasterized=True,
    )
    coords = comp[["x", "y"]].to_numpy()
    if len(coords) > 1:
        tree = cKDTree(coords)
        pairs = np.array(list(tree.query_pairs(30 / 0.325)))
        if len(pairs) > 600:
            rng = np.random.default_rng(27)
            pairs = pairs[rng.choice(len(pairs), size=600, replace=False)]
        for i, j in pairs:
            ax.plot(
                [coords[i, 0], coords[j, 0]],
                [coords[i, 1], coords[j, 1]],
                color="#111111",
                alpha=0.15,
                lw=0.45,
                zorder=1,
            )
    ax.scatter(comp["x"], comp["y"], c="#111111", s=5.2, alpha=0.82, linewidths=0, zorder=3)
    ax.set_title(f"{title}\n{len(comp):,} ductal cells, pt={pseudotime:.1f}")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#cccccc")
        spine.set_linewidth(0.5)


def plot_ductal_examples(fig: plt.Figure, subspec, spatial: pd.DataFrame, exp2: pd.DataFrame) -> None:
    sub = GridSpecFromSubplotSpec(1, 2, subplot_spec=subspec, wspace=0.05)
    for i, (component, label, pt) in enumerate(select_components(exp2)):
        ax = fig.add_subplot(sub[0, i])
        draw_component_example(ax, spatial, component, label, pt, "C" if i == 0 else None)
    fig.text(0.51, 0.665, "ductal niche examples: connected epithelial component + local cell graph", ha="center", fontsize=7)


def plot_feature_blocks(ax: plt.Axes, feature_df: pd.DataFrame) -> None:
    panel_letter(ax, "D")
    blocks = feature_block_summary(feature_df)
    blocks = blocks.sort_values("n_features", ascending=True)
    colors = sns.color_palette("crest", n_colors=len(blocks))
    ax.barh(blocks["feature_family"], blocks["n_features"], color=colors, edgecolor="none")
    ax.set_title("cell graph + niche graph\nfeature families")
    ax.set_xlabel("features entering summaries")
    ax.set_ylabel("")
    for i, value in enumerate(blocks["n_features"]):
        ax.text(value + max(blocks["n_features"]) * 0.02, i, f"{int(value)}", va="center", fontsize=6.2)
    clean_axis(ax)


def plot_module_deciles(ax: plt.Axes, df: pd.DataFrame, letter: str, title: str, ptime: str = "pooled_pseudotime") -> None:
    panel_letter(ax, letter)
    cols = {k: v for k, v in MODULE_COLUMNS.items() if k in df.columns}
    tmp = df.dropna(subset=[ptime]).copy()
    tmp["decile"] = pd.qcut(tmp[ptime], q=10, labels=False, duplicates="drop")
    med = tmp.groupby("decile")[list(cols)].median().rename(columns=cols)
    sns.heatmap(
        zscore(med, axis=0).T,
        ax=ax,
        cmap="vlag",
        center=0,
        linewidths=0.15,
        cbar_kws={"label": "z-scored median"},
    )
    ax.set_title(title)
    ax.set_xlabel("pseudotime decile")
    ax.set_ylabel("")


def plot_embedding(ax: plt.Axes, df: pd.DataFrame, x: str, y: str, color: str, cmap: str, letter: str, title: str, label: str) -> None:
    panel_letter(ax, letter)
    show = robust_sample(df.dropna(subset=[x, y, color]), 18_000, 301)
    sc = ax.scatter(show[x], show[y], c=show[color], cmap=cmap, s=1.4, alpha=0.72, linewidths=0, rasterized=True)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.01, label=label)


def plot_tissue_pseudotime(ax: plt.Axes, spatial: pd.DataFrame, letter: str, title: str) -> None:
    panel_letter(ax, letter)
    bg = robust_sample(spatial, 90_000, 401)
    ax.scatter(bg["x"], bg["y"], c="#d2d2d2", s=0.08, alpha=0.22, linewidths=0, rasterized=True)
    niche = spatial.loc[spatial["has_pooled_niche"].fillna(False)].dropna(subset=["pooled_pseudotime"]).copy()
    show = robust_sample(niche, 130_000, 402)
    sc = ax.scatter(show["x"], show["y"], c=show["pooled_pseudotime"], cmap="turbo", s=0.20, alpha=0.86, linewidths=0, rasterized=True)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.01, label="pseudotime")


def plot_score_loess(ax: plt.Axes, df: pd.DataFrame, letter: str) -> None:
    panel_letter(ax, letter)
    scores = {
        "pdac_early_duct_anchor_score": "early duct",
        "pdac_panin_like_dysplasia_score": "PanIN-like",
        "pdac_invasion_desmoplasia_axis": "invasion/desmoplasia",
        "pdac_dedifferentiation_axis": "dedifferentiation",
    }
    palette = {
        "early duct": "#2f4858",
        "PanIN-like": "#e17c05",
        "invasion/desmoplasia": "#8c1d40",
        "dedifferentiation": "#6a4c93",
    }
    for col, label in scores.items():
        if col not in df.columns:
            continue
        sub = df[["pooled_pseudotime", col]].dropna().sort_values("pooled_pseudotime")
        y = (sub[col] - sub[col].median()) / sub[col].std(ddof=0)
        line = lowess(y, sub["pooled_pseudotime"], frac=0.24, return_sorted=True)
        ax.plot(line[:, 0], line[:, 1], lw=1.7, label=label, color=palette[label])
    branch_counts = df["major_branch"].value_counts().head(5).index.tolist()
    ymin, ymax = ax.get_ylim()
    rug_y = ymin + (ymax - ymin) * 0.03
    branch_palette = dict(zip(branch_counts, sns.color_palette("tab10", n_colors=len(branch_counts))))
    for branch in branch_counts:
        sub = df.loc[df["major_branch"] == branch, "pooled_pseudotime"].dropna()
        ax.scatter(sub.sample(min(120, len(sub)), random_state=42), np.full(min(120, len(sub)), rug_y), s=2.5, color=branch_palette[branch], alpha=0.55, linewidths=0)
    ax.set_title("pathology scores change\nalong pseudotime with branch rug")
    ax.set_xlabel("pseudotime")
    ax.set_ylabel("LOESS z-score")
    ax.legend(frameon=False, loc="best")


def plot_surrounding_loess(ax: plt.Axes, df: pd.DataFrame, letter: str, title: str = "surrounding niche context") -> None:
    panel_letter(ax, letter)
    cols = {
        "surround_prop__Fibroblasts": "fibroblast fraction",
        "surround__Fibroblasts__FAP_expr_z__mean": "fibroblast FAP",
        "surround_prop__T_cells": "T-cell fraction",
        "surround_prop__B_lineage": "B-lineage fraction",
        "surround_prop__Endothelial_cells": "endothelial fraction",
        "surround_prop__Vimentin_only_mesenchyme": "VIM+ mesenchyme",
    }
    colors = sns.color_palette("Set2", n_colors=len(cols))
    for (col, label), color in zip(cols.items(), colors):
        if col not in df.columns:
            continue
        sub = df[["pooled_pseudotime", col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) < 25:
            continue
        y = (sub[col] - sub[col].median()) / sub[col].std(ddof=0)
        line = lowess(y, sub["pooled_pseudotime"], frac=0.27, return_sorted=True)
        ax.plot(line[:, 0], line[:, 1], lw=1.4, label=label, color=color)
    ax.axhline(0, lw=0.6, color="#777777")
    ax.set_title(f"{title}\nLOESS over pseudotime")
    ax.set_xlabel("pseudotime")
    ax.set_ylabel("z-scored value")
    ax.legend(frameon=False, loc="best", ncol=1)


def plot_interaction_loess(ax: plt.Axes, interactions: pd.DataFrame, letter: str, sample_id: str | None = None) -> None:
    panel_letter(ax, letter)
    df = interactions.copy()
    if sample_id is not None:
        df = df.loc[df["sample_id"] == sample_id].copy()
    cols = {
        "coloc__ductal_to__Fibroblasts__r30__fraction_epithelial_cells_with_target_neighbor": "near fibroblasts",
        "coloc__ductal_to__T_cells__r30__fraction_epithelial_cells_with_target_neighbor": "near T cells",
        "coloc__ductal_to__B_lineage__r30__fraction_epithelial_cells_with_target_neighbor": "near B lineage",
        "coloc__ductal_to__Endothelial_cells__r30__fraction_epithelial_cells_with_target_neighbor": "near endothelium",
    }
    colors = ["#2a9d8f", "#d64f4f", "#bc9b1f", "#59a14f"]
    for (col, label), color in zip(cols.items(), colors):
        sub = df[["pooled_pseudotime", col]].dropna()
        if len(sub) < 25:
            continue
        line = lowess(sub[col], sub["pooled_pseudotime"], frac=0.30, return_sorted=True)
        ax.plot(line[:, 0], line[:, 1], lw=1.45, label=label, color=color)
    ax.set_title("ductal cell-cell contact\nwithin 30 um")
    ax.set_xlabel("pseudotime")
    ax.set_ylabel("fraction of ductal cells")
    ax.legend(frameon=False, loc="best")


def plot_sample_bar(ax: plt.Axes, pooled: pd.DataFrame, letter: str) -> None:
    panel_letter(ax, letter)
    counts = pooled.groupby(["sample_id", "disease_group"]).size().reset_index(name="n_niches")
    counts["label"] = counts["sample_id"].map(SAMPLE_LABELS)
    sns.barplot(data=counts, x="n_niches", y="label", hue="disease_group", dodge=False, ax=ax, palette={"NormalPancreas": "#2f4858", "PDAC": "#a63d40"})
    ax.set_title("pooled atlas\n46,574 ductal niches")
    ax.set_xlabel("niches")
    ax.set_ylabel("")
    ax.legend(frameon=False, title="")
    clean_axis(ax)


def plot_pooled_umap_samples(ax: plt.Axes, pooled: pd.DataFrame, letter: str) -> None:
    panel_letter(ax, letter)
    show = robust_sample(pooled.dropna(subset=["UMAP1", "UMAP2"]), 28_000, 501)
    for sample_id, sdf in show.groupby("sample_id"):
        ax.scatter(
            sdf["UMAP1"],
            sdf["UMAP2"],
            s=1.2,
            alpha=0.58,
            linewidths=0,
            rasterized=True,
            color=SAMPLE_COLORS.get(sample_id, "#999999"),
            label=SAMPLE_LABELS.get(sample_id, sample_id),
        )
    ax.set_title("pooled niche UMAP\ncolored by sample")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.legend(frameon=False, markerscale=3, loc="best")


def plot_branch_occupancy(ax: plt.Axes, pooled: pd.DataFrame, letter: str) -> None:
    panel_letter(ax, letter)
    counts = pooled.groupby(["major_branch", "sample_id"]).size().unstack(fill_value=0)
    counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).head(11).index]
    frac = counts.div(counts.sum(axis=1), axis=0)
    frac.rename(columns=SAMPLE_LABELS).plot(
        kind="barh",
        stacked=True,
        ax=ax,
        width=0.82,
        color=[SAMPLE_COLORS.get(c, "#999999") for c in counts.columns],
    )
    ax.set_title("branch occupancy\nnormal pancreas versus PDAC")
    ax.set_xlabel("fraction of branch niches")
    ax.set_ylabel("")
    ax.legend(frameon=False, title="", loc="lower right")


def plot_branch_module_heatmap(ax: plt.Axes, pooled: pd.DataFrame, letter: str) -> None:
    panel_letter(ax, letter)
    cols = {k: v for k, v in MODULE_COLUMNS.items() if k in pooled.columns}
    branch_order = pooled["major_branch"].value_counts().head(10).index
    med = pooled.loc[pooled["major_branch"].isin(branch_order)].groupby("major_branch")[list(cols)].median().rename(columns=cols)
    med = med.loc[branch_order]
    sns.heatmap(
        zscore(med, axis=0).T,
        ax=ax,
        cmap="vlag",
        center=0,
        linewidths=0.15,
        cbar_kws={"label": "branch z-score"},
    )
    ax.set_title("branch-enriched\npathology modules")
    ax.set_xlabel("major branch")
    ax.set_ylabel("")


def plot_tissue_maps_four_samples(fig: plt.Figure, subspec, letter: str) -> None:
    sub = GridSpecFromSubplotSpec(2, 2, subplot_spec=subspec, wspace=0.02, hspace=0.10)
    base = ROOT / "data" / "combined_exp_2_3_4_5"
    for i, sample_id in enumerate(["40331_1", "34434_1", "33694_1", "35559_1"]):
        ax = fig.add_subplot(sub[i // 2, i % 2])
        if i == 0:
            panel_letter(ax, letter)
        path = base / f"spatial_cells_auto_branch_n24_v3_with_epithelial_{sample_id}.pkl"
        spatial = pd.read_pickle(path)
        bg = robust_sample(spatial, 35_000, 601 + i)
        ax.scatter(bg["x"], bg["y"], c="#d6d6d6", s=0.05, alpha=0.18, linewidths=0, rasterized=True)
        niche = spatial.loc[spatial["has_pooled_niche"].fillna(False)].dropna(subset=["pooled_pseudotime"])
        show = robust_sample(niche, 50_000, 611 + i)
        ax.scatter(show["x"], show["y"], c=show["pooled_pseudotime"], cmap="turbo", s=0.12, alpha=0.82, linewidths=0, rasterized=True)
        ax.set_title(SAMPLE_LABELS.get(sample_id, sample_id), fontsize=7.1)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])


def plot_trend_bars(ax: plt.Axes, trends: pd.DataFrame, letter: str, title: str, top_n: int = 8) -> None:
    panel_letter(ax, letter)
    show = trends.sort_values("spearman_r", key=lambda s: s.abs(), ascending=False).head(top_n).copy()
    label_col = "label" if "label" in show.columns else "feature"
    show["short_label"] = [shorten(x, 34) for x in show[label_col]]
    colors = ["#2a9d8f" if r >= 0 else "#e17c05" for r in show["spearman_r"]]
    sns.barplot(
        data=show,
        x="spearman_r",
        y="short_label",
        hue="short_label",
        palette=dict(zip(show["short_label"], colors)),
        legend=False,
        ax=ax,
    )
    ax.axvline(0, color="#333333", lw=0.7)
    ax.set_title(title)
    ax.set_xlabel("Spearman r")
    ax.set_ylabel("")
    for i, row in enumerate(show.itertuples()):
        offset = 0.012 if row.spearman_r >= 0 else -0.012
        ax.text(row.spearman_r + offset, i, f"{row.spearman_r:.2f}", va="center", ha="left" if row.spearman_r >= 0 else "right", fontsize=6.1)


def plot_xenium_audit(ax: plt.Axes, audit: pd.DataFrame) -> None:
    panel_letter(ax, "A")
    df = audit.copy()
    sample_col = "sample_id" if "sample_id" in df.columns else "dataset"
    cell_col = "n_cells" if "n_cells" in df.columns else "n_cells_matrix"
    gene_col = "n_genes" if "n_genes" in df.columns else "genes_detected"
    df["label"] = df[sample_col].map(XENIUM_LABELS).fillna(df[sample_col])
    ax.barh(df["label"], df[cell_col], color=[XENIUM_SAMPLE_COLORS.get(s, "#999999") for s in df[sample_col]])
    ax.set_title("Xenium data audit\ncells per sample")
    ax.set_xlabel("cells")
    ax.set_ylabel("")
    ax2 = ax.twiny()
    ax2.plot(df[gene_col], df["label"], marker="o", color="#111111", lw=1.0, ms=3.0)
    ax2.set_xlabel("panel genes")
    clean_axis(ax)


def plot_xenium_tier_counts(ax: plt.Axes, tier_counts: pd.DataFrame) -> None:
    panel_letter(ax, "B")
    df = tier_counts.copy()
    sample_col = "sample_id"
    tier_col = "Tier_A"
    count_col = "n_cells" if "n_cells" in df.columns else ("n" if "n" in df.columns else "count")
    top = df.groupby(tier_col)[count_col].sum().sort_values(ascending=False).head(10).index
    heat = df.loc[df[tier_col].isin(top)].pivot_table(index=tier_col, columns=sample_col, values=count_col, fill_value=0, aggfunc="sum")
    heat = heat.div(heat.sum(axis=0), axis=1)
    heat.index = [shorten(x, 30) for x in heat.index]
    heat.columns = [XENIUM_LABELS.get(x, x) for x in heat.columns]
    sns.heatmap(heat, ax=ax, cmap="YlGnBu", linewidths=0.1, cbar_kws={"label": "fraction of cells"})
    ax.set_title("Xenium Tier_A annotation\ncomposition")
    ax.set_xlabel("")
    ax.set_ylabel("")


def plot_xenium_feature_blocks(ax: plt.Axes, feature_blocks: pd.DataFrame) -> None:
    panel_letter(ax, "C")
    counts = feature_blocks["feature_block"].value_counts().sort_values()
    ax.barh(counts.index.map(lambda x: shorten(x, 28)), counts.values, color=sns.color_palette("flare", n_colors=len(counts)))
    ax.set_title("modality-adapted\nXenium niche features")
    ax.set_xlabel("selected features")
    ax.set_ylabel("")
    for i, v in enumerate(counts.values):
        ax.text(v + counts.max() * 0.02, i, f"{int(v)}", va="center", fontsize=6.2)
    clean_axis(ax)


def plot_xenium_umap(ax: plt.Axes, xenium: pd.DataFrame, color_by: str, letter: str, title: str, categorical: bool = False) -> None:
    panel_letter(ax, letter)
    show = robust_sample(xenium.dropna(subset=["UMAP1_sample_centered", "UMAP2_sample_centered"]), 5_074, 701)
    if categorical:
        for sample_id, sdf in show.groupby(color_by):
            ax.scatter(
                sdf["UMAP1_sample_centered"],
                sdf["UMAP2_sample_centered"],
                s=5.0,
                alpha=0.68,
                linewidths=0,
                color=XENIUM_SAMPLE_COLORS.get(sample_id, "#999999"),
                label=XENIUM_LABELS.get(sample_id, sample_id),
            )
        ax.legend(frameon=False, markerscale=1.6, loc="best")
    else:
        sc = ax.scatter(
            show["UMAP1_sample_centered"],
            show["UMAP2_sample_centered"],
            c=show[color_by],
            cmap="turbo",
            s=5.0,
            alpha=0.72,
            linewidths=0,
        )
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.01, label="sample-centered pseudotime")
    ax.set_title(title)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")


def plot_xenium_spatial_maps(fig: plt.Figure, subspec, xenium: pd.DataFrame) -> None:
    sub = GridSpecFromSubplotSpec(2, 2, subplot_spec=subspec, wspace=0.03, hspace=0.10)
    base = ROOT / "data" / "xenium_pancreas_10x" / "niche_features"
    niche_pt = xenium[[XENIUM_COMPONENT_KEY, "xenium_pseudotime_sample_centered_norm"]].drop_duplicates()
    for i, sample_id in enumerate(["normal_nondiseased_v1", "pdac_pancreas_v1", "pdac_addon_v1", "pdac_io_v1"]):
        ax = fig.add_subplot(sub[i // 2, i % 2])
        if i == 0:
            panel_letter(ax, "F")
        try:
            import anndata as ad

            adata = ad.read_h5ad(base / f"{sample_id}_with_niches.h5ad", backed="r")
            obs = adata.obs[[XENIUM_COMPONENT_KEY, "x_centroid", "y_centroid", "Tier_A"]].copy()
            adata.file.close()
            obs = obs.merge(niche_pt, on=XENIUM_COMPONENT_KEY, how="left")
            bg = robust_sample(obs, 45_000, 801 + i)
            ax.scatter(bg["x_centroid"], bg["y_centroid"], c="#d6d6d6", s=0.06, alpha=0.18, linewidths=0, rasterized=True)
            epi = obs.dropna(subset=["xenium_pseudotime_sample_centered_norm"])
            show = robust_sample(epi, 45_000, 811 + i)
            ax.scatter(
                show["x_centroid"],
                show["y_centroid"],
                c=show["xenium_pseudotime_sample_centered_norm"],
                cmap="turbo",
                s=0.10,
                alpha=0.78,
                linewidths=0,
                rasterized=True,
            )
        except Exception as exc:  # pragma: no cover - plotting fallback
            ax.text(0.5, 0.5, f"spatial map unavailable\n{exc}", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(XENIUM_LABELS.get(sample_id, sample_id), fontsize=7.0)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])


def plot_xenium_score_heatmap(ax: plt.Axes, xenium: pd.DataFrame) -> None:
    panel_letter(ax, "G")
    cols = {k: v for k, v in XENIUM_SCORE_COLUMNS.items() if k in xenium.columns}
    tmp = xenium.dropna(subset=["xenium_pseudotime_sample_centered_norm"]).copy()
    tmp["decile"] = pd.qcut(tmp["xenium_pseudotime_sample_centered_norm"], q=10, labels=False, duplicates="drop")
    med = tmp.groupby("decile")[list(cols)].median().rename(columns=cols)
    sns.heatmap(zscore(med, axis=0).T, ax=ax, cmap="vlag", center=0, linewidths=0.1, cbar_kws={"label": "z-scored median"})
    ax.set_title("Xenium biological programs\nacross pseudotime")
    ax.set_xlabel("sample-centered pseudotime decile")
    ax.set_ylabel("")


def plot_xenium_branch_heatmap(ax: plt.Axes, branch_biology: pd.DataFrame) -> None:
    panel_letter(ax, "H")
    cols = [
        "histology__normal_duct_like_score__z_enrichment",
        "histology__adm_panin_like_score__z_enrichment",
        "histology__desmoplastic_tumor_score__z_enrichment",
        "histology__immune_inflamed_score__z_enrichment",
        "histology__immune_exclusion_score__z_enrichment",
        "histology__gland_poor_undifferentiated_score__z_enrichment",
    ]
    labels = ["normal duct", "ADM/PanIN", "desmoplastic", "immune inflamed", "immune excluded", "gland-poor"]
    df = branch_biology.set_index("branch")[cols].rename(columns=dict(zip(cols, labels))).T
    sns.heatmap(df, ax=ax, cmap="vlag", center=0, linewidths=0.1, cbar_kws={"label": "z-enrichment"})
    ax.set_title("branch biology\nmodule enrichment")
    ax.set_xlabel("")
    ax.set_ylabel("")


def plot_xenium_lr_bars(ax: plt.Axes, lr: pd.DataFrame) -> None:
    panel_letter(ax, "I")
    show = lr.sort_values("spearman_r", key=lambda s: s.abs(), ascending=False).head(7).copy()
    label_col = "label" if "label" in show.columns else "feature"
    show["short_label"] = [shorten(x, 36) for x in show[label_col]]
    colors = ["#2a9d8f" if r >= 0 else "#e17c05" for r in show["spearman_r"]]
    sns.barplot(data=show, x="spearman_r", y="short_label", hue="short_label", palette=dict(zip(show["short_label"], colors)), legend=False, ax=ax)
    ax.axvline(0, color="#333333", lw=0.7)
    ax.set_title("targeted ligand-receptor\npotential over pseudotime")
    ax.set_xlabel("Spearman r")
    ax.set_ylabel("")


def make_figure_1(mpx: dict[str, pd.DataFrame], ann: pd.DataFrame) -> Path:
    fig = plt.figure(figsize=(16.6, 14.6), constrained_layout=True)
    gs = fig.add_gridspec(3, 4, height_ratios=[1.05, 1.0, 1.0], width_ratios=[1.05, 1.04, 1.05, 1.0])
    plot_spatial_tier_a(fig.add_subplot(gs[0, 0]), mpx["spatial"])
    plot_tier_b_heatmap(fig.add_subplot(gs[0, 1]), ann)
    plot_ductal_examples(fig, gs[0, 2], mpx["spatial"], mpx["exp2"])
    plot_feature_blocks(fig.add_subplot(gs[0, 3]), mpx["exp2_features"])
    plot_module_deciles(fig.add_subplot(gs[1, 0]), mpx["exp2"], "E", "PDAC pathology modules\nin 34434")
    plot_embedding(fig.add_subplot(gs[1, 1]), mpx["exp2"], "PC1", "PC2", "pooled_pseudotime", "turbo", "F", "PCA diagnostic\ncolored by pseudotime", "pseudotime")
    plot_embedding(fig.add_subplot(gs[1, 2]), mpx["exp2"], "UMAP1", "UMAP2", "pooled_pseudotime", "turbo", "G", "UMAP diagnostic\ncolored by pseudotime", "pseudotime")
    plot_tissue_pseudotime(fig.add_subplot(gs[1, 3]), mpx["spatial"], "H", "epithelial niches\ncolored by pseudotime")
    plot_score_loess(fig.add_subplot(gs[2, 0:2]), mpx["exp2"], "I")
    plot_surrounding_loess(fig.add_subplot(gs[2, 2]), mpx["exp2"], "J", "surrounding-cell programs")
    plot_interaction_loess(fig.add_subplot(gs[2, 3]), mpx["interactions"], "K", sample_id=REP_SAMPLE)
    fig.suptitle(
        "Figure 1. A worked PDAC sample introduces morphology/topology epithelial niche pseudotime",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    out = FIG_DIR / "figure_1_34434_pseudotime_workflow.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def make_figure_2(mpx: dict[str, pd.DataFrame]) -> Path:
    pooled = mpx["pooled"]
    fig = plt.figure(figsize=(16.4, 14.1), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[0.95, 1.12, 1.05], width_ratios=[1.0, 1.05, 1.05])
    plot_sample_bar(fig.add_subplot(gs[0, 0]), pooled, "A")
    plot_pooled_umap_samples(fig.add_subplot(gs[0, 1]), pooled, "B")
    plot_embedding(fig.add_subplot(gs[0, 2]), pooled, "UMAP1", "UMAP2", "pooled_pseudotime", "turbo", "C", "pooled trajectory\ncolored by pseudotime", "pseudotime")
    plot_branch_occupancy(fig.add_subplot(gs[1, 0]), pooled, "D")
    plot_branch_module_heatmap(fig.add_subplot(gs[1, 1]), pooled, "E")
    plot_tissue_maps_four_samples(fig, gs[1, 2], "F")
    plot_module_deciles(fig.add_subplot(gs[2, 0]), pooled, "G", "pooled pathology modules\nacross pseudotime")
    plot_trend_bars(fig.add_subplot(gs[2, 1]), mpx["trends"], "H", "pooled surrounding-cell\ntrend tests")
    plot_trend_bars(fig.add_subplot(gs[2, 2]), mpx["interaction_trends"], "I", "pooled ductal interaction\ntrend tests")
    fig.suptitle(
        "Figure 2. Pooled four-sample analysis separates conserved and sample-specific ductal niche programs",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    out = FIG_DIR / "figure_2_pooled_multiplexed_pseudotime.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def make_figure_3(xen: dict[str, pd.DataFrame]) -> Path:
    fig = plt.figure(figsize=(16.4, 14.3), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[0.98, 1.18, 1.08], width_ratios=[1.0, 1.05, 1.05])
    plot_xenium_audit(fig.add_subplot(gs[0, 0]), xen["audit"])
    plot_xenium_tier_counts(fig.add_subplot(gs[0, 1]), xen["tier_counts"])
    plot_xenium_feature_blocks(fig.add_subplot(gs[0, 2]), xen["feature_blocks"])
    plot_xenium_umap(fig.add_subplot(gs[1, 0]), xen["xenium"], "sample_id", "D", "Xenium niche UMAP\ncolored by sample", categorical=True)
    plot_xenium_umap(fig.add_subplot(gs[1, 1]), xen["xenium"], "xenium_pseudotime_sample_centered_norm", "E", "Xenium niche UMAP\nsample-centered pseudotime")
    plot_xenium_spatial_maps(fig, gs[1, 2], xen["xenium"])
    plot_xenium_score_heatmap(fig.add_subplot(gs[2, 0]), xen["xenium"])
    plot_xenium_branch_heatmap(fig.add_subplot(gs[2, 1]), xen["branch_biology"])
    plot_trend_bars(fig.add_subplot(gs[2, 2]), xen["micro"], "I", "Xenium microenvironment\ntrend tests")
    fig.suptitle(
        "Figure 3. Xenium transfers SpatioEv niche pseudotime to spatial transcriptomics",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    out = FIG_DIR / "figure_3_xenium_pseudotime_transfer.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # A compact ligand-receptor panel is useful in the supplement and for reuse.
    fig_lr, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    plot_xenium_lr_bars(ax, xen["lr"])
    fig_lr.savefig(FIG_DIR / "supplementary_xenium_lr_pseudotime_trends.png", bbox_inches="tight", facecolor="white")
    plt.close(fig_lr)
    return out


def main() -> None:
    configure_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    mpx = load_multiplexed()
    xen = load_xenium()
    ann = load_tier_annotations()
    write_supporting_tables(mpx, xen)
    outputs = [make_figure_1(mpx, ann), make_figure_2(mpx), make_figure_3(xen)]
    for out in outputs:
        print(out)


if __name__ == "__main__":
    main()
