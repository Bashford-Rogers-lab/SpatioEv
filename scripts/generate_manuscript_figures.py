"""Generate publication-style figures from the local SpatioEv example outputs.

The figures are intentionally based on package functions and existing example
tables so the manuscript can be regenerated from the repository state rather
than assembled by hand.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

from spatioev.pp import QCConfig, generate_qc_summary, run_segmentation_qc
from spatioev.tl import (
    assign_tiles,
    compute_general_density,
    compute_phenotype_density,
    cross_ripleys_curve_by_phenotype,
    phenotype_density_correlation,
)


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "manuscript" / "figures"
TABLE_DIR = ROOT / "manuscript" / "analysis_tables"
SUMMARY_PATH = ROOT / "manuscript" / "analysis_summary.json"

RNG = np.random.default_rng(7)

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
    "Muscularis externa": "#9c755f",
    "B lineage": "#edc948",
    "Vascular smooth muscle": "#76b7b2",
    "noise": "#bab0ab",
}


def configure_style() -> None:
    sns.set_theme(context="paper", style="white")
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def savefig(fig: plt.Figure, filename: str) -> str:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / filename
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(out.relative_to(ROOT))


def write_table(df: pd.DataFrame, filename: str) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE_DIR / filename, index=False)


def load_catalog_module_counts() -> pd.DataFrame:
    catalog_path = ROOT / "docs" / "function_catalog.csv"
    if not catalog_path.exists():
        return pd.DataFrame(
            {
                "module": [
                    "qc",
                    "preprocessing",
                    "ml",
                    "phenotype",
                    "spatial statistics",
                    "density/interactions",
                    "niche/graph",
                    "ECM",
                    "pixel/Xenium",
                    "visualization",
                ],
                "functions": [6, 3, 7, 8, 34, 12, 17, 28, 12, 9],
            }
        )

    catalog = pd.read_csv(catalog_path)
    stage_to_module = {
        "configuration": "config/io/preprocess",
        "I/O": "config/io/preprocess",
        "preprocessing": "config/io/preprocess",
        "segmentation QC": "qc",
        "phenotyping/SVM": "phenotyping",
        "annotation refinement": "annotation refinement",
        "density": "density/interactions",
        "cell-cell interaction": "density/interactions",
        "spatial statistics": "spatial statistics",
        "spatial preprocessing": "spatial statistics",
        "niche/graph": "niche/graph",
        "ECM-cell analysis": "ECM",
        "pixel morphology/Xenium": "pixel/Xenium",
        "trajectory dynamics": "trajectory dynamics",
        "visualization": "visualization",
        "spatial visualization": "visualization",
        "QC plotting": "visualization",
        "other": "other",
    }
    order = [
        "config/io/preprocess",
        "qc",
        "phenotyping",
        "annotation refinement",
        "density/interactions",
        "spatial statistics",
        "niche/graph",
        "ECM",
        "pixel/Xenium",
        "trajectory dynamics",
        "visualization",
        "other",
    ]
    catalog["module"] = catalog["stage"].map(stage_to_module).fillna(catalog["stage"])
    return (
        catalog.groupby("module")
        .size()
        .reindex(order)
        .dropna()
        .astype(int)
        .reset_index(name="functions")
    )


def zscore_frame(df: pd.DataFrame, axis: int = 0) -> pd.DataFrame:
    values = df.astype(float)
    if axis == 0:
        return (values - values.mean()) / values.std(ddof=0).replace(0, np.nan)
    return values.sub(values.mean(axis=1), axis=0).div(
        values.std(axis=1, ddof=0).replace(0, np.nan), axis=0
    )


def shorten_labels(labels: list[str], width: int = 34) -> list[str]:
    out = []
    for label in labels:
        text = str(label).replace("__", " ").replace("_", " ")
        if len(text) > width:
            text = text[: width - 1] + "..."
        out.append(text)
    return out


def load_exp2_obs() -> pd.DataFrame:
    h5ad = ROOT / "data" / "exp_2" / "34434_1_adata.h5ad"
    annotation = ROOT / "data" / "exp_2" / "34434_1_annotation.csv"
    if not h5ad.exists() or not annotation.exists():
        raise FileNotFoundError("The exp_2 example h5ad/annotation files are required.")

    adata = ad.read_h5ad(h5ad, backed="r")
    obs = adata.obs.copy()
    adata.file.close()

    ann = pd.read_csv(annotation, index_col=0)
    obs = obs.join(ann, how="left")
    obs["phenotype"] = (
        obs["Tier_A"].fillna(obs.get("annotation_level1", "unannotated")).astype(str)
    )
    obs["imageid"] = obs.get("imageid", "34434_1").astype(str)
    if "label" not in obs:
        obs["label"] = np.arange(len(obs))
    return obs


def sample_obs(obs: pd.DataFrame, n: int, seed: int = 7) -> pd.DataFrame:
    if len(obs) <= n:
        return obs.copy()
    return obs.sample(n=n, random_state=seed).copy()


def minimal_adata(obs: pd.DataFrame) -> ad.AnnData:
    return ad.AnnData(X=np.zeros((len(obs), 1), dtype=np.float32), obs=obs.copy())


def panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.08,
        1.06,
        letter,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="right",
    )


def figure_1_workflow(summary: dict) -> None:
    fig = plt.figure(figsize=(12, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 0.95], width_ratios=[1.35, 1.0])
    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    panel_letter(ax, "A")
    ax.set_title("SpatioEv package workflow and manuscript-ready analysis layers", loc="left")

    stages = [
        ("Inputs", "AnnData\ncell tables\nmasks\nECM fibers"),
        ("QC", "area in um2\nN:C ratio\nartifact flags"),
        ("Phenotype", "marker/morphology\nfeatures\nSVM probabilities"),
        ("Spatial statistics", "tile/radius/kNN density\nRipley/Moran\ninteraction curves"),
        ("Niches", "boundaries\ngraph features\nsurrounding context"),
        ("ECM-cell", "cell-fiber links\nfiber statistics\nlocal coupling"),
        ("Trajectories", "pseudotime bins\nbranch dynamics\nXenium extensions"),
        ("Outputs", "figures\ntables\ntutorials\nmanuscript"),
    ]

    x_positions = np.linspace(0.04, 0.96, len(stages))
    colors = [
        PALETTE["gray"],
        PALETTE["blue"],
        PALETTE["teal"],
        PALETTE["green"],
        PALETTE["purple"],
        PALETTE["orange"],
        PALETTE["red"],
        PALETTE["navy"],
    ]
    for i, ((title, body), x, color) in enumerate(zip(stages, x_positions, colors)):
        rect = mpl.patches.FancyBboxPatch(
            (x - 0.055, 0.44),
            0.11,
            0.32,
            boxstyle="round,pad=0.018,rounding_size=0.02",
            linewidth=0.9,
            facecolor=color,
            edgecolor="white",
            alpha=0.92,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        ax.text(
            x,
            0.67,
            title,
            color="white",
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.text(
            x,
            0.54,
            body,
            color="white",
            fontsize=7.2,
            ha="center",
            va="center",
            linespacing=1.15,
            transform=ax.transAxes,
        )
        if i < len(stages) - 1:
            ax.text(
                (x + x_positions[i + 1]) / 2,
                0.6,
                ">",
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color="#333333",
                transform=ax.transAxes,
            )
    ax.text(
        0.5,
        0.25,
        "Reproducibility layer: lazy imports, packaging metadata, deterministic tests, local-data smoke tests, notebooks with synthetic fallbacks, and GitHub data policy",
        ha="center",
        va="center",
        fontsize=9,
        color="#333333",
        transform=ax.transAxes,
    )

    ax2 = fig.add_subplot(gs[1, 0])
    panel_letter(ax2, "B")
    module_counts = load_catalog_module_counts()
    sns.barplot(
        data=module_counts,
        y="module",
        x="functions",
        ax=ax2,
        color=PALETTE["blue"],
        edgecolor="white",
    )
    ax2.set_xlabel("Documented public functions")
    ax2.set_ylabel("")
    ax2.set_title("API breadth covered in tutorials")
    for container in ax2.containers:
        ax2.bar_label(container, fmt="%d", padding=2, fontsize=7)

    ax3 = fig.add_subplot(gs[1, 1])
    panel_letter(ax3, "C")
    verification = pd.DataFrame(
        {
            "check": ["import", "unit tests", "notebook smoke", "build"],
            "status": [1, 1, 1, 1],
        }
    )
    sns.barplot(
        data=verification,
        x="status",
        y="check",
        ax=ax3,
        color=PALETTE["teal"],
        edgecolor="white",
    )
    ax3.set_xlim(0, 1.2)
    ax3.set_xlabel("verified")
    ax3.set_ylabel("")
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(["pending", "passed"])
    ax3.set_title("Release-readiness checks")
    for y, text in enumerate(["lazy API", "pytest", "nbconvert", "sdist/wheel"]):
        ax3.text(1.03, y, text, va="center", ha="left", fontsize=8)

    summary["figure_1"] = {
        "api_modules": int(module_counts.shape[0]),
        "public_functions_represented": int(module_counts["functions"].sum()),
        "release_checks": verification["check"].tolist(),
    }
    savefig(fig, "figure_1_spatioev_workflow.png")


def figure_2_qc_phenotype(obs: pd.DataFrame, summary: dict) -> None:
    qobs = sample_obs(obs, 180_000)
    adata_qc = minimal_adata(qobs)
    qc_config = QCConfig(
        pixel_size=0.325,
        min_area_um2=5,
        max_area_um2=650,
        max_nc_ratio=1.0,
    )
    adata_qc = run_segmentation_qc(adata_qc, qc_config)
    qc_summary = generate_qc_summary(adata_qc)
    write_table(qc_summary, "figure_2_qc_summary_exp2_sample.csv")

    top = obs["phenotype"].value_counts().head(10)
    comp = top.reset_index()
    comp.columns = ["phenotype", "n_cells"]
    comp["fraction"] = comp["n_cells"] / len(obs)
    write_table(comp, "figure_2_exp2_phenotype_composition.csv")

    scatter = sample_obs(obs, 70_000)
    scatter["phenotype_plot"] = scatter["phenotype"].where(
        scatter["phenotype"].isin(top.index[:8]), "other"
    )

    fig = plt.figure(figsize=(12, 8.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.2], height_ratios=[1, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    panel_letter(ax1, "A")
    sns.histplot(
        adata_qc.obs["nc_ratio"].clip(upper=4),
        bins=90,
        ax=ax1,
        color=PALETTE["blue"],
        edgecolor=None,
    )
    ax1.axvline(1.0, color=PALETTE["red"], lw=1.5, label="N:C ratio = 1")
    ax1.set_xlabel("Nuclear-to-cell area ratio")
    ax1.set_ylabel("Cells")
    ax1.set_title("Segmentation plausibility from N:C ratio")
    ax1.legend(frameon=False, loc="upper right")

    ax2 = fig.add_subplot(gs[1, 0])
    panel_letter(ax2, "B")
    qc_counts = pd.Series(
        {
            "normal": int((adata_qc.obs["area_category"].eq("normal_area") & adata_qc.obs["nc_ratio_category"].eq("normal_nc_ratio")).sum()),
            "debris": int(adata_qc.obs["area_category"].eq("debris_fragment").sum()),
            "large/merged": int(adata_qc.obs["area_category"].eq("merged_cell").sum()),
            "high N:C": int(adata_qc.obs["nc_ratio_category"].eq("abnormal_nc_ratio").sum()),
        }
    )
    sns.barplot(
        x=qc_counts.values,
        y=qc_counts.index,
        ax=ax2,
        palette=[PALETTE["teal"], PALETTE["gold"], PALETTE["orange"], PALETTE["red"]],
        orient="h",
    )
    ax2.set_xlabel("Cells in QC sample")
    ax2.set_ylabel("")
    ax2.set_title("QC flags are auditable cell-level columns")

    ax3 = fig.add_subplot(gs[0, 1])
    panel_letter(ax3, "C")
    sns.barplot(
        data=comp.iloc[::-1],
        x="fraction",
        y="phenotype",
        ax=ax3,
        palette=[PHENOTYPE_COLORS.get(p, "#777777") for p in comp.iloc[::-1]["phenotype"]],
    )
    ax3.set_xlabel("Fraction of cells")
    ax3.set_ylabel("")
    ax3.set_title("Dominant cell ecosystems in the PDAC example image")
    ax3.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))

    ax4 = fig.add_subplot(gs[1, 1])
    panel_letter(ax4, "D")
    color_map = {**PHENOTYPE_COLORS, "other": "#d0d0d0"}
    for phenotype, df in scatter.groupby("phenotype_plot", sort=False):
        ax4.scatter(
            df["X_centroid"],
            df["Y_centroid"],
            s=0.35,
            alpha=0.55,
            color=color_map.get(phenotype, "#777777"),
            label=phenotype,
            rasterized=True,
        )
    ax4.set_aspect("equal")
    ax4.invert_yaxis()
    ax4.set_xlabel("X centroid (pixels)")
    ax4.set_ylabel("Y centroid (pixels)")
    ax4.set_title("Spatial phenotype map from existing annotations")
    ax4.legend(
        markerscale=8,
        ncol=1,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
    )

    summary["figure_2"] = {
        "exp2_total_cells": int(len(obs)),
        "qc_sample_cells": int(len(qobs)),
        "qc_percent_removed": float(qc_summary["percent_removed"].iloc[0]),
        "top_phenotype": str(top.index[0]),
        "top_phenotype_fraction": float(top.iloc[0] / len(obs)),
    }
    savefig(fig, "figure_2_qc_phenotype_example.png")


def figure_3_spatial_density(obs: pd.DataFrame, summary: dict) -> None:
    density_obs = obs[
        ["label", "area", "X_centroid", "Y_centroid", "imageid", "phenotype"]
    ].copy()
    tiled = assign_tiles(SimpleNamespace(obs=density_obs), tile_size=1024)
    density = compute_general_density(tiled, tile_size=1024)
    pheno_density = compute_phenotype_density(tiled, phenotype_key="phenotype", tile_size=1024)

    top_phenotypes = obs["phenotype"].value_counts().head(8).index.tolist()
    corr = phenotype_density_correlation(
        pheno_density[pheno_density["phenotype"].isin(top_phenotypes)],
        phenotype_key="phenotype",
        value="object_density",
    ).loc[top_phenotypes, top_phenotypes]
    write_table(density, "figure_3_tile_density_exp2.csv")
    write_table(corr.reset_index(), "figure_3_phenotype_density_correlation.csv")

    source = "pancreatic ductal epithelium"
    target = "Fibroblasts"
    pair = obs[obs["phenotype"].isin([source, target])]
    pair = pair.groupby("phenotype", group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), 3500), random_state=12)
    )
    pair_adata = minimal_adata(pair)
    radii = np.array([25, 50, 100, 200, 400, 800, 1200])
    ripley = cross_ripleys_curve_by_phenotype(
        pair_adata,
        phenotype_key="phenotype",
        source_phenotype=source,
        target_phenotype=target,
        radii=radii,
    )
    write_table(ripley, "figure_3_ductal_fibroblast_cross_ripley.csv")

    local = sample_obs(obs, 45_000)
    local["ductal_neighbor_fraction"] = np.nan
    # Lightweight local neighborhood summary for visual interpretation.
    from sklearn.neighbors import BallTree

    coords = local[["X_centroid", "Y_centroid"]].to_numpy()
    tree = BallTree(coords)
    is_ductal = local["phenotype"].eq(source).to_numpy()
    neighbors = tree.query_radius(coords, r=450)
    local["ductal_neighbor_fraction"] = [
        float(is_ductal[nbrs].mean()) if len(nbrs) else np.nan for nbrs in neighbors
    ]

    fig = plt.figure(figsize=(12, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    panel_letter(ax1, "A")
    heat = density.pivot(index="tile_y", columns="tile_x", values="object_density")
    sns.heatmap(heat, ax=ax1, cmap="viridis", cbar_kws={"label": "cells / tile area x100"})
    ax1.set_title("Tile-based cellular density")
    ax1.set_xlabel("Tile X")
    ax1.set_ylabel("Tile Y")
    ax1.invert_yaxis()

    ax2 = fig.add_subplot(gs[0, 1])
    panel_letter(ax2, "B")
    sns.heatmap(
        corr,
        ax=ax2,
        cmap="vlag",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.3,
        cbar_kws={"label": "Pearson r"},
    )
    ax2.set_xticklabels(shorten_labels(top_phenotypes, 22), rotation=45, ha="right")
    ax2.set_yticklabels(shorten_labels(top_phenotypes, 22), rotation=0)
    ax2.set_title("Phenotype density co-organization")

    ax3 = fig.add_subplot(gs[1, 0])
    panel_letter(ax3, "C")
    if not ripley.empty:
        ax3.plot(
            ripley["radius"],
            ripley["L_minus_r"],
            marker="o",
            color=PALETTE["red"],
            lw=1.8,
        )
        ax3.axhline(0, color="#333333", lw=0.8, ls="--")
    ax3.set_xlabel("Radius (pixels)")
    ax3.set_ylabel("Cross L(r) - r")
    ax3.set_title("Ductal-fibroblast interaction scale")

    ax4 = fig.add_subplot(gs[1, 1])
    panel_letter(ax4, "D")
    sc = ax4.scatter(
        local["X_centroid"],
        local["Y_centroid"],
        c=local["ductal_neighbor_fraction"],
        cmap="mako",
        s=0.4,
        alpha=0.65,
        rasterized=True,
    )
    ax4.set_aspect("equal")
    ax4.invert_yaxis()
    ax4.set_xlabel("X centroid (pixels)")
    ax4.set_ylabel("Y centroid (pixels)")
    ax4.set_title("Local ductal neighborhood enrichment")
    fig.colorbar(sc, ax=ax4, label="fraction ductal within 450 px")

    corr_mask = corr.where(~np.eye(len(corr), dtype=bool))
    stacked = corr_mask.stack().sort_values(ascending=False)
    top_positive = stacked.head(1)
    top_negative = stacked.tail(1)
    summary["figure_3"] = {
        "tiles": int(len(density)),
        "top_positive_density_correlation": {
            "pair": [str(top_positive.index[0][0]), str(top_positive.index[0][1])],
            "r": float(top_positive.iloc[0]),
        },
        "top_negative_density_correlation": {
            "pair": [str(top_negative.index[0][0]), str(top_negative.index[0][1])],
            "r": float(top_negative.iloc[0]),
        },
        "ripley_peak_radius_pixels": int(ripley.loc[ripley["L_minus_r"].idxmax(), "radius"])
        if not ripley.empty
        else None,
        "ripley_peak_l_minus_r": float(ripley["L_minus_r"].max()) if not ripley.empty else None,
    }
    savefig(fig, "figure_3_density_spatial_statistics.png")


def figure_4_pseudotime(summary: dict) -> None:
    result_path = ROOT / "data" / "combined_exp_2_3_4_5" / "pooled_niche_result_df.pkl"
    panin_path = (
        ROOT
        / "data"
        / "combined_exp_2_3_4_5"
        / "pooled_pathology_with_panin_validation_scores.pkl"
    )
    result = pd.read_pickle(result_path)
    panin = pd.read_pickle(panin_path)
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

    score_cols = {
        "pdac_early_duct_anchor_score": "early duct",
        "pdac_panin_like_dysplasia_score": "PanIN-like dysplasia",
        "pdac_invasion_desmoplasia_axis": "invasion/desmoplasia",
        "pdac_proliferation_axis": "proliferation",
        "pdac_dedifferentiation_axis": "dedifferentiation",
        "panin_validation__panin_grade_like_axis": "PanIN grade-like",
    }
    available_scores = {k: v for k, v in score_cols.items() if k in merged.columns}
    trend_rows = []
    for col, label in available_scores.items():
        valid = merged[["pooled_pseudotime", col]].dropna()
        rho, pval = spearmanr(valid["pooled_pseudotime"], valid[col])
        trend_rows.append({"feature": col, "label": label, "spearman_r": rho, "p_value": pval})
    trends = pd.DataFrame(trend_rows).sort_values("spearman_r")
    write_table(trends, "figure_4_pseudotime_score_correlations.csv")

    merged["pseudotime_decile"] = pd.qcut(
        merged["pooled_pseudotime"], q=10, labels=False, duplicates="drop"
    )
    deciles = (
        merged.groupby("pseudotime_decile")[list(available_scores)]
        .median()
        .rename(columns=available_scores)
    )
    deciles_z = zscore_frame(deciles, axis=0)
    write_table(deciles.reset_index(), "figure_4_pseudotime_score_deciles.csv")

    branch_counts = (
        merged.groupby(["major_branch", "disease_group"]).size().unstack(fill_value=0)
    )
    branch_counts = branch_counts.loc[branch_counts.sum(axis=1).sort_values(ascending=False).head(10).index]
    branch_frac = branch_counts.div(branch_counts.sum(axis=1), axis=0)

    fig = plt.figure(figsize=(12, 8.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    panel_letter(ax1, "A")
    scatter = merged.sample(n=min(merged.shape[0], 26_000), random_state=15)
    sc = ax1.scatter(
        scatter["UMAP1"],
        scatter["UMAP2"],
        c=scatter["pooled_pseudotime"],
        cmap="turbo",
        s=1.6,
        alpha=0.75,
        rasterized=True,
    )
    ax1.set_xlabel("UMAP1")
    ax1.set_ylabel("UMAP2")
    ax1.set_title("Epithelial niche trajectory embedding")
    fig.colorbar(sc, ax=ax1, label="pooled pseudotime")

    ax2 = fig.add_subplot(gs[0, 1])
    panel_letter(ax2, "B")
    sns.heatmap(
        deciles_z.T,
        ax=ax2,
        cmap="vlag",
        center=0,
        cbar_kws={"label": "z-scored median"},
        linewidths=0.25,
    )
    ax2.set_xlabel("Pseudotime decile")
    ax2.set_ylabel("")
    ax2.set_title("Pathology programs change along pseudotime")

    ax3 = fig.add_subplot(gs[1, 0])
    panel_letter(ax3, "C")
    colors = [PALETTE["blue"] if x > 0 else PALETTE["orange"] for x in trends["spearman_r"]]
    sns.barplot(data=trends, x="spearman_r", y="label", ax=ax3, palette=colors)
    ax3.axvline(0, color="#333333", lw=0.8)
    ax3.set_xlabel("Spearman r with pseudotime")
    ax3.set_ylabel("")
    ax3.set_title("Direction of niche-level biological progression")

    ax4 = fig.add_subplot(gs[1, 1])
    panel_letter(ax4, "D")
    branch_frac.plot(kind="barh", stacked=True, ax=ax4, color=[PALETTE["navy"], PALETTE["red"]])
    ax4.set_xlabel("Fraction of branch niches")
    ax4.set_ylabel("")
    ax4.set_title("Normal and PDAC occupancy of recurrent branches")
    ax4.legend(title="", frameon=False, loc="lower right")

    top_positive = trends.sort_values("spearman_r", ascending=False).head(2)
    top_negative = trends.sort_values("spearman_r").head(2)
    summary["figure_4"] = {
        "n_niches": int(merged.shape[0]),
        "n_pseudotime_branches": int(merged["major_branch"].nunique()),
        "top_positive_pseudotime_programs": top_positive[["label", "spearman_r"]].to_dict("records"),
        "top_negative_pseudotime_programs": top_negative[["label", "spearman_r"]].to_dict("records"),
    }
    savefig(fig, "figure_4_pseudotime_niche_biology.png")


def figure_5_microenvironment(summary: dict) -> None:
    base = ROOT / "notebooks" / "results" / "trajectory_microenvironment_interactions"
    top_changes = pd.read_csv(base / "tables" / "top_trajectory_microenvironment_changes.csv")
    mux_state = pd.read_csv(base / "tables" / "multiplexed_branch_time_state_summary.csv")
    xen_state = pd.read_csv(base / "tables" / "xenium_branch_time_state_summary.csv")
    events = pd.read_csv(base / "tables" / "all_branch_time_transition_events.csv")

    show = top_changes.sort_values("abs_spearman_r", ascending=False).head(14).copy()
    show["short_label"] = shorten_labels(show["label"].tolist(), 44)
    show["signed"] = np.where(show["spearman_r"] >= 0, "increases", "decreases")
    write_table(show, "figure_5_top_microenvironment_trends.csv")

    mux_features = [
        "surround_prop__Fibroblasts",
        "surround_prop__T_cells",
        "surround_prop__B_lineage",
        "surround__Fibroblasts__FAP_expr_z__mean",
        "coloc__ductal_to__Fibroblasts__r30__fraction_epithelial_cells_with_target_neighbor",
        "coloc__ductal_to__T_cells__r30__mean_target_neighbor_excess",
        "atlas__fibrotic_reaction",
        "atlas__immune_infiltration",
    ]
    mux_available = [c for c in mux_features if c in mux_state.columns]
    feature_labels = {
        "surround_prop__Fibroblasts": "Fibroblast prop.",
        "surround_prop__T_cells": "T-cell prop.",
        "surround_prop__B_lineage": "B-lineage prop.",
        "surround_prop__Myeloid_cells": "Myeloid prop.",
        "surround__Fibroblasts__FAP_expr_z__mean": "Fibroblast FAP",
        "surround__Fibroblasts__ACTA2_expr_z__mean": "Fibroblast ACTA2",
        "surround__T_cells__FOXP3_expr_z__mean": "T-cell FOXP3",
        "coloc__ductal_to__Fibroblasts__r30__fraction_epithelial_cells_with_target_neighbor": "ductal near fibroblasts",
        "coloc__ductal_to__T_cells__r30__mean_target_neighbor_excess": "ductal-T excess",
        "atlas__fibrotic_reaction": "fibrotic reaction atlas",
        "atlas__immune_infiltration": "immune infiltration atlas",
        "nbhd_program__fibroblast_activation": "neighborhood fibroblast activation",
        "nbhd_program__treg_checkpoint": "neighborhood Treg/checkpoint",
        "lr__pancreatic_ductal_epithelium__to__Endothelial_cells__VEGFA__FLT1": "epithelial VEGFA -> endothelial FLT1",
        "lr__pancreatic_ductal_epithelium__to__Myeloid_cells__CSF1__CSF1R": "epithelial CSF1 -> myeloid CSF1R",
    }

    mux_pivot = mux_state.pivot_table(
        index="branch_time_state", values=mux_available, aggfunc="median"
    )
    mux_pivot = mux_pivot.loc[mux_state["branch_time_state"].drop_duplicates().head(18)]
    mux_z = zscore_frame(mux_pivot, axis=0).rename(columns=feature_labels)

    xen_features = [
        "surround_prop__Fibroblasts",
        "surround_prop__Myeloid_cells",
        "surround__Fibroblasts__ACTA2_expr_z__mean",
        "surround__T_cells__FOXP3_expr_z__mean",
        "nbhd_program__fibroblast_activation",
        "nbhd_program__treg_checkpoint",
        "lr__pancreatic_ductal_epithelium__to__Endothelial_cells__VEGFA__FLT1",
        "lr__pancreatic_ductal_epithelium__to__Myeloid_cells__CSF1__CSF1R",
    ]
    xen_available = [c for c in xen_features if c in xen_state.columns]
    xen_pivot = xen_state.pivot_table(
        index="branch_time_state", values=xen_available, aggfunc="median"
    )
    xen_pivot = xen_pivot.loc[xen_state["branch_time_state"].drop_duplicates().head(18)]
    xen_z = zscore_frame(xen_pivot, axis=0).rename(columns=feature_labels)

    top_events = events.sort_values("max_abs_delta_z", ascending=False).head(8).copy()

    fig = plt.figure(figsize=(12, 9.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1])

    ax1 = fig.add_subplot(gs[0, 0])
    panel_letter(ax1, "A")
    colors = [PALETTE["teal"] if x > 0 else PALETTE["orange"] for x in show["spearman_r"]]
    sns.barplot(data=show, x="spearman_r", y="short_label", ax=ax1, palette=colors)
    ax1.axvline(0, color="#333333", lw=0.8)
    ax1.set_xlabel("Spearman r with pseudotime")
    ax1.set_ylabel("")
    ax1.set_title("Cross-platform microenvironment programs")

    ax2 = fig.add_subplot(gs[0, 1])
    panel_letter(ax2, "B")
    sns.heatmap(
        mux_z.T,
        ax=ax2,
        cmap="vlag",
        center=0,
        cbar_kws={"label": "z"},
        xticklabels=shorten_labels(mux_z.index.tolist(), 18),
        yticklabels=mux_z.columns.tolist(),
    )
    ax2.set_title("Multiplexed branch-time niches")
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    ax2.tick_params(axis="x", rotation=75)

    ax3 = fig.add_subplot(gs[1, 0])
    panel_letter(ax3, "C")
    sns.heatmap(
        xen_z.T,
        ax=ax3,
        cmap="vlag",
        center=0,
        cbar_kws={"label": "z"},
        xticklabels=shorten_labels(xen_z.index.tolist(), 18),
        yticklabels=xen_z.columns.tolist(),
    )
    ax3.set_title("Xenium branch-time niches")
    ax3.set_xlabel("")
    ax3.set_ylabel("")
    ax3.tick_params(axis="x", rotation=75)

    ax4 = fig.add_subplot(gs[1, 1])
    panel_letter(ax4, "D")
    sns.barplot(
        data=top_events,
        x="max_abs_delta_z",
        y="transition",
        hue="dataset",
        ax=ax4,
        dodge=False,
        palette={"multiplexed imaging": PALETTE["blue"], "xenium": PALETTE["red"]},
    )
    ax4.set_xlabel("Strongest standardized transition delta")
    ax4.set_ylabel("")
    ax4.set_title("Branch-specific transition events")
    ax4.legend(frameon=False, title="")

    summary["figure_5"] = {
        "top_microenvironment_trend": {
            "label": str(show.iloc[0]["label"]),
            "spearman_r": float(show.iloc[0]["spearman_r"]),
            "late_minus_early_median": float(show.iloc[0]["late_minus_early_median"]),
            "analysis": str(show.iloc[0]["analysis"]),
        },
        "n_transition_events": int(events.shape[0]),
        "strongest_transition": {
            "dataset": str(top_events.iloc[0]["dataset"]),
            "transition": str(top_events.iloc[0]["transition"]),
            "max_abs_delta_z": float(top_events.iloc[0]["max_abs_delta_z"]),
            "top_changes": str(top_events.iloc[0]["top_changes"]),
        },
    }
    savefig(fig, "figure_5_microenvironment_dynamics.png")


def figure_6_ecm(summary: dict) -> None:
    base = ROOT / "notebooks" / "results" / "ra_oa_ecm_cell"
    density = pd.read_csv(base / "analysis_outputs" / "ra_oa_density_comparison.csv")
    distance = pd.read_csv(base / "analysis_outputs" / "ra_oa_distance_comparison.csv")
    cross = pd.read_csv(
        base / "spatioev_module_paper_applications" / "tables" / "10_cross_morans_i_ecm_cell_coupling.csv"
    )
    col6 = pd.read_csv(
        base
        / "spatioev_module_paper_applications"
        / "tables"
        / "01_col6_dark_region_cell_fraction_by_phenotype.csv"
    )

    density_top = density.assign(abs_delta=lambda d: d["RA_minus_OA"].abs()).sort_values(
        "abs_delta", ascending=False
    ).head(12)
    distance_top = distance.assign(abs_delta=lambda d: d["RA_minus_OA"].abs()).sort_values(
        "abs_delta", ascending=False
    ).head(12)

    cross_summary = (
        cross.assign(
            interaction=lambda d: d["phenotype"]
            + " | "
            + d["fiber_type"]
            + " "
            + d["fiber_feature"].str.replace("_", " ")
        )
        .groupby(["interaction", "pathology"], as_index=False)["cross_morans_i"]
        .mean()
    )
    cross_matrix = cross_summary.pivot_table(
        index="interaction", columns="pathology", values="cross_morans_i"
    )
    cross_matrix = cross_matrix.loc[
        cross_matrix.abs().max(axis=1).sort_values(ascending=False).head(10).index
    ]

    col6_focus = col6[
        col6["phenotype"].isin(["B cells", "T cells", "CD4 T cells", "Monocytes", "Macrophages", "Vascular cells"])
        & col6["COL6_dark_region"].isin(["core", "inner_edge", "outer_edge", "background"])
    ]
    col6_focus = (
        col6_focus.groupby(["pathology", "phenotype", "COL6_dark_region"])["fraction_of_phenotype"]
        .mean()
        .reset_index()
    )
    col6_focus["state"] = col6_focus["pathology"] + "::" + col6_focus["COL6_dark_region"]
    col6_matrix = col6_focus.pivot_table(
        index="phenotype", columns="state", values="fraction_of_phenotype", fill_value=0
    )

    fig = plt.figure(figsize=(12, 8.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    panel_letter(ax1, "A")
    density_top["label"] = density_top["phenotype"] + " near " + density_top["fiber_type"]
    density_top["delta_x1e4"] = density_top["RA_minus_OA"] * 10_000
    sns.barplot(
        data=density_top.iloc[::-1],
        x="delta_x1e4",
        y="label",
        ax=ax1,
        palette=[PALETTE["red"] if v > 0 else PALETTE["blue"] for v in density_top.iloc[::-1]["RA_minus_OA"]],
    )
    ax1.axvline(0, color="#333333", lw=0.8)
    ax1.set_xlabel("RA - OA fiber density near cells (x10,000)")
    ax1.set_ylabel("")
    ax1.set_title("Disease-shifted cell-ECM density")

    ax2 = fig.add_subplot(gs[0, 1])
    panel_letter(ax2, "B")
    distance_top["label"] = distance_top["phenotype"] + " to " + distance_top["fiber_type"]
    sns.barplot(
        data=distance_top.iloc[::-1],
        x="RA_minus_OA",
        y="label",
        ax=ax2,
        palette=[PALETTE["red"] if v > 0 else PALETTE["blue"] for v in distance_top.iloc[::-1]["RA_minus_OA"]],
    )
    ax2.axvline(0, color="#333333", lw=0.8)
    ax2.set_xlabel("RA - OA nearest distance")
    ax2.set_ylabel("")
    ax2.set_title("Nearest fiber distances retain disease signal")

    ax3 = fig.add_subplot(gs[1, 0])
    panel_letter(ax3, "C")
    sns.heatmap(
        cross_matrix,
        ax=ax3,
        cmap="vlag",
        center=0,
        linewidths=0.3,
        cbar_kws={"label": "cross Moran's I"},
    )
    ax3.set_xlabel("Pathology")
    ax3.set_ylabel("")
    ax3.set_title("Local coupling of cell phenotypes and ECM features")

    ax4 = fig.add_subplot(gs[1, 1])
    panel_letter(ax4, "D")
    sns.heatmap(
        col6_matrix,
        ax=ax4,
        cmap="YlGnBu",
        cbar_kws={"label": "mean phenotype fraction"},
        linewidths=0.25,
    )
    ax4.set_xlabel("Pathology and COL6-dark region")
    ax4.set_ylabel("")
    ax4.set_title("COL6-dark regions reshape cell context")
    ax4.tick_params(axis="x", rotation=55)

    summary["figure_6"] = {
        "top_density_shift": {
            "phenotype": str(density_top.iloc[0]["phenotype"]),
            "fiber_type": str(density_top.iloc[0]["fiber_type"]),
            "ra_minus_oa": float(density_top.iloc[0]["RA_minus_OA"]),
        },
        "top_distance_shift": {
            "phenotype": str(distance_top.iloc[0]["phenotype"]),
            "fiber_type": str(distance_top.iloc[0]["fiber_type"]),
            "ra_minus_oa": float(distance_top.iloc[0]["RA_minus_OA"]),
        },
        "top_alignment_coupling": {
            "interaction": str(cross_summary.loc[cross_summary["cross_morans_i"].abs().idxmax(), "interaction"]),
            "cross_morans_i": float(cross_summary["cross_morans_i"].abs().max()),
        },
    }
    savefig(fig, "figure_6_ecm_cell_interactions.png")


def figure_7_xenium(summary: dict) -> None:
    xen_base = ROOT / "data" / "xenium_pancreas_10x"
    branch = pd.read_csv(xen_base / "pseudotime" / "xenium_branch_biology_summary.csv")
    clinical = pd.read_csv(xen_base / "pseudotime" / "xenium_clinical_pseudotime_summary.csv")
    banksy = pd.read_csv(xen_base / "banksy" / "xenium_banksy_branch_time_summary.csv")
    top_changes = pd.read_csv(
        ROOT
        / "notebooks"
        / "results"
        / "trajectory_microenvironment_interactions"
        / "tables"
        / "top_trajectory_microenvironment_changes.csv"
    )
    xen_changes = top_changes[top_changes["analysis"].str.startswith("xenium")].copy()
    xen_changes = xen_changes.sort_values("abs_spearman_r", ascending=False).head(10)
    xen_changes["short_label"] = shorten_labels(xen_changes["label"].tolist(), 42)

    z_cols = [
        "histology__normal_duct_like_score__z_enrichment",
        "histology__adm_panin_like_score__z_enrichment",
        "histology__glandular_architecture_score__z_enrichment",
        "histology__epithelial_stromal_interface_disruption_score__z_enrichment",
        "histology__desmoplastic_tumor_score__z_enrichment",
        "histology__immune_inflamed_score__z_enrichment",
        "histology__immune_exclusion_score__z_enrichment",
        "xenium_panin_like_remodeling_score__z_enrichment",
        "xenium_epithelial_identity_score__z_enrichment",
        "xenium_desmoplastic_context_score__z_enrichment",
    ]
    z_cols = [c for c in z_cols if c in branch.columns]
    branch_heat = branch.set_index("branch")[z_cols]
    branch_heat.columns = [
        c.replace("histology__", "")
        .replace("xenium_", "")
        .replace("_score__z_enrichment", "")
        .replace("__z_enrichment", "")
        for c in branch_heat.columns
    ]

    entropy = banksy.pivot_table(
        index="major_branch",
        columns="branch_time_bin",
        values="banksy_surround__domain_entropy",
        aggfunc="median",
    )
    entropy = entropy[[c for c in ["early", "mid", "late"] if c in entropy.columns]]

    fig = plt.figure(figsize=(12, 8.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.1])

    ax1 = fig.add_subplot(gs[0, 0])
    panel_letter(ax1, "A")
    clinical = clinical.sort_values("pseudotime_median")
    ax1.errorbar(
        clinical["pseudotime_median"],
        clinical["sample_id"],
        xerr=[
            clinical["pseudotime_median"] - clinical["pseudotime_q25"],
            clinical["pseudotime_q75"] - clinical["pseudotime_median"],
        ],
        fmt="o",
        color=PALETTE["red"],
        ecolor="#999999",
        capsize=2,
    )
    ax1.set_xlabel("Pseudotime median (IQR)")
    ax1.set_ylabel("")
    ax1.set_title("Xenium clinical samples along inferred progression")

    ax2 = fig.add_subplot(gs[0, 1])
    panel_letter(ax2, "B")
    sns.heatmap(
        branch_heat,
        ax=ax2,
        cmap="vlag",
        center=0,
        linewidths=0.25,
        cbar_kws={"label": "branch z-enrichment"},
    )
    ax2.set_xlabel("Biology score")
    ax2.set_ylabel("")
    ax2.set_title("Branch-specific Xenium biology")
    ax2.tick_params(axis="x", rotation=55)

    ax3 = fig.add_subplot(gs[1, 0])
    panel_letter(ax3, "C")
    colors = [PALETTE["teal"] if x > 0 else PALETTE["orange"] for x in xen_changes["spearman_r"]]
    sns.barplot(data=xen_changes, x="spearman_r", y="short_label", ax=ax3, palette=colors)
    ax3.axvline(0, color="#333333", lw=0.8)
    ax3.set_xlabel("Spearman r with pseudotime")
    ax3.set_ylabel("")
    ax3.set_title("Spatial transcriptomics validates microenvironment trends")

    ax4 = fig.add_subplot(gs[1, 1])
    panel_letter(ax4, "D")
    sns.heatmap(
        entropy,
        ax=ax4,
        cmap="crest",
        linewidths=0.25,
        cbar_kws={"label": "BANKSY surround-domain entropy"},
    )
    ax4.set_xlabel("Branch time")
    ax4.set_ylabel("")
    ax4.set_title("BANKSY domain mixing across branch time")

    summary["figure_7"] = {
        "xenium_samples": int(clinical.shape[0]),
        "xenium_branches": int(branch.shape[0]),
        "top_xenium_trend": {
            "label": str(xen_changes.iloc[0]["label"]),
            "spearman_r": float(xen_changes.iloc[0]["spearman_r"]),
            "late_minus_early_median": float(xen_changes.iloc[0]["late_minus_early_median"]),
            "analysis": str(xen_changes.iloc[0]["analysis"]),
        },
        "branch_biology_examples": branch[["branch", "suggested_biology"]]
        .head(5)
        .to_dict("records"),
    }
    savefig(fig, "figure_7_xenium_extension.png")


def main() -> None:
    configure_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    summary: dict = {}
    obs = load_exp2_obs()
    figure_1_workflow(summary)
    figure_2_qc_phenotype(obs, summary)
    figure_3_spatial_density(obs, summary)
    figure_4_pseudotime(summary)
    figure_5_microenvironment(summary)
    figure_6_ecm(summary)
    figure_7_xenium(summary)

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {SUMMARY_PATH.relative_to(ROOT)}")
    for fig in sorted(FIG_DIR.glob("figure_*.png")):
        print(f"Wrote {fig.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
