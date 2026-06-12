"""Generate the ChatGPT Pro/Codex outlined SpatioEv pseudotime figures.

Outputs are written to:

- manuscript/figures/main/Fig1_workflow_scaffold.{png,pdf,svg}
- manuscript/figures/main/Fig2_single_sample_trajectory.{png,pdf,svg}
- manuscript/figures/main/Fig3_pooled_multiplexed_atlas.{png,pdf,svg}
- manuscript/figures/main/Fig4_xenium_transfer.{png,pdf,svg}
- manuscript/figures/panels/*.{png,pdf,svg}
- manuscript/figures/tables/*.csv
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpecFromSubplotSpec
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from statsmodels.nonparametric.smoothers_lowess import lowess

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_style import (  # noqa: E402
    BRANCH_CMAP,
    DISEASE_COLORS,
    MODULE_COLORS,
    PHENOTYPE_COLORS,
    PSEUDOTIME_CMAP,
    SAMPLE_COLORS,
    XENIUM_SAMPLE_COLORS,
    add_scale_bar,
    clean_axis,
    clean_spatial_axis,
    configure,
    export_figure,
    panel_letter,
    shorten,
)


ROOT = Path(__file__).resolve().parents[1]
MAIN_DIR = ROOT / "manuscript" / "figures" / "main"
PANEL_DIR = ROOT / "manuscript" / "figures" / "panels"
TABLE_DIR = ROOT / "manuscript" / "figures" / "tables"

REP_SAMPLE = "34434_1"
COMPONENT = "pancreatic ductal epithelium_mask_component"
XENIUM_COMPONENT = "xenium_ductal_epithelium_component"

SAMPLE_LABELS = {
    "40331_1": "Normal pancreas",
    "34434_1": "PDAC 34434",
    "33694_1": "PDAC 33694",
    "35559_1": "PDAC 35559",
}

XENIUM_LABELS = {
    "normal_nondiseased_v1": "Normal pancreas",
    "pdac_pancreas_v1": "PDAC pancreas",
    "pdac_addon_v1": "PDAC add-on",
    "pdac_io_v1": "PDAC invasion",
}

MODULES = {
    "pdac_early_duct_anchor_score": "early duct",
    "pdac_panin_like_dysplasia_score": "PanIN-like",
    "pdac_architectural_complexity_score": "architecture",
    "pdac_invasion_desmoplasia_axis": "invasion/desmoplasia",
    "pdac_proliferation_axis": "proliferation",
    "pdac_dedifferentiation_axis": "dedifferentiation",
}

XENIUM_SCORES = {
    "histology__normal_duct_like_score": "normal duct",
    "histology__adm_panin_like_score": "ADM/PanIN",
    "histology__desmoplastic_tumor_score": "desmoplastic",
    "histology__immune_inflamed_score": "immune inflamed",
    "histology__immune_exclusion_score": "immune excluded",
    "histology__gland_poor_undifferentiated_score": "gland-poor",
}


def zscore(df: pd.DataFrame, axis: int = 0) -> pd.DataFrame:
    values = df.astype(float)
    if axis == 1:
        return values.sub(values.mean(axis=1), axis=0).div(values.std(axis=1, ddof=0).replace(0, np.nan), axis=0)
    return (values - values.mean()) / values.std(ddof=0).replace(0, np.nan)


def robust_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=seed)


def bh_fdr(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)
    q = np.full(len(p), np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return pd.Series(q, index=p_values.index)
    valid_idx = np.where(valid)[0]
    order = valid_idx[np.argsort(p[valid])]
    ranked = p[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q[order] = np.clip(adjusted, 0, 1)
    return pd.Series(q, index=p_values.index)


def with_fdr(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    p_col = "spearman_p" if "spearman_p" in out.columns else "p"
    if "fdr" not in out.columns and p_col in out.columns:
        out["fdr"] = bh_fdr(out[p_col])
    return out


def save_table(df: pd.DataFrame, name: str) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE_DIR / f"{name}.csv", index=False)


def load_data() -> dict[str, pd.DataFrame]:
    base = ROOT / "data" / "combined_exp_2_3_4_5"
    pooled = pd.read_pickle(base / "pooled_niche_result_df.pkl")
    pooled_features = pd.read_pickle(base / "pooled_pathology_feature_df.pkl")
    exp2_features = pd.read_pickle(base / "per_sample" / "exp_2_pathology_feature_table.pkl")
    exp2 = pooled.loc[pooled["sample_id"] == REP_SAMPLE].copy()
    feature_cols = [c for c in exp2_features.columns if c not in exp2.columns or c in {COMPONENT, "image_id"}]
    exp2 = exp2.merge(exp2_features[feature_cols], on=[COMPONENT, "image_id"], how="left")
    spatial = pd.read_pickle(base / f"spatial_cells_auto_branch_n24_v3_with_epithelial_{REP_SAMPLE}.pkl")
    ann = pd.read_csv(ROOT / "data" / "exp_2" / "34434_1_annotation.csv", usecols=["Tier_A", "Tier_B"])
    interactions = pd.read_pickle(
        ROOT
        / "notebooks"
        / "results"
        / "trajectory_microenvironment_interactions"
        / "multiplexed_epithelial_niche_local_colocalization.pkl"
    )
    micro = with_fdr(
        pd.read_csv(
            ROOT
            / "notebooks"
            / "results"
            / "trajectory_microenvironment_interactions"
            / "tables"
            / "multiplexed_microenvironment_trends_contextual.csv"
        )
    )
    contact = with_fdr(
        pd.read_csv(
            ROOT
            / "notebooks"
            / "results"
            / "trajectory_microenvironment_interactions"
            / "tables"
            / "multiplexed_epithelial_niche_colocalization_trends.csv"
        )
    )
    branch_time = pd.read_csv(
        ROOT
        / "notebooks"
        / "results"
        / "trajectory_microenvironment_interactions"
        / "tables"
        / "multiplexed_branch_time_state_summary.csv"
    )
    epithelial_intrinsic = pd.read_pickle(base / "epithelial_intrinsic_pseudotime_result_df.pkl")
    xenium = pd.read_pickle(ROOT / "data" / "xenium_pancreas_10x" / "pseudotime" / "xenium_pseudotime_result_df.pkl")
    xen_audit = pd.read_csv(ROOT / "data" / "xenium_pancreas_10x" / "xenium_dataset_audit.csv")
    xen_tier = pd.read_csv(ROOT / "data" / "xenium_pancreas_10x" / "xenium_tier_a_counts.csv")
    xen_features = pd.read_csv(ROOT / "data" / "xenium_pancreas_10x" / "pseudotime" / "xenium_pseudotime_feature_blocks.csv")
    xen_branch = pd.read_csv(ROOT / "data" / "xenium_pancreas_10x" / "pseudotime" / "xenium_branch_biology_summary.csv")
    xen_micro = with_fdr(
        pd.read_csv(
            ROOT
            / "notebooks"
            / "results"
            / "trajectory_microenvironment_interactions"
            / "tables"
            / "xenium_microenvironment_trends_sample_centered.csv"
        )
    )
    xen_branch_occ = pd.read_csv(ROOT / "data" / "xenium_pancreas_10x" / "pseudotime" / "xenium_clinical_branch_occupancy.csv")
    return {
        "pooled": pooled,
        "pooled_features": pooled_features,
        "exp2": exp2,
        "exp2_features": exp2_features,
        "spatial": spatial,
        "ann": ann,
        "interactions": interactions,
        "micro": micro,
        "contact": contact,
        "branch_time": branch_time,
        "epithelial_intrinsic": epithelial_intrinsic,
        "xenium": xenium,
        "xen_audit": xen_audit,
        "xen_tier": xen_tier,
        "xen_features": xen_features,
        "xen_branch": xen_branch,
        "xen_micro": xen_micro,
        "xen_branch_occ": xen_branch_occ,
    }


def feature_block_summary(df: pd.DataFrame) -> pd.DataFrame:
    groups = [
        ("topology", lambda c: c.startswith("topology__"), "graph degree, bridges, skeleton branches"),
        ("geometry", lambda c: c.startswith("geometry__"), "hull shape, compactness, orientation"),
        ("boundary", lambda c: c.startswith("graph_boundary__"), "boundary-core contrasts"),
        ("pixel morphology", lambda c: "polarity" in c or "haralick" in c or "lacunarity" in c or "entropy" in c or "texture" in c, "polarity, entropy, lacunarity, Haralick texture"),
        ("marker state", lambda c: "CK19" in c or "NaKATPase" in c or "Ki67" in c or "Vimentin" in c, "epithelial/proliferation marker summaries"),
        ("surrounding context", lambda c: c.startswith("surround_prop__") or c.startswith("surround__"), "fibroblast, immune, endothelial, mesenchymal context"),
        ("cell-state summaries", lambda c: c.startswith("state__"), "within-niche state means and dispersion"),
        ("pathology modules", lambda c: c.startswith("pdac_"), "signed pathology-inspired scores"),
    ]
    rows = []
    used = set()
    for name, matcher, example in groups:
        cols = [c for c in df.columns if matcher(c)]
        used.update(cols)
        rows.append({"feature_family": name, "n_features": len(cols), "examples": example})
    return pd.DataFrame(rows)


def assert_plot_inputs(data: dict[str, pd.DataFrame]) -> None:
    required_single = {"UMAP1", "UMAP2", "pooled_pseudotime", "major_branch"}
    missing = required_single - set(data["exp2"].columns)
    if missing:
        raise AssertionError(f"single-sample result table missing {missing}")
    if data["exp2"]["pooled_pseudotime"].isna().any():
        raise AssertionError("single-sample plotted niches contain missing pseudotime")
    if data["pooled"]["pooled_pseudotime"].isna().any():
        raise AssertionError("pooled plotted niches contain missing pseudotime")
    if data["xenium"]["xenium_pseudotime_sample_centered_norm"].isna().any():
        raise AssertionError("Xenium plotted niches contain missing sample-centered pseudotime")
    for name in ("micro", "contact", "xen_micro"):
        if (data[name]["n"] < 50).any():
            raise AssertionError(f"{name} includes a trend with fewer than 50 niches")


def draw_tree_overlay(
    ax: plt.Axes,
    df: pd.DataFrame,
    x: str,
    y: str,
    node_col: str,
    edge_col: str,
    color: str = "#111111",
) -> None:
    if node_col not in df.columns or edge_col not in df.columns:
        return
    nodes = df.dropna(subset=[node_col, x, y]).groupby(node_col)[[x, y]].median()
    for _edge, edf in df.dropna(subset=[edge_col, node_col]).groupby(edge_col):
        node_ids = [n for n in pd.unique(edf[node_col]) if n in nodes.index]
        if len(node_ids) < 2:
            continue
        node_ids = sorted(node_ids, key=lambda n: nodes.loc[n, x])
        start, end = node_ids[0], node_ids[-1]
        ax.plot(
            [nodes.loc[start, x], nodes.loc[end, x]],
            [nodes.loc[start, y], nodes.loc[end, y]],
            color=color,
            lw=0.85,
            alpha=0.68,
            zorder=4,
        )


def draw_root_marker(ax: plt.Axes, df: pd.DataFrame, x: str, y: str, ptime: str) -> None:
    row = df.loc[df[ptime].idxmin()]
    ax.scatter([row[x]], [row[y]], marker="*", s=130, c="#111111", edgecolors="white", linewidths=0.6, zorder=6, label="root")


def plot_workflow_schematic(ax: plt.Axes) -> None:
    panel_letter(ax, "A", x=-0.02, y=1.03)
    ax.axis("off")
    stages = [
        ("image", "multiplexed\nwhole-slide field"),
        ("cells", "segmentation\nTier_A/Tier_B"),
        ("niches", "connected ductal\nepithelial units"),
        ("graph", "30 um cell graph\n+ 5-hop context"),
        ("features", "morphology\ntopology\nboundary\ncontext"),
        ("modules", "early duct\nPanIN-like\ndesmoplasia"),
        ("tree", "rooted principal\ntree + branches"),
        ("tissue", "pseudotime\nback-projection"),
    ]
    xs = np.linspace(0.05, 0.95, len(stages))
    colors = sns.color_palette("colorblind", n_colors=len(stages))
    for i, ((title, body), x0, color) in enumerate(zip(stages, xs, colors)):
        rect = mpl.patches.FancyBboxPatch(
            (x0 - 0.052, 0.44),
            0.104,
            0.28,
            boxstyle="round,pad=0.010,rounding_size=0.015",
            transform=ax.transAxes,
            facecolor=color,
            edgecolor="none",
            alpha=0.96,
        )
        ax.add_patch(rect)
        ax.text(x0, 0.64, title, ha="center", va="center", fontsize=8.0, fontweight="bold", color="white", transform=ax.transAxes)
        ax.text(x0, 0.52, body, ha="center", va="center", fontsize=6.7, color="white", linespacing=1.05, transform=ax.transAxes)
        if i < len(stages) - 1:
            ax.annotate(
                "",
                xy=(xs[i + 1] - 0.066, 0.58),
                xytext=(x0 + 0.064, 0.58),
                xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.0),
            )
    ax.text(
        0.02,
        0.22,
        "Observation unit: connected epithelial niche + graph-defined local surroundings",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color="#17324d",
    )
    ax.text(
        0.02,
        0.10,
        "Pseudotime is interpreted as a branch-resolved niche-state coordinate, not direct chronological lineage.",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=7.2,
        color="#444444",
    )


def plot_tier_a_counts(ax: plt.Axes, ann: pd.DataFrame, letter: str = "B") -> None:
    panel_letter(ax, letter)
    counts = ann["Tier_A"].value_counts().head(12).sort_values()
    colors = [PHENOTYPE_COLORS.get(label, "#bdbdbd") for label in counts.index]
    ax.barh([shorten(x, 28) for x in counts.index], counts.values, color=colors, edgecolor="none")
    ax.set_title("Sample 34434 cell compartment audit")
    ax.set_xlabel("cells")
    ax.set_ylabel("")
    ductal = int((ann["Tier_A"] == "pancreatic ductal epithelium").sum())
    ax.text(
        0.98,
        0.08,
        f"total cells: {len(ann):,}\nductal epithelial: {ductal:,}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#dddddd"),
    )
    clean_axis(ax)


def select_niche_components(exp2: pd.DataFrame) -> list[tuple[str, str, float]]:
    candidates = exp2.loc[exp2["n_cells"] >= 35].copy()
    selections = [
        ("early duct-like", candidates["pooled_pseudotime"].quantile(0.10)),
        ("remodeled/PanIN-like", candidates["pooled_pseudotime"].quantile(0.50)),
        ("desmoplastic", candidates["pooled_pseudotime"].quantile(0.90)),
    ]
    rows = []
    for label, target in selections:
        row = candidates.loc[(candidates["pooled_pseudotime"] - target).abs().idxmin()]
        rows.append((str(row[COMPONENT]), label, float(row["pooled_pseudotime"])))
    return rows


def draw_local_niche(ax: plt.Axes, spatial: pd.DataFrame, component: str, label: str, pseudotime: float, letter: str | None = None) -> None:
    if letter:
        panel_letter(ax, letter)
    comp = spatial.loc[spatial[COMPONENT].astype(str) == component].copy()
    if comp.empty:
        ax.axis("off")
        return
    pad = 230
    xmin, xmax = comp["x"].min() - pad, comp["x"].max() + pad
    ymin, ymax = comp["y"].min() - pad, comp["y"].max() + pad
    local = spatial.loc[spatial["x"].between(xmin, xmax) & spatial["y"].between(ymin, ymax)].copy()
    local = robust_sample(local, 4500, 11)
    ax.scatter(
        local["x"],
        local["y"],
        c=local["Tier_A"].map(PHENOTYPE_COLORS).fillna("#cccccc"),
        s=1.0,
        alpha=0.34,
        linewidths=0,
        rasterized=True,
    )
    coords = comp[["x", "y"]].to_numpy()
    degrees = np.zeros(len(coords), dtype=int)
    if len(coords) > 1:
        tree = cKDTree(coords)
        pairs = np.array(list(tree.query_pairs(30 / 0.325)))
        if len(pairs) > 900:
            rng = np.random.default_rng(4)
            pairs = pairs[rng.choice(len(pairs), size=900, replace=False)]
        for i, j in pairs:
            degrees[i] += 1
            degrees[j] += 1
            ax.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]], color="#111111", alpha=0.12, lw=0.35, zorder=2)
    boundary = degrees <= np.nanpercentile(degrees, 30) if len(degrees) else np.array([], dtype=bool)
    ax.scatter(comp["x"], comp["y"], c="#111111", s=4.2, alpha=0.80, linewidths=0, zorder=3, label="core")
    if len(boundary):
        ax.scatter(comp.loc[boundary, "x"], comp.loc[boundary, "y"], facecolors="none", edgecolors="#c1121f", s=15, linewidths=0.55, zorder=4, label="boundary")
    ax.set_title(f"{label}\n{len(comp):,} epithelial cells, pt={pseudotime:.1f}")
    clean_spatial_axis(ax)
    add_scale_bar(ax, length_um=50, pixel_size_um=0.325)


def plot_niche_examples(fig: plt.Figure, subspec, spatial: pd.DataFrame, exp2: pd.DataFrame, letter: str = "C") -> None:
    sub = GridSpecFromSubplotSpec(1, 3, subplot_spec=subspec, wspace=0.06)
    for i, (component, label, pt) in enumerate(select_niche_components(exp2)):
        ax = fig.add_subplot(sub[0, i])
        draw_local_niche(ax, spatial, component, label, pt, letter if i == 0 else None)


def plot_feature_grammar(ax: plt.Axes, feature_df: pd.DataFrame, letter: str = "D") -> None:
    panel_letter(ax, letter)
    blocks = feature_block_summary(feature_df)
    blocks = blocks.sort_values("n_features", ascending=True)
    colors = sns.color_palette("crest", n_colors=len(blocks))
    ax.barh([shorten(x, 24) for x in blocks["feature_family"]], blocks["n_features"], color=colors, edgecolor="none")
    for i, row in enumerate(blocks.itertuples()):
        ax.text(row.n_features + blocks["n_features"].max() * 0.02, i, f"{int(row.n_features)}", va="center", fontsize=6.3)
    ax.set_title("Feature grammar for structural niche pseudotime")
    ax.set_xlabel("feature columns")
    ax.set_ylabel("")
    clean_axis(ax)
    save_table(blocks, "feature_grammar_34434")


def plot_module_tree_schematic(ax: plt.Axes, letter: str = "E") -> None:
    panel_letter(ax, letter)
    ax.axis("off")
    ax.set_title("Root, branch assignment, and tissue projection")
    trunk = np.array([[0.12, 0.55], [0.32, 0.55], [0.48, 0.55], [0.62, 0.55]])
    branches = [
        np.array([[0.48, 0.55], [0.62, 0.73], [0.82, 0.82]]),
        np.array([[0.48, 0.55], [0.62, 0.38], [0.82, 0.30]]),
        np.array([[0.32, 0.55], [0.46, 0.73], [0.56, 0.88]]),
    ]
    ax.plot(trunk[:, 0], trunk[:, 1], color="#222222", lw=2, transform=ax.transAxes)
    for branch, color in zip(branches, ["#8c1d40", "#6a4c93", "#e17c05"]):
        ax.plot(branch[:, 0], branch[:, 1], color=color, lw=2, transform=ax.transAxes)
    ax.scatter([0.12], [0.55], marker="*", s=190, c="#111111", edgecolors="white", transform=ax.transAxes, zorder=5)
    ax.text(0.12, 0.42, "early-duct\nroot", ha="center", va="center", fontsize=7, transform=ax.transAxes)
    ax.text(0.50, 0.18, "distance from root = pseudotime\nbranch = local structural program", ha="center", va="center", fontsize=7.2, transform=ax.transAxes)
    tissue = mpl.patches.Rectangle((0.70, 0.50), 0.22, 0.20, facecolor="#f2f2f2", edgecolor="#cccccc", transform=ax.transAxes)
    ax.add_patch(tissue)
    rng = np.random.default_rng(2)
    ax.scatter(0.72 + rng.random(70) * 0.18, 0.52 + rng.random(70) * 0.16, c=rng.random(70), cmap=PSEUDOTIME_CMAP, s=7, transform=ax.transAxes, linewidths=0)
    ax.text(0.81, 0.75, "project state\nback to tissue", ha="center", va="bottom", fontsize=7, transform=ax.transAxes)


def plot_umap_tree(ax: plt.Axes, df: pd.DataFrame, letter: str, title: str, x: str = "UMAP1", y: str = "UMAP2", ptime: str = "pooled_pseudotime", node: str = "pooled_node_id", edge: str = "pooled_edge_id") -> None:
    panel_letter(ax, letter)
    show = robust_sample(df.dropna(subset=[x, y, ptime]), 18_000, 42)
    sc = ax.scatter(show[x], show[y], c=show[ptime], cmap=PSEUDOTIME_CMAP, s=1.5, alpha=0.72, linewidths=0, rasterized=True)
    draw_tree_overlay(ax, df, x, y, node, edge)
    draw_root_marker(ax, df, x, y, ptime)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.01)
    cb.set_label("pseudotime")
    ax.legend(frameon=False, loc="best", markerscale=0.7)


def plot_tissue_pseudotime(ax: plt.Axes, spatial: pd.DataFrame, letter: str = "B", title: str = "Tissue back-projection") -> None:
    panel_letter(ax, letter)
    bg = robust_sample(spatial, 90_000, 111)
    ax.scatter(bg["x"], bg["y"], c="#d5d5d5", s=0.07, alpha=0.20, linewidths=0, rasterized=True)
    niche = spatial.loc[spatial["has_pooled_niche"].fillna(False)].dropna(subset=["pooled_pseudotime"])
    show = robust_sample(niche, 140_000, 112)
    sc = ax.scatter(show["x"], show["y"], c=show["pooled_pseudotime"], cmap=PSEUDOTIME_CMAP, s=0.17, alpha=0.85, linewidths=0, rasterized=True)
    ax.set_title(title)
    clean_spatial_axis(ax)
    add_scale_bar(ax, length_um=1000, pixel_size_um=0.325)
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.01)
    cb.set_label("pseudotime")


def branch_time_module_table(df: pd.DataFrame, top_branches: int = 8, ptime: str = "pooled_pseudotime") -> pd.DataFrame:
    branch_order = df["major_branch"].value_counts().head(top_branches).index.tolist()
    rows = []
    for branch in branch_order:
        sub = df.loc[df["major_branch"] == branch].dropna(subset=[ptime]).copy()
        if len(sub) < 30:
            continue
        sub["time_bin"] = pd.qcut(sub[ptime], q=3, labels=["early", "mid", "late"], duplicates="drop")
        for time_bin, tdf in sub.groupby("time_bin", observed=False):
            if len(tdf) < 10:
                continue
            row = {"branch_time": f"{branch} | {time_bin}", "branch": branch, "time_bin": str(time_bin), "n": len(tdf)}
            for col, label in MODULES.items():
                if col in tdf.columns:
                    row[label] = tdf[col].median()
            rows.append(row)
    out = pd.DataFrame(rows)
    save_table(out, "fig2_single_sample_branch_time_modules")
    return out


def plot_branch_module_heatmap(ax: plt.Axes, df: pd.DataFrame, letter: str = "C") -> None:
    panel_letter(ax, letter)
    table = branch_time_module_table(df)
    plot_cols = [label for label in MODULES.values() if label in table.columns]
    mat = table.set_index("branch_time")[plot_cols]
    sns.heatmap(zscore(mat, axis=0), ax=ax, cmap="vlag", center=0, linewidths=0.15, cbar_kws={"label": "z-scored median"})
    ax.set_title("Branch-time pathology module states")
    ax.set_xlabel("")
    ax.set_ylabel("")


def plot_module_loess(ax: plt.Axes, df: pd.DataFrame, letter: str = "D", ptime: str = "pooled_pseudotime") -> None:
    panel_letter(ax, letter)
    rows = []
    for col, label in MODULES.items():
        if col not in df.columns:
            continue
        sub = df[[ptime, col]].dropna().sort_values(ptime)
        if len(sub) < 50:
            continue
        y = (sub[col] - sub[col].median()) / sub[col].std(ddof=0)
        sm = lowess(y, sub[ptime], frac=0.24, return_sorted=True)
        rho, pval = spearmanr(sub[ptime], sub[col])
        rows.append({"feature": col, "label": label, "n": len(sub), "spearman_r": rho, "spearman_p": pval})
        ax.plot(sm[:, 0], sm[:, 1], color=MODULE_COLORS.get(label, "#555555"), lw=1.55, label=f"{label} (rho={rho:.2f})")
    trend_stats = with_fdr(pd.DataFrame(rows))
    save_table(trend_stats, "fig2_single_sample_module_trends")
    branch_counts = df["major_branch"].value_counts().head(6).index.tolist()
    ymin, ymax = ax.get_ylim()
    rug_y = ymin + (ymax - ymin) * 0.035
    branch_palette = dict(zip(branch_counts, sns.color_palette(BRANCH_CMAP, n_colors=len(branch_counts))))
    for branch in branch_counts:
        values = df.loc[df["major_branch"] == branch, ptime].dropna()
        values = values.sample(min(130, len(values)), random_state=3)
        ax.scatter(values, np.full(len(values), rug_y), s=3, color=branch_palette[branch], alpha=0.55, linewidths=0, rasterized=True)
    ax.axhline(0, lw=0.6, color="#999999")
    ax.set_title("Core pathology module LOESS trends\nfull-context pseudotime with branch rug")
    ax.set_xlabel("pseudotime")
    ax.set_ylabel("z-scored module")
    ax.legend(frameon=False, ncol=2, loc="best")


def plot_ranked_trends(ax: plt.Axes, trend_df: pd.DataFrame, letter: str, title: str, top_n: int = 8) -> None:
    panel_letter(ax, letter)
    df = with_fdr(trend_df).copy()
    show = df.sort_values("spearman_r", key=lambda s: s.abs(), ascending=False).head(top_n).copy()
    show["short_label"] = [shorten(x, 34) for x in show["label"]]
    colors = ["#2a9d8f" if r >= 0 else "#d17a22" for r in show["spearman_r"]]
    sns.barplot(
        data=show,
        y="short_label",
        x="spearman_r",
        hue="short_label",
        palette=dict(zip(show["short_label"], colors)),
        legend=False,
        ax=ax,
    )
    ax.axvline(0, color="#333333", lw=0.7)
    max_abs = max(float(show["spearman_r"].abs().max()), 0.05)
    right_pad = max_abs * 1.85
    ax.set_xlim(-max_abs * 1.18, right_pad)
    for i, row in enumerate(show.itertuples()):
        ax.text(
            right_pad * 0.98,
            i,
            f"rho={row.spearman_r:.2f}\nq={row.fdr:.1e}",
            va="center",
            ha="right",
            fontsize=5.7,
        )
    ax.set_title(title)
    ax.set_xlabel("Spearman rho")
    ax.set_ylabel("")


def plot_single_context_contact_trends(ax: plt.Axes, data: dict[str, pd.DataFrame], letter: str = "E") -> None:
    panel_letter(ax, letter)
    micro = data["micro"].copy()
    contact = data["contact"].copy()
    micro["source"] = "surrounding context"
    contact["source"] = "ductal contact"
    comb = pd.concat([micro, contact], ignore_index=True)
    comb = with_fdr(comb)
    keep_labels = [
        "Fibroblast proportion",
        "Fibroblast FAP",
        "T-cell proportion",
        "Endothelial proportion",
        "Ductal cells near Fibroblasts",
        "Ductal cells near Vimentin only mesenchyme",
        "Ductal-T cells excess",
        "Ductal cells near Endothelial cells",
    ]
    show = comb.loc[comb["label"].isin(keep_labels)].copy()
    show = show.sort_values("spearman_r")
    colors = show["source"].map({"surrounding context": "#33658a", "ductal contact": "#8c1d40"})
    ax.scatter(show["spearman_r"], np.arange(len(show)), s=36, c=colors, zorder=3)
    for i, row in enumerate(show.itertuples()):
        ax.plot([0, row.spearman_r], [i, i], color="#bdbdbd", lw=0.8, zorder=1)
    ax.axvline(0, color="#333333", lw=0.7)
    max_abs = max(float(show["spearman_r"].abs().max()), 0.05)
    right_pad = max_abs * 1.92
    ax.set_xlim(-max_abs * 1.18, right_pad)
    for i, row in enumerate(show.itertuples()):
        ax.text(
            right_pad * 0.98,
            i,
            f"rho={row.spearman_r:.2f}  q={row.fdr:.1e}  n={int(row.n):,}",
            ha="right",
            va="center",
            fontsize=5.8,
        )
    ax.set_yticks(np.arange(len(show)))
    ax.set_yticklabels([shorten(x, 36) for x in show["label"]])
    ax.set_title("Selected full-context microenvironment and ductal-contact trends")
    ax.set_xlabel("Spearman rho")
    ax.set_ylabel("")
    save_table(show, "fig2_selected_context_contact_trends")


def state_card_components(exp2: pd.DataFrame) -> list[tuple[pd.Series, str]]:
    candidates = exp2.loc[exp2["n_cells"] >= 35].copy()
    targets = [("low pseudotime", 0.10), ("intermediate", 0.50), ("high pseudotime", 0.90)]
    rows = []
    for label, q in targets:
        target = candidates["pooled_pseudotime"].quantile(q)
        row = candidates.loc[(candidates["pooled_pseudotime"] - target).abs().idxmin()]
        rows.append((row, label))
    return rows


def plot_state_cards(fig: plt.Figure, subspec, spatial: pd.DataFrame, exp2: pd.DataFrame, letter: str = "F") -> None:
    outer = GridSpecFromSubplotSpec(1, 3, subplot_spec=subspec, wspace=0.16)
    selected = state_card_components(exp2)
    module_cols = [
        "pdac_early_duct_anchor_score",
        "pdac_panin_like_dysplasia_score",
        "pdac_invasion_desmoplasia_axis",
        "pdac_proliferation_axis",
        "pdac_dedifferentiation_axis",
    ]
    module_labels = ["duct", "PanIN", "desmo", "Ki67", "dediff"]
    module_values = exp2[module_cols].astype(float)
    module_z = (module_values - module_values.mean()) / module_values.std(ddof=0)
    module_z.index = exp2.index
    for i, (row, label) in enumerate(selected):
        sub = GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[0, i], height_ratios=[1.0, 0.38, 0.42], hspace=0.18)
        ax_crop = fig.add_subplot(sub[0, 0])
        draw_local_niche(ax_crop, spatial, str(row[COMPONENT]), f"{label}\npt={row['pooled_pseudotime']:.1f}", float(row["pooled_pseudotime"]), letter if i == 0 else None)
        ax_mod = fig.add_subplot(sub[1, 0])
        vals = module_z.loc[row.name, module_cols].clip(-2.5, 2.5)
        ax_mod.bar(module_labels, vals, color=[MODULE_COLORS["early duct"], MODULE_COLORS["PanIN-like"], MODULE_COLORS["invasion/desmoplasia"], MODULE_COLORS["proliferation"], MODULE_COLORS["dedifferentiation"]])
        ax_mod.axhline(0, color="#777777", lw=0.55)
        ax_mod.set_ylim(-2.6, 2.6)
        ax_mod.set_ylabel("module z", fontsize=5.8)
        ax_mod.tick_params(axis="x", rotation=35, labelsize=5.6)
        ax_mod.tick_params(axis="y", labelsize=5.6)
        ax_ctx = fig.add_subplot(sub[2, 0])
        ctx_cols = {
            "surround_prop__Fibroblasts": "fib",
            "surround_prop__T_cells": "T",
            "surround_prop__B_lineage": "B",
            "surround_prop__Endothelial_cells": "endo",
        }
        vals = [float(row.get(col, 0) or 0) for col in ctx_cols]
        ax_ctx.bar(list(ctx_cols.values()), vals, color=["#2a9d8f", "#d64f4f", "#edc948", "#59a14f"])
        ax_ctx.set_ylim(0, max(0.55, np.nanmax(vals) * 1.2))
        ax_ctx.set_ylabel("surround\nfraction", fontsize=5.8)
        ax_ctx.tick_params(axis="x", rotation=35, labelsize=5.6)
        ax_ctx.tick_params(axis="y", labelsize=5.6)


def plot_sample_counts(ax: plt.Axes, pooled: pd.DataFrame, letter: str = "A") -> None:
    panel_letter(ax, letter)
    counts = pooled.groupby(["sample_id", "disease_group"]).size().reset_index(name="n_niches")
    counts["sample_label"] = counts["sample_id"].map(SAMPLE_LABELS)
    sns.barplot(data=counts, y="sample_label", x="n_niches", hue="disease_group", dodge=False, palette=DISEASE_COLORS, ax=ax)
    ax.set_title("Niche counts by sample and disease group")
    ax.set_xlabel("ductal epithelial niches")
    ax.set_ylabel("")
    ax.legend(frameon=False, title="")
    clean_axis(ax)
    save_table(counts, "fig3_sample_niche_counts")


def plot_pooled_embedding_sample(ax: plt.Axes, pooled: pd.DataFrame, letter: str = "B") -> None:
    panel_letter(ax, letter)
    show = robust_sample(pooled, 28_000, 22)
    for sample, sdf in show.groupby("sample_id"):
        ax.scatter(sdf["UMAP1"], sdf["UMAP2"], s=1.2, alpha=0.58, linewidths=0, color=SAMPLE_COLORS.get(sample, "#999999"), label=SAMPLE_LABELS.get(sample, sample), rasterized=True)
    ax.set_title("Pooled embedding by sample")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.legend(frameon=False, markerscale=3, loc="best")


def plot_pooled_embedding_pt_branch(fig: plt.Figure, subspec, pooled: pd.DataFrame, letter: str = "C") -> None:
    sub = GridSpecFromSubplotSpec(1, 2, subplot_spec=subspec, wspace=0.16)
    ax1 = fig.add_subplot(sub[0, 0])
    panel_letter(ax1, letter)
    show = robust_sample(pooled, 28_000, 25)
    sc = ax1.scatter(show["UMAP1"], show["UMAP2"], c=show["pooled_pseudotime"], cmap=PSEUDOTIME_CMAP, s=1.2, alpha=0.72, linewidths=0, rasterized=True)
    draw_tree_overlay(ax1, pooled, "UMAP1", "UMAP2", "pooled_node_id", "pooled_edge_id")
    draw_root_marker(ax1, pooled, "UMAP1", "UMAP2", "pooled_pseudotime")
    ax1.set_title("pseudotime")
    ax1.set_xlabel("UMAP1")
    ax1.set_ylabel("UMAP2")
    plt.colorbar(sc, ax=ax1, fraction=0.046, pad=0.01, label="pseudotime")
    ax2 = fig.add_subplot(sub[0, 1])
    branch_order = pooled["major_branch"].value_counts().head(9).index.tolist()
    branch_show = show["major_branch"].where(show["major_branch"].isin(branch_order), "other")
    palette = dict(zip(branch_order + ["other"], sns.color_palette(BRANCH_CMAP, n_colors=len(branch_order) + 1)))
    ax2.scatter(show["UMAP1"], show["UMAP2"], c=branch_show.map(palette), s=1.2, alpha=0.68, linewidths=0, rasterized=True)
    ax2.set_title("major branch")
    ax2.set_xlabel("UMAP1")
    ax2.set_ylabel("UMAP2")


def plot_branch_occupancy(ax: plt.Axes, pooled: pd.DataFrame, letter: str = "D") -> None:
    panel_letter(ax, letter)
    counts = pooled.groupby(["major_branch", "sample_id"]).size().unstack(fill_value=0)
    counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).head(11).index]
    frac = counts.div(counts.sum(axis=1), axis=0)
    frac.rename(columns=SAMPLE_LABELS).plot(kind="barh", stacked=True, ax=ax, width=0.82, color=[SAMPLE_COLORS.get(c, "#999999") for c in counts.columns])
    ax.set_title("Branch occupancy by sample")
    ax.set_xlabel("fraction within branch")
    ax.set_ylabel("")
    ax.legend(frameon=False, title="", loc="lower right")
    save_table(frac.reset_index(), "fig3_branch_occupancy_by_sample")


def plot_pooled_branch_module_heatmap(ax: plt.Axes, pooled: pd.DataFrame, letter: str = "E") -> None:
    panel_letter(ax, letter)
    cols = {col: label for col, label in MODULES.items() if col in pooled.columns}
    order = pooled["major_branch"].value_counts().head(10).index
    med = pooled.loc[pooled["major_branch"].isin(order)].groupby("major_branch")[list(cols)].median().rename(columns=cols).loc[order]
    sns.heatmap(zscore(med, axis=0).T, ax=ax, cmap="vlag", center=0, linewidths=0.15, cbar_kws={"label": "branch z-score"})
    ax.set_title("Branch-module heatmap")
    ax.set_xlabel("major branch")
    ax.set_ylabel("")
    save_table(med.reset_index(), "fig3_branch_module_medians")


def plot_four_sample_tissue_maps(fig: plt.Figure, subspec, letter: str = "F") -> None:
    sub = GridSpecFromSubplotSpec(2, 2, subplot_spec=subspec, wspace=0.02, hspace=0.10)
    base = ROOT / "data" / "combined_exp_2_3_4_5"
    for i, sample in enumerate(["40331_1", "34434_1", "33694_1", "35559_1"]):
        ax = fig.add_subplot(sub[i // 2, i % 2])
        if i == 0:
            panel_letter(ax, letter)
        spatial = pd.read_pickle(base / f"spatial_cells_auto_branch_n24_v3_with_epithelial_{sample}.pkl")
        bg = robust_sample(spatial, 35_000, 31 + i)
        ax.scatter(bg["x"], bg["y"], c="#dddddd", s=0.05, alpha=0.17, linewidths=0, rasterized=True)
        niche = spatial.loc[spatial["has_pooled_niche"].fillna(False)].dropna(subset=["pooled_pseudotime"])
        show = robust_sample(niche, 48_000, 36 + i)
        ax.scatter(show["x"], show["y"], c=show["pooled_pseudotime"], cmap=PSEUDOTIME_CMAP, s=0.10, alpha=0.80, linewidths=0, rasterized=True)
        ax.set_title(SAMPLE_LABELS.get(sample, sample), fontsize=7.1)
        clean_spatial_axis(ax)
        if i in {0, 2}:
            add_scale_bar(ax, length_um=1000, pixel_size_um=0.325)


def plot_xenium_audit(ax: plt.Axes, data: dict[str, pd.DataFrame], letter: str = "A") -> None:
    panel_letter(ax, letter)
    audit = data["xen_audit"].copy()
    niches = data["xenium"].groupby("sample_id").size().rename("n_niches")
    audit = audit.merge(niches, on="sample_id", how="left")
    audit["label"] = audit["sample_id"].map(XENIUM_LABELS)
    audit = audit.sort_values("n_cells_matrix")
    y = np.arange(len(audit))
    ax.barh(y, audit["n_cells_matrix"], height=0.56, color=[XENIUM_SAMPLE_COLORS.get(s, "#999999") for s in audit["sample_id"]], label="cells")
    ax.set_yticks(y)
    ax.set_yticklabels(audit["label"])
    ax.set_title("Xenium sample audit")
    ax.set_xlabel("cells")
    ax.set_ylabel("")
    ax2 = ax.twiny()
    ax2.scatter(audit["n_niches"], y, s=34, color="#d6a65a", edgecolor="#111111", linewidth=0.4, zorder=4, label="epithelial niches")
    ax2.set_xlabel("epithelial niches")
    ax2.tick_params(axis="x", labelsize=6.0)
    for yi, row in enumerate(audit.itertuples()):
        ax2.text(row.n_niches + audit["n_niches"].max() * 0.03, yi, f"{int(row.n_niches):,}", va="center", fontsize=6.0, color="#5c3d00")
    clean_axis(ax)
    save_table(audit[["sample_id", "disease_group", "n_cells_matrix", "n_genes", "n_niches", "median_genes_per_cell", "median_transcripts_per_cell"]], "fig4_xenium_sample_audit")


def plot_xenium_feature_blocks(ax: plt.Axes, features: pd.DataFrame, letter: str = "B") -> None:
    panel_letter(ax, letter)
    counts = features["feature_block"].value_counts().sort_values()
    ax.barh([shorten(x, 32) for x in counts.index], counts.values, color=sns.color_palette("flare", n_colors=len(counts)))
    for i, v in enumerate(counts.values):
        ax.text(v + counts.max() * 0.02, i, str(int(v)), va="center", fontsize=6.3)
    ax.set_title("Modality-adapted Xenium feature blocks")
    ax.set_xlabel("selected features")
    ax.set_ylabel("")
    clean_axis(ax)
    save_table(counts.rename_axis("feature_block").reset_index(name="n_features"), "fig4_xenium_feature_blocks")


def plot_xenium_embedding(fig: plt.Figure, subspec, xenium: pd.DataFrame, letter: str = "C") -> None:
    sub = GridSpecFromSubplotSpec(1, 3, subplot_spec=subspec, wspace=0.12)
    show = robust_sample(xenium, 5_074, 41)
    ax1 = fig.add_subplot(sub[0, 0])
    panel_letter(ax1, letter)
    for sample, sdf in show.groupby("sample_id"):
        ax1.scatter(sdf["UMAP1_sample_centered"], sdf["UMAP2_sample_centered"], s=4.2, alpha=0.70, linewidths=0, color=XENIUM_SAMPLE_COLORS.get(sample, "#999999"), label=XENIUM_LABELS.get(sample, sample))
    ax1.set_title("sample")
    ax1.set_xlabel("UMAP1")
    ax1.set_ylabel("UMAP2")
    ax1.legend(frameon=False, markerscale=1.4, loc="best")
    ax2 = fig.add_subplot(sub[0, 1])
    sc = ax2.scatter(show["UMAP1_sample_centered"], show["UMAP2_sample_centered"], c=show["xenium_pseudotime_sample_centered_norm"], cmap=PSEUDOTIME_CMAP, s=4.2, alpha=0.76, linewidths=0)
    draw_tree_overlay(ax2, xenium, "UMAP1_sample_centered", "UMAP2_sample_centered", "xenium_node_id_sample_centered", "xenium_edge_id_sample_centered")
    draw_root_marker(ax2, xenium, "UMAP1_sample_centered", "UMAP2_sample_centered", "xenium_pseudotime_sample_centered_norm")
    ax2.set_title("sample-centered pseudotime")
    ax2.set_xlabel("UMAP1")
    ax2.set_ylabel("UMAP2")
    plt.colorbar(sc, ax=ax2, fraction=0.046, pad=0.01, label="normalized pseudotime")
    ax3 = fig.add_subplot(sub[0, 2])
    order = xenium["major_branch"].value_counts().head(8).index.tolist()
    branch = show["major_branch"].where(show["major_branch"].isin(order), "other")
    palette = dict(zip(order + ["other"], sns.color_palette(BRANCH_CMAP, n_colors=len(order) + 1)))
    ax3.scatter(show["UMAP1_sample_centered"], show["UMAP2_sample_centered"], c=branch.map(palette), s=4.2, alpha=0.74, linewidths=0)
    ax3.set_title("major branch")
    ax3.set_xlabel("UMAP1")
    ax3.set_ylabel("UMAP2")


def plot_xenium_spatial_maps(fig: plt.Figure, subspec, xenium: pd.DataFrame, letter: str = "D") -> None:
    sub = GridSpecFromSubplotSpec(2, 2, subplot_spec=subspec, wspace=0.03, hspace=0.10)
    base = ROOT / "data" / "xenium_pancreas_10x" / "niche_features"
    niche_pt = xenium[[XENIUM_COMPONENT, "xenium_pseudotime_sample_centered_norm"]].drop_duplicates()
    import anndata as ad

    for i, sample in enumerate(["normal_nondiseased_v1", "pdac_pancreas_v1", "pdac_addon_v1", "pdac_io_v1"]):
        ax = fig.add_subplot(sub[i // 2, i % 2])
        if i == 0:
            panel_letter(ax, letter)
        adata = ad.read_h5ad(base / f"{sample}_with_niches.h5ad", backed="r")
        obs = adata.obs[[XENIUM_COMPONENT, "x_centroid", "y_centroid"]].copy()
        adata.file.close()
        obs = obs.merge(niche_pt, on=XENIUM_COMPONENT, how="left")
        bg = robust_sample(obs, 45_000, 50 + i)
        ax.scatter(bg["x_centroid"], bg["y_centroid"], c="#d7d7d7", s=0.05, alpha=0.17, linewidths=0, rasterized=True)
        show = robust_sample(obs.dropna(subset=["xenium_pseudotime_sample_centered_norm"]), 45_000, 55 + i)
        ax.scatter(show["x_centroid"], show["y_centroid"], c=show["xenium_pseudotime_sample_centered_norm"], cmap=PSEUDOTIME_CMAP, s=0.09, alpha=0.80, linewidths=0, rasterized=True)
        ax.set_title(XENIUM_LABELS.get(sample, sample), fontsize=7.1)
        clean_spatial_axis(ax)
        if i in {0, 2}:
            add_scale_bar(ax, length_um=1000, pixel_size_um=1)


def plot_xenium_branch_biology(fig: plt.Figure, subspec, data: dict[str, pd.DataFrame], letter: str = "E") -> None:
    sub = GridSpecFromSubplotSpec(1, 2, subplot_spec=subspec, width_ratios=[1.25, 1.0], wspace=0.16)
    ax1 = fig.add_subplot(sub[0, 0])
    panel_letter(ax1, letter)
    branch = data["xen_branch"]
    cols = [
        "histology__normal_duct_like_score__z_enrichment",
        "histology__adm_panin_like_score__z_enrichment",
        "histology__desmoplastic_tumor_score__z_enrichment",
        "histology__immune_inflamed_score__z_enrichment",
        "histology__immune_exclusion_score__z_enrichment",
        "histology__gland_poor_undifferentiated_score__z_enrichment",
    ]
    labels = ["normal duct", "ADM/PanIN", "desmoplastic", "immune inflamed", "immune excluded", "gland-poor"]
    heat = branch.set_index("branch")[cols].rename(columns=dict(zip(cols, labels))).T
    sns.heatmap(heat, ax=ax1, cmap="vlag", center=0, linewidths=0.1, cbar_kws={"label": "z-enrichment"})
    ax1.set_title("branch biology enrichment")
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    ax2 = fig.add_subplot(sub[0, 1])
    occ = data["xen_branch_occ"].pivot_table(index="major_branch", columns="clinical_progression_label", values="fraction_within_clinical_context", fill_value=0)
    occ = occ.loc[occ.sum(axis=1).sort_values(ascending=False).head(10).index]
    occ = occ.rename(
        columns={
            "Normal pancreas": "normal",
            "Grade I-II, 50% tumor": "G1-2 50%",
            "Stage III adenocarcinoma": "stage III",
            "Stage IIB, Grade 3 PDAC": "stage IIB G3",
        }
    )
    sns.heatmap(
        occ,
        ax=ax2,
        cmap="YlGnBu",
        linewidths=0.15,
        cbar_kws={"label": "fraction within context"},
    )
    ax2.set_title("branch occupancy\nby clinical context")
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    ax2.tick_params(axis="x", rotation=35, labelsize=5.6)
    ax2.tick_params(axis="y", labelsize=5.8)
    save_table(heat.T.reset_index(), "fig4_xenium_branch_biology")


def plot_xenium_sensitivity(ax: plt.Axes, xenium: pd.DataFrame, letter: str = "G") -> None:
    panel_letter(ax, letter)
    sub = xenium[["xenium_pseudotime_sample_centered_norm", "xenium_pseudotime_intrinsic_sample_centered_norm", "sample_id"]].dropna()
    for sample, sdf in sub.groupby("sample_id"):
        ax.scatter(
            sdf["xenium_pseudotime_sample_centered_norm"],
            sdf["xenium_pseudotime_intrinsic_sample_centered_norm"],
            s=8,
            alpha=0.55,
            linewidths=0,
            color=XENIUM_SAMPLE_COLORS.get(sample, "#999999"),
            label=XENIUM_LABELS.get(sample, sample),
            rasterized=True,
        )
    rho, pval = spearmanr(sub["xenium_pseudotime_sample_centered_norm"], sub["xenium_pseudotime_intrinsic_sample_centered_norm"])
    ax.text(0.04, 0.96, f"rho={rho:.2f}\np={pval:.1e}\nn={len(sub):,}", transform=ax.transAxes, ha="left", va="top", fontsize=7, bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#dddddd"))
    ax.set_title("Sensitivity: full-context versus\nepithelial-intrinsic pseudotime")
    ax.set_xlabel("full contextual sample-centered pseudotime")
    ax.set_ylabel("epithelial-intrinsic sample-centered pseudotime")
    ax.legend(frameon=False, loc="lower right", markerscale=1.4)
    save_table(pd.DataFrame([{"comparison": "xenium full-context vs epithelial-intrinsic", "n": len(sub), "spearman_r": rho, "spearman_p": pval}]), "fig4_xenium_sensitivity")


def make_fig1(data: dict[str, pd.DataFrame]) -> plt.Figure:
    fig = plt.figure(figsize=(15.8, 10.0), constrained_layout=True)
    gs = fig.add_gridspec(3, 4, height_ratios=[0.90, 1.12, 0.95], width_ratios=[1.05, 1.0, 1.0, 1.0])
    plot_workflow_schematic(fig.add_subplot(gs[0, :]))
    plot_tier_a_counts(fig.add_subplot(gs[1, 0]), data["ann"], "B")
    plot_niche_examples(fig, gs[1, 1:4], data["spatial"], data["exp2"], "C")
    plot_feature_grammar(fig.add_subplot(gs[2, 0:2]), data["exp2_features"], "D")
    plot_module_tree_schematic(fig.add_subplot(gs[2, 2:4]), "E")
    fig.suptitle("Figure 1. SpatioEv data model for epithelial niche pseudotime", x=0.01, ha="left", fontsize=13, fontweight="bold")
    return fig


def make_fig2(data: dict[str, pd.DataFrame]) -> plt.Figure:
    fig = plt.figure(figsize=(15.8, 13.4), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.05, 1.05, 1.25], width_ratios=[1.0, 1.0, 1.08])
    plot_umap_tree(fig.add_subplot(gs[0, 0]), data["exp2"], "A", "Single-sample UMAP with principal-tree overlay")
    plot_tissue_pseudotime(fig.add_subplot(gs[0, 1]), data["spatial"], "B", "Epithelial niches in tissue coordinates")
    plot_branch_module_heatmap(fig.add_subplot(gs[0, 2]), data["exp2"], "C")
    plot_module_loess(fig.add_subplot(gs[1, 0:2]), data["exp2"], "D")
    plot_single_context_contact_trends(fig.add_subplot(gs[1, 2]), data, "E")
    plot_state_cards(fig, gs[2, :], data["spatial"], data["exp2"], "F")
    fig.suptitle("Figure 2. A representative PDAC sample reveals branch-resolved epithelial niche states", x=0.01, ha="left", fontsize=13, fontweight="bold")
    return fig


def make_fig3(data: dict[str, pd.DataFrame]) -> plt.Figure:
    fig = plt.figure(figsize=(16.0, 13.6), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[0.95, 1.10, 1.05], width_ratios=[1.0, 1.08, 1.08])
    plot_sample_counts(fig.add_subplot(gs[0, 0]), data["pooled"], "A")
    plot_pooled_embedding_sample(fig.add_subplot(gs[0, 1]), data["pooled"], "B")
    plot_pooled_embedding_pt_branch(fig, gs[0, 2], data["pooled"], "C")
    plot_branch_occupancy(fig.add_subplot(gs[1, 0]), data["pooled"], "D")
    plot_pooled_branch_module_heatmap(fig.add_subplot(gs[1, 1]), data["pooled"], "E")
    plot_four_sample_tissue_maps(fig, gs[1, 2], "F")
    plot_ranked_trends(fig.add_subplot(gs[2, 0:2]), data["micro"], "G", "Ranked contextual trend tests\nfull-context pseudotime", top_n=8)
    save_table(data["micro"], "fig3_contextual_trends_with_fdr")
    plot_ranked_trends(fig.add_subplot(gs[2, 2]), data["contact"], "H", "Ranked ductal contact trend tests", top_n=8)
    save_table(data["contact"], "fig3_ductal_contact_trends_with_fdr")
    fig.suptitle("Figure 3. Pooled multiplexed atlas separates conserved and sample-specific ductal niche programs", x=0.01, ha="left", fontsize=13, fontweight="bold")
    return fig


def make_fig4(data: dict[str, pd.DataFrame]) -> plt.Figure:
    fig = plt.figure(figsize=(16.0, 13.4), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[0.90, 1.15, 1.05], width_ratios=[1.0, 1.0, 1.1])
    plot_xenium_audit(fig.add_subplot(gs[0, 0]), data, "A")
    plot_xenium_feature_blocks(fig.add_subplot(gs[0, 1]), data["xen_features"], "B")
    plot_xenium_embedding(fig, gs[0, 2], data["xenium"], "C")
    plot_xenium_spatial_maps(fig, gs[1, 0:2], data["xenium"], "D")
    plot_xenium_branch_biology(fig, gs[1, 2], data, "E")
    plot_ranked_trends(fig.add_subplot(gs[2, 0:2]), data["xen_micro"], "F", "Xenium microenvironment trend tests\nsample-centered pseudotime", top_n=9)
    save_table(data["xen_micro"], "fig4_xenium_microenvironment_trends_with_fdr")
    plot_xenium_sensitivity(fig.add_subplot(gs[2, 2]), data["xenium"], "G")
    fig.suptitle("Figure 4. Xenium transfers niche pseudotime to spatial transcriptomics with modality-aware features", x=0.01, ha="left", fontsize=13, fontweight="bold")
    return fig


def export_panel(name: str, plotter, figsize: tuple[float, float]) -> None:
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    result = plotter(fig)
    if result is None:
        pass
    export_figure(fig, PANEL_DIR / name)
    plt.close(fig)


def make_panel_exports(data: dict[str, pd.DataFrame]) -> None:
    export_panel("Fig1A_workflow", lambda fig: plot_workflow_schematic(fig.add_subplot(111)), (10.5, 2.2))
    export_panel("Fig1B_cell_compartment_audit", lambda fig: plot_tier_a_counts(fig.add_subplot(111), data["ann"], "B"), (4.8, 3.8))
    export_panel("Fig1C_niche_examples", lambda fig: plot_niche_examples(fig, fig.add_gridspec(1, 1)[0, 0], data["spatial"], data["exp2"], "C"), (8.5, 3.2))
    export_panel("Fig1D_feature_grammar", lambda fig: plot_feature_grammar(fig.add_subplot(111), data["exp2_features"], "D"), (6.5, 3.5))
    export_panel("Fig1E_module_tree_schematic", lambda fig: plot_module_tree_schematic(fig.add_subplot(111), "E"), (6.5, 3.5))
    export_panel("Fig2A_single_sample_umap_tree", lambda fig: plot_umap_tree(fig.add_subplot(111), data["exp2"], "A", "Single-sample UMAP with tree overlay"), (4.8, 4.0))
    export_panel("Fig2B_single_sample_tissue_pseudotime", lambda fig: plot_tissue_pseudotime(fig.add_subplot(111), data["spatial"], "B", "Tissue back-projection"), (4.8, 4.2))
    export_panel("Fig2C_branch_module_heatmap", lambda fig: plot_branch_module_heatmap(fig.add_subplot(111), data["exp2"], "C"), (5.4, 4.2))
    export_panel("Fig2D_module_loess", lambda fig: plot_module_loess(fig.add_subplot(111), data["exp2"], "D"), (7.5, 3.8))
    export_panel("Fig2E_context_contact_trends", lambda fig: plot_single_context_contact_trends(fig.add_subplot(111), data, "E"), (5.8, 4.2))
    export_panel("Fig2F_state_cards", lambda fig: plot_state_cards(fig, fig.add_gridspec(1, 1)[0, 0], data["spatial"], data["exp2"], "F"), (10.5, 4.4))
    export_panel("Fig3A_sample_counts", lambda fig: plot_sample_counts(fig.add_subplot(111), data["pooled"], "A"), (4.8, 3.6))
    export_panel("Fig3B_pooled_embedding_sample", lambda fig: plot_pooled_embedding_sample(fig.add_subplot(111), data["pooled"], "B"), (4.8, 4.0))
    export_panel("Fig3C_pooled_pseudotime_branch", lambda fig: plot_pooled_embedding_pt_branch(fig, fig.add_gridspec(1, 1)[0, 0], data["pooled"], "C"), (7.8, 3.8))
    export_panel("Fig3D_branch_occupancy", lambda fig: plot_branch_occupancy(fig.add_subplot(111), data["pooled"], "D"), (5.4, 4.0))
    export_panel("Fig3E_branch_module_heatmap", lambda fig: plot_pooled_branch_module_heatmap(fig.add_subplot(111), data["pooled"], "E"), (5.4, 4.0))
    export_panel("Fig3F_four_sample_tissue_maps", lambda fig: plot_four_sample_tissue_maps(fig, fig.add_gridspec(1, 1)[0, 0], "F"), (6.8, 5.2))
    export_panel("Fig3G_context_trends", lambda fig: plot_ranked_trends(fig.add_subplot(111), data["micro"], "G", "Contextual trend tests", 8), (6.3, 4.0))
    export_panel("Fig3H_contact_trends", lambda fig: plot_ranked_trends(fig.add_subplot(111), data["contact"], "H", "Ductal contact trend tests", 8), (6.3, 4.0))
    export_panel("Fig4A_xenium_audit", lambda fig: plot_xenium_audit(fig.add_subplot(111), data, "A"), (5.4, 3.8))
    export_panel("Fig4B_xenium_feature_blocks", lambda fig: plot_xenium_feature_blocks(fig.add_subplot(111), data["xen_features"], "B"), (5.4, 3.8))
    export_panel("Fig4C_xenium_embedding", lambda fig: plot_xenium_embedding(fig, fig.add_gridspec(1, 1)[0, 0], data["xenium"], "C"), (9.5, 3.4))
    export_panel("Fig4D_xenium_spatial_maps", lambda fig: plot_xenium_spatial_maps(fig, fig.add_gridspec(1, 1)[0, 0], data["xenium"], "D"), (8.6, 5.2))
    export_panel("Fig4E_xenium_branch_biology", lambda fig: plot_xenium_branch_biology(fig, fig.add_gridspec(1, 1)[0, 0], data, "E"), (8.4, 3.8))
    export_panel("Fig4F_xenium_microenvironment_trends", lambda fig: plot_ranked_trends(fig.add_subplot(111), data["xen_micro"], "F", "Xenium microenvironment trend tests", 9), (6.3, 4.2))
    export_panel("Fig4G_xenium_sensitivity", lambda fig: plot_xenium_sensitivity(fig.add_subplot(111), data["xenium"], "G"), (4.8, 4.0))


def main() -> None:
    configure()
    MAIN_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    assert_plot_inputs(data)

    figures = [
        ("Fig1_workflow_scaffold", make_fig1(data)),
        ("Fig2_single_sample_trajectory", make_fig2(data)),
        ("Fig3_pooled_multiplexed_atlas", make_fig3(data)),
        ("Fig4_xenium_transfer", make_fig4(data)),
    ]
    for name, fig in figures:
        export_figure(fig, MAIN_DIR / name)
        plt.close(fig)
        print(MAIN_DIR / f"{name}.png")

    make_panel_exports(data)
    print(f"Wrote panel files to {PANEL_DIR}")
    print(f"Wrote statistics tables to {TABLE_DIR}")


if __name__ == "__main__":
    main()
