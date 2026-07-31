from pathlib import Path

import nbformat as nbf


ROOT = Path("/Users/shihongwu/SpatioEv")
NOTEBOOK = ROOT / "notebooks" / "08_trajectory_microenvironment_interactions.ipynb"


def md(text):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text):
    return nbf.v4.new_code_cell(text.strip() + "\n")


cells = [
    md(
        """
# Trajectory Microenvironment and Communication Atlas

This notebook asks what changes in fibroblasts and immune cells along the SpatioEv epithelial-niche trajectories, and which spatially local ligand-receptor interaction proxies change along the Xenium trajectory.

Method framing:

- For neighborhood/state changes, we use niche-centered surrounding-cell summaries and branch/pseudotime trend analysis.
- For spatial cell-cell communication, we use a CellPhoneDB/CellChat-style ligand-receptor expression product, but constrain the source/target expression to cells in each epithelial niche and its local graph surround. This is a lightweight approximation to spatial CCC methods such as Squidpy ligand-receptor tests and COMMOT spatially constrained communication.
- For downstream interpretation, we treat the ligand-receptor results as candidates for validation, in the spirit of NicheNet/CCC workflows, not as proof of signaling.

Key references:

- Squidpy neighborhood enrichment and ligand-receptor analysis: https://doi.org/10.1038/s41592-021-01358-2
- CellPhoneDB protocol: https://doi.org/10.1038/s41596-020-0292-x
- CellChat: https://doi.org/10.1038/s41467-021-21246-9
- COMMOT: https://doi.org/10.1038/s41592-022-01728-4
- NicheNet: https://doi.org/10.1038/s41592-019-0667-5
"""
    ),
    code(
        """
%matplotlib inline

from pathlib import Path
import json
import gc
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import sparse
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.neighbors import BallTree

import scanpy as sc
import spatioev as se

plt.rcParams.update({
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
})
sns.set_style("white")

ROOT = Path("/Users/shihongwu/SpatioEv")
OUTPUT_DIR = ROOT / "notebooks" / "results" / "trajectory_microenvironment_interactions"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
for d in [OUTPUT_DIR, FIG_DIR, TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MI_DIR = ROOT / "data" / "combined_exp_2_3_4_5"
XENIUM_DIR = ROOT / "data" / "xenium_pancreas_10x"
XENIUM_NICHE_DIR = XENIUM_DIR / "niche_features"
XENIUM_PTIME_DIR = XENIUM_DIR / "pseudotime"

MI_KEY = "pancreatic ductal epithelium_mask_component"
XENIUM_KEY = "xenium_ductal_epithelium_component"
RANDOM_STATE = 42

def save_df(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_pickle(path)
    return path

def zscore_series(values):
    values = pd.to_numeric(values, errors="coerce")
    sd = values.std(ddof=0)
    if not np.isfinite(sd) or np.isclose(sd, 0):
        return pd.Series(np.nan, index=values.index)
    return (values - values.mean()) / sd

def safe_label(value):
    return str(value).replace(" ", "_").replace("/", "_")

def clean_label(label):
    out = str(label)
    replacements = {
        "surround_prop__": "",
        "surround__": "",
        "__mean": "",
        "_expr_z": "",
        "histology__": "",
        "xenium_": "",
        "pdac_": "",
        "panin_validation__": "",
        "_score": "",
    }
    for old, new in replacements.items():
        out = out.replace(old, new)
    out = out.replace("__", " ").replace("_", " ")
    return out

def zmean(df, columns):
    columns = [c for c in columns if c in df.columns]
    if len(columns) == 0:
        return pd.Series(np.nan, index=df.index)
    return pd.concat([zscore_series(df[c]) for c in columns], axis=1).mean(axis=1, skipna=True)
"""
    ),
    md("## Load trajectory and feature tables"),
    code(
        """
mi_result_df = pd.read_pickle(MI_DIR / "pooled_niche_result_df.pkl")
mi_feature_df = pd.read_pickle(MI_DIR / "pooled_pathology_with_panin_validation_scores.pkl")
mi_epi_df = pd.read_pickle(MI_DIR / "epithelial_intrinsic_pseudotime_result_df.pkl")

xenium_result_df = pd.read_pickle(XENIUM_PTIME_DIR / "xenium_pseudotime_result_df.pkl")
xenium_feature_df = pd.read_pickle(XENIUM_DIR / "pooled_xenium_niche_feature_df.pkl")
xenium_branch_summary_df = pd.read_csv(XENIUM_PTIME_DIR / "xenium_branch_biology_summary.csv")

mi_traj_cols = [
    MI_KEY, "image_id", "sample_id", "disease_group", "differentiation_label",
    "pooled_pseudotime", "simple_pseudotime", "major_branch",
    "pdac_early_duct_anchor_score", "pdac_panin_like_dysplasia_score",
    "pdac_invasion_desmoplasia_axis", "pdac_proliferation_axis",
    "pdac_dedifferentiation_axis",
]
mi_traj_cols = [c for c in mi_traj_cols if c in mi_result_df.columns]
mi_context_df = mi_feature_df.merge(
    mi_result_df[mi_traj_cols].drop_duplicates([MI_KEY, "image_id", "sample_id"]),
    on=[MI_KEY, "image_id", "sample_id"],
    how="left",
    suffixes=("", "__traj"),
)

x_traj_cols = [
    XENIUM_KEY, "sample_id", "display_name", "disease_group",
    "xenium_pseudotime", "xenium_pseudotime_sample_centered",
    "xenium_pseudotime_intrinsic_sample_centered",
    "xenium_pseudotime_sample_centered_norm",
    "major_branch",
]
x_traj_cols += [c for c in xenium_result_df.columns if c.startswith("histology__") or c.startswith("xenium_")]
x_traj_cols = list(dict.fromkeys([c for c in x_traj_cols if c in xenium_result_df.columns]))
xenium_context_df = xenium_feature_df.merge(
    xenium_result_df[x_traj_cols].drop_duplicates([XENIUM_KEY, "sample_id"]),
    on=[XENIUM_KEY, "sample_id"],
    how="left",
    suffixes=("", "__traj"),
)

print("MI contextual table:", mi_context_df.shape)
print("MI epithelial-intrinsic table:", mi_epi_df.shape)
print("Xenium contextual table:", xenium_context_df.shape)
display(xenium_branch_summary_df[["branch", "n_niches", "dominant_sample", "suggested_biology"]])
"""
    ),
    md("## Helper functions for trajectory trends and branch contrasts"),
    code(
        """
def available_feature_specs(df, specs):
    out = []
    seen = set()
    for spec in specs:
        feature = spec["feature"]
        if feature in seen:
            continue
        if feature in df.columns:
            out.append(spec)
            seen.add(feature)
    return out

def compute_feature_trend_table(df, specs, pseudotime_col, group_name, min_n=30):
    rows = []
    pt = pd.to_numeric(df[pseudotime_col], errors="coerce")
    for spec in specs:
        feature = spec["feature"]
        y = pd.to_numeric(df[feature], errors="coerce")
        tmp = pd.DataFrame({"pt": pt, "y": y}).dropna()
        if len(tmp) < min_n or tmp["pt"].nunique() < 5:
            continue
        rho, p = spearmanr(tmp["pt"], tmp["y"])
        q10, q90 = tmp["pt"].quantile([0.1, 0.9])
        early = tmp.loc[tmp["pt"] <= q10, "y"]
        late = tmp.loc[tmp["pt"] >= q90, "y"]
        delta = late.median() - early.median()
        try:
            mw_p = mannwhitneyu(early, late, alternative="two-sided").pvalue
        except Exception:
            mw_p = np.nan
        rows.append({
            "group": group_name,
            "category": spec.get("category", ""),
            "feature": feature,
            "label": spec.get("label", clean_label(feature)),
            "n": len(tmp),
            "spearman_r": rho,
            "spearman_p": p,
            "early_q10_median": early.median(),
            "late_q90_median": late.median(),
            "late_minus_early_median": delta,
            "mannwhitney_p_early_vs_late": mw_p,
        })
    return pd.DataFrame(rows).sort_values("spearman_r", ascending=False)

def plot_lowess_feature_panel(
    df,
    specs,
    pseudotime_col,
    title,
    ylabel="z-scored expression / proportion",
    n_cols=3,
    figsize_scale=2.4,
    lowess_frac=0.28,
    min_n=30,
    save_name=None,
):
    specs = available_feature_specs(df, specs)
    if len(specs) == 0:
        print(f"No available features for {title}")
        return pd.DataFrame()

    n_rows = int(np.ceil(len(specs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, figsize_scale * n_rows), squeeze=False)
    axes = axes.ravel()
    trend_df = compute_feature_trend_table(df, specs, pseudotime_col, title, min_n=min_n)

    for ax, spec in zip(axes, specs):
        feature = spec["feature"]
        tmp = df[[pseudotime_col, feature]].copy()
        tmp[pseudotime_col] = pd.to_numeric(tmp[pseudotime_col], errors="coerce")
        tmp[feature] = pd.to_numeric(tmp[feature], errors="coerce")
        tmp = tmp.dropna().sort_values(pseudotime_col)
        if len(tmp) >= min_n and tmp[pseudotime_col].nunique() >= 5:
            if len(tmp) > 6000:
                point_df = tmp.sample(6000, random_state=RANDOM_STATE)
            else:
                point_df = tmp
            ax.scatter(point_df[pseudotime_col], point_df[feature], s=3, alpha=0.035, color=spec.get("color", "#555555"), linewidths=0)
            low = sm.nonparametric.lowess(
                tmp[feature],
                tmp[pseudotime_col],
                frac=lowess_frac,
                return_sorted=True,
            )
            ax.plot(low[:, 0], low[:, 1], color=spec.get("color", "#222222"), linewidth=1.6)
        ax.set_title(spec.get("label", clean_label(feature)))
        ax.set_xlabel("Pseudotime")
        ax.set_ylabel(ylabel)
        ax.grid(False)

    for ax in axes[len(specs):]:
        ax.axis("off")
    fig.suptitle(title, y=1.01, fontsize=10)
    plt.tight_layout()
    if save_name:
        fig.savefig(FIG_DIR / save_name, dpi=220, bbox_inches="tight")
    plt.show()
    return trend_df

def plot_branch_heatmap(df, specs, branch_col, title, save_name=None, top_n=None):
    specs = available_feature_specs(df, specs)
    features = [s["feature"] for s in specs]
    if len(features) == 0 or branch_col not in df.columns:
        print(f"No branch heatmap for {title}")
        return pd.DataFrame()
    order = df[branch_col].value_counts().index.tolist()
    if top_n is not None:
        order = order[:top_n]
    use = df[df[branch_col].isin(order)].copy()
    mat = use.groupby(branch_col, observed=True)[features].median().reindex(order)
    mat_z = mat.apply(zscore_series, axis=0)
    mat_z.columns = [s.get("label", clean_label(s["feature"])) for s in specs]
    fig_w = max(7, 0.45 * len(features) + 2)
    fig_h = max(3.2, 0.32 * len(order) + 1.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(mat_z, cmap="RdBu_r", center=0, linewidths=0.2, linecolor="white", cbar_kws={"label": "branch median z"}, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    if save_name:
        fig.savefig(FIG_DIR / save_name, dpi=220, bbox_inches="tight")
    plt.show()
    return mat

TIME_BIN_ORDER = ["early", "mid", "late"]

def add_branch_time_bins(df, branch_col, pseudotime_col, min_branch_n=30):
    # Split each branch into early/mid/late pseudotime states within that branch.
    out = df.copy()
    out["_branch_time_bin"] = pd.NA
    out["_branch_time_order"] = np.nan
    pt = pd.to_numeric(out[pseudotime_col], errors="coerce")

    for branch, idx in out.groupby(branch_col, observed=True).groups.items():
        idx = list(idx)
        vals = pt.loc[idx]
        valid_idx = vals.dropna().index
        if len(valid_idx) < min_branch_n or vals.loc[valid_idx].nunique() < 3:
            continue
        try:
            bins = pd.qcut(vals.loc[valid_idx], q=3, labels=TIME_BIN_ORDER, duplicates="drop")
        except ValueError:
            continue
        if len(set(bins.dropna().astype(str))) < 2:
            continue
        out.loc[valid_idx, "_branch_time_bin"] = bins.astype(str)
        out.loc[valid_idx, "_branch_time_order"] = bins.astype(str).map({b: i for i, b in enumerate(TIME_BIN_ORDER)})

    out["_branch_time_state"] = (
        out[branch_col].astype(str) + " | " + out["_branch_time_bin"].astype(str)
    )
    out.loc[out["_branch_time_bin"].isna(), "_branch_time_state"] = pd.NA
    return out

def state_feature_matrix(state_df, specs, branch_col, pseudotime_col, dataset_label):
    specs = available_feature_specs(state_df, specs)
    features = [s["feature"] for s in specs]
    if len(features) == 0:
        return pd.DataFrame(), pd.DataFrame()

    state_df = state_df[state_df["_branch_time_state"].notna()].copy()
    if state_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows = []
    for state, sub in state_df.groupby("_branch_time_state", observed=True):
        branch = sub[branch_col].astype(str).iloc[0]
        time_bin = sub["_branch_time_bin"].astype(str).iloc[0]
        row = {
            "dataset": dataset_label,
            "branch": branch,
            "time_bin": time_bin,
            "time_order": TIME_BIN_ORDER.index(time_bin) if time_bin in TIME_BIN_ORDER else np.nan,
            "branch_time_state": state,
            "n_niches": len(sub),
            "pseudotime_median": pd.to_numeric(sub[pseudotime_col], errors="coerce").median(),
            "pseudotime_q25": pd.to_numeric(sub[pseudotime_col], errors="coerce").quantile(0.25),
            "pseudotime_q75": pd.to_numeric(sub[pseudotime_col], errors="coerce").quantile(0.75),
        }
        for spec in specs:
            row[spec["feature"]] = pd.to_numeric(sub[spec["feature"]], errors="coerce").median()
        rows.append(row)

    summary_df = pd.DataFrame(rows).sort_values(["branch", "time_order", "pseudotime_median"]).reset_index(drop=True)
    feature_df = summary_df[["dataset", "branch", "time_bin", "time_order", "branch_time_state", "n_niches", "pseudotime_median"] + features].copy()
    return summary_df, feature_df

def build_state_cards(state_feature_df, specs, branch_biology_map=None, top_n=5):
    specs = available_feature_specs(state_feature_df, specs)
    features = [s["feature"] for s in specs]
    if state_feature_df.empty or len(features) == 0:
        return pd.DataFrame()
    label_map = {s["feature"]: s.get("label", clean_label(s["feature"])) for s in specs}
    category_map = {s["feature"]: s.get("category", "") for s in specs}
    zmat = state_feature_df[features].apply(zscore_series, axis=0)
    rows = []
    for i, row in state_feature_df.reset_index(drop=True).iterrows():
        vals = zmat.iloc[i].dropna()
        positive = vals.sort_values(ascending=False).head(top_n)
        negative = vals.sort_values(ascending=True).head(top_n)
        branch = row["branch"]
        rows.append({
            "dataset": row["dataset"],
            "branch": branch,
            "time_bin": row["time_bin"],
            "branch_time_state": row["branch_time_state"],
            "n_niches": int(row["n_niches"]),
            "pseudotime_median": row["pseudotime_median"],
            "branch_interpretation": (branch_biology_map or {}).get(branch, branch),
            "top_enriched_features": "; ".join(
                f"{label_map[f]} ({category_map[f]}, z={v:+.2f})" for f, v in positive.items()
            ),
            "top_depleted_features": "; ".join(
                f"{label_map[f]} ({category_map[f]}, z={v:+.2f})" for f, v in negative.items()
            ),
        })
    return pd.DataFrame(rows)

def compute_branch_time_transitions(state_feature_df, specs, dataset_label):
    specs = available_feature_specs(state_feature_df, specs)
    features = [s["feature"] for s in specs]
    if state_feature_df.empty or len(features) == 0:
        return pd.DataFrame()
    label_map = {s["feature"]: s.get("label", clean_label(s["feature"])) for s in specs}
    category_map = {s["feature"]: s.get("category", "") for s in specs}
    zmat = state_feature_df[features].apply(zscore_series, axis=0)
    z_lookup = zmat.to_dict(orient="index")
    rows = []
    state_feature_df = state_feature_df.sort_values(["branch", "time_order"])
    for branch, sub in state_feature_df.groupby("branch", observed=True):
        sub = sub.sort_values("time_order")
        for (left_idx, left), (right_idx, right) in zip(sub.iloc[:-1].iterrows(), sub.iloc[1:].iterrows()):
            transition = f"{branch}: {left['time_bin']} -> {right['time_bin']}"
            for feature in features:
                before = pd.to_numeric(pd.Series([left[feature]]), errors="coerce").iloc[0]
                after = pd.to_numeric(pd.Series([right[feature]]), errors="coerce").iloc[0]
                if not (np.isfinite(before) and np.isfinite(after)):
                    continue
                delta = after - before
                before_z = z_lookup.get(left_idx, {}).get(feature, np.nan)
                after_z = z_lookup.get(right_idx, {}).get(feature, np.nan)
                delta_z = after_z - before_z if np.isfinite(before_z) and np.isfinite(after_z) else np.nan
                rows.append({
                    "dataset": dataset_label,
                    "branch": branch,
                    "transition": transition,
                    "from_state": left["branch_time_state"],
                    "to_state": right["branch_time_state"],
                    "from_time_bin": left["time_bin"],
                    "to_time_bin": right["time_bin"],
                    "feature": feature,
                    "label": label_map[feature],
                    "category": category_map[feature],
                    "from_median": before,
                    "to_median": after,
                    "delta": delta,
                    "abs_delta": abs(delta),
                    "from_state_z": before_z,
                    "to_state_z": after_z,
                    "delta_z": delta_z,
                    "abs_delta_z": abs(delta_z) if np.isfinite(delta_z) else np.nan,
                    "direction": "increase" if delta > 0 else "decrease",
                    "n_from": int(left["n_niches"]),
                    "n_to": int(right["n_niches"]),
                })
    return pd.DataFrame(rows).sort_values("abs_delta_z", ascending=False).reset_index(drop=True)

def summarize_transition_events(transition_df, top_n_per_transition=5):
    if transition_df.empty:
        return pd.DataFrame()
    rows = []
    for transition, sub in transition_df.groupby("transition", observed=True):
        top = sub.sort_values("abs_delta_z", ascending=False).head(top_n_per_transition)
        rows.append({
            "dataset": top["dataset"].iloc[0],
            "branch": top["branch"].iloc[0],
            "transition": transition,
            "max_abs_delta": float(top["abs_delta"].max()),
            "max_abs_delta_z": float(top["abs_delta_z"].max()),
            "top_changes": "; ".join(
                f"{r.label} {r.direction} (raw {r.delta:+.2g}, z {r.delta_z:+.2g})"
                for r in top.itertuples(index=False)
            ),
            "top_categories": ", ".join(top["category"].dropna().astype(str).value_counts().head(3).index.tolist()),
        })
    return pd.DataFrame(rows).sort_values("max_abs_delta_z", ascending=False).reset_index(drop=True)

def plot_branch_time_state_heatmap(state_feature_df, specs, title, save_name=None):
    specs = available_feature_specs(state_feature_df, specs)
    features = [s["feature"] for s in specs]
    if state_feature_df.empty or len(features) == 0:
        print(f"No branch-time heatmap for {title}")
        return pd.DataFrame()
    plot_df = state_feature_df.sort_values(["branch", "time_order", "pseudotime_median"]).copy()
    mat = plot_df.set_index("branch_time_state")[features]
    mat_z = mat.apply(zscore_series, axis=0)
    mat_z.columns = [s.get("label", clean_label(s["feature"])) for s in specs]
    fig_w = max(8, 0.36 * len(features) + 2.5)
    fig_h = max(4, 0.27 * len(mat_z) + 1.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        mat_z,
        cmap="RdBu_r",
        center=0,
        linewidths=0.2,
        linecolor="white",
        cbar_kws={"label": "branch-time state z"},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    if save_name:
        fig.savefig(FIG_DIR / save_name, dpi=220, bbox_inches="tight")
    plt.show()
    return mat

def plot_transition_event_bars(transition_df, title, save_name=None, top_n=25):
    if transition_df.empty:
        print(f"No transition events for {title}")
        return
    top = transition_df.sort_values("abs_delta_z", ascending=False).head(top_n).copy()
    top["change_label"] = top["transition"] + " | " + top["label"]
    fig_h = max(4, 0.25 * len(top) + 1.0)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    sns.barplot(
        data=top,
        y="change_label",
        x="delta_z",
        hue="category",
        dodge=False,
        palette="tab20",
        ax=ax,
    )
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Late-state median minus early-state median, within-feature z units")
    ax.set_ylabel("")
    ax.grid(False)
    ax.legend(frameon=False, fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    if save_name:
        fig.savefig(FIG_DIR / save_name, dpi=220, bbox_inches="tight")
    plt.show()

def make_branch_curve_df(df, branch_col, pseudotime_col, specs, n_bins=18, min_bin_n=12):
    specs = available_feature_specs(df, specs)
    features = [s["feature"] for s in specs]
    rows = []
    for branch, sub in df.groupby(branch_col, observed=True):
        sub = sub[[pseudotime_col] + features].copy()
        sub[pseudotime_col] = pd.to_numeric(sub[pseudotime_col], errors="coerce")
        for feature in features:
            sub[feature] = pd.to_numeric(sub[feature], errors="coerce")
        sub = sub.dropna(subset=[pseudotime_col]).sort_values(pseudotime_col)
        if len(sub) < min_bin_n or sub[pseudotime_col].nunique() < 3:
            continue
        q = int(min(n_bins, max(3, sub[pseudotime_col].nunique())))
        try:
            sub["_pt_bin"] = pd.qcut(sub[pseudotime_col], q=q, duplicates="drop")
        except ValueError:
            continue
        for _, bdf in sub.groupby("_pt_bin", observed=True):
            if len(bdf) < min_bin_n:
                continue
            row = {
                "branch": str(branch),
                "pseudotime": float(bdf[pseudotime_col].median()),
                "n_niches": int(len(bdf)),
            }
            for feature in features:
                row[feature] = pd.to_numeric(bdf[feature], errors="coerce").median()
            rows.append(row)
    return pd.DataFrame(rows)

def branch_lane_order(curve_df, trunk_label="trunk"):
    if curve_df.empty:
        return []
    branch_min = curve_df.groupby("branch", observed=True)["pseudotime"].min().sort_values()
    order = branch_min.index.tolist()
    if trunk_label in order:
        order = [trunk_label] + [b for b in order if b != trunk_label]
    return order

def plot_trajectory_subway_map(
    df,
    branch_col,
    pseudotime_col,
    feature,
    title,
    feature_label=None,
    cmap="RdBu_r",
    center=0,
    n_bins=18,
    min_bin_n=12,
    save_name=None,
):
    specs = [{"feature": feature, "label": feature_label or clean_label(feature)}]
    curve_df = make_branch_curve_df(df, branch_col, pseudotime_col, specs, n_bins=n_bins, min_bin_n=min_bin_n)
    if curve_df.empty or feature not in curve_df.columns:
        print(f"No trajectory subway map for {title}")
        return pd.DataFrame()

    order = branch_lane_order(curve_df)
    y_map = {branch: i for i, branch in enumerate(order)}
    max_n = max(curve_df["n_niches"].max(), 1)
    vals = pd.to_numeric(curve_df[feature], errors="coerce")
    if center is None:
        norm = plt.Normalize(vmin=np.nanquantile(vals, 0.02), vmax=np.nanquantile(vals, 0.98))
    else:
        vmax = np.nanmax(np.abs(vals - center))
        if not np.isfinite(vmax) or np.isclose(vmax, 0):
            vmax = 1
        norm = plt.Normalize(vmin=center - vmax, vmax=center + vmax)
    cmap_obj = plt.get_cmap(cmap)

    fig_h = max(3.8, 0.36 * len(order) + 1.4)
    fig, ax = plt.subplots(figsize=(9.5, fig_h))

    trunk_df = curve_df[curve_df["branch"] == "trunk"].sort_values("pseudotime")
    trunk_x = trunk_df["pseudotime"].to_numpy() if not trunk_df.empty else np.array([])
    trunk_y = np.zeros_like(trunk_x)

    for branch in order:
        sub = curve_df[curve_df["branch"] == branch].sort_values("pseudotime").copy()
        if sub.empty:
            continue
        y = y_map[branch]
        x = sub["pseudotime"].to_numpy()
        linewidth = 0.8 + 5.2 * np.sqrt(sub["n_niches"].to_numpy() / max_n)
        colors = cmap_obj(norm(pd.to_numeric(sub[feature], errors="coerce").to_numpy()))
        if branch != "trunk" and len(trunk_x) > 0:
            x0 = x[0]
            trunk_idx = int(np.argmin(np.abs(trunk_x - x0)))
            ax.plot(
                [trunk_x[trunk_idx], x0],
                [trunk_y[trunk_idx], y],
                color="#bdbdbd",
                linewidth=1.0,
                alpha=0.6,
                zorder=1,
            )
        for i in range(len(x) - 1):
            ax.plot(
                [x[i], x[i + 1]],
                [y, y],
                color=colors[i],
                linewidth=float(linewidth[i]),
                solid_capstyle="round",
                alpha=0.88,
                zorder=2,
            )
        ax.scatter(
            x,
            np.full_like(x, y, dtype=float),
            s=np.clip(8 + 45 * sub["n_niches"].to_numpy() / max_n, 10, 55),
            c=colors,
            edgecolor="white",
            linewidth=0.25,
            zorder=3,
        )

    smap = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    smap.set_array([])
    cbar = plt.colorbar(smap, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(feature_label or clean_label(feature))
    ax.set_yticks([y_map[b] for b in order])
    ax.set_yticklabels(order)
    ax.set_xlabel("Pseudotime")
    ax.set_ylabel("Trajectory branch lane")
    ax.set_title(title)
    ax.grid(False)
    ax.invert_yaxis()
    plt.tight_layout()
    if save_name:
        fig.savefig(FIG_DIR / save_name, dpi=240, bbox_inches="tight")
    plt.show()
    out = curve_df.copy()
    out["lane_y"] = out["branch"].map(y_map)
    return out

def plot_trajectory_subway_map_grid(
    df,
    branch_col,
    pseudotime_col,
    feature_specs,
    title_prefix,
    n_cols=2,
    n_bins=18,
    min_bin_n=12,
    save_name=None,
):
    feature_specs = available_feature_specs(df, feature_specs)
    if len(feature_specs) == 0:
        return pd.DataFrame()
    n_rows = int(np.ceil(len(feature_specs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, max(3.0, 2.7 * n_rows)), squeeze=False)
    axes = axes.ravel()
    all_curves = []
    for ax, spec in zip(axes, feature_specs):
        curve_df = make_branch_curve_df(
            df,
            branch_col,
            pseudotime_col,
            [spec],
            n_bins=n_bins,
            min_bin_n=min_bin_n,
        )
        if curve_df.empty:
            ax.axis("off")
            continue
        feature = spec["feature"]
        order = branch_lane_order(curve_df)
        y_map = {branch: i for i, branch in enumerate(order)}
        max_n = max(curve_df["n_niches"].max(), 1)
        vals = pd.to_numeric(curve_df[feature], errors="coerce")
        vmax = np.nanmax(np.abs(vals))
        if not np.isfinite(vmax) or np.isclose(vmax, 0):
            vmax = 1
        norm = plt.Normalize(vmin=-vmax, vmax=vmax)
        cmap_obj = plt.get_cmap(spec.get("cmap", "RdBu_r"))
        trunk_df = curve_df[curve_df["branch"] == "trunk"].sort_values("pseudotime")
        trunk_x = trunk_df["pseudotime"].to_numpy() if not trunk_df.empty else np.array([])
        for branch in order:
            sub = curve_df[curve_df["branch"] == branch].sort_values("pseudotime")
            if sub.empty:
                continue
            y = y_map[branch]
            x = sub["pseudotime"].to_numpy()
            colors = cmap_obj(norm(pd.to_numeric(sub[feature], errors="coerce").to_numpy()))
            linewidth = 0.55 + 3.2 * np.sqrt(sub["n_niches"].to_numpy() / max_n)
            if branch != "trunk" and len(trunk_x) > 0:
                trunk_idx = int(np.argmin(np.abs(trunk_x - x[0])))
                ax.plot([trunk_x[trunk_idx], x[0]], [0, y], color="#d0d0d0", linewidth=0.8, alpha=0.55)
            for i in range(len(x) - 1):
                ax.plot([x[i], x[i + 1]], [y, y], color=colors[i], linewidth=float(linewidth[i]), solid_capstyle="round")
            ax.scatter(x, np.full_like(x, y, dtype=float), s=7 + 25 * sub["n_niches"].to_numpy() / max_n, c=colors, linewidth=0)
        ax.set_title(spec.get("label", clean_label(feature)))
        ax.set_xlabel("Pseudotime")
        ax.set_yticks([y_map[b] for b in order])
        ax.set_yticklabels(order, fontsize=5)
        ax.invert_yaxis()
        ax.grid(False)
        smap = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        smap.set_array([])
        cbar = plt.colorbar(smap, ax=ax, fraction=0.028, pad=0.01)
        cbar.ax.tick_params(labelsize=5)
        curve_copy = curve_df.copy()
        curve_copy["feature"] = feature
        all_curves.append(curve_copy)
    for ax in axes[len(feature_specs):]:
        ax.axis("off")
    fig.suptitle(title_prefix, y=1.01, fontsize=10)
    plt.tight_layout()
    if save_name:
        fig.savefig(FIG_DIR / save_name, dpi=240, bbox_inches="tight")
    plt.show()
    return pd.concat(all_curves, ignore_index=True) if all_curves else pd.DataFrame()
"""
    ),
    md("## Multiplexed imaging: fibroblast and immune changes along trajectory"),
    code(
        """
MI_MICROENV_FEATURES = [
    {"feature": "surround_prop__Fibroblasts", "label": "Fibroblast proportion", "category": "composition", "color": "#e41a1c"},
    {"feature": "surround_prop__Vimentin_only_mesenchyme", "label": "Vimentin-only mesenchyme prop.", "category": "composition", "color": "#4daf4a"},
    {"feature": "surround_prop__Endothelial_cells", "label": "Endothelial proportion", "category": "composition", "color": "#377eb8"},
    {"feature": "surround_prop__T_cells", "label": "T-cell proportion", "category": "composition", "color": "#984ea3"},
    {"feature": "surround_prop__B_lineage", "label": "B-lineage proportion", "category": "composition", "color": "#ff7f00"},
    {"feature": "surround__Fibroblasts__FAP_expr_z__mean", "label": "Fibroblast FAP", "category": "fibroblast status", "color": "#e41a1c"},
    {"feature": "surround__Fibroblasts__aSMA_expr_z__mean", "label": "Fibroblast aSMA", "category": "fibroblast status", "color": "#a65628"},
    {"feature": "surround__Fibroblasts__PDPN_expr_z__mean", "label": "Fibroblast PDPN", "category": "fibroblast status", "color": "#f781bf"},
    {"feature": "surround__Fibroblasts__Thy1_expr_z__mean", "label": "Fibroblast Thy1", "category": "fibroblast status", "color": "#999999"},
]

MI_COLOC_CACHE = OUTPUT_DIR / "multiplexed_epithelial_niche_local_colocalization.pkl"
FORCE_REBUILD_MI_COLOCALIZATION = False
MI_COLOCALIZATION_RADIUS_UM = 30
MI_COLOCALIZATION_TARGETS = [
    "Fibroblasts",
    "T cells",
    "B lineage",
    "Endothelial cells",
    "Vimentin only mesenchyme",
]

def latest_mi_spatial_cell_paths():
    preferred = sorted(MI_DIR.glob("spatial_cells_auto_branch_n24_v3_with_epithelial_*.pkl"))
    if preferred:
        return preferred
    return sorted(MI_DIR.glob("spatial_cells_*_*.pkl"))

def compute_epithelial_niche_colocalization_for_sample(path, targets, radius_um=30):
    df = pd.read_pickle(path)
    need = [
        "cell_id", "image_id", "sample_id", "x", "y", "Tier_A", MI_KEY,
        "has_pooled_niche", "pooled_pseudotime", "major_branch",
    ]
    keep = [c for c in need if c in df.columns]
    df = df[keep].copy()
    df = df.dropna(subset=["x", "y", "Tier_A"])
    if "has_pooled_niche" in df.columns:
        source = df[df["has_pooled_niche"].fillna(False)].copy()
    else:
        source = df[df[MI_KEY].notna() & ~df[MI_KEY].astype(str).eq("unassigned")].copy()
    source = source[source[MI_KEY].notna() & ~source[MI_KEY].astype(str).eq("unassigned")].copy()
    if source.empty:
        return pd.DataFrame()

    coords_all = df[["x", "y"]].to_numpy(dtype=float)
    x_span = np.nanmax(coords_all[:, 0]) - np.nanmin(coords_all[:, 0])
    y_span = np.nanmax(coords_all[:, 1]) - np.nanmin(coords_all[:, 1])
    window_area = x_span * y_span
    if not np.isfinite(window_area) or window_area <= 0:
        return pd.DataFrame()

    source_coords = source[["x", "y"]].to_numpy(dtype=float)
    meta_cols = [MI_KEY, "image_id", "sample_id", "pooled_pseudotime", "major_branch"]
    source_meta = source[meta_cols].copy()
    rows = []
    for target in targets:
        target_df = df[df["Tier_A"].astype(str).eq(target)].copy()
        if len(target_df) < 5:
            continue
        target_coords = target_df[["x", "y"]].to_numpy(dtype=float)
        tree = BallTree(target_coords)
        neighbors = tree.query_radius(source_coords, r=radius_um)
        counts = np.array([len(nbrs) for nbrs in neighbors], dtype=float)
        expected = (len(target_coords) / window_area) * np.pi * radius_um ** 2
        if expected > 0:
            ratio = counts / expected
        else:
            ratio = np.full_like(counts, np.nan)
        safe_target = safe_label(target)
        tmp = source_meta.copy()
        tmp["_count"] = counts
        tmp["_excess"] = counts - expected
        tmp["_ratio"] = ratio
        tmp["_has_neighbor"] = counts > 0
        agg = (
            tmp.groupby([MI_KEY, "image_id", "sample_id"], observed=True)
            .agg(
                pooled_pseudotime=("pooled_pseudotime", "median"),
                major_branch=("major_branch", lambda s: s.dropna().astype(str).mode().iat[0] if len(s.dropna()) else np.nan),
                n_source_epithelial_cells=("_count", "size"),
                **{
                    f"coloc__ductal_to__{safe_target}__r{radius_um}__mean_target_neighbors_per_epithelial_cell": ("_count", "mean"),
                    f"coloc__ductal_to__{safe_target}__r{radius_um}__mean_target_neighbor_excess": ("_excess", "mean"),
                    f"coloc__ductal_to__{safe_target}__r{radius_um}__mean_target_neighbor_ratio": ("_ratio", "mean"),
                    f"coloc__ductal_to__{safe_target}__r{radius_um}__fraction_epithelial_cells_with_target_neighbor": ("_has_neighbor", "mean"),
                },
            )
            .reset_index()
        )
        rows.append(agg)
    if len(rows) == 0:
        return pd.DataFrame()
    out = rows[0]
    merge_keys = [MI_KEY, "image_id", "sample_id", "pooled_pseudotime", "major_branch", "n_source_epithelial_cells"]
    for extra in rows[1:]:
        out = out.merge(extra, on=merge_keys, how="outer")
    return out

if MI_COLOC_CACHE.exists() and not FORCE_REBUILD_MI_COLOCALIZATION:
    mi_colocalization_df = pd.read_pickle(MI_COLOC_CACHE)
    print(f"Using cached multiplexed co-localization: {MI_COLOC_CACHE}")
else:
    coloc_frames = []
    for path in latest_mi_spatial_cell_paths():
        print(f"Computing epithelial-niche co-localization from {path.name}")
        coloc_frames.append(
            compute_epithelial_niche_colocalization_for_sample(
                path,
                MI_COLOCALIZATION_TARGETS,
                radius_um=MI_COLOCALIZATION_RADIUS_UM,
            )
        )
        gc.collect()
    mi_colocalization_df = pd.concat([f for f in coloc_frames if not f.empty], ignore_index=True)
    save_df(mi_colocalization_df, MI_COLOC_CACHE)

mi_coloc_feature_cols = [
    c for c in mi_colocalization_df.columns
    if c.startswith("coloc__") and (
        c.endswith("__fraction_epithelial_cells_with_target_neighbor")
        or c.endswith("__mean_target_neighbor_excess")
    )
]
mi_context_df = mi_context_df.merge(
    mi_colocalization_df[[MI_KEY, "image_id", "sample_id"] + mi_coloc_feature_cols],
    on=[MI_KEY, "image_id", "sample_id"],
    how="left",
)

MI_COLOCALIZATION_FEATURES = []
for target in MI_COLOCALIZATION_TARGETS:
    safe_target = safe_label(target)
    feature = f"coloc__ductal_to__{safe_target}__r{MI_COLOCALIZATION_RADIUS_UM}__fraction_epithelial_cells_with_target_neighbor"
    if feature in mi_context_df.columns:
        MI_COLOCALIZATION_FEATURES.append({
            "feature": feature,
            "label": f"Ductal cells near {target}",
            "category": "spatial co-localization",
            "color": "#252525",
        })
    excess_feature = f"coloc__ductal_to__{safe_target}__r{MI_COLOCALIZATION_RADIUS_UM}__mean_target_neighbor_excess"
    if excess_feature in mi_context_df.columns:
        MI_COLOCALIZATION_FEATURES.append({
            "feature": excess_feature,
            "label": f"Ductal-{target} excess",
            "category": "spatial co-localization",
            "color": "#525252",
        })

mi_context_df["atlas__fibrotic_reaction"] = zmean(
    mi_context_df,
    [
        "surround_prop__Fibroblasts",
        "surround__Fibroblasts__FAP_expr_z__mean",
        "surround__Fibroblasts__aSMA_expr_z__mean",
        "surround__Fibroblasts__PDPN_expr_z__mean",
    ],
)
mi_context_df["atlas__immune_infiltration"] = zmean(
    mi_context_df,
    ["surround_prop__T_cells", "surround_prop__B_lineage"],
)
mi_context_df["atlas__ductal_fibroblast_colocalization"] = zmean(
    mi_context_df,
    [
        f"coloc__ductal_to__Fibroblasts__r{MI_COLOCALIZATION_RADIUS_UM}__fraction_epithelial_cells_with_target_neighbor",
        f"coloc__ductal_to__Fibroblasts__r{MI_COLOCALIZATION_RADIUS_UM}__mean_target_neighbor_excess",
    ],
)
MI_TREE_FEATURES = [
    {"feature": "atlas__fibrotic_reaction", "label": "Fibrotic reaction", "category": "atlas"},
    {"feature": "atlas__immune_infiltration", "label": "Immune infiltration", "category": "atlas"},
    {"feature": "atlas__ductal_fibroblast_colocalization", "label": "Ductal-fibroblast co-localization", "category": "atlas"},
]

mi_microenv_trend_df = plot_lowess_feature_panel(
    mi_context_df,
    MI_MICROENV_FEATURES,
    pseudotime_col="pooled_pseudotime",
    title="Multiplexed imaging: fibroblast/immune context along contextual pseudotime",
    ylabel="Proportion or z-scored marker",
    n_cols=3,
    save_name="multiplexed_microenvironment_trends_contextual.png",
)
display(mi_microenv_trend_df)
save_df(mi_microenv_trend_df, TABLE_DIR / "multiplexed_microenvironment_trends_contextual.csv")

mi_colocalization_trend_df = plot_lowess_feature_panel(
    mi_context_df,
    MI_COLOCALIZATION_FEATURES,
    pseudotime_col="pooled_pseudotime",
    title="Multiplexed imaging: epithelial-niche spatial co-localization along pseudotime",
    ylabel="Local cross-Ripley-style score",
    n_cols=3,
    save_name="multiplexed_epithelial_niche_colocalization_trends.png",
)
display(mi_colocalization_trend_df)
save_df(mi_colocalization_trend_df, TABLE_DIR / "multiplexed_epithelial_niche_colocalization_trends.csv")

mi_tree_curve_df = plot_trajectory_subway_map_grid(
    mi_context_df,
    branch_col="major_branch",
    pseudotime_col="pooled_pseudotime",
    feature_specs=MI_TREE_FEATURES,
    title_prefix="Multiplexed imaging: trajectory subway map with niche-density weighted branches",
    n_cols=1,
    n_bins=22,
    min_bin_n=18,
    save_name="multiplexed_trajectory_subway_map_key_programs.png",
)
save_df(mi_tree_curve_df, TABLE_DIR / "multiplexed_trajectory_subway_map_key_programs.csv")

mi_branch_microenv_df = plot_branch_heatmap(
    mi_context_df,
    MI_MICROENV_FEATURES + MI_COLOCALIZATION_FEATURES + MI_TREE_FEATURES,
    branch_col="major_branch",
    title="Multiplexed imaging: microenvironment status by branch",
    save_name="multiplexed_microenvironment_branch_heatmap.png",
    top_n=16,
)
display(mi_branch_microenv_df.head())
save_df(mi_branch_microenv_df.reset_index(), TABLE_DIR / "multiplexed_microenvironment_branch_medians.csv")
"""
    ),
    md(
        """
## Multiplexed imaging: branch-time niche state atlas

Here we reinterpret the trajectory as ordered niche states instead of only global pseudotime trends. Each branch is split into early/mid/late pseudotime states, then each state is summarized by fibroblast, immune, endothelial, and mesenchymal context. The transition table highlights the largest within-branch changes from early to mid and mid to late states.
"""
    ),
    code(
        """
mi_branch_time_df = add_branch_time_bins(
    mi_context_df,
    branch_col="major_branch",
    pseudotime_col="pooled_pseudotime",
    min_branch_n=45,
)
mi_state_summary_df, mi_state_feature_df = state_feature_matrix(
    mi_branch_time_df,
    MI_MICROENV_FEATURES + MI_COLOCALIZATION_FEATURES + MI_TREE_FEATURES,
    branch_col="major_branch",
    pseudotime_col="pooled_pseudotime",
    dataset_label="multiplexed imaging",
)
mi_state_card_df = build_state_cards(mi_state_feature_df, MI_MICROENV_FEATURES + MI_COLOCALIZATION_FEATURES + MI_TREE_FEATURES, top_n=5)
mi_transition_df = compute_branch_time_transitions(
    mi_state_feature_df,
    MI_MICROENV_FEATURES + MI_COLOCALIZATION_FEATURES + MI_TREE_FEATURES,
    dataset_label="multiplexed imaging",
)
mi_transition_event_df = summarize_transition_events(mi_transition_df, top_n_per_transition=4)

display(mi_state_card_df.head(20))
display(mi_transition_event_df.head(20))

plot_branch_time_state_heatmap(
    mi_state_feature_df,
    MI_MICROENV_FEATURES + MI_COLOCALIZATION_FEATURES + MI_TREE_FEATURES,
    title="Multiplexed imaging: branch-time niche state atlas",
    save_name="multiplexed_branch_time_state_heatmap.png",
)
plot_transition_event_bars(
    mi_transition_df,
    title="Multiplexed imaging: strongest within-branch niche transitions",
    save_name="multiplexed_branch_time_transition_events.png",
    top_n=25,
)

save_df(mi_state_summary_df, TABLE_DIR / "multiplexed_branch_time_state_summary.csv")
save_df(mi_state_card_df, TABLE_DIR / "multiplexed_branch_time_state_cards.csv")
save_df(mi_transition_df, TABLE_DIR / "multiplexed_branch_time_transition_feature_deltas.csv")
save_df(mi_transition_event_df, TABLE_DIR / "multiplexed_branch_time_transition_events.csv")
"""
    ),
    md("## Xenium: fibroblast, immune, myeloid, and endothelial changes along trajectory"),
    code(
        """
XENIUM_MICROENV_FEATURES = [
    {"feature": "surround_prop__Fibroblasts", "label": "Fibroblast prop.", "category": "composition", "color": "#e41a1c"},
    {"feature": "surround_prop__T_cells", "label": "T-cell prop.", "category": "composition", "color": "#984ea3"},
    {"feature": "surround_prop__B_lineage", "label": "B-lineage prop.", "category": "composition", "color": "#ff7f00"},
    {"feature": "surround_prop__Myeloid_cells", "label": "Myeloid prop.", "category": "composition", "color": "#4daf4a"},
    {"feature": "surround_prop__Endothelial_cells", "label": "Endothelial prop.", "category": "composition", "color": "#377eb8"},
    {"feature": "surround__Fibroblasts__ACTA2_expr_z__mean", "label": "Fibroblast ACTA2", "category": "fibroblast status", "color": "#a65628"},
    {"feature": "surround__Fibroblasts__PDGFRA_expr_z__mean", "label": "Fibroblast PDGFRA", "category": "fibroblast status", "color": "#e41a1c"},
    {"feature": "surround__Fibroblasts__THY1_expr_z__mean", "label": "Fibroblast THY1", "category": "fibroblast status", "color": "#fb9a99"},
    {"feature": "surround__Fibroblasts__PDPN_expr_z__mean", "label": "Fibroblast PDPN", "category": "fibroblast status", "color": "#f781bf"},
    {"feature": "surround__Fibroblasts__DCN_expr_z__mean", "label": "Fibroblast DCN", "category": "fibroblast status", "color": "#b15928"},
    {"feature": "surround__Fibroblasts__LUM_expr_z__mean", "label": "Fibroblast LUM", "category": "fibroblast status", "color": "#cab2d6"},
    {"feature": "surround__T_cells__CD3D_expr_z__mean", "label": "T CD3D", "category": "T-cell status", "color": "#984ea3"},
    {"feature": "surround__T_cells__CD8A_expr_z__mean", "label": "T CD8A", "category": "T-cell status", "color": "#6a3d9a"},
    {"feature": "surround__T_cells__FOXP3_expr_z__mean", "label": "T FOXP3", "category": "T-cell status", "color": "#e7298a"},
    {"feature": "surround__T_cells__GZMB_expr_z__mean", "label": "T GZMB", "category": "T-cell status", "color": "#7570b3"},
    {"feature": "surround__T_cells__NKG7_expr_z__mean", "label": "T/NK NKG7", "category": "T-cell status", "color": "#66a61e"},
    {"feature": "surround__B_lineage__CD79A_expr_z__mean", "label": "B CD79A", "category": "B-lineage status", "color": "#ff7f00"},
    {"feature": "surround__B_lineage__MS4A1_expr_z__mean", "label": "B MS4A1", "category": "B-lineage status", "color": "#fdbf6f"},
    {"feature": "surround__B_lineage__MZB1_expr_z__mean", "label": "Plasma MZB1", "category": "B-lineage status", "color": "#b2df8a"},
    {"feature": "surround__B_lineage__JCHAIN_expr_z__mean", "label": "Plasma JCHAIN", "category": "B-lineage status", "color": "#33a02c"},
    {"feature": "surround__Myeloid_cells__CD68_expr_z__mean", "label": "Myeloid CD68", "category": "myeloid status", "color": "#4daf4a"},
    {"feature": "surround__Myeloid_cells__CD163_expr_z__mean", "label": "Myeloid CD163", "category": "myeloid status", "color": "#1b9e77"},
    {"feature": "surround__Myeloid_cells__C1QB_expr_z__mean", "label": "Myeloid C1QB", "category": "myeloid status", "color": "#66c2a5"},
    {"feature": "surround__Myeloid_cells__S100A9_expr_z__mean", "label": "Myeloid S100A9", "category": "myeloid status", "color": "#b3de69"},
]

def add_xenium_neighborhood_program_scores(df):
    df = df.copy()
    program_defs = {
        "nbhd_program__fibroblast_activation": [
            "surround_prop__Fibroblasts",
            "surround__Fibroblasts__ACTA2_expr_z__mean",
            "surround__Fibroblasts__THY1_expr_z__mean",
            "surround__Fibroblasts__PDPN_expr_z__mean",
            "surround__Fibroblasts__DCN_expr_z__mean",
            "surround__Fibroblasts__LUM_expr_z__mean",
            "graph_surround__ACTA2_expr_z__surround_minus_niche",
            "graph_surround__PDPN_expr_z__surround_minus_niche",
            "graph_surround__PDGFRA_expr_z__surround_minus_niche",
        ],
        "nbhd_program__cytotoxic_t": [
            "surround_prop__T_cells",
            "surround__T_cells__CD8A_expr_z__mean",
            "surround__T_cells__GZMB_expr_z__mean",
            "surround__T_cells__NKG7_expr_z__mean",
            "graph_surround__CD8A_expr_z__surround_minus_niche",
            "graph_surround__GZMB_expr_z__surround_minus_niche",
            "graph_surround__NKG7_expr_z__surround_minus_niche",
        ],
        "nbhd_program__treg_checkpoint": [
            "surround__T_cells__FOXP3_expr_z__mean",
            "graph_surround__FOXP3_expr_z__surround_minus_niche",
            "graph_surround__PDCD1_expr_z__surround_minus_niche",
            "graph_surround__CD274_expr_z__surround_minus_niche",
        ],
        "nbhd_program__b_plasma": [
            "surround_prop__B_lineage",
            "surround__B_lineage__CD79A_expr_z__mean",
            "surround__B_lineage__MS4A1_expr_z__mean",
            "surround__B_lineage__MZB1_expr_z__mean",
            "surround__B_lineage__JCHAIN_expr_z__mean",
            "graph_surround__CD79A_expr_z__surround_minus_niche",
            "graph_surround__MS4A1_expr_z__surround_minus_niche",
            "graph_surround__MZB1_expr_z__surround_minus_niche",
            "graph_surround__JCHAIN_expr_z__surround_minus_niche",
        ],
        "nbhd_program__myeloid_inflammatory": [
            "surround_prop__Myeloid_cells",
            "surround__Myeloid_cells__CD68_expr_z__mean",
            "surround__Myeloid_cells__CD163_expr_z__mean",
            "surround__Myeloid_cells__C1QB_expr_z__mean",
            "surround__Myeloid_cells__S100A9_expr_z__mean",
            "graph_surround__CD68_expr_z__surround_minus_niche",
            "graph_surround__CD163_expr_z__surround_minus_niche",
            "graph_surround__C1QB_expr_z__surround_minus_niche",
            "graph_surround__S100A9_expr_z__surround_minus_niche",
            "graph_surround__AIF1_expr_z__surround_minus_niche",
        ],
        "nbhd_program__endothelial_angiogenic": [
            "surround_prop__Endothelial_cells",
            "surround__Endothelial_cells__PECAM1_expr_z__mean",
            "surround__Endothelial_cells__CD34_expr_z__mean",
            "graph_surround__PECAM1_expr_z__surround_minus_niche",
            "graph_surround__CD34_expr_z__surround_minus_niche",
            "graph_surround__VWF_expr_z__surround_minus_niche",
            "graph_surround__FLT1_expr_z__surround_minus_niche",
        ],
        "nbhd_program__neighborhood_mixing": [
            "graph_surround__phenotype_entropy",
            "graph_surround__cross_edges_per_niche_cell",
            "graph_surround__surround_to_niche_ratio",
        ],
    }
    availability_rows = []
    for program, cols in program_defs.items():
        present = [c for c in cols if c in df.columns]
        df[program] = zmean(df, present)
        availability_rows.append({
            "program": program,
            "n_features": len(present),
            "features": ", ".join(present),
        })
    return df, pd.DataFrame(availability_rows)

xenium_context_df, xenium_neighborhood_program_availability_df = add_xenium_neighborhood_program_scores(xenium_context_df)
display(xenium_neighborhood_program_availability_df)
save_df(xenium_neighborhood_program_availability_df, TABLE_DIR / "xenium_neighborhood_program_availability.csv")

XENIUM_NEIGHBORHOOD_PROGRAM_FEATURES = [
    {"feature": "nbhd_program__fibroblast_activation", "label": "Neighborhood fibroblast activation", "category": "neighborhood program", "color": "#a65628"},
    {"feature": "nbhd_program__cytotoxic_t", "label": "Neighborhood cytotoxic T", "category": "neighborhood program", "color": "#7570b3"},
    {"feature": "nbhd_program__treg_checkpoint", "label": "Neighborhood Treg/checkpoint", "category": "neighborhood program", "color": "#e7298a"},
    {"feature": "nbhd_program__b_plasma", "label": "Neighborhood B/plasma", "category": "neighborhood program", "color": "#ff7f00"},
    {"feature": "nbhd_program__myeloid_inflammatory", "label": "Neighborhood myeloid", "category": "neighborhood program", "color": "#4daf4a"},
    {"feature": "nbhd_program__endothelial_angiogenic", "label": "Neighborhood endothelial/angiogenic", "category": "neighborhood program", "color": "#377eb8"},
    {"feature": "nbhd_program__neighborhood_mixing", "label": "Neighborhood mixing", "category": "neighborhood program", "color": "#636363"},
]

xenium_context_df["atlas__fibrotic_reaction"] = zmean(
    xenium_context_df,
    [
        "surround_prop__Fibroblasts",
        "surround__Fibroblasts__ACTA2_expr_z__mean",
        "surround__Fibroblasts__THY1_expr_z__mean",
        "surround__Fibroblasts__PDPN_expr_z__mean",
        "nbhd_program__fibroblast_activation",
    ],
)
xenium_context_df["atlas__immune_infiltration"] = zmean(
    xenium_context_df,
    [
        "surround_prop__T_cells",
        "surround_prop__B_lineage",
        "surround_prop__Myeloid_cells",
        "nbhd_program__cytotoxic_t",
        "nbhd_program__b_plasma",
        "nbhd_program__myeloid_inflammatory",
    ],
)
xenium_context_df["atlas__immune_suppression_checkpoint"] = zmean(
    xenium_context_df,
    [
        "surround__T_cells__FOXP3_expr_z__mean",
        "nbhd_program__treg_checkpoint",
    ],
)
XENIUM_TREE_FEATURES = [
    {"feature": "atlas__fibrotic_reaction", "label": "Fibrotic reaction", "category": "atlas"},
    {"feature": "atlas__immune_infiltration", "label": "Immune infiltration", "category": "atlas"},
    {"feature": "atlas__immune_suppression_checkpoint", "label": "Treg/checkpoint-like context", "category": "atlas"},
    {"feature": "nbhd_program__endothelial_angiogenic", "label": "Endothelial/angiogenic neighborhood", "category": "neighborhood program"},
]

xenium_microenv_trend_df = plot_lowess_feature_panel(
    xenium_context_df,
    XENIUM_MICROENV_FEATURES,
    pseudotime_col="xenium_pseudotime_sample_centered",
    title="Xenium: fibroblast/immune context along sample-centered pseudotime",
    ylabel="Proportion or z-scored marker",
    n_cols=4,
    figsize_scale=2.2,
    save_name="xenium_microenvironment_trends_sample_centered.png",
)
display(xenium_microenv_trend_df)
save_df(xenium_microenv_trend_df, TABLE_DIR / "xenium_microenvironment_trends_sample_centered.csv")

xenium_neighborhood_program_trend_df = plot_lowess_feature_panel(
    xenium_context_df,
    XENIUM_NEIGHBORHOOD_PROGRAM_FEATURES,
    pseudotime_col="xenium_pseudotime_sample_centered",
    title="Xenium: BANKSY-like collective neighborhood programs along pseudotime",
    ylabel="Neighborhood program score",
    n_cols=3,
    figsize_scale=2.3,
    save_name="xenium_neighborhood_program_trends_sample_centered.png",
)
display(xenium_neighborhood_program_trend_df)
save_df(xenium_neighborhood_program_trend_df, TABLE_DIR / "xenium_neighborhood_program_trends_sample_centered.csv")

xenium_branch_microenv_df = plot_branch_heatmap(
    xenium_context_df,
    XENIUM_MICROENV_FEATURES + XENIUM_NEIGHBORHOOD_PROGRAM_FEATURES + XENIUM_TREE_FEATURES,
    branch_col="major_branch",
    title="Xenium: microenvironment status by branch",
    save_name="xenium_microenvironment_branch_heatmap.png",
)
display(xenium_branch_microenv_df.head())
save_df(xenium_branch_microenv_df.reset_index(), TABLE_DIR / "xenium_microenvironment_branch_medians.csv")
"""
    ),
    md("## Xenium: targeted spatial ligand-receptor communication potential"),
    code(
        """
LR_CACHE = OUTPUT_DIR / "xenium_targeted_lr_context_log1p_nonnegative.pkl"
FORCE_REBUILD_LR_CONTEXT = False

XENIUM_SAMPLE_CONFIGS = [
    {"sample_id": "pdac_pancreas_v1"},
    {"sample_id": "pdac_io_v1"},
    {"sample_id": "pdac_addon_v1"},
    {"sample_id": "normal_nondiseased_v1"},
]

CANDIDATE_LR_PAIRS = [
    {"source": "Fibroblasts", "target": "pancreatic ductal epithelium", "ligand": "CXCL12", "receptor": "CXCR4", "label": "Fibroblast CXCL12 -> epithelial CXCR4"},
    {"source": "pancreatic ductal epithelium", "target": "pancreatic ductal epithelium", "ligand": "AREG", "receptor": "EGFR", "label": "Epithelial AREG -> epithelial EGFR"},
    {"source": "pancreatic ductal epithelium", "target": "Myeloid cells", "ligand": "CSF1", "receptor": "CSF1R", "label": "Epithelial CSF1 -> myeloid CSF1R"},
    {"source": "pancreatic ductal epithelium", "target": "Myeloid cells", "ligand": "CCL2", "receptor": "CCR2", "label": "Epithelial CCL2 -> myeloid CCR2"},
    {"source": "pancreatic ductal epithelium", "target": "Endothelial cells", "ligand": "VEGFA", "receptor": "FLT1", "label": "Epithelial VEGFA -> endothelial FLT1"},
    {"source": "pancreatic ductal epithelium", "target": "T cells", "ligand": "CD274", "receptor": "PDCD1", "label": "Epithelial CD274 -> T-cell PDCD1"},
    {"source": "Myeloid cells", "target": "T cells", "ligand": "CD274", "receptor": "PDCD1", "label": "Myeloid CD274 -> T-cell PDCD1"},
    {"source": "pancreatic ductal epithelium", "target": "B lineage", "ligand": "MIF", "receptor": "CD74", "label": "Epithelial MIF -> B-lineage CD74"},
]

LR_GENES = sorted(set([p["ligand"] for p in CANDIDATE_LR_PAIRS] + [p["receptor"] for p in CANDIDATE_LR_PAIRS]))
LR_PHENOTYPES = sorted(set([p["source"] for p in CANDIDATE_LR_PAIRS] + [p["target"] for p in CANDIDATE_LR_PAIRS] + ["T cells", "B lineage", "Myeloid cells", "Fibroblasts", "Endothelial cells"]))

def extract_gene_matrix(adata, genes):
    present = [g for g in genes if g in adata.var_names]
    if len(present) == 0:
        return pd.DataFrame(index=adata.obs_names)
    if "log1p" in adata.layers:
        X = adata[:, present].layers["log1p"]
    elif "counts" in adata.layers:
        X = adata[:, present].layers["counts"]
    else:
        X = adata[:, present].X
    if sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=float)
    X = np.clip(X, 0, None)
    return pd.DataFrame(X, index=adata.obs_names.astype(str), columns=present)

def add_lr_gene_obs_columns(adata, genes):
    expr = extract_gene_matrix(adata, genes)
    for gene in expr.columns:
        vals = pd.to_numeric(expr[gene], errors="coerce")
        adata.obs[f"{gene}_lr_expr"] = vals.to_numpy()
    return list(expr.columns)

def summarize_lr_context_for_sample(sample_id):
    adata_path = XENIUM_NICHE_DIR / f"{sample_id}_with_niches.h5ad"
    if not adata_path.exists():
        print(f"Missing {adata_path}")
        return pd.DataFrame()
    adata = sc.read_h5ad(adata_path)
    present_genes = add_lr_gene_obs_columns(adata, LR_GENES)
    if len(present_genes) == 0:
        return pd.DataFrame()

    source_mask = (
        adata.obs[XENIUM_KEY].notna()
        & (adata.obs["Tier_A"].astype(str) == "pancreatic ductal epithelium")
    )
    source_cols = [f"{g}_lr_expr" for g in present_genes]
    source_df = (
        adata.obs.loc[source_mask, [XENIUM_KEY] + source_cols]
        .groupby(XENIUM_KEY, observed=True)[source_cols]
        .mean()
        .reset_index()
    )
    source_df = source_df.rename(columns={c: f"source__pancreatic_ductal_epithelium__{c}__mean" for c in source_cols})

    phenotype_labels = [
        label
        for label in LR_PHENOTYPES
        if label in set(adata.obs["Tier_A"].dropna().astype(str))
    ]
    phenotype_feature_map = {label: source_cols for label in phenotype_labels}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        context_df = se.summarize_niche_surrounding_context(
            adata,
            niche_key=XENIUM_KEY,
            phenotype_key="Tier_A",
            phenotype_labels=phenotype_labels,
            phenotype_feature_map=phenotype_feature_map,
            image_key="sample_id",
            adjacency_key="cell_graph_connectivities",
            surround_hops=5,
            min_cells=5,
            summary_stats=("mean",),
            show_progress=True,
            progress_desc=f"{sample_id} LR surroundings",
        )
    out = source_df.merge(context_df.drop(columns=["n_cells"], errors="ignore"), on=XENIUM_KEY, how="left")
    out["sample_id"] = sample_id
    out["lr_genes_present"] = ",".join(present_genes)
    del adata
    gc.collect()
    return out

if LR_CACHE.exists() and not FORCE_REBUILD_LR_CONTEXT:
    xenium_lr_context_df = pd.read_pickle(LR_CACHE)
    print(f"Using cached LR context: {LR_CACHE}")
else:
    frames = []
    for cfg in XENIUM_SAMPLE_CONFIGS:
        frames.append(summarize_lr_context_for_sample(cfg["sample_id"]))
    xenium_lr_context_df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    save_df(xenium_lr_context_df, LR_CACHE)

print(xenium_lr_context_df.shape)
xenium_lr_context_df.head()
"""
    ),
    code(
        """
def expr_col(compartment, gene, source_or_target):
    gene_col = f"{gene}_lr_expr"
    if compartment == "pancreatic ductal epithelium":
        return f"source__pancreatic_ductal_epithelium__{gene_col}__mean"
    return f"surround__{safe_label(compartment)}__{gene_col}__mean"

def prop_col(compartment):
    if compartment == "pancreatic ductal epithelium":
        return None
    return f"surround_prop__{safe_label(compartment)}"

def add_lr_scores(lr_context_df, trajectory_df):
    merged = lr_context_df.merge(
        trajectory_df[
            [
                XENIUM_KEY, "sample_id", "display_name", "disease_group",
                "major_branch", "xenium_pseudotime_sample_centered",
                "xenium_pseudotime", "xenium_pseudotime_intrinsic_sample_centered",
            ]
            + [c for c in trajectory_df.columns if c.startswith("histology__")]
        ].drop_duplicates([XENIUM_KEY, "sample_id"]),
        on=[XENIUM_KEY, "sample_id"],
        how="left",
    )
    score_rows = []
    wide = merged[[XENIUM_KEY, "sample_id", "display_name", "disease_group", "major_branch", "xenium_pseudotime_sample_centered"]].copy()
    for pair in CANDIDATE_LR_PAIRS:
        ligand_col = expr_col(pair["source"], pair["ligand"], "source")
        receptor_col = expr_col(pair["target"], pair["receptor"], "target")
        required = [ligand_col, receptor_col]
        if not all(c in merged.columns for c in required):
            continue
        ligand = pd.to_numeric(merged[ligand_col], errors="coerce")
        receptor = pd.to_numeric(merged[receptor_col], errors="coerce")
        score = ligand * receptor
        for compartment in [pair["source"], pair["target"]]:
            pcol = prop_col(compartment)
            if pcol is not None and pcol in merged.columns:
                score = score * np.sqrt(pd.to_numeric(merged[pcol], errors="coerce").fillna(0).clip(lower=0))
        score_name = f"lr__{safe_label(pair['source'])}__to__{safe_label(pair['target'])}__{pair['ligand']}__{pair['receptor']}"
        wide[score_name] = score
        score_rows.append({**pair, "score": score_name, "ligand_col": ligand_col, "receptor_col": receptor_col, "n_nonnull": int(score.notna().sum())})
    score_meta_df = pd.DataFrame(score_rows)
    return wide, score_meta_df

xenium_lr_score_df, xenium_lr_score_meta_df = add_lr_scores(xenium_lr_context_df, xenium_result_df)
display(xenium_lr_score_meta_df)
save_df(xenium_lr_score_meta_df, TABLE_DIR / "xenium_lr_score_metadata.csv")
save_df(xenium_lr_score_df, TABLE_DIR / "xenium_lr_scores_by_niche.pkl")

LR_FEATURE_SPECS = [
    {"feature": row["score"], "label": row["label"], "category": "LR", "color": "#333333"}
    for _, row in xenium_lr_score_meta_df.iterrows()
    if row["n_nonnull"] >= 30
]
lr_score_cols = [spec["feature"] for spec in LR_FEATURE_SPECS if spec["feature"] in xenium_lr_score_df.columns]

xenium_lr_trend_df = plot_lowess_feature_panel(
    xenium_lr_score_df,
    LR_FEATURE_SPECS,
    pseudotime_col="xenium_pseudotime_sample_centered",
    title="Xenium spatial ligand-receptor communication potential along pseudotime",
    ylabel="local LR product score",
    n_cols=2,
    figsize_scale=2.5,
    lowess_frac=0.35,
    min_n=30,
    save_name="xenium_lr_potential_trends.png",
)
display(xenium_lr_trend_df)
save_df(xenium_lr_trend_df, TABLE_DIR / "xenium_lr_potential_trends.csv")

xenium_lr_branch_df = plot_branch_heatmap(
    xenium_lr_score_df,
    LR_FEATURE_SPECS,
    branch_col="major_branch",
    title="Xenium spatial ligand-receptor potential by branch",
    save_name="xenium_lr_branch_heatmap.png",
)
display(xenium_lr_branch_df)
save_df(xenium_lr_branch_df.reset_index(), TABLE_DIR / "xenium_lr_branch_medians.csv")

lr_label_to_score = dict(zip(xenium_lr_score_meta_df["label"], xenium_lr_score_meta_df["score"])) if not xenium_lr_score_meta_df.empty else {}
lr_tree_df = xenium_lr_score_df[[XENIUM_KEY, "sample_id"] + lr_score_cols].copy()
xenium_context_df = xenium_context_df.merge(lr_tree_df, on=[XENIUM_KEY, "sample_id"], how="left")
xenium_context_df["atlas__vascular_lr_potential"] = zmean(
    xenium_context_df,
    [lr_label_to_score.get("Epithelial VEGFA -> endothelial FLT1", "__missing__")],
)
xenium_context_df["atlas__checkpoint_lr_potential"] = zmean(
    xenium_context_df,
    [
        lr_label_to_score.get("Epithelial CD274 -> T-cell PDCD1", "__missing__"),
        lr_label_to_score.get("Myeloid CD274 -> T-cell PDCD1", "__missing__"),
    ],
)
xenium_context_df["atlas__myeloid_recruitment_lr_potential"] = zmean(
    xenium_context_df,
    [
        lr_label_to_score.get("Epithelial CSF1 -> myeloid CSF1R", "__missing__"),
        lr_label_to_score.get("Epithelial CCL2 -> myeloid CCR2", "__missing__"),
    ],
)
XENIUM_TREE_FEATURES_WITH_LR = XENIUM_TREE_FEATURES + [
    {"feature": "atlas__vascular_lr_potential", "label": "VEGFA-FLT1 vascular LR potential", "category": "LR atlas"},
    {"feature": "atlas__checkpoint_lr_potential", "label": "CD274-PDCD1 checkpoint LR potential", "category": "LR atlas"},
    {"feature": "atlas__myeloid_recruitment_lr_potential", "label": "CSF1/CCL2 myeloid LR potential", "category": "LR atlas"},
]

xenium_tree_curve_df = plot_trajectory_subway_map_grid(
    xenium_context_df,
    branch_col="major_branch",
    pseudotime_col="xenium_pseudotime_sample_centered",
    feature_specs=XENIUM_TREE_FEATURES_WITH_LR,
    title_prefix="Xenium: trajectory subway map with niche-density weighted branches",
    n_cols=2,
    n_bins=16,
    min_bin_n=10,
    save_name="xenium_trajectory_subway_map_key_programs.png",
)
save_df(xenium_tree_curve_df, TABLE_DIR / "xenium_trajectory_subway_map_key_programs.csv")
"""
    ),
    md(
        """
## Xenium: branch-time niche evolution and communication atlas

This is the main post-trajectory interpretation layer. We combine microenvironment composition/status features with candidate ligand-receptor potential, then split each automatic branch into early/mid/late states. This gives us state cards and transition events such as fibroblast activation, myeloid recruitment, endothelial coupling, checkpoint-like signaling, or immune exclusion emerging within a branch.
"""
    ),
    code(
        """
xenium_branch_biology_map = {}
if {"branch", "suggested_biology"}.issubset(xenium_branch_summary_df.columns):
    xenium_branch_biology_map = dict(
        zip(xenium_branch_summary_df["branch"].astype(str), xenium_branch_summary_df["suggested_biology"].astype(str))
    )

lr_score_cols = [
    row["score"]
    for _, row in xenium_lr_score_meta_df.iterrows()
    if row["score"] in xenium_lr_score_df.columns and row["n_nonnull"] >= 30
]
missing_lr_cols = [c for c in lr_score_cols if c not in xenium_context_df.columns]
if missing_lr_cols:
    xenium_atlas_df = xenium_context_df.merge(
        xenium_lr_score_df[[XENIUM_KEY, "sample_id"] + missing_lr_cols],
        on=[XENIUM_KEY, "sample_id"],
        how="left",
    )
else:
    xenium_atlas_df = xenium_context_df.copy()
xenium_atlas_specs = (
    XENIUM_MICROENV_FEATURES
    + XENIUM_NEIGHBORHOOD_PROGRAM_FEATURES
    + XENIUM_TREE_FEATURES_WITH_LR
    + LR_FEATURE_SPECS
)

xenium_branch_time_df = add_branch_time_bins(
    xenium_atlas_df,
    branch_col="major_branch",
    pseudotime_col="xenium_pseudotime_sample_centered",
    min_branch_n=30,
)
xenium_state_summary_df, xenium_state_feature_df = state_feature_matrix(
    xenium_branch_time_df,
    xenium_atlas_specs,
    branch_col="major_branch",
    pseudotime_col="xenium_pseudotime_sample_centered",
    dataset_label="xenium",
)
xenium_state_card_df = build_state_cards(
    xenium_state_feature_df,
    xenium_atlas_specs,
    branch_biology_map=xenium_branch_biology_map,
    top_n=5,
)
xenium_transition_df = compute_branch_time_transitions(
    xenium_state_feature_df,
    xenium_atlas_specs,
    dataset_label="xenium",
)
xenium_transition_event_df = summarize_transition_events(xenium_transition_df, top_n_per_transition=5)

display(xenium_state_card_df.head(25))
display(xenium_transition_event_df.head(25))

plot_branch_time_state_heatmap(
    xenium_state_feature_df,
    xenium_atlas_specs,
    title="Xenium: branch-time niche evolution and communication atlas",
    save_name="xenium_branch_time_state_heatmap.png",
)
plot_transition_event_bars(
    xenium_transition_df,
    title="Xenium: strongest within-branch niche transition events",
    save_name="xenium_branch_time_transition_events.png",
    top_n=35,
)

save_df(xenium_state_summary_df, TABLE_DIR / "xenium_branch_time_state_summary.csv")
save_df(xenium_state_card_df, TABLE_DIR / "xenium_branch_time_state_cards.csv")
save_df(xenium_transition_df, TABLE_DIR / "xenium_branch_time_transition_feature_deltas.csv")
save_df(xenium_transition_event_df, TABLE_DIR / "xenium_branch_time_transition_events.csv")
"""
    ),
    md("## Integrated interpretation tables"),
    code(
        """
def summarize_top_changes(trend_df, label, top_n=8):
    if trend_df is None or trend_df.empty:
        return pd.DataFrame()
    out = trend_df.copy()
    out["analysis"] = label
    out["abs_spearman_r"] = out["spearman_r"].abs()
    return out.sort_values("abs_spearman_r", ascending=False).head(top_n)

top_change_df = pd.concat(
    [
        summarize_top_changes(mi_microenv_trend_df, "multiplexed microenvironment"),
        summarize_top_changes(mi_colocalization_trend_df, "multiplexed spatial co-localization"),
        summarize_top_changes(xenium_microenv_trend_df, "xenium microenvironment"),
        summarize_top_changes(xenium_neighborhood_program_trend_df, "xenium neighborhood programs"),
        summarize_top_changes(xenium_lr_trend_df, "xenium ligand-receptor"),
    ],
    ignore_index=True,
)
display(top_change_df[["analysis", "label", "category", "spearman_r", "late_minus_early_median", "n"]])
save_df(top_change_df, TABLE_DIR / "top_trajectory_microenvironment_changes.csv")

all_transition_df = pd.concat(
    [
        mi_transition_df,
        xenium_transition_df,
    ],
    ignore_index=True,
)
all_transition_event_df = pd.concat(
    [
        mi_transition_event_df,
        xenium_transition_event_df,
    ],
    ignore_index=True,
)
transition_category_summary_df = (
    all_transition_df
    .assign(abs_delta_z=lambda d: d["delta_z"].abs())
    .groupby(["dataset", "category"], observed=True)
    .agg(
        n_events=("delta", "size"),
        median_abs_delta_z=("abs_delta_z", "median"),
        max_abs_delta_z=("abs_delta_z", "max"),
        n_increases=("direction", lambda s: int((s == "increase").sum())),
        n_decreases=("direction", lambda s: int((s == "decrease").sum())),
    )
    .reset_index()
    .sort_values(["dataset", "max_abs_delta_z"], ascending=[True, False])
)

all_transition_event_df = all_transition_event_df.sort_values("max_abs_delta_z", ascending=False).reset_index(drop=True)

display(all_transition_event_df.head(30))
display(transition_category_summary_df)
save_df(all_transition_df, TABLE_DIR / "all_branch_time_transition_feature_deltas.csv")
save_df(all_transition_event_df, TABLE_DIR / "all_branch_time_transition_events.csv")
save_df(transition_category_summary_df, TABLE_DIR / "transition_category_summary.csv")

summary_lines = []
for analysis, sub in top_change_df.groupby("analysis", observed=True):
    summary_lines.append(f"## {analysis}")
    for _, row in sub.iterrows():
        direction = "increases" if row["spearman_r"] > 0 else "decreases"
        summary_lines.append(
            f"- {row['label']} {direction} along pseudotime "
            f"(Spearman r={row['spearman_r']:.2f}, late-early median={row['late_minus_early_median']:.3g})."
        )
summary_lines.append("## branch-time niche evolution events")
for _, row in all_transition_event_df.head(20).iterrows():
    summary_lines.append(
        f"- {row['dataset']} {row['transition']}: {row['top_changes']} "
        f"(strongest standardized |delta|={row['max_abs_delta_z']:.2g})."
    )
summary_text = "\\n".join(summary_lines)
(OUTPUT_DIR / "trajectory_microenvironment_interpretation_summary.md").write_text(summary_text)
print(summary_text)
"""
    ),
    md(
        """
## Notes

Interpretation guardrails:

- Multiplexed imaging has stronger protein/pixel-level fibroblast status readouts but limited immune subtype marker status in the pooled table.
- Multiplexed imaging co-localization uses local cross-Ripley-style counts around epithelial cells assigned to each ductal niche. This is more spatially explicit than simple surround proportions, but still depends on the chosen radius.
- Xenium has richer immune/fibroblast/myeloid status genes, but ligand-receptor coverage is panel-limited and varies by sample.
- Xenium neighborhood programs are transparent BANKSY-like collective scores: they combine local graph-surround expression contrasts, compartment-specific surrounding-cell expression, and composition. They are not a full BANKSY clustering run, but they capture the same idea of cell state plus neighborhood context.
- The LR score is a local, abundance-weighted expression product constrained to niche/surround neighborhoods. It is useful for prioritizing interactions, not proving active signaling.
- Branch-time states are not literal clock stages. They are local waypoints along morphology/topology-defined branches, useful for recapitulating likely niche-state transitions.
- The most robust biological claims are those that recur across modalities: for example fibroblast activation in multiplexed imaging and fibroblast/stromal gene activation in Xenium.
- Strong candidate interactions should be checked spatially and, if possible, validated with protein or pathway activity readouts.
"""
    ),
]


def main():
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {
        "display_name": "spatioev_env",
        "language": "python",
        "name": "spatioev_env",
    }
    nb["metadata"]["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NOTEBOOK)
    print(f"Wrote {NOTEBOOK}")


if __name__ == "__main__":
    main()
