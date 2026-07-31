#!/usr/bin/env python
"""Write separate Xenium BANKSY and SpatialCellChat integration notebooks."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BANKSY_NOTEBOOK = ROOT  / "paper" / "notebooks" / "09_xenium_banksy_pseudotime_integration.ipynb"
SPATIALCELLCHAT_NOTEBOOK = ROOT  / "paper" / "notebooks" / "10_xenium_spatialcellchat_pseudotime_integration.ipynb"


def md(text: str) -> dict:
    source = textwrap.dedent(text).strip()
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(text: str) -> dict:
    source = textwrap.dedent(text).strip()
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "spatioev_env",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


banksy_cells = [
    md(
        """
        # Xenium BANKSY domains on the SpatioEv pseudotime trajectory

        This notebook runs **proper pyBANKSY** independently in the existing `/Users/shihongwu/anaconda3/envs/pybanksy` environment, then integrates BANKSY domains back onto the SpatioEv epithelial niche pseudotime table.

        This version follows the logic of the reference notebook:

        - concatenate multiple Xenium samples using common genes
        - stagger sample coordinates so samples do not overlap spatially
        - construct BANKSY matrices on the combined object
        - Harmony-correct BANKSY PCs by sample
        - run Leiden clustering on Harmony-corrected BANKSY PCs
        - export per-sample BANKSY labels and summarize them by epithelial niche and niche surround
        """
    ),
    code(
        """
        from pathlib import Path
        import os
        import subprocess

        import numpy as np
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt

        ROOT = Path("/Users/shihongwu/SpatioEv")
        DATA_DIR = ROOT / "data" / "xenium_pancreas_10x"
        BANKSY_DIR = DATA_DIR / "banksy"
        FIG_DIR = ROOT  / "paper" / "notebooks" / "results" / "xenium_banksy_pseudotime"

        BANKSY_PYTHON = Path("/Users/shihongwu/anaconda3/envs/pybanksy/bin/python")
        SPATIOEV_PYTHON = Path("/Users/shihongwu/anaconda3/envs/spatioev_env/bin/python")

        SAMPLE_IDS = [
            "pdac_pancreas_v1",
            "pdac_io_v1",
            "pdac_addon_v1",
            "normal_nondiseased_v1",
        ]

        BANKSY_DIR.mkdir(parents=True, exist_ok=True)
        FIG_DIR.mkdir(parents=True, exist_ok=True)

        RUN_BANKSY = False
        RUN_BANKSY_INTEGRATION = True

        BANKSY_MAX_CELLS_PER_SAMPLE = 0  # 0 = all cells; set e.g. 50000 for a quick exploratory run.
        BANKSY_NUM_NEIGHBOURS = 20
        BANKSY_LAMBDAS = ["0.8"]
        BANKSY_RESOLUTIONS = ["0.5"]
        BANKSY_PCA_DIMS = ["20"]
        BANKSY_PRIMARY_RUN_KEY = None

        plt.rcParams.update({"font.size": 8})
        sns.set_style("white")

        print("BANKSY python:", BANKSY_PYTHON)
        print("BANKSY python exists:", BANKSY_PYTHON.exists())
        """
    ),
    md("## 1. Run multi-sample BANKSY + Harmony"),
    code(
        """
        banksy_cmd = [
            str(BANKSY_PYTHON),
            str(ROOT / "scripts" / "run_xenium_banksy_harmony_multisample.py"),
            "--sample-id", "all",
            "--output-dir", str(BANKSY_DIR),
            "--num-neighbours", str(BANKSY_NUM_NEIGHBOURS),
            "--lambda-list", *BANKSY_LAMBDAS,
            "--resolutions", *BANKSY_RESOLUTIONS,
            "--pca-dims", *BANKSY_PCA_DIMS,
            "--max-cells-per-sample", str(BANKSY_MAX_CELLS_PER_SAMPLE),
        ]

        banksy_env = os.environ.copy()
        banksy_env.setdefault("NUMBA_CACHE_DIR", "/private/tmp/numba_cache")
        banksy_env.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig")

        if RUN_BANKSY:
            subprocess.run(banksy_cmd, cwd=ROOT, env=banksy_env, check=True)
        else:
            print("Set RUN_BANKSY=True to run multi-sample pyBANKSY + Harmony.")
            print("Command:")
            print(" ".join(banksy_cmd))

        for sample_id in SAMPLE_IDS:
            path = BANKSY_DIR / sample_id / f"{sample_id}_banksy_cell_domains.csv.gz"
            print(sample_id, "BANKSY domains exist:", path.exists(), path)
        """
    ),
    md("## 2. Integrate BANKSY domains onto epithelial niche pseudotime"),
    code(
        """
        banksy_domain_paths = [
            BANKSY_DIR / sample_id / f"{sample_id}_banksy_cell_domains.csv.gz"
            for sample_id in SAMPLE_IDS
        ]

        if all(path.exists() for path in banksy_domain_paths) and RUN_BANKSY_INTEGRATION:
            integrate_cmd = [
                str(SPATIOEV_PYTHON),
                str(ROOT / "scripts" / "integrate_xenium_banksy_to_pseudotime.py"),
                "--sample-id", "all",
                "--banksy-dir", str(BANKSY_DIR),
                "--force",
            ]
            if BANKSY_PRIMARY_RUN_KEY is not None:
                integrate_cmd.extend(["--primary-run-key", BANKSY_PRIMARY_RUN_KEY])
            subprocess.run(integrate_cmd, cwd=ROOT, check=True)
        else:
            print("BANKSY outputs are not complete yet, or RUN_BANKSY_INTEGRATION=False.")

        banksy_integrated_path = BANKSY_DIR / "xenium_pseudotime_with_banksy.pkl"
        print("Integrated BANKSY table exists:", banksy_integrated_path.exists())
        """
    ),
    md("## 3. Visualize BANKSY domains along the SpatioEv trajectory"),
    code(
        """
        def plot_embedding(ax, df, color, title, cmap="viridis", palette=None, s=5, alpha=0.75):
            if color not in df.columns:
                ax.axis("off")
                ax.set_title(f"Missing: {color}")
                return
            if pd.api.types.is_numeric_dtype(df[color]):
                sc = ax.scatter(df["UMAP1"], df["UMAP2"], c=df[color], s=s, cmap=cmap, alpha=alpha, linewidths=0)
                plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            else:
                sns.scatterplot(data=df, x="UMAP1", y="UMAP2", hue=color, palette=palette, s=s, alpha=alpha, linewidth=0, ax=ax)
                ax.legend(frameon=False, fontsize=6, loc="best", markerscale=3)
            ax.set_title(title)
            ax.grid(False)


        def plot_banksy_overview(df):
            fig, axes = plt.subplots(2, 2, figsize=(11, 8))
            axes = axes.ravel()
            plot_embedding(axes[0], df, "xenium_pseudotime", "SpatioEv pseudotime")
            plot_embedding(axes[1], df, "major_branch", "SpatioEv branch", palette="tab20")
            plot_embedding(axes[2], df, "banksy_intrinsic__dominant_domain", "BANKSY domain inside epithelial niche", palette="tab20")
            plot_embedding(axes[3], df, "banksy_surround__dominant_domain", "BANKSY domain in niche surround", palette="tab20")
            plt.tight_layout()
            out = FIG_DIR / "xenium_banksy_umap_overview.png"
            plt.savefig(out, dpi=300, bbox_inches="tight")
            plt.show()
            print(out)


        def plot_banksy_branch_heatmap(df, prefix="banksy_surround"):
            prop_cols = [c for c in df.columns if c.startswith(f"{prefix}__domain_") and c.endswith("__prop")]
            if len(prop_cols) == 0:
                print(f"No {prefix} domain proportion columns available.")
                return pd.DataFrame()
            summary = (
                df.dropna(subset=["branch_time_state"])
                .groupby(["major_branch", "branch_time_bin", "branch_time_state"], observed=True)[prop_cols]
                .mean()
            )
            renamed = summary.rename(columns=lambda c: c.replace(f"{prefix}__domain_", "").replace("__prop", ""))
            plt.figure(figsize=(max(8, 0.35 * renamed.shape[1]), max(4, 0.28 * renamed.shape[0])))
            sns.heatmap(renamed, cmap="mako", linewidths=0.2, linecolor="white")
            plt.title(f"{prefix.replace('_', ' ').title()} BANKSY domain occupancy by branch-time state")
            plt.xlabel("BANKSY domain")
            plt.ylabel("Branch-time state")
            plt.tight_layout()
            out = FIG_DIR / f"xenium_{prefix}_banksy_branch_time_heatmap.png"
            plt.savefig(out, dpi=300, bbox_inches="tight")
            plt.show()
            print(out)
            return renamed


        banksy_integrated_path = BANKSY_DIR / "xenium_pseudotime_with_banksy.pkl"
        if banksy_integrated_path.exists():
            banksy_pt_df = pd.read_pickle(banksy_integrated_path)
            display(banksy_pt_df.head())
            plot_banksy_overview(banksy_pt_df)
            banksy_surround_heatmap_df = plot_banksy_branch_heatmap(banksy_pt_df, prefix="banksy_surround")
            banksy_intrinsic_heatmap_df = plot_banksy_branch_heatmap(banksy_pt_df, prefix="banksy_intrinsic")
        else:
            print("No integrated BANKSY table yet.")
        """
    ),
    md("## 4. Spatial scatter plots of BANKSY domains per sample"),
    code(
        """
        from matplotlib.lines import Line2D


        def infer_banksy_domain_col(df, preferred=None, use_relabeled=True):
            if preferred is not None and preferred in df.columns:
                base_col = preferred
            else:
                candidates = [
                    c for c in df.columns
                    if c.startswith(("banksy_harmony__", "banksy__")) and not c.endswith("__relabeled")
                ]
                if len(candidates) == 0:
                    raise ValueError("No BANKSY domain columns found.")
                preferred_candidates = [
                    c for c in candidates
                    if "__lambda_0p80__" in c and "__res_0p50" in c
                ]
                base_col = preferred_candidates[0] if len(preferred_candidates) > 0 else candidates[0]
            relabeled_col = f"{base_col}__relabeled"
            if use_relabeled and relabeled_col in df.columns:
                return relabeled_col
            return base_col


        def load_banksy_cell_domains(sample_ids=SAMPLE_IDS):
            frames = []
            for sample_id in sample_ids:
                path = BANKSY_DIR / sample_id / f"{sample_id}_banksy_cell_domains.csv.gz"
                if not path.exists():
                    print(f"Missing BANKSY cell-domain file for {sample_id}: {path}")
                    continue
                frames.append(pd.read_csv(path))
            if len(frames) == 0:
                return pd.DataFrame()
            return pd.concat(frames, ignore_index=True)


        def make_domain_palette(values):
            values = pd.Series(values).dropna().astype(str).unique().tolist()
            try:
                values = sorted(values, key=lambda x: int(float(x)))
            except ValueError:
                values = sorted(values)
            colors = sns.color_palette("tab20", n_colors=max(len(values), 1))
            return dict(zip(values, colors))


        def plot_banksy_domain_spatial_by_sample(
            cell_df,
            domain_col,
            sample_ids=SAMPLE_IDS,
            phenotype_filter=None,
            point_size=0.08,
            alpha=0.9,
            n_cols=2,
            save_name="xenium_banksy_domains_spatial_by_sample.png",
        ):
            plot_df = cell_df.copy()
            plot_df[domain_col] = plot_df[domain_col].astype(str)
            if phenotype_filter is not None:
                plot_df = plot_df.loc[plot_df["Tier_A"].isin(phenotype_filter)].copy()
                save_name = save_name.replace(".png", "_filtered.png")

            palette = make_domain_palette(plot_df[domain_col])
            n_rows = int(np.ceil(len(sample_ids) / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.8 * n_rows))
            axes = np.array(axes).reshape(-1)

            for ax, sample_id in zip(axes, sample_ids):
                tmp = plot_df.loc[plot_df["sample_id"] == sample_id].copy()
                if tmp.empty:
                    ax.axis("off")
                    ax.set_title(f"{sample_id}: no cells")
                    continue
                colors = tmp[domain_col].map(palette)
                ax.scatter(
                    tmp["banksy_x_original"],
                    tmp["banksy_y_original"],
                    c=colors,
                    s=point_size,
                    alpha=alpha,
                    linewidths=0,
                    rasterized=True,
                )
                ax.set_title(f"{sample_id} BANKSY domains")
                ax.set_aspect("equal", adjustable="box")
                ax.invert_yaxis()
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)

            for ax in axes[len(sample_ids):]:
                ax.axis("off")

            handles = [
                Line2D([0], [0], marker="o", color="none", label=str(domain), markerfacecolor=color, markersize=5)
                for domain, color in palette.items()
            ]
            fig.legend(
                handles=handles,
                title="BANKSY domain",
                frameon=False,
                bbox_to_anchor=(1.01, 0.5),
                loc="center left",
                fontsize=7,
                title_fontsize=8,
            )
            fig.suptitle(
                f"Spatial distribution of BANKSY domains"
                + ("" if phenotype_filter is None else " among selected phenotypes"),
                y=1.02,
            )
            plt.tight_layout()
            out = FIG_DIR / save_name
            plt.savefig(out, dpi=300, bbox_inches="tight")
            plt.show()
            print(out)
            return out


        banksy_cell_df = load_banksy_cell_domains()
        if banksy_cell_df.empty:
            print("No BANKSY cell-domain tables found yet.")
        else:
            banksy_spatial_domain_col = infer_banksy_domain_col(
                banksy_cell_df,
                preferred=BANKSY_PRIMARY_RUN_KEY,
                use_relabeled=True,
            )
            print("Plotting BANKSY domain column:", banksy_spatial_domain_col)
            display(banksy_cell_df[["sample_id", banksy_spatial_domain_col]].value_counts().rename("n").reset_index().head(20))

            plot_banksy_domain_spatial_by_sample(
                banksy_cell_df,
                domain_col=banksy_spatial_domain_col,
                point_size=0.06,
                save_name="xenium_banksy_domains_spatial_by_sample.png",
            )

            # Optional epithelial-only view, useful for comparing BANKSY domains with the epithelial trajectory backbone.
            plot_banksy_domain_spatial_by_sample(
                banksy_cell_df,
                domain_col=banksy_spatial_domain_col,
                phenotype_filter=["pancreatic ductal epithelium", "Mucosa gland"],
                point_size=0.18,
                save_name="xenium_banksy_domains_spatial_epithelial_by_sample.png",
            )
        """
    ),
    md(
        """
        ## Notes

        BANKSY domains are spatially smoothed transcriptional domains. Here they are not used to define pseudotime; they are projected onto the morphology/topology-derived SpatioEv trajectory to ask which epithelial niche states and surrounding tissue states occupy each branch.
        """
    ),
]


spatialcellchat_cells = [
    md(
        """
        # Xenium SpatialCellChat integration with SpatioEv pseudotime

        This notebook is separate from BANKSY. It prepares trajectory-aware Xenium inputs for **SpatialCellChat / CellChat v3**, runs the R workflow when the package is available, and summarizes inferred ligand-receptor communication by sample and SpatioEv branch-time state.
        """
    ),
    code(
        """
        from pathlib import Path
        import subprocess

        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt

        ROOT = Path("/Users/shihongwu/SpatioEv")
        DATA_DIR = ROOT / "data" / "xenium_pancreas_10x"
        SPATIALCELLCHAT_DIR = DATA_DIR / "spatialcellchat"
        FIG_DIR = ROOT  / "paper" / "notebooks" / "results" / "xenium_spatialcellchat_pseudotime"

        SPATIOEV_PYTHON = Path("/Users/shihongwu/anaconda3/envs/spatioev_env/bin/python")
        RSCRIPT = "Rscript"

        SAMPLE_IDS = [
            "pdac_pancreas_v1",
            "pdac_io_v1",
            "pdac_addon_v1",
            "normal_nondiseased_v1",
        ]

        SPATIALCELLCHAT_DIR.mkdir(parents=True, exist_ok=True)
        FIG_DIR.mkdir(parents=True, exist_ok=True)

        RUN_SPATIALCELLCHAT_EXPORT = False
        RUN_SPATIALCELLCHAT_R = False
        RUN_SPATIALCELLCHAT_INTEGRATION = True

        MAX_CELLS_PER_STATE = 12000
        MIN_CELLS_PER_STATE = 300
        MIN_CELLS_PER_GROUP = 20

        plt.rcParams.update({"font.size": 8})
        sns.set_style("white")
        """
    ),
    md("## 1. Export trajectory-aware SpatialCellChat inputs"),
    code(
        """
        export_scc_cmd = [
            str(SPATIOEV_PYTHON),
            str(ROOT / "scripts" / "export_xenium_spatialcellchat_inputs.py"),
            "--sample-id", "all",
            "--output-dir", str(SPATIALCELLCHAT_DIR / "input"),
        ]

        if RUN_SPATIALCELLCHAT_EXPORT:
            subprocess.run(export_scc_cmd, cwd=ROOT, check=True)
        else:
            print("Set RUN_SPATIALCELLCHAT_EXPORT=True to export SpatialCellChat inputs.")
            print("Command:")
            print(" ".join(export_scc_cmd))

        for sample_id in SAMPLE_IDS:
            path = SPATIALCELLCHAT_DIR / "input" / sample_id / "metadata.csv.gz"
            print(sample_id, "SpatialCellChat input exists:", path.exists(), path)
        """
    ),
    md("## 2. Run SpatialCellChat in R"),
    code(
        """
        r_check = subprocess.run(
            [RSCRIPT, "-e", "cat(requireNamespace('SpatialCellChat', quietly=TRUE))"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        print("SpatialCellChat installed:", r_check.stdout.strip())

        scc_cmd = [
            RSCRIPT,
            str(ROOT / "scripts" / "run_xenium_spatialcellchat.R"),
            "--sample-id", "all",
            "--input-dir", str(SPATIALCELLCHAT_DIR / "input"),
            "--output-dir", str(SPATIALCELLCHAT_DIR / "results"),
            "--group-col", "Tier_A",
            "--state-col", "branch_time_state",
            "--max-cells-per-state", str(MAX_CELLS_PER_STATE),
            "--min-cells-per-state", str(MIN_CELLS_PER_STATE),
            "--min-cells-per-group", str(MIN_CELLS_PER_GROUP),
        ]

        if RUN_SPATIALCELLCHAT_R:
            subprocess.run(scc_cmd, cwd=ROOT, check=True)
        else:
            print("Set RUN_SPATIALCELLCHAT_R=True after installing SpatialCellChat.")
            print("Command:")
            print(" ".join(scc_cmd))
        """
    ),
    md("## 3. Integrate SpatialCellChat outputs with pseudotime states"),
    code(
        """
        scc_results = sorted((SPATIALCELLCHAT_DIR / "results").glob("*/*__communication_lr.csv"))
        print("SpatialCellChat result files:", len(scc_results))

        if len(scc_results) > 0 and RUN_SPATIALCELLCHAT_INTEGRATION:
            subprocess.run(
                [
                    str(SPATIOEV_PYTHON),
                    str(ROOT / "scripts" / "integrate_xenium_spatialcellchat_to_pseudotime.py"),
                    "--results-dir", str(SPATIALCELLCHAT_DIR / "results"),
                    "--output-dir", str(SPATIALCELLCHAT_DIR / "integrated"),
                ],
                cwd=ROOT,
                check=True,
            )
        else:
            print("No SpatialCellChat result files yet, or RUN_SPATIALCELLCHAT_INTEGRATION=False.")

        scc_summary_path = SPATIALCELLCHAT_DIR / "integrated" / "xenium_spatialcellchat_trajectory_summary.csv"
        print("Integrated SpatialCellChat summary exists:", scc_summary_path.exists())
        """
    ),
    md("## 4. Visualize communication programs across trajectory states"),
    code(
        """
        scc_summary_path = SPATIALCELLCHAT_DIR / "integrated" / "xenium_spatialcellchat_trajectory_summary.csv"

        if scc_summary_path.exists():
            scc_summary_df = pd.read_csv(scc_summary_path)
            display(scc_summary_df.head())

            top = (
                scc_summary_df.dropna(subset=["branch_time_state"])
                .sort_values("mean_probability", ascending=False)
                .head(30)
                .copy()
            )
            if {"interaction_name", "branch_time_state", "mean_probability"}.issubset(top.columns):
                pivot = top.pivot_table(
                    index="interaction_name",
                    columns="branch_time_state",
                    values="mean_probability",
                    aggfunc="max",
                ).fillna(0)
                plt.figure(figsize=(max(8, 0.35 * pivot.shape[1]), max(6, 0.25 * pivot.shape[0])))
                sns.heatmap(pivot, cmap="rocket_r", linewidths=0.2, linecolor="white")
                plt.title("SpatialCellChat top interactions by trajectory state")
                plt.xlabel("Branch-time state")
                plt.ylabel("Ligand-receptor interaction")
                plt.tight_layout()
                out = FIG_DIR / "xenium_spatialcellchat_top_interactions_heatmap.png"
                plt.savefig(out, dpi=300, bbox_inches="tight")
                plt.show()
                print(out)
        else:
            print("No integrated SpatialCellChat summary yet.")
        """
    ),
    md(
        """
        ## Notes

        SpatialCellChat outputs are inferred communication probabilities, not proof of signaling. Here the biological use is comparative: which inferred source-target pathways become enriched in specific early/mid/late states of morphology-derived SpatioEv branches?
        """
    ),
]


def write_notebook(path: Path, cells: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook(cells), indent=1), encoding="utf-8")
    print(path)


def main() -> None:
    write_notebook(BANKSY_NOTEBOOK, banksy_cells)
    write_notebook(SPATIALCELLCHAT_NOTEBOOK, spatialcellchat_cells)


if __name__ == "__main__":
    main()
