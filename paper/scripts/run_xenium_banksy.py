#!/usr/bin/env python
"""Run official pyBANKSY on Xenium sample-level AnnData files.

This script is intentionally independent from ``spatioev_env``. Run it with the
existing BANKSY environment, for example:

    NUMBA_CACHE_DIR=/private/tmp/numba_cache MPLCONFIGDIR=/private/tmp/mplconfig \
      /Users/shihongwu/anaconda3/envs/pybanksy/bin/python \
      scripts/run_xenium_banksy.py --sample-id all

The output is one cell-level CSV per sample containing BANKSY domain labels for
each requested lambda/resolution setting. Those labels can then be summarized
back onto SpatioEv epithelial niches by
``scripts/integrate_xenium_banksy_to_pseudotime.py``.
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

# These need to be set before importing scanpy/numba/matplotlib-heavy modules.
os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig")

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from banksy.initialize_banksy import initialize_banksy
from banksy.run_banksy import run_banksy_multiparam


SAMPLE_IDS = (
    "pdac_pancreas_v1",
    "pdac_io_v1",
    "pdac_addon_v1",
    "normal_nondiseased_v1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/xenium_pancreas_10x"),
        help="Root Xenium analysis directory.",
    )
    parser.add_argument(
        "--sample-id",
        nargs="+",
        default=["all"],
        help="Sample IDs to run, or 'all'.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/xenium_pancreas_10x/banksy"),
        help="Directory for BANKSY outputs.",
    )
    parser.add_argument("--layer", default="log1p", help="AnnData layer to use as expression.")
    parser.add_argument("--x-key", default="x_centroid")
    parser.add_argument("--y-key", default="y_centroid")
    parser.add_argument("--phenotype-key", default="Tier_A")
    parser.add_argument("--tier-b-key", default="Tier_B")
    parser.add_argument("--niche-key", default="xenium_ductal_epithelium_component")
    parser.add_argument("--num-neighbours", type=int, default=15)
    parser.add_argument("--max-m", type=int, default=1)
    parser.add_argument("--lambda-list", type=float, nargs="+", default=[0.2, 0.5])
    parser.add_argument("--resolutions", type=float, nargs="+", default=[0.4, 0.8])
    parser.add_argument("--pca-dims", type=int, nargs="+", default=[20])
    parser.add_argument(
        "--max-cells",
        type=int,
        default=0,
        help="Optional random downsample per sample. Use 0 for all cells.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument(
        "--save-figures",
        action="store_true",
        help="Save pyBANKSY diagnostic figures. Off by default for speed.",
    )
    return parser.parse_args()


def selected_samples(sample_args: list[str]) -> list[str]:
    if len(sample_args) == 1 and sample_args[0].lower() == "all":
        return list(SAMPLE_IDS)
    unknown = sorted(set(sample_args).difference(SAMPLE_IDS))
    if unknown:
        raise ValueError(f"Unknown sample IDs: {unknown}")
    return sample_args


def safe_float(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def banksy_run_key(decay: str, lambda_param: float, num_pcs: int, resolution: float) -> str:
    decay_safe = str(decay).replace(" ", "_")
    return (
        f"banksy__{decay_safe}"
        f"__lambda_{safe_float(float(lambda_param))}"
        f"__pc_{int(num_pcs)}"
        f"__res_{safe_float(float(resolution))}"
    )


def make_color_list(n: int = 500) -> list[str]:
    base = list(sc.pl.palettes.default_102)
    if len(base) >= n:
        return base[:n]
    rng = np.random.default_rng(12345)
    extra = ["#%02x%02x%02x" % tuple(rng.integers(0, 256, size=3)) for _ in range(n - len(base))]
    return base + extra


def load_and_prepare_adata(path: Path, args: argparse.Namespace, sample_id: str):
    adata = sc.read_h5ad(path)

    valid = np.isfinite(pd.to_numeric(adata.obs[args.x_key], errors="coerce")) & np.isfinite(
        pd.to_numeric(adata.obs[args.y_key], errors="coerce")
    )
    if valid.sum() < adata.n_obs:
        adata = adata[np.asarray(valid)].copy()

    if args.max_cells and adata.n_obs > args.max_cells:
        rng = np.random.default_rng(args.random_state)
        idx = np.sort(rng.choice(adata.n_obs, size=args.max_cells, replace=False))
        adata = adata[idx].copy()

    if args.layer in adata.layers:
        adata.X = adata.layers[args.layer].copy()
    elif args.layer != "X":
        raise KeyError(f"{path.name} does not contain layer {args.layer!r}")

    if sparse.issparse(adata.X):
        gene_sum = np.asarray(adata.X.sum(axis=0)).ravel()
        gene_sq_sum = np.asarray(adata.X.power(2).sum(axis=0)).ravel()
        gene_var = gene_sq_sum / max(adata.n_obs, 1) - (gene_sum / max(adata.n_obs, 1)) ** 2
    else:
        gene_sum = np.asarray(adata.X.sum(axis=0)).ravel()
        gene_var = np.asarray(adata.X.var(axis=0)).ravel()
    keep_genes = np.isfinite(gene_var) & (gene_sum > 0) & (gene_var > 0)
    adata = adata[:, keep_genes].copy()

    adata.obsm["spatial"] = adata.obs[[args.x_key, args.y_key]].to_numpy(dtype=float)
    adata.obs["banksy_cell_id"] = adata.obs_names.astype(str)
    adata.obs["banksy_sample_id"] = sample_id
    return adata


def extract_result_labels(results_df: pd.DataFrame, n_cells: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_cols = {}
    param_rows = []
    for params_name, row in results_df.iterrows():
        labels = row["labels"]
        if hasattr(labels, "dense"):
            labels = labels.dense
        labels = np.asarray(labels).astype(str)
        if labels.shape[0] != n_cells:
            raise ValueError(
                f"BANKSY label length mismatch for {params_name}: {labels.shape[0]} vs {n_cells}"
            )
        key = banksy_run_key(
            decay=row["decay"],
            lambda_param=row["lambda_param"],
            num_pcs=row["num_pcs"],
            resolution=row["resolution"],
        )
        label_cols[key] = labels
        param_rows.append(
            {
                "banksy_run_key": key,
                "params_name": params_name,
                "decay": row["decay"],
                "lambda_param": row["lambda_param"],
                "num_pcs": row["num_pcs"],
                "resolution": row["resolution"],
                "num_labels": row["num_labels"],
            }
        )
    return pd.DataFrame(label_cols), pd.DataFrame(param_rows)


def run_sample(sample_id: str, args: argparse.Namespace) -> None:
    input_path = args.data_dir / "niche_features" / f"{sample_id}_with_niches.h5ad"
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    sample_output_dir = args.output_dir / sample_id
    sample_output_dir.mkdir(parents=True, exist_ok=True)
    cell_output_path = sample_output_dir / f"{sample_id}_banksy_cell_domains.csv.gz"
    param_output_path = sample_output_dir / f"{sample_id}_banksy_run_params.csv"

    if cell_output_path.exists() and param_output_path.exists() and not args.force:
        print(f"Using cached BANKSY output for {sample_id}: {cell_output_path}")
        return

    print(f"Loading {sample_id} from {input_path}")
    adata = load_and_prepare_adata(input_path, args, sample_id)
    print(f"{sample_id}: running BANKSY on {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    banksy_dict = initialize_banksy(
        adata,
        coord_keys=(args.x_key, args.y_key, "spatial"),
        num_neighbours=args.num_neighbours,
        nbr_weight_decay="scaled_gaussian",
        max_m=args.max_m,
        plt_edge_hist=False,
        plt_nbr_weights=False,
        plt_agf_angles=False,
        plt_theta=False,
    )
    results_df = run_banksy_multiparam(
        adata,
        banksy_dict,
        lambda_list=args.lambda_list,
        resolutions=args.resolutions,
        color_list=make_color_list(),
        max_m=args.max_m,
        filepath=str(sample_output_dir / "pybanksy_figures"),
        key=(args.x_key, args.y_key, "spatial"),
        annotation_key=None,
        pca_dims=args.pca_dims,
        savefig=args.save_figures,
        save_all_h5ad=False,
        add_nonspatial=False,
        partition_seed=args.random_state,
    )

    labels_df, params_df = extract_result_labels(results_df, adata.n_obs)
    metadata_cols = [
        "banksy_cell_id",
        "banksy_sample_id",
        args.x_key,
        args.y_key,
        args.phenotype_key,
        args.tier_b_key,
        args.niche_key,
    ]
    metadata_cols = [c for c in metadata_cols if c in adata.obs.columns]
    cell_df = adata.obs[metadata_cols].copy()
    cell_df = cell_df.reset_index(names="cell_id")
    cell_df = pd.concat([cell_df.reset_index(drop=True), labels_df.reset_index(drop=True)], axis=1)

    cell_df.to_csv(cell_output_path, index=False)
    params_df.to_csv(param_output_path, index=False)
    print(f"Saved {cell_output_path}")
    print(f"Saved {param_output_path}")

    del adata, banksy_dict, results_df, cell_df, labels_df, params_df
    gc.collect()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for sample_id in selected_samples(args.sample_id):
        run_sample(sample_id, args)


if __name__ == "__main__":
    main()
