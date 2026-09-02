#!/usr/bin/env python
"""Run reference-style multi-sample pyBANKSY + Harmony for Xenium.

This follows the structure of the user's earlier BANKSY notebook:

1. Load several Xenium samples into one AnnData object using common genes.
2. Stagger sample coordinates so samples do not overlap in BANKSY's spatial graph.
3. Run official pyBANKSY matrix construction.
4. Harmony-correct BANKSY PCs by sample.
5. Run Leiden clustering on the Harmony-corrected BANKSY PCs.
6. Export BANKSY labels back into one per-sample CSV for SpatioEv integration.

Run with:

    NUMBA_CACHE_DIR=/private/tmp/numba_cache MPLCONFIGDIR=/private/tmp/mplconfig \
      /Users/shihongwu/anaconda3/envs/pybanksy/bin/python \
      scripts/run_xenium_banksy_harmony_multisample.py
"""

from __future__ import annotations

import argparse
import gc
import os
import pickle
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig")

import anndata as ad
import harmonypy as hm
import numpy as np
import pandas as pd
import scanpy as sc
import umap
from scipy import sparse

from banksy.cluster_methods import run_Leiden_partition
from banksy.embed_banksy import generate_banksy_matrix
from banksy.initialize_banksy import initialize_banksy
from banksy_utils.umap_pca import pca_umap


SAMPLE_IDS = (
    "pdac_pancreas_v1",
    "pdac_io_v1",
    "pdac_addon_v1",
    "normal_nondiseased_v1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/xenium_pancreas_10x"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/xenium_pancreas_10x/banksy"))
    parser.add_argument("--sample-id", nargs="+", default=["all"])
    parser.add_argument("--layer", default="log1p")
    parser.add_argument("--x-key", default="x_centroid")
    parser.add_argument("--y-key", default="y_centroid")
    parser.add_argument("--phenotype-key", default="Tier_A")
    parser.add_argument("--tier-b-key", default="Tier_B")
    parser.add_argument("--niche-key", default="xenium_ductal_epithelium_component")
    parser.add_argument("--num-neighbours", type=int, default=20)
    parser.add_argument("--max-m", type=int, default=1)
    parser.add_argument("--lambda-list", type=float, nargs="+", default=[0.8])
    parser.add_argument("--resolutions", type=float, nargs="+", default=[0.5])
    parser.add_argument("--pca-dims", type=int, nargs="+", default=[20])
    parser.add_argument("--num-nn", type=int, default=50)
    parser.add_argument("--n-top-genes", type=int, default=5000)
    parser.add_argument(
        "--max-cells-per-sample",
        type=int,
        default=0,
        help="Optional random downsample per sample. Use 0 for all cells.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--force", action="store_true")
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
        f"banksy_harmony__{decay_safe}"
        f"__lambda_{safe_float(float(lambda_param))}"
        f"__pc_{int(num_pcs)}"
        f"__res_{safe_float(float(resolution))}"
    )


def sample_h5ad_path(data_dir: Path, sample_id: str) -> Path:
    return data_dir / "niche_features" / f"{sample_id}_with_niches.h5ad"


def load_sample(path: Path, sample_id: str, args: argparse.Namespace) -> ad.AnnData:
    adata = sc.read_h5ad(path)
    adata.var_names_make_unique()
    valid = np.isfinite(pd.to_numeric(adata.obs[args.x_key], errors="coerce")) & np.isfinite(
        pd.to_numeric(adata.obs[args.y_key], errors="coerce")
    )
    adata = adata[np.asarray(valid)].copy()
    if args.max_cells_per_sample and adata.n_obs > args.max_cells_per_sample:
        rng = np.random.default_rng(args.random_state)
        idx = np.sort(rng.choice(adata.n_obs, size=args.max_cells_per_sample, replace=False))
        adata = adata[idx].copy()
    if args.layer in adata.layers:
        adata.X = adata.layers[args.layer].copy()
    elif args.layer != "X":
        raise KeyError(f"{path.name} does not contain layer {args.layer!r}")
    adata.obs["banksy_cell_id"] = adata.obs_names.astype(str)
    adata.obs["sample_id"] = sample_id
    adata.obs["banksy_x_original"] = pd.to_numeric(adata.obs[args.x_key], errors="coerce").to_numpy()
    adata.obs["banksy_y_original"] = pd.to_numeric(adata.obs[args.y_key], errors="coerce").to_numpy()
    return adata


def filter_nonzero_variable_genes(adata: ad.AnnData) -> ad.AnnData:
    if sparse.issparse(adata.X):
        gene_sum = np.asarray(adata.X.sum(axis=0)).ravel()
        gene_sq_sum = np.asarray(adata.X.power(2).sum(axis=0)).ravel()
        gene_var = gene_sq_sum / max(adata.n_obs, 1) - (gene_sum / max(adata.n_obs, 1)) ** 2
    else:
        gene_sum = np.asarray(adata.X.sum(axis=0)).ravel()
        gene_var = np.asarray(adata.X.var(axis=0)).ravel()
    keep = np.isfinite(gene_var) & (gene_sum > 0) & (gene_var > 0)
    return adata[:, keep].copy()


def load_multisample(args: argparse.Namespace) -> ad.AnnData:
    sample_ids = selected_samples(args.sample_id)
    samples = []
    common_genes: set[str] | None = None
    for sample_id in sample_ids:
        path = sample_h5ad_path(args.data_dir, sample_id)
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"Loading {sample_id}: {path}")
        sample = load_sample(path, sample_id, args)
        genes = set(map(str, sample.var_names))
        common_genes = genes if common_genes is None else common_genes.intersection(genes)
        samples.append(sample)

    if not common_genes:
        raise ValueError("No common genes across selected samples.")
    common_genes_sorted = sorted(common_genes)
    samples = [sample[:, common_genes_sorted].copy() for sample in samples]
    adata = ad.concat(samples, join="inner", merge="same", index_unique=None)
    adata = filter_nonzero_variable_genes(adata)

    if args.n_top_genes and args.n_top_genes < adata.n_vars:
        sc.pp.highly_variable_genes(adata, n_top_genes=args.n_top_genes, flavor="seurat")
        adata = adata[:, adata.var["highly_variable"].to_numpy()].copy()

    coords = adata.obs[["banksy_x_original", "banksy_y_original", "sample_id"]].copy()
    coords["banksy_x"] = coords.groupby("sample_id")["banksy_x_original"].transform(lambda x: x - x.min())
    coords["banksy_y"] = coords.groupby("sample_id")["banksy_y_original"].transform(lambda y: y - y.min())
    global_max_x = coords["banksy_x"].max() * 1.5
    sample_order = {sample_id: i for i, sample_id in enumerate(sample_ids)}
    coords["sample_no"] = coords["sample_id"].map(sample_order).astype(int)
    coords["banksy_x"] = coords["banksy_x"] + coords["sample_no"] * global_max_x
    adata.obs["banksy_x"] = coords["banksy_x"].to_numpy()
    adata.obs["banksy_y"] = coords["banksy_y"].to_numpy()
    adata.obsm["spatial"] = adata.obs[["banksy_x", "banksy_y"]].to_numpy(dtype=float)
    return adata


def run_harmony_on_banksy_dict(banksy_dict: dict, decay: str, lambda_list: list[float], pca_dims: list[int]) -> None:
    for lambda_value in lambda_list:
        bdata = banksy_dict[decay][lambda_value]["adata"]
        for pca_dim in pca_dims:
            key = f"reduced_pc_{pca_dim}"
            print(f"Harmony correcting {decay}, lambda={lambda_value}, {key}")
            harmony_res = hm.run_harmony(
                bdata.obsm[key],
                bdata.obs,
                "sample_id",
                random_state=42,
            )
            corrected = np.asarray(harmony_res.Z_corr)
            if corrected.shape[0] != bdata.n_obs and corrected.shape[1] == bdata.n_obs:
                corrected = corrected.T
            if corrected.shape[0] != bdata.n_obs:
                raise ValueError(f"Harmony output shape mismatch: {corrected.shape} vs {bdata.n_obs}")
            bdata.obsm[key] = corrected
            reducer = umap.UMAP(random_state=42)
            bdata.obsm[f"{key}_umap"] = reducer.fit_transform(corrected)


def extract_labels(results_df: pd.DataFrame, adata_obs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_cols = {}
    param_rows = []
    for params_name, row in results_df.iterrows():
        labels = row["labels"]
        if hasattr(labels, "dense"):
            labels = labels.dense
        labels = np.asarray(labels).astype(str)
        if len(labels) != len(adata_obs):
            raise ValueError(f"Label length mismatch for {params_name}: {len(labels)} vs {len(adata_obs)}")
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
                "harmony_corrected": True,
            }
        )
        if "relabeled" in results_df.columns:
            relabeled = row["relabeled"]
            if hasattr(relabeled, "dense"):
                relabeled = relabeled.dense
            label_cols[f"{key}__relabeled"] = np.asarray(relabeled).astype(str)
    return pd.DataFrame(label_cols), pd.DataFrame(param_rows)


def export_outputs(results_df: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> None:
    first_adata = results_df.iloc[0]["adata"]
    metadata_cols = [
        "banksy_cell_id",
        "sample_id",
        "banksy_x_original",
        "banksy_y_original",
        "banksy_x",
        "banksy_y",
        args.phenotype_key,
        args.tier_b_key,
        args.niche_key,
    ]
    metadata_cols = [c for c in metadata_cols if c in first_adata.obs.columns]
    meta = first_adata.obs[metadata_cols].copy().reset_index(names="cell_id")
    labels_df, params_df = extract_labels(results_df, first_adata.obs)
    all_df = pd.concat([meta.reset_index(drop=True), labels_df.reset_index(drop=True)], axis=1)

    for sample_id, sample_df in all_df.groupby("sample_id", observed=True):
        sample_dir = output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_df.to_csv(sample_dir / f"{sample_id}_banksy_cell_domains.csv.gz", index=False)
        params_df.to_csv(sample_dir / f"{sample_id}_banksy_run_params.csv", index=False)

    all_df.to_csv(output_dir / "xenium_banksy_harmony_multisample_cell_domains.csv.gz", index=False)
    params_df.to_csv(output_dir / "xenium_banksy_harmony_multisample_run_params.csv", index=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_pickle = args.output_dir / "xenium_banksy_harmony_multisample_results_df.pkl"
    if result_pickle.exists() and not args.force:
        print(f"Using cached results: {result_pickle}")
        with open(result_pickle, "rb") as handle:
            results_df = pickle.load(handle)
        export_outputs(results_df, args.output_dir, args)
        return

    adata = load_multisample(args)
    print(f"Combined BANKSY input: {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    banksy_dict = initialize_banksy(
        adata,
        coord_keys=("banksy_x", "banksy_y", "spatial"),
        num_neighbours=args.num_neighbours,
        nbr_weight_decay="scaled_gaussian",
        max_m=args.max_m,
        plt_edge_hist=False,
        plt_nbr_weights=False,
        plt_agf_angles=False,
        plt_theta=False,
    )
    banksy_dict, _ = generate_banksy_matrix(
        adata,
        banksy_dict,
        args.lambda_list,
        args.max_m,
        variance_balance=False,
    )
    pca_umap(banksy_dict, pca_dims=args.pca_dims, plt_remaining_var=False, add_umap=False)
    run_harmony_on_banksy_dict(
        banksy_dict,
        decay="scaled_gaussian",
        lambda_list=args.lambda_list,
        pca_dims=args.pca_dims,
    )
    results_df, max_num_labels = run_Leiden_partition(
        banksy_dict,
        args.resolutions,
        num_nn=args.num_nn,
        num_iterations=-1,
        partition_seed=args.random_state,
        match_labels=True,
    )
    results_df.attrs["max_num_labels"] = max_num_labels
    with open(result_pickle, "wb") as handle:
        pickle.dump(results_df, handle)
    export_outputs(results_df, args.output_dir, args)
    print(f"Saved {result_pickle}")
    del adata, banksy_dict, results_df
    gc.collect()


if __name__ == "__main__":
    main()
