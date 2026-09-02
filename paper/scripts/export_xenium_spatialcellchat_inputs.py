#!/usr/bin/env python
"""Export Xenium trajectory-state inputs for SpatialCellChat.

SpatialCellChat runs in R, so this script writes minimal, method-friendly inputs:

* normalized log-expression matrix as Matrix Market, genes x cells
* cell metadata with Tier_A/Tier_B, coordinates, and assigned trajectory state
* genes and cell IDs

The trajectory state assignment maps epithelial cells directly by their
SpatioEv epithelial component and maps neighboring non-epithelial cells to the
nearest graph-hop epithelial niche state. This gives SpatialCellChat a proper
spatial single-cell input while keeping the output directly linkable to the
pseudotime tree.
"""

from __future__ import annotations

import argparse
import gc
import gzip
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import io, sparse


SAMPLE_IDS = (
    "pdac_pancreas_v1",
    "pdac_io_v1",
    "pdac_addon_v1",
    "normal_nondiseased_v1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/xenium_pancreas_10x"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/xenium_pancreas_10x/spatialcellchat/input"),
    )
    parser.add_argument("--sample-id", nargs="+", default=["all"])
    parser.add_argument("--layer", default="log1p")
    parser.add_argument("--niche-key", default="xenium_ductal_epithelium_component")
    parser.add_argument("--pseudotime-key", default="xenium_pseudotime")
    parser.add_argument("--branch-key", default="major_branch")
    parser.add_argument("--surround-hops", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def selected_samples(sample_args: list[str]) -> list[str]:
    if len(sample_args) == 1 and sample_args[0].lower() == "all":
        return list(SAMPLE_IDS)
    unknown = sorted(set(sample_args).difference(SAMPLE_IDS))
    if unknown:
        raise ValueError(f"Unknown sample IDs: {unknown}")
    return sample_args


def add_branch_time_bin(df: pd.DataFrame, pseudotime_key: str, branch_key: str) -> pd.DataFrame:
    df = df.copy()
    df["branch_time_bin"] = pd.NA
    for _, idx in df.groupby(branch_key, observed=True).groups.items():
        values = pd.to_numeric(df.loc[idx, pseudotime_key], errors="coerce")
        valid_idx = values.dropna().index
        if len(valid_idx) < 9 or values.loc[valid_idx].nunique() < 3:
            continue
        df.loc[valid_idx, "branch_time_bin"] = pd.qcut(
            values.loc[valid_idx],
            q=3,
            labels=["early", "mid", "late"],
            duplicates="drop",
        ).astype(str)
    df["branch_time_state"] = df[branch_key].astype(str) + "::" + df["branch_time_bin"].astype(str)
    df.loc[df["branch_time_bin"].isna(), "branch_time_state"] = pd.NA
    return df


def assign_cells_to_trajectory_states(
    adata: ad.AnnData,
    sample_pt: pd.DataFrame,
    niche_key: str,
    surround_hops: int,
) -> pd.DataFrame:
    obs = adata.obs[[niche_key, "Tier_A", "Tier_B", "x_centroid", "y_centroid"]].copy()
    obs["cell_id"] = adata.obs_names.astype(str)
    obs["trajectory_niche"] = pd.NA
    obs["trajectory_assignment_hop"] = pd.NA

    valid_niches = set(sample_pt[niche_key].dropna().tolist())
    direct = obs[niche_key].isin(valid_niches)
    obs.loc[direct, "trajectory_niche"] = obs.loc[direct, niche_key]
    obs.loc[direct, "trajectory_assignment_hop"] = 0

    A = adata.obsp["cell_graph_connectivities"].tocsr()
    niche_values = sample_pt[niche_key].dropna().unique().tolist()
    row_index = pd.Series(np.arange(adata.n_obs), index=adata.obs_names)
    niche_values_by_row = obs[niche_key].to_numpy()

    for hop in range(1, surround_hops + 1):
        unassigned = obs["trajectory_niche"].isna().to_numpy()
        if not unassigned.any():
            break
        for niche_value in niche_values:
            niche_idx = np.flatnonzero(niche_values_by_row == niche_value)
            if len(niche_idx) == 0:
                continue
            frontier = set(niche_idx.tolist())
            visited = set(niche_idx.tolist())
            for _ in range(hop):
                next_frontier = set()
                for node in frontier:
                    for nbr in A.getrow(node).indices:
                        nbr = int(nbr)
                        if nbr in visited:
                            continue
                        next_frontier.add(nbr)
                visited.update(next_frontier)
                frontier = next_frontier
                if not frontier:
                    break
            if not frontier:
                continue
            candidate_idx = np.fromiter(frontier, dtype=int)
            candidate_idx = candidate_idx[unassigned[candidate_idx]]
            if len(candidate_idx) == 0:
                continue
            obs.iloc[candidate_idx, obs.columns.get_loc("trajectory_niche")] = niche_value
            obs.iloc[candidate_idx, obs.columns.get_loc("trajectory_assignment_hop")] = hop

    state_map = sample_pt.set_index(niche_key)[
        [
            "branch_time_state",
            "branch_time_bin",
            "major_branch",
            "xenium_pseudotime",
            "xenium_pseudotime_norm",
        ]
    ]
    obs = obs.merge(
        state_map,
        left_on="trajectory_niche",
        right_index=True,
        how="left",
    )
    obs["has_trajectory_state"] = obs["branch_time_state"].notna()
    return obs


def write_matrix_market_gz(matrix: sparse.spmatrix, path: Path) -> None:
    with gzip.open(path, "wb") as handle:
        io.mmwrite(handle, matrix)


def export_sample(sample_id: str, args: argparse.Namespace, pseudotime_df: pd.DataFrame) -> None:
    out_dir = args.output_dir / sample_id
    metadata_path = out_dir / "metadata.csv.gz"
    matrix_path = out_dir / "expression_log1p_genes_by_cells.mtx.gz"
    genes_path = out_dir / "genes.csv"
    cells_path = out_dir / "cells.csv"
    if metadata_path.exists() and matrix_path.exists() and not args.force:
        print(f"Using cached SpatialCellChat input for {sample_id}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    h5ad_path = args.data_dir / "niche_features" / f"{sample_id}_with_niches.h5ad"
    print(f"Loading {sample_id}: {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    sample_pt = pseudotime_df.loc[pseudotime_df["sample_id"] == sample_id].copy()
    sample_pt = add_branch_time_bin(sample_pt, args.pseudotime_key, args.branch_key)

    metadata = assign_cells_to_trajectory_states(
        adata,
        sample_pt,
        niche_key=args.niche_key,
        surround_hops=args.surround_hops,
    )
    metadata["sample_id"] = sample_id
    metadata = metadata.rename(columns={"x_centroid": "x", "y_centroid": "y"})

    if args.layer in adata.layers:
        X = adata.layers[args.layer]
    elif args.layer == "X":
        X = adata.X
    else:
        raise KeyError(f"{sample_id} missing layer {args.layer!r}")
    X = sparse.csr_matrix(X).T.tocoo()

    metadata.to_csv(metadata_path, index=False)
    pd.DataFrame({"gene": adata.var_names.astype(str)}).to_csv(genes_path, index=False)
    pd.DataFrame({"cell_id": adata.obs_names.astype(str)}).to_csv(cells_path, index=False)
    write_matrix_market_gz(X, matrix_path)

    state_counts = (
        metadata.groupby(["branch_time_state", "Tier_A"], dropna=False)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    state_counts.to_csv(out_dir / "trajectory_state_cell_counts.csv", index=False)

    print(f"Saved SpatialCellChat input for {sample_id} to {out_dir}")
    del adata, metadata, X
    gc.collect()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pseudotime_df = pd.read_pickle(args.data_dir / "pseudotime" / "xenium_pseudotime_result_df.pkl")
    for sample_id in selected_samples(args.sample_id):
        export_sample(sample_id, args, pseudotime_df)


if __name__ == "__main__":
    main()
