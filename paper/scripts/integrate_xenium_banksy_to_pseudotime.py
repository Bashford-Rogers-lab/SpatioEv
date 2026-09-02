#!/usr/bin/env python
"""Summarize pyBANKSY cell domains onto Xenium epithelial niches.

The script consumes outputs from ``scripts/run_xenium_banksy.py`` and creates:

* cell-domain proportions inside each epithelial niche
* cell-domain proportions in graph-defined niche surroundings
* branch/time summaries that can be plotted alongside SpatioEv pseudotime
"""

from __future__ import annotations

import argparse
import gc
import math
from collections import Counter
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


SAMPLE_IDS = (
    "pdac_pancreas_v1",
    "pdac_io_v1",
    "pdac_addon_v1",
    "normal_nondiseased_v1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/xenium_pancreas_10x"))
    parser.add_argument("--banksy-dir", type=Path, default=Path("data/xenium_pancreas_10x/banksy"))
    parser.add_argument("--sample-id", nargs="+", default=["all"])
    parser.add_argument("--niche-key", default="xenium_ductal_epithelium_component")
    parser.add_argument("--pseudotime-key", default="xenium_pseudotime")
    parser.add_argument("--branch-key", default="major_branch")
    parser.add_argument("--surround-hops", type=int, default=5)
    parser.add_argument(
        "--primary-run-key",
        default=None,
        help="BANKSY run key to summarize. Defaults to the first cached run key.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def selected_samples(sample_args: list[str]) -> list[str]:
    if len(sample_args) == 1 and sample_args[0].lower() == "all":
        return list(SAMPLE_IDS)
    unknown = sorted(set(sample_args).difference(SAMPLE_IDS))
    if unknown:
        raise ValueError(f"Unknown sample IDs: {unknown}")
    return sample_args


def safe_label(value) -> str:
    return (
        str(value)
        .replace(" ", "_")
        .replace("/", "_")
        .replace(".", "p")
        .replace("-", "m")
        .replace(":", "_")
    )


def entropy_from_counts(counts: Counter) -> float:
    n = sum(counts.values())
    if n <= 0:
        return np.nan
    probs = np.array([v / n for v in counts.values() if v > 0], dtype=float)
    if len(probs) <= 1:
        return 0.0
    return float(-(probs * np.log2(probs)).sum() / math.log2(len(probs)))


def domain_summary(values: pd.Series, prefix: str) -> dict:
    values = values.dropna().astype(str)
    out = {f"{prefix}__n_cells": int(len(values))}
    if len(values) == 0:
        out[f"{prefix}__dominant_domain"] = pd.NA
        out[f"{prefix}__domain_entropy"] = np.nan
        return out
    counts = Counter(values)
    out[f"{prefix}__dominant_domain"] = counts.most_common(1)[0][0]
    out[f"{prefix}__domain_entropy"] = entropy_from_counts(counts)
    for domain, count in sorted(counts.items(), key=lambda item: str(item[0])):
        out[f"{prefix}__domain_{safe_label(domain)}__prop"] = count / len(values)
        out[f"{prefix}__domain_{safe_label(domain)}__n"] = int(count)
    return out


def get_surrounding_cells(A: sparse.spmatrix, niche_idx: np.ndarray, n_hops: int) -> np.ndarray:
    niche_set = set(np.asarray(niche_idx, dtype=int).tolist())
    visited = set(niche_set)
    frontier = set(niche_set)
    layers = []
    for _ in range(n_hops):
        next_frontier = set()
        for node in frontier:
            for nbr in A.getrow(node).indices:
                nbr = int(nbr)
                if nbr in visited:
                    continue
                next_frontier.add(nbr)
        if not next_frontier:
            break
        layers.append(np.array(sorted(next_frontier), dtype=int))
        visited.update(next_frontier)
        frontier = next_frontier
    if not layers:
        return np.empty(0, dtype=int)
    return np.concatenate(layers)


def choose_run_key(cell_df: pd.DataFrame, requested: str | None) -> str:
    run_keys = [c for c in cell_df.columns if c.startswith(("banksy__", "banksy_harmony__"))]
    run_keys = [c for c in run_keys if not c.endswith("__relabeled")]
    if not run_keys:
        raise ValueError("No BANKSY run columns found in cell-domain table.")
    if requested is not None:
        if requested not in run_keys:
            raise KeyError(f"Requested BANKSY run key not found: {requested}")
        return requested
    preferred = [c for c in run_keys if "__lambda_0p80__" in c and "__res_0p50" in c]
    if not preferred:
        preferred = [c for c in run_keys if "__lambda_0p20__" in c and "__res_0p80" in c]
    return preferred[0] if preferred else run_keys[0]


def summarize_sample(sample_id: str, args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    cell_path = args.banksy_dir / sample_id / f"{sample_id}_banksy_cell_domains.csv.gz"
    h5ad_path = args.data_dir / "niche_features" / f"{sample_id}_with_niches.h5ad"
    if not cell_path.exists():
        raise FileNotFoundError(f"Missing BANKSY cell domains for {sample_id}: {cell_path}")
    if not h5ad_path.exists():
        raise FileNotFoundError(h5ad_path)

    cell_df = pd.read_csv(cell_path)
    run_key = choose_run_key(cell_df, args.primary_run_key)
    print(f"{sample_id}: summarizing {run_key}")

    adata = ad.read_h5ad(h5ad_path)
    A = adata.obsp["cell_graph_connectivities"].tocsr()
    obs = adata.obs[[args.niche_key]].copy()
    obs["_cell_id"] = adata.obs_names.astype(str)
    obs["_row"] = np.arange(adata.n_obs)

    domain_map = cell_df.set_index("cell_id")[run_key].astype(str)
    obs["_banksy_domain"] = obs["_cell_id"].map(domain_map)

    rows = []
    niche_values = obs[args.niche_key].dropna().unique().tolist()
    for niche_value in niche_values:
        niche_mask = obs[args.niche_key] == niche_value
        niche_idx = obs.loc[niche_mask, "_row"].to_numpy(dtype=int)
        if len(niche_idx) == 0:
            continue

        surround_idx = get_surrounding_cells(A, niche_idx, args.surround_hops)
        row = {
            "sample_id": sample_id,
            args.niche_key: niche_value,
            "banksy_run_key": run_key,
        }
        row.update(domain_summary(obs.iloc[niche_idx]["_banksy_domain"], "banksy_intrinsic"))
        row.update(domain_summary(obs.iloc[surround_idx]["_banksy_domain"], "banksy_surround"))
        rows.append(row)

    del adata, A, obs, cell_df
    gc.collect()
    return pd.DataFrame(rows), run_key


def add_branch_time_bin(df: pd.DataFrame, pseudotime_key: str, branch_key: str) -> pd.DataFrame:
    df = df.copy()
    df["branch_time_bin"] = pd.NA
    for branch, idx in df.groupby(branch_key, observed=True).groups.items():
        values = pd.to_numeric(df.loc[idx, pseudotime_key], errors="coerce")
        valid_idx = values.dropna().index
        if len(valid_idx) < 9 or values.loc[valid_idx].nunique() < 3:
            continue
        labels = ["early", "mid", "late"]
        df.loc[valid_idx, "branch_time_bin"] = pd.qcut(
            values.loc[valid_idx],
            q=3,
            labels=labels,
            duplicates="drop",
        ).astype(str)
    df["branch_time_state"] = df[branch_key].astype(str) + "::" + df["branch_time_bin"].astype(str)
    df.loc[df["branch_time_bin"].isna(), "branch_time_state"] = pd.NA
    return df


def main() -> None:
    args = parse_args()
    output_path = args.banksy_dir / "xenium_banksy_niche_context.pkl"
    merged_path = args.banksy_dir / "xenium_pseudotime_with_banksy.pkl"
    summary_path = args.banksy_dir / "xenium_banksy_branch_time_summary.csv"
    args.banksy_dir.mkdir(parents=True, exist_ok=True)

    if merged_path.exists() and summary_path.exists() and not args.force:
        print(f"Using cached integrated BANKSY table: {merged_path}")
        return

    sample_frames = []
    run_keys = {}
    for sample_id in selected_samples(args.sample_id):
        frame, run_key = summarize_sample(sample_id, args)
        sample_frames.append(frame)
        run_keys[sample_id] = run_key

    banksy_context_df = pd.concat(sample_frames, ignore_index=True)
    banksy_context_df.to_pickle(output_path)
    banksy_context_df.to_csv(args.banksy_dir / "xenium_banksy_niche_context.csv", index=False)

    pseudotime_path = args.data_dir / "pseudotime" / "xenium_pseudotime_result_df.pkl"
    pseudotime_df = pd.read_pickle(pseudotime_path)
    merged = pseudotime_df.merge(
        banksy_context_df,
        on=["sample_id", args.niche_key],
        how="left",
        validate="one_to_one",
    )
    merged = add_branch_time_bin(merged, args.pseudotime_key, args.branch_key)
    merged.to_pickle(merged_path)
    merged.to_csv(args.banksy_dir / "xenium_pseudotime_with_banksy.csv", index=False)

    value_cols = [
        c
        for c in merged.columns
        if c.startswith("banksy_intrinsic__domain_")
        or c.startswith("banksy_surround__domain_")
        or c.endswith("__domain_entropy")
    ]
    summary = (
        merged.dropna(subset=["branch_time_state"])
        .groupby(["major_branch", "branch_time_bin", "branch_time_state"], observed=True)[value_cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    summary.to_csv(summary_path, index=False)
    print(f"Saved {output_path}")
    print(f"Saved {merged_path}")
    print(f"Saved {summary_path}")
    print("BANKSY run keys by sample:")
    print(run_keys)


if __name__ == "__main__":
    main()
