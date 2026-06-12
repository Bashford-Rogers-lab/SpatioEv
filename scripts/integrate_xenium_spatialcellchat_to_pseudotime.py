#!/usr/bin/env python
"""Collect SpatialCellChat outputs and summarize them by trajectory state."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("data/xenium_pancreas_10x/spatialcellchat/results"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/xenium_pancreas_10x/spatialcellchat/integrated"),
    )
    parser.add_argument("--top-n-interactions", type=int, default=40)
    return parser.parse_args()


def read_communication_files(results_dir: Path) -> pd.DataFrame:
    paths = sorted(results_dir.glob("*/*__communication_lr.csv"))
    frames = []
    for path in paths:
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        df["source_file"] = str(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename = {
        "source": "source_celltype",
        "target": "target_celltype",
        "ligand": "ligand",
        "receptor": "receptor",
        "prob": "communication_probability",
        "pval": "p_value",
        "pathway_name": "pathway",
        "interaction_name": "interaction_name",
    }
    for old, new in rename.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
    if "interaction_name" not in df.columns and {"ligand", "receptor"}.issubset(df.columns):
        df["interaction_name"] = df["ligand"].astype(str) + "_" + df["receptor"].astype(str)
    if "communication_probability" not in df.columns:
        for candidate in ["prob", "weight", "score"]:
            if candidate in df.columns:
                df["communication_probability"] = pd.to_numeric(df[candidate], errors="coerce")
                break
    if "communication_probability" not in df.columns:
        df["communication_probability"] = np.nan
    return df


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = read_communication_files(args.results_dir)
    if raw_df.empty:
        print(f"No SpatialCellChat communication CSVs found under {args.results_dir}")
        return

    comm_df = standardize_columns(raw_df)
    comm_df.to_csv(args.output_dir / "xenium_spatialcellchat_communication_long.csv", index=False)

    group_cols = [
        c
        for c in [
            "sample_id",
            "branch_time_state",
            "source_celltype",
            "target_celltype",
            "pathway",
            "interaction_name",
        ]
        if c in comm_df.columns
    ]
    summary = (
        comm_df.groupby(group_cols, dropna=False, observed=True)
        .agg(
            mean_probability=("communication_probability", "mean"),
            max_probability=("communication_probability", "max"),
            n_interactions=("interaction_name", "size"),
        )
        .reset_index()
    )
    summary.to_csv(args.output_dir / "xenium_spatialcellchat_trajectory_summary.csv", index=False)

    top = (
        summary.dropna(subset=["branch_time_state"])
        .sort_values("mean_probability", ascending=False)
        .head(args.top_n_interactions)
    )
    top.to_csv(args.output_dir / "xenium_spatialcellchat_top_trajectory_interactions.csv", index=False)

    print(f"Saved {args.output_dir / 'xenium_spatialcellchat_communication_long.csv'}")
    print(f"Saved {args.output_dir / 'xenium_spatialcellchat_trajectory_summary.csv'}")


if __name__ == "__main__":
    main()
