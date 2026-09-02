#!/usr/bin/env python3
"""Extract only the missing ark-analysis regionprops columns and merge them into a cell table.

This script is intended for cases where the existing cell/pixel feature table is
missing a small set of morphology features that were added later to a local
`ark-analysis` checkout. It computes only:

- orientation
- solidity
- feret_diameter_max
- circularity
- fractual_dimension
- boundary_irregularity

The extraction runs one FOV at a time from the whole-cell segmentation masks and
merges the results back into an existing CSV keyed by ``fov`` + ``label``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage.measure import regionprops


TARGET_BASE_PROPS = [
    "orientation",
    "solidity",
    "feret_diameter_max",
]

TARGET_CUSTOM_PROPS = [
    "circularity",
    "fractual_dimension",
    "boundary_irregularity",
]

TARGET_COLUMNS = TARGET_BASE_PROPS + TARGET_CUSTOM_PROPS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract missing ark-analysis regionprops columns from segmentation masks.",
    )
    parser.add_argument(
        "--pixel-features-csv",
        required=True,
        help="Path to the existing pixel/cell feature CSV containing at least fov and label.",
    )
    parser.add_argument(
        "--seg-dir",
        required=True,
        help="Directory containing per-FOV whole-cell segmentation masks.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Defaults to <pixel-features stem>_with_regionprops.csv",
    )
    parser.add_argument(
        "--ark-src",
        default="/Users/shihongwu/ark-analysis/src",
        help="Path to the local ark-analysis src directory.",
    )
    parser.add_argument(
        "--mask-suffix",
        default="_whole_cell.tiff",
        help="Filename suffix appended to each FOV name to find the segmentation mask.",
    )
    return parser.parse_args()


def load_ark_regionprops_functions(ark_src: str):
    sys.path.insert(0, ark_src)
    # Suppress matplotlib cache warnings that can appear during scientific stack imports.
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    from ark.segmentation.regionprops_extraction import REGIONPROPS_FUNCTION

    missing = [name for name in TARGET_CUSTOM_PROPS if name not in REGIONPROPS_FUNCTION]
    if missing:
        raise ValueError(
            f"The following custom ark regionprops are missing from REGIONPROPS_FUNCTION: {missing}"
        )
    return REGIONPROPS_FUNCTION


def compute_missing_props_for_fov(seg_path: Path, fov: str, valid_labels: set[int], regionprops_function):
    seg = tifffile.memmap(seg_path)
    seg = np.asarray(seg).squeeze()
    if seg.ndim != 2:
        raise ValueError(f"Expected a 2D segmentation mask at {seg_path}, got shape {seg.shape}")

    rows = []
    for prop in regionprops(seg):
        label = int(prop.label)
        if label not in valid_labels:
            continue
        rows.append(
            {
                "fov": fov,
                "label": label,
                "orientation": prop.orientation,
                "solidity": prop.solidity,
                "feret_diameter_max": prop.feret_diameter_max,
                "circularity": regionprops_function["circularity"](prop),
                "fractual_dimension": regionprops_function["fractual_dimension"](prop),
                "boundary_irregularity": regionprops_function["boundary_irregularity"](prop),
            }
        )

    return pd.DataFrame(rows, columns=["fov", "label"] + TARGET_COLUMNS)


def main() -> None:
    args = parse_args()

    pixel_features_path = Path(args.pixel_features_csv)
    seg_dir = Path(args.seg_dir)
    output_path = (
        Path(args.output)
        if args.output is not None
        else pixel_features_path.with_name(pixel_features_path.stem + "_with_regionprops.csv")
    )

    if not pixel_features_path.exists():
        raise FileNotFoundError(f"pixel features CSV not found: {pixel_features_path}")
    if not seg_dir.exists():
        raise FileNotFoundError(f"segmentation directory not found: {seg_dir}")

    regionprops_function = load_ark_regionprops_functions(args.ark_src)

    pixel_df = pd.read_csv(pixel_features_path)
    required_cols = {"fov", "label"}
    missing_required = required_cols.difference(pixel_df.columns)
    if missing_required:
        raise ValueError(f"pixel features CSV is missing required columns: {sorted(missing_required)}")

    pixel_df["label"] = pd.to_numeric(pixel_df["label"], errors="raise").astype(int)

    duplicate_ct = pixel_df.duplicated(subset=["fov", "label"]).sum()
    if duplicate_ct > 0:
        raise ValueError(
            f"pixel features CSV contains {duplicate_ct} duplicated (fov, label) rows; "
            "cannot safely merge one-to-one."
        )

    fovs = sorted(pixel_df["fov"].dropna().astype(str).unique().tolist())
    extracted_parts = []

    for idx, fov in enumerate(fovs, start=1):
        seg_path = seg_dir / f"{fov}{args.mask_suffix}"
        if not seg_path.exists():
            raise FileNotFoundError(f"Missing segmentation mask for {fov}: {seg_path}")

        valid_labels = set(pixel_df.loc[pixel_df["fov"] == fov, "label"].astype(int).tolist())
        print(f"[{idx}/{len(fovs)}] extracting missing regionprops for {fov} from {seg_path.name}")
        fov_df = compute_missing_props_for_fov(seg_path, fov, valid_labels, regionprops_function)
        print(f"    extracted {len(fov_df):,} rows")
        extracted_parts.append(fov_df)

    extracted_df = pd.concat(extracted_parts, ignore_index=True)

    extracted_dup_ct = extracted_df.duplicated(subset=["fov", "label"]).sum()
    if extracted_dup_ct > 0:
        raise ValueError(
            f"Extracted regionprops contain {extracted_dup_ct} duplicated (fov, label) rows."
        )

    # Replace any stale versions of the target columns.
    pixel_df = pixel_df.drop(columns=[col for col in TARGET_COLUMNS if col in pixel_df.columns])
    merged_df = pixel_df.merge(extracted_df, on=["fov", "label"], how="left", validate="one_to_one")

    missing_counts = merged_df[TARGET_COLUMNS].isna().sum().to_dict()
    print("Missing counts after merge:")
    for col, count in missing_counts.items():
        print(f"    {col}: {count:,}")

    merged_df.to_csv(output_path, index=False)
    print(f"Saved updated feature table to: {output_path}")


if __name__ == "__main__":
    main()
