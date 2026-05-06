#!/usr/bin/env python3
"""Extract per-nucleus DAPI pixel features from a 10x Xenium outs directory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/numba")

import pandas as pd

from spatioev.spatial.cell_pixel_features import extract_xenium_dapi_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outs-path", required=True, help="Path to the Xenium outs directory.")
    parser.add_argument("--output-csv", required=True, help="Output CSV path.")
    parser.add_argument(
        "--annotated-h5ad",
        default=None,
        help="Optional annotated h5ad used to restrict cells by Tier_A or provide cell IDs.",
    )
    parser.add_argument(
        "--tier-a",
        default=None,
        help="Optional Tier_A label to extract, for example 'pancreatic ductal epithelium'. Requires --annotated-h5ad.",
    )
    parser.add_argument(
        "--cell-id-csv",
        default=None,
        help="Optional CSV containing a cell_id column. Applied after --tier-a if both are provided.",
    )
    parser.add_argument("--cell-id-column", default="cell_id", help="Cell ID column for --cell-id-csv.")
    parser.add_argument(
        "--image-kind",
        default="auto",
        choices=["auto", "mip", "focus", "focus_dir0", "zstack"],
        help="Morphology image to use. auto prefers MIP, then focus, then focus-folder image, then z-stack.",
    )
    parser.add_argument(
        "--image-path",
        default=None,
        help="Optional explicit morphology OME-TIFF path. Overrides --image-kind selection.",
    )
    parser.add_argument(
        "--channel-index",
        type=int,
        default=0,
        help="Channel index for multi-channel morphology images. Xenium focus DAPI is usually channel 0.",
    )
    parser.add_argument(
        "--z-projection",
        default="max",
        choices=["max", "mean", "middle"],
        help="Projection used only when reading morphology.ome.tif z-stacks.",
    )
    parser.add_argument("--max-cells", type=int, default=None, help="Optional random pilot subset size.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-texture", action="store_true", help="Skip entropy/lacunarity.")
    parser.add_argument("--compute-haralick", action="store_true", help="Compute slower Haralick texture features.")
    parser.add_argument("--progress-every", type=int, default=10000)
    return parser.parse_args()


def load_cell_ids(args: argparse.Namespace):
    cell_ids = None
    if args.annotated_h5ad is not None:
        import scanpy as sc

        adata = sc.read_h5ad(args.annotated_h5ad, backed="r")
        try:
            if args.tier_a is not None:
                if "Tier_A" not in adata.obs.columns:
                    raise ValueError("The annotated h5ad does not contain obs['Tier_A'].")
                cell_ids = adata.obs_names[adata.obs["Tier_A"].astype(str) == args.tier_a].astype(str).tolist()
            else:
                cell_ids = adata.obs_names.astype(str).tolist()
        finally:
            adata.file.close()

    if args.cell_id_csv is not None:
        ids = pd.read_csv(args.cell_id_csv)[args.cell_id_column].astype(str).tolist()
        if cell_ids is None:
            cell_ids = ids
        else:
            keep = set(ids)
            cell_ids = [cell_id for cell_id in cell_ids if cell_id in keep]
    return cell_ids


def main() -> None:
    args = parse_args()
    cell_ids = load_cell_ids(args)
    if cell_ids is not None:
        print(f"Restricting extraction to {len(cell_ids):,} requested cell IDs.")

    output_csv = Path(args.output_csv)
    df = extract_xenium_dapi_features(
        outs_path=args.outs_path,
        cell_ids=cell_ids,
        output_path=output_csv,
        image_kind=args.image_kind,
        image_path=args.image_path,
        channel_index=args.channel_index,
        z_projection=args.z_projection,
        max_cells=args.max_cells,
        random_state=args.random_state,
        compute_texture=not args.no_texture,
        compute_haralick=args.compute_haralick,
        progress_every=args.progress_every,
    )
    print(f"Wrote {len(df):,} rows to {output_csv}")


if __name__ == "__main__":
    main()
