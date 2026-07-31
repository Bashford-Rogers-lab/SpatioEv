#!/usr/bin/env python3
"""Open a clustering checkpoint as a spatial overlay in napari."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spatioev_clustering_review_matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "spatioev_clustering_review_numba"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

import anndata as ad

import spatioev as se


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adata", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--label", default="leiden")
    parser.add_argument("--imageid", default=None, help="Only display cells from this image/FOV")
    args = parser.parse_args()
    adata = ad.read_h5ad(args.adata)
    if args.label not in adata.obs:
        raise KeyError(f"{args.label!r} not present in {args.adata}")
    if args.imageid is not None:
        if "imageid" not in adata.obs:
            raise KeyError("AnnData has no 'imageid' column for FOV-specific review")
        adata = adata[adata.obs["imageid"].astype(str).eq(str(args.imageid))].copy()
        if adata.n_obs == 0:
            raise ValueError(f"No cells have imageid={args.imageid!r}")
    se.inspect_clusters(adata, image_path=str(args.image), label=args.label, block=True)


if __name__ == "__main__":
    main()
