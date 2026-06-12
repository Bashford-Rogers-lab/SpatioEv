from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from spatioev.pp import QCConfig, run_segmentation_qc, validate_spatial_coordinates
from spatioev.tl import (
    assign_tiles,
    build_feature_matrix,
    compute_general_density,
    compute_radius_density,
)

EXAMPLE_H5AD = Path("data/exp_2/34434_1_adata.h5ad")


@pytest.mark.skipif(
    not EXAMPLE_H5AD.exists(),
    reason="Local example dataset is not included in lightweight GitHub checkouts.",
)
def test_existing_exp2_dataset_smoke():
    adata = ad.read_h5ad(EXAMPLE_H5AD, backed="r")[:250].to_memory()

    validate_spatial_coordinates(adata)
    qc = run_segmentation_qc(adata.copy(), QCConfig(pixel_size=0.325))
    assert "area_category" in qc.obs

    tiled = assign_tiles(qc, tile_size=256)
    density = compute_general_density(tiled, tile_size=256)
    assert not density.empty

    density_adata = compute_radius_density(qc.copy(), radius=100)
    assert density_adata.obs["radius_density"].notna().any()

    X = build_feature_matrix(qc, markers=["CD8", "Ki67"], morph_weight=0.2)
    assert X.shape[0] == qc.n_obs
    assert np.isfinite(X).all()
