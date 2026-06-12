from __future__ import annotations

import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "spatioev-mplconfig"),
)


@pytest.fixture()
def toy_adata() -> ad.AnnData:
    n = 12
    rng = np.random.default_rng(7)
    obs = pd.DataFrame(
        {
            "label": np.arange(n),
            "imageid": ["img1"] * 6 + ["img2"] * 6,
            "phenotype": ["duct", "duct", "immune", "immune", "stromal", "stromal"] * 2,
            "X_centroid": [0, 10, 20, 30, 40, 50, 0, 12, 25, 33, 48, 60],
            "Y_centroid": [0, 5, 10, 10, 5, 0, 40, 45, 52, 55, 50, 44],
            "area": np.linspace(60, 220, n),
            "eccentricity": np.linspace(0.2, 0.8, n),
            "major_axis_length": np.linspace(10, 30, n),
            "minor_axis_length": np.linspace(5, 15, n),
            "perimeter": np.linspace(30, 90, n),
            "convex_area": np.linspace(70, 250, n),
            "equivalent_diameter": np.linspace(8, 20, n),
            "orientation": np.linspace(-1, 1, n),
            "solidity": np.linspace(0.7, 0.98, n),
            "feret_diameter_max": np.linspace(12, 34, n),
            "major_minor_axis_ratio": np.linspace(1.1, 2.2, n),
            "perim_square_over_area": np.linspace(12, 18, n),
            "major_axis_equiv_diam_ratio": np.linspace(1.1, 1.8, n),
            "convex_hull_resid": np.linspace(0.01, 0.2, n),
            "centroid_dif": np.linspace(0.1, 2.0, n),
            "num_concavities": np.arange(n) % 4,
            "circularity": np.linspace(0.35, 0.95, n),
            "fractal_dimension": np.linspace(1.0, 1.4, n),
            "boundary_irregularity": np.linspace(0.1, 0.6, n),
            "nc_ratio": np.linspace(0.2, 0.8, n),
            "feature_a": np.linspace(-1, 1, n),
            "feature_b": rng.normal(size=n),
            "pseudotime": np.linspace(0, 1, n),
        },
        index=[f"cell_{i}" for i in range(n)],
    )
    X = rng.normal(size=(n, 3))
    var = pd.DataFrame(index=["CD8", "Ki67", "PCK"])
    return ad.AnnData(X=X, obs=obs, var=var)
