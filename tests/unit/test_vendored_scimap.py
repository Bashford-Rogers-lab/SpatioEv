"""The vendored scimap functions must work without scimap installed.

These guard the vendoring decision: ``rescale`` and ``phenotype_cells`` are
copied into ``spatioev._vendor.scimap`` precisely so that prior-knowledge
phenotyping does not drag in scimap's ~28 transitive dependencies (napari,
PyQt6, numba, gensim, ...) or its ``zarr==2.10.3`` pin.
"""

from __future__ import annotations

import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import spatioev as sv


@pytest.fixture
def marker_adata() -> ad.AnnData:
    """400 cells, three mutually exclusive marker-high populations."""
    rng = np.random.default_rng(0)
    n = 400

    def marker(high: np.ndarray) -> np.ndarray:
        values = rng.lognormal(0.2, 0.3, n)
        values[high] = rng.lognormal(2.5, 0.3, int(high.sum()))
        return values

    tcell = np.zeros(n, dtype=bool)
    tcell[:150] = True
    bcell = np.zeros(n, dtype=bool)
    bcell[150:280] = True
    tumour = np.zeros(n, dtype=bool)
    tumour[280:] = True

    adata = ad.AnnData(X=np.column_stack([marker(tcell), marker(bcell), marker(tumour)]))
    adata.var_names = ["CD3", "CD20", "PANCK"]
    adata.obs_names = [f"c{i}" for i in range(n)]
    adata.obs["imageid"] = "img1"
    adata.uns["_truth"] = np.where(tcell, "Tcell", np.where(bcell, "Bcell", "Tumour"))
    return adata


def _workflow() -> pd.DataFrame:
    # scimap layout: two leading unnamed columns, then one column per marker.
    return pd.DataFrame(
        [
            ["all", "Tcell", "pos", "", ""],
            ["all", "Bcell", "", "pos", ""],
            ["all", "Tumour", "", "", "pos"],
        ],
        columns=["", "_", "CD3", "CD20", "PANCK"],
    )


def test_vendored_modules_do_not_import_scimap():
    from spatioev._vendor import scimap as vendored

    assert vendored.rescale is not None
    assert vendored.phenotype_cells is not None
    # Importing the vendored code must not have pulled in the real package.
    assert "scimap" not in sys.modules


def test_rescale_maps_into_unit_interval(marker_adata):
    rescaled = sv.tl.scimap_rescale(marker_adata)
    values = np.asarray(rescaled.X, dtype=float)

    assert not np.isnan(values).any()
    # MinMaxScaler can overshoot the bound by ~1 ulp.
    assert values.min() >= -1e-9
    assert values.max() <= 1.0 + 1e-9
    # A gate-based rescale must place the marker-high cells above the 0.5 gate.
    assert values[:150, 0].mean() > 0.5
    assert values[150:280, 1].mean() > 0.5


def test_phenotype_cells_recovers_marker_high_populations(marker_adata):
    rescaled = sv.tl.scimap_rescale(marker_adata)
    out = sv.tl.scimap_phenotype_cells(rescaled, phenotype=_workflow(), label="phenotype")

    assert "phenotype" in out.obs
    labels = out.obs["phenotype"].to_numpy()
    assert set(labels) <= {"Tcell", "Bcell", "Tumour", "Unknown"}

    # CD3-high cells are the cleanest separated population; require most of them.
    truth = marker_adata.uns["_truth"]
    tcell_recall = (labels[truth == "Tcell"] == "Tcell").mean()
    assert tcell_recall > 0.8


def test_combined_workflow_runs_without_scimap(marker_adata):
    out = sv.tl.run_scimap_prior_knowledge_phenotyping(
        marker_adata, phenotype_workflow=_workflow(), label="phenotype"
    )
    assert "phenotype" in out.obs
    assert "scimap" not in sys.modules
