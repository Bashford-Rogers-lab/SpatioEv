import anndata as ad
import numpy as np
import pandas as pd
import pytest

from spatioev.workflows.clustering import scoped_sample_id, subset_for_clustering


def test_fov_specific_clustering_scope_preserves_cell_ids():
    source = ad.AnnData(
        X=np.arange(12, dtype=float).reshape(4, 3),
        obs=pd.DataFrame(
            {"imageid": ["fov1", "fov1", "fov2", "fov2"]},
            index=["cell1", "cell2", "cell3", "cell4"],
        ),
    )

    selected = subset_for_clustering(source, {"cluster_imageid": "fov2"})

    assert list(selected.obs_names) == ["cell3", "cell4"]
    assert set(selected.obs["imageid"].astype(str)) == {"fov2"}
    assert scoped_sample_id("TMA1", {"cluster_imageid": "fov2"}) == "TMA1_fov2"
    assert subset_for_clustering(source, {}).n_obs == source.n_obs


def test_fov_specific_clustering_rejects_unknown_fov():
    source = ad.AnnData(
        X=np.zeros((2, 2)),
        obs=pd.DataFrame(
            {"imageid": ["fov1", "fov1"]}, index=["cell1", "cell2"]
        ),
    )
    with pytest.raises(ValueError, match="fov2"):
        subset_for_clustering(source, {"cluster_imageid": "fov2"})
