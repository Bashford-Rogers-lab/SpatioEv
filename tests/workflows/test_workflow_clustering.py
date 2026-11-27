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


def test_zscore_within_each_fov_removes_a_core_wide_shift():
    from spatioev.pp.normalize import zscore_normalize

    rng = np.random.default_rng(0)
    base = rng.normal(size=(40, 2))
    # fov2 stained twice as bright and shifted: the same cells, a core-wide offset.
    source = ad.AnnData(
        X=np.vstack([base, 2 * base + 5]).astype(np.float32),
        obs=pd.DataFrame({"imageid": ["fov1"] * 40 + ["fov2"] * 40}, index=[f"c{i}" for i in range(80)]),
    )

    pooled = zscore_normalize(source.copy())
    within = zscore_normalize(source.copy(), groupby="imageid")

    assert pooled.X[:40].mean() < -0.5 < 0.5 < pooled.X[40:].mean()
    for rows in (slice(0, 40), slice(40, 80)):
        np.testing.assert_allclose(within.X[rows].mean(axis=0), 0, atol=1e-5)
        np.testing.assert_allclose(within.X[rows].std(axis=0), 1, atol=1e-5)
    np.testing.assert_allclose(within.X[:40], within.X[40:], atol=1e-5)
    np.testing.assert_allclose(within.layers["raw"], source.X)
    with pytest.raises(KeyError, match="batch"):
        zscore_normalize(source.copy(), groupby="batch")


def test_scale_within_image_is_off_unless_asked():
    from spatioev.workflows.clustering import scale_groupby

    assert scale_groupby({}) is None
    assert scale_groupby({"scale_within_image": False}) is None
    assert scale_groupby({"scale_within_image": True}) == "imageid"


def test_trimmed_knn_graph_keeps_the_nearest_neighbours_as_scanpy_does():
    import scipy.sparse

    from spatioev.tl.phenotype import trimmed_knn_graph

    # Four cells, each row holding itself and its three nearest, as
    # pynndescent 0.7 returns it: a csr_array, one entry more than kept, and
    # (in row 1) not in distance order.
    indices = np.array([[0, 1, 2, 3], [1, 3, 0, 2], [2, 1, 0, 3], [3, 2, 1, 0]])
    distances = np.array([[0, 1, 2, 3], [0, 3, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.float32)
    graph = scipy.sparse.csr_array(
        (distances.ravel(), indices.ravel(), np.arange(0, 17, 4)), shape=(4, 4)
    )

    trimmed = trimmed_knn_graph(graph, n_neighbors=3)

    assert isinstance(trimmed, scipy.sparse.csr_matrix)
    assert hasattr(trimmed, "getnnz")
    np.testing.assert_array_equal(np.diff(trimmed.indptr), [3, 3, 3, 3])
    rows = trimmed.indices.reshape(4, 3)
    np.testing.assert_array_equal(rows[:, 0], np.arange(4))
    np.testing.assert_array_equal(rows[1], [1, 0, 2])
    assert trimmed.data.reshape(4, 3).flags["C_CONTIGUOUS"]
    # Rows of unequal length are only converted.
    ragged = scipy.sparse.csr_array(scipy.sparse.random(5, 5, density=0.5, random_state=0))
    assert isinstance(trimmed_knn_graph(ragged, 2), scipy.sparse.csr_matrix)
    assert trimmed_knn_graph(ragged, 2).nnz == ragged.nnz


def test_knn_transformer_only_replaces_scanpys_approximate_search():
    from spatioev.tl.phenotype import APPROXIMATE_KNN_MIN_CELLS, knn_transformer

    assert knn_transformer(APPROXIMATE_KNN_MIN_CELLS - 1, 15) is None
    pytest.importorskip("pynndescent")
    pytest.importorskip("scanpy")
    transformer = knn_transformer(APPROXIMATE_KNN_MIN_CELLS, 15)
    assert transformer.get_params()["n_neighbors"] == 15


def test_cluster_cells_runs_on_more_cells_than_scanpys_exact_search_takes():
    """Regression: every real TMA failed here (scanpy 1.11.5, pynndescent 0.7, umap-learn 0.6)."""
    pytest.importorskip("scanpy")
    pytest.importorskip("pynndescent")
    from spatioev.config import ClusteringConfig
    from spatioev.tl.phenotype import APPROXIMATE_KNN_MIN_CELLS, cluster_cells

    rng = np.random.default_rng(0)
    n = APPROXIMATE_KNN_MIN_CELLS + 100
    groups = rng.integers(0, 3, n)
    centers = np.array([[4, 0, 0, 0], [0, 4, 0, 0], [0, 0, 4, 0]], dtype=np.float32)
    source = ad.AnnData(
        X=(centers[groups] + rng.normal(size=(n, 4))).astype(np.float32),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n)]),
    )
    source.var_names = ["A", "B", "C", "D"]

    clustered = cluster_cells(source, ClusteringConfig(markers=["A", "B", "C", "D"], resolution=0.1, n_neighbors=10, n_pcs=3))

    assert clustered.obsm["X_umap"].shape == (n, 2)
    assert clustered.uns["neighbors"]["params"]["n_neighbors"] == 10
    # The three groups come out as clusters.
    purity = pd.crosstab(clustered.obs["leiden"], groups).max(axis=1).sum() / n
    assert purity > 0.95
