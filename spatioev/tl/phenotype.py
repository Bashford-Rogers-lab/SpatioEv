"""Cell phenotyping helpers: clustering, subsetting, annotation merging, and refinement."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import anndata as ad

from spatioev.config import ClusteringConfig


# ============================================================
# Section 1: Cell clustering  (from archive/phenotype/clustering.py)
# ============================================================

def _require_scanpy():
    try:
        import scanpy as sc
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        raise ImportError(
            "cluster_cells requires the optional Scanpy dependency. "
            "Install SpatioEv with `pip install -e '.[scanpy]'` or "
            "`pip install scanpy`."
        ) from exc

    return sc


def cluster_cells(
    adata: ad.AnnData,
    config: ClusteringConfig,
) -> ad.AnnData:
    """Run PCA → kNN → UMAP → Leiden clustering on a marker subset.

    Parameters
    ----------
    adata : anndata.AnnData
        Full AnnData object.  Only the markers listed in *config* are used.
    config : ClusteringConfig
        Clustering parameters (markers, resolution, n_neighbors, n_pcs).

    Returns
    -------
    anndata.AnnData
        Subset AnnData (cells × selected markers) with ``leiden`` cluster
        labels in ``adata.obs["leiden"]`` and UMAP coordinates in
        ``adata.obsm["X_umap"]``.
    """
    sc = _require_scanpy()

    markers = config.markers

    adata_sub = adata[:, markers].copy()

    sc.tl.pca(adata_sub)
    sc.pp.neighbors(
        adata_sub,
        n_neighbors=config.n_neighbors,
        n_pcs=config.n_pcs,
    )

    sc.tl.umap(adata_sub)

    sc.tl.leiden(
        adata_sub,
        resolution=config.resolution,
    )

    return adata_sub


# ============================================================
# Section 2: Cell subsetting  (from archive/phenotype/subset.py)
# ============================================================

def subset_cells(
    adata: ad.AnnData,
    annotation_key: str,
    labels: str | list[str],
) -> ad.AnnData:
    """Return a copy of *adata* keeping only cells with the given labels.

    Parameters
    ----------
    adata : anndata.AnnData
        Full AnnData object.
    annotation_key : str
        Column in ``adata.obs`` to filter on.
    labels : str or list of str
        One or more label values to retain.

    Returns
    -------
    anndata.AnnData
        Copy of *adata* containing only matching cells.
    """
    if isinstance(labels, str):
        labels = [labels]

    subset = adata[
        adata.obs[annotation_key].isin(labels)
    ].copy()

    return subset


# ============================================================
# Section 3: Annotation merging  (from archive/phenotype/merge.py)
# ============================================================

def merge_annotations(
    adata: ad.AnnData,
    subset: ad.AnnData,
    new_key: str = "annotation_level2",
) -> ad.AnnData:
    """Write sub-cluster annotations from *subset* back into *adata*.

    Parameters
    ----------
    adata : anndata.AnnData
        Parent AnnData whose ``obs`` will be updated.
    subset : anndata.AnnData
        Sub-clustered AnnData with an ``"annotation"`` column in
        ``subset.obs``.  Its ``obs_names`` must be a subset of
        *adata*'s ``obs_names``.
    new_key : str
        Target column in ``adata.obs``.  Created if absent.

    Returns
    -------
    anndata.AnnData
        *adata* with *new_key* updated in-place (and returned).
    """
    if new_key not in adata.obs:
        adata.obs[new_key] = None

    adata.obs.loc[
        subset.obs_names,
        new_key
    ] = subset.obs["annotation"]

    return adata


def merge_refinements(
    adata: ad.AnnData,
    refined_results: dict[str, ad.AnnData],
    new_key: str = "annotation_level2",
) -> ad.AnnData:
    """Merge refined sub-cluster annotations from multiple subsets into *adata*.

    Parameters
    ----------
    adata : anndata.AnnData
        Parent AnnData object.
    refined_results : dict of str → AnnData
        Mapping from cluster name to a sub-clustered AnnData (output of
        :func:`refine_clusters`).  Each AnnData must have ``"annotation"``
        in ``obs``.
    new_key : str
        Target column in ``adata.obs``.  Initialised from the existing
        ``"annotation"`` column if absent.

    Returns
    -------
    anndata.AnnData
        *adata* with *new_key* updated in-place (and returned).
    """
    if new_key not in adata.obs:
        adata.obs[new_key] = adata.obs.get("annotation", None)

    # convert categorical to string to allow new labels
    adata.obs[new_key] = adata.obs[new_key].astype("object")

    for subset in refined_results.values():

        if "annotation" not in subset.obs:
            continue

        adata.obs.loc[
            subset.obs_names,
            new_key
        ] = subset.obs["annotation"].astype("object")

    return adata


# ============================================================
# Section 4: Cluster refinement  (from archive/phenotype/refine.py)
# ============================================================

def refine_clusters(
    adata: ad.AnnData,
    clusters: list[str],
    configs: dict[str, ClusteringConfig],
    annotation_key: str = "annotation",
) -> dict[str, ad.AnnData]:
    """Re-cluster a set of named clusters with per-cluster configurations.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData with annotation labels in ``adata.obs[annotation_key]``.
    clusters : list of str
        Cluster names to refine.
    configs : dict of str → ClusteringConfig
        Per-cluster clustering configuration.  Each key must appear in
        *clusters*.
    annotation_key : str
        Column in ``adata.obs`` identifying cluster membership.

    Returns
    -------
    dict of str → AnnData
        Mapping from cluster name to its sub-clustered AnnData.
    """
    results: dict[str, ad.AnnData] = {}

    for cluster in clusters:

        if cluster not in configs:
            raise ValueError(f"No config provided for cluster '{cluster}'")

        subset = adata[
            adata.obs[annotation_key] == cluster
        ].copy()

        if subset.n_obs == 0:
            print(f"Skipping {cluster} (no cells)")
            continue

        config = configs[cluster]

        clustered = cluster_cells(subset, config)

        results[cluster] = clustered

    return results


__all__ = [
    "cluster_cells",
    "subset_cells",
    "merge_annotations",
    "merge_refinements",
    "refine_clusters",
]
