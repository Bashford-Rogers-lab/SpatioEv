"""Cell phenotyping helpers: clustering, subsetting, annotation merging, and refinement."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from spatioev.config import ClusteringConfig

if TYPE_CHECKING:
    import anndata as ad


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


def _require_scimap():
    """Import scimap, which is only needed for the interactive Napari gater.

    The two algorithmic scimap functions SpatioEv uses (``rescale`` and
    ``phenotype_cells``) are vendored under :mod:`spatioev._vendor.scimap`
    and need no scimap install. Only the Napari-backed gating UI still
    requires the real package.
    """
    try:
        import scimap as sm
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        raise ImportError(
            "Interactive Napari gating requires the optional scimap dependency. "
            "Install SpatioEv with `pip install -e '.[gating]'` or install "
            "`scimap[napari]`. Note that prior-knowledge phenotyping "
            "(scimap_rescale / scimap_phenotype_cells) does not need this."
        ) from exc

    return sm


def _read_csv_if_path(value: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value
    return pd.read_csv(value)


def scimap_napari_gater(
    adata: ad.AnnData,
    image_path: str | Path,
    **kwargs: Any,
) -> ad.AnnData:
    """Launch Scimap's Napari gate finder for manual marker thresholds.

    This is a light wrapper around ``scimap.pl.napariGater``. The function
    mutates ``adata.uns["gates"]`` in the same way as Scimap and returns
    ``adata`` for workflow chaining.
    """
    sm = _require_scimap()
    sm.pl.napariGater(str(image_path), adata, **kwargs)
    return adata


def scimap_rescale(
    adata: ad.AnnData,
    gate: pd.DataFrame | str | Path | None = None,
    **kwargs: Any,
) -> ad.AnnData:
    """Rescale marker intensities with Scimap gates.

    Wraps Scimap's ``rescale``. ``gate`` may be a manual-gates DataFrame,
    a CSV path, or ``None`` to use ``adata.uns["gates"]`` or the fallback
    GMM gate estimation.

    Uses the vendored implementation in :mod:`spatioev._vendor.scimap`, so
    no scimap install is required.
    """
    from spatioev._vendor.scimap import rescale as _rescale

    gate_df = None if gate is None else _read_csv_if_path(gate)
    return _rescale(adata, gate=gate_df, **kwargs)


def scimap_phenotype_cells(
    adata: ad.AnnData,
    phenotype: pd.DataFrame | str | Path,
    label: str = "phenotype",
    **kwargs: Any,
) -> ad.AnnData:
    """Run Scimap prior-knowledge hierarchical phenotyping.

    Wraps Scimap's ``phenotype_cells`` using a phenotype workflow table.
    The table may be provided as a DataFrame or CSV path.

    Uses the vendored implementation in :mod:`spatioev._vendor.scimap`, so
    no scimap install is required.
    """
    from spatioev._vendor.scimap import phenotype_cells as _phenotype_cells

    phenotype_df = _read_csv_if_path(phenotype)
    return _phenotype_cells(adata, phenotype=phenotype_df, label=label, **kwargs)


def run_scimap_prior_knowledge_phenotyping(
    adata: ad.AnnData,
    phenotype_workflow: pd.DataFrame | str | Path,
    manual_gates: pd.DataFrame | str | Path | None = None,
    label: str = "phenotype",
    rescale_kwargs: dict[str, Any] | None = None,
    phenotype_kwargs: dict[str, Any] | None = None,
) -> ad.AnnData:
    """Run Scimap's gate-rescale-phenotype prior-knowledge workflow.

    This follows the Scimap prior-knowledge tutorial sequence:
    manual gates or Napari-derived gates, ``sm.pp.rescale``, then
    ``sm.tl.phenotype_cells``.
    """
    adata = scimap_rescale(
        adata,
        gate=manual_gates,
        **(rescale_kwargs or {}),
    )
    return scimap_phenotype_cells(
        adata,
        phenotype=phenotype_workflow,
        label=label,
        **(phenotype_kwargs or {}),
    )


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
# Section 3: Interactive annotation  (from archive/phenotype/annotation.py)
# ============================================================

def _natural_cluster_sort_key(value: object) -> tuple[int, int | str]:
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def annotate_interactive(
    adata: ad.AnnData,
    cluster_key: str = "leiden",
    new_key: str = "annotation",
) -> tuple[ad.AnnData, dict[str, str]]:
    """Prompt for one annotation label per cluster and write it to ``adata.obs``.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData with cluster labels in ``adata.obs[cluster_key]``.
    cluster_key : str
        Column containing cluster IDs to annotate.
    new_key : str
        Target column for the entered annotation labels.

    Returns
    -------
    tuple
        ``(adata, mapping)`` where ``mapping`` maps cluster ID strings to
        entered annotation labels.
    """
    if cluster_key not in adata.obs:
        raise KeyError(f"{cluster_key!r} not found in adata.obs.")

    clusters = sorted(
        adata.obs[cluster_key].astype(str).dropna().unique(),
        key=_natural_cluster_sort_key,
    )
    mapping: dict[str, str] = {}

    existing = None
    if new_key in adata.obs:
        existing = (
            adata.obs[[cluster_key, new_key]]
            .astype(str)
            .dropna()
            .drop_duplicates(subset=[cluster_key])
            .set_index(cluster_key)[new_key]
            .to_dict()
        )

    print("\nEnter annotation for each cluster. Press Enter to keep the current label.\n")

    for cluster in clusters:
        current = existing.get(cluster, cluster) if existing is not None else cluster
        label = input(f"Cluster {cluster} | current={current}: ").strip()
        mapping[cluster] = label if label else current

    adata.obs[new_key] = adata.obs[cluster_key].astype(str).map(mapping)

    return adata, mapping


def annotate_from_csv(
    adata: ad.AnnData,
    csv_file: str,
    cluster_key: str = "leiden",
    new_key: str = "annotation",
) -> ad.AnnData:
    """Attach cluster annotations from a CSV with ``cluster`` and ``annotation``.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData with cluster labels in ``adata.obs[cluster_key]``.
    csv_file : str
        Path to a CSV containing ``cluster`` and ``annotation`` columns.
    cluster_key : str
        Column containing cluster IDs to annotate.
    new_key : str
        Target column for annotation labels.
    """
    if cluster_key not in adata.obs:
        raise KeyError(f"{cluster_key!r} not found in adata.obs.")

    mapping_df = pd.read_csv(csv_file)
    missing = {"cluster", "annotation"} - set(mapping_df.columns)
    if missing:
        raise ValueError(f"Annotation CSV is missing columns: {sorted(missing)}")

    mapping = dict(
        zip(
            mapping_df["cluster"].astype(str),
            mapping_df["annotation"].astype(str),
        )
    )
    adata.obs[new_key] = adata.obs[cluster_key].astype(str).map(mapping)

    return adata


# ============================================================
# Section 4: Annotation merging  (from archive/phenotype/merge.py)
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
# Section 5: Cluster refinement  (from archive/phenotype/refine.py)
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
    "scimap_napari_gater",
    "scimap_rescale",
    "scimap_phenotype_cells",
    "run_scimap_prior_knowledge_phenotyping",
    "subset_cells",
    "annotate_interactive",
    "annotate_from_csv",
    "merge_annotations",
    "merge_refinements",
    "refine_clusters",
]
