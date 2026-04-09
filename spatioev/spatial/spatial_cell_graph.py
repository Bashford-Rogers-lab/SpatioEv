"""
Spatial cell graph construction utilities.

This module builds a Layer 1 cell graph where:
- nodes = cells
- edges = spatial proximity within each image
- node features = selected morphology/intensity features plus optional phenotype encoding

The graph is stored in AnnData-friendly form:
- ``adata.obsm["cell_features"]`` for processed node features
- ``adata.obsp["cell_graph_connectivities"]`` for the weighted or binary adjacency
- ``adata.obsp["cell_graph_distances"]`` for edge distances
- ``adata.uns["cell_graph"]`` for graph metadata

These outputs are designed to support niche-level induced subgraphs for
downstream trajectory or representation analyses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scipy.sparse import coo_matrix, csr_matrix, triu
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def _auto_log_transform(df, feature_cols, skew_threshold=1.0):
    """
    Apply ``log1p`` to non-negative, skewed continuous features.
    """
    df_out = df.copy()
    transformed = []

    for col in feature_cols:
        values = pd.to_numeric(df_out[col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)

        if not finite.any():
            continue

        finite_values = values[finite]
        if np.all(finite_values >= 0):
            skewness = pd.Series(finite_values).skew()
            if pd.notna(skewness) and skewness > skew_threshold:
                df_out.loc[finite, col] = np.log1p(finite_values)
                transformed.append(col)

    return df_out, transformed


def _prepare_node_features(
    adata,
    feature_cols,
    phenotype_key=None,
    auto_log=True,
    skew_threshold=1.0,
    scale_features=True,
    scale_binary_features=False,
    phenotype_weight=1.0,
):
    """
    Prepare a processed node-feature matrix aligned to ``adata.obs``.

    Continuous features are optionally log-transformed and scaled.
    Phenotype dummies are optionally appended and are not scaled by default.
    Rows with non-finite continuous features are retained in the output but
    marked invalid via the returned mask.
    """
    obs = adata.obs.copy()

    missing = [col for col in feature_cols if col not in obs.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    df_cont = obs[feature_cols].apply(pd.to_numeric, errors="coerce")

    transformed_cols = []
    if auto_log:
        df_cont, transformed_cols = _auto_log_transform(
            df_cont,
            feature_cols=feature_cols,
            skew_threshold=skew_threshold,
        )

    valid_cont_mask = np.isfinite(df_cont.to_numpy(dtype=float)).all(axis=1)
    n_obs = adata.n_obs

    cont_array = df_cont.to_numpy(dtype=float)
    if scale_features and len(feature_cols) > 0 and valid_cont_mask.any():
        scaler = StandardScaler()
        cont_scaled = np.full_like(cont_array, np.nan, dtype=float)
        cont_scaled[valid_cont_mask] = scaler.fit_transform(cont_array[valid_cont_mask])
    else:
        cont_scaled = cont_array.astype(float, copy=True)

    matrices = []
    feature_names = []

    if len(feature_cols) > 0:
        matrices.append(cont_scaled)
        feature_names.extend(feature_cols)

    phenotype_cols = []
    if phenotype_key is not None:
        if phenotype_key not in obs.columns:
            raise ValueError(f"{phenotype_key} not found in adata.obs")

        pheno_onehot = pd.get_dummies(obs[phenotype_key], prefix="pheno", dtype=float)
        pheno_array = pheno_onehot.to_numpy(dtype=float)

        if scale_binary_features and pheno_array.shape[1] > 0:
            scaler = StandardScaler()
            pheno_array = scaler.fit_transform(pheno_array)

        pheno_array = pheno_array * float(phenotype_weight)
        matrices.append(pheno_array)
        phenotype_cols = pheno_onehot.columns.tolist()
        feature_names.extend(phenotype_cols)

    if matrices:
        X = np.concatenate(matrices, axis=1)
    else:
        X = np.empty((n_obs, 0), dtype=float)

    # Missing phenotype values are allowed; invalidity is driven by continuous features.
    return X, feature_names, transformed_cols, valid_cont_mask


def _build_radius_graph(coords, radius):
    """
    Build an undirected radius graph from coordinates.

    Returns local edge endpoints and distances with each pair included once.
    """
    if len(coords) < 2:
        empty = np.empty(0, dtype=int)
        return empty, empty, np.empty(0, dtype=float)

    nbrs = NearestNeighbors(radius=radius)
    nbrs.fit(coords)
    graph = nbrs.radius_neighbors_graph(coords, mode="distance")
    graph = triu(graph, k=1).tocoo()

    return graph.row.astype(int), graph.col.astype(int), graph.data.astype(float)


def _resolve_sigma(values, sigma, default=1.0):
    """
    Resolve kernel width from data when not provided explicitly.
    """
    if sigma is not None:
        return float(sigma)

    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    positive = finite[finite > 0]

    if len(positive) == 0:
        return float(default)

    return float(np.median(positive))


def _compute_edge_weights(row, col, dists, features, sigma_space=None, sigma_feat=None):
    """
    Compute edge weights using spatial decay and feature similarity.
    """
    sigma_space = _resolve_sigma(dists, sigma_space, default=1.0)
    spatial_weights = np.exp(-(dists ** 2) / (2.0 * sigma_space ** 2))

    if features.shape[1] == 0:
        return spatial_weights, sigma_space, None

    diffs = features[row] - features[col]
    feat_dists = np.linalg.norm(diffs, axis=1)
    sigma_feat = _resolve_sigma(feat_dists, sigma_feat, default=1.0)
    feat_weights = np.exp(-(feat_dists ** 2) / (2.0 * sigma_feat ** 2))

    return spatial_weights * feat_weights, sigma_space, sigma_feat


def build_cell_graph(
    adata,
    feature_cols,
    phenotype_key=None,
    radius=40.0,
    x_key="X_centroid",
    y_key="Y_centroid",
    image_key="imageid",
    auto_log=True,
    skew_threshold=1.0,
    scale_features=True,
    scale_binary_features=False,
    phenotype_weight=1.0,
    compute_weights=True,
    sigma_space=None,
    sigma_feat=None,
    feature_obsm_key="cell_features",
    adjacency_key="cell_graph_connectivities",
    distance_key="cell_graph_distances",
    graph_obs_key="cell_graph_valid",
):
    """
    Build a Layer 1 spatial cell graph across all images.

    The resulting adjacency is block-diagonal across images because edges are
    only built within each image.
    """
    obs = adata.obs
    required_cols = [x_key, y_key, image_key]
    missing = [col for col in required_cols if col not in obs.columns]
    if missing:
        raise ValueError(f"Missing required columns in adata.obs: {missing}")

    adata_out = adata.copy()

    X, feature_names, transformed_cols, valid_feature_mask = _prepare_node_features(
        adata_out,
        feature_cols=feature_cols,
        phenotype_key=phenotype_key,
        auto_log=auto_log,
        skew_threshold=skew_threshold,
        scale_features=scale_features,
        scale_binary_features=scale_binary_features,
        phenotype_weight=phenotype_weight,
    )

    coords_df = adata_out.obs[[x_key, y_key]].apply(pd.to_numeric, errors="coerce")
    coords = coords_df.to_numpy(dtype=float)
    valid_coord_mask = np.isfinite(coords).all(axis=1)

    if compute_weights:
        graph_valid_mask = valid_coord_mask & valid_feature_mask
    else:
        graph_valid_mask = valid_coord_mask

    adata_out.obsm[feature_obsm_key] = X
    adata_out.obs[graph_obs_key] = graph_valid_mask

    rows = []
    cols = []
    weights = []
    distances = []
    image_graphs = {}

    for img in adata_out.obs[image_key].dropna().unique():
        img_mask = adata_out.obs[image_key] == img
        local_mask = img_mask.to_numpy() & graph_valid_mask
        local_idx = np.flatnonzero(local_mask)

        if len(local_idx) < 2:
            continue

        local_coords = coords[local_idx]
        row_local, col_local, dist_local = _build_radius_graph(local_coords, radius=radius)

        if len(row_local) == 0:
            image_graphs[img] = {
                "cell_ids": adata_out.obs_names[local_idx].tolist(),
                "obs_positions": local_idx.tolist(),
                "n_nodes": int(len(local_idx)),
                "n_edges": 0,
            }
            continue

        row_global = local_idx[row_local]
        col_global = local_idx[col_local]

        if compute_weights:
            weight_local, sigma_space_img, sigma_feat_img = _compute_edge_weights(
                row=row_local,
                col=col_local,
                dists=dist_local,
                features=X[local_idx],
                sigma_space=sigma_space,
                sigma_feat=sigma_feat,
            )
        else:
            weight_local = np.ones(len(dist_local), dtype=float)
            sigma_space_img = None
            sigma_feat_img = None

        rows.extend(row_global.tolist())
        cols.extend(col_global.tolist())
        weights.extend(weight_local.tolist())
        distances.extend(dist_local.tolist())

        image_graphs[img] = {
            "cell_ids": adata_out.obs_names[local_idx].tolist(),
            "obs_positions": local_idx.tolist(),
            "n_nodes": int(len(local_idx)),
            "n_edges": int(len(row_local)),
            "sigma_space": sigma_space_img,
            "sigma_feat": sigma_feat_img,
        }

    n_obs = adata_out.n_obs
    if len(rows) > 0:
        row_array = np.asarray(rows, dtype=int)
        col_array = np.asarray(cols, dtype=int)
        weight_array = np.asarray(weights, dtype=float)
        dist_array = np.asarray(distances, dtype=float)

        adjacency = coo_matrix(
            (
                np.concatenate([weight_array, weight_array]),
                (
                    np.concatenate([row_array, col_array]),
                    np.concatenate([col_array, row_array]),
                ),
            ),
            shape=(n_obs, n_obs),
        ).tocsr()

        distance_matrix = coo_matrix(
            (
                np.concatenate([dist_array, dist_array]),
                (
                    np.concatenate([row_array, col_array]),
                    np.concatenate([col_array, row_array]),
                ),
            ),
            shape=(n_obs, n_obs),
        ).tocsr()
    else:
        adjacency = csr_matrix((n_obs, n_obs), dtype=float)
        distance_matrix = csr_matrix((n_obs, n_obs), dtype=float)

    adata_out.obsp[adjacency_key] = adjacency
    adata_out.obsp[distance_key] = distance_matrix
    adata_out.uns["cell_graph"] = {
        "feature_obsm_key": feature_obsm_key,
        "adjacency_key": adjacency_key,
        "distance_key": distance_key,
        "graph_obs_key": graph_obs_key,
        "feature_names": feature_names,
        "log_transformed": transformed_cols,
        "radius": float(radius),
        "x_key": x_key,
        "y_key": y_key,
        "image_key": image_key,
        "phenotype_key": phenotype_key,
        "compute_weights": bool(compute_weights),
        "scale_features": bool(scale_features),
        "scale_binary_features": bool(scale_binary_features),
        "phenotype_weight": float(phenotype_weight),
        "images": image_graphs,
        "node_ids": adata_out.obs_names.tolist(),
    }

    return adata_out


def extract_niche_subgraph(
    adata,
    niche_key,
    niche_value,
    adjacency_key="cell_graph_connectivities",
    distance_key="cell_graph_distances",
    feature_obsm_key="cell_features",
):
    """
    Extract the induced cell subgraph for one niche label.

    Returns a dictionary containing the niche membership, adjacency, distances,
    and processed node features aligned to the returned cell IDs.
    """
    if niche_key not in adata.obs.columns:
        raise ValueError(f"{niche_key} not found in adata.obs")

    if adjacency_key not in adata.obsp:
        raise ValueError(f"{adjacency_key} not found in adata.obsp")

    mask = (adata.obs[niche_key] == niche_value).to_numpy()
    idx = np.flatnonzero(mask)

    adjacency = adata.obsp[adjacency_key][idx][:, idx].copy()

    result = {
        "niche_key": niche_key,
        "niche_value": niche_value,
        "cell_ids": adata.obs_names[idx].tolist(),
        "obs_positions": idx.tolist(),
        "adjacency": adjacency,
    }

    if distance_key in adata.obsp:
        result["distances"] = adata.obsp[distance_key][idx][:, idx].copy()

    if feature_obsm_key in adata.obsm:
        result["node_features"] = np.asarray(adata.obsm[feature_obsm_key][idx])

    return result


def extract_all_niche_subgraphs(
    adata,
    niche_key,
    adjacency_key="cell_graph_connectivities",
    distance_key="cell_graph_distances",
    feature_obsm_key="cell_features",
    include_values=None,
    include_prefix=None,
    exclude_values=("unassigned", "noise"),
    exclude_prefixes=None,
    min_cells=1,
):
    """
    Extract induced cell subgraphs for multiple niche labels.

    This is a convenience wrapper around ``extract_niche_subgraph`` that reads
    niche labels from ``adata.obs[niche_key]``, filters them, and returns a
    dictionary keyed by niche label.

    Parameters
    ----------
    niche_key : str
        Column in ``adata.obs`` containing niche labels.
    include_values : iterable, optional
        Explicit niche labels to include. If provided, only these labels are
        considered.
    include_prefix : str, optional
        If provided, keep only labels whose string form starts with this prefix.
    exclude_values : iterable, optional
        Labels to exclude exactly. By default excludes ``"unassigned"`` and
        ``"noise"``.
    exclude_prefixes : iterable, optional
        Exclude labels whose string form starts with any of these prefixes.
        Useful for filtering singleton labels.
    min_cells : int
        Minimum number of cells required for a niche to be returned.

    Returns
    -------
    dict
        ``{niche_value: subgraph_dict}``
    """
    if niche_key not in adata.obs.columns:
        raise ValueError(f"{niche_key} not found in adata.obs")

    labels = pd.Series(adata.obs[niche_key]).dropna()
    if labels.empty:
        return {}

    unique_values = labels.unique().tolist()

    if include_values is not None:
        include_set = set(include_values)
        unique_values = [value for value in unique_values if value in include_set]

    if include_prefix is not None:
        unique_values = [
            value for value in unique_values
            if str(value).startswith(include_prefix)
        ]

    if exclude_values is not None:
        exclude_set = set(exclude_values)
        unique_values = [value for value in unique_values if value not in exclude_set]

    if exclude_prefixes is not None:
        exclude_prefixes = tuple(map(str, exclude_prefixes))
        unique_values = [
            value for value in unique_values
            if not str(value).startswith(exclude_prefixes)
        ]

    subgraphs = {}

    for niche_value in unique_values:
        subgraph = extract_niche_subgraph(
            adata,
            niche_key=niche_key,
            niche_value=niche_value,
            adjacency_key=adjacency_key,
            distance_key=distance_key,
            feature_obsm_key=feature_obsm_key,
        )

        if len(subgraph["cell_ids"]) < int(min_cells):
            continue

        subgraphs[niche_value] = subgraph

    return subgraphs
