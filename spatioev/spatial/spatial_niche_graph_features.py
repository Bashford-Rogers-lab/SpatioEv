"""
Niche-level graph descriptor utilities.

This module summarizes each niche as the induced subgraph of the Layer 1
cell graph plus a set of handcrafted descriptors that capture:

- graph topology
- geometry-aware graph structure
- node-feature organization on the graph
- optional boundary/core contrasts

The output is a per-niche feature table suitable for downstream clustering,
trajectory inference, or pseudotime modeling.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import ConvexHull
from scipy.sparse import csr_matrix, triu
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm

from .preprocessing import compute_convex_hull_area


def _safe_scalar(value):
    """
    Convert scalar-like results to a float, returning NaN when invalid.
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan

    if np.isfinite(value):
        return value

    return np.nan


def _safe_cv(values):
    """
    Coefficient of variation with NaN-safe handling.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    mean = values.mean()
    if np.isclose(mean, 0.0):
        return np.nan

    return float(values.std(ddof=0) / mean)


def _safe_ratio(num, denom):
    """
    Safe scalar ratio with NaN on zero or non-finite denominator.
    """
    num = _safe_scalar(num)
    denom = _safe_scalar(denom)

    if not np.isfinite(num) or not np.isfinite(denom) or np.isclose(denom, 0.0):
        return np.nan

    return float(num / denom)


def _nanmean_if_any(values):
    """
    NaN-safe mean that returns NaN when no finite values are present.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    return float(values.mean())


def _normalized_entropy(values, n_bins):
    """
    Compute entropy normalized to [0, 1].
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    counts, _ = np.histogram(values, bins=n_bins)
    total = counts.sum()
    if total == 0:
        return np.nan

    probs = counts[counts > 0] / total
    if len(probs) <= 1:
        return 0.0

    entropy = -np.sum(probs * np.log(probs))
    return float(entropy / np.log(n_bins))


def _sanitize_label(value):
    """
    Convert arbitrary labels into stable column-name fragments.
    """
    text = str(value)
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def _graph_morans_i_from_adjacency(adjacency, values):
    """
    Moran-like autocorrelation on a graph adjacency matrix.
    """
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)

    if adjacency.shape[0] != len(values):
        return np.nan

    if valid.sum() < 3:
        return np.nan

    idx = np.flatnonzero(valid)
    sub_adj = adjacency[idx][:, idx].astype(float).tocsr()
    x = values[idx]

    W = float(sub_adj.sum())
    n = len(x)

    if n < 3 or np.isclose(W, 0.0):
        return np.nan

    x_centered = x - x.mean()
    denom = float(np.sum(x_centered ** 2))
    if np.isclose(denom, 0.0):
        return np.nan

    numerator = float(x_centered @ (sub_adj @ x_centered))
    return float((n / W) * (numerator / denom))


def _build_knn_skeleton(coords, k=6):
    """
    Build a sparse Euclidean skeleton graph from a symmetric kNN graph and
    extract its minimum spanning tree.

    This provides a less trivial topology summary than the original niche graph
    when niches were themselves defined as connected components of a radius
    graph.
    """
    coords = np.asarray(coords, dtype=float)
    n = len(coords)

    if n < 2:
        return csr_matrix((n, n), dtype=float)

    k_eff = min(int(k), n - 1)
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1)
    nbrs.fit(coords)
    knn_graph = nbrs.kneighbors_graph(coords, mode="distance")
    knn_graph = knn_graph.maximum(knn_graph.T).tocsr()

    mst = minimum_spanning_tree(knn_graph)
    mst = mst.maximum(mst.T).tocsr()

    return mst


def _edge_feature_stats(values, edge_rows, edge_cols):
    """
    Edge-based coherence summaries for one node feature.
    """
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)

    keep = valid[edge_rows] & valid[edge_cols]
    if keep.sum() < 2:
        return {
            "neighbor_corr": np.nan,
            "edge_abs_diff_mean": np.nan,
        }

    x = values[edge_rows[keep]]
    y = values[edge_cols[keep]]

    if np.std(x) == 0 or np.std(y) == 0:
        corr = np.nan
    else:
        corr = np.corrcoef(x, y)[0, 1]

    return {
        "neighbor_corr": _safe_scalar(corr),
        "edge_abs_diff_mean": _safe_scalar(np.mean(np.abs(x - y))),
    }


def _quantile_bin(values, n_bins=3):
    """
    Bin a continuous feature by quantiles for assortativity calculations.
    """
    series = pd.Series(values, dtype=float)
    valid = series.notna()

    if valid.sum() < n_bins:
        return None

    try:
        binned = pd.qcut(series[valid], q=n_bins, duplicates="drop")
    except ValueError:
        return None

    if binned.nunique(dropna=True) < 2:
        return None

    out = pd.Series(index=series.index, dtype=object)
    out.loc[valid] = binned.astype(str).values
    return out.to_numpy(dtype=object)


def _spearman_feature(values, reference):
    """
    Spearman correlation with NaN-safe handling.
    """
    values = np.asarray(values, dtype=float)
    reference = np.asarray(reference, dtype=float)
    valid = np.isfinite(values) & np.isfinite(reference)

    if valid.sum() < 3:
        return np.nan

    x = values[valid]
    y = reference[valid]
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return np.nan

    stat = spearmanr(x, y).statistic
    return _safe_scalar(stat)


def _summarize_topology(G, include_path_metrics=True):
    """
    Topological summaries for one induced niche graph.
    """
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    degrees = np.array([deg for _, deg in G.degree()], dtype=float)

    out = {
        "topology__n_nodes": float(n_nodes),
        "topology__n_edges": float(n_edges),
        "topology__density": _safe_scalar(nx.density(G)) if n_nodes > 1 else 0.0,
        "topology__avg_degree": _safe_scalar(degrees.mean()) if len(degrees) > 0 else np.nan,
        "topology__degree_var": _safe_scalar(degrees.var(ddof=0)) if len(degrees) > 0 else np.nan,
        "topology__degree_cv": _safe_cv(degrees),
        "topology__leaf_fraction": _safe_scalar(np.mean(degrees == 1)) if len(degrees) > 0 else np.nan,
        "topology__isolates_fraction": _safe_scalar(np.mean(degrees == 0)) if len(degrees) > 0 else np.nan,
        "topology__avg_clustering": _safe_scalar(nx.average_clustering(G)) if n_nodes > 1 else np.nan,
        "topology__transitivity": _safe_scalar(nx.transitivity(G)) if n_nodes > 2 else np.nan,
        "topology__n_connected_components": float(nx.number_connected_components(G)) if n_nodes > 0 else 0.0,
    }

    if n_nodes > 0:
        component_sizes = np.array([len(c) for c in nx.connected_components(G)], dtype=float)
        out["topology__largest_component_size"] = _safe_scalar(component_sizes.max())
        out["topology__largest_component_fraction"] = _safe_scalar(component_sizes.max() / n_nodes)
        out["topology__component_size_cv"] = _safe_cv(component_sizes)
    else:
        out["topology__largest_component_size"] = np.nan
        out["topology__largest_component_fraction"] = np.nan
        out["topology__component_size_cv"] = np.nan

    if n_nodes < 3 or n_edges == 0 or len(np.unique(degrees)) < 2:
        out["topology__degree_assortativity"] = np.nan
    else:
        try:
            out["topology__degree_assortativity"] = _safe_scalar(
                nx.degree_assortativity_coefficient(G)
            )
        except Exception:
            out["topology__degree_assortativity"] = np.nan

    if n_nodes > 0 and n_edges > 0:
        core_numbers = np.array(list(nx.core_number(G).values()), dtype=float)
        out["topology__mean_core_number"] = _safe_scalar(core_numbers.mean())
        out["topology__max_core_number"] = _safe_scalar(core_numbers.max())
        if include_path_metrics:
            out["topology__bridge_fraction"] = _safe_scalar(
                len(list(nx.bridges(G))) / n_edges
            )
        else:
            out["topology__bridge_fraction"] = np.nan
    else:
        out["topology__mean_core_number"] = np.nan
        out["topology__max_core_number"] = np.nan
        out["topology__bridge_fraction"] = np.nan

    if n_nodes == 0 or not include_path_metrics:
        out["topology__diameter_lcc"] = np.nan
        out["topology__avg_shortest_path_lcc"] = np.nan
        return out

    largest_nodes = max(nx.connected_components(G), key=len)
    G_lcc = G.subgraph(largest_nodes).copy()

    if G_lcc.number_of_nodes() <= 1:
        out["topology__diameter_lcc"] = 0.0
        out["topology__avg_shortest_path_lcc"] = 0.0
    else:
        out["topology__diameter_lcc"] = _safe_scalar(nx.diameter(G_lcc))
        out["topology__avg_shortest_path_lcc"] = _safe_scalar(
            nx.average_shortest_path_length(G_lcc)
        )

    return out


def _summarize_skeleton_topology(coords):
    """
    Pathology-oriented topology summaries from a sparse Euclidean skeleton.

    In PDAC this tends to separate compact gland-forming niches from elongated,
    budding, or infiltrative epithelial structures better than dense radius
    graphs alone.
    """
    coords = np.asarray(coords, dtype=float)
    n_nodes = len(coords)

    out = {
        "topology__skeleton_n_edges": np.nan,
        "topology__skeleton_leaf_fraction": np.nan,
        "topology__skeleton_branchpoint_fraction": np.nan,
        "topology__skeleton_avg_degree": np.nan,
        "topology__skeleton_degree_cv": np.nan,
        "topology__skeleton_total_length": np.nan,
        "topology__skeleton_mean_edge_length": np.nan,
        "topology__skeleton_edge_length_cv": np.nan,
        "topology__skeleton_diameter": np.nan,
        "topology__skeleton_avg_shortest_path": np.nan,
        "topology__skeleton_tortuosity": np.nan,
    }

    if n_nodes < 2:
        return out

    skeleton = _build_knn_skeleton(coords, k=6)
    if skeleton.nnz == 0:
        return out

    G_skel = nx.from_scipy_sparse_array(skeleton)
    degrees = np.array([deg for _, deg in G_skel.degree()], dtype=float)
    edge_lengths = skeleton.data.astype(float)

    out["topology__skeleton_n_edges"] = float(G_skel.number_of_edges())
    out["topology__skeleton_leaf_fraction"] = _safe_scalar(np.mean(degrees == 1))
    out["topology__skeleton_branchpoint_fraction"] = _safe_scalar(np.mean(degrees >= 3))
    out["topology__skeleton_avg_degree"] = _safe_scalar(degrees.mean())
    out["topology__skeleton_degree_cv"] = _safe_cv(degrees)
    out["topology__skeleton_total_length"] = _safe_scalar(edge_lengths.sum())
    out["topology__skeleton_mean_edge_length"] = _safe_scalar(edge_lengths.mean())
    out["topology__skeleton_edge_length_cv"] = _safe_cv(edge_lengths)

    if G_skel.number_of_nodes() <= 1:
        out["topology__skeleton_diameter"] = 0.0
        out["topology__skeleton_avg_shortest_path"] = 0.0
        out["topology__skeleton_tortuosity"] = np.nan
        return out

    largest_nodes = max(nx.connected_components(G_skel), key=len)
    G_lcc = G_skel.subgraph(largest_nodes).copy()

    out["topology__skeleton_diameter"] = _safe_scalar(nx.diameter(G_lcc))
    out["topology__skeleton_avg_shortest_path"] = _safe_scalar(
        nx.average_shortest_path_length(G_lcc)
    )

    if len(coords) >= 2:
        centered = coords - coords.mean(axis=0)
        cov = np.cov(coords.T)
        eigvecs = np.linalg.eigh(cov)[1]
        major_axis = eigvecs[:, -1]
        major_span = np.ptp(centered @ major_axis)
        out["topology__skeleton_tortuosity"] = _safe_ratio(edge_lengths.sum(), major_span)

    return out


def _summarize_geometry(coords, edge_rows, edge_cols, edge_lengths):
    """
    Geometry-aware summaries for one niche subgraph.
    """
    coords = np.asarray(coords, dtype=float)
    out = {}

    out["geometry__mean_edge_length"] = _safe_scalar(np.mean(edge_lengths)) if len(edge_lengths) > 0 else np.nan
    out["geometry__edge_length_std"] = _safe_scalar(np.std(edge_lengths, ddof=0)) if len(edge_lengths) > 0 else np.nan
    out["geometry__edge_length_cv"] = _safe_cv(edge_lengths)
    out["geometry__median_edge_length"] = _safe_scalar(np.median(edge_lengths)) if len(edge_lengths) > 0 else np.nan

    if len(edge_lengths) > 0:
        vecs = coords[edge_cols] - coords[edge_rows]
        angles = np.mod(np.arctan2(vecs[:, 1], vecs[:, 0]), np.pi)
        out["geometry__orientation_entropy"] = _normalized_entropy(angles, n_bins=8)

        doubled = 2.0 * angles
        cos_mean = np.mean(np.cos(doubled))
        sin_mean = np.mean(np.sin(doubled))
        out["geometry__orientation_coherence"] = _safe_scalar(
            np.sqrt(cos_mean ** 2 + sin_mean ** 2)
        )
    else:
        out["geometry__orientation_entropy"] = np.nan
        out["geometry__orientation_coherence"] = np.nan

    if len(coords) >= 2:
        centroid = coords.mean(axis=0)
        radial = np.linalg.norm(coords - centroid, axis=1)
        out["geometry__mean_radial_distance"] = _safe_scalar(radial.mean())
        out["geometry__radial_distance_cv"] = _safe_cv(radial)

        cov = np.cov(coords.T)
        eigvals = np.sort(np.real(np.linalg.eigvalsh(cov)))[::-1]
        major = eigvals[0]
        minor = eigvals[1] if len(eigvals) > 1 else 0.0

        denom = major + minor
        out["geometry__node_cloud_anisotropy"] = (
            _safe_scalar((major - minor) / denom) if not np.isclose(denom, 0.0) else np.nan
        )
        out["geometry__node_cloud_elongation"] = (
            _safe_scalar(major / minor) if minor > 0 else np.nan
        )
        proj = coords - centroid
        eigvecs = np.linalg.eigh(cov)[1]
        major_axis = eigvecs[:, -1]
        minor_axis = eigvecs[:, 0]
        major_proj = proj @ major_axis
        minor_proj = proj @ minor_axis
        major_span = np.ptp(major_proj)
        minor_span = np.ptp(minor_proj)
        out["geometry__major_axis_span"] = _safe_scalar(major_span)
        out["geometry__minor_axis_span"] = _safe_scalar(minor_span)
        out["geometry__span_ratio"] = _safe_ratio(major_span, minor_span)

        width = max(2, int(np.ceil(np.sqrt(len(coords)))))
        x_bins = np.linspace(coords[:, 0].min(), coords[:, 0].max(), width + 1)
        y_bins = np.linspace(coords[:, 1].min(), coords[:, 1].max(), width + 1)
        hist, _, _ = np.histogram2d(coords[:, 0], coords[:, 1], bins=[x_bins, y_bins])
        probs = hist.ravel()
        probs = probs[probs > 0] / probs.sum()
        if len(probs) <= 1:
            out["geometry__spatial_entropy"] = 0.0
        else:
            out["geometry__spatial_entropy"] = _safe_scalar(
                -np.sum(probs * np.log(probs)) / np.log(len(hist.ravel()))
            )

        if len(coords) >= 3:
            nn = NearestNeighbors(n_neighbors=2)
            nn.fit(coords)
            dists, _ = nn.kneighbors(coords)
            nn_dists = dists[:, 1]
            out["geometry__mean_nearest_neighbor_distance"] = _safe_scalar(nn_dists.mean())
            out["geometry__nearest_neighbor_distance_cv"] = _safe_cv(nn_dists)
        else:
            out["geometry__mean_nearest_neighbor_distance"] = np.nan
            out["geometry__nearest_neighbor_distance_cv"] = np.nan

        hull_area = compute_convex_hull_area(coords)
        out["geometry__convex_hull_area"] = _safe_scalar(hull_area)
        if len(coords) >= 3:
            try:
                hull = ConvexHull(coords)
                hull_perimeter = float(hull.area)
            except Exception:
                hull_perimeter = np.nan
        else:
            hull_perimeter = np.nan
        out["geometry__convex_hull_perimeter"] = _safe_scalar(hull_perimeter)
        if np.isfinite(hull_area) and hull_area > 0 and np.isfinite(hull_perimeter) and hull_perimeter > 0:
            out["geometry__hull_circularity"] = _safe_scalar(
                4.0 * np.pi * hull_area / (hull_perimeter ** 2)
            )
        else:
            out["geometry__hull_circularity"] = np.nan
        out["geometry__cell_density_hull"] = (
            _safe_scalar(len(coords) / hull_area) if np.isfinite(hull_area) and hull_area > 0 else np.nan
        )
        out["geometry__edge_density_hull"] = (
            _safe_scalar(len(edge_lengths) / hull_area) if np.isfinite(hull_area) and hull_area > 0 else np.nan
        )
        out["geometry__cells_per_major_axis_span"] = _safe_ratio(len(coords), major_span)
    else:
        out["geometry__mean_radial_distance"] = np.nan
        out["geometry__radial_distance_cv"] = np.nan
        out["geometry__node_cloud_anisotropy"] = np.nan
        out["geometry__node_cloud_elongation"] = np.nan
        out["geometry__major_axis_span"] = np.nan
        out["geometry__minor_axis_span"] = np.nan
        out["geometry__span_ratio"] = np.nan
        out["geometry__spatial_entropy"] = np.nan
        out["geometry__mean_nearest_neighbor_distance"] = np.nan
        out["geometry__nearest_neighbor_distance_cv"] = np.nan
        out["geometry__convex_hull_area"] = np.nan
        out["geometry__convex_hull_perimeter"] = np.nan
        out["geometry__hull_circularity"] = np.nan
        out["geometry__cell_density_hull"] = np.nan
        out["geometry__edge_density_hull"] = np.nan
        out["geometry__cells_per_major_axis_span"] = np.nan

    return out


def _summarize_feature_organization(
    obs_sub,
    adjacency_binary,
    edge_rows,
    edge_cols,
    feature_cols=None,
    phenotype_key=None,
    morphology_bin_count=3,
):
    """
    Node-feature organization descriptors within one niche graph.
    """
    out = {}
    G = nx.from_scipy_sparse_array(adjacency_binary)

    if phenotype_key is not None and phenotype_key in obs_sub.columns:
        phenotype_values = obs_sub[phenotype_key].astype("object").to_numpy()
        valid = pd.notna(phenotype_values)

        if valid.sum() >= 2 and pd.Series(phenotype_values[valid]).nunique() >= 2:
            G_pheno = G.subgraph(np.flatnonzero(valid)).copy()
            mapping = {old: i for i, old in enumerate(G_pheno.nodes())}
            G_pheno = nx.relabel_nodes(G_pheno, mapping)
            pheno_valid = phenotype_values[valid]
            for i, value in enumerate(pheno_valid):
                G_pheno.nodes[i]["phenotype"] = value
            if G_pheno.number_of_edges() == 0:
                out["features__phenotype_assortativity"] = np.nan
            else:
                try:
                    out["features__phenotype_assortativity"] = _safe_scalar(
                        nx.attribute_assortativity_coefficient(G_pheno, "phenotype")
                    )
                except Exception:
                    out["features__phenotype_assortativity"] = np.nan
        else:
            out["features__phenotype_assortativity"] = np.nan

    if feature_cols is None:
        return out

    degree = np.asarray(adjacency_binary.sum(axis=1)).ravel().astype(float)

    for feature in feature_cols:
        if feature not in obs_sub.columns:
            continue

        values = pd.to_numeric(obs_sub[feature], errors="coerce").to_numpy(dtype=float)

        out[f"features__{feature}__graph_morans_i"] = _graph_morans_i_from_adjacency(
            adjacency_binary,
            values,
        )

        edge_stats = _edge_feature_stats(values, edge_rows, edge_cols)
        out[f"features__{feature}__neighbor_corr"] = edge_stats["neighbor_corr"]
        out[f"features__{feature}__edge_abs_diff_mean"] = edge_stats["edge_abs_diff_mean"]
        out[f"features__{feature}__degree_spearman"] = _spearman_feature(values, degree)

        binned = _quantile_bin(values, n_bins=morphology_bin_count)
        if binned is not None:
            valid = pd.notna(binned)
            G_bin = G.subgraph(np.flatnonzero(valid)).copy()
            mapping = {old: i for i, old in enumerate(G_bin.nodes())}
            G_bin = nx.relabel_nodes(G_bin, mapping)
            binned_valid = binned[valid]
            for i, value in enumerate(binned_valid):
                G_bin.nodes[i]["feature_bin"] = value
            if G_bin.number_of_edges() == 0 or pd.Series(binned_valid).nunique() < 2:
                out[f"features__{feature}__bin_assortativity"] = np.nan
            else:
                try:
                    out[f"features__{feature}__bin_assortativity"] = _safe_scalar(
                        nx.attribute_assortativity_coefficient(G_bin, "feature_bin")
                    )
                except Exception:
                    out[f"features__{feature}__bin_assortativity"] = np.nan
        else:
            out[f"features__{feature}__bin_assortativity"] = np.nan

    return out


def _summarize_boundary_core(
    obs_sub,
    adjacency_binary,
    feature_cols=None,
    phenotype_key=None,
    region_key=None,
    boundary_labels=("inner_border",),
    core_labels=("core",),
):
    """
    Boundary/core summaries for one niche graph.
    """
    out = {}

    if region_key is None or region_key not in obs_sub.columns:
        return out

    region_values = obs_sub[region_key].astype("object")
    boundary_mask = region_values.isin(boundary_labels).to_numpy()
    core_mask = region_values.isin(core_labels).to_numpy()

    out["boundary__boundary_fraction"] = _safe_scalar(boundary_mask.mean())

    degree = np.asarray(adjacency_binary.sum(axis=1)).ravel().astype(float)
    if boundary_mask.any():
        out["boundary__mean_degree_boundary"] = _safe_scalar(np.nanmean(degree[boundary_mask]))
    else:
        out["boundary__mean_degree_boundary"] = np.nan

    if core_mask.any():
        out["boundary__mean_degree_core"] = _safe_scalar(np.nanmean(degree[core_mask]))
    else:
        out["boundary__mean_degree_core"] = np.nan

    if boundary_mask.any() and core_mask.any():
        out["boundary__degree_boundary_minus_core"] = _safe_scalar(
            np.nanmean(degree[boundary_mask]) - np.nanmean(degree[core_mask])
        )
    else:
        out["boundary__degree_boundary_minus_core"] = np.nan

    if feature_cols is not None:
        for feature in feature_cols:
            if feature not in obs_sub.columns:
                continue

            values = pd.to_numeric(obs_sub[feature], errors="coerce").to_numpy(dtype=float)
            if boundary_mask.any() and core_mask.any():
                out[f"boundary__{feature}__boundary_minus_core"] = _safe_scalar(
                    np.nanmean(values[boundary_mask]) - np.nanmean(values[core_mask])
                )
            else:
                out[f"boundary__{feature}__boundary_minus_core"] = np.nan

    if phenotype_key is not None and phenotype_key in obs_sub.columns:
        phenotypes = obs_sub[phenotype_key].astype("object")
        if boundary_mask.any():
            out["boundary__phenotype_entropy_boundary"] = _safe_scalar(
                phenotypes[boundary_mask].value_counts(normalize=True, dropna=True).pipe(
                    lambda probs: -np.sum(probs * np.log(probs)) if len(probs) > 1 else 0.0
                )
            )
        else:
            out["boundary__phenotype_entropy_boundary"] = np.nan

        if core_mask.any():
            out["boundary__phenotype_entropy_core"] = _safe_scalar(
                phenotypes[core_mask].value_counts(normalize=True, dropna=True).pipe(
                    lambda probs: -np.sum(probs * np.log(probs)) if len(probs) > 1 else 0.0
                )
            )
        else:
            out["boundary__phenotype_entropy_core"] = np.nan

    return out


def _get_n_hop_external_layers(adjacency_full, niche_idx, max_hops=1):
    """
    Collect external nodes around a niche in successive graph hops.

    Hop 1 contains external neighbors directly adjacent to niche cells.
    Hop 2 contains previously unseen external neighbors adjacent to hop-1 nodes,
    and so on. Niche nodes themselves are never returned.
    """
    n_total = adjacency_full.shape[0]
    niche_idx = np.asarray(niche_idx, dtype=int)

    if len(niche_idx) == 0 or max_hops < 1:
        return {}

    niche_set = set(niche_idx.tolist())
    visited = set(niche_idx.tolist())
    frontier = set(niche_idx.tolist())
    layers = {}

    for hop in range(1, int(max_hops) + 1):
        next_frontier = set()
        for node in frontier:
            neighbors = adjacency_full.getrow(node).indices
            for neighbor in neighbors:
                if neighbor in niche_set or neighbor in visited:
                    continue
                next_frontier.add(int(neighbor))

        if not next_frontier:
            break

        layers[hop] = np.array(sorted(next_frontier), dtype=int)
        visited.update(next_frontier)
        frontier = next_frontier

    return layers


def _summarize_graph_surroundings(
    adata,
    niche_idx,
    adjacency_full,
    niche_key,
    niche_value,
    feature_cols=None,
    phenotype_key=None,
    surround_hops=1,
    numeric_cache=None,
    phenotype_array=None,
):
    """
    Fast graph-defined niche boundary/core/surround summaries.

    Boundary cells are niche cells with at least one edge to a non-niche cell.
    Core cells are niche cells with no direct external neighbors.
    Surrounding cells are non-niche cells reached within ``surround_hops``.
    """
    niche_idx = np.asarray(niche_idx, dtype=int)
    out = {}

    if len(niche_idx) == 0:
        return out

    niche_set = set(niche_idx.tolist())
    niche_pos = {node: i for i, node in enumerate(niche_idx.tolist())}
    niche_adj = adjacency_full[niche_idx][:, niche_idx].tocsr()
    niche_degree = np.asarray(niche_adj.sum(axis=1)).ravel().astype(float)

    boundary_local = []
    external_degree = np.zeros(len(niche_idx), dtype=float)
    cross_edges = []

    for local_i, global_i in enumerate(niche_idx):
        neighbors = adjacency_full.getrow(global_i).indices
        external_neighbors = [j for j in neighbors if j not in niche_set]
        external_degree[local_i] = len(external_neighbors)

        if external_neighbors:
            boundary_local.append(local_i)
            for j in external_neighbors:
                cross_edges.append((int(global_i), int(j)))

    boundary_local = np.array(boundary_local, dtype=int)
    core_local = np.array(
        sorted(set(range(len(niche_idx))) - set(boundary_local.tolist())),
        dtype=int,
    )

    boundary_idx = niche_idx[boundary_local] if len(boundary_local) > 0 else np.empty(0, dtype=int)
    core_idx = niche_idx[core_local] if len(core_local) > 0 else np.empty(0, dtype=int)
    surround_layers = _get_n_hop_external_layers(
        adjacency_full=adjacency_full,
        niche_idx=niche_idx,
        max_hops=surround_hops,
    )

    surround_idx = (
        np.concatenate(list(surround_layers.values()))
        if len(surround_layers) > 0
        else np.empty(0, dtype=int)
    )

    out["graph_boundary__n_boundary_cells"] = float(len(boundary_idx))
    out["graph_boundary__boundary_fraction"] = _safe_scalar(len(boundary_idx) / len(niche_idx))
    out["graph_boundary__n_core_cells"] = float(len(core_idx))
    out["graph_boundary__core_fraction"] = _safe_scalar(len(core_idx) / len(niche_idx))
    out["graph_boundary__mean_external_degree"] = _safe_scalar(external_degree.mean())
    out["graph_boundary__max_external_degree"] = _safe_scalar(external_degree.max()) if len(external_degree) > 0 else np.nan
    out["graph_boundary__boundary_external_degree_mean"] = (
        _safe_scalar(external_degree[boundary_local].mean()) if len(boundary_local) > 0 else np.nan
    )
    out["graph_surround__n_total"] = float(len(surround_idx))
    out["graph_surround__surround_to_niche_ratio"] = _safe_scalar(len(surround_idx) / len(niche_idx))
    out["graph_surround__n_cross_edges"] = float(len(cross_edges))
    out["graph_surround__cross_edges_per_niche_cell"] = _safe_scalar(len(cross_edges) / len(niche_idx))

    for hop in range(1, int(surround_hops) + 1):
        hop_idx = surround_layers.get(hop, np.empty(0, dtype=int))
        out[f"graph_surround__hop_{hop}__n_cells"] = float(len(hop_idx))
        out[f"graph_surround__hop_{hop}__fraction_of_niche"] = _safe_scalar(
            len(hop_idx) / len(niche_idx)
        )

    if phenotype_key is not None and phenotype_array is not None:
        phenotype_series = pd.Series(phenotype_array)

        if len(cross_edges) > 0:
            same = []
            for src, dst in cross_edges:
                p_src = phenotype_series.iloc[src]
                p_dst = phenotype_series.iloc[dst]
                same.append(pd.notna(p_src) and pd.notna(p_dst) and p_src == p_dst)
            out["graph_surround__cross_edge_same_phenotype_fraction"] = _safe_scalar(np.mean(same))
        else:
            out["graph_surround__cross_edge_same_phenotype_fraction"] = np.nan

        if len(surround_idx) > 0:
            surround_pheno = phenotype_series.iloc[surround_idx]
            probs = surround_pheno.value_counts(normalize=True, dropna=True)
            out["graph_surround__phenotype_entropy"] = _safe_scalar(
                -np.sum(probs * np.log(probs)) if len(probs) > 1 else 0.0
            )
        else:
            out["graph_surround__phenotype_entropy"] = np.nan

        for hop in range(1, int(surround_hops) + 1):
            hop_idx = surround_layers.get(hop, np.empty(0, dtype=int))
            if len(hop_idx) > 0:
                hop_probs = phenotype_series.iloc[hop_idx].value_counts(normalize=True, dropna=True)
                out[f"graph_surround__hop_{hop}__phenotype_entropy"] = _safe_scalar(
                    -np.sum(hop_probs * np.log(hop_probs)) if len(hop_probs) > 1 else 0.0
                )
            else:
                out[f"graph_surround__hop_{hop}__phenotype_entropy"] = np.nan

    if feature_cols is not None:
        for feature in feature_cols:
            if numeric_cache is None or feature not in numeric_cache:
                continue

            values = numeric_cache[feature]
            niche_values = values[niche_idx]

            if len(boundary_idx) > 0 and len(core_idx) > 0:
                out[f"graph_boundary__{feature}__boundary_minus_core"] = _safe_scalar(
                    _nanmean_if_any(values[boundary_idx]) - _nanmean_if_any(values[core_idx])
                )
            else:
                out[f"graph_boundary__{feature}__boundary_minus_core"] = np.nan

            if len(surround_idx) > 0:
                out[f"graph_surround__{feature}__surround_minus_niche"] = _safe_scalar(
                    _nanmean_if_any(values[surround_idx]) - _nanmean_if_any(niche_values)
                )
            else:
                out[f"graph_surround__{feature}__surround_minus_niche"] = np.nan

            for hop in range(1, int(surround_hops) + 1):
                hop_idx = surround_layers.get(hop, np.empty(0, dtype=int))
                if len(hop_idx) > 0:
                    out[f"graph_surround__hop_{hop}__{feature}__minus_niche"] = _safe_scalar(
                        _nanmean_if_any(values[hop_idx]) - _nanmean_if_any(niche_values)
                    )
                else:
                    out[f"graph_surround__hop_{hop}__{feature}__minus_niche"] = np.nan

    return out


def _summarize_niche_state(
    obs_sub,
    state_feature_cols=None,
    phenotype_key=None,
    state_summary_stats=("mean", "median", "std", "iqr", "p10", "p90"),
):
    """
    Summarize original cell-level features at the niche level.

    This keeps more of the single-cell information than a pure mean-only
    aggregation by using richer distribution summaries.
    """
    out = {}

    if state_feature_cols is not None:
        for feature in state_feature_cols:
            if feature not in obs_sub.columns:
                continue

            values = pd.to_numeric(obs_sub[feature], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]

            if len(values) == 0:
                for stat in state_summary_stats:
                    out[f"state__{feature}__{stat}"] = np.nan
                continue

            if "mean" in state_summary_stats:
                out[f"state__{feature}__mean"] = _safe_scalar(np.mean(values))
            if "median" in state_summary_stats:
                out[f"state__{feature}__median"] = _safe_scalar(np.median(values))
            if "std" in state_summary_stats:
                out[f"state__{feature}__std"] = _safe_scalar(np.std(values, ddof=0))
            if "iqr" in state_summary_stats:
                q75, q25 = np.percentile(values, [75, 25])
                out[f"state__{feature}__iqr"] = _safe_scalar(q75 - q25)
            if "p10" in state_summary_stats:
                out[f"state__{feature}__p10"] = _safe_scalar(np.percentile(values, 10))
            if "p90" in state_summary_stats:
                out[f"state__{feature}__p90"] = _safe_scalar(np.percentile(values, 90))

    if phenotype_key is not None and phenotype_key in obs_sub.columns:
        phenotypes = obs_sub[phenotype_key].astype("object")
        probs = phenotypes.value_counts(normalize=True, dropna=True)

        out["state__phenotype_entropy"] = _safe_scalar(
            -np.sum(probs * np.log(probs)) if len(probs) > 1 else 0.0
        )
        out["state__phenotype_dominant_fraction"] = _safe_scalar(
            probs.max() if len(probs) > 0 else np.nan
        )

        for label, proportion in probs.items():
            safe_label = _sanitize_label(label)
            out[f"state__phenotype__{safe_label}__proportion"] = _safe_scalar(proportion)

    return out


def summarize_niche_graph_features(
    adata,
    niche_key,
    feature_cols=None,
    state_feature_cols=None,
    include_values=None,
    phenotype_key=None,
    region_key=None,
    image_key="imageid",
    x_key="X_centroid",
    y_key="Y_centroid",
    adjacency_key="cell_graph_connectivities",
    distance_key="cell_graph_distances",
    boundary_labels=("inner_border",),
    core_labels=("core",),
    exclude_labels=("unassigned", "noise"),
    morphology_bin_count=3,
    min_cells=3,
    include_graph_surroundings=True,
    surround_hops=1,
    include_state_summaries=True,
    state_summary_stats=("mean", "median", "std", "iqr", "p10", "p90"),
    lightweight=False,
    show_progress=False,
    progress_desc="Niche features",
):
    """
    Summarize each niche as a graph-derived descriptor vector.

    Parameters
    ----------
    niche_key : str
        Column in ``adata.obs`` containing niche labels.
    feature_cols : list, optional
        Continuous node features for feature-organization descriptors.
    state_feature_cols : list, optional
        Continuous cell features to summarize as niche-level state descriptors.
        If ``None``, defaults to ``feature_cols``.
    include_values : iterable, optional
        If provided, only summarize these niche labels.
    phenotype_key : str, optional
        Categorical phenotype label for assortativity and entropy summaries.
    region_key : str, optional
        Optional region label column such as ``core`` / ``inner_border``.
    adjacency_key : str
        Key in ``adata.obsp`` containing the Layer 1 cell graph adjacency.
    distance_key : str
        Key in ``adata.obsp`` containing edge distances.
    boundary_labels, core_labels : tuple
        Region labels used for boundary/core contrasts.
    exclude_labels : tuple
        Niche labels to skip.
    min_cells : int
        Minimum number of cells required to summarize one niche.
    include_graph_surroundings : bool
        If ``True``, add fast graph-defined boundary/core/surround features that
        do not require precomputed geometric region labels.
    surround_hops : int
        Number of external graph hops used to define niche surroundings.
    include_state_summaries : bool
        If ``True``, append niche-level summaries of the original cell features.
    state_summary_stats : tuple
        Summary statistics used for the niche state block.
    lightweight : bool
        If ``True``, skip the most expensive topology metrics such as bridge
        fraction, diameter, and average shortest path length.
    show_progress : bool
        If ``True``, display a progress bar over niche groups.
    progress_desc : str
        Description shown in the progress bar.

    Returns
    -------
    DataFrame
        One row per niche with graph-derived descriptors.
    """
    if niche_key not in adata.obs.columns:
        raise ValueError(f"{niche_key} not found in adata.obs")

    if adjacency_key not in adata.obsp:
        raise ValueError(f"{adjacency_key} not found in adata.obsp")

    if x_key not in adata.obs.columns or y_key not in adata.obs.columns:
        raise ValueError(f"{x_key} and/or {y_key} not found in adata.obs")

    feature_cols = list(feature_cols) if feature_cols is not None else None
    if state_feature_cols is None:
        state_feature_cols = feature_cols
    else:
        state_feature_cols = list(state_feature_cols)
    adjacency = adata.obsp[adjacency_key].tocsr()
    distance_matrix = adata.obsp[distance_key] if distance_key in adata.obsp else None
    include_set = set(include_values) if include_values is not None else None
    numeric_cache = {}
    if feature_cols is not None:
        for feature in feature_cols:
            if feature in adata.obs.columns:
                numeric_cache[feature] = pd.to_numeric(
                    adata.obs[feature],
                    errors="coerce",
                ).to_numpy(dtype=float)
    phenotype_array = None
    if phenotype_key is not None and phenotype_key in adata.obs.columns:
        phenotype_array = adata.obs[phenotype_key].astype("object").to_numpy()

    rows = []
    group_cols = [image_key, niche_key] if image_key in adata.obs.columns else [niche_key]
    grouped = adata.obs.groupby(group_cols, dropna=True, observed=False)
    grouped_iter = grouped
    if show_progress:
        grouped_iter = tqdm(grouped, total=grouped.ngroups, desc=progress_desc)

    for group_key, obs_sub in grouped_iter:
        if len(group_cols) == 2:
            image_value, niche_value = group_key
        else:
            image_value = np.nan
            niche_value = group_key

        if include_set is not None and niche_value not in include_set:
            continue

        if pd.isna(niche_value) or niche_value in exclude_labels:
            continue

        idx = adata.obs_names.get_indexer(obs_sub.index)
        idx = idx[idx >= 0]

        if len(idx) < min_cells:
            continue

        obs_sub = adata.obs.iloc[idx].copy()
        coords = obs_sub[[x_key, y_key]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        valid_coords = np.isfinite(coords).all(axis=1)
        if valid_coords.sum() < min_cells:
            continue

        idx = idx[valid_coords]
        obs_sub = obs_sub.iloc[np.flatnonzero(valid_coords)].copy()
        coords = coords[valid_coords]

        adjacency_sub = adjacency[idx][:, idx].astype(float).tocsr()
        adjacency_binary = adjacency_sub.copy()
        adjacency_binary.data = np.ones_like(adjacency_binary.data, dtype=float)

        if adjacency_binary.nnz == 0:
            edge_rows = np.empty(0, dtype=int)
            edge_cols = np.empty(0, dtype=int)
            edge_lengths = np.empty(0, dtype=float)
        else:
            upper = triu(adjacency_binary, k=1).tocoo()
            edge_rows = upper.row.astype(int)
            edge_cols = upper.col.astype(int)

            if distance_matrix is not None:
                dist_sub = distance_matrix[idx][:, idx].tocsr()
                edge_lengths = np.asarray(dist_sub[edge_rows, edge_cols]).ravel().astype(float)
            else:
                edge_lengths = np.linalg.norm(coords[edge_cols] - coords[edge_rows], axis=1)

        G = nx.from_scipy_sparse_array(adjacency_binary)

        row = {
            niche_key: niche_value,
            "image_id": image_value,
            "n_cells": float(len(obs_sub)),
        }

        row.update(_summarize_topology(G, include_path_metrics=not lightweight))
        row.update(_summarize_skeleton_topology(coords))
        row.update(_summarize_geometry(coords, edge_rows, edge_cols, edge_lengths))
        row.update(
            _summarize_feature_organization(
                obs_sub=obs_sub,
                adjacency_binary=adjacency_binary,
                edge_rows=edge_rows,
                edge_cols=edge_cols,
                feature_cols=feature_cols,
                phenotype_key=phenotype_key,
                morphology_bin_count=morphology_bin_count,
            )
        )
        row.update(
            _summarize_boundary_core(
                obs_sub=obs_sub,
                adjacency_binary=adjacency_binary,
                feature_cols=feature_cols,
                phenotype_key=phenotype_key,
                region_key=region_key,
                boundary_labels=boundary_labels,
                core_labels=core_labels,
            )
        )
        if include_graph_surroundings:
            row.update(
                _summarize_graph_surroundings(
                    adata=adata,
                    niche_idx=idx,
                    adjacency_full=adjacency,
                    niche_key=niche_key,
                    niche_value=niche_value,
                    feature_cols=feature_cols,
                    phenotype_key=phenotype_key,
                    surround_hops=surround_hops,
                    numeric_cache=numeric_cache,
                    phenotype_array=phenotype_array,
                )
            )
        if include_state_summaries:
            row.update(
                _summarize_niche_state(
                    obs_sub=obs_sub,
                    state_feature_cols=state_feature_cols,
                    phenotype_key=phenotype_key,
                    state_summary_stats=state_summary_stats,
                )
            )

        rows.append(row)

    return pd.DataFrame(rows)


def build_niche_feature_table(
    adata,
    niche_key,
    feature_cols=None,
    state_feature_cols=None,
    phenotype_key=None,
    region_key=None,
    image_key="imageid",
    x_key="X_centroid",
    y_key="Y_centroid",
    adjacency_key="cell_graph_connectivities",
    distance_key="cell_graph_distances",
    include_values=None,
    include_prefix=None,
    exclude_values=("unassigned", "noise"),
    exclude_prefixes=None,
    boundary_labels=("inner_border",),
    core_labels=("core",),
    morphology_bin_count=3,
    min_cells=3,
    include_graph_surroundings=True,
    surround_hops=1,
    include_state_summaries=True,
    state_summary_stats=("mean", "median", "std", "iqr", "p10", "p90"),
    lightweight=False,
    show_progress=False,
    progress_desc="Niche features",
):
    """
    Build a filtered per-niche graph-descriptor table in one call.

    This is a notebook-friendly wrapper around
    ``summarize_niche_graph_features`` that first filters niche labels from
    ``adata.obs[niche_key]`` and then computes graph-derived descriptors only
    for the retained niches.

    Parameters
    ----------
    include_values : iterable, optional
        Explicit niche labels to include.
    include_prefix : str, optional
        Keep only niche labels whose string form starts with this prefix.
    exclude_values : iterable, optional
        Exact niche labels to exclude.
    exclude_prefixes : iterable, optional
        Exclude niche labels whose string form starts with any of these prefixes.

    Returns
    -------
    DataFrame
        One row per retained niche with graph-derived descriptors.
    """
    if niche_key not in adata.obs.columns:
        raise ValueError(f"{niche_key} not found in adata.obs")

    niche_series = pd.Series(adata.obs[niche_key]).dropna()
    if niche_series.empty:
        return pd.DataFrame()

    niche_values = niche_series.unique().tolist()

    if include_values is not None:
        include_set = set(include_values)
        niche_values = [value for value in niche_values if value in include_set]

    if include_prefix is not None:
        niche_values = [
            value for value in niche_values
            if str(value).startswith(include_prefix)
        ]

    if exclude_values is not None:
        exclude_set = set(exclude_values)
        niche_values = [value for value in niche_values if value not in exclude_set]

    if exclude_prefixes is not None:
        exclude_prefixes = tuple(map(str, exclude_prefixes))
        niche_values = [
            value for value in niche_values
            if not str(value).startswith(exclude_prefixes)
        ]

    if len(niche_values) == 0:
        return pd.DataFrame()

    summary_df = summarize_niche_graph_features(
        adata=adata,
        niche_key=niche_key,
        feature_cols=feature_cols,
        state_feature_cols=state_feature_cols,
        include_values=niche_values,
        phenotype_key=phenotype_key,
        region_key=region_key,
        image_key=image_key,
        x_key=x_key,
        y_key=y_key,
        adjacency_key=adjacency_key,
        distance_key=distance_key,
        boundary_labels=boundary_labels,
        core_labels=core_labels,
        exclude_labels=(),
        morphology_bin_count=morphology_bin_count,
        min_cells=min_cells,
        include_graph_surroundings=include_graph_surroundings,
        surround_hops=surround_hops,
        include_state_summaries=include_state_summaries,
        state_summary_stats=state_summary_stats,
        lightweight=lightweight,
        show_progress=show_progress,
        progress_desc=progress_desc,
    )

    if summary_df.empty:
        return summary_df

    return summary_df[summary_df[niche_key].isin(niche_values)].reset_index(drop=True)


def build_niche_feature_table_batched(
    adata,
    niche_key,
    batch_size=100,
    include_values=None,
    show_progress=False,
    progress_desc="Niche batches",
    **kwargs,
):
    """
    Build the niche feature table in batches of niche labels.

    This is a safer option for large whole-slide datasets than trying to process
    every niche in one notebook call.
    """
    if niche_key not in adata.obs.columns:
        raise ValueError(f"{niche_key} not found in adata.obs")

    if include_values is None:
        niche_values = pd.Series(adata.obs[niche_key]).dropna().unique().tolist()
    else:
        niche_values = list(include_values)

    include_prefix = kwargs.get("include_prefix", None)
    exclude_values = kwargs.get("exclude_values", ("unassigned", "noise"))
    exclude_prefixes = kwargs.get("exclude_prefixes", None)

    if include_prefix is not None:
        niche_values = [
            value for value in niche_values
            if str(value).startswith(include_prefix)
        ]

    if exclude_values is not None:
        exclude_set = set(exclude_values)
        niche_values = [value for value in niche_values if value not in exclude_set]

    if exclude_prefixes is not None:
        exclude_prefixes = tuple(map(str, exclude_prefixes))
        niche_values = [
            value for value in niche_values
            if not str(value).startswith(exclude_prefixes)
        ]

    if len(niche_values) == 0:
        return pd.DataFrame()

    inner_show_progress = kwargs.pop("show_progress_inner", False)
    inner_progress_desc = kwargs.pop("progress_desc_inner", "Niches in batch")
    batch_frames = []
    batch_starts = range(0, len(niche_values), int(batch_size))
    if show_progress:
        total_batches = int(np.ceil(len(niche_values) / int(batch_size)))
        batch_starts = tqdm(batch_starts, total=total_batches, desc=progress_desc)

    for start in batch_starts:
        batch_values = niche_values[start:start + int(batch_size)]
        batch_df = build_niche_feature_table(
            adata=adata,
            niche_key=niche_key,
            include_values=batch_values,
            show_progress=inner_show_progress,
            progress_desc=inner_progress_desc,
            **kwargs,
        )
        if not batch_df.empty:
            batch_frames.append(batch_df)

    if len(batch_frames) == 0:
        return pd.DataFrame()

    return pd.concat(batch_frames, ignore_index=True)


def summarize_niche_surrounding_context(
    adata,
    niche_key,
    phenotype_key,
    feature_cols=None,
    phenotype_feature_map=None,
    include_values=None,
    image_key="imageid",
    adjacency_key="cell_graph_connectivities",
    surround_hops=1,
    phenotype_labels=None,
    min_cells=3,
    summary_stats=("mean", "median"),
    show_progress=False,
    progress_desc="Niche surroundings",
):
    """
    Summarize the graph-defined surroundings of each niche.

    This is a notebook-friendly wrapper around the same n-hop surround logic
    used by ``summarize_niche_graph_features``, but exposes phenotype
    composition and phenotype-restricted feature summaries in a reusable table.

    Parameters
    ----------
    adata : AnnData
        Annotated table containing cell graph and metadata.
    niche_key : str
        Column in ``adata.obs`` containing niche labels.
    phenotype_key : str
        Column in ``adata.obs`` containing phenotype labels for surrounding-cell
        composition summaries.
    feature_cols : list, optional
        Numeric features in ``adata.obs`` to summarize across all surrounding
        cells.
    phenotype_feature_map : dict, optional
        Mapping ``{phenotype_label: [feature_cols...]}`` describing phenotype-
        restricted feature summaries to compute in the surround.
    include_values : iterable, optional
        If provided, restrict output to these niche labels.
    image_key : str
        Optional image/FOV identifier used to keep niches image-local.
    adjacency_key : str
        Key in ``adata.obsp`` holding the cell graph adjacency.
    surround_hops : int
        Number of graph hops used to define the surrounding context.
    phenotype_labels : list, optional
        Explicit phenotype labels to report. If ``None``, all observed labels
        are used.
    min_cells : int
        Minimum niche size required for summarization.
    summary_stats : tuple
        Subset of ``("mean", "median", "std")`` to compute.

    Returns
    -------
    DataFrame
        One row per niche with surrounding-context summaries.
    """
    if niche_key not in adata.obs.columns:
        raise ValueError(f"{niche_key} not found in adata.obs")
    if phenotype_key not in adata.obs.columns:
        raise ValueError(f"{phenotype_key} not found in adata.obs")
    if adjacency_key not in adata.obsp:
        raise ValueError(f"{adjacency_key} not found in adata.obsp")

    feature_cols = list(feature_cols) if feature_cols is not None else []
    phenotype_feature_map = phenotype_feature_map or {}
    include_set = set(include_values) if include_values is not None else None

    missing_features = [col for col in feature_cols if col not in adata.obs.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns in adata.obs: {missing_features}")

    missing_map_features = []
    for cols in phenotype_feature_map.values():
        for col in cols:
            if col not in adata.obs.columns:
                missing_map_features.append(col)
    if missing_map_features:
        missing_map_features = sorted(set(missing_map_features))
        raise ValueError(f"Missing phenotype_feature_map columns in adata.obs: {missing_map_features}")

    if phenotype_labels is None:
        phenotype_labels = pd.Series(adata.obs[phenotype_key]).dropna().unique().tolist()
    else:
        phenotype_labels = list(phenotype_labels)

    numeric_cache = {}
    for feature in sorted(set(feature_cols).union(*[set(v) for v in phenotype_feature_map.values()])):
        numeric_cache[feature] = pd.to_numeric(
            adata.obs[feature],
            errors="coerce",
        ).to_numpy(dtype=float)

    adjacency = adata.obsp[adjacency_key].tocsr()
    group_cols = [image_key, niche_key] if image_key in adata.obs.columns else [niche_key]
    grouped = adata.obs.groupby(group_cols, dropna=True, observed=False)
    grouped_iter = grouped
    if show_progress:
        grouped_iter = tqdm(grouped, total=grouped.ngroups, desc=progress_desc)

    rows = []
    for group_key, obs_sub in grouped_iter:
        if len(group_cols) == 2:
            image_value, niche_value = group_key
        else:
            image_value = np.nan
            niche_value = group_key

        if include_set is not None and niche_value not in include_set:
            continue
        if pd.isna(niche_value) or niche_value in ("unassigned", "noise"):
            continue

        idx = adata.obs_names.get_indexer(obs_sub.index)
        idx = idx[idx >= 0]
        if len(idx) < int(min_cells):
            continue

        surround_layers = _get_n_hop_external_layers(
            adjacency_full=adjacency,
            niche_idx=idx,
            max_hops=surround_hops,
        )
        surround_idx = (
            np.concatenate(list(surround_layers.values()))
            if len(surround_layers) > 0
            else np.empty(0, dtype=int)
        )

        row = {
            niche_key: niche_value,
            "image_id": image_value,
            "n_cells": float(len(idx)),
            "n_surround": float(len(surround_idx)),
        }
        for hop in range(1, int(surround_hops) + 1):
            hop_idx = surround_layers.get(hop, np.empty(0, dtype=int))
            row[f"surround__hop_{hop}__n_cells"] = float(len(hop_idx))

        if len(surround_idx) == 0:
            for label in phenotype_labels:
                safe_label = _sanitize_label(label)
                row[f"surround_prop__{safe_label}"] = 0.0
                row[f"surround__{safe_label}__n_cells"] = 0.0
            for feature in feature_cols:
                for stat in summary_stats:
                    row[f"surround__{feature}__{stat}"] = np.nan
            for label, cols in phenotype_feature_map.items():
                safe_label = _sanitize_label(label)
                for feature in cols:
                    row[f"surround__{safe_label}__{feature}__n_cells"] = 0.0
                    for stat in summary_stats:
                        row[f"surround__{safe_label}__{feature}__{stat}"] = np.nan
            rows.append(row)
            continue

        surround_obs = adata.obs.iloc[surround_idx]
        surround_pheno = surround_obs[phenotype_key].astype("object")
        pheno_probs = surround_pheno.value_counts(normalize=True, dropna=True)

        for label in phenotype_labels:
            safe_label = _sanitize_label(label)
            label_mask = surround_pheno == label
            row[f"surround_prop__{safe_label}"] = _safe_scalar(pheno_probs.get(label, 0.0))
            row[f"surround__{safe_label}__n_cells"] = float(label_mask.sum())

        for feature in feature_cols:
            values = numeric_cache[feature][surround_idx]
            finite = values[np.isfinite(values)]
            if len(finite) == 0:
                for stat in summary_stats:
                    row[f"surround__{feature}__{stat}"] = np.nan
                continue
            if "mean" in summary_stats:
                row[f"surround__{feature}__mean"] = _safe_scalar(np.mean(finite))
            if "median" in summary_stats:
                row[f"surround__{feature}__median"] = _safe_scalar(np.median(finite))
            if "std" in summary_stats:
                row[f"surround__{feature}__std"] = _safe_scalar(np.std(finite, ddof=0))

        for label, cols in phenotype_feature_map.items():
            safe_label = _sanitize_label(label)
            label_idx = surround_idx[surround_pheno.to_numpy(dtype=object) == label]
            for feature in cols:
                row[f"surround__{safe_label}__{feature}__n_cells"] = float(len(label_idx))
                values = numeric_cache[feature][label_idx] if len(label_idx) > 0 else np.array([], dtype=float)
                finite = values[np.isfinite(values)]
                if len(finite) == 0:
                    for stat in summary_stats:
                        row[f"surround__{safe_label}__{feature}__{stat}"] = np.nan
                    continue
                if "mean" in summary_stats:
                    row[f"surround__{safe_label}__{feature}__mean"] = _safe_scalar(np.mean(finite))
                if "median" in summary_stats:
                    row[f"surround__{safe_label}__{feature}__median"] = _safe_scalar(np.median(finite))
                if "std" in summary_stats:
                    row[f"surround__{safe_label}__{feature}__std"] = _safe_scalar(np.std(finite, ddof=0))

        rows.append(row)

    return pd.DataFrame(rows)


def score_pdac_niche_pathology_modules(
    feature_df,
    niche_key=None,
    image_key="image_id",
    polarity_high_is_organized=True,
    min_features_per_module=2,
):
    """
    Score PDAC pathology-inspired niche modules from a niche feature table.

    The function is intentionally tolerant of partially available inputs: each
    module uses whatever relevant columns are present in ``feature_df`` and
    averages signed per-column z-scores across the resolved features.

    Returns a compact per-niche table containing:
    - base pathology scores, including a separate proliferation module
    - condensed trajectory-oriented scores/axes
    - per-module feature counts for transparency
    """
    if not isinstance(feature_df, pd.DataFrame):
        raise TypeError("feature_df must be a pandas DataFrame")

    df = feature_df.copy()
    out = pd.DataFrame(index=df.index)

    if niche_key is not None and niche_key in df.columns:
        out[niche_key] = df[niche_key]
    if image_key is not None and image_key in df.columns:
        out[image_key] = df[image_key]

    def _first_existing(candidates):
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def _zscore(col):
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        out_arr = np.full(len(values), np.nan, dtype=float)
        if finite.sum() < 2:
            return out_arr
        mean = values[finite].mean()
        std = values[finite].std(ddof=0)
        if np.isclose(std, 0.0):
            return out_arr
        out_arr[finite] = (values[finite] - mean) / std
        return out_arr

    polarity_sign_pos = 1.0 if polarity_high_is_organized else -1.0
    polarity_sign_neg = -1.0 * polarity_sign_pos

    module_specs = {
        "pdac_duct_organization_score": [
            (1.0, ["geometry__hull_circularity"]),
            (1.0, ["topology__largest_component_fraction"]),
            (1.0, ["geometry__mean_nearest_neighbor_distance", "geometry__median_edge_length", "topology__skeleton_mean_edge_length"]),
            (1.0, ["state__CK19_expr_z__mean", "state__CK19_expr__mean", "state__CK19_expr__median"]),
            (1.0, ["state__NaKATPase_expr_z__mean", "state__NaKATPase_expr__mean", "state__NaKATPase_expr__median"]),
            (polarity_sign_pos, ["state__polarity_score__median", "state__polarity_score__mean"]),
            (-1.0, ["graph_boundary__boundary_fraction"]),
            (-1.0, ["graph_boundary__mean_external_degree", "graph_boundary__boundary_external_degree_mean"]),
            (-1.0, ["topology__bridge_fraction"]),
            (-1.0, ["topology__skeleton_branchpoint_fraction"]),
            (-1.0, ["geometry__spatial_entropy"]),
        ],
        "pdac_dysplasia_score": [
            (1.0, ["state__nc_ratio_z__mean", "state__nc_ratio__mean", "state__nc_ratio__median", "state__nc_ratio__p90"]),
            (1.0, ["state__boundary_irregularity_z__mean", "state__boundary_irregularity__mean"]),
            (1.0, ["state__major_minor_axis_ratio_z__mean", "state__major_minor_axis_ratio__mean"]),
            (1.0, ["state__centroid_dif_z__mean", "state__centroid_dif__mean"]),
            (1.0, ["state__num_concavities_z__mean", "state__num_concavities__mean"]),
            (polarity_sign_neg, ["state__polarity_score__median", "state__polarity_score__mean"]),
            (1.0, ["features__nc_ratio__graph_morans_i"]),
        ],
        "pdac_architectural_complexity_score": [
            (1.0, ["topology__degree_var"]),
            (1.0, ["topology__avg_clustering"]),
            (1.0, ["topology__skeleton_branchpoint_fraction"]),
            (1.0, ["topology__skeleton_leaf_fraction"]),
            (1.0, ["geometry__orientation_entropy"]),
            (1.0, ["geometry__edge_length_cv"]),
            (1.0, ["state__lacunarity_z__mean", "state__lacunarity__mean"]),
            (1.0, ["state__boundary_irregularity__mean"]),
            (-1.0, ["geometry__hull_circularity"]),
        ],
        "pdac_invasion_desmoplasia_score": [
            (1.0, ["graph_boundary__boundary_fraction"]),
            (1.0, ["graph_boundary__mean_external_degree", "graph_boundary__boundary_external_degree_mean"]),
            (1.0, ["graph_surround__cross_edges_per_niche_cell", "graph_surround__n_cross_edges"]),
            (1.0, ["graph_surround__phenotype_entropy", "graph_surround__hop_1__phenotype_entropy"]),
            (1.0, ["graph_surround__hop_1__fraction_of_niche", "graph_surround__surround_to_niche_ratio"]),
            (1.0, ["surround_prop__Fibroblasts", "surround__Fibroblasts__fraction"]),
            (1.0, ["surround__Fibroblasts__FAP_expr_z__mean", "surround__Fibroblasts__FAP_expr__mean", "fibro_surround__FAP_expr_z__mean", "fibro_surround__FAP_expr__mean"]),
            (1.0, ["surround__Fibroblasts__aSMA_expr_z__mean", "surround__Fibroblasts__aSMA_expr__mean", "fibro_surround__aSMA_expr_z__mean", "fibro_surround__aSMA_expr__mean"]),
            (-1.0, ["geometry__hull_circularity"]),
        ],
        "pdac_proliferation_score": [
            (1.0, ["state__Ki67_expr_z__mean", "state__Ki67_expr__mean", "state__Ki67_expr__median"]),
            (1.0, ["state__Ki67_expr_z__p90", "state__Ki67_expr__p90"]),
        ],
        "pdac_dedifferentiation_score": [
            (1.0, ["state__entropy_z__mean", "state__entropy__mean"]),
            (1.0, ["state__inertia_z__mean", "state__inertia__mean"]),
            (1.0, ["state__haralick_contrast__mean"]),
            (1.0, ["geometry__cell_density_hull", "geometry__edge_density_hull"]),
            (1.0, ["topology__skeleton_branchpoint_fraction"]),
            (-1.0, ["geometry__mean_nearest_neighbor_distance", "topology__skeleton_mean_edge_length"]),
            (-1.0, ["geometry__hull_circularity"]),
            (polarity_sign_neg, ["state__polarity_score__median", "state__polarity_score__mean"]),
        ],
    }

    resolved_map = {}
    z_cache = {}

    for module_name, spec in module_specs.items():
        signed_parts = []
        resolved_features = []
        for sign, candidates in spec:
            resolved = _first_existing(candidates)
            if resolved is None:
                continue
            if resolved not in z_cache:
                z_cache[resolved] = _zscore(resolved)
            signed_parts.append(sign * z_cache[resolved])
            resolved_features.append((resolved, float(sign)))

        resolved_map[module_name] = resolved_features
        out[f"{module_name}__n_features"] = float(len(resolved_features))

        if len(signed_parts) < int(min_features_per_module):
            out[module_name] = np.nan
            continue

        stacked = np.vstack(signed_parts)
        out[module_name] = np.nanmean(stacked, axis=0)

    def _module_z(col):
        values = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        result = np.full(len(values), np.nan, dtype=float)
        if finite.sum() < 2:
            return result
        mean = values[finite].mean()
        std = values[finite].std(ddof=0)
        if np.isclose(std, 0.0):
            return result
        result[finite] = (values[finite] - mean) / std
        return result

    base_cols = [
        "pdac_duct_organization_score",
        "pdac_dysplasia_score",
        "pdac_architectural_complexity_score",
        "pdac_invasion_desmoplasia_score",
        "pdac_proliferation_score",
        "pdac_dedifferentiation_score",
    ]
    base_z = {col: _module_z(col) for col in base_cols}

    condensed_specs = {
        "pdac_early_duct_anchor_score": [
            base_z["pdac_duct_organization_score"],
            -base_z["pdac_dysplasia_score"],
            -base_z["pdac_invasion_desmoplasia_score"],
            -base_z["pdac_proliferation_score"],
            -base_z["pdac_dedifferentiation_score"],
        ],
        "pdac_panin_like_dysplasia_score": [
            base_z["pdac_dysplasia_score"],
            base_z["pdac_architectural_complexity_score"],
            -base_z["pdac_duct_organization_score"],
        ],
        "pdac_invasive_gland_forming_score": [
            base_z["pdac_duct_organization_score"],
            base_z["pdac_invasion_desmoplasia_score"],
            base_z["pdac_proliferation_score"],
            -base_z["pdac_dedifferentiation_score"],
        ],
        "pdac_invasion_desmoplasia_axis": [
            base_z["pdac_invasion_desmoplasia_score"],
        ],
        "pdac_proliferation_axis": [
            base_z["pdac_proliferation_score"],
        ],
        "pdac_dedifferentiation_axis": [
            base_z["pdac_dedifferentiation_score"],
            -base_z["pdac_duct_organization_score"],
        ],
    }

    for module_name, arrays in condensed_specs.items():
        stacked = np.vstack(arrays)
        finite_counts = np.isfinite(stacked).sum(axis=0)
        out[f"{module_name}__n_features"] = finite_counts.astype(float)
        out[module_name] = np.nanmean(stacked, axis=0)
        out.loc[finite_counts == 0, module_name] = np.nan

    out.attrs["resolved_module_features"] = resolved_map
    return out
