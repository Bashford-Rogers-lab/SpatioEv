"""
Spatial statistics module for SpatioEv.

This module implements a comprehensive set of spatial point-process
and spatial autocorrelation statistics for analyzing multiplexed
imaging and spatial omics datasets.

The methods are designed to quantify spatial organization of cells,
cell–cell interactions, and spatial coupling between biological
features within tissue sections.

The module supports three major classes of spatial analysis:

1. Point-pattern statistics (Ripley family)
   • Global clustering or dispersion of cells
   • Phenotype-specific clustering
   • Cross-phenotype interaction analysis
   • Spatial interaction curves across distance scales
   • Monte Carlo envelope testing for statistical significance
   • Permutation-based null models preserving tissue architecture

2. Spatial autocorrelation statistics (Moran family)
   • Global spatial autocorrelation of continuous features
   • Detection of spatial clustering or dispersion of features
   • Local hotspot detection at the single-cell level

3. Cross-feature spatial association
   • Spatial coupling between two biological features
   • Detection of regions where features co-localize or exclude each other

Key features of this implementation:

• Convex-hull tissue boundary estimation to avoid bias from empty regions  
• Monte Carlo envelope testing for Ripley statistics  
• Permutation-based interaction tests preserving tissue architecture  
• Multi-scale spatial analysis across distance radii  
• Efficient neighbor queries using BallTree and K-nearest-neighbor graphs

These methods enable quantitative investigation of spatial
organization in tumor microenvironments, immune infiltration
patterns, stromal remodeling, and cellular ecosystem structure.
"""

import numpy as np
import pandas as pd

from sklearn.neighbors import BallTree, kneighbors_graph
from .preprocessing import compute_convex_hull_area
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point


def _random_points_in_hull(coords, n_points):
    """
    Generate random points inside the convex hull of coordinates.

    Used for Monte Carlo envelope testing.
    """

    hull = ConvexHull(coords)

    polygon = Polygon(coords[hull.vertices])

    minx, miny, maxx, maxy = polygon.bounds

    points = []

    while len(points) < n_points:

        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)

        if polygon.contains(Point(x, y)):
            points.append((x, y))

    return np.array(points)


def _permute_values_within_groups(values, groups, rng):
    """
    Permute values independently within each group.
    """
    values = np.asarray(values)
    groups = np.asarray(groups)

    permuted = values.copy()

    for group in pd.unique(groups):
        idx = np.where(groups == group)[0]
        permuted[idx] = rng.permutation(values[idx])

    return permuted


def _clean_coords(coords):
    """
    Drop rows with non-finite x/y coordinates.
    """
    coords = np.asarray(coords, dtype=float)

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("Coordinates must be an Nx2 array.")

    return coords[np.isfinite(coords).all(axis=1)]


def _clean_paired_coords(source_coords, target_coords):
    """
    Drop non-finite rows from two coordinate arrays independently.
    """
    return _clean_coords(source_coords), _clean_coords(target_coords)


def _resolve_k(n_obs, k):
    """
    Ensure kNN graph construction uses a valid number of neighbors.
    """
    if n_obs < 2:
        return None

    return min(k, n_obs - 1)


# ============================================================
# 1. GLOBAL RIPLEY'S K
# ============================================================

def ripleys_k(coords, radius):
    """
    Compute Ripley's K statistic and derived transforms.

    Returns
    -------
    dict
        K_observed
        K_expected
        L
        L_minus_r
    """

    coords = _clean_coords(coords)

    n = coords.shape[0]

    if n < 2:
        return {
            "K_observed": np.nan,
            "K_expected": np.nan,
            "L": np.nan,
            "L_minus_r": np.nan,
        }

    # tissue area
    area = compute_convex_hull_area(coords)

    if np.isnan(area):
        return {
            "K_observed": np.nan,
            "K_expected": np.nan,
            "L": np.nan,
            "L_minus_r": np.nan,
        }

    # neighbor search
    tree = BallTree(coords)

    neighbors = tree.query_radius(coords, r=radius)

    counts = np.array([len(nbrs) - 1 for nbrs in neighbors], dtype=float)

    # observed K
    K_obs = (area / (n * (n - 1))) * counts.sum()

    # CSR expectation
    K_exp = np.pi * radius**2

    # L transform
    L = np.sqrt(K_obs / np.pi)

    return {
        "K_observed": K_obs,
        "K_expected": K_exp,
        "L": L,
        "L_minus_r": L - radius,
    }


def ripleys_curve(coords, radii):
    """
    Compute Ripley statistics across radii.

    Returns
    -------
    DataFrame
        radius
        K
        L
        L_minus_r
    """

    coords = _clean_coords(coords)

    n = len(coords)

    if n < 2:
        return pd.DataFrame()

    area = compute_convex_hull_area(coords)

    if np.isnan(area):
        return pd.DataFrame()

    tree = BallTree(coords)

    K_vals = []

    for r in radii:

        neighbors = tree.query_radius(coords, r=r)

        counts = np.array([len(n)-1 for n in neighbors])

        K_r = (area/(n*(n-1))) * counts.sum()

        K_vals.append(K_r)

    K_vals = np.array(K_vals)

    L_vals = np.sqrt(K_vals/np.pi)

    return pd.DataFrame({
        "radius": radii,
        "K": K_vals,
        "L": L_vals,
        "L_minus_r": L_vals - radii
    })


def ripley_envelope(coords, radii, n_sim=99):
    """
    Monte Carlo envelope test for Ripley statistics.

    Randomizes point locations inside the convex hull.
    """

    coords = _clean_coords(coords)

    observed = ripleys_curve(coords, radii)

    if observed.empty:
        return observed

    sims = []

    for i in range(n_sim):

        sim_coords = _random_points_in_hull(coords, len(coords))

        sim_curve = ripleys_curve(sim_coords, radii)

        sims.append(sim_curve["L_minus_r"].values)

    sims = np.array(sims)

    observed["envelope_low"] = np.percentile(sims, 2.5, axis=0)
    observed["envelope_high"] = np.percentile(sims, 97.5, axis=0)

    return observed


# ============================================================
# 2. PHENOTYPE-AGNOSTIC RIPLEY'S K
# ============================================================

def ripleys_k_by_image(
    adata,
    radius,
    x_key="X_centroid",
    y_key="Y_centroid",
    image_key="imageid",
):
    """
    Compute Ripley's K statistics for each image.

    Parameters
    ----------
    x_key, y_key : str
        Column names in ``adata.obs`` containing spatial coordinates.
    image_key : str
        Column in ``adata.obs`` identifying which image each cell belongs to.
    """

    rows = []

    for img in adata.obs[image_key].unique():

        coords = adata.obs.loc[
            adata.obs[image_key] == img,
            [x_key, y_key],
        ].to_numpy()

        stats = ripleys_k(coords, radius)

        rows.append({
            image_key: img,
            "radius": radius,
            **stats
        })

    out = pd.DataFrame(rows)

    if "spatial_stats" not in adata.uns:
        adata.uns["spatial_stats"] = {}

    adata.uns["spatial_stats"]["ripley_k"] = out.copy()

    return out

def ripleys_k_by_phenotype(
    adata,
    phenotype_key,
    radius,
    x_key="X_centroid",
    y_key="Y_centroid",
    image_key="imageid",
):
    """
    Compute Ripley's K separately for each phenotype.

    Biological question
    -------------------
    Are cells of a given phenotype spatially clustered?

    Parameters
    ----------
    phenotype_key : str
        Column in ``adata.obs`` containing phenotype labels.
    x_key, y_key : str
        Column names in ``adata.obs`` containing spatial coordinates.
    image_key : str
        Column in ``adata.obs`` identifying which image each cell belongs to.
    """

    rows = []

    for img in adata.obs[image_key].unique():

        df = adata.obs[adata.obs[image_key] == img]

        phenotypes = df[phenotype_key].dropna().unique()

        for pheno in phenotypes:

            subset = df[df[phenotype_key] == pheno]

            coords = subset[[x_key, y_key]].to_numpy()

            if len(coords) < 3:

                stats = {
                    "K_observed": np.nan,
                    "K_expected": np.nan,
                    "L": np.nan,
                    "L_minus_r": np.nan,
                }

            else:

                stats = ripleys_k(coords, radius)

            rows.append({
                image_key: img,
                "phenotype": pheno,
                "radius": radius,
                **stats
            })

    out = pd.DataFrame(rows)

    if "spatial_stats" not in adata.uns:
        adata.uns["spatial_stats"] = {}

    adata.uns["spatial_stats"]["ripley_k_by_phenotype"] = out.copy()

    return out


# ============================================================
# 3. CROSS-PHENOTYPE RIPLEY'S K (INTERACTION)
# ============================================================
def cross_ripleys_k(
    source_coords,
    target_coords,
    radius,
):
    """
    Cross Ripley's K statistic.

    Measures spatial interaction between two point sets.
    """

    source_coords, target_coords = _clean_paired_coords(
        source_coords,
        target_coords,
    )

    if len(source_coords) == 0 or len(target_coords) == 0:
        return np.nan

    # remove NaN coordinates
    source_coords = source_coords[np.isfinite(source_coords).all(axis=1)]
    target_coords = target_coords[np.isfinite(target_coords).all(axis=1)]

    if len(source_coords) == 0 or len(target_coords) == 0:
        return np.nan

    area = compute_convex_hull_area(
        np.vstack([source_coords, target_coords])
    )

    tree = BallTree(target_coords)

    neighbors = tree.query_radius(source_coords, r=radius)

    counts = np.array([len(n) for n in neighbors], dtype=float)

    n_source = len(source_coords)
    n_target = len(target_coords)

    K = (area / (n_source * n_target)) * counts.sum()

    return K


def cross_ripleys_k_by_phenotype(
    adata,
    phenotype_key,
    source_phenotype,
    target_phenotype,
    radius,
    x_key="X_centroid",
    y_key="Y_centroid",
    image_key="imageid",
):
    """
    Compute cross Ripley's K between two phenotypes.

    Biological interpretation
    -------------------------
    Tests whether cells of one phenotype spatially cluster
    around another phenotype.

    Examples
    --------
    CD8 → tumor interaction
    macrophage → fibroblast interaction

    Parameters
    ----------
    phenotype_key : str
        Column in ``adata.obs`` containing phenotype labels.
    source_phenotype, target_phenotype : str
        The two phenotype labels to compare.
    radius : float
        Spatial radius at which cross-K is evaluated.
    x_key, y_key : str
        Column names in ``adata.obs`` containing spatial coordinates.
    image_key : str
        Column in ``adata.obs`` identifying which image each cell belongs to.

    Returns
    -------
    DataFrame
        cross Ripley's K per image
    """

    rows = []

    for img in adata.obs[image_key].unique():

        df = adata.obs[adata.obs[image_key] == img]

        source = df[
            df[phenotype_key] == source_phenotype
        ]

        target = df[
            df[phenotype_key] == target_phenotype
        ]

        source_coords = source[[x_key, y_key]].to_numpy()
        target_coords = target[[x_key, y_key]].to_numpy()

        K = cross_ripleys_k(
            source_coords,
            target_coords,
            radius,
        )

        rows.append({
            image_key: img,
            "source": source_phenotype,
            "target": target_phenotype,
            "cross_ripley_k": K,
        })

    return pd.DataFrame(rows)


def cross_ripleys_k_all_pairs(
    adata,
    phenotype_key,
    radius,
    x_key="X_centroid",
    y_key="Y_centroid",
    image_key="imageid",
):

    phenotypes = adata.obs[phenotype_key].dropna().unique()

    rows = []

    for p1 in phenotypes:
        for p2 in phenotypes:

            res = cross_ripleys_k_by_phenotype(
                adata,
                phenotype_key,
                p1,
                p2,
                radius,
                x_key,
                y_key,
                image_key,
            )

            rows.append(res)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)

# ============================================================
# 4. Cross Ripley curves and significance testing
# ============================================================

def cross_ripleys_curve(
    source_coords,
    target_coords,
    radii,
):
    """
    Compute cross Ripley curve K(r), L(r), and L(r)-r.

    Parameters
    ----------
    source_coords : Nx2 array
    target_coords : Mx2 array
    radii : iterable of radii

    Returns
    -------
    DataFrame
        radius
        K
        L
        L_minus_r
    """

    source_coords, target_coords = _clean_paired_coords(
        source_coords,
        target_coords,
    )

    if len(source_coords) == 0 or len(target_coords) == 0:
        return pd.DataFrame()

    area = compute_convex_hull_area(
        np.vstack([source_coords, target_coords])
    )

    if np.isnan(area):
        return pd.DataFrame()

    tree = BallTree(target_coords)

    n_source = len(source_coords)
    n_target = len(target_coords)

    K_vals = []

    for r in radii:

        neighbors = tree.query_radius(source_coords, r=r)

        counts = np.array([len(n) for n in neighbors])

        K_r = (area / (n_source * n_target)) * counts.sum()

        K_vals.append(K_r)

    K_vals = np.array(K_vals)

    L_vals = np.sqrt(K_vals / np.pi)

    return pd.DataFrame({
        "radius": radii,
        "K": K_vals,
        "L": L_vals,
        "L_minus_r": L_vals - radii
    })


def cross_ripley_envelope(source_coords, target_coords, radii, n_sim=99):
    """
    Monte Carlo envelope for cross Ripley analysis.

    Tests whether two phenotypes interact more than expected.
    """

    source_coords, target_coords = _clean_paired_coords(
        source_coords,
        target_coords,
    )

    observed = cross_ripleys_curve(source_coords, target_coords, radii)

    if observed.empty:
        return observed

    sims = []

    for i in range(n_sim):

        rand_source = _random_points_in_hull(source_coords, len(source_coords))

        sim_curve = cross_ripleys_curve(rand_source, target_coords, radii)

        sims.append(sim_curve["L_minus_r"].values)

    sims = np.array(sims)

    observed["envelope_low"] = np.percentile(sims, 2.5, axis=0)
    observed["envelope_high"] = np.percentile(sims, 97.5, axis=0)

    return observed


def cross_ripleys_curve_by_phenotype(
    adata,
    phenotype_key,
    source_phenotype,
    target_phenotype,
    radii,
    x_key="X_centroid",
    y_key="Y_centroid",
    image_key="imageid",
):
    """
    Compute cross Ripley curve between two phenotypes.

    Returns curves for each image.

    Parameters
    ----------
    phenotype_key : str
        Column in ``adata.obs`` containing phenotype labels.
    source_phenotype, target_phenotype : str
        The two phenotype labels to compare.
    x_key, y_key : str
        Column names in ``adata.obs`` containing spatial coordinates.
    image_key : str
        Column in ``adata.obs`` identifying which image each cell belongs to.
    """

    rows = []

    for img in adata.obs[image_key].unique():

        df = adata.obs[adata.obs[image_key] == img]

        source = df[
            df[phenotype_key] == source_phenotype
        ]

        target = df[
            df[phenotype_key] == target_phenotype
        ]

        source_coords = source[[x_key, y_key]].to_numpy()
        target_coords = target[[x_key, y_key]].to_numpy()

        curve = cross_ripleys_curve(
            source_coords,
            target_coords,
            radii,
        )

        if curve.empty:
            continue

        curve[image_key] = img
        curve["source"] = source_phenotype
        curve["target"] = target_phenotype

        rows.append(curve)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def cross_ripley_envelope_by_phenotype(
    adata,
    phenotype_key,
    source_phenotype,
    target_phenotype,
    radii,
    n_sim=99,
    x_key="X_centroid",
    y_key="Y_centroid",
    image_key="imageid",
):
    """
    Monte Carlo cross-Ripley envelope between two phenotypes.

    Returns envelopes for each image.

    Biological interpretation
    -------------------------
    Determines whether spatial association between two phenotypes
    is stronger than expected under spatial randomness.

    Parameters
    ----------
    phenotype_key : str
        Column in ``adata.obs`` containing phenotype labels.
    source_phenotype, target_phenotype : str
        The two phenotype labels to compare.
    radii : array-like
        Distances at which the cross-Ripley curve is evaluated.
    x_key, y_key : str
        Column names in ``adata.obs`` containing spatial coordinates.
    image_key : str
        Column in ``adata.obs`` identifying which image each cell belongs to.
    """

    rows = []

    for img in adata.obs[image_key].unique():

        df = adata.obs[adata.obs[image_key] == img]

        source = df[df[phenotype_key] == source_phenotype]
        target = df[df[phenotype_key] == target_phenotype]

        source_coords = source[[x_key, y_key]].to_numpy()
        target_coords = target[[x_key, y_key]].to_numpy()

        envelope = cross_ripley_envelope(
            source_coords,
            target_coords,
            radii,
            n_sim=n_sim,
        )

        if envelope.empty:
            continue

        envelope[image_key] = img
        envelope["source"] = source_phenotype
        envelope["target"] = target_phenotype

        rows.append(envelope)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def cross_ripley_permutation_envelope(
    adata,
    phenotype_key,
    source_phenotype,
    target_phenotype,
    radii,
    n_sim=199,
    x_key="X_centroid",
    y_key="Y_centroid",
    image_key="imageid",
    random_state=None,
):
    """
    Cross-Ripley permutation envelope.

    This method randomizes phenotype labels while preserving
    the spatial structure of the tissue.

    This is the preferred null model for multiplex imaging data.

    Parameters
    ----------
    phenotype_key : str
        Column in ``adata.obs`` containing phenotype labels.
    source_phenotype, target_phenotype : str
        The two phenotype labels to compare.
    radii : array-like
        Distances at which the cross-Ripley curve is evaluated.
    x_key, y_key : str
        Column names in ``adata.obs`` containing spatial coordinates.
    image_key : str
        Column in ``adata.obs`` identifying which image each cell belongs to.
    """
    coords = _clean_coords(adata.obs[[x_key, y_key]].to_numpy())

    valid = np.isfinite(adata.obs[[x_key, y_key]].to_numpy()).all(axis=1)
    phenotypes = adata.obs.loc[valid, phenotype_key].to_numpy()
    image_ids = adata.obs.loc[valid, image_key].to_numpy()
    rng = np.random.default_rng(random_state)

    source_coords = coords[phenotypes == source_phenotype]
    target_coords = coords[phenotypes == target_phenotype]

    observed = cross_ripleys_curve(source_coords, target_coords, radii)

    if observed.empty:
        return observed

    sims = []

    for i in range(n_sim):
        permuted = _permute_values_within_groups(
            phenotypes,
            image_ids,
            rng,
        )

        sim_source = coords[permuted == source_phenotype]
        sim_target = coords[permuted == target_phenotype]

        sim_curve = cross_ripleys_curve(sim_source, sim_target, radii)

        sims.append(sim_curve["L_minus_r"].values)

    sims = np.array(sims)

    observed["envelope_low"] = np.percentile(sims, 2.5, axis=0)
    observed["envelope_high"] = np.percentile(sims, 97.5, axis=0)

    return observed


#============================================================
# 5. Extracting interaction scales from Ripley curves
#============================================================
def ripley_interaction_scale(curve_df):
    """
    Extract characteristic interaction scale from a Ripley curve.

    Parameters
    ----------
    curve_df : DataFrame
        Output from ripleys_curve() or cross_ripleys_curve()

    Returns
    -------
    dict
        interaction_radius
        max_L_minus_r
        interaction_strength
    """

    if curve_df.empty:
        return {
            "interaction_radius": np.nan,
            "max_L_minus_r": np.nan,
            "interaction_strength": np.nan,
        }

    idx = curve_df["L_minus_r"].idxmax()

    r_star = curve_df.loc[idx, "radius"]
    max_val = curve_df.loc[idx, "L_minus_r"]

    return {
        "interaction_radius": r_star,
        "max_L_minus_r": max_val,
        "interaction_strength": max_val,
    }


def ripley_spatial_scales(curve_df):
    """
    Extract biologically meaningful spatial scales from a Ripley curve.

    Parameters
    ----------
    curve_df : DataFrame
        Output from ripleys_curve() or cross_ripleys_curve()

    Returns
    -------
    dict
        first_clustering_radius
        peak_clustering_radius
        peak_strength
        repulsion_radius
    """

    if curve_df.empty:
        return {
            "first_clustering_radius": np.nan,
            "peak_clustering_radius": np.nan,
            "peak_strength": np.nan,
            "repulsion_radius": np.nan,
        }

    r = curve_df["radius"].values
    Lr = curve_df["L_minus_r"].values

    # ------------------------------------------------
    # first clustering scale
    # ------------------------------------------------

    positive_idx = np.where(Lr > 0)[0]

    first_cluster = np.nan
    if len(positive_idx) > 0:
        first_cluster = r[positive_idx[0]]

    # ------------------------------------------------
    # peak clustering
    # ------------------------------------------------

    peak_idx = np.argmax(Lr)

    peak_radius = r[peak_idx]
    peak_strength = Lr[peak_idx]

    # ------------------------------------------------
    # repulsion scale
    # ------------------------------------------------

    repulsion = np.nan

    after_peak = Lr[peak_idx:]

    negative_idx = np.where(after_peak < 0)[0]

    if len(negative_idx) > 0:
        repulsion = r[peak_idx + negative_idx[0]]

    return {
        "first_clustering_radius": first_cluster,
        "peak_clustering_radius": peak_radius,
        "peak_strength": peak_strength,
        "repulsion_radius": repulsion,
    }


# ============================================================
# 6. MORAN'S I (GLOBAL SPATIAL AUTOCORRELATION)
# ============================================================

def morans_i(coords, values, k=8):
    """
    Global Moran's I.

    Measures whether similar values cluster spatially.

    Interpretation:
        I > 0 → clustering
        I = 0 → random
        I < 0 → dispersion

    Parameters
    ----------
    coords : array-like of shape (n, 2)
        Spatial coordinates for each observation.
    values : array-like of shape (n,)
        Continuous feature measured at each coordinate.
    k : int
        Number of nearest neighbors used to define the spatial graph.
    """

    coords = np.asarray(coords)
    values = np.asarray(values, dtype=float)

    valid = np.isfinite(values)

    coords = coords[valid]
    values = values[valid]

    n = len(values)

    if n < 3:
        return np.nan

    k_eff = _resolve_k(n, k)

    if k_eff is None:
        return np.nan

    # build spatial neighbor graph
    W = kneighbors_graph(
        coords,
        k_eff,
        mode="connectivity",
        include_self=False,
    )

    W = W.toarray()

    x = values - values.mean()

    denom = np.sum(x ** 2)

    if denom == 0:
        return np.nan

    numerator = np.sum(W * np.outer(x, x))

    I = (n / W.sum()) * (numerator / denom)

    return I


def morans_i_permutation_test(coords, values, k=8, n_sim=999, random_state=None):
    """
    Permutation test for global Moran's I.
    """
    coords = np.asarray(coords)
    values = np.asarray(values, dtype=float)

    valid = np.isfinite(values)
    coords = coords[valid]
    values = values[valid]

    observed = morans_i(coords, values, k=k)

    if not np.isfinite(observed):
        return {
            "observed": np.nan,
            "p_value": np.nan,
            "null_mean": np.nan,
            "null_std": np.nan,
            "z_score": np.nan,
            "n_sim": 0,
        }

    rng = np.random.default_rng(random_state)
    sims = []

    for i in range(n_sim):
        sim_values = rng.permutation(values)
        sim_stat = morans_i(coords, sim_values, k=k)

        if np.isfinite(sim_stat):
            sims.append(sim_stat)

    sims = np.asarray(sims, dtype=float)

    if len(sims) == 0:
        raise ValueError("All permutation simulations returned invalid statistics.")

    null_mean = sims.mean()
    null_std = sims.std(ddof=1) if len(sims) > 1 else 0.0
    p_value = (np.sum(np.abs(sims) >= abs(observed)) + 1) / (len(sims) + 1)
    z_score = np.nan if null_std == 0 else (observed - null_mean) / null_std

    return {
        "observed": observed,
        "p_value": p_value,
        "null_mean": null_mean,
        "null_std": null_std,
        "z_score": z_score,
        "n_sim": len(sims),
    }


def morans_i_by_image(
    adata,
    value_key,
    x_key="X_centroid",
    y_key="Y_centroid",
    image_key="imageid",
    k=8, # number of neighbors for spatial weights
):
    """
    Compute Moran's I per image for a spatial feature.

    Parameters
    ----------
    value_key : str
        Column in ``adata.obs`` containing the numeric feature to analyze.
    x_key, y_key : str
        Column names in ``adata.obs`` containing spatial coordinates.
    image_key : str
        Column in ``adata.obs`` identifying which image each cell belongs to.
    k : int
        Number of nearest neighbors used to define the spatial graph.
    """

    rows = []

    for img in adata.obs[image_key].unique():

        idx = adata.obs.index[adata.obs[image_key] == img]

        coords = adata.obs.loc[idx, [x_key, y_key]].to_numpy()

        values = adata.obs.loc[idx, value_key].to_numpy()

        rows.append({
            image_key: img,
            "morans_i": morans_i(coords, values, k=k)
        })

    return pd.DataFrame(rows)


def morans_i_by_image_permutation_test(
    adata,
    value_key,
    x_key="X_centroid",
    y_key="Y_centroid",
    image_key="imageid",
    k=8,
    n_sim=999,
    random_state=None,
):
    """
    Permutation test for Moran's I per image.
    """
    rows = []
    rng = np.random.default_rng(random_state)

    for img in adata.obs[image_key].unique():
        idx = adata.obs.index[adata.obs[image_key] == img]
        coords = adata.obs.loc[idx, [x_key, y_key]].to_numpy()
        values = adata.obs.loc[idx, value_key].to_numpy()

        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        stats = morans_i_permutation_test(
            coords,
            values,
            k=k,
            n_sim=n_sim,
            random_state=seed,
        )

        rows.append({
            image_key: img,
            **stats,
        })

    return pd.DataFrame(rows)


# ============================================================
# 7. LOCAL MORAN'S I (SPATIAL HOTSPOTS)
# ============================================================

def local_morans_i(
        coords, 
        values, 
        k=8 # number of neighbors for spatial weights
    ):
    """
    Local Moran's I.

    Detects spatial hotspots of similar values.

    High positive values indicate clusters of high values.

    Parameters
    ----------
    coords : array-like of shape (n, 2)
        Spatial coordinates for each observation.
    values : array-like of shape (n,)
        Continuous feature measured at each coordinate.
    k : int
        Number of nearest neighbors used to define the spatial graph.
    """

    coords = np.asarray(coords)
    values = np.asarray(values, dtype=float)

    valid = np.isfinite(values)

    coords_v = coords[valid]
    values_v = values[valid]

    out = np.full(len(values), np.nan)

    n = len(values_v)

    if n < 3:
        return out

    k_eff = _resolve_k(n, k)

    if k_eff is None:
        return out

    W = kneighbors_graph(
        coords_v,
        k_eff,
        mode="connectivity",
        include_self=False,
    ).toarray()

    x = values_v - values_v.mean()

    m2 = np.sum(x ** 2) / n

    if m2 == 0:
        return out

    local_I = x * (W @ x) / m2

    out[np.where(valid)[0]] = local_I

    return out


def add_local_morans_i(
    adata,
    value_key,
    out_key=None,
    x_key="X_centroid",
    y_key="Y_centroid",
    image_key="imageid",
    k=8,
):
    """
    Compute Local Moran's I and add to adata.obs.

    Each cell receives a spatial autocorrelation value.

    Parameters
    ----------
    value_key : str
        Column in ``adata.obs`` containing the numeric feature to analyze.
    out_key : str, optional
        Name of the output column added to ``adata.obs``.
    x_key, y_key : str
        Column names in ``adata.obs`` containing spatial coordinates.
    image_key : str
        Column in ``adata.obs`` identifying which image each cell belongs to.
    k : int
        Number of nearest neighbors used to define the spatial graph.
    """

    if out_key is None:
        out_key = f"local_morans_i__{value_key}"

    adata.obs[out_key] = np.nan

    for img in adata.obs[image_key].unique():

        idx = adata.obs.index[adata.obs[image_key] == img]

        coords = adata.obs.loc[idx, [x_key, y_key]].to_numpy()

        values = adata.obs.loc[idx, value_key].to_numpy()

        local_I = local_morans_i(coords, values, k=k)

        adata.obs.loc[idx, out_key] = local_I

    return adata

# ============================================================
# 8. CROSS MORAN'S I
# ============================================================

def cross_morans_i(coords, x_values, y_values, k=8):
    """
    Cross Moran's I.

    Measures spatial association between two features.

    Example
    -------
    tumor_density vs fibroblast_density

    Parameters
    ----------
    coords : array-like of shape (n, 2)
        Spatial coordinates shared by both features.
    x_values, y_values : array-like of shape (n,)
        Continuous features measured at the same coordinates.
    k : int
        Number of nearest neighbors used to define the spatial graph.
    """

    coords = np.asarray(coords)

    x = np.asarray(x_values)
    y = np.asarray(y_values)

    valid = np.isfinite(x) & np.isfinite(y)

    coords = coords[valid]
    x = x[valid]
    y = y[valid]

    n = len(x)

    if n < 3:
        return np.nan

    k_eff = _resolve_k(n, k)

    if k_eff is None:
        return np.nan

    W = kneighbors_graph(coords, k_eff, mode="connectivity", include_self=False)

    W = W.toarray()

    x = x - x.mean()
    y = y - y.mean()

    numerator = np.sum(W * np.outer(x, y))

    denom = np.sqrt(np.sum(x**2) * np.sum(y**2))

    if denom == 0:
        return np.nan

    I = (n / W.sum()) * (numerator / denom)

    return I


def cross_morans_i_permutation_test(
    coords,
    x_values,
    y_values,
    k=8,
    n_sim=999,
    permute="y",
    random_state=None,
):
    """
    Permutation test for global cross Moran's I.

    Parameters
    ----------
    permute : {"x", "y"}
        Which feature to shuffle for the null model.
    """
    if permute not in {"x", "y"}:
        raise ValueError("permute must be either 'x' or 'y'")

    coords = np.asarray(coords)
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)

    valid = np.isfinite(x_values) & np.isfinite(y_values)
    coords = coords[valid]
    x_values = x_values[valid]
    y_values = y_values[valid]

    observed = cross_morans_i(coords, x_values, y_values, k=k)

    if not np.isfinite(observed):
        return {
            "observed": np.nan,
            "p_value": np.nan,
            "null_mean": np.nan,
            "null_std": np.nan,
            "z_score": np.nan,
            "n_sim": 0,
        }

    rng = np.random.default_rng(random_state)
    sims = []

    for i in range(n_sim):
        if permute == "x":
            sim_x = rng.permutation(x_values)
            sim_y = y_values
        else:
            sim_x = x_values
            sim_y = rng.permutation(y_values)

        sim_stat = cross_morans_i(coords, sim_x, sim_y, k=k)

        if np.isfinite(sim_stat):
            sims.append(sim_stat)

    sims = np.asarray(sims, dtype=float)

    if len(sims) == 0:
        raise ValueError("All permutation simulations returned invalid statistics.")

    null_mean = sims.mean()
    null_std = sims.std(ddof=1) if len(sims) > 1 else 0.0
    p_value = (np.sum(np.abs(sims) >= abs(observed)) + 1) / (len(sims) + 1)
    z_score = np.nan if null_std == 0 else (observed - null_mean) / null_std

    return {
        "observed": observed,
        "p_value": p_value,
        "null_mean": null_mean,
        "null_std": null_std,
        "z_score": z_score,
        "n_sim": len(sims),
    }

def local_cross_morans_i(coords, x_values, y_values, k=8):
    """
    Local cross Moran's I.

    Detects spatial regions where two features interact.

    Example
    -------
    tumor density correlated with collagen alignment.

    Parameters
    ----------
    coords : array-like of shape (n, 2)
        Spatial coordinates shared by both features.
    x_values, y_values : array-like of shape (n,)
        Continuous features measured at the same coordinates.
    k : int
        Number of nearest neighbors used to define the spatial graph.
    """

    coords = np.asarray(coords)

    x = np.asarray(x_values)
    y = np.asarray(y_values)

    valid = np.isfinite(x) & np.isfinite(y)

    coords_v = coords[valid]
    x_v = x[valid]
    y_v = y[valid]

    out = np.full(len(x), np.nan)

    n = len(x_v)

    if n < 3:
        return out

    k_eff = _resolve_k(n, k)

    if k_eff is None:
        return out

    W = kneighbors_graph(
        coords_v,
        k_eff,
        mode="connectivity",
        include_self=False,
    ).toarray()

    x_c = x_v - x_v.mean()
    y_c = y_v - y_v.mean()

    m2 = np.sum(y_c**2) / n

    if m2 == 0:
        return out

    local_I = x_c * (W @ y_c) / m2

    out[np.where(valid)[0]] = local_I

    return out
