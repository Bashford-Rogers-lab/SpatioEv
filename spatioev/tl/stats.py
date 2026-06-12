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
from __future__ import annotations


import numpy as np
import pandas as pd

from sklearn.neighbors import BallTree, kneighbors_graph
from .preprocessing import compute_convex_hull_area
from scipy.spatial import ConvexHull


def _random_points_in_hull(coords, n_points):
    """
    Generate random points inside the convex hull of coordinates.

    Used for Monte Carlo envelope testing.
    """
    try:
        from shapely.geometry import Point, Polygon
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        raise ImportError(
            "Ripley envelope simulations require shapely. Install SpatioEv "
            "with its core dependencies or `pip install shapely`."
        ) from exc

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


def _resolve_window_area(coords, window_coords=None):
    """
    Compute the observation-window area used for Ripley statistics.

    If ``window_coords`` is provided, its convex hull defines the tissue
    window; otherwise the hull of ``coords`` is used.
    """
    if window_coords is None:
        return compute_convex_hull_area(coords)

    window_coords = _clean_coords(window_coords)
    if len(window_coords) < 3:
        return np.nan

    return compute_convex_hull_area(window_coords)


# ============================================================
# 1. GLOBAL RIPLEY'S K
# ============================================================

def ripleys_k(coords: np.ndarray, radius: float, window_coords: np.ndarray | None=None) -> float:
    """Compute Ripley's K statistic and derived transforms at a single radius.

    Ripley's K measures how many neighbours a typical point has within a given
    radius relative to the expectation under complete spatial randomness (CSR).
    Values of L − r > 0 indicate clustering; < 0 indicate dispersion.

    Parameters
    ----------
    coords : array-like of shape (n, 2)
        Spatial coordinates (x, y) for each cell.
    radius : float
        Search radius in the same units as *coords*.
    window_coords : array-like of shape (m, 2) or None
        Optional coordinates used to define the observation window (convex
        hull).  If ``None``, the hull of *coords* is used.

    Returns
    -------
    dict
        ``K_observed`` — observed Ripley's K value.
        ``K_expected`` — CSR expectation (π r²).
        ``L`` — Besag's L transform (√(K / π)).
        ``L_minus_r`` — L − r, centred at 0 under CSR.

    Examples
    --------
    >>> import numpy as np
    >>> from spatioev.tl.stats import ripleys_k
    >>> coords = np.random.uniform(0, 500, (200, 2))
    >>> result = ripleys_k(coords, radius=50)
    >>> result["L_minus_r"]  # positive → clustering
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

    # tissue area / observation window area
    area = _resolve_window_area(coords, window_coords=window_coords)

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


def ripleys_curve(coords: np.ndarray, radii: np.ndarray | list[float], window_coords: np.ndarray | None=None) -> pd.DataFrame:
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

    area = _resolve_window_area(coords, window_coords=window_coords)

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


def ripley_envelope(coords: np.ndarray, radii: np.ndarray | list[float], n_sim: int=99, window_coords: np.ndarray | None=None) -> pd.DataFrame:
    """
    Monte Carlo envelope test for Ripley statistics.

    Randomizes point locations inside the convex hull.
    """

    coords = _clean_coords(coords)

    observed = ripleys_curve(coords, radii, window_coords=window_coords)

    if observed.empty:
        return observed

    sims = []

    for i in range(n_sim):

        sim_coords = _random_points_in_hull(coords, len(coords))

        sim_curve = ripleys_curve(sim_coords, radii, window_coords=window_coords)

        sims.append(sim_curve["L_minus_r"].values)

    sims = np.array(sims)

    observed["envelope_low"] = np.percentile(sims, 2.5, axis=0)
    observed["envelope_high"] = np.percentile(sims, 97.5, axis=0)

    return observed


# ============================================================
# 2. PHENOTYPE-AGNOSTIC RIPLEY'S K
# ============================================================

def ripleys_k_by_image(
    adata: ad.AnnData,
    radius: float,
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    image_key: str="imageid",
) -> pd.DataFrame:
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
    adata: ad.AnnData,
    phenotype_key: str,
    radius: float,
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    image_key: str="imageid",
) -> pd.DataFrame:
    """Compute Ripley's K per phenotype across all images.

    **Biological question:** Are cells of a given phenotype spatially
    clustered within each image?

    The observation window for each image is the convex hull of all cells
    in that image (not just the phenotype of interest), which avoids
    overestimating clustering at tissue boundaries.

    Results are stored in ``adata.uns["spatial_stats"]["ripley_k_by_phenotype"]``
    and returned as a DataFrame.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData with spatial coordinates and phenotype labels in ``adata.obs``.
    phenotype_key : str
        Column in ``adata.obs`` containing phenotype labels.
    radius : float
        Search radius in the same units as the coordinate columns.
    x_key : str
        Column in ``adata.obs`` for the x-coordinate.  Default ``"X_centroid"``.
    y_key : str
        Column in ``adata.obs`` for the y-coordinate.  Default ``"Y_centroid"``.
    image_key : str
        Column in ``adata.obs`` identifying the image/FOV each cell belongs
        to.  Default ``"imageid"``.

    Returns
    -------
    pandas.DataFrame
        One row per (image, phenotype) combination with columns:
        ``imageid``, ``phenotype``, ``radius``, ``K_observed``,
        ``K_expected``, ``L``, ``L_minus_r``.

    Examples
    --------
    >>> import spatioev as sv
    >>> k_df = sv.tl.ripleys_k_by_phenotype(adata, phenotype_key="phenotype", radius=50)
    >>> k_df[k_df["phenotype"] == "Tumor"][["imageid", "L_minus_r"]]
    """

    rows = []

    for img in adata.obs[image_key].unique():

        df = adata.obs[adata.obs[image_key] == img]

        phenotypes = df[phenotype_key].dropna().unique()
        window_coords = df[[x_key, y_key]].to_numpy()

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

                stats = ripleys_k(
                    coords,
                    radius,
                    window_coords=window_coords,
                )

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
    source_coords: np.ndarray,
    target_coords: np.ndarray,
    radius: float,
    window_coords: np.ndarray | None=None,
) -> float:
    """
    Cross Ripley's K statistic.

    Measures spatial interaction between two point sets.
    """

    source_coords, target_coords = _clean_paired_coords(
        source_coords,
        target_coords,
    )

    if len(source_coords) == 0 or len(target_coords) == 0:
        return {
            "K_observed": np.nan,
            "K_expected": np.nan,
            "L": np.nan,
            "L_minus_r": np.nan,
        }

    # remove NaN coordinates
    source_coords = source_coords[np.isfinite(source_coords).all(axis=1)]
    target_coords = target_coords[np.isfinite(target_coords).all(axis=1)]

    if len(source_coords) == 0 or len(target_coords) == 0:
        return {
            "K_observed": np.nan,
            "K_expected": np.nan,
            "L": np.nan,
            "L_minus_r": np.nan,
        }

    area = _resolve_window_area(
        np.vstack([source_coords, target_coords]),
        window_coords=window_coords,
    )
    if np.isnan(area):
        return {
            "K_observed": np.nan,
            "K_expected": np.nan,
            "L": np.nan,
            "L_minus_r": np.nan,
        }

    tree = BallTree(target_coords)

    neighbors = tree.query_radius(source_coords, r=radius)

    counts = np.array([len(n) for n in neighbors], dtype=float)

    n_source = len(source_coords)
    n_target = len(target_coords)

    K_obs = (area / (n_source * n_target)) * counts.sum()
    K_exp = np.pi * radius**2
    L = np.sqrt(K_obs / np.pi)

    return {
        "K_observed": K_obs,
        "K_expected": K_exp,
        "L": L,
        "L_minus_r": L - radius,
    }


def _cross_source_centered_stats(
    source_coords,
    target_coords,
    radius,
    window_coords=None,
):
    """
    Source-centered directional neighborhood summaries for a phenotype pair.

    These metrics answer questions like:
    - how many target cells does a typical source cell see within ``radius``?
    - how much larger is that than expected from global target density?
    - what fraction of source cells have at least one target neighbor?
    """
    source_coords, target_coords = _clean_paired_coords(
        source_coords,
        target_coords,
    )

    if len(source_coords) == 0 or len(target_coords) == 0:
        return {
            "mean_target_neighbors_per_source": np.nan,
            "expected_target_neighbors_per_source": np.nan,
            "source_neighbor_excess": np.nan,
            "source_neighbor_ratio": np.nan,
            "fraction_source_with_target_neighbor": np.nan,
        }

    area = _resolve_window_area(
        np.vstack([source_coords, target_coords]),
        window_coords=window_coords,
    )
    if np.isnan(area):
        return {
            "mean_target_neighbors_per_source": np.nan,
            "expected_target_neighbors_per_source": np.nan,
            "source_neighbor_excess": np.nan,
            "source_neighbor_ratio": np.nan,
            "fraction_source_with_target_neighbor": np.nan,
        }

    tree = BallTree(target_coords)
    neighbors = tree.query_radius(source_coords, r=radius)
    counts = np.array([len(n) for n in neighbors], dtype=float)

    mean_count = float(np.mean(counts))
    expected_count = float((len(target_coords) / area) * np.pi * radius**2)
    frac_with_neighbor = float(np.mean(counts > 0))

    if expected_count > 0:
        ratio = mean_count / expected_count
    else:
        ratio = np.nan

    return {
        "mean_target_neighbors_per_source": mean_count,
        "expected_target_neighbors_per_source": expected_count,
        "source_neighbor_excess": mean_count - expected_count,
        "source_neighbor_ratio": ratio,
        "fraction_source_with_target_neighbor": frac_with_neighbor,
        "directional_observed": mean_count,
        "directional_expected": expected_count,
        "directional_excess": mean_count - expected_count,
        "directional_ratio": ratio,
    }


def cross_ripleys_k_by_phenotype(
    adata: ad.AnnData,
    phenotype_key: str,
    source_phenotype: str,
    target_phenotype: str,
    radius: float,
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    image_key: str="imageid",
) -> pd.DataFrame:
    """
    Compute cross Ripley's K between two phenotypes.

    Biological interpretation
    -------------------------
    Tests whether cells of one phenotype spatially cluster
    around another phenotype.

    Notes
    -----
    The classical cross-Ripley fields returned here
    (``K_observed``, ``K_expected``, ``L``, ``L_minus_r``)
    are symmetric for a phenotype pair.

    To support directional interpretation from source -> target,
    this function also returns source-centered neighborhood metrics:
    ``directional_observed``, ``directional_expected``,
    ``directional_excess``, ``directional_ratio``,
    and ``fraction_source_with_target_neighbor``.

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
        window_coords = df[[x_key, y_key]].to_numpy()

        stats = cross_ripleys_k(
            source_coords,
            target_coords,
            radius,
            window_coords=window_coords,
        )

        rows.append({
            image_key: img,
            "source": source_phenotype,
            "target": target_phenotype,
            "radius": radius,
            **stats,
            **_cross_source_centered_stats(
                source_coords,
                target_coords,
                radius,
                window_coords=window_coords,
            ),
        })

    return pd.DataFrame(rows)


def cross_ripleys_k_all_pairs(
    adata: ad.AnnData,
    phenotype_key: str,
    radius: float,
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    image_key: str="imageid",
    include_self_pairs: bool=False,
) -> pd.DataFrame:
    """Compute cross-Ripley K for all ordered phenotype pairs.

    Iterates over every ordered (source, target) phenotype combination and
    calls :func:`cross_ripleys_k_by_phenotype` for each pair.  Results are
    concatenated into a single long-form DataFrame and cached in
    ``adata.uns["spatial_stats"]["cross_ripley_k_all_pairs"]``.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData with cell coordinates and phenotype labels in ``adata.obs``.
    phenotype_key : str
        Column in ``adata.obs`` containing phenotype labels.
    radius : float
        Search radius for Ripley K computation.
    x_key, y_key : str
        Column names for x and y cell coordinates.
    image_key : str
        Column identifying image/FOV membership.
    include_self_pairs : bool
        If ``True``, include same-phenotype (source == target) pairs.

    Returns
    -------
    pandas.DataFrame
        Long-form table with columns ``source_phenotype``, ``target_phenotype``,
        ``imageid``, ``K``, ``L``, and ``L_minus_r``.
    """
    phenotypes = adata.obs[phenotype_key].dropna().unique()

    rows = []

    for p1 in phenotypes:
        for p2 in phenotypes:
            if not include_self_pairs and p1 == p2:
                continue

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
    out = pd.concat(rows, ignore_index=True)

    if "spatial_stats" not in adata.uns:
        adata.uns["spatial_stats"] = {}

    adata.uns["spatial_stats"]["cross_ripley_k_all_pairs"] = out.copy()

    return out

# ============================================================
# 4. Cross Ripley curves and significance testing
# ============================================================

def cross_ripleys_curve(
    source_coords: np.ndarray,
    target_coords: np.ndarray,
    radii: np.ndarray | list[float],
    window_coords: np.ndarray | None=None,
) -> pd.DataFrame:
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

    area = _resolve_window_area(
        np.vstack([source_coords, target_coords]),
        window_coords=window_coords,
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


def cross_ripley_envelope(
    source_coords: np.ndarray,
    target_coords: np.ndarray,
    radii: np.ndarray | list[float],
    n_sim: int=99,
    window_coords: np.ndarray | None=None,
) -> pd.DataFrame:
    """
    Monte Carlo envelope for cross Ripley analysis.

    Tests whether two phenotypes interact more than expected.
    """

    source_coords, target_coords = _clean_paired_coords(
        source_coords,
        target_coords,
    )

    observed = cross_ripleys_curve(
        source_coords,
        target_coords,
        radii,
        window_coords=window_coords,
    )

    if observed.empty:
        return observed

    sims = []

    for i in range(n_sim):

        if window_coords is not None and len(_clean_coords(window_coords)) >= 3:
            random_window = _clean_coords(window_coords)
        elif len(source_coords) >= 3:
            random_window = source_coords
        else:
            random_window = target_coords
        rand_source = _random_points_in_hull(random_window, len(source_coords))

        sim_curve = cross_ripleys_curve(
            rand_source,
            target_coords,
            radii,
            window_coords=random_window,
        )

        sims.append(sim_curve["L_minus_r"].values)

    sims = np.array(sims)

    observed["envelope_low"] = np.percentile(sims, 2.5, axis=0)
    observed["envelope_high"] = np.percentile(sims, 97.5, axis=0)

    return observed


def cross_ripleys_curve_by_phenotype(
    adata: ad.AnnData,
    phenotype_key: str,
    source_phenotype: str,
    target_phenotype: str,
    radii: np.ndarray | list[float],
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    image_key: str="imageid",
) -> pd.DataFrame:
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
        window_coords = df[[x_key, y_key]].to_numpy()

        curve = cross_ripleys_curve(
            source_coords,
            target_coords,
            radii,
            window_coords=window_coords,
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
    adata: ad.AnnData,
    phenotype_key: str,
    source_phenotype: str,
    target_phenotype: str,
    radii: np.ndarray | list[float],
    n_sim: int=99,
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    image_key: str="imageid",
) -> pd.DataFrame:
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
        window_coords = df[[x_key, y_key]].to_numpy()

        envelope = cross_ripley_envelope(
            source_coords,
            target_coords,
            radii,
            n_sim=n_sim,
            window_coords=window_coords,
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
    adata: ad.AnnData,
    phenotype_key: str,
    source_phenotype: str,
    target_phenotype: str,
    radii: np.ndarray | list[float],
    n_sim: int=199,
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    image_key: str="imageid",
    random_state: int=None,
) -> pd.DataFrame:
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

    observed = cross_ripleys_curve(
        source_coords,
        target_coords,
        radii,
        window_coords=coords,
    )

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

        sim_curve = cross_ripleys_curve(
            sim_source,
            sim_target,
            radii,
            window_coords=coords,
        )

        sims.append(sim_curve["L_minus_r"].values)

    sims = np.array(sims)

    observed["envelope_low"] = np.percentile(sims, 2.5, axis=0)
    observed["envelope_high"] = np.percentile(sims, 97.5, axis=0)

    return observed


def ripley_local_counts_by_phenotype(
    adata: ad.AnnData,
    phenotype_key: str,
    radius: float,
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    image_key: str="imageid",
    min_cells_per_phenotype: int=3,
    add_to_obs: bool=False,
    count_key: str=None,
    excess_key: str=None,
    ratio_key: str=None,
    hotspot_key: str=None,
) -> pd.DataFrame:
    """
    Identify cells contributing to phenotype clustering at a chosen radius.

    For each cell, this computes the number of same-phenotype neighbors within
    ``radius`` inside each image. It also estimates the expected neighbor count
    under CSR using the phenotype density in the full image window.

    Returns
    -------
    pd.DataFrame
        One row per valid cell with local Ripley-style neighborhood statistics.
    """
    rows = []

    for img in adata.obs[image_key].dropna().unique():
        df = adata.obs[adata.obs[image_key] == img]
        window_coords = df[[x_key, y_key]].to_numpy()
        area = _resolve_window_area(window_coords, window_coords=window_coords)

        if np.isnan(area):
            continue

        for pheno in df[phenotype_key].dropna().unique():
            subset = df[df[phenotype_key] == pheno]
            coords = subset[[x_key, y_key]].to_numpy()

            if len(coords) < min_cells_per_phenotype:
                continue

            tree = BallTree(coords)
            neighbors = tree.query_radius(coords, r=radius)
            counts = np.array([len(nbrs) - 1 for nbrs in neighbors], dtype=float)

            density = len(coords) / area
            expected_count = density * np.pi * radius**2
            excess = counts - expected_count
            ratio = counts / expected_count if expected_count > 0 else np.full_like(counts, np.nan)
            is_hotspot = excess > 0

            for idx, cell_id in enumerate(subset.index):
                rows.append({
                    "cell_id": cell_id,
                    image_key: img,
                    "phenotype": pheno,
                    "radius": radius,
                    "same_type_neighbor_count": counts[idx],
                    "expected_same_type_neighbor_count": expected_count,
                    "same_type_neighbor_excess": excess[idx],
                    "same_type_neighbor_ratio": ratio[idx],
                    "is_ripley_hotspot": bool(is_hotspot[idx]),
                })

    out = pd.DataFrame(rows)

    if add_to_obs and not out.empty:
        count_key = count_key or f"ripley_local_count__{phenotype_key}__r{radius}"
        excess_key = excess_key or f"ripley_local_excess__{phenotype_key}__r{radius}"
        ratio_key = ratio_key or f"ripley_local_ratio__{phenotype_key}__r{radius}"
        hotspot_key = hotspot_key or f"ripley_local_hotspot__{phenotype_key}__r{radius}"

        adata.obs[count_key] = np.nan
        adata.obs[excess_key] = np.nan
        adata.obs[ratio_key] = np.nan
        adata.obs[hotspot_key] = False

        adata.obs.loc[out["cell_id"], count_key] = out["same_type_neighbor_count"].to_numpy()
        adata.obs.loc[out["cell_id"], excess_key] = out["same_type_neighbor_excess"].to_numpy()
        adata.obs.loc[out["cell_id"], ratio_key] = out["same_type_neighbor_ratio"].to_numpy()
        adata.obs.loc[out["cell_id"], hotspot_key] = out["is_ripley_hotspot"].to_numpy()

    return out


def cross_ripley_local_counts(
    adata: ad.AnnData,
    phenotype_key: str,
    source_phenotype: str,
    target_phenotype: str,
    radius: float,
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    image_key: str="imageid",
    min_source_cells: int=1,
    min_target_cells: int=1,
    add_to_obs: bool=False,
    count_key: str=None,
    excess_key: str=None,
    ratio_key: str=None,
    hotspot_key: str=None,
) -> pd.DataFrame:
    """
    Identify source cells embedded in target-enriched local neighborhoods.

    For each source-phenotype cell, this computes the number of nearby
    target-phenotype cells within ``radius`` inside each image and compares
    that count to the expectation under CSR using the target density in the
    full image window.

    Returns
    -------
    pd.DataFrame
        One row per source cell with local cross-Ripley-style statistics.
    """
    rows = []

    for img in adata.obs[image_key].dropna().unique():
        df = adata.obs[adata.obs[image_key] == img]
        window_coords = df[[x_key, y_key]].to_numpy()
        area = _resolve_window_area(window_coords, window_coords=window_coords)

        if np.isnan(area):
            continue

        source = df[df[phenotype_key] == source_phenotype]
        target = df[df[phenotype_key] == target_phenotype]

        if len(source) < min_source_cells or len(target) < min_target_cells:
            continue

        source_coords = source[[x_key, y_key]].to_numpy()
        target_coords = target[[x_key, y_key]].to_numpy()

        tree = BallTree(target_coords)
        neighbors = tree.query_radius(source_coords, r=radius)
        counts = np.array([len(nbrs) for nbrs in neighbors], dtype=float)

        if source_phenotype == target_phenotype:
            counts = np.maximum(counts - 1, 0)

        target_density = len(target_coords) / area
        expected_count = target_density * np.pi * radius**2
        excess = counts - expected_count
        ratio = (
            counts / expected_count
            if expected_count > 0
            else np.full_like(counts, np.nan)
        )
        is_hotspot = excess > 0

        for idx, cell_id in enumerate(source.index):
            rows.append({
                "cell_id": cell_id,
                image_key: img,
                "source": source_phenotype,
                "target": target_phenotype,
                "radius": radius,
                "target_neighbor_count": counts[idx],
                "expected_target_neighbor_count": expected_count,
                "target_neighbor_excess": excess[idx],
                "target_neighbor_ratio": ratio[idx],
                "is_cross_ripley_hotspot": bool(is_hotspot[idx]),
            })

    out = pd.DataFrame(rows)

    if add_to_obs and not out.empty:
        if source_phenotype == target_phenotype:
            prefix = f"cross_ripley_local__{source_phenotype}__self__r{radius}"
        else:
            prefix = f"cross_ripley_local__{source_phenotype}__to__{target_phenotype}__r{radius}"

        count_key = count_key or f"{prefix}__count"
        excess_key = excess_key or f"{prefix}__excess"
        ratio_key = ratio_key or f"{prefix}__ratio"
        hotspot_key = hotspot_key or f"{prefix}__hotspot"

        adata.obs[count_key] = np.nan
        adata.obs[excess_key] = np.nan
        adata.obs[ratio_key] = np.nan
        adata.obs[hotspot_key] = False

        adata.obs.loc[out["cell_id"], count_key] = out["target_neighbor_count"].to_numpy()
        adata.obs.loc[out["cell_id"], excess_key] = out["target_neighbor_excess"].to_numpy()
        adata.obs.loc[out["cell_id"], ratio_key] = out["target_neighbor_ratio"].to_numpy()
        adata.obs.loc[out["cell_id"], hotspot_key] = out["is_cross_ripley_hotspot"].to_numpy()

    return out


#============================================================
# 5. Extracting interaction scales from Ripley curves
#============================================================
def ripley_interaction_scale(curve_df: pd.DataFrame) -> pd.DataFrame:
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


def ripley_spatial_scales(curve_df: pd.DataFrame) -> pd.DataFrame:
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

def morans_i(coords: np.ndarray, values: np.ndarray, k: int=8) -> float:
    """Compute the global Moran's I spatial autocorrelation statistic.

    Moran's I measures whether similar feature values cluster spatially
    (positive autocorrelation) or repel each other (negative autocorrelation).

    Interpretation:
        I > 0  → spatial clustering (similar values are neighbours)
        I ≈ 0  → spatial randomness (CSR)
        I < 0  → spatial dispersion (dissimilar values are neighbours)

    Parameters
    ----------
    coords : array-like of shape (n, 2)
        Spatial (x, y) coordinates for each cell.
    values : array-like of shape (n,)
        Continuous feature measured at each coordinate (e.g. marker
        expression, area, local density).  Non-finite values are dropped.
    k : int
        Number of nearest neighbours used to build the spatial weight
        matrix.  Default is 8.

    Returns
    -------
    float
        Moran's I statistic.  Returns ``np.nan`` if fewer than 3 valid
        observations are present or if all values are identical.

    Examples
    --------
    >>> import numpy as np
    >>> import spatioev as sv
    >>> coords = np.random.uniform(0, 500, (300, 2))
    >>> values = np.random.normal(size=300)
    >>> I = sv.tl.morans_i(coords, values)
    >>> -1.0 < I < 1.0
    True
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

    x = values - values.mean()

    denom = np.sum(x ** 2)

    if denom == 0:
        return np.nan

    Wx = W @ x
    numerator = np.dot(x, Wx)

    I = (n / W.sum()) * (numerator / denom)

    return I


def morans_i_permutation_test(coords: np.ndarray, values: np.ndarray, k: int=8, n_sim: int=999, random_state: int=None) -> dict:
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
    adata: ad.AnnData,
    value_key: str,
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    image_key: str="imageid",
    k: int=8, # number of neighbors for spatial weights
) -> pd.DataFrame:
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
    adata: ad.AnnData,
    value_key: str,
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    image_key: str="imageid",
    k: int=8,
    n_sim: int=999,
    random_state: int=None,
) -> pd.DataFrame:
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
        coords: np.ndarray, 
        values: np.ndarray, 
        k: int=8 # number of neighbors for spatial weights
    ) -> np.ndarray:
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
    )

    x = values_v - values_v.mean()

    m2 = np.sum(x ** 2) / n

    if m2 == 0:
        return out

    local_I = x * (W @ x) / m2

    out[np.where(valid)[0]] = local_I

    return out


def add_local_morans_i(
    adata: ad.AnnData,
    value_key: str,
    out_key: str=None,
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    image_key: str="imageid",
    k: int=8,
) -> ad.AnnData:
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


def classify_local_morans_i(
    values: np.ndarray,
    local_i: np.ndarray,
    value_threshold: float=0.0,
    local_i_threshold: float=0.0,
) -> np.ndarray:
    """
    Classify local Moran's I into hotspot/coldspot/outlier quadrants.

    Parameters
    ----------
    values : array-like
        Feature values, typically centered or z-scored so that 0 separates
        high from low.
    local_i : array-like
        Local Moran's I values for the same observations.
    value_threshold : float, default 0.0
        Threshold separating high vs low feature values.
    local_i_threshold : float, default 0.0
        Threshold separating positive vs negative local Moran's I.

    Returns
    -------
    np.ndarray of dtype object
        One of: ``high-high``, ``low-low``, ``high-low``, ``low-high``,
        or ``unclassified`` for missing values.
    """
    values = np.asarray(values, dtype=float)
    local_i = np.asarray(local_i, dtype=float)

    if values.shape[0] != local_i.shape[0]:
        raise ValueError("values and local_i must have the same length.")

    labels = np.full(values.shape[0], "unclassified", dtype=object)
    valid = np.isfinite(values) & np.isfinite(local_i)

    high_value = values > value_threshold
    low_value = values < value_threshold
    positive_i = local_i > local_i_threshold
    negative_i = local_i < local_i_threshold

    labels[valid & high_value & positive_i] = "high-high"
    labels[valid & low_value & positive_i] = "low-low"
    labels[valid & high_value & negative_i] = "high-low"
    labels[valid & low_value & negative_i] = "low-high"

    return labels


def add_local_morans_i_quadrants(
    adata: ad.AnnData,
    value_key: str,
    local_i_key: str=None,
    out_key: str=None,
    value_threshold: float=0.0,
    local_i_threshold: float=0.0,
) -> ad.AnnData:
    """
    Add quadrant-style local Moran classification to ``adata.obs``.

    This labels each cell as:
    - ``high-high``: high value surrounded by high values
    - ``low-low``: low value surrounded by low values
    - ``high-low``: high value surrounded by low values
    - ``low-high``: low value surrounded by high values
    """
    if local_i_key is None:
        local_i_key = f"local_morans_i__{value_key}"

    if local_i_key not in adata.obs.columns:
        raise ValueError(
            f"{local_i_key!r} not found in adata.obs. "
            "Run add_local_morans_i(...) first or provide local_i_key."
        )

    if value_key not in adata.obs.columns:
        raise ValueError(f"{value_key!r} not found in adata.obs.")

    if out_key is None:
        out_key = f"local_morans_quadrant__{value_key}"

    adata.obs[out_key] = classify_local_morans_i(
        values=adata.obs[value_key].to_numpy(),
        local_i=adata.obs[local_i_key].to_numpy(),
        value_threshold=value_threshold,
        local_i_threshold=local_i_threshold,
    )

    return adata

# ============================================================
# 8. CROSS MORAN'S I
# ============================================================

def cross_morans_i(coords: np.ndarray, x_values: np.ndarray, y_values: np.ndarray, k: int=8) -> float:
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

    x = x - x.mean()
    y = y - y.mean()

    Wy = W @ y
    numerator = np.dot(x, Wy)

    denom = np.sqrt(np.sum(x**2) * np.sum(y**2))

    if denom == 0:
        return np.nan

    I = (n / W.sum()) * (numerator / denom)

    return I


def cross_morans_i_by_image(
    adata: ad.AnnData,
    x_value_key: str,
    y_value_key: str,
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    image_key: str="imageid",
    k: int=8,
) -> pd.DataFrame:
    """
    Compute cross Moran's I per image for two spatial features.

    Parameters
    ----------
    x_value_key, y_value_key : str
        Columns in ``adata.obs`` containing the numeric features to analyze.
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
        x_values = adata.obs.loc[idx, x_value_key].to_numpy()
        y_values = adata.obs.loc[idx, y_value_key].to_numpy()

        rows.append({
            image_key: img,
            "x_feature": x_value_key,
            "y_feature": y_value_key,
            "cross_morans_i": cross_morans_i(
                coords,
                x_values,
                y_values,
                k=k,
            ),
        })

    return pd.DataFrame(rows)


def cross_morans_i_permutation_test(
    coords: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    k: int=8,
    n_sim: int=999,
    permute: bool="y",
    random_state: int=None,
) -> dict:
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


def cross_morans_i_by_image_permutation_test(
    adata: ad.AnnData,
    x_value_key: str,
    y_value_key: str,
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    image_key: str="imageid",
    k: int=8,
    n_sim: int=999,
    permute: bool="y",
    random_state: int=None,
) -> pd.DataFrame:
    """
    Permutation test for cross Moran's I per image.
    """

    rows = []
    rng = np.random.default_rng(random_state)

    for img in adata.obs[image_key].unique():
        idx = adata.obs.index[adata.obs[image_key] == img]

        coords = adata.obs.loc[idx, [x_key, y_key]].to_numpy()
        x_values = adata.obs.loc[idx, x_value_key].to_numpy()
        y_values = adata.obs.loc[idx, y_value_key].to_numpy()

        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        stats = cross_morans_i_permutation_test(
            coords,
            x_values,
            y_values,
            k=k,
            n_sim=n_sim,
            permute=permute,
            random_state=seed,
        )

        rows.append({
            image_key: img,
            "x_feature": x_value_key,
            "y_feature": y_value_key,
            **stats,
        })

    return pd.DataFrame(rows)


def local_cross_morans_i(coords: np.ndarray, x_values: np.ndarray, y_values: np.ndarray, k: int=8) -> np.ndarray:
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
    )

    x_c = x_v - x_v.mean()
    y_c = y_v - y_v.mean()

    m2 = np.sum(y_c**2) / n

    if m2 == 0:
        return out

    local_I = x_c * (W @ y_c) / m2

    out[np.where(valid)[0]] = local_I

    return out


def add_local_cross_morans_i(
    adata: ad.AnnData,
    x_value_key: str,
    y_value_key: str,
    out_key: str=None,
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    image_key: str="imageid",
    k: int=8,
) -> ad.AnnData:
    """
    Compute local cross Moran's I and add it to ``adata.obs``.

    Each cell receives a local spatial association value describing
    how strongly ``x_value_key`` aligns with neighboring ``y_value_key``.
    """

    if out_key is None:
        out_key = f"local_cross_morans_i__{x_value_key}__{y_value_key}"

    adata.obs[out_key] = np.nan

    for img in adata.obs[image_key].unique():
        idx = adata.obs.index[adata.obs[image_key] == img]

        coords = adata.obs.loc[idx, [x_key, y_key]].to_numpy()
        x_values = adata.obs.loc[idx, x_value_key].to_numpy()
        y_values = adata.obs.loc[idx, y_value_key].to_numpy()

        local_I = local_cross_morans_i(
            coords,
            x_values,
            y_values,
            k=k,
        )

        adata.obs.loc[idx, out_key] = local_I

    return adata


def classify_local_cross_morans_i(
    source_values: np.ndarray,
    target_neighbor_values: np.ndarray,
    local_cross_i: np.ndarray,
    source_threshold: float=0.0,
    target_threshold: float=0.0,
    local_i_threshold: float=0.0,
) -> np.ndarray:
    """
    Classify local cross Moran's I into quadrant-style labels.

    Parameters
    ----------
    source_values : array-like
        Source-cell feature values.
    target_neighbor_values : array-like
        Source-centered neighborhood summaries of the target feature.
    local_cross_i : array-like
        Local cross Moran's I values.
    source_threshold, target_threshold : float, default 0.0
        Thresholds separating high vs low values. These work best when the
        features are centered or z-scored.
    local_i_threshold : float, default 0.0
        Threshold separating positive vs negative local cross Moran's I.

    Returns
    -------
    np.ndarray of dtype object
        One of: ``high-high``, ``low-low``, ``high-low``, ``low-high``,
        or ``unclassified``.
    """
    source_values = np.asarray(source_values, dtype=float)
    target_neighbor_values = np.asarray(target_neighbor_values, dtype=float)
    local_cross_i = np.asarray(local_cross_i, dtype=float)

    n = len(source_values)
    if len(target_neighbor_values) != n or len(local_cross_i) != n:
        raise ValueError("All input arrays must have the same length.")

    labels = np.full(n, "unclassified", dtype=object)
    valid = (
        np.isfinite(source_values)
        & np.isfinite(target_neighbor_values)
        & np.isfinite(local_cross_i)
    )

    source_high = source_values > source_threshold
    source_low = source_values < source_threshold
    target_high = target_neighbor_values > target_threshold
    target_low = target_neighbor_values < target_threshold
    local_pos = local_cross_i > local_i_threshold
    local_neg = local_cross_i < local_i_threshold

    labels[valid & source_high & target_high & local_pos] = "high-high"
    labels[valid & source_low & target_low & local_pos] = "low-low"
    labels[valid & source_high & target_low & local_neg] = "high-low"
    labels[valid & source_low & target_high & local_neg] = "low-high"

    return labels


def add_local_cross_morans_i_quadrants(
    adata: ad.AnnData,
    source_value_key: str,
    target_neighbor_value_key: str,
    local_i_key: str=None,
    out_key: str=None,
    source_threshold: float=0.0,
    target_threshold: float=0.0,
    local_i_threshold: float=0.0,
) -> ad.AnnData:
    """
    Add quadrant-style local cross Moran classification to ``adata.obs``.

    This labels each source cell as:
    - ``high-high``: high source value in a high target-neighborhood context
    - ``low-low``: low source value in a low target-neighborhood context
    - ``high-low``: high source value in a low target-neighborhood context
    - ``low-high``: low source value in a high target-neighborhood context

    The sign of local cross Moran's I determines whether the association is
    locally concordant (positive) or discordant (negative).
    """
    if local_i_key is None:
        local_i_key = f"local_cross_morans_i__{source_value_key}__{target_neighbor_value_key}"

    missing = [
        key for key in [source_value_key, target_neighbor_value_key, local_i_key]
        if key not in adata.obs.columns
    ]
    if missing:
        raise ValueError(f"Required keys not found in adata.obs: {missing}")

    if out_key is None:
        out_key = f"local_cross_morans_quadrant__{source_value_key}__{target_neighbor_value_key}"

    adata.obs[out_key] = classify_local_cross_morans_i(
        source_values=adata.obs[source_value_key].to_numpy(),
        target_neighbor_values=adata.obs[target_neighbor_value_key].to_numpy(),
        local_cross_i=adata.obs[local_i_key].to_numpy(),
        source_threshold=source_threshold,
        target_threshold=target_threshold,
        local_i_threshold=local_i_threshold,
    )

    return adata


def summarize_target_features_around_source_cells(
    adata: ad.AnnData,
    phenotype_key: str,
    source_phenotype: str,
    target_phenotype: str,
    target_feature_keys: list[str],
    radius: float=None,
    k_neighbors: int=None,
    agg: str="mean",
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    image_key: str="imageid",
    source_only: bool=True,
) -> pd.DataFrame:
    """
    Summarize target-cell features in neighborhoods around source cells.

    Parameters
    ----------
    target_feature_keys : list of str
        Numeric target-cell features to aggregate around each source cell.
    radius : float, optional
        Radius-based neighborhood. Exactly one of ``radius`` or ``k_neighbors``
        must be provided.
    k_neighbors : int, optional
        kNN-based neighborhood in the target phenotype.
    agg : {"mean", "median"}
        Aggregation applied to target-feature values in each source-centered
        neighborhood.
    source_only : bool, default True
        If True, returns only source cells with aggregated target summaries.
        If False, merges the summary columns back onto a copy of ``adata.obs``.

    Returns
    -------
    pd.DataFrame
        Source-cell table with neighborhood target-feature summaries.
    """
    if (radius is None) == (k_neighbors is None):
        raise ValueError("Specify exactly one of radius or k_neighbors.")

    if agg not in {"mean", "median"}:
        raise ValueError("agg must be either 'mean' or 'median'.")

    missing = [c for c in target_feature_keys if c not in adata.obs.columns]
    if missing:
        raise ValueError(f"Target features not found in adata.obs: {missing}")

    rows = []

    for img in adata.obs[image_key].dropna().unique():
        df = adata.obs[adata.obs[image_key] == img]
        source = df[df[phenotype_key] == source_phenotype]
        target = df[df[phenotype_key] == target_phenotype]

        if source.empty or target.empty:
            continue

        source_coords = source[[x_key, y_key]].to_numpy()
        target_coords = target[[x_key, y_key]].to_numpy()

        tree = BallTree(target_coords)
        if radius is not None:
            neighbor_idx = tree.query_radius(source_coords, r=radius)
        else:
            k_eff = _resolve_k(len(target_coords), k_neighbors + 1 if source_phenotype == target_phenotype else k_neighbors)
            if k_eff is None:
                continue
            _, neighbor_idx = tree.query(source_coords, k=k_eff, return_distance=True)
            if k_eff == 1:
                neighbor_idx = neighbor_idx.reshape(-1, 1)

        target_values = target[target_feature_keys]

        for src_pos, cell_id in enumerate(source.index):
            idx = np.asarray(neighbor_idx[src_pos], dtype=int)

            if source_phenotype == target_phenotype and len(idx) > 0:
                src_label = source.index[src_pos]
                target_ids = target.index.to_numpy()
                idx = idx[target_ids[idx] != src_label]

            row = {
                "cell_id": cell_id,
                image_key: img,
                "source_phenotype": source_phenotype,
                "target_phenotype": target_phenotype,
                "n_target_neighbors": len(idx),
            }

            if len(idx) == 0:
                for feature in target_feature_keys:
                    row[f"neighbor_{agg}__{feature}"] = np.nan
            else:
                subset = target_values.iloc[idx]
                if agg == "mean":
                    vals = subset.mean(axis=0, numeric_only=True)
                else:
                    vals = subset.median(axis=0, numeric_only=True)
                for feature in target_feature_keys:
                    row[f"neighbor_{agg}__{feature}"] = vals.get(feature, np.nan)

            rows.append(row)

    out = pd.DataFrame(rows)

    if source_only:
        return out

    merged = adata.obs.copy()
    if not out.empty:
        merged = merged.merge(out, how="left", left_index=True, right_on="cell_id")
    return merged


def cross_morans_i_feature_matrix(
    adata: ad.AnnData,
    phenotype_key: str,
    source_phenotype: str,
    target_phenotype: str,
    source_feature_keys: list[str],
    target_feature_keys: list[str],
    radius: float=None,
    k_neighbors: int=None,
    agg: str="mean",
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    image_key: str="imageid",
    k: int=8,
) -> pd.DataFrame:
    """
    Compute a feature-by-feature cross Moran's I matrix between two phenotypes.

    Workflow:
    - summarize target-feature neighborhoods around each source cell
    - compute cross Moran's I between each source feature and each summarized
      target feature at the source-cell coordinates
    """
    missing_source = [c for c in source_feature_keys if c not in adata.obs.columns]
    missing_target = [c for c in target_feature_keys if c not in adata.obs.columns]
    if missing_source:
        raise ValueError(f"Source features not found in adata.obs: {missing_source}")
    if missing_target:
        raise ValueError(f"Target features not found in adata.obs: {missing_target}")

    source_df = adata.obs[adata.obs[phenotype_key] == source_phenotype].copy()
    neighbor_df = summarize_target_features_around_source_cells(
        adata=adata,
        phenotype_key=phenotype_key,
        source_phenotype=source_phenotype,
        target_phenotype=target_phenotype,
        target_feature_keys=target_feature_keys,
        radius=radius,
        k_neighbors=k_neighbors,
        agg=agg,
        x_key=x_key,
        y_key=y_key,
        image_key=image_key,
        source_only=True,
    )

    if neighbor_df.empty or source_df.empty:
        return pd.DataFrame()

    source_df = source_df.merge(
        neighbor_df,
        how="left",
        left_index=True,
        right_on="cell_id",
        suffixes=("", "__neighbor"),
    )

    if f"{image_key}__neighbor" in source_df.columns:
        source_df = source_df.drop(columns=[f"{image_key}__neighbor"])

    rows = []
    neighbor_cols = [f"neighbor_{agg}__{f}" for f in target_feature_keys]

    for src_feat in source_feature_keys:
        for tgt_feat, tgt_col in zip(target_feature_keys, neighbor_cols):
            for img in source_df[image_key].dropna().unique():
                df_img = source_df[source_df[image_key] == img]
                coords = df_img[[x_key, y_key]].to_numpy()
                x_values = df_img[src_feat].to_numpy()
                y_values = df_img[tgt_col].to_numpy()

                rows.append({
                    image_key: img,
                    "source_phenotype": source_phenotype,
                    "target_phenotype": target_phenotype,
                    "source_feature": src_feat,
                    "target_feature": tgt_feat,
                    "target_summary_feature": tgt_col,
                    "cross_morans_i": cross_morans_i(
                        coords,
                        x_values,
                        y_values,
                        k=k,
                    ),
                    "n_source_cells": len(df_img),
                    "n_nonmissing_pairs": int(np.sum(np.isfinite(x_values) & np.isfinite(y_values))),
                })

    return pd.DataFrame(rows)


def add_local_cross_morans_i_between_phenotypes(
    adata: ad.AnnData,
    phenotype_key: str,
    source_phenotype: str,
    target_phenotype: str,
    source_feature_key: str,
    target_feature_key: str,
    radius: float=None,
    k_neighbors: int=None,
    agg: str="mean",
    out_key: str=None,
    neighbor_feature_key: str=None,
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    image_key: str="imageid",
    k: int=8,
) -> ad.AnnData:
    """
    Compute local cross Moran's I between a source-cell feature and a
    source-centered neighborhood summary of a target-cell feature.

    This is a one-step wrapper around:
    1. ``summarize_target_features_around_source_cells(...)``
    2. merging the neighborhood summary onto source cells
    3. ``add_local_cross_morans_i(...)``

    Returns
    -------
    anndata.AnnData
        AnnData containing only source-phenotype cells, with both the
        summarized target feature and local cross Moran's I added to ``obs``.
    """
    if source_feature_key not in adata.obs.columns:
        raise ValueError(f"Source feature {source_feature_key!r} not found in adata.obs.")
    if target_feature_key not in adata.obs.columns:
        raise ValueError(f"Target feature {target_feature_key!r} not found in adata.obs.")

    neighbor_df = summarize_target_features_around_source_cells(
        adata=adata,
        phenotype_key=phenotype_key,
        source_phenotype=source_phenotype,
        target_phenotype=target_phenotype,
        target_feature_keys=[target_feature_key],
        radius=radius,
        k_neighbors=k_neighbors,
        agg=agg,
        x_key=x_key,
        y_key=y_key,
        image_key=image_key,
        source_only=True,
    )

    source_adata = adata[adata.obs[phenotype_key] == source_phenotype].copy()
    if source_adata.n_obs == 0:
        return source_adata

    default_neighbor_key = f"neighbor_{agg}__{target_feature_key}"
    if neighbor_feature_key is None:
        neighbor_feature_key = default_neighbor_key

    if not neighbor_df.empty:
        merge_df = neighbor_df[["cell_id", default_neighbor_key]].rename(
            columns={default_neighbor_key: neighbor_feature_key}
        )
        merge_df = merge_df.set_index("cell_id")
        source_adata.obs = source_adata.obs.join(merge_df, how="left")
    else:
        source_adata.obs[neighbor_feature_key] = np.nan

    if out_key is None:
        out_key = (
            f"local_cross_morans_i__{source_feature_key}"
            f"__{source_phenotype}__vs__{target_phenotype}"
            f"__{neighbor_feature_key}"
        )

    source_adata = add_local_cross_morans_i(
        source_adata,
        x_value_key=source_feature_key,
        y_value_key=neighbor_feature_key,
        out_key=out_key,
        x_key=x_key,
        y_key=y_key,
        image_key=image_key,
        k=k,
    )

    return source_adata


__all__ = [
    "ripleys_k",
    "ripleys_curve",
    "ripley_envelope",
    "ripleys_k_by_image",
    "ripleys_k_by_phenotype",
    "cross_ripleys_k",
    "cross_ripleys_k_by_phenotype",
    "cross_ripleys_k_all_pairs",
    "cross_ripleys_curve",
    "cross_ripley_envelope",
    "cross_ripleys_curve_by_phenotype",
    "cross_ripley_envelope_by_phenotype",
    "cross_ripley_permutation_envelope",
    "ripley_local_counts_by_phenotype",
    "cross_ripley_local_counts",
    "ripley_interaction_scale",
    "ripley_spatial_scales",
    "morans_i",
    "morans_i_permutation_test",
    "morans_i_by_image",
    "morans_i_by_image_permutation_test",
    "local_morans_i",
    "add_local_morans_i",
    "classify_local_morans_i",
    "add_local_morans_i_quadrants",
    "cross_morans_i",
    "cross_morans_i_by_image",
    "cross_morans_i_permutation_test",
    "cross_morans_i_by_image_permutation_test",
    "local_cross_morans_i",
    "add_local_cross_morans_i",
    "classify_local_cross_morans_i",
    "add_local_cross_morans_i_quadrants",
    "summarize_target_features_around_source_cells",
    "cross_morans_i_feature_matrix",
    "add_local_cross_morans_i_between_phenotypes",
]
