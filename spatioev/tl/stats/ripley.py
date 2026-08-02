"""Ripley's K and derived cross-type spatial statistics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from ._helpers import (
    _clean_coords,
    _clean_paired_coords,
    _permute_values_within_groups,
    _random_points_in_hull,
    _resolve_window_area,
)

if TYPE_CHECKING:  # pragma: no cover
    import anndata as ad


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
