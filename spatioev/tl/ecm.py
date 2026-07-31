# ============================================================
# Section 1: ECM cell-fiber links  (from archive/spatial/spatial_ecm_links.py)
# ============================================================

"""
Spatial ECM Link Construction
=============================

Builds reusable spatial relationship tables linking cells and ECM fibers.

Relationship table structure
----------------------------

imageid
cell_id
fiber_id
distance

One cell may link to many fibers.
One fiber may link to many cells.

This table can be reused across all ECM spatial statistics.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import anndata as ad


import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

# ------------------------------------------------
# Radius-based adjacency map
# ------------------------------------------------

def build_cell_fiber_links(
    adata: ad.AnnData,
    fiber_df: pd.DataFrame,
    radius: float=50,
    cell_x: str="X_centroid",
    cell_y: str="Y_centroid",
    fiber_x: str="X_centroid",
    fiber_y: str="Y_centroid",
    image_key: str="imageid",
    fiber_image_key: str="imageid",
) -> pd.DataFrame:
    """
    Build bidirectional adjacency table linking cells and fibers.

    Returns
    -------
    DataFrame

        imageid
        cell_id
        fiber_id
        distance
    """

    rows = []

    images = np.intersect1d(
        adata.obs[image_key].unique(),
        fiber_df[fiber_image_key].unique(),
    )

    for img in images:

        cells = adata.obs[adata.obs[image_key] == img]
        fibers = fiber_df[fiber_df[fiber_image_key] == img]

        if cells.empty or fibers.empty:
            continue

        cell_coords = cells[[cell_x, cell_y]].to_numpy()
        fiber_coords = fibers[[fiber_x, fiber_y]].to_numpy()

        tree = BallTree(cell_coords)

        neighbors = tree.query_radius(fiber_coords, r=radius)

        for i, idxs in enumerate(neighbors):

            if len(idxs) == 0:
                continue

            fiber_id = fibers.index[i]
            fiber_point = fiber_coords[i]

            dists = np.linalg.norm(cell_coords[idxs] - fiber_point, axis=1)

            for j, dist in zip(idxs, dists):

                rows.append(
                    {
                        "imageid": img,
                        "cell_id": cells.index[j],
                        "fiber_id": fiber_id,
                        "distance": dist,
                    }
                )

    return pd.DataFrame(rows)



# ------------------------------------------------
# Nearest cell-fiber mapping
# ------------------------------------------------

def build_nearest_cell_fiber_map(
    adata: ad.AnnData,
    fiber_df: pd.DataFrame,
    cell_x: str="X_centroid",
    cell_y: str="Y_centroid",
    fiber_x: str="X_centroid",
    fiber_y: str="Y_centroid",
    image_key: str="imageid",
    fiber_image_key: str="imageid",
) -> pd.DataFrame:
    """
    Compute nearest fiber for every cell.

    Returns
    -------
    DataFrame

        imageid
        cell_id
        fiber_id
        distance
    """

    rows = []

    images = np.intersect1d(
        adata.obs[image_key].unique(),
        fiber_df[fiber_image_key].unique(),
    )

    for img in images:

        cells = adata.obs[adata.obs[image_key] == img]
        fibers = fiber_df[fiber_df[fiber_image_key] == img]

        if cells.empty or fibers.empty:
            continue

        cell_coords = cells[[cell_x, cell_y]].to_numpy()
        fiber_coords = fibers[[fiber_x, fiber_y]].to_numpy()

        tree = BallTree(fiber_coords)

        dist, idx = tree.query(cell_coords, k=1)

        for i in range(len(cells)):

            rows.append(
                {
                    "imageid": img,
                    "cell_id": cells.index[i],
                    "fiber_id": fibers.index[idx[i, 0]],
                    "distance": dist[i, 0],
                }
            )

    return pd.DataFrame(rows)



# ============================================================
# Section 2: ECM spatial statistics  (from archive/spatial/spatial_ecm_stats.py)
# ============================================================

"""
Spatial ECM Statistics Module
=============================

This module implements spatial statistics linking extracellular matrix (ECM)
fibers with cells in multiplexed imaging datasets.

The goal is to quantify how ECM structure influences cell behaviour,
immune infiltration, tumor invasion, and stromal remodeling.

Inputs
------

Cells (AnnData):
    adata.obs must contain:
        X_centroid
        Y_centroid
        orientation (optional)
        phenotype (optional)
        imageid

Fibers (DataFrame):
    fiber_df must contain:
        X_centroid
        Y_centroid
        orientation
        major_axis_length
        minor_axis_length
        area
        eccentricity
        alignment_score
        fiber_type (optional)
        imageid


Spatial analyses implemented
----------------------------

1. Cell → fiber proximity
2. Local ECM density near cells
3. Cross Ripley’s K curve (cell–ECM interaction scale)
4. Moran’s I (ECM structural clustering)
5. Cross Moran’s I (ECM–cell coupling)
6. Local cross Moran’s I (spatial map of ECM–cell interaction)
7. Spatial regression models
8. ECM vector field representation
9. Cell–fiber orientation alignment
"""



from sklearn.linear_model import LinearRegression
from sklearn.neighbors import kneighbors_graph

from .preprocessing import compute_convex_hull_area


# ============================================================
# Helper functions
# ============================================================
def _filter_fibers(
    fiber_df,
    fiber_type_key,
    fiber_type,
):
    fiber_type = _ensure_list(fiber_type)

    if fiber_type is None:
        return fiber_df

    return fiber_df[
        fiber_df[fiber_type_key].isin(fiber_type)
    ]

def _filter_cells(
    adata,
    phenotype_key,
    phenotype,
):
    phenotype = _ensure_list(phenotype)

    cells = adata.obs

    if phenotype is None:
        return cells

    if phenotype_key is None:
        raise ValueError(
            "phenotype_key must be provided when phenotype filtering is used."
        )

    return cells[cells[phenotype_key].isin(phenotype)]


def _subset_links(
    links_df, 
    cells, 
    fibers,
    cell_image_key="imageid",
    fiber_image_key="imageid",
    link_image_key="imageid",
):
    """
    CRITICAL FIX:
    Ensures links are restricted to same image + valid ids
    """
    valid_images = np.intersect1d(
        cells[cell_image_key].unique(),
        fibers[fiber_image_key].unique()
    )

    links = links_df.copy()

    links = links[links[link_image_key].isin(valid_images)]
    links = links[links["cell_id"].isin(cells.index)]
    links = links[links["fiber_id"].isin(fibers.index)]

    return links


def _ensure_list(value):
    """
    Normalize a scalar or iterable label selection to a list.
    """
    if value is None:
        return None

    if isinstance(value, str):
        return [value]

    return list(value)


def _label_suffix(value, default):
    """
    Build a stable suffix for derived column names.
    """
    value = _ensure_list(value)

    if value is None or len(value) == 0:
        return default

    return "_".join(map(str, value))


def _clean_coords(coords):
    """
    Drop rows with non-finite x/y coordinates.
    """
    coords = np.asarray(coords, dtype=float)

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("Coordinates must be an Nx2 array.")

    return coords[np.isfinite(coords).all(axis=1)]


def _resolve_k(n_obs, k):
    """
    Ensure kNN graph construction uses a valid number of neighbors.
    """
    if n_obs < 2:
        return None

    return min(k, n_obs - 1)


def _cross_ripleys_curve_from_links(
    cells,
    fibers,
    links,
    radii,
    cell_x="X_centroid",
    cell_y="Y_centroid",
    fiber_x="X_centroid",
    fiber_y="Y_centroid",
    check_radius=True,
):
    """
    Compute cross Ripley's K/L curve from pre-filtered cells, fibers, and links.
    """
    if cells.empty or fibers.empty:
        return pd.DataFrame()

    cell_valid = cells[[cell_x, cell_y]].notna().all(axis=1)
    fiber_valid = fibers[[fiber_x, fiber_y]].notna().all(axis=1)

    cells = cells.loc[cell_valid]
    fibers = fibers.loc[fiber_valid]
    links = _subset_links(links, cells, fibers)

    if cells.empty or fibers.empty:
        return pd.DataFrame()

    cell_coords = _clean_coords(cells[[cell_x, cell_y]].to_numpy())
    fiber_coords = _clean_coords(fibers[[fiber_x, fiber_y]].to_numpy())

    area = compute_convex_hull_area(
        np.vstack([cell_coords, fiber_coords])
    )

    if np.isnan(area):
        return pd.DataFrame()

    if links.empty:
        if check_radius:
            raise ValueError(
                "No cell-fiber links found. Increase link radius to cover the requested radii."
            )

        counts = np.zeros(len(radii), dtype=float)
    else:
        if check_radius and max(radii) > links["distance"].max():
            raise ValueError("Increase link radius to cover all radii")

        counts = np.array(
            [(links["distance"] <= r).sum() for r in radii],
            dtype=float,
        )

    n_cells = len(cells)
    n_fibers = len(fibers)

    K_vals = (area / (n_cells * n_fibers)) * counts
    L_vals = np.sqrt(K_vals / np.pi)

    return pd.DataFrame({
        "radius": radii,
        "K": K_vals,
        "L": L_vals,
        "L_minus_r": L_vals - radii,
    })


def _permute_mask_within_images(mask, image_ids, rng):
    """
    Permute a boolean selection mask independently within each image.
    """
    mask = np.asarray(mask, dtype=bool)
    image_ids = np.asarray(image_ids)

    permuted = np.zeros_like(mask, dtype=bool)

    for img in pd.unique(image_ids):
        idx = np.where(image_ids == img)[0]
        permuted[idx] = rng.permutation(mask[idx])

    return permuted


def _permute_values_within_images(values, image_ids, rng):
    """
    Permute feature values independently within each image.
    """
    values = np.asarray(values)
    image_ids = np.asarray(image_ids)

    permuted = values.copy()

    for img in pd.unique(image_ids):
        idx = np.where(image_ids == img)[0]
        permuted[idx] = rng.permutation(values[idx])

    return permuted


def _map_cell_feature_to_fibers(
    adata,
    fiber_df,
    links_df,
    cell_feature,
    cell_values=None,
    image_key="imageid",
    fiber_image_key="imageid",
    weight_scale=50,
):
    """
    Aggregate a cell feature onto fibers using distance-weighted links.
    """
    valid_images = np.intersect1d(
        adata.obs[image_key].unique(),
        fiber_df[fiber_image_key].unique(),
    )

    cells = adata.obs[adata.obs[image_key].isin(valid_images)]
    fibers = fiber_df[fiber_df[fiber_image_key].isin(valid_images)]
    links = _subset_links(
        links_df,
        cells,
        fibers,
        cell_image_key=image_key,
        fiber_image_key=fiber_image_key,
        link_image_key="imageid",
    )

    if cell_values is None:
        cell_values = adata.obs[cell_feature]

    cell_values = pd.Series(cell_values, index=adata.obs.index, name=cell_feature)

    merged = links.merge(
        cell_values.to_frame(),
        left_on="cell_id",
        right_index=True,
    )

    merged = merged[np.isfinite(merged[cell_feature])]

    if merged.empty:
        return pd.Series(dtype=float)

    merged["weight"] = np.exp(-merged["distance"] / weight_scale)

    return merged.groupby("fiber_id")[[cell_feature, "weight"]].apply(
        lambda x: np.average(x[cell_feature], weights=x["weight"])
    )


def _compute_cross_morans_i_from_fiber_table(
    fiber_df,
    feature_fiber,
    feature_cell,
    k=8,
    fiber_x="X_centroid",
    fiber_y="Y_centroid",
):
    """
    Compute global cross Moran's I from fiber-level features.
    """
    valid = fiber_df[[fiber_x, fiber_y, feature_fiber, feature_cell]].dropna()

    coords = fiber_df.loc[valid.index, [fiber_x, fiber_y]].to_numpy()
    x = valid[feature_fiber].to_numpy(dtype=float)
    y = valid[feature_cell].to_numpy(dtype=float)

    n = len(x)

    if n < 3:
        return np.nan

    k_eff = _resolve_k(n, k)

    if k_eff is None:
        return np.nan

    W = kneighbors_graph(
        coords,
        k_eff,
        mode="connectivity",
        include_self=False,
    ).toarray()

    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W = W / row_sums

    x = x - x.mean()
    y = y - y.mean()

    denom = np.sqrt(np.sum(x**2) * np.sum(y**2))

    if denom == 0:
        return np.nan

    return (n / W.sum()) * (
        np.sum(W * np.outer(x, y)) / denom
    )


# ============================================================
# 1. Cell → ECM proximity
# ============================================================
def cell_to_fiber_distance(
    adata: ad.AnnData,
    fiber_df: pd.DataFrame,
    links_df: pd.DataFrame,
    fiber_type_key: str="fiber_type",
    fiber_type: str=None,
    phenotype_key: str=None,
    phenotype: str=None,
    summary_phenotype_key: str=None,
) -> object:
    """
    Compute distance from cells to nearest ECM fiber.
    
    This function quantifies how close each cell lies to ECM structures.
    It can optionally restrict the analysis to specific fiber types
    (e.g. collagen, fibronectin) and/or specific cell phenotypes.
    
    Biological applications
    -----------------------
    Immune exclusion
        CD8 T cells far from collagen bundles → collagen barrier

    Tumor invasion tracks
        tumor cells near aligned collagen fibers → migration tracks

    Fibroblast niches
        CAFs near fibronectin → ECM remodeling zones

    Parameters
    ----------
    fiber_type_key : str
        Column in ``fiber_df`` containing fiber-class labels.
    fiber_type
        Fiber label or list of labels to keep. Use ``None`` to keep all fibers.

    phenotype_key : str, optional
        Column in ``adata.obs`` containing cell phenotype labels.
    phenotype
        Cell phenotype label or list of labels to keep. Use ``None`` to keep all cells.

    summary_phenotype_key : str, optional
        If provided, return median distance grouped by this cell-level column.
    """

    fibers = _filter_fibers(fiber_df, 
                            fiber_type_key, 
                            fiber_type)
    
    cells = _filter_cells(adata, 
                          phenotype_key, 
                          phenotype)

    links = _subset_links(links_df, 
                          cells, 
                          fibers)

    nearest = (
        links.sort_values("distance")
        .groupby("cell_id")
        .first()
    )

    col = f"dist_to_{_label_suffix(fiber_type, 'fiber')}"
    adata.obs[col] = np.nan

    adata.obs.loc[nearest.index, col] = nearest["distance"]

    if summary_phenotype_key:
        return (
            adata.obs.groupby(summary_phenotype_key)[col]
            .median()
            .reset_index()
        )

    return adata



# ============================================================
# 2. ECM density around cells
# ============================================================
def fiber_density_near_cells(
    adata: ad.AnnData,
    fiber_df: pd.DataFrame,
    links_df: pd.DataFrame,
    fiber_type_key: str="fiber_type",
    fiber_type: str=None,
    phenotype_key: str=None,
    phenotype: str=None,
    normalize: bool=False,
    density_radius: float=50,
) -> object:
    """
    Count ECM fibers within radius of each cell.

    Biological application
    ----------------------
    Estimates ECM density experienced by each cell.

    Allows optional filtering by ECM fiber type and/or cell phenotype.

    Examples
    --------
    collagen density around CD8 T cells
    fibronectin density around tumor cells
    total ECM density around macrophages

    Interpretation
    --------------
    high density
        fibrotic region
        tumor stroma

    low density
        open parenchyma
        immune infiltration zones

    Parameters
    ----------
    fiber_type_key : str
        Column in ``fiber_df`` containing fiber-class labels.
    fiber_type : str or list, optional
        Fiber label or list of labels to keep. Use ``None`` to keep all fibers.
    phenotype_key : str, optional
        Column in ``adata.obs`` containing cell phenotype labels.
    phenotype : str or list, optional
        Cell phenotype label or list of labels to keep. Use ``None`` to keep all cells.
    normalize : bool
        If ``True``, divide counts by the area of a circle with radius
        ``density_radius``.
    density_radius : float
        Radius used for density normalization in the same units as coordinates.
        This should usually match the radius used to build ``links_df``.
    """

    fibers = _filter_fibers(fiber_df, 
                            fiber_type_key, 
                            fiber_type)
    
    cells = _filter_cells(adata, 
                          phenotype_key, 
                          phenotype)

    links = _subset_links(links_df, 
                          cells, 
                          fibers)

    density = links.groupby("cell_id").size()

    if normalize:
        density = density / (np.pi * density_radius**2)

    col = f"{_label_suffix(fiber_type, 'fiber')}_density"
    adata.obs[col] = np.nan

    adata.obs.loc[density.index, col] = density

    return adata


# ============================================================
# 3. Cross Ripley’s K curve
# ============================================================
def cross_ripleys_k(
    adata: ad.AnnData,
    fiber_df: pd.DataFrame,
    links_df: pd.DataFrame,
    radii: np.ndarray | list[float],
    fiber_type_key: str="fiber_type",
    fiber_type: str=None,
    phenotype_key: str=None,
    phenotype: str=None,
    check_radius: float=True,
) -> float:
    """
    Cross Ripley's K statistic between cells and ECM fibers.

    Biological application
    ----------------------
    Quantifies spatial attraction or repulsion between cells
    and ECM fibers across spatial scales.

    Examples
    --------
    CD8 vs collagen → immune exclusion
    tumor vs collagen → invasion tracks
    macrophage vs ECM → stromal niches

    Returns
    -------
    DataFrame
        radius
        K
        L
        L_minus_r

    Parameters
    ----------
    radii : array-like
        Distances at which the cell-fiber interaction curve is evaluated.
    fiber_type_key : str
        Column in ``fiber_df`` containing fiber-class labels.
    fiber_type : str or list, optional
        Fiber label or list of labels to keep. Use ``None`` to keep all fibers.
    phenotype_key : str, optional
        Column in ``adata.obs`` containing cell phenotype labels.
    phenotype : str or list, optional
        Cell phenotype label or list of labels to keep. Use ``None`` to keep all cells.
    check_radius : bool
        If ``True``, raise when the filtered link table does not contain links out to
        the largest requested radius. Set to ``False`` when you know ``links_df`` was
        built with a radius that covers ``radii``.
    """

    fibers = _filter_fibers(fiber_df, 
                            fiber_type_key, 
                            fiber_type)
    
    cells = _filter_cells(adata, 
                          phenotype_key, 
                          phenotype)

    links = _subset_links(links_df, 
                          cells, 
                          fibers)

    return _cross_ripleys_curve_from_links(
        cells,
        fibers,
        links,
        radii,
        check_radius=check_radius,
    )


def cross_ripleys_k_permutation_envelope(
    adata: ad.AnnData,
    fiber_df: pd.DataFrame,
    links_df: pd.DataFrame,
    radii: np.ndarray | list[float],
    fiber_type_key: str="fiber_type",
    fiber_type: str=None,
    phenotype_key: str=None,
    phenotype: str=None,
    permute: bool="cells",
    n_sim: int=199,
    image_key: str="imageid",
    fiber_image_key: str="imageid",
    random_state: int=None,
    check_radius: float=True,
) -> pd.DataFrame:
    """
    Permutation envelope for cell-ECM cross Ripley's K.

    Preserves cell and fiber coordinates while shuffling labels within each image.

    Parameters
    ----------
    radii : array-like
        Distances at which the cell-fiber interaction curve is evaluated.
    fiber_type_key : str
        Column in ``fiber_df`` containing fiber-class labels.
    fiber_type : str or list, optional
        Fiber label or list of labels to keep when defining the observed subset.
    phenotype_key : str, optional
        Column in ``adata.obs`` containing cell phenotype labels.
    phenotype : str or list, optional
        Cell phenotype label or list of labels to keep when defining the observed subset.
    permute : {"cells", "fibers"}
        Which labels to randomize for the null model.
    image_key : str
        Column in ``adata.obs`` identifying which image each cell belongs to.
    fiber_image_key : str
        Column in ``fiber_df`` identifying which image each fiber belongs to.
    check_radius : bool
        If ``True``, raise when the observed filtered link table does not contain
        links out to the largest requested radius. Set to ``False`` when ``links_df``
        was built with a known radius that covers ``radii``.

    Returns
    -------
    DataFrame
        radius
        K
        L
        L_minus_r
        envelope_low
        envelope_high
    """
    if permute not in {"cells", "fibers"}:
        raise ValueError("permute must be either 'cells' or 'fibers'")

    fibers = _filter_fibers(fiber_df,
                            fiber_type_key,
                            fiber_type)

    cells = _filter_cells(adata,
                          phenotype_key,
                          phenotype)

    links = _subset_links(links_df,
                          cells,
                          fibers)

    observed = _cross_ripleys_curve_from_links(
        cells,
        fibers,
        links,
        radii,
        check_radius=check_radius,
    )

    if observed.empty:
        return observed

    rng = np.random.default_rng(random_state)
    sims = []

    if permute == "cells":
        if phenotype_key is None or phenotype is None:
            raise ValueError(
                "phenotype_key and phenotype must be provided when permute='cells'."
            )

        phenotype = _ensure_list(phenotype)

        cell_mask = adata.obs[phenotype_key].isin(phenotype).to_numpy()
        cell_images = adata.obs[image_key].to_numpy()

        for i in range(n_sim):
            permuted_mask = _permute_mask_within_images(
                cell_mask,
                cell_images,
                rng,
            )

            sim_cells = adata.obs.loc[permuted_mask]
            sim_links = _subset_links(
                links_df,
                sim_cells,
                fibers,
                cell_image_key=image_key,
                fiber_image_key=fiber_image_key,
                link_image_key="imageid",
            )
            

            sim_curve = _cross_ripleys_curve_from_links(
                sim_cells,
                fibers,
                sim_links,
                radii,
                check_radius=False,
            )

            if sim_curve.empty:
                continue

            sims.append(sim_curve["L_minus_r"].values)

    else:
        if fiber_type is None:
            raise ValueError(
                "fiber_type must be provided when permute='fibers'."
            )

        fiber_type = _ensure_list(fiber_type)

        fiber_mask = fiber_df[fiber_type_key].isin(fiber_type).to_numpy()
        fiber_images = fiber_df[fiber_image_key].to_numpy()

        for i in range(n_sim):
            permuted_mask = _permute_mask_within_images(
                fiber_mask,
                fiber_images,
                rng,
            )

            sim_fibers = fiber_df.loc[permuted_mask]
            sim_links = _subset_links(
                links_df,
                cells,
                sim_fibers,
                cell_image_key=image_key,
                fiber_image_key=fiber_image_key,
                link_image_key="imageid",
            )

            sim_curve = _cross_ripleys_curve_from_links(
                cells,
                sim_fibers,
                sim_links,
                radii,
                check_radius=False,
            )

            if sim_curve.empty:
                continue

            sims.append(sim_curve["L_minus_r"].values)

    if len(sims) == 0:
        raise ValueError(
            "All permutation simulations were empty. Check the selected labels and link table."
        )

    sims = np.array(sims)

    observed["envelope_low"] = np.percentile(sims, 2.5, axis=0)
    observed["envelope_high"] = np.percentile(sims, 97.5, axis=0)

    return observed


# ============================================================
# 4. Moran’s I for ECM structure
# ============================================================

def morans_i_fibers(
    fiber_df: pd.DataFrame,
    feature: str,
    k: int=8,
    fiber_x: str="X_centroid",
    fiber_y: str="Y_centroid",
    fiber_type_key: str="fiber_type",
    fiber_type: str=None,
) -> pd.DataFrame:
    """
    Spatial autocorrelation of ECM features.

    Biological application
    ----------------------
    Detects spatial clustering of ECM morphology or architecture.

    Example questions
    -----------------
    Do long collagen fibers cluster spatially?
    Do aligned fibers form stromal invasion tracks?
    Do dense ECM bundles form fibrotic niches?

    Parameters
    ----------
    fiber_df : DataFrame
        ECM fiber dataset

    feature : str
        Fiber feature to analyze
        Example:
            major_axis_length
            eccentricity
            alignment_score

    k : int
        Number of nearest neighbors used to construct spatial graph

    fiber_x, fiber_y : str
        Column names in ``fiber_df`` containing fiber coordinates.

    fiber_type : str or list, optional
        Restrict analysis to specific fiber types
        Example:
            "collagen"
            ["collagen", "fibronectin"]

    Returns
    -------
    float
        Moran's I spatial autocorrelation statistic

    Interpretation
    --------------
    Moran's I > 0
        clustering of similar ECM features
        stromal remodeling zones

    Moran's I ≈ 0
        random ECM organization

    Moran's I < 0
        spatial dispersion of ECM features
    """

    fiber_subset = fiber_df.copy()

    # --------------------------------------------------
    # Filter fiber types
    # --------------------------------------------------

    fiber_type = _ensure_list(fiber_type)

    if fiber_type is not None:

        fiber_subset = fiber_subset[
            fiber_subset[fiber_type_key].isin(fiber_type)
        ]

    if fiber_subset.empty:
        raise ValueError("No fibers found for specified fiber_type")

    # --------------------------------------------------
    # Extract coordinates and feature
    # --------------------------------------------------

    valid = (
        fiber_subset[[fiber_x, fiber_y, feature]]
        .notna()
        .all(axis=1)
    )

    fiber_subset = fiber_subset[valid]

    coords = fiber_subset[[fiber_x, fiber_y]].to_numpy()
    values = fiber_subset[feature].to_numpy()

    if len(values) < 3:
        return np.nan

    k_eff = _resolve_k(len(values), k)

    if k_eff is None:
        return np.nan

    # --------------------------------------------------
    # Spatial weights (kNN graph)
    # --------------------------------------------------

    W = kneighbors_graph(coords, k_eff, mode="connectivity").toarray()

    # NORMALIZATION FIX
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W = W / row_sums

    # --------------------------------------------------
    # Moran's I calculation
    # --------------------------------------------------

    x = values - values.mean()

    denom = np.sum(x**2)

    if denom == 0:
        return np.nan

    return (len(x) / W.sum()) * (
        np.sum(W * np.outer(x, x)) / denom
    )


# ============================================================
# 5. Local Moran’s I for ECM features
# ============================================================
def local_morans_i_fibers(
    fiber_df: pd.DataFrame,
    feature: str,
    k: int=8,
    fiber_x: str="X_centroid",
    fiber_y: str="Y_centroid",
    fiber_type_key: str="fiber_type",
    fiber_type: str=None,
) -> np.ndarray:
    """
    Local Moran's I spatial autocorrelation for ECM features.

    Computes a spatial clustering statistic for each fiber.

    Biological application
    ----------------------
    Detects spatial hotspots of ECM remodeling.

    Examples
    --------
    aligned collagen tracks
    fibrotic ECM bundles
    stromal remodeling zones near tumors

    Parameters
    ----------
    fiber_df : DataFrame
        ECM fiber dataset

    feature : str
        Fiber feature to analyze
        Example:
            major_axis_length
            eccentricity
            alignment_score

    k : int
        Number of nearest neighbors

    fiber_x, fiber_y : str
        Column names in ``fiber_df`` containing fiber coordinates.

    fiber_type : str or list, optional
        Restrict analysis to specific fiber types

    Returns
    -------
    DataFrame
        Original dataframe with added column:
            local_moran_<feature>

    Interpretation
    --------------
    high positive values
        clusters of similar ECM features

    negative values
        spatial contrast / boundary regions
    """

    fiber_subset = fiber_df.copy()

    # --------------------------------------------------
    # Filter fiber types
    # --------------------------------------------------

    fiber_type = _ensure_list(fiber_type)

    if fiber_type is not None:

        fiber_subset = fiber_subset[
            fiber_subset[fiber_type_key].isin(fiber_type)
        ]

    if fiber_subset.empty:
        raise ValueError("No fibers found for specified fiber_type")

    # --------------------------------------------------
    # Remove missing values
    # --------------------------------------------------

    valid = (
        fiber_subset[[fiber_x, fiber_y, feature]]
        .notna()
        .all(axis=1)
    )

    fiber_subset = fiber_subset[valid]

    coords = fiber_subset[[fiber_x, fiber_y]].to_numpy()
    values = fiber_subset[feature].to_numpy()

    n = len(values)

    if n < 3:
        result = fiber_df.copy()
        result[f"local_moran_{feature}"] = np.nan
        return result

    k_eff = _resolve_k(n, k)

    if k_eff is None:
        result = fiber_df.copy()
        result[f"local_moran_{feature}"] = np.nan
        return result

    # --------------------------------------------------
    # Build spatial weights
    # --------------------------------------------------

    W = kneighbors_graph(
        coords,
        k_eff,
        mode="connectivity",
        include_self=False
    )

    W = W.toarray()

    # --------------------------------------------------
    # Center feature values
    # --------------------------------------------------

    x = values - values.mean()

    m2 = np.sum(x**2) / n

    if m2 == 0:
        result = fiber_df.copy()
        result[f"local_moran_{feature}"] = np.nan
        return result

    # --------------------------------------------------
    # Compute Local Moran's I
    # --------------------------------------------------

    local_I = x * (W @ x) / m2

    # --------------------------------------------------
    # Store result
    # --------------------------------------------------

    result = fiber_df.copy()

    result.loc[fiber_subset.index, f"local_moran_{feature}"] = local_I

    return result


# ============================================================
# 6. Cross Moran’s I (ECM–cell coupling)
# ============================================================
def cross_morans_i_ecm_cells(
    adata: ad.AnnData,
    fiber_df: pd.DataFrame,
    links_df: pd.DataFrame,
    feature_fiber: str,
    cell_feature: str,
    k: int=8,
) -> pd.DataFrame:
    
    """
    Cross Moran's I between ECM features and nearby cell features.

    Automatically maps cell features from adata.obs onto fiber locations
    using spatial proximity.

    Biological application
    ----------------------
    Detects spatial coupling between ECM architecture and nearby
    cellular niches.

    Example questions
    -----------------
    Do aligned collagen fibers occur near tumor cells?
    Do thick ECM bundles occur near macrophage clusters?

    Parameters
    ----------
    feature_fiber : str
        Column in ``fiber_df`` containing the ECM feature to analyze.
    cell_feature : str
        Column in ``adata.obs`` containing the cell feature to aggregate onto fibers.
    k : int
        Number of nearest neighbors used to define the fiber-level spatial graph.

    Returns
    -------
    float
        cross Moran's I statistic
    """
    fiber_cell = _map_cell_feature_to_fibers(
        adata,
        fiber_df,
        links_df,
        cell_feature,
    )

    fiber_local = fiber_df.copy()
    fiber_local["cell_local"] = fiber_local.index.map(fiber_cell)

    return _compute_cross_morans_i_from_fiber_table(
        fiber_local,
        feature_fiber,
        "cell_local",
        k=k,
    )


def cross_morans_i_ecm_cells_permutation_test(
    adata: ad.AnnData,
    fiber_df: pd.DataFrame,
    links_df: pd.DataFrame,
    feature_fiber: str,
    cell_feature: str,
    k: int=8,
    n_sim: int=999,
    image_key: str="imageid",
    random_state: int=None,
) -> pd.DataFrame:
    """
    Permutation test for global cross Moran's I between ECM and cell features.

    Preserves cell and fiber coordinates while shuffling the cell feature
    within each image.

    Parameters
    ----------
    feature_fiber : str
        Column in ``fiber_df`` containing the ECM feature to analyze.
    cell_feature : str
        Column in ``adata.obs`` containing the cell feature to aggregate onto fibers.
    k : int
        Number of nearest neighbors used to define the fiber-level spatial graph.
    image_key : str
        Column in ``adata.obs`` identifying which image each cell belongs to.

    Returns
    -------
    dict
        observed
        p_value
        null_mean
        null_std
        z_score
        n_sim
    """
    observed = cross_morans_i_ecm_cells(
        adata,
        fiber_df,
        links_df,
        feature_fiber,
        cell_feature,
        k=k,
    )

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
    cell_values = adata.obs[cell_feature].to_numpy()
    cell_images = adata.obs[image_key].to_numpy()
    sims = []

    for i in range(n_sim):
        permuted = _permute_values_within_images(
            cell_values,
            cell_images,
            rng,
        )

        fiber_cell = _map_cell_feature_to_fibers(
            adata,
            fiber_df,
            links_df,
            cell_feature,
            cell_values=permuted,
            image_key=image_key,
        )

        fiber_local = fiber_df.copy()
        fiber_local["cell_local"] = fiber_local.index.map(fiber_cell)

        sim_stat = _compute_cross_morans_i_from_fiber_table(
            fiber_local,
            feature_fiber,
            "cell_local",
            k=k,
        )

        if np.isfinite(sim_stat):
            sims.append(sim_stat)

    if len(sims) == 0:
        raise ValueError(
            "All permutation simulations returned invalid statistics."
        )

    sims = np.asarray(sims, dtype=float)
    null_mean = sims.mean()
    null_std = sims.std(ddof=1) if len(sims) > 1 else 0.0

    # Two-sided Monte Carlo p-value with +1 correction.
    p_value = (np.sum(np.abs(sims) >= abs(observed)) + 1) / (len(sims) + 1)

    if null_std == 0:
        z_score = np.nan
    else:
        z_score = (observed - null_mean) / null_std

    return {
        "observed": observed,
        "p_value": p_value,
        "null_mean": null_mean,
        "null_std": null_std,
        "z_score": z_score,
        "n_sim": len(sims),
    }



# helper function for local cross Moran's I
def map_cells_to_fibers(
    adata: ad.AnnData,
    fiber_df: pd.DataFrame,
    links_df: pd.DataFrame,
    phenotype_key: str,
    phenotype: str,
) -> pd.DataFrame:
    """
    Map nearby cell features onto ECM fiber locations.

    Computes spatial relationships between cells and fibers.

    Generated features
    ------------------
    <phenotype>_density
        number of cells within radius of each fiber

    <phenotype>_presence
        whether any cells exist near fiber

    <phenotype>_mean_distance
        mean distance to nearby cells

    Biological applications
    -----------------------
    tumor_density
        tumor invasion tracks

    CD8_density
        immune infiltration

    macrophage_density
        stromal immune niches

    Parameters
    ----------
    phenotype_key : str
        Column in ``adata.obs`` containing cell phenotype labels.
    phenotype : str or list
        Cell phenotype label or list of labels to map onto fibers.
    """

    phenotype = _ensure_list(phenotype)
    cells = _filter_cells(adata, phenotype_key, phenotype)
    links = _subset_links(links_df, cells, fiber_df)

    density = links.groupby("fiber_id").size()
    presence = density.gt(0).astype(float)
    mean_dist = links.groupby("fiber_id")["distance"].mean()

    name = _label_suffix(phenotype, "cells")

    fiber_df = fiber_df.copy()
    fiber_df[f"{name}_density"] = np.nan
    fiber_df[f"{name}_presence"] = np.nan
    fiber_df[f"{name}_mean_distance"] = np.nan

    fiber_df[f"{name}_density"] = fiber_df.index.map(density)
    fiber_df[f"{name}_presence"] = fiber_df.index.map(presence)
    fiber_df[f"{name}_mean_distance"] = fiber_df.index.map(mean_dist)

    return fiber_df



def map_cells_to_fibers_kernel(
    adata: ad.AnnData,
    fiber_df: pd.DataFrame,
    links_df: pd.DataFrame,
    phenotype_key: str,
    phenotype: str,
    bandwidth: float=50,
) -> pd.DataFrame:
    """
    Map cell influence onto ECM fibers using Gaussian kernel density.

    Produces smooth spatial signals of cell presence around fibers.

    Generated features
    ------------------
    <phenotype>_kernel_density
        distance-weighted cell density

    <phenotype>_nearest_distance
        distance to nearest cell

    Biological interpretation
    -------------------------
    high kernel density
        strong cellular niche around fiber

    low kernel density
        fiber located in cell-poor region

    Parameters
    ----------
    phenotype_key : str
        Column in ``adata.obs`` containing cell phenotype labels.
    phenotype : str or list
        Cell phenotype label or list of labels to map onto fibers.
    bandwidth : float
        Gaussian kernel bandwidth in the same spatial units as coordinates.
    """

    phenotype = _ensure_list(phenotype)
    cells = _filter_cells(adata, phenotype_key, phenotype)

    links = _subset_links(links_df, cells, fiber_df).copy()

    d = links["distance"].to_numpy()
    links["weight"] = np.exp(-(d**2)/(2*bandwidth**2))

    kernel = links.groupby("fiber_id")["weight"].sum()
    nearest = links.groupby("fiber_id")["distance"].min()

    name = _label_suffix(phenotype, "cells")

    fiber_df = fiber_df.copy()
    fiber_df[f"{name}_kernel_density"] = np.nan
    fiber_df[f"{name}_nearest_distance"] = np.nan

    fiber_df[f"{name}_kernel_density"] = fiber_df.index.map(kernel)
    fiber_df[f"{name}_nearest_distance"] = fiber_df.index.map(nearest)

    return fiber_df


# ============================================================
# 7. Local cross Moran’s I
# ============================================================

def local_cross_morans_i_ecm_cells(
    fiber_df: pd.DataFrame,
    feature_fiber: str,
    feature_cell: str,
    k: int=8,
    fiber_x: str="X_centroid",
    fiber_y: str="Y_centroid",
    fiber_type_key: str="fiber_type",
    fiber_type: str=None,
) -> np.ndarray:
    """
    Local cross Moran's I.

    Produces spatial map of ECM–cell coupling.

    Biological application
    ----------------------
    Detects spatial hotspots where ECM morphology
    interacts with cellular niches.

    Example interpretations
    -----------------------
    high positive
        aligned ECM near tumor invasion fronts

    high negative
        aligned ECM associated with immune exclusion

    Parameters
    ----------
    feature_fiber : str
        Column in ``fiber_df`` containing the ECM feature to analyze.
    feature_cell : str
        Column in ``fiber_df`` containing a cell-derived feature already mapped to fibers.
    k : int
        Number of nearest neighbors used to define the fiber-level spatial graph.
    fiber_x, fiber_y : str
        Column names in ``fiber_df`` containing fiber coordinates.
    """

    fiber_subset = fiber_df.copy()

    # --------------------------------------------------
    # Filter fiber types
    # --------------------------------------------------

    fiber_type = _ensure_list(fiber_type)

    if fiber_type is not None:

        fiber_subset = fiber_subset[
            fiber_subset[fiber_type_key].isin(fiber_type)
        ]

    if fiber_subset.empty:
        raise ValueError("No fibers found for specified fiber_type")

    # --------------------------------------------------
    # Remove missing values
    # --------------------------------------------------

    valid = (
        fiber_subset[[fiber_x, fiber_y, feature_fiber, feature_cell]]
        .notna()
        .all(axis=1)
    )

    fiber_subset = fiber_subset[valid]

    coords = fiber_subset[[fiber_x, fiber_y]].to_numpy()

    x = fiber_subset[feature_fiber].to_numpy(dtype=float)
    y = fiber_subset[feature_cell].to_numpy(dtype=float)

    n = len(x)

    if n < 3:
        result = fiber_df.copy()
        result[f"local_cross_moran_{feature_fiber}_{feature_cell}"] = np.nan
        return result

    k_eff = _resolve_k(n, k)

    if k_eff is None:
        result = fiber_df.copy()
        result[f"local_cross_moran_{feature_fiber}_{feature_cell}"] = np.nan
        return result

    # --------------------------------------------------
    # Spatial weights
    # --------------------------------------------------

    W = kneighbors_graph(
        coords,
        k_eff,
        mode="connectivity",
        include_self=False
    ).toarray()

    # row normalization
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W = W / row_sums

    # --------------------------------------------------
    # Center variables
    # --------------------------------------------------

    x_c = x - x.mean()
    y_c = y - y.mean()

    m2 = np.sum(y_c**2) / n

    if m2 == 0:
        result = fiber_df.copy()
        result[f"local_cross_moran_{feature_fiber}_{feature_cell}"] = np.nan
        return result

    local_I = x_c * (W @ y_c) / m2

    result = fiber_df.copy()

    result.loc[
        fiber_subset.index,
        f"local_cross_moran_{feature_fiber}_{feature_cell}"
    ] = local_I

    return result



# ============================================================
# 8. Spatial regression models
# ============================================================

def spatial_linear_regression(
    df: pd.DataFrame,
    feature_predictor: str,
    feature_response: str,
    phenotype_key: str=None,
    phenotype: str=None,
) -> object:
    """
    Linear regression between spatial features.

    Can be applied to:
        ECM ↔ cell coupling
        ECM ↔ ECM coupling
        cell ↔ cell coupling

    Example biological questions
    ----------------------------
    Does collagen alignment predict tumor density?
    Does ECM thickness correlate with macrophage infiltration?

    Parameters
    ----------
    df : DataFrame
        Data containing spatial features (fiber_df or adata.obs)

    feature_predictor : str
        predictor variable

    feature_response : str
        response variable

    phenotype : str or list, optional
        Optional phenotype filter when ``df`` contains cell-level data.

    Returns
    -------
    dict
        regression statistics
    """

    data = df.copy()

    if phenotype is not None and phenotype_key is not None:

        phenotype = _ensure_list(phenotype)

        data = data[data[phenotype_key].isin(phenotype)]

    data = data[[feature_predictor, feature_response]].dropna()

    if data.empty:
        raise ValueError("No valid rows remain after filtering and dropping missing values.")

    X = data[[feature_predictor]]
    y = data[feature_response]

    model = LinearRegression()
    model.fit(X, y)

    return {
        "coef": model.coef_[0],
        "intercept": model.intercept_,
        "r2": model.score(X, y),
        "n": len(data)
    }


def spatial_mixed_model(
    df: pd.DataFrame,
    formula: str,
    group_key: str="imageid",
    phenotype_key: str=None,
    phenotype: str=None,
) -> object:
    """
    Mixed effects spatial regression.

    Controls for variability across images or patients.

    Example
    -------
    Tumor_density ~ alignment_score + (1|imageid)

    Parameters
    ----------
    formula : str
        Patsy-style mixed model formula.
    group_key : str
        Column in ``df`` defining the grouping structure, typically image or patient ID.
    """

    data = df.copy()

    if phenotype is not None and phenotype_key is not None:

        phenotype = _ensure_list(phenotype)

        data = data[data[phenotype_key].isin(phenotype)]

    data = data.dropna()

    if data.empty:
        raise ValueError("No valid rows remain after filtering and dropping missing values.")

    try:
        from statsmodels.formula.api import mixedlm
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        raise ImportError(
            "spatial_mixed_model requires statsmodels with a SciPy-compatible "
            "installation. Install or update statsmodels to use this function; "
            "other ECM statistics do not require it."
        ) from exc

    model = mixedlm(formula, data, groups=data[group_key])

    return model.fit()



def spatial_enrichment_score(
    df: pd.DataFrame,
    feature: str,
    phenotype_key: str,
    phenotype: str=None,
) -> object:
    """
    Spatial enrichment score.

    Measures whether cell phenotypes are enriched in regions
    with high values of a spatial feature.

    Biological application
    ----------------------
    Detect cell types associated with ECM remodeling.

    Examples
    --------
    CD8 enrichment in low collagen regions
    macrophage enrichment in fibrotic zones

    Parameters
    ----------
    feature : str
        Spatial feature whose mean value is summarized by phenotype.
    phenotype_key : str
        Column in ``df`` containing phenotype labels.
    phenotype : str or list, optional
        Optional subset of phenotype labels to keep before summarizing.
    """

    data = df.copy()

    if phenotype is not None:

        phenotype = _ensure_list(phenotype)

        data = data[data[phenotype_key].isin(phenotype)]

    data = data[[feature, phenotype_key]].dropna()

    scores = (
        data.groupby(phenotype_key)[feature]
        .mean()
        .sort_values(ascending=False)
    )

    return scores



# ============================================================
# 8. ECM vector field
# ============================================================

def fiber_vectors(fiber_df: pd.DataFrame) -> np.ndarray:
    """
    Convert ECM fiber orientation into vector field representation.

    This allows modeling ECM as a directional scaffold.
    """
    fiber_df = fiber_df.copy()

    angles = np.deg2rad(fiber_df["orientation"])

    fiber_df["vx"] = np.cos(angles)
    fiber_df["vy"] = np.sin(angles)

    return fiber_df



# ============================================================
# 9. Cell–fiber alignment
# ============================================================

def cell_fiber_alignment(
    adata: ad.AnnData,
    fiber_df: pd.DataFrame,
    cell_orientation_key: str="orientation",
    cell_x: str="X_centroid",
    cell_y: str="Y_centroid",
    fiber_x: str="X_centroid",
    fiber_y: str="Y_centroid",
) -> object:
    """
    Compute alignment between cell orientation and ECM fiber orientation.

    Biological application
    ----------------------
    Detects whether cells orient themselves along ECM fibers.

    Example interpretation
    ----------------------
    high alignment
        cells migrate along collagen bundles

    low alignment
        cells move independently of ECM structure

    Parameters
    ----------
    cell_orientation_key : str
        Column in ``adata.obs`` containing cell orientation angles in degrees.
    cell_x, cell_y : str
        Column names in ``adata.obs`` containing cell coordinates.
    fiber_x, fiber_y : str
        Column names in ``fiber_df`` containing fiber coordinates.
    """

    fiber_df = fiber_vectors(fiber_df)

    fiber_coords = fiber_df[[fiber_x, fiber_y]].to_numpy()

    tree = BallTree(fiber_coords)

    cell_coords = adata.obs[[cell_x, cell_y]].to_numpy()

    _, idx = tree.query(cell_coords, k=1)

    fiber_vec = fiber_df[["vx", "vy"]].to_numpy()[idx.flatten()]

    cell_angles = np.deg2rad(adata.obs[cell_orientation_key])

    cell_vec = np.vstack(
        (np.cos(cell_angles), np.sin(cell_angles))
    ).T

    alignment = np.abs(np.sum(cell_vec * fiber_vec, axis=1))

    adata = adata.copy()
    adata.obs["cell_fiber_alignment"] = alignment

    return adata



# ============================================================
# Section 3: ECM bipartite graph  (from archive/spatial/spatial_ecm_graph.py)
# ============================================================

import networkx as nx
from networkx.algorithms import bipartite

try:
    import community as community_louvain
except ImportError:
    community_louvain = None


def _fiber_node_id(fiber_id):
    """
    Canonical graph node id for a fiber.
    """
    return ("fiber", fiber_id)


def _cell_node_id(cell_id):
    """
    Canonical graph node id for a cell.
    """
    return ("cell", cell_id)


def build_ecm_bipartite_graph_per_image(
    adata: ad.AnnData,
    fiber_df: pd.DataFrame,
    links_df: pd.DataFrame,
    image_key: str="imageid",
    link_image_key: str="imageid",
    distance_scale: float=50,
    include_cell_attrs: bool=True,
    include_fiber_attrs: bool=True,
) -> pd.DataFrame:
    """
    Build bipartite graphs per image.

    Each graph contains two node types:
    - ``("cell", cell_id)`` for cells
    - ``("fiber", fiber_id)`` for fibers

    Edges represent precomputed spatial links from ``links_df``.
    The edge weight decays with distance as ``exp(-distance / distance_scale)``.

    Parameters
    ----------
    image_key : str
        Column shared by ``adata.obs`` and ``fiber_df`` identifying each image.
    link_image_key : str
        Column in ``links_df`` identifying each image.
    distance_scale : float
        Distance-decay scale used to convert link distance into edge weight.
    include_cell_attrs, include_fiber_attrs : bool
        If ``True``, copy row attributes from the source table onto graph nodes.

    Returns
    -------
    dict
        {image_id: graph}
    """

    graphs = {}

    images = np.intersect1d(
        adata.obs[image_key].unique(),
        fiber_df[image_key].unique()
    )

    for img in images:

        # -----------------------------
        # Subset data
        # -----------------------------
        cells = adata.obs[adata.obs[image_key] == img]
        fibers = fiber_df[fiber_df[image_key] == img]
        links = links_df[links_df[link_image_key] == img]

        G = nx.Graph()

        # -----------------------------
        # Add cell nodes
        # -----------------------------
        for cell_id in cells.index:
            attrs = {
                "node_type": "cell",
                image_key: img,
            }
            if include_cell_attrs:
                attrs.update(cells.loc[cell_id].to_dict())

            G.add_node(
                _cell_node_id(cell_id),
                **attrs,
            )

        # -----------------------------
        # Add fiber nodes
        # -----------------------------
        for fiber_id in fibers.index:
            attrs = {
                "node_type": "fiber",
                image_key: img,
            }
            if include_fiber_attrs:
                attrs.update(fibers.loc[fiber_id].to_dict())

            G.add_node(
                _fiber_node_id(fiber_id),
                **attrs,
            )

        # -----------------------------
        # Add edges (SAFE)
        # -----------------------------
        for _, row in links.iterrows():

            if row["cell_id"] not in cells.index:
                continue
            if row["fiber_id"] not in fibers.index:
                continue

            G.add_edge(
                _cell_node_id(row["cell_id"]),
                _fiber_node_id(row["fiber_id"]),
                distance=row["distance"],
                weight=np.exp(-row["distance"] / distance_scale),
                **{image_key: img},
            )

        graphs[img] = G

    return graphs

def project_fiber_graph_per_image(graphs: dict) -> dict:
    """
    Project each bipartite graph onto a fiber-only graph.

    Two fibers become connected when they share neighboring cells
    in the bipartite graph.
    """

    projected = {}

    for img, G in graphs.items():

        fiber_nodes = [
            n for n, d in G.nodes(data=True)
            if d["node_type"] == "fiber"
        ]

        if len(fiber_nodes) < 2:
            continue

        projected[img] = bipartite.weighted_projected_graph(G, fiber_nodes)

    return projected


def _best_louvain_partition(G, weight="weight", resolution=1.0, random_state=None):
    """
    Run Louvain community detection with either python-louvain or NetworkX.
    """
    if community_louvain is not None:
        return community_louvain.best_partition(
            G,
            weight=weight,
            resolution=resolution,
            random_state=random_state,
        )

    if not hasattr(nx.algorithms.community, "louvain_communities"):
        raise ImportError(
            "ECM niche detection requires either the 'python-louvain' package "
            "or a NetworkX version with louvain_communities."
        )

    communities = nx.algorithms.community.louvain_communities(
        G,
        weight=weight,
        resolution=resolution,
        seed=random_state,
    )

    partition = {}
    for niche, nodes in enumerate(communities):
        for node in nodes:
            partition[node] = niche

    return partition


def detect_ecm_niches_per_image(
    fiber_graphs: dict,
    weight: float="weight",
    resolution: float=1.0,
    random_state: int=None,
) -> pd.DataFrame:
    """
    Detect ECM niches separately per image.

    Runs Louvain community detection on each projected fiber graph.

    Uses the optional ``python-louvain`` package when installed. If it is not
    available, falls back to NetworkX's built-in Louvain implementation.
    """

    niche_maps = {}

    for img, G in fiber_graphs.items():

        if len(G.nodes) < 2 or G.number_of_edges() == 0:
            continue

        partition = _best_louvain_partition(
            G,
            weight=weight,
            resolution=resolution,
            random_state=random_state,
        )

        niche_maps[img] = partition

    return niche_maps


def assign_niches_to_fibers(fiber_df: pd.DataFrame, niche_maps: dict, image_key: str="imageid") -> pd.DataFrame:
    """
    Map per-image fiber-graph community labels back onto ``fiber_df``.
    """

    fiber_df = fiber_df.copy()

    niche_labels = {}

    for img, partition in niche_maps.items():
        for node, niche in partition.items():
            if not (isinstance(node, tuple) and len(node) == 2):
                continue

            node_type, fiber_id = node

            if node_type != "fiber":
                continue

            niche_labels[(img, fiber_id)] = niche

    fiber_df["niche"] = [
        niche_labels.get((img, fiber_id), np.nan)
        for fiber_id, img in zip(fiber_df.index, fiber_df[image_key])
    ]

    return fiber_df



def compute_invasion_score(
    fiber_df: pd.DataFrame,
    tumor_density_col: str="tumor_density",
    alignment_col: str="alignment_score",
    image_key: str="imageid",
    niche_col: str="niche",
) -> pd.DataFrame:
    """
    Summarize each ECM niche by tumor density, alignment, and a simple invasion score.

    Parameters
    ----------
    tumor_density_col : str
        Column in ``fiber_df`` containing tumor-density values per fiber.
    alignment_col : str
        Column in ``fiber_df`` containing fiber alignment values.
    image_key : str
        Column in ``fiber_df`` identifying each image.
    niche_col : str
        Column in ``fiber_df`` containing niche labels assigned to fibers.
    """

    df = fiber_df.copy()

    summary = (
        df.groupby([image_key, niche_col])
        [[tumor_density_col, alignment_col]]
        .mean()
        .reset_index()
    )

    summary["invasion_score"] = (
        summary[tumor_density_col] *
        summary[alignment_col]
    )

    return summary.sort_values("invasion_score", ascending=False)



# ============================================================
# Section 4: ECM cell neighborhoods  (from archive/spatial/spatial_ecm_neighborhoods.py)
# ============================================================

"""
ECM-cell spatial neighborhood analysis.

This module implements a scimap-style radius-neighborhood workflow for
cell-ECM data. Around each cell, it summarizes nearby cell phenotypes and
nearby ECM fibers, then clusters those local profiles into spatial
neighborhoods.
"""

import re

from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler


def _safe_label(value):
    """
    Convert category labels into stable column-name suffixes.
    """
    value = str(value)
    value = re.sub(r"[^0-9a-zA-Z]+", "_", value)
    return value.strip("_")


def _circle_area_mm2(radius, pixel_size_um):
    """
    Area of the circular neighborhood in mm2.
    """
    radius_um = radius * pixel_size_um
    return np.pi * (radius_um ** 2) / 1e6


def detect_tissue_regions_dbscan(
    adata: ad.AnnData,
    eps: float=200,
    min_samples: int=50,
    min_cells: int=1600,
    image_key: str="imageid",
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    cluster_key: str="tissue_region",
    keep_key: list[str] | None="keep_tissue",
) -> pd.DataFrame:
    """
    Identify large tissue pieces from cell coordinates using DBSCAN.

    This is useful before radius-neighborhood analysis because isolated cells,
    dust-like segmentation artifacts, and tiny disconnected fragments can
    otherwise form misleading low-ECM neighborhoods.

    Parameters
    ----------
    adata : AnnData
        Cell-level object. ``adata.obs`` must contain image ids and spatial
        coordinates.
    eps : float
        DBSCAN neighborhood radius in the same coordinate units as
        ``x_key``/``y_key``. In pixel coordinates, this should be pixels.
    min_samples : int
        Minimum local cell count for DBSCAN core points.
    min_cells : int
        Minimum total number of cells required to retain a tissue component.
        Components smaller than this are marked as not kept.
    image_key : str
        Column in ``adata.obs`` identifying each tissue image/sample.
    x_key, y_key : str
        Coordinate columns in ``adata.obs``.
    cluster_key : str
        Output column for tissue-component labels.
    keep_key : str
        Output boolean column indicating retained tissue cells.

    Returns
    -------
    tissue_df : DataFrame
        One row per cell with DBSCAN labels, component size, and keep flag.
    summary_df : DataFrame
        One row per DBSCAN component with size and retained/removed status.
    """
    obs = adata.obs
    required = [image_key, x_key, y_key]
    missing = [col for col in required if col not in obs.columns]
    if missing:
        raise KeyError(f"adata.obs is missing required columns: {missing}")

    tissue_rows = []
    summary_rows = []

    for img, sub in obs.groupby(image_key, observed=True):
        sub = sub.dropna(subset=[x_key, y_key]).copy()
        if sub.empty:
            continue

        coords = sub[[x_key, y_key]].to_numpy(dtype=float)
        valid = np.isfinite(coords).all(axis=1)
        sub = sub.loc[valid].copy()
        coords = coords[valid]

        if len(sub) < min_samples:
            labels = np.full(len(sub), -1, dtype=int)
        else:
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)

        raw_col = f"{cluster_key}_raw"
        sub[raw_col] = labels
        sizes = sub[raw_col].value_counts()
        keep_labels = [
            int(label)
            for label, size in sizes.items()
            if label >= 0 and size >= min_cells
        ]
        keep_labels = set(keep_labels)

        sub[keep_key] = sub[raw_col].isin(keep_labels)
        sub[f"{cluster_key}_size"] = sub[raw_col].map(sizes).astype(int)
        sub[cluster_key] = pd.Series(pd.NA, index=sub.index, dtype="object")
        for label in keep_labels:
            mask = sub[raw_col].eq(label)
            sub.loc[mask, cluster_key] = f"{img}_tissue_{label}"

        for label, size in sizes.items():
            summary_rows.append({
                image_key: img,
                "dbscan_label": int(label),
                "n_cells": int(size),
                "kept": bool(label in keep_labels),
            })

        tissue_rows.append(
            sub[[image_key, x_key, y_key, raw_col, f"{cluster_key}_size", cluster_key, keep_key]]
            .assign(cell_id=sub.index)
        )

    if not tissue_rows:
        return pd.DataFrame(), pd.DataFrame(summary_rows)

    tissue_df = pd.concat(tissue_rows, axis=0)
    tissue_df = tissue_df.set_index("cell_id", drop=False)
    summary_df = pd.DataFrame(summary_rows)

    return tissue_df, summary_df


def filter_ecm_cell_inputs_by_tissue(
    adata: ad.AnnData,
    fiber_df: pd.DataFrame,
    links_df: pd.DataFrame,
    tissue_df: pd.DataFrame,
    keep_key: list[str] | None="keep_tissue",
    cell_id_col: str="cell_id",
    fiber_id_col: str="fiber_id",
) -> ad.AnnData:
    """
    Filter cells, fibers, and links to retained tissue components.

    Fibers are retained when they are linked to at least one retained tissue
    cell. This keeps ECM features aligned with the tissue-cleaned cell anchors
    used for radius-neighborhood analysis.
    """
    if tissue_df.empty:
        raise ValueError("tissue_df is empty.")
    if keep_key not in tissue_df.columns:
        raise KeyError(f"{keep_key!r} is not present in tissue_df.")

    keep_cells = tissue_df.loc[tissue_df[keep_key], cell_id_col].astype(str)
    keep_cells = set(keep_cells)

    adata_clean = adata[adata.obs.index.astype(str).isin(keep_cells)].copy()
    links_clean = links_df[links_df[cell_id_col].astype(str).isin(keep_cells)].copy()

    keep_fibers = set(links_clean[fiber_id_col].astype(str))
    fiber_clean = fiber_df[fiber_df.index.astype(str).isin(keep_fibers)].copy()
    links_clean = links_clean[links_clean[fiber_id_col].astype(str).isin(fiber_clean.index.astype(str))].copy()

    return adata_clean, fiber_clean, links_clean


def build_ecm_cell_neighborhood_features(
    adata: ad.AnnData,
    fiber_df: pd.DataFrame,
    links_df: pd.DataFrame,
    radius: float,
    phenotype_key: str="phenotype",
    image_key: str="imageid",
    group_key: str="pathology",
    x_key: str="X_centroid",
    y_key: str="Y_centroid",
    fiber_type_key: str="fiber_type",
    fiber_types: list[str] | None=("COL6A1", "CHP"),
    cell_phenotypes: list[str] | None=None,
    include_self: bool=False,
    pixel_size_um: float=0.325,
    distance_weight_scale: float=None,
) -> pd.DataFrame:
    """
    Build radius-based ECM-cell neighborhood features around each cell.

    This mirrors the idea of scimap radius neighborhoods: every cell becomes
    an anchor point, and the local neighborhood is represented by nearby cell
    phenotype composition plus nearby ECM fiber abundance.

    Parameters
    ----------
    adata : AnnData
        Cell-level object. ``adata.obs`` must contain coordinates, image ids,
        and phenotype annotations.
    fiber_df : DataFrame
        Fiber table containing coordinates, image ids, and fiber types.
    links_df : DataFrame
        Precomputed cell-fiber links with ``cell_id``, ``fiber_id``,
        ``imageid``, and ``distance`` columns.
    radius : float
        Spatial radius in the same units as ``X_centroid``/``Y_centroid``.
        If coordinates are pixels, pass a pixel radius.
    phenotype_key : str
        Column in ``adata.obs`` containing cell phenotype labels.
    image_key : str
        Column shared by cells, fibers, and links identifying each image.
    group_key : str
        Optional biological group column, for example RA/OA pathology.
    x_key, y_key : str
        Coordinate columns used for cell neighborhoods.
    fiber_type_key : str
        Column in ``fiber_df`` containing fiber type labels.
    fiber_types : sequence of str
        Fiber types to include as ECM features, for example ``COL6A1`` and
        ``CHP`` for collagen VI dark-zone discovery.
    cell_phenotypes : sequence of str, optional
        Phenotypes to include. If ``None``, all observed phenotypes are used.
    include_self : bool
        If ``False``, subtract each anchor cell from its own phenotype count.
        This makes the features describe the surrounding neighborhood rather
        than the anchor label itself.
    pixel_size_um : float
        Pixel size used to convert radius-neighborhood counts into densities.
    distance_weight_scale : float, optional
        Distance decay scale for fiber weights. Defaults to ``radius / 2``.

    Returns
    -------
    DataFrame
        One row per cell with local cell-composition and ECM-fiber features.
    """
    fiber_types = _ensure_list(fiber_types)
    if fiber_types is None:
        fiber_types = []

    obs = adata.obs.copy()
    required_cell_cols = [image_key, x_key, y_key, phenotype_key]
    missing = [col for col in required_cell_cols if col not in obs.columns]
    if missing:
        raise KeyError(f"adata.obs is missing required columns: {missing}")

    required_fiber_cols = [image_key, fiber_type_key]
    missing = [col for col in required_fiber_cols if col not in fiber_df.columns]
    if missing:
        raise KeyError(f"fiber_df is missing required columns: {missing}")

    required_link_cols = [image_key, "cell_id", "fiber_id", "distance"]
    missing = [col for col in required_link_cols if col not in links_df.columns]
    if missing:
        raise KeyError(f"links_df is missing required columns: {missing}")

    if cell_phenotypes is None:
        cell_phenotypes = sorted(obs[phenotype_key].dropna().unique())
    else:
        cell_phenotypes = list(cell_phenotypes)

    distance_weight_scale = radius / 2 if distance_weight_scale is None else distance_weight_scale
    neighborhood_area = _circle_area_mm2(radius, pixel_size_um)

    rows = []
    images = np.intersect1d(obs[image_key].dropna().unique(), fiber_df[image_key].dropna().unique())

    fiber_meta = fiber_df[[fiber_type_key]].copy()

    for img in images:
        cells = obs.loc[obs[image_key].eq(img)].dropna(subset=[x_key, y_key]).copy()
        if cells.empty:
            continue

        coords = cells[[x_key, y_key]].to_numpy(dtype=float)
        valid = np.isfinite(coords).all(axis=1)
        cells = cells.loc[valid].copy()
        coords = coords[valid]
        if cells.empty:
            continue

        image_features = pd.DataFrame(index=cells.index)
        image_features["cell_id"] = cells.index
        image_features[image_key] = img
        if group_key in cells.columns:
            image_features[group_key] = cells[group_key].to_numpy()
        image_features[phenotype_key] = cells[phenotype_key].to_numpy()
        image_features[x_key] = cells[x_key].to_numpy(dtype=float)
        image_features[y_key] = cells[y_key].to_numpy(dtype=float)

        for phenotype in cell_phenotypes:
            suffix = _safe_label(phenotype)
            phenotype_mask = cells[phenotype_key].eq(phenotype).to_numpy()
            phenotype_coords = coords[phenotype_mask]

            if len(phenotype_coords) == 0:
                counts = np.zeros(len(cells), dtype=float)
            else:
                tree = KDTree(phenotype_coords)
                counts = tree.query_radius(coords, r=radius, count_only=True).astype(float)
                if not include_self:
                    counts = counts - phenotype_mask.astype(float)
                    counts[counts < 0] = 0

            image_features[f"cell_count_{suffix}"] = counts
            image_features[f"cell_density_{suffix}"] = counts / neighborhood_area

        cell_count_cols = [f"cell_count_{_safe_label(p)}" for p in cell_phenotypes]
        total_cells = image_features[cell_count_cols].sum(axis=1).to_numpy(dtype=float)
        image_features["neighbor_cell_count"] = total_cells
        image_features["neighbor_cell_density"] = total_cells / neighborhood_area

        denom = np.where(total_cells > 0, total_cells, 1.0)
        for phenotype in cell_phenotypes:
            suffix = _safe_label(phenotype)
            image_features[f"cell_fraction_{suffix}"] = (
                image_features[f"cell_count_{suffix}"].to_numpy(dtype=float) / denom
            )

        image_links = links_df.loc[
            links_df[image_key].eq(img) & links_df["distance"].le(radius),
            [image_key, "cell_id", "fiber_id", "distance"],
        ].copy()

        if not image_links.empty and fiber_types:
            image_links = image_links.merge(
                fiber_meta,
                left_on="fiber_id",
                right_index=True,
                how="left",
            )
            image_links = image_links[image_links[fiber_type_key].isin(fiber_types)].copy()

        fiber_count_total = np.zeros(len(image_features), dtype=float)
        if image_links.empty or not fiber_types:
            for fiber_type in fiber_types:
                suffix = _safe_label(fiber_type)
                image_features[f"fiber_count_{suffix}"] = 0.0
                image_features[f"fiber_density_{suffix}"] = 0.0
                image_features[f"fiber_weight_{suffix}"] = 0.0
                image_features[f"nearest_distance_{suffix}"] = np.nan
        else:
            image_links["weight"] = np.exp(-image_links["distance"] / distance_weight_scale)
            count_table = (
                image_links
                .groupby(["cell_id", fiber_type_key], observed=True)
                .size()
                .unstack(fill_value=0)
            )
            weight_table = (
                image_links
                .groupby(["cell_id", fiber_type_key], observed=True)["weight"]
                .sum()
                .unstack(fill_value=0)
            )
            nearest_table = (
                image_links
                .groupby(["cell_id", fiber_type_key], observed=True)["distance"]
                .min()
                .unstack()
            )

            for fiber_type in fiber_types:
                suffix = _safe_label(fiber_type)
                counts = count_table[fiber_type] if fiber_type in count_table else pd.Series(dtype=float)
                weights = weight_table[fiber_type] if fiber_type in weight_table else pd.Series(dtype=float)
                nearest = nearest_table[fiber_type] if fiber_type in nearest_table else pd.Series(dtype=float)

                image_features[f"fiber_count_{suffix}"] = image_features.index.map(counts).fillna(0).to_numpy(dtype=float)
                image_features[f"fiber_density_{suffix}"] = image_features[f"fiber_count_{suffix}"] / neighborhood_area
                image_features[f"fiber_weight_{suffix}"] = image_features.index.map(weights).fillna(0).to_numpy(dtype=float)
                image_features[f"nearest_distance_{suffix}"] = image_features.index.map(nearest).to_numpy(dtype=float)
                fiber_count_total += image_features[f"fiber_count_{suffix}"].to_numpy(dtype=float)

        image_features["neighbor_fiber_count"] = fiber_count_total
        fiber_denom = np.where(fiber_count_total > 0, fiber_count_total, 1.0)
        for fiber_type in fiber_types:
            suffix = _safe_label(fiber_type)
            image_features[f"fiber_fraction_{suffix}"] = image_features[f"fiber_count_{suffix}"] / fiber_denom

        if {"COL6A1", "CHP"}.issubset(set(map(str, fiber_types))):
            col6 = image_features["fiber_count_COL6A1"].to_numpy(dtype=float)
            chp = image_features["fiber_count_CHP"].to_numpy(dtype=float)
            image_features["COL6A1_CHP_count_ratio"] = (col6 + 1.0) / (chp + 1.0)

        rows.append(image_features)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, axis=0)


def default_ecm_cell_neighborhood_feature_columns(
    neighborhood_df: pd.DataFrame,
    include_cell_fractions: bool=True,
    include_fiber_features: bool=True,
    include_cell_density: bool=False,
) -> object:
    """
    Select sensible default features for K-means neighborhood clustering.
    """
    cols = []
    if include_cell_fractions:
        cols.extend([c for c in neighborhood_df.columns if c.startswith("cell_fraction_")])
    if include_fiber_features:
        cols.extend([c for c in neighborhood_df.columns if c.startswith("fiber_count_")])
        cols.extend([c for c in neighborhood_df.columns if c.startswith("fiber_density_")])
        cols.extend([c for c in neighborhood_df.columns if c.startswith("fiber_weight_")])
        cols.extend([c for c in neighborhood_df.columns if c.startswith("nearest_distance_")])
    if include_cell_density and "neighbor_cell_density" in neighborhood_df.columns:
        cols.append("neighbor_cell_density")
    return cols


def cluster_ecm_cell_neighborhoods(
    neighborhood_df: pd.DataFrame,
    feature_columns: list[str]=None,
    feature_weights: dict[str, float] | None=None,
    n_clusters: int=10,
    label_key: str="ecm_cell_neighborhood",
    random_state: int=0,
    scale: bool=True,
) -> ad.AnnData:
    """
    Cluster ECM-cell neighborhood features with K-means.

    Parameters
    ----------
    neighborhood_df : DataFrame
        Output of :func:`build_ecm_cell_neighborhood_features`.
    feature_columns : sequence of str, optional
        Columns used for clustering. If ``None``, a default set of cell
        fractions, fiber densities, fiber fractions, and total cell density is
        used.
    feature_weights : dict, optional
        Optional per-feature weights applied after scaling. This is useful when
        one feature block has many more columns than another, for example all
        cell-type fractions versus a smaller COL6A1/CHP ECM block.
    n_clusters : int
        Number of K-means neighborhoods.
    label_key : str
        Name of the output label column.
    random_state : int
        Random seed for reproducibility.
    scale : bool
        If ``True``, standardize features before K-means.

    Returns
    -------
    clustered_df : DataFrame
        Input table with added K-means labels.
    model : KMeans
        Fitted K-means model.
    scaler : StandardScaler or None
        Fitted scaler when ``scale=True``.
    feature_columns : list
        Columns used for clustering.
    """
    if neighborhood_df.empty:
        raise ValueError("neighborhood_df is empty.")

    if feature_columns is None:
        feature_columns = default_ecm_cell_neighborhood_feature_columns(neighborhood_df)
    feature_columns = list(feature_columns)

    missing = [col for col in feature_columns if col not in neighborhood_df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")

    X = neighborhood_df[feature_columns].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X)
    else:
        X_fit = X.to_numpy(dtype=float)

    if feature_weights is not None:
        weights = np.array(
            [feature_weights.get(col, 1.0) for col in feature_columns],
            dtype=float,
        )
        X_fit = X_fit * weights

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=20,
    )
    labels = model.fit_predict(X_fit)

    out = neighborhood_df.copy()
    out[label_key] = labels
    out[f"{label_key}_label"] = pd.Categorical([f"N{label}" for label in labels])
    return out, model, scaler, feature_columns


def summarize_ecm_cell_neighborhoods(
    neighborhood_df: pd.DataFrame,
    label_key: str="ecm_cell_neighborhood",
    phenotype_key: str="phenotype",
    group_key: str="pathology",
) -> pd.DataFrame:
    """
    Summarize K-means neighborhoods by size, phenotype composition, and features.
    """
    if label_key not in neighborhood_df.columns:
        raise KeyError(f"{label_key!r} is not present in neighborhood_df.")

    feature_cols = [
        c for c in neighborhood_df.columns
        if c.startswith((
            "cell_fraction_",
            "fiber_count_",
            "fiber_density_",
            "fiber_weight_",
            "fiber_fraction_",
            "nearest_distance_",
        ))
        or c in {"neighbor_cell_density", "neighbor_cell_count", "neighbor_fiber_count", "COL6A1_CHP_count_ratio"}
    ]

    summary = (
        neighborhood_df
        .groupby(label_key, observed=True)
        .agg(
            n_cells=("cell_id", "size"),
            **{f"median_{col}": (col, "median") for col in feature_cols}
        )
        .reset_index()
    )

    if group_key in neighborhood_df.columns:
        group_counts = (
            neighborhood_df
            .groupby([group_key, label_key], observed=True)
            .size()
            .reset_index(name="n_cells")
        )
        totals = group_counts.groupby(group_key, observed=True)["n_cells"].transform("sum")
        group_counts["fraction"] = group_counts["n_cells"] / totals
    else:
        group_counts = pd.DataFrame()

    if phenotype_key in neighborhood_df.columns:
        phenotype_counts = (
            neighborhood_df
            .groupby([label_key, phenotype_key], observed=True)
            .size()
            .reset_index(name="n_cells")
        )
        totals = phenotype_counts.groupby(label_key, observed=True)["n_cells"].transform("sum")
        phenotype_counts["fraction"] = phenotype_counts["n_cells"] / totals
    else:
        phenotype_counts = pd.DataFrame()

    return summary, group_counts, phenotype_counts


def score_col6_dark_neighborhoods(
    neighborhood_df: pd.DataFrame,
    label_key: str="ecm_cell_neighborhood",
    col6_density_col: str="fiber_density_COL6A1",
    chp_density_col: str="fiber_density_CHP",
    cell_density_col: str="neighbor_cell_density",
    low_quantile: float=0.35,
    min_cell_density_quantile: float=0.10,
    output_key: str="COL6_dark_neighborhood",
) -> pd.DataFrame:
    """
    Mark K-means neighborhoods that look COL6A1/CHP-poor inside tissue.

    The rule is intentionally transparent:
    - low median local COL6A1 fiber density;
    - low median local CHP fiber density;
    - enough local cells to avoid selecting sparse/tissue-edge artifacts.

    Returns a copy of ``neighborhood_df`` plus a cluster-level score table.
    """
    required = [label_key, col6_density_col, chp_density_col, cell_density_col]
    missing = [col for col in required if col not in neighborhood_df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    cluster_scores = (
        neighborhood_df
        .groupby(label_key, observed=True)
        .agg(
            n_cells=("cell_id", "size"),
            median_col6_density=(col6_density_col, "median"),
            median_chp_density=(chp_density_col, "median"),
            median_cell_density=(cell_density_col, "median"),
        )
        .reset_index()
    )

    col6_cutoff = cluster_scores["median_col6_density"].quantile(low_quantile)
    chp_cutoff = cluster_scores["median_chp_density"].quantile(low_quantile)
    cell_density_cutoff = cluster_scores["median_cell_density"].quantile(min_cell_density_quantile)

    cluster_scores["low_col6"] = cluster_scores["median_col6_density"] <= col6_cutoff
    cluster_scores["low_chp"] = cluster_scores["median_chp_density"] <= chp_cutoff
    cluster_scores["inside_tissue_like"] = cluster_scores["median_cell_density"] >= cell_density_cutoff
    cluster_scores["col6_dark_candidate"] = (
        cluster_scores["low_col6"]
        & cluster_scores["low_chp"]
        & cluster_scores["inside_tissue_like"]
    )

    out = neighborhood_df.copy()
    dark_clusters = set(
        cluster_scores.loc[
            cluster_scores["col6_dark_candidate"],
            label_key,
        ]
    )
    out[output_key] = np.where(out[label_key].isin(dark_clusters), "COL6_dark_candidate", "other")
    out[output_key] = pd.Categorical(out[output_key], categories=["COL6_dark_candidate", "other"])

    return out, cluster_scores


def add_neighborhoods_to_obs(
    adata: ad.AnnData,
    neighborhood_df: pd.DataFrame,
    columns: list[str],
) -> ad.AnnData:
    """
    Copy neighborhood labels/features back into ``adata.obs`` by cell id.

    Parameters
    ----------
    adata : AnnData
        Cell-level object to annotate.
    neighborhood_df : DataFrame
        Table indexed by cell id or containing a ``cell_id`` column.
    columns : sequence of str
        Columns to copy into ``adata.obs``.

    Returns
    -------
    AnnData
        A copy of ``adata`` with added observation columns.
    """
    out = adata.copy()
    columns = list(columns)
    missing = [col for col in columns if col not in neighborhood_df.columns]
    if missing:
        raise KeyError(f"neighborhood_df is missing columns: {missing}")

    if "cell_id" in neighborhood_df.columns:
        mapped = neighborhood_df.set_index("cell_id")[columns]
    else:
        mapped = neighborhood_df[columns]

    for col in columns:
        out.obs[col] = out.obs.index.map(mapped[col])

    return out



__all__ = [
    "build_cell_fiber_links",
    "build_nearest_cell_fiber_map",
    "cell_to_fiber_distance",
    "fiber_density_near_cells",
    "morans_i_fibers",
    "local_morans_i_fibers",
    "cross_morans_i_ecm_cells",
    "cross_morans_i_ecm_cells_permutation_test",
    "map_cells_to_fibers",
    "map_cells_to_fibers_kernel",
    "local_cross_morans_i_ecm_cells",
    "spatial_linear_regression",
    "spatial_mixed_model",
    "spatial_enrichment_score",
    "fiber_vectors",
    "cell_fiber_alignment",
    "cross_ripleys_k_permutation_envelope",
    "build_ecm_bipartite_graph_per_image",
    "project_fiber_graph_per_image",
    "detect_ecm_niches_per_image",
    "assign_niches_to_fibers",
    "compute_invasion_score",
    "detect_tissue_regions_dbscan",
    "filter_ecm_cell_inputs_by_tissue",
    "build_ecm_cell_neighborhood_features",
    "default_ecm_cell_neighborhood_feature_columns",
    "cluster_ecm_cell_neighborhoods",
    "summarize_ecm_cell_neighborhoods",
    "score_col6_dark_neighborhoods",
    "add_neighborhoods_to_obs",
    # helper also defined in this module
    "compute_convex_hull_area",
    "cross_ripleys_k",
]
