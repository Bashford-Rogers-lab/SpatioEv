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


import numpy as np
import pandas as pd

from sklearn.neighbors import BallTree, kneighbors_graph
from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import mixedlm
from scipy.spatial import ConvexHull


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


def compute_convex_hull_area(coords):
    """
    Compute area of convex hull.
    """
    coords = _clean_coords(coords)

    if len(coords) < 3:
        return np.nan

    hull = ConvexHull(coords)
    return hull.volume  # in 2D, volume = area


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
    adata,
    fiber_df,
    links_df,
    fiber_type_key="fiber_type",
    fiber_type=None,
    phenotype_key=None,
    phenotype=None,
    summary_phenotype_key=None,
):
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
    adata,
    fiber_df,
    links_df,
    fiber_type_key="fiber_type",
    fiber_type=None,
    phenotype_key=None,
    phenotype=None,
    normalize=False,
    density_radius=50,
):
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
    adata,
    fiber_df,
    links_df,
    radii,
    fiber_type_key="fiber_type",
    fiber_type=None,
    phenotype_key=None,
    phenotype=None,
    check_radius=True,
):
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
    adata,
    fiber_df,
    links_df,
    radii,
    fiber_type_key="fiber_type",
    fiber_type=None,
    phenotype_key=None,
    phenotype=None,
    permute="cells",
    n_sim=199,
    image_key="imageid",
    fiber_image_key="imageid",
    random_state=None,
    check_radius=True,
):
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
    fiber_df,
    feature,
    k=8,
    fiber_x="X_centroid",
    fiber_y="Y_centroid",
    fiber_type_key="fiber_type",
    fiber_type=None,
):
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
    fiber_df,
    feature,
    k=8,
    fiber_x="X_centroid",
    fiber_y="Y_centroid",
    fiber_type_key="fiber_type",
    fiber_type=None,
):
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
    adata,
    fiber_df,
    links_df,
    feature_fiber,
    cell_feature,
    k=8,
):
    
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
    adata,
    fiber_df,
    links_df,
    feature_fiber,
    cell_feature,
    k=8,
    n_sim=999,
    image_key="imageid",
    random_state=None,
):
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
    adata,
    fiber_df,
    links_df,
    phenotype_key,
    phenotype,
):
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
    adata,
    fiber_df,
    links_df,
    phenotype_key,
    phenotype,
    bandwidth=50,
):
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
    fiber_df,
    feature_fiber,
    feature_cell,
    k=8,
    fiber_x="X_centroid",
    fiber_y="Y_centroid",
    fiber_type_key="fiber_type",
    fiber_type=None,
):
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
    df,
    feature_predictor,
    feature_response,
    phenotype_key=None,
    phenotype=None,
):
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
    df,
    formula,
    group_key="imageid",
    phenotype_key=None,
    phenotype=None,
):
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

    model = mixedlm(formula, data, groups=data[group_key])

    return model.fit()



def spatial_enrichment_score(
    df,
    feature,
    phenotype_key,
    phenotype=None,
):
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

def fiber_vectors(fiber_df):
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
    adata,
    fiber_df,
    cell_orientation_key="orientation",
    cell_x="X_centroid",
    cell_y="Y_centroid",
    fiber_x="X_centroid",
    fiber_y="Y_centroid",
):
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
