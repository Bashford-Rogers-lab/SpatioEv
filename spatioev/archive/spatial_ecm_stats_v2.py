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
        x
        y
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


def _filter_fibers(
    fiber_df,
    fiber_type_key,
    fiber_type,
):

    if fiber_type is None:
        return fiber_df

    if isinstance(fiber_type, str):
        fiber_type = [fiber_type]

    return fiber_df[
        fiber_df[fiber_type_key].isin(fiber_type)
    ]

def _filter_cells(
    adata,
    phenotype_key,
    phenotype,
):

    cells = adata.obs

    if phenotype is None:
        return cells

    if phenotype_key is None:
        raise ValueError(
            "phenotype_key must be provided when phenotype filtering is used."
        )

    if isinstance(phenotype, str):
        phenotype = [phenotype]

    return cells[cells[phenotype_key].isin(phenotype)]



def compute_convex_hull_area(coords):
    """
    Compute area of convex hull.
    """
    if len(coords) < 3:
        return np.nan

    hull = ConvexHull(coords)
    return hull.volume  # in 2D, volume = area


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
    fiber_type
        restrict analysis to specific ECM fiber classes

    phenotype
        restrict analysis to specific cell phenotypes

    summary_phenotype_key
        return median distances grouped by phenotype
    """

    fibers = _filter_fibers(
        fiber_df,
        fiber_type_key,
        fiber_type,
    )

    cells = _filter_cells(
        adata,
        phenotype_key,
        phenotype,
    )

    links = links_df.copy()

    links = links[
        links["fiber_id"].isin(fibers.index)
    ]

    links = links[
        links["cell_id"].isin(cells.index)
    ]

    nearest = (
        links
        .sort_values("distance")
        .groupby("cell_id")
        .first()
    )

    if fiber_type is None:
        col_name = "dist_to_fiber"
    else:
        col_name = f"dist_to_{'_'.join(fiber_type)}"

    adata.obs.loc[
        nearest.index,
        col_name
    ] = nearest["distance"]

    if summary_phenotype_key is not None:

        return (
            adata.obs
            .groupby(summary_phenotype_key)[col_name]
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
    """

    fibers = _filter_fibers(
        fiber_df,
        fiber_type_key,
        fiber_type,
    )

    cells = _filter_cells(
        adata,
        phenotype_key,
        phenotype,
    )

    links = links_df.copy()

    links = links[
        links["fiber_id"].isin(fibers.index)
    ]

    links = links[
        links["cell_id"].isin(cells.index)
    ]

    density = links.groupby("cell_id").size()

    if fiber_type is None:
        col_name = "fiber_density"
    else:
        col_name = f"{'_'.join(fiber_type)}_density"

    adata.obs.loc[density.index, col_name] = density

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
    """

    fibers = _filter_fibers(
        fiber_df,
        fiber_type_key,
        fiber_type,
    )

    cells = _filter_cells(
        adata,
        phenotype_key,
        phenotype,
    )

    links = links_df.copy()

    links = links[
        links["fiber_id"].isin(fibers.index)
    ]

    links = links[
        links["cell_id"].isin(cells.index)
    ]

    N_cells = len(cells)
    N_fibers = len(fibers)

    if N_cells < 2 or N_fibers < 2:
        raise ValueError("Too few cells or fibers")

    cell_coords = cells[["X_centroid", "Y_centroid"]].to_numpy()
    fiber_coords = fibers[["X_centroid", "Y_centroid"]].to_numpy()

    all_coords = np.vstack([cell_coords, fiber_coords])
    area = compute_convex_hull_area(all_coords)
    
    if np.isnan(area):
        raise ValueError("Invalid convex hull area")

    K_values = []

    for r in radii:

        count = (links["distance"] <= r).sum()

        K_r = (area/(N_cells*N_fibers))*count

        K_values.append(K_r)

    K_values = np.array(K_values)

    L_values = np.sqrt(K_values/np.pi)

    return pd.DataFrame(
        {
            "radius": radii,
            "K": K_values,
            "L": L_values,
            "L_minus_r": L_values - radii,
        }
    )


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

    if fiber_type is not None:

        if isinstance(fiber_type, str):
            fiber_type = [fiber_type]

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

    # --------------------------------------------------
    # Spatial weights (kNN graph)
    # --------------------------------------------------

    W = kneighbors_graph(
        coords,
        k,
        mode="connectivity",
        include_self=False,
    )

    W = W.toarray()

    # --------------------------------------------------
    # Moran's I calculation
    # --------------------------------------------------

    x = values - values.mean()

    numerator = np.sum(W * np.outer(x, x))
    denominator = np.sum(x**2)

    n = len(values)

    I = (n / W.sum()) * (numerator / denominator)

    return I


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

    if fiber_type is not None:

        if isinstance(fiber_type, str):
            fiber_type = [fiber_type]

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
        fiber_df[f"local_moran_{feature}"] = np.nan
        return fiber_df

    # --------------------------------------------------
    # Build spatial weights
    # --------------------------------------------------

    W = kneighbors_graph(
        coords,
        k,
        mode="connectivity",
        include_self=False
    )

    W = W.toarray()

    # --------------------------------------------------
    # Center feature values
    # --------------------------------------------------

    x = values - values.mean()

    m2 = np.sum(x**2) / n

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

    Returns
    -------
    float
        cross Moran's I statistic
    """

    merged = links_df.merge(
        adata.obs[[cell_feature]],
        left_on="cell_id",
        right_index=True,
    )

    fiber_cell = (
        merged.groupby("fiber_id")[cell_feature]
        .mean()
    )

    fiber_df = fiber_df.copy()

    fiber_df["cell_feature_local"] = fiber_df.index.map(
        fiber_cell
    )

    valid = fiber_df[
        [feature_fiber, "cell_feature_local"]
    ].dropna()

    coords = fiber_df.loc[
        valid.index,
        ["X_centroid","Y_centroid"]
    ].to_numpy()

    x = valid[feature_fiber].to_numpy()
    y = valid["cell_feature_local"].to_numpy()

    W = kneighbors_graph(
        coords,
        k,
        mode="connectivity",
        include_self=False
    ).toarray()

    x -= x.mean()
    y -= y.mean()

    numerator = np.sum(W*np.outer(x,y))
    denom = np.sqrt(np.sum(x**2)*np.sum(y**2))

    I = (len(x)/W.sum())*(numerator/denom)

    return I


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
    """

    cells = adata.obs

    if isinstance(phenotype,str):
        phenotype=[phenotype]

    cells=cells[cells[phenotype_key].isin(phenotype)]

    links=links_df[
        links_df["cell_id"].isin(cells.index)
    ]

    density=links.groupby("fiber_id").size()

    presence=(density>0).astype(int)

    mean_dist=links.groupby("fiber_id")["distance"].mean()

    name="_".join(phenotype)

    fiber_df[f"{name}_density"]=fiber_df.index.map(density)
    fiber_df[f"{name}_presence"]=fiber_df.index.map(presence)
    fiber_df[f"{name}_mean_distance"]=fiber_df.index.map(mean_dist)

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
    """

    cells=adata.obs

    if isinstance(phenotype,str):
        phenotype=[phenotype]

    cells=cells[cells[phenotype_key].isin(phenotype)]

    links=links_df[
        links_df["cell_id"].isin(cells.index)
    ]

    d=links["distance"].to_numpy()

    weights=np.exp(-(d**2)/(2*bandwidth**2))

    links = links.copy()
    links["weight"] = weights

    kernel=links.groupby("fiber_id")["weight"].sum()

    nearest=links.groupby("fiber_id")["distance"].min()

    name="_".join(phenotype)

    fiber_df[f"{name}_kernel_density"]=fiber_df.index.map(kernel)
    fiber_df[f"{name}_nearest_distance"]=fiber_df.index.map(nearest)

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
    """

    fiber_subset = fiber_df.copy()

    # --------------------------------------------------
    # Filter fiber types
    # --------------------------------------------------

    if fiber_type is not None:

        if isinstance(fiber_type, str):
            fiber_type = [fiber_type]

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
        fiber_df[f"local_cross_moran_{feature_fiber}_{feature_cell}"] = np.nan
        return fiber_df

    # --------------------------------------------------
    # Spatial weights
    # --------------------------------------------------

    W = kneighbors_graph(
        coords,
        k,
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

    phenotype : optional filter if df contains cell data

    Returns
    -------
    dict
        regression statistics
    """

    data = df.copy()

    if phenotype is not None and phenotype_key is not None:

        if isinstance(phenotype, str):
            phenotype = [phenotype]

        data = data[data[phenotype_key].isin(phenotype)]

    data = data[[feature_predictor, feature_response]].dropna()

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
    """

    data = df.copy()

    if phenotype is not None and phenotype_key is not None:

        if isinstance(phenotype, str):
            phenotype = [phenotype]

        data = data[data[phenotype_key].isin(phenotype)]

    data = data.dropna()

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
    """

    data = df.copy()

    if phenotype is not None:

        if isinstance(phenotype, str):
            phenotype = [phenotype]

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
    """

    fiber_df = fiber_vectors(fiber_df)

    fiber_coords = fiber_df[["x", "y"]].to_numpy()

    tree = BallTree(fiber_coords)

    cell_coords = adata.obs[["X_centroid", "Y_centroid"]].to_numpy()

    _, idx = tree.query(cell_coords, k=1)

    fiber_vec = fiber_df[["vx", "vy"]].to_numpy()[idx.flatten()]

    cell_angles = np.deg2rad(adata.obs[cell_orientation_key])

    cell_vec = np.vstack(
        (np.cos(cell_angles), np.sin(cell_angles))
    ).T

    alignment = np.abs(np.sum(cell_vec * fiber_vec, axis=1))

    adata.obs["cell_fiber_alignment"] = alignment

    return adata