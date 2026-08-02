"""Spatial regression, enrichment, and fiber orientation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    import anndata as ad
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import BallTree

from ._helpers import _ensure_list

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

