"""Cell-ECM proximity, fiber density, and cross-type Ripley K."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    import anndata as ad

from ._helpers import (
    _cross_ripleys_curve_from_links,
    _ensure_list,
    _filter_cells,
    _filter_fibers,
    _label_suffix,
    _permute_mask_within_images,
    _subset_links,
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
