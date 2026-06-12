"""QC visualisation helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

try:
    import anndata as ad
    _AnnData = ad.AnnData
except ImportError:  # pragma: no cover
    _AnnData = object  # type: ignore[assignment,misc]


def plot_area_distribution(
    adata: _AnnData,
    min_area: float | None = None,
    max_area: float | None = None,
) -> plt.Figure:
    """Plot a histogram of cell area (µm²) with optional QC threshold lines.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object with ``area_um2`` in ``adata.obs``.
    min_area : float or None
        Lower area threshold drawn as a red vertical line.
    max_area : float or None
        Upper area threshold drawn as a green vertical line.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object. Call ``plt.show()`` in a notebook or save with
        ``fig.savefig(...)`` as needed.
    """
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.histplot(adata.obs["area_um2"], bins=50, ax=ax)

    if min_area is not None:
        ax.axvline(min_area, color="red", linewidth=1, label=f"min {min_area}")
    if max_area is not None:
        ax.axvline(max_area, color="green", linewidth=1, label=f"max {max_area}")

    ax.set_title("Cell area distribution")
    ax.set_xlabel("Area (µm²)")
    ax.set_ylabel("Cell count")
    if min_area is not None or max_area is not None:
        ax.legend(fontsize=8)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_nc_ratio_distribution(
    adata: _AnnData,
    max_ratio: float | None = None,
) -> plt.Figure:
    """Plot a histogram of nucleus-to-cytoplasm ratios with an optional cutoff.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object with ``nc_ratio`` in ``adata.obs``.
    max_ratio : float or None
        NC-ratio threshold drawn as a red vertical line.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object. Call ``plt.show()`` in a notebook or save with
        ``fig.savefig(...)`` as needed.
    """
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.histplot(adata.obs["nc_ratio"], bins=50, ax=ax)

    if max_ratio:
        ax.axvline(max_ratio, color="red", linewidth=1, label=f"max NC {max_ratio}")
        ax.legend(fontsize=8)

    ax.set_title("Nuclear-to-cell ratio distribution")
    ax.set_xlabel("NC ratio")
    ax.set_ylabel("Cell count")
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


__all__ = [
    "plot_area_distribution",
    "plot_nc_ratio_distribution",
]
