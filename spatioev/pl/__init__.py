"""Plotting API for SpatioEv.

``spatioev.pl`` provides plotting functions for QC, spatial scatter,
density, interaction, niche, and feature visualizations.

Functions are grouped into submodules::

    from spatioev.pl.qc import plot_area_distribution
    from spatioev.pl.spatial import plot_niche_boundaries, spatial_scatter_plot

Or via the namespace shorthand::

    import spatioev as sv
    sv.pl.spatial_scatter_plot(adata, color="phenotype")
    sv.pl.plot_niche_boundaries(adata, boundaries)

Submodules
----------
qc       Morphology distribution plots (area, NC ratio)
spatial  Spatial scatter, cluster heatmaps, density overlays, niche plots
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    # QC plots
    "plot_area_distribution": "spatioev.pl.qc",
    "plot_nc_ratio_distribution": "spatioev.pl.qc",
    # Phenotype / cluster visualization
    "plot_cluster_heatmap": "spatioev.pl.spatial",
    "plot_refinement_umaps": "spatioev.pl.spatial",
    "spatial_scatter_plot": "spatioev.pl.spatial",
    "inspect_clusters": "spatioev.pl.spatial",
    # Spatial feature plots
    "plot_spatial_feature": "spatioev.pl.spatial",
    "plot_spatial_category": "spatioev.pl.spatial",
    "plot_niche_boundaries": "spatioev.pl.spatial",
    "plot_correlation_heatmap": "spatioev.pl.spatial",
    # Density plots (re-exported from tl.density for convenience)
    "plot_density_heatmap": "spatioev.tl.density",
    "plot_phenotype_density_heatmap": "spatioev.tl.density",
    "plot_density_correlation": "spatioev.tl.density",
    "plot_kde_density": "spatioev.tl.density",
    "plot_local_density_map": "spatioev.tl.density",
    # Interaction plots
    "plot_interaction_density": "spatioev.tl.density",
    "plot_interaction_overlay": "spatioev.tl.density",
    "plot_interaction_distribution": "spatioev.tl.density",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module = import_module(_EXPORTS[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
