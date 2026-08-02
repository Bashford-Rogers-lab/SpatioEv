"""Spatial density computation.

Split into submodules by the density definition used:

    tiles        regular-grid binning and Gaussian KDE
    knn          k-nearest-neighbour local density
    radius       fixed-radius local density
    interaction  pairwise phenotype interaction density

The plotting counterparts live in :mod:`spatioev.pl.density` and are
re-exported from :mod:`spatioev.pl`; they are re-exported here too so that
``from spatioev.tl.density import plot_density_heatmap`` keeps working.

All names remain importable directly from this package, so
``from spatioev.tl.density import compute_general_density`` is unchanged.
"""

from __future__ import annotations

from ...pl.density import (
    plot_density_correlation,
    plot_density_heatmap,
    plot_interaction_density,
    plot_interaction_distribution,
    plot_interaction_overlay,
    plot_kde_density,
    plot_local_density_map,
    plot_phenotype_density_heatmap,
)
from .interaction import phenotype_interaction_density
from .knn import compute_local_density_all_cells, compute_local_density_by_phenotype
from .radius import compute_radius_density
from .tiles import (
    assign_tiles,
    compute_general_density,
    compute_kde_density,
    compute_phenotype_density,
    phenotype_density_correlation,
)

__all__ = [
    "assign_tiles",
    "compute_general_density",
    "compute_phenotype_density",
    "phenotype_density_correlation",
    "compute_kde_density",
    "plot_density_heatmap",
    "plot_phenotype_density_heatmap",
    "plot_density_correlation",
    "plot_kde_density",
    "compute_local_density_by_phenotype",
    "compute_local_density_all_cells",
    "plot_local_density_map",
    "compute_radius_density",
    "phenotype_interaction_density",
    "plot_interaction_density",
    "plot_interaction_overlay",
    "plot_interaction_distribution",
]
