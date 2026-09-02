"""Vendored subset of scimap (MIT, Laboratory of Systems Pharmacology @ Harvard).

Only the two dependency-light algorithmic functions SpatioEv uses are vendored:

- :func:`phenotype_cells` — prior-knowledge hierarchical phenotyping (numpy/pandas)
- :func:`rescale` — gate-based 0–1 marker rescaling with GMM fallback (adds sklearn)

The interactive Napari functions (``napariGater``, ``image_viewer``) are *not*
vendored — they genuinely need napari and PyQt6 and remain an optional import
from an installed scimap. See ``spatioev.tl.phenotype``.
"""

from .phenotype_cells import phenotype_cells
from .rescale import rescale

__all__ = ["phenotype_cells", "rescale"]
