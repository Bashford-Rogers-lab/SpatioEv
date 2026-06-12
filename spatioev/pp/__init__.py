"""Preprocessing and quality-control API for SpatioEv.

``spatioev.pp`` follows the familiar ``pp`` convention used by Scanpy and
SCIMAP. Functions are grouped into focused submodules::

    from spatioev.pp.qc import run_segmentation_qc
    from spatioev.pp.normalize import zscore_normalize
    from spatioev.pp.pixel import extract_cell_pixel_features
    from spatioev.pp.spatial_prep import validate_spatial_coordinates

Or via the namespace shorthand::

    import spatioev as sv
    adata = sv.pp.run_segmentation_qc(adata, QCConfig(pixel_size=0.325))

Submodules
----------
qc           Segmentation quality control (area, NC ratio filtering, summaries)
normalize    Marker z-score normalization and obs feature construction
pixel        Per-cell pixel feature extraction (texture, morphology, DAPI)
spatial_prep Spatial coordinate validation, convex hull, tissue areas, edge cells
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    # Config objects (re-exported here for convenience)
    "QCConfig": "spatioev.config",
    "ClusteringConfig": "spatioev.config",
    # QC
    "compute_area_um2": "spatioev.pp.qc",
    "categorize_area": "spatioev.pp.qc",
    "categorize_nc_ratio": "spatioev.pp.qc",
    "filter_segmentation_errors": "spatioev.pp.qc",
    "run_segmentation_qc": "spatioev.pp.qc",
    "generate_qc_summary": "spatioev.pp.qc",
    # Normalization
    "zscore_normalize": "spatioev.pp.normalize",
    "add_obs_from_var": "spatioev.pp.normalize",
    "add_zscore_obs_features": "spatioev.pp.normalize",
    # Spatial prep
    "validate_spatial_coordinates": "spatioev.pp.spatial_prep",
    "compute_tissue_areas": "spatioev.pp.spatial_prep",
    "detect_edge_cells": "spatioev.pp.spatial_prep",
    # Pixel features
    "extract_cell_pixel_features_for_fov": "spatioev.pp.pixel",
    "extract_cell_pixel_features": "spatioev.pp.pixel",
    "extract_xenium_dapi_features": "spatioev.pp.pixel",
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
