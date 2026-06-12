"""Xenium and spatial-transcriptomics API for SpatioEv.

``spatioev.xe`` provides cell-type annotation, niche feature scoring, and
DAPI nuclear feature extraction for 10x Xenium data.

Submodules::

    from spatioev.xe.annotation import compute_marker_set_scores
    from spatioev.xe.features import score_xenium_histology_modules

Or via the namespace shorthand::

    import spatioev as sv
    fm = sv.xe.available_feature_map()
    adata = sv.xe.score_xenium_histology_modules(adata, feature_map=fm)

Submodules
----------
annotation  Marker-set scoring, cluster summaries, rule-based labelling
features    Pre-built histology module scoring and DAPI feature map
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "compute_marker_set_scores": "spatioev.xe.annotation",
    "summarize_cluster_marker_scores": "spatioev.xe.annotation",
    "assign_labels_from_marker_rules": "spatioev.xe.annotation",
    "available_feature_map": "spatioev.xe.features",
    "score_xenium_histology_modules": "spatioev.xe.features",
    "add_module_scores": "spatioev.xe.features",
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
