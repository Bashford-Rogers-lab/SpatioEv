"""Uniform handling of optional heavy dependencies.

SpatioEv keeps Scanpy, scimap, Napari and Shapely-backed geometry out of the
import path until they are actually needed. Each call site used to write its
own try/except with its own wording; this centralises the message so the
install hint is consistent and correct.

Private to SpatioEv: nothing here is public API.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

__all__ = ["require"]

# Maps a module name to the extra that provides it.
_EXTRA_FOR = {
    "scanpy": "scanpy",
    "scimap": "gating",
    "napari": "gating",
    "shapely": None,  # core dependency; a missing shapely means a broken env
    "squidpy": "spatialdata",
    "spatialdata": "spatialdata",
    "umap": "trajectory",
    "elpigraph": "trajectory",
}


def require(module: str, feature: str) -> ModuleType:
    """Import *module*, or raise an ImportError naming *feature* and the fix.

    Parameters
    ----------
    module : str
        Importable module name, e.g. ``"scanpy"``.
    feature : str
        Human-readable description of what needs it, used in the message.

    Returns
    -------
    ModuleType
    """
    try:
        return import_module(module)
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        extra = _EXTRA_FOR.get(module)
        if extra is None:
            hint = f"Install it with `pip install {module}`."
        else:
            hint = (
                f"Install SpatioEv with `pip install 'spatioev[{extra}]'` "
                f"or install `{module}` directly."
            )
        raise ImportError(
            f"{feature} requires the optional dependency '{module}'. {hint}"
        ) from exc
