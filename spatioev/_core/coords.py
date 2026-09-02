"""Coordinate extraction, per-image iteration, and label utilities.

These are the primitives the analysis modules were each re-deriving. Keeping
one implementation means a change to (say) how non-finite coordinates are
dropped applies everywhere instead of in one module at a time.

Private to SpatioEv: nothing here is public API.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "ensure_list",
    "label_suffix",
    "clean_coords",
    "get_coords",
    "require_obs_columns",
    "per_image",
    "per_image_table",
]


def ensure_list(value: Any) -> list | None:
    """Normalize a scalar or iterable label selection to a list."""
    if value is None:
        return None

    if isinstance(value, str):
        return [value]

    return list(value)


def label_suffix(value: Any, default: str) -> str:
    """Build a stable suffix for derived column names."""
    value = ensure_list(value)

    if value is None or len(value) == 0:
        return default

    return "_".join(map(str, value))


def clean_coords(coords: np.ndarray) -> np.ndarray:
    """Drop rows with non-finite x/y coordinates."""
    coords = np.asarray(coords, dtype=float)

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("Coordinates must be an Nx2 array.")

    return coords[np.isfinite(coords).all(axis=1)]


def require_obs_columns(obs: pd.DataFrame, columns, name: str = "table") -> None:
    """Raise a single, informative error listing every missing column."""
    columns = ensure_list(columns) or []
    missing = [c for c in columns if c is not None and c not in obs.columns]
    if missing:
        raise ValueError(
            f"{name} is missing required column(s): {', '.join(map(str, missing))}"
        )


def get_coords(
    source,
    x_key: str = "X_centroid",
    y_key: str = "Y_centroid",
    *,
    clean: bool = False,
) -> np.ndarray:
    """Extract an (n, 2) coordinate array from an AnnData or DataFrame.

    Parameters
    ----------
    source : anndata.AnnData or pandas.DataFrame
        ``AnnData`` inputs read from ``.obs``.
    clean : bool
        If ``True``, drop rows with non-finite coordinates.
    """
    obs = getattr(source, "obs", source)
    require_obs_columns(obs, [x_key, y_key], name="coordinate table")

    coords = obs[[x_key, y_key]].to_numpy(dtype=float)
    return clean_coords(coords) if clean else coords


def per_image(
    source,
    image_key: str = "imageid",
) -> Iterator[tuple[Any, pd.Index]]:
    """Yield ``(image_id, index)`` for each image, in order of appearance.

    Replaces the hand-written ``for img in obs[image_key].unique(): idx =
    obs.index[obs[image_key] == img]`` block, which appeared eight times in
    the statistics module alone.
    """
    obs = getattr(source, "obs", source)
    require_obs_columns(obs, [image_key], name="table")

    # Series.unique() preserves order of appearance, matching the hand-written
    # loops this replaces (important: results are compared row-by-row).
    for image_id in obs[image_key].unique():
        yield image_id, obs.index[obs[image_key] == image_id]


def per_image_table(
    adata,
    value_keys: list[str],
    fn,
    *,
    x_key: str = "X_centroid",
    y_key: str = "Y_centroid",
    image_key: str = "imageid",
    extra: dict | None = None,
    result_key: str | None = None,
    rng=None,
    **fn_kwargs,
) -> pd.DataFrame:
    """Apply a coordinate-and-values statistic to each image, as a table.

    Collapses the ``for img in obs[image_key].unique(): ...`` block that every
    ``*_by_image`` wrapper repeated verbatim.

    Parameters
    ----------
    value_keys : list of str
        ``adata.obs`` columns passed positionally to *fn* after ``coords``.
    fn : callable
        Called as ``fn(coords, *values, **fn_kwargs)``.
    extra : dict, optional
        Constant columns inserted after the image id, before the result.
    result_key : str, optional
        If given, the return of *fn* is stored under this column. Otherwise
        the return is treated as a mapping and merged into the row.
    rng : numpy.random.Generator, optional
        When supplied, a fresh ``random_state`` is drawn per image and passed
        to *fn*, so per-image permutation tests stay independent yet
        reproducible from one seed.
    """
    obs = getattr(adata, "obs", adata)
    require_obs_columns(obs, [x_key, y_key, image_key, *value_keys], name="adata.obs")

    rows = []
    for image_id, idx in per_image(obs, image_key):
        coords = obs.loc[idx, [x_key, y_key]].to_numpy()
        values = [obs.loc[idx, key].to_numpy() for key in value_keys]

        call_kwargs = dict(fn_kwargs)
        if rng is not None:
            call_kwargs["random_state"] = int(rng.integers(0, np.iinfo(np.int32).max))

        result = fn(coords, *values, **call_kwargs)

        row = {image_key: image_id, **(extra or {})}
        if result_key is not None:
            row[result_key] = result
        else:
            row.update(result)
        rows.append(row)

    return pd.DataFrame(rows)
