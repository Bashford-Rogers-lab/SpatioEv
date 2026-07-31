"""Spatial neighbour graphs and weight matrices.

The k-nearest-neighbour weight matrix ``W`` is the single most reused object
in SpatioEv's spatial statistics, and it depends only on the coordinates.
Building it repeatedly — in particular once per iteration of a permutation
loop, where the coordinates never change — dominates the cost of the
statistics pipeline.

Treat ``W`` as a value to compute once and pass around.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.neighbors import kneighbors_graph

__all__ = ["knn_weights", "resolve_k"]


def resolve_k(n_obs: int, k: int) -> int | None:
    """Clamp *k* to a value a kNN graph over *n_obs* points can support.

    Returns ``None`` when no graph is possible (fewer than two points).
    """
    if n_obs < 2:
        return None
    return min(k, n_obs - 1)


def knn_weights(
    coords: np.ndarray,
    k: int,
    *,
    normalize: bool = False,
) -> sparse.csr_matrix:
    """Build a k-nearest-neighbour spatial weight matrix.

    Parameters
    ----------
    coords : array-like of shape (n, 2)
        Spatial coordinates. Must already be finite; callers are expected to
        have dropped non-finite rows.
    k : int
        Number of neighbours. Clamped to ``n - 1``.
    normalize : bool
        If ``True``, row-normalise so each row sums to 1. If ``False``
        (default) return the binary connectivity matrix, which is what the
        classical Moran's I normalisation ``n / S0`` expects.

    Returns
    -------
    scipy.sparse.csr_matrix of shape (n, n)

    Notes
    -----
    Self-neighbours are excluded.
    """
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]

    k_eff = resolve_k(n, k)
    if k_eff is None:
        raise ValueError("At least two points are required to build a kNN graph.")

    W = kneighbors_graph(coords, k_eff, mode="connectivity", include_self=False)
    W = sparse.csr_matrix(W)

    if normalize:
        row_sums = np.asarray(W.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        W = sparse.diags(1.0 / row_sums) @ W
        W = sparse.csr_matrix(W)

    return W


def morans_i_from_weights(W: sparse.spmatrix, values: np.ndarray) -> float:
    """Global Moran's I given a precomputed weight matrix.

    ``I = (n / S0) * (x' W x) / (x' x)`` with ``x`` the mean-centred values
    and ``S0`` the sum of all weights.
    """
    values = np.asarray(values, dtype=float)
    n = values.size

    x = values - values.mean()
    denom = float(x @ x)
    if denom == 0:
        return np.nan

    s0 = float(W.sum())
    if s0 == 0:
        return np.nan

    return (n / s0) * float(x @ (W @ x)) / denom


def morans_i_batch(W: sparse.spmatrix, value_matrix: np.ndarray) -> np.ndarray:
    """Global Moran's I for many value vectors sharing one weight matrix.

    Parameters
    ----------
    W : sparse matrix of shape (n, n)
    value_matrix : array of shape (n, b)
        Each column is an independent set of values over the same points.

    Returns
    -------
    numpy.ndarray of shape (b,)

    Notes
    -----
    This is the vectorised form used by the permutation tests: a single sparse
    matrix product replaces ``b`` separate ones, and ``W`` is built once
    instead of ``b`` times.
    """
    V = np.asarray(value_matrix, dtype=float)
    if V.ndim == 1:
        V = V[:, None]

    n = V.shape[0]
    Vc = V - V.mean(axis=0, keepdims=True)

    denom = np.einsum("ij,ij->j", Vc, Vc)
    numer = np.einsum("ij,ij->j", Vc, W @ Vc)

    s0 = float(W.sum())
    out = np.full(V.shape[1], np.nan, dtype=float)
    valid = (denom != 0) & np.isfinite(denom)
    if s0 != 0:
        out[valid] = (n / s0) * numer[valid] / denom[valid]
    return out
