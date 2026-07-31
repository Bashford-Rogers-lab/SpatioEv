"""Shared internal building blocks.

Private to SpatioEv: nothing here is part of the public API. These helpers
exist so that the analysis modules stop re-deriving the same spatial
primitives (neighbour graphs, weight matrices, coordinate extraction) in
every function.
"""

from .neighbors import knn_weights, resolve_k

__all__ = ["knn_weights", "resolve_k"]
