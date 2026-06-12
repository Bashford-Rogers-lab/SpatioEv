"""Configuration dataclasses for SpatioEv workflows.

These lightweight dataclasses carry validated parameters through QC,
normalization, and clustering steps, making it easy to persist and
reproduce analysis configurations.

Examples
--------
>>> from spatioev.config import QCConfig, ClusteringConfig
>>> qc_cfg = QCConfig(pixel_size=0.325, min_area_um2=10, max_area_um2=800)
>>> cl_cfg = ClusteringConfig(markers=["CD8", "CD3", "Ki67"], resolution=0.8)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class QCConfig:
    """Parameters for segmentation quality control.

    Parameters
    ----------
    pixel_size : float
        Physical size of one pixel in micrometres (µm).  Used to convert
        pixel-area to µm².  Typical values: 0.325 µm (CODEX/MIBI), 0.5 µm
        (CyCIF).
    min_area_um2 : float
        Minimum cell area in µm².  Cells smaller than this threshold are
        flagged as debris or fragments.  Default is 10 µm².
    max_area_um2 : float
        Maximum cell area in µm².  Cells larger than this threshold are
        flagged as potential over-segmentation or merged cells.
        Default is 1000 µm².
    max_nc_ratio : float
        Maximum nucleus-to-cytoplasm ratio.  Values > 1.0 indicate the
        nucleus is larger than the cytoplasm estimate and likely reflect
        segmentation errors.  Default is 1.0.

    Examples
    --------
    >>> from spatioev.config import QCConfig
    >>> cfg = QCConfig(pixel_size=0.325)
    >>> cfg.min_area_um2
    10
    """

    pixel_size: float = 0.325
    min_area_um2: float = 10
    max_area_um2: float = 1000
    max_nc_ratio: float = 1.0


@dataclass
class ClusteringConfig:
    """Parameters for unsupervised cell clustering.

    Parameters
    ----------
    markers : list[str]
        Names of marker columns in ``adata.var_names`` to use for
        clustering.  Must be present in the AnnData object.
    resolution : float
        Leiden/Louvain community detection resolution.  Higher values
        produce more, smaller clusters.  Default is 0.5.
    n_neighbors : int
        Number of neighbours for the kNN graph construction (passed to
        ``scanpy.pp.neighbors``).  Default is 10.
    n_pcs : int
        Number of principal components used before kNN graph
        construction.  Default is 15.
    scale : bool
        If ``True``, z-score each marker column before PCA.
        Default is ``True``.

    Examples
    --------
    >>> from spatioev.config import ClusteringConfig
    >>> cfg = ClusteringConfig(
    ...     markers=["CD8", "CD3", "Ki67", "panCK"],
    ...     resolution=0.8,
    ...     n_neighbors=15,
    ... )
    """

    markers: list[str] = field(default_factory=list)
    resolution: float = 0.5
    n_neighbors: int = 10
    n_pcs: int = 15
    scale: bool = True


__all__ = ["QCConfig", "ClusteringConfig"]
