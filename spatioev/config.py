from dataclasses import dataclass
from typing import List

@dataclass
class QCConfig:
    pixel_size: float = 0.325
    min_area_um2: float = 10
    max_area_um2: float = 1000
    max_nc_ratio: float = 1.0

@dataclass
class ClusteringConfig:

    markers: List[str]

    resolution: float = 0.5
    n_neighbors: int = 10
    n_pcs: int = 15

    scale: bool = True