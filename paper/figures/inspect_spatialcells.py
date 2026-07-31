"""
Diagnostic: sweep alpha values to find one that reproduces ~16 components
and maximises biologically meaningful holes.

Run:
    conda activate spatial-cells-env
    python notebooks/inspect_spatialcells.py
"""

import numpy as np
import pandas as pd
import anndata as ad
import spatialcells as spc

SAMPLE    = "JRP112"
FIBER_CSV = f"data/RA_OA/{SAMPLE}/ark_wdir/fiber_segmentation_processed_data_CHP/fiber_object_table.csv"
X_KEY, Y_KEY = "X_centroid", "Y_centroid"
EPS_PX    = 1000
MIN_SAMPLES = 5
MIN_COMMUNITY_FIBERS = 20

# ── Load and cluster ───────────────────────────────────────────────────────────
df = pd.read_csv(FIBER_CSV)
df = df.rename(columns={"centroid-0": Y_KEY, "centroid-1": X_KEY}).dropna(subset=[X_KEY, Y_KEY])
print(f"CHP fibers: {len(df)}")

obs = df[[X_KEY, Y_KEY]].copy().reset_index(drop=True)
obs.index = obs.index.astype(str)
obs["CHP"] = True
fiber_adata = ad.AnnData(X=np.zeros((len(obs), 1), dtype=np.float32), obs=obs)

community_col = "COI_community"
ret = spc.spatial.getCommunities(
    fiber_adata, ["CHP"], eps=EPS_PX,
    newcolumn=community_col, min_samples=MIN_SAMPLES, core_only=True,
)
community_list = ret[0]
community_idx_list = [idx for n_fibers, idx in community_list if n_fibers >= MIN_COMMUNITY_FIBERS]
print(f"Communities >= {MIN_COMMUNITY_FIBERS} fibers: {len(community_idx_list)}")
print(f"Community sizes: {community_list}")

# ── Sweep alpha ────────────────────────────────────────────────────────────────
print(f"\n{'alpha':>6}  {'n_geoms':>8}  {'n_holes_raw':>12}  {'n_holes_ge10k':>14}  {'n_holes_ge50k':>14}")
print("-" * 60)

for alpha in [50, 100, 150, 200, 300, 500, 750, 1000]:
    try:
        boundaries = spc.spatial.getBoundary(
            fiber_adata, community_col, community_idx_list, alpha=alpha, debug=False,
        )
        if isinstance(boundaries, tuple):
            boundaries = boundaries[0]

        n_geoms = len(list(boundaries.geoms)) if hasattr(boundaries, "geoms") else 1

        all_holes = spc.spa.getHoles(boundaries)
        areas = [h.area for h in all_holes]
        n_raw   = len(areas)
        n_10k   = sum(1 for a in areas if a >= 10_000)
        n_50k   = sum(1 for a in areas if a >= 50_000)
        print(f"{alpha:>6}  {n_geoms:>8}  {n_raw:>12}  {n_10k:>14}  {n_50k:>14}")
    except Exception as e:
        print(f"{alpha:>6}  ERROR: {e}")
