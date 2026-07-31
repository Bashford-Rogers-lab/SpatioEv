#!/usr/bin/env python3
"""
09b_spatialcells_col6_dark_segmentation.py

Replaces the raster-based COL6 dark zone segmentation notebook
(09_RA_OA_ECM_cell_05_chp_density_micro_holes_col6_dark_zone_segmentation.ipynb)
with the spatialcells CHP community + alpha-shape hole detection approach.

For each of the 6 RA/OA samples the pipeline does:
  1. Load CHP fiber centroids from fiber_object_table.csv
  2. Detect CHP fiber communities via DBSCAN (spatialcells.spatial.getCommunities)
  3. Build alpha-shape polygon boundaries per community (spatialcells.spatial.getBoundary)
  4. Prune small/noisy boundary components (spatialcells.spatial.pruneSmallComponents)
  5. Extract geometric holes inside CHP communities (spatialcells.spa.getHoles)
  6. Assign cells (from tmp6.h5ad) to dark-zone regions using Shapely:
       core         =  inside a hole
       inner_edge   =  0 – INNER_EDGE_UM µm outside the hole boundary
       outer_edge   =  INNER_EDGE_UM – OUTER_EDGE_UM µm outside the hole boundary
       full_edge    =  inner_edge ∪ outer_edge  (convenience aggregate)
       background   =  everything else
  7. Compute region areas from polygon geometry (mm²)
  8. Save one QC scatter plot per sample

Output files (written to DARK_DIR, replacing raster-based CSVs):
  04_cell_chp_micro_hole_region_assignments.csv
  03_chp_micro_hole_image_region_edge_metrics.csv
  06_cell_region_phenotype_counts.csv

These three files are consumed unchanged by:
  09_RA_OA_ECM_cell_spatioev_module_paper_applications.ipynb

Run from the SpatioEv project root or from the notebooks/ directory:
  python notebooks/09b_spatialcells_col6_dark_segmentation.py
"""

from __future__ import annotations

from pathlib import Path
import warnings

import anndata as ad
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spatialcells as spc
from shapely.ops import unary_union
from shapely.prepared import prep

# ── Paths ──────────────────────────────────────────────────────────────────────
# Resolve project root whether the script is run from root or notebooks/.
_HERE = Path(__file__).resolve().parent
# Walk up to the repository root (the directory containing pyproject.toml).
PROJECT_ROOT = next(
    (p for p in (_HERE, *_HERE.parents) if (p / "pyproject.toml").is_file()),
    _HERE,
)

DATA_DIR    = PROJECT_ROOT / "data" / "RA_OA"
RESULTS_DIR = PROJECT_ROOT  / "paper" / "notebooks" / "results" / "ra_oa_ecm_cell"
DARK_DIR    = RESULTS_DIR / "chp_density_micro_holes_col6_dark_segmentation" / "outputs"
FIGURES_DIR = RESULTS_DIR / "chp_density_micro_holes_col6_dark_segmentation" / "figures" / "spatialcells_qc"
ADATA_PATH  = DATA_DIR / "tmp6.h5ad"

DARK_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Column keys ────────────────────────────────────────────────────────────────
IMAGE_KEY     = "imageid"
GROUP_KEY     = "pathology"
PHENOTYPE_KEY = "phenotype"
X_KEY         = "X_centroid"
Y_KEY         = "Y_centroid"
FOV_KEY       = "fov"

# ── Physical constants ──────────────────────────────────────────────────────────
PIXEL_SIZE_UM = 0.325          # µm per pixel
PIXEL_SIZE_MM = PIXEL_SIZE_UM / 1000
PX2_TO_MM2    = PIXEL_SIZE_MM ** 2    # mm² per px²
UM2_TO_MM2    = 1e-6

# ── spatialcells parameters ─────────────────────────────────────────────────────
# DBSCAN eps for CHP fiber community detection.
# 325 µm = 1000 px at 0.325 µm/px — matches the original JRP112 notebook.
EPS_UM      = 325.0
EPS_PX      = EPS_UM / PIXEL_SIZE_UM    # 1000 px
MIN_SAMPLES = 5

# Alpha value for alpha-shape boundary construction.
# Matches the original JRP112 spatialcells notebook (alpha=100).
# With the current fiber data (~2,727 fibers vs ~4,252 in the original) this
# produces 98 alpha-shape components and ~44 holes. Use alpha=150 if you want
# to match the original's 16-component structure (add to SAMPLE_OVERRIDES).
ALPHA = 100

# Minimum number of CHP fibers for a community to be included in boundary/hole detection.
MIN_COMMUNITY_FIBERS = 20

# ── Concentric zone widths ──────────────────────────────────────────────────────
INNER_EDGE_UM = 30.0    # µm: distance outside hole → inner_edge
OUTER_EDGE_UM = 100.0   # µm: distance outside inner_edge → outer_edge
INNER_EDGE_PX = INNER_EDGE_UM / PIXEL_SIZE_UM
OUTER_EDGE_PX = OUTER_EDGE_UM / PIXEL_SIZE_UM

# ── Samples ────────────────────────────────────────────────────────────────────
SAMPLE_IDS = ["JRP112", "JRP122", "JRP141", "JRP144", "S00292623", "S00293087"]

# Per-sample parameter overrides.
# After reviewing QC plots, add entries here if a sample needs different eps or alpha.
# Example: SAMPLE_OVERRIDES = {"JRP141": {"EPS_PX": 500, "ALPHA": 150}}
SAMPLE_OVERRIDES: dict[str, dict] = {}

# ── matplotlib publication style ───────────────────────────────────────────────
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})


# ──────────────────────────────────────────────────────────────────────────────
# Helper: JRP141 FOV coordinate correction
# (inherited from 09_RA_OA_ECM_cell_00_prepare_links.ipynb)
# ──────────────────────────────────────────────────────────────────────────────
def apply_jrp141_fiber_corrections(fiber_df: pd.DataFrame) -> pd.DataFrame:
    """Apply FOV stitching offsets for JRP141 fiber coordinates."""
    df = fiber_df.copy()
    is_141 = df[IMAGE_KEY].eq("JRP141")
    df.loc[is_141 & df[FOV_KEY].eq("fov1"), X_KEY] += 25818
    df.loc[is_141 & df[FOV_KEY].eq("fov2"), Y_KEY] += 24880
    df.loc[is_141 & df[FOV_KEY].eq("fov3"), X_KEY] += 25818
    df.loc[is_141 & df[FOV_KEY].eq("fov3"), Y_KEY] += 24880
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Fiber loading
# ──────────────────────────────────────────────────────────────────────────────
def load_chp_fibers(image_id: str) -> pd.DataFrame:
    """Load CHP fiber centroids for one sample and apply coordinate corrections."""
    csv_path = (
        DATA_DIR / image_id / "ark_wdir"
        / "fiber_segmentation_processed_data_CHP"
        / "fiber_object_table.csv"
    )
    df = pd.read_csv(csv_path)
    # centroid-0 is row (Y), centroid-1 is column (X) in pixel coordinates
    df = df.rename(columns={"centroid-0": Y_KEY, "centroid-1": X_KEY})
    df[IMAGE_KEY] = image_id
    df = df.dropna(subset=[X_KEY, Y_KEY])
    if image_id == "JRP141":
        df = apply_jrp141_fiber_corrections(df)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Build minimal AnnData for spatialcells
# ──────────────────────────────────────────────────────────────────────────────
def make_fiber_adata(chp_df: pd.DataFrame) -> ad.AnnData:
    """
    Create a minimal AnnData from CHP fiber coordinates for spatialcells.

    spatialcells.spatial.getCommunities reads X_centroid and Y_centroid from
    adata.obs and uses a boolean column ('CHP') to select points for clustering.
    """
    obs = chp_df[[X_KEY, Y_KEY, FOV_KEY]].copy().reset_index(drop=True)
    obs.index = obs.index.astype(str)
    obs["CHP"] = True
    # AnnData requires at least a 1-column X matrix
    dummy_X = np.zeros((len(obs), 1), dtype=np.float32)
    return ad.AnnData(X=dummy_X, obs=obs)


# ──────────────────────────────────────────────────────────────────────────────
# spatialcells pipeline: communities → boundaries → holes
# ──────────────────────────────────────────────────────────────────────────────
def run_spatialcells_pipeline(
    fiber_adata: ad.AnnData,
    eps_px: float = EPS_PX,
    alpha: int = ALPHA,
) -> tuple:
    """
    Run CHP community detection → alpha-shape boundaries → hole extraction.

    Returns (holes, boundaries):
      - holes      : list of Shapely Polygon objects (geometric holes inside communities)
      - boundaries : raw Shapely geometry returned by getBoundary (for QC plotting),
                     or None if the pipeline failed early.
    """
    if len(fiber_adata) == 0:
        print("    No CHP fibers — skipping.")
        return [], None

    # Step 1: DBSCAN community detection on CHP fiber centroids
    community_col = "COI_community"
    ret = spc.spatial.getCommunities(
        fiber_adata,
        ["CHP"],
        eps=eps_px,
        newcolumn=community_col,
        min_samples=MIN_SAMPLES,
        core_only=True,
    )
    community_list = ret[0]   # [(n_fibers, cluster_id), ...] sorted descending
    print(f"    DBSCAN communities found: {len(community_list)}")
    if not community_list:
        print("    No communities — skipping.")
        return [], None

    # Step 2: select communities above minimum size threshold
    community_idx_list = [
        idx for n_fibers, idx in community_list if n_fibers >= MIN_COMMUNITY_FIBERS
    ]
    print(f"    Communities ≥ {MIN_COMMUNITY_FIBERS} fibers: {len(community_idx_list)}")
    if not community_idx_list:
        print("    No qualifying communities — skipping.")
        return [], None

    # Step 3: alpha-shape boundary construction.
    # spatialcells 1.0.1 returns a single geometry object (MultiPolygon),
    # not the 3-tuple (boundaries, polygons, edge_components) of older versions.
    try:
        boundaries = spc.spatial.getBoundary(
            fiber_adata,
            community_col,
            community_idx_list,
            alpha=alpha,
            debug=False,
        )
    except Exception as exc:
        warnings.warn(f"    getBoundary failed: {exc}")
        return [], None

    if boundaries is None or (hasattr(boundaries, "is_empty") and boundaries.is_empty):
        print("    getBoundary returned empty geometry — skipping.")
        return [], None
    print(f"    Boundary geometry type: {type(boundaries).__name__}")

    # Step 4: extract ALL geometric holes from the unpruned boundaries.
    # The original notebook used all holes from getHoles with no area filter
    # (it used a manual ROI polygon instead, which we skip here to analyse the
    # full tissue).  We keep every hole getHoles returns.
    holes = spc.spa.getHoles(boundaries)
    print(f"    Holes extracted: {len(holes)}")
    return holes, boundaries


# ──────────────────────────────────────────────────────────────────────────────
# Point-in-polygon (Shapely version-agnostic)
# ──────────────────────────────────────────────────────────────────────────────
def _contains_points(geom, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """
    Boolean array: which (xs[i], ys[i]) points lie inside geom.
    Tries the vectorised Shapely 2.x API; falls back to a prepared-geometry loop.
    """
    if geom is None or geom.is_empty:
        return np.zeros(len(xs), dtype=bool)
    try:
        from shapely import prepare, contains_xy
        prepare(geom)
        return np.asarray(contains_xy(geom, xs, ys), dtype=bool)
    except (ImportError, Exception):
        from shapely.geometry import Point as _Pt
        p = prep(geom)
        return np.array([p.contains(_Pt(x, y)) for x, y in zip(xs, ys)])


# ──────────────────────────────────────────────────────────────────────────────
# Cell region assignment
# ──────────────────────────────────────────────────────────────────────────────
def assign_cells_to_regions(
    cell_df: pd.DataFrame,
    holes: list,
) -> tuple[pd.Series, pd.Series]:
    """
    Assign cells to COL6 dark-zone regions based on proximity to holes.

    Returns
    -------
    region_series    : "core" | "inner_edge" | "outer_edge" | "background"
    component_series : integer hole index for non-background cells, -1 otherwise
    """
    index = cell_df.index
    if not holes:
        return (
            pd.Series("background", index=index),
            pd.Series(-1, index=index),
        )

    xs = cell_df[X_KEY].to_numpy(dtype=float)
    ys = cell_df[Y_KEY].to_numpy(dtype=float)

    # Build merged zone geometries for fast vectorised lookup
    core_geom  = unary_union(holes)
    inner_geom = unary_union([h.buffer(INNER_EDGE_PX) for h in holes])
    outer_geom = unary_union([h.buffer(OUTER_EDGE_PX) for h in holes])

    in_core  = _contains_points(core_geom,  xs, ys)
    in_inner = _contains_points(inner_geom, xs, ys) & ~in_core
    in_outer = _contains_points(outer_geom, xs, ys) & ~in_core & ~in_inner

    region = np.where(
        in_core,  "core",
        np.where(
            in_inner, "inner_edge",
            np.where(in_outer, "outer_edge", "background"),
        ),
    )

    # Assign the index of the nearest enclosing hole as dark_component.
    # For speed, use the union geometry to first pre-filter candidates.
    component = np.full(len(xs), -1, dtype=int)
    in_any = in_core | in_inner | in_outer
    if in_any.any():
        for h_idx, hole in enumerate(holes):
            zone = hole.buffer(OUTER_EDGE_PX)
            zone_prep = prep(zone)
            from shapely.geometry import Point as _Pt
            for i in np.where(in_any & (component == -1))[0]:
                if zone_prep.contains(_Pt(xs[i], ys[i])):
                    component[i] = h_idx

    return (
        pd.Series(region, index=index),
        pd.Series(component, index=index),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Region area calculation
# ──────────────────────────────────────────────────────────────────────────────
def compute_region_areas_mm2(holes: list, cell_df: pd.DataFrame) -> dict[str, float]:
    """
    Compute mm² area for each region type.

    Background area is estimated as convex-hull of all cells minus the outer
    buffer zone — a reasonable tissue-area proxy when no explicit mask exists.
    """
    if not holes:
        from shapely.geometry import MultiPoint
        tissue_geom = MultiPoint(
            list(zip(cell_df[X_KEY].tolist(), cell_df[Y_KEY].tolist()))
        ).convex_hull
        return {
            "core":       0.0,
            "inner_edge": 0.0,
            "outer_edge": 0.0,
            "full_edge":  0.0,
            "background": tissue_geom.area * PX2_TO_MM2,
        }

    core_geom  = unary_union(holes)
    inner_geom = unary_union([h.buffer(INNER_EDGE_PX) for h in holes])
    outer_geom = unary_union([h.buffer(OUTER_EDGE_PX) for h in holes])

    core_area_mm2  = core_geom.area  * PX2_TO_MM2
    inner_area_mm2 = (inner_geom.area - core_geom.area)  * PX2_TO_MM2
    outer_area_mm2 = (outer_geom.area - inner_geom.area) * PX2_TO_MM2
    full_edge_mm2  = inner_area_mm2 + outer_area_mm2

    from shapely.geometry import MultiPoint
    tissue_geom = MultiPoint(
        list(zip(cell_df[X_KEY].tolist(), cell_df[Y_KEY].tolist()))
    ).convex_hull
    bg_area_mm2 = max(0.0, tissue_geom.area * PX2_TO_MM2 - outer_geom.area * PX2_TO_MM2)

    return {
        "core":       core_area_mm2,
        "inner_edge": inner_area_mm2,
        "outer_edge": outer_area_mm2,
        "full_edge":  full_edge_mm2,
        "background": bg_area_mm2,
    }


# ──────────────────────────────────────────────────────────────────────────────
# QC plot
# ──────────────────────────────────────────────────────────────────────────────
REGION_COLORS = {
    "core":       "#2b6cb0",
    "inner_edge": "#f6ad55",
    "outer_edge": "#c53030",
    "background": "#cbd5e0",
}

def plot_qc(
    image_id: str,
    cell_df: pd.DataFrame,
    chp_df: pd.DataFrame,
    holes: list,
    region_series: pd.Series,
    boundaries=None,
) -> None:
    """Three-panel QC plot:
      Left   : cells coloured by dark-zone region assignment
      Middle : CHP fibers + hole boundaries (extracted dark zones)
      Right  : raw alpha-shape boundary mesh (equivalent to original plotBoundary)
    """
    from shapely.geometry import Polygon as ShapelyPolygon
    fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharex=True, sharey=True)

    region_order = ["core", "inner_edge", "outer_edge", "background"]

    # ── Left panel: cells coloured by region ──────────────────────────────────
    ax = axes[0]
    for region in region_order:
        mask = region_series.eq(region)
        sub = cell_df.loc[mask]
        if sub.empty:
            continue
        ax.scatter(
            sub[X_KEY], sub[Y_KEY],
            s=0.5, color=REGION_COLORS.get(region, "grey"),
            alpha=0.6, linewidth=0, rasterized=True,
            label=f"{region} (n={mask.sum():,})",
        )
    ax.set_title(f"{image_id}: cells by dark-zone region")
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()   # shared axis — invert once; all panels inherit via sharey
    ax.legend(loc="upper right", markerscale=5, fontsize=6, framealpha=0.7)

    # ── Middle panel: CHP fibers + extracted hole boundaries ──────────────────
    ax = axes[1]
    ax.scatter(
        cell_df[X_KEY], cell_df[Y_KEY],
        s=0.3, color="#cbd5e0", alpha=0.4, linewidth=0,
        rasterized=True, label="all cells",
    )
    chp_sample = chp_df if len(chp_df) <= 20_000 else chp_df.sample(20_000, random_state=0)
    ax.scatter(
        chp_sample[X_KEY], chp_sample[Y_KEY],
        s=0.5, color="#e11d48", alpha=0.35, linewidth=0,
        rasterized=True, label="CHP fibers",
    )
    # Draw extracted hole boundaries (the true dark zones)
    first_hole_labeled = False
    for hole in holes:
        sub_geoms = list(hole.geoms) if hasattr(hole, "geoms") else [hole]
        for sub in sub_geoms:
            if not hasattr(sub, "exterior") or sub.is_empty:
                continue
            xs_h, ys_h = sub.exterior.xy
            lbl = "_nolegend_" if first_hole_labeled else f"holes (n={len(holes)})"
            ax.plot(xs_h, ys_h, color="#111827", linewidth=0.8, alpha=0.8, label=lbl)
            first_hole_labeled = True
    ax.set_title(f"{image_id}: CHP fibers + extracted holes (n={len(holes)})")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", markerscale=5, fontsize=6, framealpha=0.7)

    # ── Right panel: raw alpha-shape boundary mesh (like original plotBoundary) ──
    # Shows every polygon component of the alpha-shape — equivalent to:
    #   spc.plt.plotBoundary(boundaries, linewidth=1, color="k", ax=ax)
    ax = axes[2]
    ax.scatter(
        chp_sample[X_KEY], chp_sample[Y_KEY],
        s=0.5, color="#e11d48", alpha=0.35, linewidth=0,
        rasterized=True, label="CHP fibers",
    )
    if boundaries is not None:
        geoms = list(boundaries.geoms) if hasattr(boundaries, "geoms") else [boundaries]
        n_geoms = len(geoms)
        for geom in geoms:
            if not isinstance(geom, ShapelyPolygon) or geom.is_empty:
                continue
            # Draw exterior
            xs_b, ys_b = geom.exterior.xy
            ax.plot(xs_b, ys_b, color="#111827", linewidth=0.5, alpha=0.7)
            # Draw interior rings (interior holes within each component)
            for interior in geom.interiors:
                xs_i, ys_i = interior.xy
                ax.plot(xs_i, ys_i, color="#1d4ed8", linewidth=0.5, alpha=0.6)
        ax.set_title(
            f"{image_id}: alpha-shape boundary mesh\n"
            f"({n_geoms} components, black=exterior, blue=interior rings)"
        )
    else:
        ax.set_title(f"{image_id}: alpha-shape boundary mesh (unavailable)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", markerscale=5, fontsize=6, framealpha=0.7)

    for ax in axes:
        ax.set_xlabel(X_KEY)
        ax.set_ylabel(Y_KEY)

    fig.tight_layout()
    out_path = FIGURES_DIR / f"{image_id}_qc_spatialcells_dark_zones.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    QC plot saved → {out_path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    # ── Load combined cell metadata ────────────────────────────────────────────
    print("Loading tmp6.h5ad obs …")
    adata_full = ad.read_h5ad(ADATA_PATH)
    cell_obs = adata_full.obs[[IMAGE_KEY, GROUP_KEY, PHENOTYPE_KEY, X_KEY, Y_KEY]].copy()
    cell_obs["cell_id"] = cell_obs.index.astype(str)
    del adata_full
    print(f"  Total cells: {len(cell_obs):,}")

    # ── Per-sample results collectors ──────────────────────────────────────────
    all_assignment_rows: list[pd.DataFrame]    = []
    all_region_metric_rows: list[pd.DataFrame] = []
    all_count_rows: list[pd.DataFrame]         = []

    # ── Process each sample ────────────────────────────────────────────────────
    for image_id in SAMPLE_IDS:
        print(f"\n{'─'*60}")
        print(f"  {image_id}")

        # Resolve per-sample parameter overrides
        overrides = SAMPLE_OVERRIDES.get(image_id, {})
        eps_px = overrides.get("EPS_PX", EPS_PX)
        alpha  = overrides.get("ALPHA",  ALPHA)
        print(f"    eps={eps_px:.0f} px ({eps_px * PIXEL_SIZE_UM:.0f} µm)  alpha={alpha}")

        # ── Load CHP fibers ────────────────────────────────────────────────────
        chp_df = load_chp_fibers(image_id)
        print(f"    CHP fibers: {len(chp_df):,}")

        # ── Cell subset for this sample ────────────────────────────────────────
        cell_df = cell_obs[cell_obs[IMAGE_KEY].eq(image_id)].copy()
        pathology = cell_df[GROUP_KEY].iloc[0] if len(cell_df) else "unknown"
        print(f"    Cells: {len(cell_df):,}  pathology: {pathology}")

        # ── Run spatialcells pipeline ──────────────────────────────────────────
        fiber_adata = make_fiber_adata(chp_df)
        holes, boundaries = run_spatialcells_pipeline(fiber_adata, eps_px=eps_px, alpha=alpha)

        # ── Assign cells to regions ────────────────────────────────────────────
        region_series, component_series = assign_cells_to_regions(cell_df, holes)
        region_counts = region_series.value_counts()
        print(f"    Region counts: {dict(region_counts)}")

        # ── QC plot ────────────────────────────────────────────────────────────
        plot_qc(image_id, cell_df, chp_df, holes, region_series, boundaries=boundaries)

        # ── Build per-cell assignment table ───────────────────────────────────
        assign_df = cell_df[[IMAGE_KEY, GROUP_KEY, PHENOTYPE_KEY, X_KEY, Y_KEY, "cell_id"]].copy()
        assign_df["COL6_dark_region"] = region_series.values
        assign_df["dark_component"]   = component_series.values
        assign_df["COL6_dark_score"]  = np.nan   # spatialcells has no equivalent score
        assign_df["CHP_density_rank"] = np.nan   # spatialcells has no equivalent rank
        all_assignment_rows.append(assign_df)

        # ── Compute region areas (mm²) ─────────────────────────────────────────
        area_dict = compute_region_areas_mm2(holes, cell_df)
        print(f"    Region areas (mm²): { {k: round(v, 4) for k, v in area_dict.items()} }")

        region_metric_rows = []
        for region, area_mm2 in area_dict.items():
            region_metric_rows.append({
                IMAGE_KEY:        image_id,
                GROUP_KEY:        pathology,
                "dark_component": "all",
                "region":         region,
                "area_um2":       area_mm2 / UM2_TO_MM2,
                "area_mm2":       area_mm2,
            })
        all_region_metric_rows.append(pd.DataFrame(region_metric_rows))

        # ── Per-phenotype × region counts ─────────────────────────────────────
        count_df = cell_df[[PHENOTYPE_KEY]].copy()
        count_df["COL6_dark_region"] = region_series.values
        counts = (
            count_df.groupby([PHENOTYPE_KEY, "COL6_dark_region"], observed=True)
            .size()
            .reset_index(name="n_cells")
        )
        totals = counts.groupby(PHENOTYPE_KEY, observed=True)["n_cells"].transform("sum")
        counts["fraction_of_phenotype"] = counts["n_cells"] / totals
        counts.insert(0, IMAGE_KEY, image_id)
        counts.insert(1, GROUP_KEY, pathology)
        all_count_rows.append(counts)

    # ── Concatenate and save ───────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("Saving output CSV files …")

    cell_assignments = pd.concat(all_assignment_rows, ignore_index=True)
    # Reorder columns to match applications notebook expectations
    cell_assignments = cell_assignments[[
        IMAGE_KEY, GROUP_KEY, PHENOTYPE_KEY, X_KEY, Y_KEY,
        "cell_id", "COL6_dark_region", "dark_component",
        "COL6_dark_score", "CHP_density_rank",
    ]]
    out_04 = DARK_DIR / "04_cell_chp_micro_hole_region_assignments.csv"
    cell_assignments.to_csv(out_04, index=False)
    print(f"  04 cell assignments  → {out_04}  ({len(cell_assignments):,} rows)")

    image_region_metrics = pd.concat(all_region_metric_rows, ignore_index=True)
    out_03 = DARK_DIR / "03_chp_micro_hole_image_region_edge_metrics.csv"
    image_region_metrics.to_csv(out_03, index=False)
    print(f"  03 region metrics    → {out_03}  ({len(image_region_metrics)} rows)")

    cell_region_counts = pd.concat(all_count_rows, ignore_index=True)
    out_06 = DARK_DIR / "06_cell_region_phenotype_counts.csv"
    cell_region_counts.to_csv(out_06, index=False)
    print(f"  06 phenotype counts  → {out_06}  ({len(cell_region_counts):,} rows)")

    print("\nDone. Review QC plots in:")
    print(f"  {FIGURES_DIR}")
    print("\nIf holes look wrong for a sample, add a SAMPLE_OVERRIDES entry and re-run.")
    print("Then run 09_RA_OA_ECM_cell_spatioev_module_paper_applications.ipynb as usual.")


if __name__ == "__main__":
    main()
