"""A synthetic multi-FOV testis TMA in the ARK layout, for steps 00-03.

:func:`make_demo_testis_tma` writes a small artificial dataset laid out like
a real cyclic-immunofluorescence TMA after dearraying, background
subtraction and ARK segmentation::

    <destination>/<dataset_id>/
        background/markers_bs.csv          cycle-based panel, one row per image plane
        dearray/1.ome.tif ... N.ome.tif    one unnamed multi-channel image per FOV
        ark_wdir_1/segmentation/cell_table/cell_table_size_normalized.csv
                                           cell_table_arcsinh_transformed.csv
        ark_wdir_1/segmentation/deepcell_output/fov1_whole_cell.tiff ...
        ark_wdir_2/...                     FOVs are split over several ARK runs
        fov_metadata.csv                   age group of each FOV
        ground_truth.csv                   true cell type of every cell (for checking)
        data/<dataset_id>_marker_condition_for_gating.csv
                                           step 02's questionnaire answers
        manual_gates.csv                   per-FOV gates (markers,fov1,fov2,...)
        tubule_phenotype_workflow.csv      scimap-style prior-knowledge rules
        interstitium_phenotype_workflow.csv

It reproduces the details that make the real data awkward:

- the image planes carry no channel names (``C0``, ``C1``, ...), so only the
  row order of ``markers_bs.csv`` says which marker each plane is;
- ``markers_bs.csv`` numbers channels *within* each imaging cycle, so sorting
  by ``channel_number`` scrambles the panel;
- ARK writes the cell-table marker columns in alphabetical order, not image
  order, and adds a ``nuclear`` row for every cell;
- staining strength differs between FOVs, so one gate per marker does not fit
  every FOV.

The tissue is a prepubertal-to-peripubertal testis: seminiferous tubules lined
by Sertoli cells and spermatogonia, with spermatocytes and spermatids only in
the older FOVs, a ring of peritubular myoid cells around each tubule, and an
interstitium of Leydig cells, vessels and immune cells. The cell tables are
measured from the images and masks that are written, so every reader sees the
same numbers. Nothing comes from a patient.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["COMPARTMENTS", "TESTIS_PANEL", "make_demo_testis_tma"]

#: Image-plane order: (marker, imaging cycle, channel within the cycle).
TESTIS_PANEL = (
    ("DNA_1", 0, 0), ("FMRP", 1, 1), ("AMH", 1, 2), ("GFRA1", 1, 3),
    ("SOX9", 3, 1), ("CREM", 3, 2), ("MAGEA4", 3, 3), ("INSL3", 5, 1),
    ("KRT18", 5, 2), ("PIWIL4", 5, 3), ("NaKATPase", 7, 1), ("panCad", 7, 3),
    ("DMRT1", 9, 3), ("STAR", 11, 2), ("CYP17A1", 11, 3), ("SYCP3", 13, 1),
    ("CYP11A1", 13, 2), ("PCNA", 13, 3), ("CD45", 15, 1), ("CD68", 15, 2),
    ("NF2F2", 15, 3), ("CD31", 17, 1), ("UTF1_2", 17, 2), ("Vimentin_2", 18, 3),
    ("CX43_2", 20, 1), ("Acrosin", 20, 2), ("MYH11", 20, 3), ("SMA", 22, 1),
    ("CD8", 22, 3), ("CLDN11_2", 24, 2),
)
MARKERS = tuple(marker for marker, _, _ in TESTIS_PANEL)

#: Markers each cell type expresses (besides DNA_1 and low membrane staining).
_EXPRESSION = {
    "Sertoli": ("SOX9", "AMH", "CLDN11_2", "CX43_2", "KRT18", "Vimentin_2", "DMRT1", "panCad"),
    "Spermatogonia": ("MAGEA4", "FMRP"),
    "Spermatocyte": ("SYCP3", "CREM", "MAGEA4", "PCNA"),
    "Spermatid": ("Acrosin", "CREM"),
    "Myoid": ("MYH11", "SMA", "Vimentin_2"),
    "Leydig": ("INSL3", "CYP17A1", "CYP11A1", "STAR", "NF2F2"),
    "Endothelial": ("CD31", "Vimentin_2"),
    "Macrophage": ("CD45", "CD68"),
    "T cell": ("CD45", "CD8"),
    "Fibroblast": ("Vimentin_2", "NF2F2"),
}
#: Where each marker stains, for step 02's marker-condition questionnaire.
_NUCLEAR_MARKERS = {"DNA_1", "SOX9", "CREM", "DMRT1", "PCNA", "UTF1_2", "PIWIL4", "SYCP3", "NF2F2", "MAGEA4"}
_MEMBRANE_MARKERS = {"CD31", "CD45", "CD8", "CLDN11_2", "CX43_2", "NaKATPase", "panCad", "GFRA1"}

#: Tissue compartment of each cell type: what broad clustering should find.
COMPARTMENTS = {
    "Sertoli": "Tubule", "Spermatogonia": "Tubule", "Spermatocyte": "Tubule", "Spermatid": "Tubule",
    "Myoid": "Myoid",
    "Leydig": "Interstitium", "Endothelial": "Interstitium", "Macrophage": "Interstitium",
    "T cell": "Interstitium", "Fibroblast": "Interstitium",
}
#: Spermatogonial states and the extra markers that define them.
_SPG_STATES = {
    "state0": ("UTF1_2", "PIWIL4"),
    "state1": ("GFRA1",),
    "state2": ("DMRT1", "PCNA"),
    "state3-4": (),
}
_CELL_RADIUS = 5
_NUCLEUS_RADIUS = 3
_BACKGROUND = 25.0
_POSITIVE = 900.0
_ARCSINH_COFACTOR = 5.0

_MORPHOLOGY = (
    "area", "eccentricity", "major_axis_length", "minor_axis_length", "perimeter",
    "convex_area", "equivalent_diameter", "centroid-0", "centroid-1",
)


def _disc(radius: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    keep = yy**2 + xx**2 <= radius**2
    return yy[keep], xx[keep]


def _ring_points(rng, centre, r_min, r_max, n):
    angle = rng.uniform(0, 2 * np.pi, n)
    radius = np.sqrt(rng.uniform(r_min**2, r_max**2, n))
    return np.column_stack([centre[0] + radius * np.sin(angle), centre[1] + radius * np.cos(angle)])


def _layout(rng, size: int, older: bool):
    """Cell centres and types for one FOV."""
    # Fewer tubules in small test images, so they still fit side by side.
    n_tubules = 4 if size >= 360 else 2 if size >= 260 else 1
    radius = size * 0.17
    centres: list[np.ndarray] = []
    while len(centres) < n_tubules:
        candidate = rng.uniform(radius + 14, size - radius - 14, 2)
        if all(np.hypot(*(candidate - other)) > 2 * radius + 34 for other in centres):
            centres.append(candidate)
        if len(centres) < n_tubules and rng.random() < 0.002:
            radius *= 0.95  # give up gracefully on small images
    points, kinds, states, tubules = [], [], [], []

    def add(xy, kind, state="", tubule=0):
        points.append(xy)
        kinds.extend([kind] * len(xy))
        states.extend([state] * len(xy) if isinstance(state, str) else list(state))
        tubules.extend([tubule] * len(xy))

    for number, centre in enumerate(centres, start=1):
        add(_ring_points(rng, centre, radius - 20, radius - 9, 26), "Sertoli", tubule=number)
        spg = _ring_points(rng, centre, radius - 10, radius - 3, 18)
        spg_states = rng.choice(list(_SPG_STATES), len(spg), p=[0.3, 0.2, 0.3, 0.2])
        add(spg, "Spermatogonia", spg_states.tolist(), tubule=number)
        if older:
            inner = radius * 0.45
            add(_ring_points(rng, centre, inner, max(inner + 2, radius - 21), 30), "Spermatocyte", tubule=number)
            add(_ring_points(rng, centre, radius * 0.25, radius * 0.45, 8), "Spermatid", tubule=number)
        add(_ring_points(rng, centre, radius + 3, radius + 9, 20), "Myoid", tubule=number)
    n_interstitial = int(size * size / 900)
    candidates = rng.uniform(6, size - 6, (n_interstitial * 3, 2))
    outside = np.ones(len(candidates), dtype=bool)
    for centre in centres:
        outside &= np.hypot(*(candidates - centre).T) > radius + 15
    candidates = candidates[outside][:n_interstitial]
    interstitial_kinds = rng.choice(
        ["Leydig", "Endothelial", "Macrophage", "T cell", "Fibroblast"],
        len(candidates),
        p=[0.35, 0.2, 0.12, 0.08, 0.25],
    )
    points.append(candidates)
    kinds.extend(interstitial_kinds.tolist())
    states.extend([""] * len(candidates))
    tubules.extend([0] * len(candidates))

    xy = np.vstack(points)
    kinds = np.asarray(kinds)
    states = np.asarray(states)
    tubules = np.asarray(tubules)
    # Place cells in random order, rejecting any that would touch one already
    # placed, so masks never overlap.
    order = rng.permutation(len(xy))
    xy, kinds, states, tubules = xy[order], kinds[order], states[order], tubules[order]
    accepted = []
    occupied = np.zeros((size, size), dtype=bool)
    ys, xs = _disc(_CELL_RADIUS + 1)
    cys, cxs = _disc(_CELL_RADIUS)
    for index, (y, x) in enumerate(np.round(xy).astype(int)):
        if not (_CELL_RADIUS < y < size - _CELL_RADIUS - 1 and _CELL_RADIUS < x < size - _CELL_RADIUS - 1):
            continue
        if occupied[y + ys, x + xs].any():
            continue
        occupied[y + cys, x + cxs] = True
        accepted.append(index)
    accepted = np.asarray(accepted, dtype=int)
    return xy[accepted], kinds[accepted], states[accepted], tubules[accepted]


def _render_fov(rng, size: int, older: bool, brightness: float):
    xy, kinds, states, tubules = _layout(rng, size, older)
    n_cells = len(xy)
    cell_mask = np.zeros((size, size), dtype=np.int32)
    nuclear_mask = np.zeros((size, size), dtype=np.int32)
    cy, cx = _disc(_CELL_RADIUS)
    ny, nx = _disc(_NUCLEUS_RADIUS)
    centres = np.round(xy).astype(int)
    for label, (y, x) in enumerate(centres, start=1):
        cell_mask[y + cy, x + cx] = label
        nuclear_mask[y + ny, x + nx] = label

    levels = np.zeros((n_cells, len(MARKERS)))
    index = {marker: i for i, marker in enumerate(MARKERS)}
    for cell, (kind, state) in enumerate(zip(kinds, states)):
        markers = list(_EXPRESSION[kind]) + list(_SPG_STATES.get(state, ()))
        for marker in markers:
            levels[cell, index[marker]] = _POSITIVE * rng.lognormal(0.0, 0.35)
        # Na/K-ATPase lines every membrane, but on real testis cores it stains
        # tubules far more strongly than the interstitium.
        membrane = 480 if COMPARTMENTS[kind] == "Tubule" else 120
        levels[cell, index["NaKATPase"]] += membrane * rng.lognormal(0, 0.3)
    levels *= brightness

    stack = np.empty((len(MARKERS), size, size), dtype=np.float32)
    for plane, marker in enumerate(MARKERS):
        signal = np.zeros(n_cells + 1)
        signal[1:] = levels[:, plane]
        mask = nuclear_mask if marker == "DNA_1" else cell_mask
        image = signal[mask] if marker != "DNA_1" else np.where(nuclear_mask > 0, 1500.0 * brightness, 0.0)
        image = image + rng.gamma(2.0, _BACKGROUND / 2.0, (size, size))
        stack[plane] = image
    return stack.clip(0, 65535).astype(np.uint16), cell_mask, nuclear_mask, kinds, states, tubules


def _measure(stack, mask, fov, mask_type) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = mask.ravel()
    n = int(labels.max())
    area = np.bincount(labels, minlength=n + 1)[1:].astype(float)
    present = area > 0
    rows = np.arange(1, n + 1)[present]
    area = area[present]
    yy, xx = np.indices(mask.shape)
    cy = np.bincount(labels, weights=yy.ravel(), minlength=n + 1)[1:][present] / area
    cx = np.bincount(labels, weights=xx.ravel(), minlength=n + 1)[1:][present] / area
    means = {
        marker: np.bincount(labels, weights=stack[plane].ravel().astype(float), minlength=n + 1)[1:][present] / area
        for plane, marker in enumerate(MARKERS)
    }
    diameter = np.sqrt(4 * area / np.pi)
    morphology = {
        "label": rows,
        "area": area,
        "eccentricity": np.full(len(rows), 0.2),
        "major_axis_length": diameter,
        "minor_axis_length": diameter * 0.95,
        "perimeter": np.pi * diameter,
        "convex_area": area,
        "equivalent_diameter": diameter,
        "centroid-0": cy,
        "centroid-1": cx,
        "fov": fov,
        "mask_type": mask_type,
    }
    # ARK writes the marker columns alphabetically, after cell_size.
    alphabetical = sorted(MARKERS, key=str.lower)
    size_normalized = pd.DataFrame({"cell_size": area, **{m: means[m] for m in alphabetical}, **morphology})
    arcsinh = size_normalized.copy()
    for marker in alphabetical:
        arcsinh[marker] = np.arcsinh(arcsinh[marker] / _ARCSINH_COFACTOR)
    return size_normalized, arcsinh


def _workflows() -> tuple[pd.DataFrame, pd.DataFrame]:
    tubule = pd.DataFrame(
        [
            ["all", "Sertoli", {"SOX9": "anypos", "AMH": "anypos"}],
            ["all", "Germ cells", {"MAGEA4": "anypos", "SYCP3": "anypos", "Acrosin": "anypos"}],
            # 'allpos'/'allneg', not 'pos'/'neg': scimap averages a lone
            # 'pos' or 'neg' marker into the score instead of requiring it, so
            # a dim SYCP3+ spermatocyte could still score highest here.
            ["Germ cells", "Spermatogonia", {"MAGEA4": "allpos", "SYCP3": "allneg"}],
            ["Germ cells", "Spermatocytes", {"SYCP3": "pos"}],
            ["Germ cells", "Spermatids", {"Acrosin": "pos"}],
            ["Spermatogonia", "SPG state0", {"UTF1_2": "anypos", "PIWIL4": "anypos"}],
            ["Spermatogonia", "SPG state1", {"GFRA1": "pos"}],
            ["Spermatogonia", "SPG state2", {"PCNA": "anypos", "DMRT1": "anypos"}],
        ],
        columns=["parent", "phenotype", "rules"],
    )
    interstitium = pd.DataFrame(
        [
            ["all", "Leydig", {"INSL3": "anypos", "CYP17A1": "anypos", "CYP11A1": "anypos"}],
            ["all", "Endothelial", {"CD31": "pos"}],
            # Two 'pos' markers are both required, so NF2F2+ Leydig cells and
            # Vimentin+ endothelial cells do not qualify.
            ["all", "Fibroblast", {"NF2F2": "pos", "Vimentin_2": "pos"}],
            ["all", "Immune", {"CD45": "pos"}],
            ["Immune", "Macrophage", {"CD68": "pos"}],
            ["Immune", "CD8 T cell", {"CD8": "pos"}],
        ],
        columns=["parent", "phenotype", "rules"],
    )

    def wide(frame: pd.DataFrame, rename: dict[str, str] | None = None) -> pd.DataFrame:
        markers = list(dict.fromkeys(m for rules in frame["rules"] for m in rules))
        table = pd.DataFrame(
            [{"parent": p, "phenotype": n, **rules} for p, n, rules in frame.itertuples(index=False)],
            columns=["parent", "phenotype", *markers],
        )
        return table.rename(columns=rename or {})

    # Spelled the way an older panel did (CLDN11_1 rather than CLDN11_2) to
    # show that workflow columns are matched to the AnnData's marker names.
    tubule_table = wide(tubule)
    tubule_table.insert(len(tubule_table.columns), "CLDN11_1", ["anypos"] + [np.nan] * (len(tubule_table) - 1))
    return tubule_table, wide(interstitium)


def make_demo_testis_tma(
    destination: str | Path,
    *,
    dataset_id: str = "DEMO_TESTIS",
    n_fovs: int = 6,
    fovs_per_batch: int = 2,
    size: int = 400,
    seed: int = 0,
    overwrite: bool = False,
) -> Path:
    """Write the synthetic testis TMA; return the dataset folder.

    ``n_fovs`` FOVs of ``size`` x ``size`` pixels are split over ARK batches
    of ``fovs_per_batch`` FOVs. The second half of the FOVs are "older"
    (spermatocytes and spermatids present) and staining brightness varies by
    FOV, so per-FOV gates differ.
    """
    import tifffile

    if size < 160:
        raise ValueError("size must be at least 160 pixels")
    root = Path(destination).expanduser().resolve() / dataset_id
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"{root} already exists; pass overwrite=True to replace it")
        shutil.rmtree(root)
    rng = np.random.default_rng(seed)
    (root / "background").mkdir(parents=True)
    (root / "dearray").mkdir()
    pd.DataFrame(
        {
            "channel_number": [channel for _, _, channel in TESTIS_PANEL],
            "cycle_number": [cycle for _, cycle, _ in TESTIS_PANEL],
            "marker_name": MARKERS,
        }
    ).to_csv(root / "background" / "markers_bs.csv", index=False)

    metadata, gate_rows, truth = [], {}, []
    tables: dict[int, list[tuple[pd.DataFrame, pd.DataFrame]]] = {}
    for number in range(1, n_fovs + 1):
        fov = f"fov{number}"
        older = number > n_fovs // 2
        brightness = float(rng.uniform(0.55, 1.6))
        stack, cell_mask, nuclear_mask, kinds, states, tubules = _render_fov(rng, size, older, brightness)
        # Mask label i is the i-th placed cell.
        truth.append(
            pd.DataFrame(
                {
                    "fov": fov,
                    "label": np.arange(1, len(kinds) + 1),
                    "cell_type": kinds,
                    "spg_state": np.where(kinds == "Spermatogonia", states, ""),
                    "compartment": [COMPARTMENTS[kind] for kind in kinds],
                    # Tubule a cell belongs to (its myoid ring included); 0 = interstitium.
                    "tubule": tubules,
                }
            )
        )
        tifffile.imwrite(
            root / "dearray" / f"{number}.ome.tif",
            stack,
            ome=True,
            photometric="minisblack",
            compression="zlib",
            metadata={"axes": "CYX"},
        )
        batch = (number - 1) // fovs_per_batch + 1
        segmentation = root / f"ark_wdir_{batch}" / "segmentation"
        (segmentation / "deepcell_output").mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(segmentation / "deepcell_output" / f"{fov}_whole_cell.tiff", cell_mask, compression="zlib")
        tifffile.imwrite(segmentation / "deepcell_output" / f"{fov}_nuclear.tiff", nuclear_mask, compression="zlib")
        whole = _measure(stack, cell_mask, fov, "whole_cell")
        nuclear = _measure(stack, nuclear_mask, fov, "nuclear")
        tables.setdefault(batch, []).append((whole, nuclear))
        metadata.append(
            {
                "fov": fov,
                "image": f"dearray/{number}.ome.tif",
                "ark_wdir": f"ark_wdir_{batch}",
                "age_group": "peripubertal" if older else "prepubertal",
                "n_cells": int(cell_mask.max()),
                "brightness": round(brightness, 3),
            }
        )
        # Gates sit between background and positive cells on the log1p scale
        # of the size-normalised intensity, scaled with the FOV's brightness.
        gate_rows[fov] = round(float(np.log1p(220.0 * brightness)), 2)

    for batch, parts in tables.items():
        folder = root / f"ark_wdir_{batch}" / "segmentation" / "cell_table"
        folder.mkdir(parents=True, exist_ok=True)
        # Whole-cell rows first, then the nuclear rows, as ARK writes them.
        size_normalized = pd.concat(
            [whole[0] for whole, _ in parts] + [nuclear[0] for _, nuclear in parts], ignore_index=True
        )
        arcsinh = pd.concat(
            [whole[1] for whole, _ in parts] + [nuclear[1] for _, nuclear in parts], ignore_index=True
        )
        size_normalized.to_csv(folder / "cell_table_size_normalized.csv", index=False)
        arcsinh.to_csv(folder / "cell_table_arcsinh_transformed.csv", index=False)

    pd.DataFrame(metadata).to_csv(root / "fov_metadata.csv", index=False)
    pd.concat(truth, ignore_index=True).to_csv(root / "ground_truth.csv", index=False)
    # Like a real hand-made gate table: one column per FOV, and only some FOVs
    # reviewed so far.
    reviewed = [f"fov{number}" for number in range(1, n_fovs + 1, 2)]
    gate_markers = [marker for marker in MARKERS if marker != "DNA_1"]
    pd.DataFrame(
        {"markers": gate_markers, **{fov: [gate_rows[fov]] * len(gate_markers) for fov in reviewed}}
    ).to_csv(root / "manual_gates.csv", index=False)
    # Step 02's questionnaire, where the gating page looks for saved answers.
    # Expression shape is left blank so the gate calculation infers it from
    # each marker's distribution.
    conditions = root / "data" / f"{dataset_id}_marker_condition_for_gating.csv"
    conditions.parent.mkdir(exist_ok=True)
    pd.DataFrame(
        {
            "marker": gate_markers,
            "staining_condition": "clear_specific",
            "compartment_pattern": [
                "nuclear" if m in _NUCLEAR_MARKERS else "membrane" if m in _MEMBRANE_MARKERS else "cytoplasmic"
                for m in gate_markers
            ],
            "expression_condition": "",
            "artifact_level": "low",
        }
    ).to_csv(conditions, index=False)
    tubule, interstitium = _workflows()
    tubule.to_csv(root / "tubule_phenotype_workflow.csv", index=False)
    interstitium.to_csv(root / "interstitium_phenotype_workflow.csv", index=False)
    return root
