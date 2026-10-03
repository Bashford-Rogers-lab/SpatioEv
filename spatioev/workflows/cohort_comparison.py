#!/usr/bin/env python3
"""Background worker: compare spatial summaries between groups of samples.

Reads a sample sheet assigning samples (or individual TMA cores) to groups,
collects the summaries the per-sample stages wrote, averages them per
independent unit (patient) and tests every feature between groups.

Sample sheet columns
--------------------
sample_id     required; the Sample ID used in the per-sample stages
group         required; e.g. HPV+ / HPV-
imageid       optional; restrict the row to one image (TMA core) of the sample
patient_id    optional; cores or slides from the same patient are averaged
project_root  optional; where that sample's ``results/`` folder lives
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import traceback
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spatioev_cohort_matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spatioev.tl.compare import compare_groups

from ._io import now, write_json

FAMILIES = ("composition", "matrix", "cell_cell", "cell_matrix")
MATRIX_METRICS = (
    "area_fraction_pct", "fibers_per_mm2", "avg_alignment_score", "orientation_coherence",
    "avg_major_axis_length", "avg_minor_axis_length",
    # TWOMBLI-style architecture. lacunarity_anamorf is left out: it is a
    # transform of coverage (|1/p - 2|), so testing it repeats area fraction.
    "hdm_pct", "alignment_coherency", "length_density_mm_per_mm2", "branchpoints_per_mm2",
    "endpoints_per_mm2", "branchpoints_per_mm_length", "mean_branch_length_um", "curvature_per_um",
    "fractal_dimension", "lacunarity_gliding", "fiber_thickness_um",
)


def update_status(path: Path, state: str, message: str, *, stage: str, progress: float, **extra) -> None:
    write_json(path, {"state": state, "message": message, "stage": stage, "progress": progress, "updated_at": now(), **extra})


def read_sample_sheet(path: str | Path) -> pd.DataFrame:
    sheet = pd.read_csv(Path(path).expanduser(), dtype=str).fillna("")
    sheet.columns = [c.strip() for c in sheet.columns]
    missing = {"sample_id", "group"} - set(sheet.columns)
    if missing:
        raise ValueError(f"Sample sheet is missing column(s): {sorted(missing)}")
    for column in ("imageid", "patient_id", "project_root"):
        if column not in sheet:
            sheet[column] = ""
    sheet = sheet[sheet["sample_id"].str.strip() != ""]
    for column in sheet.columns:
        sheet[column] = sheet[column].str.strip()
    return sheet


def stage_files(root: Path, sample_id: str) -> dict[str, Path]:
    regions = root / "results" / f"{sample_id}_tissue_regions"
    coloc = root / "results" / f"{sample_id}_colocalization"
    return {
        "composition": regions / f"{sample_id}_region_composition.csv",
        "matrix": regions / f"{sample_id}_fiber_region_stats.csv",
        "cell_cell_curves": coloc / f"{sample_id}_cell_cell_curves.csv",
        "cell_cell_local": coloc / f"{sample_id}_cell_cell_local.csv",
        "cell_matrix": coloc / f"{sample_id}_cell_matrix_summary.csv",
    }


def _read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if "imageid" in frame:
        frame["imageid"] = frame["imageid"].astype(str)
    return frame


def sample_features(sample_id: str, root: Path, families, min_region_cells: int) -> tuple[pd.DataFrame, dict]:
    files = stage_files(root, sample_id)
    found = {key: path.exists() for key, path in files.items()}
    rows = []

    composition = _read(files["composition"]) if "composition" in families else None
    if composition is not None and len(composition):
        phenotype_key = [c for c in composition.columns if c not in {"imageid", "tissue_region", "count", "region_total", "proportion"}][0]
        valid = composition[composition["region_total"] >= min_region_cells]
        rows.append(pd.DataFrame({
            "imageid": valid["imageid"],
            "family": "composition",
            "feature": "composition | " + valid["tissue_region"].astype(str) + " | " + valid[phenotype_key].astype(str),
            "value": valid["proportion"],
        }))

    matrix = _read(files["matrix"]) if "matrix" in families else None
    if matrix is not None and len(matrix):
        metrics = [m for m in MATRIX_METRICS if m in matrix]
        long = matrix.melt(id_vars=["imageid", "fiber_type", "region"], value_vars=metrics, var_name="metric")
        rows.append(pd.DataFrame({
            "imageid": long["imageid"],
            "family": "matrix",
            "feature": "matrix | " + long["fiber_type"] + " | " + long["region"] + " | " + long["metric"],
            "value": long["value"],
        }))

    if "cell_cell" in families:
        curves = _read(files["cell_cell_curves"])
        if curves is not None and len(curves):
            rows.append(pd.DataFrame({
                "imageid": curves["imageid"],
                "family": "cell_cell",
                "feature": "cell-cell | " + curves["source"] + " → " + curves["target"] + " | L(r)−r excess over null @ "
                + curves["radius_um"].round(1).astype(str) + " µm",
                "value": curves["excess_over_null_um"],
            }))
        local = _read(files["cell_cell_local"])
        if local is not None and len(local):
            rows.append(pd.DataFrame({
                "imageid": local["imageid"],
                "family": "cell_cell",
                "feature": "cell-cell | " + local["source"] + " → " + local["target"] + " | " + local["region"] + " | neighbour ratio @ "
                + local["radius_um"].round(1).astype(str) + " µm",
                "value": local["mean_neighbour_ratio"],
            }))

    cell_matrix = _read(files["cell_matrix"]) if "cell_matrix" in families else None
    if cell_matrix is not None and len(cell_matrix):
        long = cell_matrix.melt(
            id_vars=["imageid", "fiber_type", "region", "phenotype"],
            value_vars=["mean_matrix_fraction", "median_distance_to_matrix_um", "fraction_in_contact"],
            var_name="metric",
        )
        rows.append(pd.DataFrame({
            "imageid": long["imageid"],
            "family": "cell_matrix",
            "feature": "cell-matrix | " + long["phenotype"] + " | " + long["fiber_type"] + " | " + long["region"] + " | " + long["metric"],
            "value": long["value"],
        }))

    features = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["imageid", "family", "feature", "value"])
    features.insert(0, "sample_id", sample_id)
    return features, found


def save_feature_plot(features: pd.DataFrame, comparison: pd.DataFrame, unit_values: pd.DataFrame, order: list[str], path: Path, top: int = 12) -> Path | None:
    tested = comparison.dropna(subset=["p_value"]).head(top)
    if tested.empty:
        return None
    n = len(tested)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.6 * rows), squeeze=False)
    for ax, (_, row) in zip(axes.ravel(), tested.iterrows()):
        sub = unit_values[unit_values["feature"] == row["feature"]]
        data = [sub.loc[sub["group"] == g, "value"].to_numpy(float) for g in order]
        ax.boxplot(data, showfliers=False, widths=0.5)
        for index, values in enumerate(data, start=1):
            jitter = np.random.default_rng(index).uniform(-0.12, 0.12, len(values))
            ax.scatter(np.full(len(values), index) + jitter, values, s=14, alpha=0.8)
        ax.set_xticks(range(1, len(order) + 1), order)
        title = row["feature"] if len(row["feature"]) < 70 else row["feature"][:67] + "…"
        ax.set_title(f"{title}\np={row['p_value']:.3g}, q={row['q_value']:.3g}", fontsize=8)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def run_workflow(config: dict, status_path: Path) -> dict:
    output_dir = Path(config["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    name = config.get("comparison_name", "cohort").strip() or "cohort"
    families = [f for f in config.get("families", FAMILIES) if f in FAMILIES]
    default_root = Path(config.get("project_root", ".")).expanduser()

    update_status(status_path, "running", "Reading sample sheet", stage="load", progress=0.05)
    sheet = read_sample_sheet(config["sample_sheet"])
    collected, availability = [], []
    samples = sheet["sample_id"].unique().tolist()
    for index, sample_id in enumerate(samples):
        rows = sheet[sheet["sample_id"] == sample_id]
        root = Path(rows["project_root"].iloc[0]).expanduser() if rows["project_root"].iloc[0] else default_root
        features, found = sample_features(sample_id, root, families, int(config.get("min_region_cells", 20)))
        availability.append({"sample_id": sample_id, "project_root": str(root), **found, "n_features": len(features)})
        if not features.empty:
            per_image = rows[rows["imageid"] != ""]
            whole = rows[rows["imageid"] == ""]
            if len(whole):
                meta = whole.iloc[0]
                features = features.assign(group=meta["group"], patient_id=meta["patient_id"])
            else:
                features = features.merge(per_image[["imageid", "group", "patient_id"]], on="imageid", how="inner")
            collected.append(features)
        update_status(status_path, "running", f"Collected {sample_id}", stage="collect", progress=round(0.05 + 0.6 * (index + 1) / len(samples), 4))

    availability = pd.DataFrame(availability)
    availability_path = output_dir / f"{name}_inputs_found.csv"
    availability.to_csv(availability_path, index=False)
    if not collected:
        raise FileNotFoundError(
            "No per-sample summaries were found. Run the Tissue regions and Co-localisation stages first; "
            f"see {availability_path} for where each sample was looked for."
        )
    features = pd.concat(collected, ignore_index=True)
    features["unit"] = np.where(
        features["patient_id"].astype(str) != "",
        features["patient_id"].astype(str),
        features["sample_id"].astype(str) + np.where(features["imageid"].astype(str) != "", "::" + features["imageid"].astype(str), ""),
    )
    features_path = output_dir / f"{name}_features_long.csv"
    features.to_csv(features_path, index=False)

    update_status(status_path, "running", "Testing features between groups", stage="test", progress=0.75)
    groups = [g for g in config.get("groups", []) if g] or sorted(features["group"].astype(str).unique())
    comparison = compare_groups(
        features, group_key="group", feature_key="feature", value_key="value", unit_key="unit",
        groups=groups, min_units_per_group=int(config.get("min_units_per_group", 2)),
    )
    family_of = features.drop_duplicates("feature").set_index("feature")["family"]
    comparison.insert(1, "family", comparison["feature"].map(family_of))
    comparison_path = output_dir / f"{name}_group_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    unit_values = (
        features[features["group"].isin(groups)]
        .groupby(["unit", "group", "feature"], observed=True)["value"].mean().reset_index()
    )
    unit_path = output_dir / f"{name}_unit_values.csv"
    unit_values.pivot_table(index=["unit", "group"], columns="feature", values="value").to_csv(unit_path)
    plot = save_feature_plot(features, comparison, unit_values, groups, output_dir / f"{name}_top_features.png")

    units_per_group = unit_values.drop_duplicates(["unit", "group"])["group"].value_counts().to_dict()
    outputs = {
        "comparison": str(comparison_path),
        "features_long": str(features_path),
        "unit_values": str(unit_path),
        "inputs_found": str(availability_path),
        "top_features_png": str(plot) if plot else None,
        "groups": groups,
        "units_per_group": {str(k): int(v) for k, v in units_per_group.items()},
        "n_features": int(comparison["feature"].nunique()) if len(comparison) else 0,
        "n_significant_q05": int((comparison["q_value"] < 0.05).sum()) if len(comparison) else 0,
    }
    manifest_path = output_dir / f"{name}_cohort_manifest.json"
    write_json(manifest_path, {"created_at": now(), "config": config, "outputs": outputs})
    outputs["manifest"] = str(manifest_path)
    update_status(status_path, "complete", f"Compared {outputs['n_features']:,} features across {len(groups)} groups", stage="complete", progress=1.0, outputs=outputs)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    try:
        run_workflow(config, args.status)
    except Exception as error:
        update_status(args.status, "failed", str(error), stage="failed", progress=1.0, traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
