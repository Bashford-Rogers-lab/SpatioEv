#!/usr/bin/env python3
"""Convert paired CellSAM quantification tables into a validated AnnData file."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from xml.etree import ElementTree as ET

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spatioev_cellsam_matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile


CHANNEL_ALIASES = {
    "EPCAM": "EpCAM",
    "HOECHST": "HOECHST2",
    "HOECHST 2": "HOECHST2",
    "LYVE-1": "LYVE1",
    "HNFalpha": "HNF4a",
    "pancytokeratin": "panCK",
    "pancyto": "panCK",
    "Cd16": "CD16",
    "alphaSMA": "aSMA",
    "TCRValpha": "TCRVa",
    "ILR18a": "IL18Ra",
    "FOXP3": "FoxP3",
    "CD4 good": "CD4",
}


@dataclass(frozen=True)
class TableSchema:
    path: Path
    columns: list[str]
    marker_columns: list[str]
    obs_columns: list[str]


@dataclass(frozen=True)
class ConversionPlan:
    primary_csv: Path
    secondary_csv: Path
    image_path: Path
    imageid: str
    output_path: Path
    layer_name: str = "size_normalized"
    make_qc: bool = True


StatusCallback = Callable[[str, str, float], None]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return next(csv.reader(handle))


def table_schema(path: Path) -> TableSchema:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    columns = read_header(path)
    if len(columns) != len(set(columns)):
        duplicated = sorted({column for column in columns if columns.count(column) > 1})
        raise ValueError(f"Duplicate columns in {path.name}: {duplicated}")
    if "label" not in columns:
        raise ValueError(f"{path.name} does not contain the required 'label' column")
    label_index = columns.index("label")
    marker_columns = [column for column in columns[:label_index] if column != "cell_size"]
    obs_columns = columns[label_index:]
    if not marker_columns:
        raise ValueError(f"No marker columns were found before 'label' in {path.name}")
    return TableSchema(path, columns, marker_columns, obs_columns)


def discover_tables(cell_table_dir: Path) -> list[Path]:
    directory = Path(cell_table_dir).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    preferred = [
        directory / "cell_table_arcsinh_transformed.csv",
        directory / "cell_table_size_normalized.csv",
    ]
    existing = [path for path in preferred if path.exists()]
    if len(existing) == 2:
        return existing
    return sorted(
        path for path in directory.glob("*.csv")
        if not path.name.startswith("._") and "raw" not in path.stem.lower()
    )


def image_channel_names(image_path: Path) -> list[str]:
    image_path = Path(image_path).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    with tifffile.TiffFile(image_path) as tif:
        ome_xml = tif.ome_metadata
        if not ome_xml:
            shape = tif.series[0].shape
            return [f"C{index}" for index in range(shape[0])]
    root = ET.fromstring(ome_xml)
    channels = [node for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "Channel"]
    return [node.attrib.get("Name", f"C{index}") for index, node in enumerate(channels)]


def canonical_name(name: str) -> str:
    clean = str(name).strip()
    return CHANNEL_ALIASES.get(clean, clean)


def marker_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", canonical_name(name).lower())


def ordered_marker_mapping(image_path: Path, table_markers: list[str]) -> tuple[list[str], list[str]]:
    image_channels = image_channel_names(image_path)
    table_by_key: dict[str, str] = {}
    for marker in table_markers:
        key = marker_key(marker)
        if key in table_by_key:
            raise ValueError(
                f"Table marker names {table_by_key[key]!r} and {marker!r} are ambiguous after normalization"
            )
        table_by_key[key] = marker

    output_markers: list[str] = []
    source_columns: list[str] = []
    missing_from_table: list[str] = []
    seen_keys: set[str] = set()
    for channel in image_channels:
        key = marker_key(channel)
        if key in seen_keys:
            raise ValueError(f"OME image contains an ambiguous duplicate channel: {channel!r}")
        seen_keys.add(key)
        source = table_by_key.get(key)
        if source is None:
            missing_from_table.append(channel)
            continue
        output_markers.append(canonical_name(channel))
        source_columns.append(source)

    extra_table_markers = [marker for marker in table_markers if marker_key(marker) not in seen_keys]
    if missing_from_table or extra_table_markers:
        raise ValueError(
            "Image/table marker mismatch. "
            f"Missing from table: {missing_from_table or 'none'}; "
            f"missing from image: {extra_table_markers or 'none'}"
        )
    if len(output_markers) != len(set(output_markers)):
        raise ValueError("Canonical OME channel names are not unique")
    return output_markers, source_columns


def preflight(plan: ConversionPlan) -> dict:
    if not plan.imageid.strip():
        raise ValueError("imageid cannot be empty")
    if not plan.layer_name.strip():
        raise ValueError("layer_name cannot be empty")
    primary = table_schema(plan.primary_csv)
    secondary = table_schema(plan.secondary_csv)
    if primary.path == secondary.path:
        raise ValueError("Primary and secondary CSV files must be different")
    if primary.marker_columns != secondary.marker_columns:
        raise ValueError("The two CSV files do not have identical marker columns")
    if primary.obs_columns != secondary.obs_columns:
        raise ValueError("The two CSV files do not have identical metadata columns")
    output_markers, source_columns = ordered_marker_mapping(plan.image_path, primary.marker_columns)
    mapping = pd.DataFrame(
        {
            "channel_index": np.arange(len(output_markers), dtype=int),
            "marker": output_markers,
            "source_column": source_columns,
        }
    )
    return {
        "n_markers": len(output_markers),
        "n_obs_columns": len(primary.obs_columns) + 1,
        "primary_csv": str(primary.path),
        "secondary_csv": str(secondary.path),
        "image_path": str(Path(plan.image_path).expanduser().resolve()),
        "output_path": str(Path(plan.output_path).expanduser().resolve()),
        "layer_name": plan.layer_name,
        "marker_mapping": mapping.to_dict(orient="records"),
    }


def _cell_keys(frame: pd.DataFrame) -> pd.Index:
    key_columns = ["label"]
    if "fov" in frame.columns:
        key_columns.insert(0, "fov")
    if len(key_columns) == 1:
        keys = pd.Index(frame["label"].astype(str), name="label")
    else:
        keys = pd.Index(
            frame["fov"].astype(str) + "\x1f" + frame["label"].astype(str),
            name="fov_label",
        )
    if keys.has_duplicates:
        raise ValueError(f"Cell identity columns {key_columns} do not uniquely identify rows")
    return keys


def _obs_names(obs: pd.DataFrame, imageid: str) -> pd.Index:
    labels = obs["label"].astype(str)
    if "fov" in obs.columns and obs["fov"].astype(str).nunique() > 1:
        names = imageid + "_" + obs["fov"].astype(str) + "_" + labels
    else:
        names = imageid + "_" + labels
    names = pd.Index(names, name="cell_id")
    if names.has_duplicates:
        raise ValueError("Generated AnnData cell IDs are not unique")
    return names


def _read_primary(schema: TableSchema) -> pd.DataFrame:
    dtype = {marker: np.float32 for marker in schema.marker_columns}
    return pd.read_csv(schema.path, usecols=schema.marker_columns + schema.obs_columns, dtype=dtype)


def _read_secondary(schema: TableSchema, identity_columns: list[str]) -> pd.DataFrame:
    dtype = {marker: np.float32 for marker in schema.marker_columns}
    usecols = schema.marker_columns + identity_columns
    return pd.read_csv(schema.path, usecols=usecols, dtype=dtype)


def _align_secondary(primary: pd.DataFrame, secondary: pd.DataFrame) -> pd.DataFrame:
    primary_keys = _cell_keys(primary)
    secondary_keys = _cell_keys(secondary)
    if primary_keys.equals(secondary_keys):
        return secondary
    missing = primary_keys.difference(secondary_keys)
    extra = secondary_keys.difference(primary_keys)
    if len(missing) or len(extra):
        raise ValueError(
            "The two tables contain different cells: "
            f"{len(missing)} missing from secondary and {len(extra)} extra in secondary"
        )
    secondary = secondary.copy()
    secondary.index = secondary_keys
    return secondary.loc[primary_keys].reset_index(drop=True)


def _coordinate_columns(obs: pd.DataFrame) -> tuple[str, str] | None:
    candidates = [
        ("centroid-1", "centroid-0"),
        ("X_centroid", "Y_centroid"),
        ("x", "y"),
    ]
    return next(((x, y) for x, y in candidates if x in obs and y in obs), None)


def write_qc(adata: ad.AnnData, output_path: Path, *, max_cells: int = 50_000) -> dict[str, str]:
    qc_dir = output_path.parent / f"{output_path.stem}_qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    indexes = np.arange(adata.n_obs)
    if len(indexes) > max_cells:
        indexes = np.sort(rng.choice(indexes, size=max_cells, replace=False))
    values = np.asarray(adata.X[indexes], dtype=float)

    stats = pd.DataFrame(
        {
            "marker": adata.var_names,
            "mean": np.nanmean(values, axis=0),
            "median": np.nanmedian(values, axis=0),
            "q95": np.nanpercentile(values, 95, axis=0),
            "q99": np.nanpercentile(values, 99, axis=0),
            "maximum": np.nanmax(values, axis=0),
        }
    )
    stats_path = qc_dir / "marker_expression_summary.csv"
    stats.to_csv(stats_path, index=False)

    mapping_path = qc_dir / "marker_channel_order.csv"
    adata.var.reset_index(names="marker").to_csv(mapping_path, index=False)

    ncols = 5
    nrows = int(np.ceil(adata.n_vars / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 2.35 * nrows), dpi=150)
    axes = np.atleast_1d(axes).ravel()
    for marker_index, (axis, marker) in enumerate(zip(axes, adata.var_names)):
        marker_values = values[:, marker_index]
        finite = marker_values[np.isfinite(marker_values)]
        axis.hist(finite, bins=80, color="#287D8E", alpha=0.9)
        axis.set_title(str(marker), fontsize=9)
        axis.tick_params(labelsize=7)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in axes[adata.n_vars:]:
        axis.axis("off")
    fig.suptitle(f"{adata.uns['cellsam_conversion']['imageid']} marker distributions", fontsize=14, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    histogram_path = qc_dir / "marker_distributions.png"
    fig.savefig(histogram_path, bbox_inches="tight")
    plt.close(fig)

    outputs = {
        "marker_summary_csv": str(stats_path),
        "marker_order_csv": str(mapping_path),
        "marker_histograms_png": str(histogram_path),
    }
    coordinates = _coordinate_columns(adata.obs)
    if coordinates is not None:
        x_column, y_column = coordinates
        spatial_indexes = indexes
        x = pd.to_numeric(adata.obs.iloc[spatial_indexes][x_column], errors="coerce").to_numpy()
        y = pd.to_numeric(adata.obs.iloc[spatial_indexes][y_column], errors="coerce").to_numpy()
        color = (
            pd.to_numeric(adata.obs.iloc[spatial_indexes]["area"], errors="coerce").to_numpy()
            if "area" in adata.obs
            else np.ones(len(spatial_indexes))
        )
        fig, axis = plt.subplots(figsize=(8, 8), dpi=160)
        points = axis.scatter(x, y, c=np.log1p(color), s=1.0, cmap="viridis", linewidths=0, rasterized=True)
        axis.invert_yaxis()
        axis.set_aspect("equal")
        axis.set_title("Cell centroid coverage")
        axis.set_xlabel(x_column)
        axis.set_ylabel(y_column)
        fig.colorbar(points, ax=axis, label="log1p(area)")
        fig.tight_layout()
        spatial_path = qc_dir / "cell_centroid_coverage.png"
        fig.savefig(spatial_path, bbox_inches="tight")
        plt.close(fig)
        outputs["spatial_qc_png"] = str(spatial_path)
    return outputs


def build_anndata(plan: ConversionPlan, status: StatusCallback | None = None) -> tuple[ad.AnnData, dict]:
    def update(stage: str, message: str, progress: float) -> None:
        if status is not None:
            status(stage, message, progress)

    update("preflight", "Validating table schemas and OME channel order", 0.05)
    report = preflight(plan)
    primary_schema = table_schema(plan.primary_csv)
    secondary_schema = table_schema(plan.secondary_csv)
    output_markers = [row["marker"] for row in report["marker_mapping"]]
    source_columns = [row["source_column"] for row in report["marker_mapping"]]

    update("read_primary", "Reading the primary expression table", 0.18)
    primary = _read_primary(primary_schema)
    identity_columns = [column for column in ["label", "fov"] if column in primary.columns]
    update("read_secondary", "Reading the secondary expression layer", 0.40)
    secondary = _read_secondary(secondary_schema, identity_columns)
    update("align", "Checking cell identities and aligning both matrices", 0.56)
    secondary = _align_secondary(primary, secondary)

    obs = primary[primary_schema.obs_columns].copy()
    obs.insert(0, "imageid", plan.imageid.strip())
    obs.index = _obs_names(obs, plan.imageid.strip())
    var = pd.DataFrame(
        {
            "channel_index": np.arange(len(output_markers), dtype=int),
            "source_column": source_columns,
        },
        index=pd.Index(output_markers, name="marker"),
    )
    x = primary[source_columns].to_numpy(dtype=np.float32, copy=True)
    layer = secondary[source_columns].to_numpy(dtype=np.float32, copy=True)
    update("assemble", "Assembling AnnData in OME channel order", 0.68)
    adata = ad.AnnData(X=x, obs=obs, var=var)
    adata.layers[plan.layer_name] = layer
    adata.uns["all_markers"] = np.asarray(output_markers, dtype=str)
    adata.uns["cellsam_conversion"] = {
        "created_at": now(),
        "imageid": plan.imageid.strip(),
        "primary_csv": str(primary_schema.path),
        "secondary_csv": str(secondary_schema.path),
        "source_image": str(Path(plan.image_path).expanduser().resolve()),
        "x_source": primary_schema.path.stem,
        "layer_name": plan.layer_name,
        "layer_source": secondary_schema.path.stem,
        "cell_size_removed": True,
        "channel_order_source": "OME metadata",
    }

    output_path = Path(plan.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    update("write", "Writing compressed H5AD", 0.78)
    adata.write_h5ad(output_path, compression="gzip")

    qc_outputs: dict[str, str] = {}
    if plan.make_qc:
        update("qc", "Generating marker-distribution and spatial QC", 0.90)
        qc_outputs = write_qc(adata, output_path)

    manifest = {
        **report,
        "created_at": now(),
        "n_cells": int(adata.n_obs),
        "n_markers": int(adata.n_vars),
        "obs_columns": list(adata.obs.columns),
        "all_markers": list(adata.var_names),
        "qc_outputs": qc_outputs,
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    update("complete", "AnnData conversion complete", 1.0)
    return adata, manifest


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--primary-csv", type=Path, required=True)
    cli.add_argument("--secondary-csv", type=Path, required=True)
    cli.add_argument("--image", type=Path, required=True)
    cli.add_argument("--imageid", required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--layer-name", default="size_normalized")
    cli.add_argument("--no-qc", action="store_true")
    cli.add_argument("--inspect-only", action="store_true")
    cli.add_argument("--status", type=Path)
    return cli


def main() -> None:
    args = parser().parse_args()
    plan = ConversionPlan(
        primary_csv=args.primary_csv,
        secondary_csv=args.secondary_csv,
        image_path=args.image,
        imageid=args.imageid,
        output_path=args.output,
        layer_name=args.layer_name,
        make_qc=not args.no_qc,
    )

    def status(stage: str, message: str, progress: float) -> None:
        if args.status is not None:
            write_json(
                args.status,
                {
                    "state": "complete" if stage == "complete" else "running",
                    "stage": stage,
                    "message": message,
                    "progress": progress,
                    "updated_at": now(),
                },
            )
        print(f"[{stage}] {message}", flush=True)

    try:
        if args.inspect_only:
            print(json.dumps(preflight(plan), indent=2))
            return
        _, manifest = build_anndata(plan, status=status)
        if args.status is not None:
            payload = json.loads(args.status.read_text(encoding="utf-8"))
            payload["outputs"] = manifest
            write_json(args.status, payload)
        print(json.dumps(manifest, indent=2))
    except Exception as error:
        if args.status is not None:
            write_json(
                args.status,
                {
                    "state": "failed",
                    "stage": "failed",
                    "message": str(error),
                    "updated_at": now(),
                    "traceback": traceback.format_exc(),
                },
            )
        raise


if __name__ == "__main__":
    main()
