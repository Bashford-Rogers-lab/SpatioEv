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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spatioev_cellsam_matplotlib")
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

from ._io import now, write_json
from .image_collection import generic_channel_names

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

ROLE_MARKER = "marker"
ROLE_CELL_ID = "cell_id"
ROLE_X = "x_coordinate"
ROLE_Y = "y_coordinate"
ROLE_GROUP = "group_id"
ROLE_OBSERVATION = "observation"
ROLE_IGNORE = "ignore"
ROLE_LABELS = {
    ROLE_MARKER: "Expression marker (.X / layer)",
    ROLE_CELL_ID: "Cell identifier",
    ROLE_X: "X coordinate",
    ROLE_Y: "Y coordinate",
    ROLE_GROUP: "FOV / group identifier",
    ROLE_OBSERVATION: "Observation metadata (.obs)",
    ROLE_IGNORE: "Ignore",
}
ROLE_OPTIONS = tuple(ROLE_LABELS)

CELL_ID_ALIASES = ("label", "cell_id", "cellid", "object_id", "objectid")
GROUP_ID_ALIASES = ("fov", "field_of_view", "roi", "region")
COORDINATE_ALIASES = (
    ("X_centroid", "Y_centroid"),
    ("centroid_x", "centroid_y"),
    ("centroid-1", "centroid-0"),
    ("x", "y"),
)


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
    column_roles: dict[str, str] | None = None
    marker_targets: dict[str, str] | None = None
    # Optional marker order CSV. When supplied it defines the channel order
    # instead of the OME metadata, because TMA exports routinely lose their
    # channel names and an unnamed image cannot arbitrate its own marker order.
    marker_manifest: Path | None = None


StatusCallback = Callable[[str, str, float], None]


def read_marker_manifest(path: Path) -> pd.DataFrame:
    """Read a marker order CSV, whose *row order* defines the channel order.

    Row 1 is the first image plane, row 2 the second, and so on.

    ``channel_number`` is optional metadata and is interpreted by what it
    contains. Unique values are a *global* plane index, so they must ascend
    with the rows -- a manifest that contradicts itself there is rejected
    rather than silently resolved in one direction, because an image whose OME
    channel names were lost cannot arbitrate between the two. Repeated values
    mean the column counts channels *within* an imaging cycle (a cycle-based
    CODEX/PhenoCycler panel restarts at 1 every round), which says nothing
    about global order; it is then kept as metadata and the row order stands.
    """
    manifest = pd.read_csv(Path(path).expanduser().resolve())
    if "marker_name" not in manifest:
        raise ValueError("Marker manifest must contain a 'marker_name' column")
    manifest["marker_name"] = manifest["marker_name"].astype(str).str.strip()
    if manifest["marker_name"].duplicated().any():
        duplicated = manifest.loc[
            manifest["marker_name"].duplicated(keep=False), "marker_name"
        ].tolist()
        raise ValueError(f"Marker manifest contains duplicated names: {duplicated}")
    if "channel_number" in manifest:
        numbers = pd.to_numeric(manifest["channel_number"], errors="raise")
        # Repeats mean the column is cycle-local (1, 2, 3, 1, 2, 3, ...) rather
        # than a global plane index, so it carries no claim about global order
        # and must not be checked against the row order.
        cycle_local = bool(numbers.duplicated().any())
        out_of_order = (
            []
            if cycle_local
            else [
                f"row {index + 2} ({name}, channel_number={number})"
                for index, (name, number, previous) in enumerate(
                    zip(
                        manifest["marker_name"][1:],
                        numbers[1:],
                        numbers[:-1],
                        strict=True,
                    )
                )
                if number <= previous
            ]
        )
        if out_of_order:
            raise ValueError(
                "Marker order CSV row order disagrees with its own "
                f"'channel_number' column at: {out_of_order}. The row order "
                "defines the marker order, so sort the rows by channel_number "
                "(or delete the column) and run the conversion again."
            )
        manifest["channel_number"] = numbers
    else:
        manifest.insert(0, "channel_number", np.arange(1, len(manifest) + 1, dtype=int))
    return manifest.reset_index(drop=True)


def manifest_marker_order(plan: ConversionPlan) -> list[str] | None:
    """Marker order from ``plan.marker_manifest``, or ``None`` when unset."""
    if plan.marker_manifest is None:
        return None
    order = read_marker_manifest(plan.marker_manifest)["marker_name"].tolist()
    planes = len(image_channel_names(plan.image_path))
    if len(order) != planes:
        raise ValueError(
            f"Marker order CSV lists {len(order)} markers but "
            f"{Path(plan.image_path).name} has {planes} image channels"
        )
    return order


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
    if "label" in columns:
        label_index = columns.index("label")
        marker_columns = [
            column for column in columns[:label_index] if column != "cell_size"
        ]
        obs_columns = columns[label_index:]
    else:
        marker_columns = []
        obs_columns = columns.copy()
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
        path
        for path in directory.glob("*.csv")
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
    channels = [
        node for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "Channel"
    ]
    return [node.attrib.get("Name", f"C{index}") for index, node in enumerate(channels)]


def display_channel_names(image_path: Path) -> list[str]:
    """One display name per *physical* image plane, with repeats disambiguated.

    ``adata.uns["all_markers"]`` is consumed by ``scimap.pl.image_viewer``,
    which asserts one name per channel present in the image::

        AssertionError: number of channel names (17) must match
        number of channels (18)

    ``var_names`` holds one entry per *distinct* marker, so the two diverge as
    soon as an image repeats a channel name -- routine in cyclic imaging. This
    returns a name per plane so they line up again, suffixing repeats
    (``DAPI``, ``DAPI (2)``) to keep napari layers distinguishable.
    """
    seen: dict[str, int] = {}
    display: list[str] = []
    for name in image_channel_names(image_path):
        canonical = canonical_name(name)
        seen[canonical] = seen.get(canonical, 0) + 1
        display.append(
            canonical if seen[canonical] == 1 else f"{canonical} ({seen[canonical]})"
        )
    return display


def canonical_name(name: str) -> str:
    clean = str(name).strip()
    return CHANNEL_ALIASES.get(clean, clean)


def marker_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", canonical_name(name).lower())


def ordered_marker_mapping(
    image_path: Path, table_markers: list[str]
) -> tuple[list[str], list[str]]:
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
            # Repeated channel names are normal in cyclic imaging; the first
            # plane is authoritative, matching inspect_inputs().
            continue
        seen_keys.add(key)
        source = table_by_key.get(key)
        if source is None:
            missing_from_table.append(channel)
            continue
        output_markers.append(canonical_name(channel))
        source_columns.append(source)

    if missing_from_table:
        raise ValueError(
            "Image/table marker mismatch. "
            f"Missing from table: {missing_from_table or 'none'}; "
            "non-marker table columns are permitted and will be stored in .obs"
        )
    if len(output_markers) != len(set(output_markers)):
        raise ValueError("Canonical OME channel names are not unique")
    return output_markers, source_columns


def _find_named_column(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    by_key = {marker_key(column): column for column in columns}
    return next(
        (by_key[marker_key(alias)] for alias in aliases if marker_key(alias) in by_key),
        None,
    )


def _find_coordinate_pair(columns: list[str]) -> tuple[str, str] | None:
    by_key = {marker_key(column): column for column in columns}
    for x_alias, y_alias in COORDINATE_ALIASES:
        x_column = by_key.get(marker_key(x_alias))
        y_column = by_key.get(marker_key(y_alias))
        if x_column is not None and y_column is not None:
            return x_column, y_column
    return None


def _automatic_roles(
    columns: list[str], image_channels: list[str]
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    roles = {column: ROLE_OBSERVATION for column in columns}
    targets: dict[str, str] = {}
    reasons = {
        column: "Not an image channel; preserve as observation metadata"
        for column in columns
    }

    columns_by_key: dict[str, list[str]] = {}
    for column in columns:
        columns_by_key.setdefault(marker_key(column), []).append(column)
    for channel in image_channels:
        candidates = columns_by_key.get(marker_key(channel), [])
        if len(candidates) == 1:
            column = candidates[0]
            roles[column] = ROLE_MARKER
            targets[column] = channel
            reasons[column] = f"Matches OME channel {channel!r}"

    cell_id = _find_named_column(columns, CELL_ID_ALIASES)
    if cell_id is not None and roles[cell_id] != ROLE_MARKER:
        roles[cell_id] = ROLE_CELL_ID
        reasons[cell_id] = "Recognized cell identifier"

    group_id = _find_named_column(columns, GROUP_ID_ALIASES)
    if group_id is not None and roles[group_id] != ROLE_MARKER:
        roles[group_id] = ROLE_GROUP
        reasons[group_id] = "Recognized FOV or group identifier"

    coordinates = _find_coordinate_pair(columns)
    if coordinates is not None:
        x_column, y_column = coordinates
        if roles[x_column] != ROLE_MARKER:
            roles[x_column] = ROLE_X
            reasons[x_column] = "Recognized X-coordinate alias"
        if roles[y_column] != ROLE_MARKER:
            roles[y_column] = ROLE_Y
            reasons[y_column] = "Recognized Y-coordinate alias"
    return roles, targets, reasons


def inspect_inputs(plan: ConversionPlan) -> dict:
    """Inspect inputs and infer editable CSV-column roles without reading all rows."""

    if not plan.imageid.strip():
        raise ValueError("imageid cannot be empty")
    if not plan.layer_name.strip():
        raise ValueError("layer_name cannot be empty")
    primary = table_schema(plan.primary_csv)
    secondary = table_schema(plan.secondary_csv)
    if primary.path == secondary.path:
        raise ValueError("Primary and secondary CSV files must be different")
    file_channels = image_channel_names(plan.image_path)
    manifest_order = manifest_marker_order(plan)
    # The manifest replaces the OME channel names positionally: manifest row i
    # names image plane i. Everything downstream keys off `image_channels`, so
    # substituting here is enough to put var_names in marker-order-CSV order.
    image_channels = manifest_order if manifest_order is not None else file_channels
    roles, targets, reasons = _automatic_roles(primary.columns, image_channels)

    for column, role in (plan.column_roles or {}).items():
        if column not in roles:
            raise ValueError(
                f"Column-role override references unknown column: {column!r}"
            )
        if role not in ROLE_OPTIONS:
            raise ValueError(f"Unsupported role {role!r} for column {column!r}")
        roles[column] = role
        reasons[column] = "User-selected role"
    for column, channel in (plan.marker_targets or {}).items():
        if column not in roles:
            raise ValueError(f"Marker target references unknown column: {column!r}")
        targets[column] = channel
        reasons[column] = "User-selected marker target"
    targets = {
        column: target
        for column, target in targets.items()
        if roles[column] == ROLE_MARKER and target
    }

    errors: list[str] = []
    warnings: list[str] = []
    # Overriding a *named* image is legitimate but must never be silent: say so
    # when the file carried real names and the manifest disagrees with them.
    if (
        manifest_order is not None
        and not generic_channel_names(file_channels)
        and [marker_key(name) for name in file_channels]
        != [marker_key(name) for name in manifest_order]
    ):
        warnings.append(
            f"The marker order CSV overrides the channel names in "
            f"{Path(plan.image_path).name}. Image order: {file_channels}. "
            f"Marker order CSV: {manifest_order}."
        )
    # A column sitting in the table's own marker block (everything before
    # "label" in an ARK/CellSAM export) that finds no channel is dropped from .X
    # and kept as .obs. That is a legitimate outcome, but silently returning
    # fewer markers than the table offers reads as "my markers do not line up",
    # so name them.
    demoted = [
        column
        for column in primary.marker_columns
        if roles.get(column) == ROLE_OBSERVATION
    ]
    if demoted:
        warnings.append(
            f"These expression columns have no matching image channel and are "
            f"kept as .obs metadata rather than markers: {demoted}. Supply a "
            f"marker order CSV if they should be markers."
        )
    image_channel_keys = [marker_key(channel) for channel in image_channels]
    duplicate_image_channels = sorted(
        {
            channel
            for channel, key in zip(image_channels, image_channel_keys, strict=True)
            if image_channel_keys.count(key) > 1
        }
    )
    # Repeated channel names are normal in cyclic imaging: CODEX/PhenoCycler
    # re-image the nuclear stain every round, so several planes legitimately
    # share a name. That is only *ambiguous* if a marker column has to choose
    # between them, which is checked once the marker targets are resolved
    # below. An unused repeat is harmless and must not block the conversion.
    role_columns = {
        role: [column for column, value in roles.items() if value == role]
        for role in ROLE_OPTIONS
    }
    if len(role_columns[ROLE_CELL_ID]) != 1:
        errors.append(
            f"Select exactly one cell identifier; found {len(role_columns[ROLE_CELL_ID])}"
        )
    for role, label in (
        (ROLE_X, "X coordinate"),
        (ROLE_Y, "Y coordinate"),
        (ROLE_GROUP, "FOV/group identifier"),
    ):
        if len(role_columns[role]) > 1:
            errors.append(
                f"Select at most one {label}; found {len(role_columns[role])}"
            )
    if not role_columns[ROLE_X] or not role_columns[ROLE_Y]:
        warnings.append(
            "Spatial coordinates were not identified; AnnData can be built, but spatial workflow pages require them"
        )

    # Keep the FIRST plane when a channel name repeats. A dict comprehension
    # would keep the last, which is both arbitrary and surprising: in cyclic
    # imaging the first occurrence is the reference round.
    channel_by_key: dict[str, str] = {}
    for channel in image_channels:
        channel_by_key.setdefault(marker_key(channel), channel)
    resolved_targets: dict[str, str] = {}
    for column in role_columns[ROLE_MARKER]:
        target = targets.get(column)
        if target is None:
            target = channel_by_key.get(marker_key(column))
        if target is None or marker_key(target) not in channel_by_key:
            errors.append(
                f"Marker column {column!r} does not have a valid OME channel target"
            )
            continue
        resolved_targets[column] = channel_by_key[marker_key(target)]

    assigned_channels = list(resolved_targets.values())

    # Repeated channel names do not make the AnnData ambiguous: markers come
    # from CSV columns, which are unique, so var_names stay unique either way.
    # The repeat only decides which image plane a marker is displayed against,
    # and that is resolved deterministically to the first occurrence above.
    if duplicate_image_channels:
        warnings.append(
            f"OME image repeats channel name(s) {duplicate_image_channels}; the "
            "first plane of each is used for image review. This is expected for "
            "cyclic imaging, where the nuclear stain is re-imaged every round."
        )

    duplicates = sorted(
        {
            channel
            for channel in assigned_channels
            if assigned_channels.count(channel) > 1
        }
    )
    if duplicates:
        errors.append(
            f"Multiple CSV columns are assigned to OME channels: {duplicates}"
        )
    # Compare on marker keys, not raw names: a channel name that appears twice
    # is covered once the CSV supplies a single column for it.
    covered_keys = {marker_key(channel) for channel in assigned_channels}
    missing_channels = sorted(
        {
            channel
            for channel in image_channels
            if marker_key(channel) not in covered_keys
        },
        key=image_channels.index,
    )
    if missing_channels:
        errors.append(f"OME channels without an expression column: {missing_channels}")

    secondary_missing = [
        column
        for column in [
            *role_columns[ROLE_MARKER],
            *role_columns[ROLE_CELL_ID],
            *role_columns[ROLE_GROUP],
        ]
        if column not in secondary.columns
    ]
    if secondary_missing:
        errors.append(
            f"Columns required from the secondary table are missing: {secondary_missing}"
        )

    marker_rows = []
    # Emit one row per distinct channel name. Iterating raw image_channels
    # would add a row per repeated plane, giving duplicate var_names and
    # reading the same CSV column several times.
    seen_channel_keys: set[str] = set()
    for channel_index, channel in enumerate(image_channels):
        channel_key = marker_key(channel)
        if channel_key in seen_channel_keys:
            continue
        seen_channel_keys.add(channel_key)
        source = next(
            (
                column
                for column, target in resolved_targets.items()
                if marker_key(target) == channel_key
            ),
            None,
        )
        if source is not None:
            marker_rows.append(
                {
                    "channel_index": channel_index,
                    "marker": canonical_name(channel),
                    "source_column": source,
                }
            )

    role_rows = []
    for column in primary.columns:
        role_rows.append(
            {
                "column": column,
                "role": roles[column],
                "role_label": ROLE_LABELS[roles[column]],
                "image_channel": resolved_targets.get(column, ""),
                "reason": reasons[column],
            }
        )
    obs_columns = [
        column
        for column in primary.columns
        if roles[column] not in {ROLE_MARKER, ROLE_IGNORE}
    ]
    return {
        "n_markers": len(marker_rows),
        "n_obs_columns": len(obs_columns) + 1,
        "primary_csv": str(primary.path),
        "secondary_csv": str(secondary.path),
        "image_path": str(Path(plan.image_path).expanduser().resolve()),
        "output_path": str(Path(plan.output_path).expanduser().resolve()),
        "layer_name": plan.layer_name,
        "image_channels": image_channels,
        "marker_mapping": marker_rows,
        "column_roles": role_rows,
        "errors": errors,
        "warnings": warnings,
        "valid": not errors,
    }


def preflight(plan: ConversionPlan) -> dict:
    report = inspect_inputs(plan)
    if report["errors"]:
        raise ValueError("Input schema is not valid: " + "; ".join(report["errors"]))
    return report


def _cell_keys(
    frame: pd.DataFrame, cell_id_column: str, group_column: str | None
) -> pd.Index:
    key_columns = [cell_id_column]
    if group_column is not None:
        key_columns.insert(0, group_column)
    if group_column is None:
        keys = pd.Index(frame[cell_id_column].astype(str), name=cell_id_column)
    else:
        keys = pd.Index(
            frame[group_column].astype(str)
            + "\x1f"
            + frame[cell_id_column].astype(str),
            name=f"{group_column}_{cell_id_column}",
        )
    if keys.has_duplicates:
        raise ValueError(
            f"Cell identity columns {key_columns} do not uniquely identify rows"
        )
    return keys


def _obs_names(
    obs: pd.DataFrame, imageid: str, cell_id_column: str, group_column: str | None
) -> pd.Index:
    labels = obs[cell_id_column].astype(str)
    if group_column is not None and obs[group_column].astype(str).nunique() > 1:
        names = imageid + "_" + obs[group_column].astype(str) + "_" + labels
    else:
        names = imageid + "_" + labels
    names = pd.Index(names, name="cell_id")
    if names.has_duplicates:
        raise ValueError("Generated AnnData cell IDs are not unique")
    return names


def _read_primary(
    schema: TableSchema, marker_columns: list[str], obs_columns: list[str]
) -> pd.DataFrame:
    dtype = {marker: np.float32 for marker in marker_columns}
    return pd.read_csv(schema.path, usecols=marker_columns + obs_columns, dtype=dtype)


def _read_secondary(
    schema: TableSchema, marker_columns: list[str], identity_columns: list[str]
) -> pd.DataFrame:
    dtype = {marker: np.float32 for marker in marker_columns}
    usecols = marker_columns + identity_columns
    return pd.read_csv(schema.path, usecols=usecols, dtype=dtype)


def _align_secondary(
    primary: pd.DataFrame,
    secondary: pd.DataFrame,
    cell_id_column: str,
    group_column: str | None,
) -> pd.DataFrame:
    primary_keys = _cell_keys(primary, cell_id_column, group_column)
    secondary_keys = _cell_keys(secondary, cell_id_column, group_column)
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
        ("X_centroid", "Y_centroid"),
        ("centroid_x", "centroid_y"),
        ("centroid-1", "centroid-0"),
        ("x", "y"),
    ]
    return next(((x, y) for x, y in candidates if x in obs and y in obs), None)


def _drop_duplicate_cell_size(obs: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    if "cell_size" not in obs:
        return obs, False
    for area_column in ("area", "cell_area_px2", "cell_area"):
        if area_column not in obs:
            continue
        cell_size = pd.to_numeric(obs["cell_size"], errors="coerce").to_numpy(
            dtype=float
        )
        area = pd.to_numeric(obs[area_column], errors="coerce").to_numpy(dtype=float)
        if np.allclose(cell_size, area, equal_nan=True):
            return obs.drop(columns="cell_size"), True
    return obs, False


def write_qc(
    adata: ad.AnnData, output_path: Path, *, max_cells: int = 50_000
) -> dict[str, str]:
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
    for axis in axes[adata.n_vars :]:
        axis.axis("off")
    fig.suptitle(
        f"{adata.uns['cellsam_conversion']['imageid']} marker distributions",
        fontsize=14,
        y=0.998,
    )
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
        x = pd.to_numeric(
            adata.obs.iloc[spatial_indexes][x_column], errors="coerce"
        ).to_numpy()
        y = pd.to_numeric(
            adata.obs.iloc[spatial_indexes][y_column], errors="coerce"
        ).to_numpy()
        color = (
            pd.to_numeric(
                adata.obs.iloc[spatial_indexes]["area"], errors="coerce"
            ).to_numpy()
            if "area" in adata.obs
            else np.ones(len(spatial_indexes))
        )
        fig, axis = plt.subplots(figsize=(8, 8), dpi=160)
        points = axis.scatter(
            x,
            y,
            c=np.log1p(color),
            s=1.0,
            cmap="viridis",
            linewidths=0,
            rasterized=True,
        )
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


def build_anndata(
    plan: ConversionPlan, status: StatusCallback | None = None
) -> tuple[ad.AnnData, dict]:
    def update(stage: str, message: str, progress: float) -> None:
        if status is not None:
            status(stage, message, progress)

    update("preflight", "Validating table schemas and OME channel order", 0.05)
    report = preflight(plan)
    primary_schema = table_schema(plan.primary_csv)
    secondary_schema = table_schema(plan.secondary_csv)
    output_markers = [row["marker"] for row in report["marker_mapping"]]
    source_columns = [row["source_column"] for row in report["marker_mapping"]]
    roles = {row["column"]: row["role"] for row in report["column_roles"]}
    obs_columns = [
        column
        for column in primary_schema.columns
        if roles[column] not in {ROLE_MARKER, ROLE_IGNORE}
    ]
    cell_id_column = next(
        column for column, role in roles.items() if role == ROLE_CELL_ID
    )
    group_column = next(
        (column for column, role in roles.items() if role == ROLE_GROUP), None
    )
    x_column = next((column for column, role in roles.items() if role == ROLE_X), None)
    y_column = next((column for column, role in roles.items() if role == ROLE_Y), None)

    update("read_primary", "Reading the primary expression table", 0.18)
    primary = _read_primary(primary_schema, source_columns, obs_columns)
    identity_columns = [cell_id_column] + (
        [group_column] if group_column is not None else []
    )
    update("read_secondary", "Reading the secondary expression layer", 0.40)
    secondary = _read_secondary(secondary_schema, source_columns, identity_columns)
    update("align", "Checking cell identities and aligning both matrices", 0.56)
    secondary = _align_secondary(primary, secondary, cell_id_column, group_column)

    obs = primary[obs_columns].copy()
    obs, cell_size_removed = _drop_duplicate_cell_size(obs)
    if "imageid" in obs:
        obs["imageid"] = plan.imageid.strip()
    else:
        obs.insert(0, "imageid", plan.imageid.strip())
    if x_column is not None and y_column is not None:
        obs["X_centroid"] = pd.to_numeric(obs[x_column], errors="raise").to_numpy(
            dtype=float
        )
        obs["Y_centroid"] = pd.to_numeric(obs[y_column], errors="raise").to_numpy(
            dtype=float
        )
    if "area" not in obs:
        for area_column in ("cell_area_px2", "cell_area"):
            if area_column in obs:
                obs["area"] = pd.to_numeric(obs[area_column], errors="coerce").to_numpy(
                    dtype=float
                )
                break
    obs.index = _obs_names(obs, plan.imageid.strip(), cell_id_column, group_column)
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
    if {"X_centroid", "Y_centroid"}.issubset(adata.obs.columns):
        adata.obsm["spatial"] = adata.obs[["X_centroid", "Y_centroid"]].to_numpy(
            dtype=float
        )
    # One name per *image plane*, not per marker: scimap.pl.image_viewer
    # asserts these line up with the channels in the file. A marker order CSV
    # already supplies exactly one name per plane, so it is used verbatim;
    # otherwise the OME names are de-duplicated for the repeated-channel case.
    manifest_order = manifest_marker_order(plan)
    adata.uns["all_markers"] = np.asarray(
        manifest_order
        if manifest_order is not None
        else display_channel_names(plan.image_path),
        dtype=str,
    )
    adata.uns["cellsam_conversion"] = {
        "created_at": now(),
        "imageid": plan.imageid.strip(),
        "primary_csv": str(primary_schema.path),
        "secondary_csv": str(secondary_schema.path),
        "source_image": str(Path(plan.image_path).expanduser().resolve()),
        "x_source": primary_schema.path.stem,
        "layer_name": plan.layer_name,
        "layer_source": secondary_schema.path.stem,
        "cell_id_source": cell_id_column,
        "group_id_source": group_column,
        "x_coordinate_source": x_column,
        "y_coordinate_source": y_column,
        "cell_size_removed": cell_size_removed,
        "channel_order_source": (
            "marker order CSV" if manifest_order is not None else "OME metadata"
        ),
        "marker_manifest": (
            str(Path(plan.marker_manifest).expanduser().resolve())
            if plan.marker_manifest is not None
            else None
        ),
        "column_roles": roles,
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
    cli.add_argument(
        "--marker-manifest",
        type=Path,
        help="Marker order CSV. Its row order defines the channel order, "
        "overriding the OME channel names.",
    )
    cli.add_argument("--schema-config", type=Path)
    cli.add_argument("--no-qc", action="store_true")
    cli.add_argument("--inspect-only", action="store_true")
    cli.add_argument("--status", type=Path)
    return cli


def main() -> None:
    args = parser().parse_args()
    schema_config = {}
    if args.schema_config is not None:
        schema_config = json.loads(args.schema_config.read_text(encoding="utf-8"))
    plan = ConversionPlan(
        primary_csv=args.primary_csv,
        secondary_csv=args.secondary_csv,
        image_path=args.image,
        imageid=args.imageid,
        output_path=args.output,
        layer_name=args.layer_name,
        make_qc=not args.no_qc,
        column_roles=schema_config.get("column_roles"),
        marker_targets=schema_config.get("marker_targets"),
        marker_manifest=args.marker_manifest,
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
            print(json.dumps(inspect_inputs(plan), indent=2))
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
