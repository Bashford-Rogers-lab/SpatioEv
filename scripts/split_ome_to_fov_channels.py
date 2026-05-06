#!/usr/bin/env python3
"""Split a whole-slide OME-TIFF into per-FOV, per-channel TIFFs.

This is a preprocessing helper for ``spatioev.spatial.cell_pixel_features``,
which expects an ``img_dir`` layout like:

    img_dir/
      fov0/
        CK19.tiff
        DNA_1.tiff
        NaKATPase.tiff
      fov1/
        ...

The script infers the FOV tile size from the segmentation masks and then slices
the OME stack into row-major FOV tiles. If the OME image is slightly larger
than the exact tiled layout, the extra pixels are trimmed from the bottom and
right edges.
"""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import tifffile


OME_NS = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}


@dataclass(frozen=True)
class Layout:
    fovs: list[str]
    tile_height: int
    tile_width: int
    rows: int
    cols: int
    used_height: int
    used_width: int
    trim_bottom: int
    trim_right: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a whole-slide OME-TIFF into per-FOV per-channel TIFFs.",
    )
    parser.add_argument(
        "--ome-tif",
        required=True,
        help="Path to the whole-slide OME-TIFF (expected axes CYX).",
    )
    parser.add_argument(
        "--seg-dir",
        required=True,
        help="Directory containing per-FOV segmentation masks such as fov0_whole_cell.tiff.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory to create img_dir/fovX/channel.tiff files in.",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=None,
        help="Channel names to export. Defaults to all channels in the OME image.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Optional manual row count for the FOV grid.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=None,
        help="Optional manual column count for the FOV grid.",
    )
    parser.add_argument(
        "--seg-suffix",
        default="_whole_cell.tiff",
        help="Suffix used to identify per-FOV whole-cell masks.",
    )
    parser.add_argument(
        "--layout-csv",
        default=None,
        help="Optional path to save the inferred FOV tile layout CSV.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing channel TIFFs if they already exist.",
    )
    return parser.parse_args()


def natural_fov_key(name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", name)
    if match:
        return (int(match.group(1)), name)
    return (10**9, name)


def parse_channel_names(ome_xml: str, expected_n: int) -> list[str]:
    if not ome_xml:
        raise ValueError("OME metadata is missing; cannot recover channel names.")

    root = ET.fromstring(ome_xml)
    channels = root.findall(".//ome:Channel", OME_NS)
    names = [ch.attrib.get("Name", f"channel_{idx}") for idx, ch in enumerate(channels)]

    if len(names) != expected_n:
        raise ValueError(
            f"Found {len(names)} channel names in OME metadata, expected {expected_n}."
        )
    return names


def get_segmentation_fovs(seg_dir: Path, seg_suffix: str) -> tuple[list[str], tuple[int, int]]:
    seg_paths = sorted(seg_dir.glob(f"*{seg_suffix}"), key=lambda p: natural_fov_key(p.stem.replace(seg_suffix[:-5], "")))
    if not seg_paths:
        raise FileNotFoundError(f"No segmentation masks ending with {seg_suffix!r} found in {seg_dir}")

    fovs: list[str] = []
    shapes: set[tuple[int, int]] = set()
    for seg_path in seg_paths:
        fov = seg_path.name[: -len(seg_suffix)]
        with tifffile.TiffFile(seg_path) as tif:
            shape = tuple(tif.series[0].shape)
        if len(shape) != 2:
            raise ValueError(f"Expected 2D segmentation mask for {seg_path}, got shape {shape}")
        fovs.append(fov)
        shapes.add(shape)

    if len(shapes) != 1:
        raise ValueError(f"Expected all segmentation masks to have the same shape, found: {sorted(shapes)}")

    return sorted(fovs, key=natural_fov_key), next(iter(shapes))


def infer_grid(
    full_height: int,
    full_width: int,
    tile_height: int,
    tile_width: int,
    n_fovs: int,
    rows: int | None,
    cols: int | None,
) -> tuple[int, int]:
    if rows is not None and cols is not None:
        if rows * cols != n_fovs:
            raise ValueError(f"rows*cols must equal number of FOVs ({n_fovs}), got {rows}*{cols}")
        return rows, cols

    candidates: list[tuple[int, int, int, int]] = []
    for r in range(1, n_fovs + 1):
        if n_fovs % r != 0:
            continue
        c = n_fovs // r
        used_h = r * tile_height
        used_w = c * tile_width
        if used_h > full_height or used_w > full_width:
            continue
        trim_h = full_height - used_h
        trim_w = full_width - used_w
        score = trim_h + trim_w
        candidates.append((score, trim_h * trim_w, r, c))

    if not candidates:
        raise ValueError(
            "Could not infer a valid FOV grid from the OME image and segmentation mask sizes."
        )

    candidates.sort()
    _, _, best_rows, best_cols = candidates[0]
    return best_rows, best_cols


def build_layout(
    fovs: list[str],
    tile_shape: tuple[int, int],
    full_shape: tuple[int, int],
    rows: int | None,
    cols: int | None,
) -> Layout:
    tile_height, tile_width = tile_shape
    full_height, full_width = full_shape

    grid_rows, grid_cols = infer_grid(
        full_height=full_height,
        full_width=full_width,
        tile_height=tile_height,
        tile_width=tile_width,
        n_fovs=len(fovs),
        rows=rows,
        cols=cols,
    )

    used_height = grid_rows * tile_height
    used_width = grid_cols * tile_width
    return Layout(
        fovs=fovs,
        tile_height=tile_height,
        tile_width=tile_width,
        rows=grid_rows,
        cols=grid_cols,
        used_height=used_height,
        used_width=used_width,
        trim_bottom=full_height - used_height,
        trim_right=full_width - used_width,
    )


def sanitize_channel_name(name: str) -> str:
    return name.replace("/", "_")


def write_layout_csv(layout: Layout, path: Path) -> None:
    rows = []
    for idx, fov in enumerate(layout.fovs):
        row_idx = idx // layout.cols
        col_idx = idx % layout.cols
        y0 = row_idx * layout.tile_height
        y1 = y0 + layout.tile_height
        x0 = col_idx * layout.tile_width
        x1 = x0 + layout.tile_width
        rows.append(
            {
                "fov": fov,
                "row": row_idx,
                "col": col_idx,
                "y0": y0,
                "y1": y1,
                "x0": x0,
                "x1": x1,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def iter_fov_tiles(layout: Layout):
    for idx, fov in enumerate(layout.fovs):
        row_idx = idx // layout.cols
        col_idx = idx % layout.cols
        y0 = row_idx * layout.tile_height
        y1 = y0 + layout.tile_height
        x0 = col_idx * layout.tile_width
        x1 = x0 + layout.tile_width
        yield idx, fov, y0, y1, x0, x1


def main() -> None:
    args = parse_args()

    ome_tif = Path(args.ome_tif)
    seg_dir = Path(args.seg_dir)
    output_dir = Path(args.output_dir)

    if not ome_tif.exists():
        raise FileNotFoundError(f"OME-TIFF not found: {ome_tif}")
    if not seg_dir.exists():
        raise FileNotFoundError(f"Segmentation directory not found: {seg_dir}")

    fovs, tile_shape = get_segmentation_fovs(seg_dir, args.seg_suffix)

    with tifffile.TiffFile(ome_tif) as tif:
        series = tif.series[0]
        if series.axes != "CYX":
            raise ValueError(f"Expected OME image axes CYX, got {series.axes}")
        full_shape = tuple(series.shape[1:])
        channel_names = parse_channel_names(tif.ome_metadata or "", expected_n=series.shape[0])

    layout = build_layout(
        fovs=fovs,
        tile_shape=tile_shape,
        full_shape=full_shape,
        rows=args.rows,
        cols=args.cols,
    )

    requested_channels = channel_names if args.channels is None else args.channels
    missing_channels = [ch for ch in requested_channels if ch not in channel_names]
    if missing_channels:
        raise ValueError(
            f"Requested channels not present in OME image: {missing_channels}. "
            f"Available channels: {channel_names}"
        )

    channel_to_index = {name: idx for idx, name in enumerate(channel_names)}
    output_dir.mkdir(parents=True, exist_ok=True)

    layout_csv = Path(args.layout_csv) if args.layout_csv else output_dir / "fov_layout.csv"
    write_layout_csv(layout, layout_csv)

    print(f"OME image: {ome_tif}")
    print(f"Segmentation dir: {seg_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Detected {len(fovs)} FOVs: {fovs}")
    print(
        f"Tile shape: {layout.tile_height} x {layout.tile_width}; "
        f"grid {layout.rows} x {layout.cols}; "
        f"trim bottom={layout.trim_bottom}, right={layout.trim_right}"
    )
    print(f"Channels to export ({len(requested_channels)}): {requested_channels}")
    print(f"Saved layout CSV to: {layout_csv}")

    try:
        ome_stack = tifffile.memmap(ome_tif)
        print("Using memory-mapped OME access.")

        for idx, fov, y0, y1, x0, x1 in iter_fov_tiles(layout):
            fov_dir = output_dir / fov
            fov_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n[{idx + 1}/{len(layout.fovs)}] writing {fov} -> y:{y0}:{y1}, x:{x0}:{x1}")

            for channel_name in requested_channels:
                channel_idx = channel_to_index[channel_name]
                out_name = sanitize_channel_name(channel_name) + ".tiff"
                out_path = fov_dir / out_name

                if out_path.exists() and not args.overwrite:
                    print(f"  skip existing {out_name}")
                    continue

                tile = ome_stack[channel_idx, y0:y1, x0:x1]
                tifffile.imwrite(out_path, tile)
                print(f"  wrote {out_name}")

    except ValueError as exc:
        if "memory-mappable" not in str(exc):
            raise

        print("OME is not memory-mappable; falling back to channel-by-channel reads.")
        with tifffile.TiffFile(ome_tif) as tif:
            for channel_name in requested_channels:
                channel_idx = channel_to_index[channel_name]
                out_name = sanitize_channel_name(channel_name) + ".tiff"
                print(f"\nReading channel {channel_name} (index {channel_idx})")

                try:
                    channel_image = tif.pages[channel_idx].asarray()
                except Exception:
                    channel_image = tif.asarray(key=channel_idx)

                if channel_image.shape != full_shape:
                    raise ValueError(
                        f"Channel {channel_name} produced shape {channel_image.shape}, expected {full_shape}"
                    )

                for idx, fov, y0, y1, x0, x1 in iter_fov_tiles(layout):
                    fov_dir = output_dir / fov
                    fov_dir.mkdir(parents=True, exist_ok=True)
                    out_path = fov_dir / out_name

                    if out_path.exists() and not args.overwrite:
                        if idx == 0:
                            print(f"  existing outputs found for {out_name}; skipping where present")
                        continue

                    tile = channel_image[y0:y1, x0:x1]
                    tifffile.imwrite(out_path, tile)

                del channel_image


if __name__ == "__main__":
    main()
