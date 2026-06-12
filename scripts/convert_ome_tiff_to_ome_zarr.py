#!/usr/bin/env python3
"""Convert a pyramidal OME-TIFF to compressed OME-Zarr without loading it all.

The converter preserves the full stitched coordinate frame and writes Zarr-v2 /
OME-Zarr-0.4 for broad compatibility with napari/spatialdata-era tooling.
Empty all-zero chunks are not stored, which is important for sparse slides with
large blank glass regions.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import dask
import dask.array as da
import numpy as np
import tifffile
import zarr
from dask.diagnostics import ProgressBar
from numcodecs import Blosc
from ome_zarr import format, writer


OME_NS = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a pyramidal OME-TIFF to compressed OME-Zarr."
    )
    parser.add_argument("--source", required=True, type=Path, help="Input OME-TIFF.")
    parser.add_argument("--output", required=True, type=Path, help="Output .ome.zarr.")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of dask worker threads. Default: 4.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=5,
        help="ZSTD compression level for Blosc. Default: 5.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing output directory before writing.",
    )
    return parser.parse_args()


def read_ome_metadata(tif: tifffile.TiffFile) -> dict[str, object]:
    ome = tif.ome_metadata or ""
    if not ome:
        raise ValueError("Source TIFF does not contain OME metadata.")

    root = ET.fromstring(ome)
    pixels = root.find(".//ome:Pixels", OME_NS)
    if pixels is None:
        raise ValueError("OME metadata does not contain a Pixels element.")

    channel_names: list[str] = []
    for index, channel in enumerate(root.findall(".//ome:Channel", OME_NS)):
        channel_names.append(channel.attrib.get("Name") or f"channel_{index}")

    return {
        "channel_names": channel_names,
        "pixel_size_x": float(pixels.attrib.get("PhysicalSizeX", "1.0")),
        "pixel_size_y": float(pixels.attrib.get("PhysicalSizeY", "1.0")),
        "pixel_size_x_unit": pixels.attrib.get("PhysicalSizeXUnit", "pixel"),
        "pixel_size_y_unit": pixels.attrib.get("PhysicalSizeYUnit", "pixel"),
    }


def normalize_unit(unit: object) -> str:
    text = str(unit)
    if text in {"µm", "um", "micron", "micrometer", "micrometre"}:
        return "micrometer"
    return text


def omero_metadata(channel_names: list[str]) -> dict[str, object]:
    channels = []
    for name in channel_names:
        channels.append(
            {
                "active": True,
                "coefficient": 1,
                "color": "FFFFFF",
                "family": "linear",
                "inverted": False,
                "label": name,
                "window": {"start": 0, "end": 65535, "min": 0, "max": 65535},
            }
        )
    return {
        "channels": channels,
        "rdefs": {"defaultT": 0, "defaultZ": 0, "model": "color"},
    }


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{path} already exists. Pass --overwrite to replace it."
            )
        shutil.rmtree(path, ignore_errors=True)
        if path.exists():
            raise OSError(f"Could not fully remove existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.exists():
        raise FileNotFoundError(source)

    prepare_output(output, args.overwrite)

    compressor = Blosc(
        cname="zstd",
        clevel=args.compression_level,
        shuffle=Blosc.BITSHUFFLE,
    )
    group = zarr.open_group(str(output), mode="w", zarr_format=2)

    with tifffile.TiffFile(source) as tif:
        if not tif.is_ome:
            raise ValueError("Input is not an OME-TIFF.")

        series = tif.series[0]
        metadata = read_ome_metadata(tif)
        channel_names = metadata["channel_names"]
        pixel_size_y = float(metadata["pixel_size_y"])
        pixel_size_x = float(metadata["pixel_size_x"])
        unit_y = normalize_unit(metadata["pixel_size_y_unit"])
        unit_x = normalize_unit(metadata["pixel_size_x_unit"])

        base_shape = series.levels[0].shape
        if len(base_shape) != 3:
            raise ValueError(f"Expected CYX data, found shape {base_shape}.")

        datasets: list[dict[str, object]] = []
        level_shapes: list[tuple[int, ...]] = []
        level_chunks: list[tuple[int, ...]] = []

        print(
            json.dumps(
                {
                    "event": "start",
                    "source": str(source),
                    "output": str(output),
                    "base_shape": base_shape,
                    "levels": len(series.levels),
                    "workers": args.workers,
                    "compression": {
                        "codec": "blosc",
                        "cname": "zstd",
                        "clevel": args.compression_level,
                        "shuffle": "bitshuffle",
                    },
                }
            ),
            flush=True,
        )

        for level_index, level in enumerate(series.levels):
            store = series.aszarr(level=level_index)
            try:
                source_array = zarr.open(store, mode="r")
                dask_array = da.from_zarr(source_array, chunks=source_array.chunks)
                target = group.create_array(
                    str(level_index),
                    shape=dask_array.shape,
                    dtype=dask_array.dtype,
                    chunks=dask_array.chunksize,
                    compressor=compressor,
                    fill_value=0,
                    config={"write_empty_chunks": False},
                )

                downscale_y = base_shape[1] / dask_array.shape[1]
                downscale_x = base_shape[2] / dask_array.shape[2]
                datasets.append(
                    {
                        "path": str(level_index),
                        "coordinateTransformations": [
                            {
                                "type": "scale",
                                "scale": [
                                    1,
                                    pixel_size_y * downscale_y,
                                    pixel_size_x * downscale_x,
                                ],
                            }
                        ],
                    }
                )
                level_shapes.append(tuple(int(v) for v in dask_array.shape))
                level_chunks.append(tuple(int(v) for v in dask_array.chunksize))

                print(
                    json.dumps(
                        {
                            "event": "write_level",
                            "level": level_index,
                            "shape": dask_array.shape,
                            "chunks": dask_array.chunksize,
                        }
                    ),
                    flush=True,
                )
                with dask.config.set(scheduler="threads", num_workers=args.workers):
                    with ProgressBar():
                        da.store(dask_array, target, lock=False, compute=True)
            finally:
                close = getattr(store, "close", None)
                if close is not None:
                    close()

        axes = [
            {"name": "c", "type": "channel"},
            {"name": "y", "type": "space", "unit": unit_y},
            {"name": "x", "type": "space", "unit": unit_x},
        ]
        writer.write_multiscales_metadata(
            group,
            datasets=datasets,
            fmt=format.FormatV04(),
            axes=axes,
            name=source.stem,
        )
        group.attrs["omero"] = omero_metadata(list(channel_names))
        group.attrs["conversion"] = {
            "source": str(source),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "tool": "scripts/convert_ome_tiff_to_ome_zarr.py",
            "preserves_full_canvas": True,
            "zarr_format": 2,
            "ome_zarr_version": "0.4",
            "level_shapes": level_shapes,
            "level_chunks": level_chunks,
            "compression": {
                "codec": "blosc",
                "cname": "zstd",
                "clevel": args.compression_level,
                "shuffle": "bitshuffle",
                "write_empty_chunks": False,
            },
            "pixel_size": {
                "x": pixel_size_x,
                "y": pixel_size_y,
                "x_unit": unit_x,
                "y_unit": unit_y,
            },
        }

    zarr.consolidate_metadata(str(output))
    total_bytes = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    print(
        json.dumps(
            {
                "event": "done",
                "output": str(output),
                "size_gib": total_bytes / 1024**3,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
