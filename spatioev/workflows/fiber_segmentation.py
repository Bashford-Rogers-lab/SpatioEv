#!/usr/bin/env python3
"""Background worker: segment ECM fibers in OME-TIFF channels and quantify them.

Segmentation follows ark-analysis (see :mod:`spatioev.pp.fibers`). For each
image and matrix channel the worker writes a fiber label TIFF and an overlay
PNG; across all of them it writes one fiber table, whole-image summaries and
tile summaries.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import traceback
from pathlib import Path
from xml.etree import ElementTree as ET

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spatioev_fiber_matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

from spatioev.pl.fibers import plot_fiber_qc, plot_fiber_segmentation_steps
from spatioev.pp.fibers import (
    calculate_fiber_alignment,
    resolve_clahe_kernel_size,
    segment_fibers,
    segment_fibers_tiled,
)
from spatioev.tl.ecm.fiber_stats import fiber_image_stats, fiber_tile_stats
from spatioev.workflows.image_collection import (
    channel_names,
    image_files,
    imageid_key,
    zarr_array,
)

from ._io import now, write_json

#: Channel-name fragments that usually mark an ECM / matrix stain.
MATRIX_HINTS = ("COL", "FN", "FIBRONECTIN", "ELASTIN", "LAMININ", "SHG", "TNC", "TENASCIN", "PERIOSTIN", "POSTN", "HSPG", "VERSICAN")

DEFAULT_PARAMETERS = {
    "blur": 2.0,
    "contrast_scaling_divisor": 128.0,
    "clahe_kernel_size": None,
    "fiber_widths": [1, 3, 5, 7, 9],
    "ridge_cutoff": 0.1,
    "sobel_blur": 1.0,
    "min_fiber_size": 15,
    "include_bright": False,
    "bright_local_window_um": 65.0,
    "bright_floor": 0.25,
    "tile_size": 4096,
    "overlap": 128,
    "alignment_k": 4,
    "axis_thresh": 2.0,
    "axial": True,
    "stats_tile_size_px": 512,
    "min_fiber_num": 5,
    # TWOMBLI-style architecture (spatioev.tl.ecm.architecture); lengths in µm.
    "architecture": True,
    "min_branch_length_um": 3.0,
    "curvature_window_um": 10.0,
    "lacunarity_box_um": 10.0,
    "max_display_hdm": 200.0,
    "hdm_saturation_pct": 0.35,
    "fd_coarse_factor": None,
}


def bright_window_px(params: dict, pixel_size_um: float | None) -> int:
    """Local window of the bright-matrix test in pixels (0 = brightest class only)."""
    window_um = float(params.get("bright_local_window_um") or 0)
    return int(round(window_um / float(pixel_size_um or 1.0))) if window_um > 0 else 0


def architecture_settings(params: dict, pixel_size_um: float | None) -> dict:
    """Convert the µm architecture settings to pixels for one pixel size."""
    scale = float(pixel_size_um or 1.0)
    return {
        "pixel_size_um": scale,
        "min_branch_length": float(params["min_branch_length_um"]) / scale,
        "curvature_window": max(1, int(round(float(params["curvature_window_um"]) / scale))),
        "lacunarity_box": max(2, int(round(float(params["lacunarity_box_um"]) / scale))),
        "coarse_factor": params.get("fd_coarse_factor"),
    }


def tile_grid_groups(tile_px: int, shape: tuple[int, int]):
    """Group reader and names that assign every pixel to its square tile."""
    n_cols = int(np.ceil(shape[1] / tile_px))
    n_rows = int(np.ceil(shape[0] / tile_px))

    def reader(y0, y1, x0, x1):
        rows = np.arange(y0, y1)[:, None] // tile_px
        cols = np.arange(x0, x1)[None, :] // tile_px
        return rows * n_cols + cols + 1

    names = {r * n_cols + c + 1: f"{r * tile_px},{c * tile_px}" for r in range(n_rows) for c in range(n_cols)}
    return reader, names


def measure_architecture(mask_reader, intensity_reader, shape, threshold, params, pixel_size_um, group_reader, group_names, all_name, progress=None):
    """Run :func:`matrix_architecture_tiled` with the run's settings."""
    from spatioev.tl.ecm.architecture import matrix_architecture_tiled

    settings = architecture_settings(params, pixel_size_um)
    return matrix_architecture_tiled(
        mask_reader, shape, group_reader, group_names, intensity_reader, threshold,
        pixel_size_um=settings["pixel_size_um"],
        min_branch_length=settings["min_branch_length"],
        curvature_window=settings["curvature_window"],
        lacunarity_box=settings["lacunarity_box"],
        tile_size=int(params["tile_size"]),
        all_name=all_name,
        coarse_factor=settings["coarse_factor"],
        progress=progress,
    )


def update_status(path: Path, state: str, message: str, *, stage: str, progress: float, **extra) -> None:
    write_json(path, {"state": state, "message": message, "stage": stage, "progress": progress, "updated_at": now(), **extra})


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "image"


class ChannelPlane:
    """Lazy 2-D view of one channel of an open OME-TIFF (slices read on demand)."""

    def __init__(self, array, channel_index: int):
        self._array = array
        self._channel = channel_index
        self.shape = tuple(int(s) for s in array.shape[-2:])
        self.ndim = 2

    def __getitem__(self, key):
        return np.asarray(self._array[(self._channel, *key)] if isinstance(key, tuple) else self._array[self._channel, key])


class OMEImage:
    """Open an OME-TIFF and expose channels by name, lazily."""

    def __init__(self, path: Path):
        import zarr

        self.path = Path(path)
        self.tif = tifffile.TiffFile(self.path)
        series = self.tif.series[0]
        if not series.axes.endswith("YX") or "C" not in series.axes:
            raise ValueError(f"{self.path.name}: expected CYX axes, found {series.axes!r}")
        self.channels = channel_names(self.path)
        self._store = series.levels[0].aszarr()
        self.array = zarr_array(zarr.open(self._store, mode="r"))
        self.shape = tuple(int(s) for s in series.shape[-2:])

    def plane(self, channel: str) -> ChannelPlane:
        if channel not in self.channels:
            raise KeyError(f"{self.path.name}: channel {channel!r} not found; available: {self.channels}")
        return ChannelPlane(self.array, self.channels.index(channel))

    def close(self) -> None:
        close = getattr(self._store, "close", None)
        if close is not None:
            close()
        self.tif.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def physical_pixel_size(path: Path) -> float | None:
    """``PhysicalSizeX`` in micrometres from OME metadata, if recorded."""
    with tifffile.TiffFile(path) as tif:
        if not tif.ome_metadata:
            return None
        root = ET.fromstring(tif.ome_metadata)
    for node in root.iter():
        if node.tag.rsplit("}", 1)[-1] == "Pixels" and "PhysicalSizeX" in node.attrib:
            value = float(node.attrib["PhysicalSizeX"])
            unit = node.attrib.get("PhysicalSizeXUnit", "µm")
            return value / 1000.0 if unit == "nm" else value * 1000.0 if unit == "mm" else value
    return None


def resolve_images(image_path: str, imageids: list[str] | None = None, single_imageid: str | None = None) -> list[dict]:
    """Return ``[{imageid, path}]`` for a single OME-TIFF or a folder of FOVs.

    For a folder, images are matched to ``imageids`` (taken from the H5AD) the
    same way the phenotyping pages match them; without ``imageids`` every file
    is used and named by its stem.
    """
    source = Path(image_path).expanduser().resolve()
    if source.is_file():
        return [{"imageid": single_imageid or source.name.split(".")[0], "path": str(source)}]
    if not source.is_dir():
        raise FileNotFoundError(
            f"Image not found or not readable: {source}. Check the path; on macOS, a folder such as "
            "Downloads may also need Files and Folders permission for the app running SpatioEv."
        )
    files = image_files(source)
    if not files:
        raise FileNotFoundError(f"No OME-TIFF images found under {source}")
    if not imageids:
        return [{"imageid": path.name.split(".")[0], "path": str(path)} for path in files]
    by_key = {imageid_key(path.stem): path for path in files}
    rows, missing = [], []
    for imageid in imageids:
        path = by_key.get(imageid_key(imageid))
        (rows.append({"imageid": str(imageid), "path": str(path)}) if path else missing.append(str(imageid)))
    if missing:
        raise ValueError(f"No OME-TIFF found for image IDs: {missing[:10]}")
    return rows


def adata_imageids(adata_path: str | None) -> list[str] | None:
    if not adata_path or not Path(adata_path).expanduser().exists():
        return None
    import anndata as ad

    adata = ad.read_h5ad(Path(adata_path).expanduser(), backed="r")
    try:
        return [str(value) for value in pd.unique(adata.obs["imageid"].astype(str))]
    finally:
        adata.file.close()


def inspect_inputs(config: dict) -> dict:
    """Describe the images the worker would process (used by the UI)."""
    imageids = adata_imageids(config.get("adata_path"))
    single = imageids[0] if imageids and len(imageids) == 1 else config.get("sample_id")
    images = resolve_images(config["image_path"], imageids, single)
    with OMEImage(Path(images[0]["path"])) as first:
        channels, shape = first.channels, first.shape
    return {
        "images": images,
        "n_images": len(images),
        "channels": channels,
        "matrix_channels": [c for c in channels if any(hint in c.upper() for hint in MATRIX_HINTS)],
        "first_shape": shape,
        "pixel_size_um": physical_pixel_size(Path(images[0]["path"])),
        "imageids_from_adata": imageids is not None,
    }


def preview_steps(config: dict, imageid: str, channel: str, centre: tuple[int, int] | None = None, size: int = 768) -> dict:
    """Run every segmentation step on a crop, for parameter tuning.

    The CLAHE window is resolved from the *full* image so the crop is
    processed with the window a real run would use.
    """
    params = {**DEFAULT_PARAMETERS, **config.get("parameters", {})}
    images = {row["imageid"]: row["path"] for row in resolve_images(config["image_path"], [imageid] if Path(config["image_path"]).is_dir() else None, imageid)}
    with OMEImage(Path(images[imageid])) as image:
        rows, cols = image.shape
        kernel = resolve_clahe_kernel_size(image.shape, params["contrast_scaling_divisor"], params["clahe_kernel_size"])
        cy, cx = centre if centre is not None else (rows // 2, cols // 2)
        half = size // 2
        y0, x0 = max(0, min(cy - half, rows - size)), max(0, min(cx - half, cols - size))
        crop = image.plane(channel)[y0 : y0 + size, x0 : x0 + size]
    labels, steps = segment_fibers(
        crop,
        blur=params["blur"],
        fiber_widths=params["fiber_widths"],
        ridge_cutoff=params["ridge_cutoff"],
        sobel_blur=params["sobel_blur"],
        min_fiber_size=int(params["min_fiber_size"]),
        clahe_kernel_size=kernel,
        include_bright=bool(params.get("include_bright", False)),
        bright_local_window=bright_window_px(params, config.get("pixel_size_um") or physical_pixel_size(Path(images[imageid]))),
        bright_floor=float(params.get("bright_floor", 0.25)),
        return_steps=True,
    )
    n_ridge = int(steps.get("n_ridge_objects", labels.max()))
    return {
        "steps": steps,
        "n_fibers": n_ridge,
        "n_bright_matrix": int(labels.max()) - n_ridge,
        "origin": (y0, x0),
        "kernel": kernel,
        "area_fraction_pct": float((labels > 0).mean() * 100),
    }


def run_workflow(config: dict, status_path: Path) -> dict:
    params = {**DEFAULT_PARAMETERS, **config.get("parameters", {})}
    sample_id = config["sample_id"].strip()
    output_dir = Path(config["output_dir"]).expanduser().resolve()
    labels_dir = output_dir / "labels"
    qc_dir = output_dir / "qc"
    labels_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    channels = [str(c) for c in config["channels"]]
    if not channels:
        raise ValueError("Select at least one matrix channel")

    update_status(status_path, "running", "Resolving images", stage="inspect", progress=0.02)
    imageids = adata_imageids(config.get("adata_path"))
    single = imageids[0] if imageids and len(imageids) == 1 else sample_id
    images = resolve_images(config["image_path"], imageids, single)
    pixel_size = config.get("pixel_size_um") or physical_pixel_size(Path(images[0]["path"]))

    tables, shapes, label_files, overlay_files, architecture_rows = [], {}, [], [], []
    jobs = [(row, channel) for row in images for channel in channels]
    for job_index, (row, channel) in enumerate(jobs):
        imageid = row["imageid"]
        base = 0.05 + 0.80 * job_index / len(jobs)
        span = 0.80 / len(jobs)

        def report(fraction: float, message: str, base=base, span=span, imageid=imageid, channel=channel) -> None:
            update_status(status_path, "running", f"{imageid} / {channel}: {message}", stage="segment", progress=round(base + span * fraction, 4))

        with OMEImage(Path(row["path"])) as image:
            shapes[imageid] = image.shape
            plane = image.plane(channel)
            kernel = resolve_clahe_kernel_size(image.shape, params["contrast_scaling_divisor"], params["clahe_kernel_size"])
            label_path = labels_dir / f"{safe_name(imageid)}__{safe_name(channel)}_fiber_labels.tiff"
            labels, table = segment_fibers_tiled(
                plane,
                blur=params["blur"],
                fiber_widths=params["fiber_widths"],
                ridge_cutoff=params["ridge_cutoff"],
                sobel_blur=params["sobel_blur"],
                min_fiber_size=int(params["min_fiber_size"]),
                clahe_kernel_size=kernel,
                include_bright=bool(params.get("include_bright", False)),
                bright_local_window=bright_window_px(params, pixel_size),
                bright_floor=float(params.get("bright_floor", 0.25)),
                tile_size=int(params["tile_size"]),
                overlap=int(params["overlap"]),
                imageid=imageid,
                fiber_type=channel,
                work_dir=output_dir,
                progress=report,
            )
            dtype = np.uint16 if labels.max() < 65535 else np.uint32
            temporary = label_path.with_suffix(".tmp.tiff")
            tifffile.imwrite(temporary, np.asarray(labels).astype(dtype), compression="zlib", tile=(512, 512), bigtiff=labels.size * np.dtype(dtype).itemsize > 2**31)
            temporary.replace(label_path)
            label_row = {"imageid": imageid, "fiber_type": channel, "path": str(label_path), "kernel": kernel, "image_path": row["path"]}

            if params.get("architecture", True):
                from spatioev.tl.ecm.architecture import hdm_threshold

                report(0.97, "TWOMBLI-style architecture")
                threshold = hdm_threshold(plane, float(params["max_display_hdm"]), float(params["hdm_saturation_pct"]))
                label_row["hdm_threshold"] = threshold
                reader, names = tile_grid_groups(int(params["stats_tile_size_px"]), image.shape)
                measured = measure_architecture(
                    labels, plane, image.shape, threshold, params, pixel_size, reader, names, "whole_image"
                )
                measured.insert(0, "fiber_type", channel)
                measured.insert(0, "imageid", imageid)
                architecture_rows.append(measured)
            label_files.append(label_row)

            bright_labels = table.loc[table["object_type"] == "bright_matrix", "label"].to_numpy() if "object_type" in table else None
            fig = plot_fiber_qc(
                plane, labels, bright_labels,
                title=f"{imageid} {channel}: {int((table.get('object_type', pd.Series()) != 'bright_matrix').sum()):,} ridge fibers"
                + (f" + {len(bright_labels):,} bright-matrix objects" if bright_labels is not None and len(bright_labels) else ""),
            )
            overlay_path = qc_dir / f"{safe_name(imageid)}__{safe_name(channel)}_fiber_overlay.png"
            fig.savefig(overlay_path, dpi=100)
            plt.close(fig)
            overlay_files.append(str(overlay_path))
        tables.append(table)

    update_status(status_path, "running", "Scoring fiber alignment", stage="alignment", progress=0.87)
    fiber_table = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    fiber_table = calculate_fiber_alignment(fiber_table, k=int(params["alignment_k"]), axis_thresh=float(params["axis_thresh"]), axial=bool(params["axial"]))
    if pixel_size:
        fiber_table["major_axis_length_um"] = fiber_table["major_axis_length"] * pixel_size
        fiber_table["minor_axis_length_um"] = fiber_table["minor_axis_length"] * pixel_size
        fiber_table["area_um2"] = fiber_table["area"] * pixel_size**2

    update_status(status_path, "running", "Summarising fibers per image and tile", stage="stats", progress=0.92)
    image_stats = fiber_image_stats(fiber_table, shapes, pixel_size_um=pixel_size)
    tile_stats = fiber_tile_stats(
        fiber_table, shapes, tile_size=int(params["stats_tile_size_px"]), min_fiber_num=int(params["min_fiber_num"]), pixel_size_um=pixel_size
    )
    architecture_path = None
    if architecture_rows:
        architecture = pd.concat(architecture_rows, ignore_index=True)
        whole = architecture[architecture["group"] == "whole_image"].drop(columns="group")
        tiles = architecture[architecture["group"] != "whole_image"].copy()
        tiles[["tile_y", "tile_x"]] = tiles["group"].str.split(",", expand=True).astype(int)
        tiles = tiles.drop(columns="group")
        tile_stats = tile_stats.merge(tiles, on=["imageid", "fiber_type", "tile_y", "tile_x"], how="left")
        architecture_path = output_dir / f"{sample_id}_matrix_architecture.csv"
        whole.to_csv(architecture_path, index=False)

    table_path = output_dir / f"{sample_id}_fiber_table.csv"
    image_stats_path = output_dir / f"{sample_id}_fiber_image_stats.csv"
    tile_stats_path = output_dir / f"{sample_id}_fiber_tile_stats.csv"
    for frame, path in ((fiber_table, table_path), (image_stats, image_stats_path), (tile_stats, tile_stats_path)):
        temporary = path.with_suffix(".tmp.csv")
        frame.to_csv(temporary, index=False)
        temporary.replace(path)

    n_ridge = int((fiber_table["object_type"] != "bright_matrix").sum()) if "object_type" in fiber_table else int(len(fiber_table))
    outputs = {
        "sample_id": sample_id,
        "fiber_table": str(table_path),
        "image_stats": str(image_stats_path),
        "tile_stats": str(tile_stats_path),
        "architecture": str(architecture_path) if architecture_path else None,
        "stats_tile_size_px": int(params["stats_tile_size_px"]),
        "labels": label_files,
        "overlays": overlay_files,
        "image_shapes": {key: list(value) for key, value in shapes.items()},
        "pixel_size_um": pixel_size,
        "channels": channels,
        "n_fibers": n_ridge,
        "n_bright_matrix": int(len(fiber_table)) - n_ridge,
    }
    manifest_path = output_dir / f"{sample_id}_fiber_manifest.json"
    write_json(manifest_path, {"created_at": now(), "config": config, "parameters": params, "outputs": outputs})
    outputs["manifest"] = str(manifest_path)
    update_status(status_path, "complete", f"Segmented {n_ridge:,} fibers" + (f" and {len(fiber_table) - n_ridge:,} bright-matrix objects" if len(fiber_table) > n_ridge else ""), stage="complete", progress=1.0, outputs=outputs)
    return outputs


def label_image_path(label_row: dict, manifest: dict) -> Path | None:
    """Source OME-TIFF of a label image (older manifests did not record it)."""
    if label_row.get("image_path"):
        return Path(label_row["image_path"])
    config = manifest.get("config", {})
    try:
        source = Path(config["image_path"]).expanduser()
        imageid = str(label_row["imageid"])
        return Path(resolve_images(str(source), [imageid] if source.is_dir() else None, imageid)[0]["path"])
    except Exception:
        return None


def raster_group_reader(raster: np.ndarray, downsample: int):
    """Group reader returning a downsampled region raster at full resolution."""
    def reader(y0, y1, x0, x1):
        rows = np.minimum(np.arange(y0, y1) // downsample, raster.shape[0] - 1)
        cols = np.minimum(np.arange(x0, x1) // downsample, raster.shape[1] - 1)
        return raster[rows[:, None], cols[None, :]]
    return reader


def open_labels(path: str | Path):
    """Open a fiber label TIFF lazily; slices are read on demand."""
    import zarr

    store = tifffile.imread(Path(path), aszarr=True)
    return zarr_array(zarr.open(store, mode="r"))


def read_fiber_table(path: str | Path) -> pd.DataFrame:
    """Load a fiber table written by this worker, indexed by ``fiber_id``."""
    table = pd.read_csv(path)
    table["imageid"] = table["imageid"].astype(str)
    return table.set_index("fiber_id", drop=False)


def save_preview_figure(preview: dict, path: Path, title: str) -> Path:
    fig = plot_fiber_segmentation_steps(preview["steps"], title=title)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    try:
        if args.inspect_only:
            print(json.dumps(inspect_inputs(config), indent=2, default=str))
            return
        run_workflow(config, args.status)
    except Exception as error:
        update_status(args.status, "failed", str(error), stage="failed", progress=1.0, traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
