import os
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import find_objects
from scipy.stats import pearsonr
from skimage.draw import polygon as draw_polygon
from skimage.exposure import rescale_intensity
from skimage import img_as_ubyte
from skimage.feature import graycomatrix, graycoprops
from skimage.io import imread
from skimage.measure import shannon_entropy
from skimage.util import view_as_windows


def _require_columns(df, required, name="cell table"):
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {name}: {missing}")


def calculate_polarity_score(geometric_centroid, intensity_centroid, cell_size):
    return np.linalg.norm(np.asarray(geometric_centroid) - np.asarray(intensity_centroid)) / cell_size


def calculate_moment_of_inertia(masked_intensity, cx, cy):
    y_coords, x_coords = np.indices(masked_intensity.shape)
    return np.sum(((x_coords - cx) ** 2 + (y_coords - cy) ** 2) * masked_intensity)


def calculate_haralick_features(masked_image):
    masked_image = img_as_ubyte(masked_image)
    glcm = graycomatrix(
        masked_image,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True,
    )
    return {
        "haralick_contrast": graycoprops(glcm, "contrast")[0, 0],
        "haralick_correlation": graycoprops(glcm, "correlation")[0, 0],
        "haralick_energy": graycoprops(glcm, "energy")[0, 0],
        "haralick_homogeneity": graycoprops(glcm, "homogeneity")[0, 0],
    }


def calculate_entropy(masked_image):
    return shannon_entropy(masked_image)


def calculate_lacunarity(masked_image, box_size=5):
    if masked_image.shape[0] < box_size or masked_image.shape[1] < box_size:
        return np.nan

    windows = view_as_windows(masked_image, (box_size, box_size))
    local_sums = windows.sum(axis=(2, 3))
    mean = np.mean(local_sums)
    std = np.std(local_sums)

    return (std / mean) ** 2 if mean != 0 else np.nan


def calculate_channel_correlation(mask, img1, img2):
    masked1 = img1[mask]
    masked2 = img2[mask]

    if np.std(masked1) == 0 or np.std(masked2) == 0:
        return np.nan, np.nan

    pcc, _ = pearsonr(masked1, masked2)
    mean1 = np.mean(masked1)
    mean2 = np.mean(masked2)

    return pcc, (mean2 / mean1) if mean1 != 0 else np.nan


def _rescale_to_uint8(image):
    image = np.asarray(image)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = finite.min(), finite.max()
    if hi <= lo:
        return np.zeros(image.shape, dtype=np.uint8)
    return rescale_intensity(image, in_range=(lo, hi), out_range=(0, 255)).astype(np.uint8)


def calculate_haralick_features_rescaled(masked_image):
    masked_image = _rescale_to_uint8(masked_image)
    glcm = graycomatrix(
        masked_image,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True,
    )
    return {
        "haralick_contrast": graycoprops(glcm, "contrast")[0, 0],
        "haralick_correlation": graycoprops(glcm, "correlation")[0, 0],
        "haralick_energy": graycoprops(glcm, "energy")[0, 0],
        "haralick_homogeneity": graycoprops(glcm, "homogeneity")[0, 0],
    }


def _build_label_slices(seg_labels, padding=2):
    raw_slices = find_objects(seg_labels)
    label_slices = {}
    height, width = seg_labels.shape

    for label, slc in enumerate(raw_slices, start=1):
        if slc is None:
            continue

        y_slice, x_slice = slc
        y_min = max(y_slice.start - padding, 0)
        y_max = min(y_slice.stop + padding, height)
        x_min = max(x_slice.start - padding, 0)
        x_max = min(x_slice.stop + padding, width)
        label_slices[label] = (slice(y_min, y_max), slice(x_min, x_max))

    return label_slices


def _process_single_cell(
    row,
    label_slices,
    seg_labels,
    polarity_img,
    texture_img,
    corr_img1,
    corr_img2,
):
    label, cell_size, centroid_x, centroid_y = row
    label = int(label)
    geometric_centroid = (centroid_x, centroid_y)

    bbox = label_slices.get(label)
    if bbox is None:
        return None

    y_slice, x_slice = bbox
    cropped_labels = seg_labels[y_slice, x_slice]
    cropped_mask = cropped_labels == label
    if not np.any(cropped_mask):
        return None

    cropped_polarity = polarity_img[y_slice, x_slice]
    cropped_texture = texture_img[y_slice, x_slice]
    cropped_corr1 = corr_img1[y_slice, x_slice]
    cropped_corr2 = corr_img2[y_slice, x_slice]

    masked_intensity = cropped_polarity * cropped_mask
    total_intensity = np.sum(masked_intensity)
    if total_intensity == 0:
        return None

    y_idx, x_idx = np.indices(cropped_mask.shape)
    cx = np.sum(x_idx * masked_intensity) / total_intensity
    cy = np.sum(y_idx * masked_intensity) / total_intensity
    intensity_centroid = (cx + x_slice.start, cy + y_slice.start)

    masked_texture = cropped_texture * cropped_mask

    pcc, intensity_ratio = calculate_channel_correlation(cropped_mask, cropped_corr1, cropped_corr2)
    out = {
        "label": label,
        "geometric_centroid_x": geometric_centroid[0],
        "geometric_centroid_y": geometric_centroid[1],
        "intensity_centroid_x": intensity_centroid[0],
        "intensity_centroid_y": intensity_centroid[1],
        "polarity_score": calculate_polarity_score(geometric_centroid, intensity_centroid, cell_size),
        "entropy": calculate_entropy(masked_texture),
        "channel_pcc": pcc,
        "intensity_ratio": intensity_ratio,
        "inertia": calculate_moment_of_inertia(masked_intensity, cx, cy),
        "lacunarity": calculate_lacunarity(masked_texture),
    }
    out.update(calculate_haralick_features(masked_texture))
    return out


def extract_cell_pixel_features_for_fov(
    fov,
    cell_table,
    seg_dir,
    img_dir,
    polarity_channel,
    texture_channel,
    corr_channels,
    label_key="label",
    fov_key="fov",
    size_key="cell_size",
    centroid_x_key="centroid-1",
    centroid_y_key="centroid-0",
    seg_suffix="_whole_cell.tiff",
    image_suffix=".tiff",
    padding=2,
    n_workers=12,
    pcc_key=None,
):
    """
    Extract per-cell pixel/morphotexture features for one FOV.
    """
    _require_columns(
        cell_table,
        [label_key, fov_key, size_key, centroid_x_key, centroid_y_key],
    )

    if len(corr_channels) != 2:
        raise ValueError("corr_channels must contain exactly two channel names.")

    seg_path = os.path.join(seg_dir, f"{fov}{seg_suffix}")
    if not os.path.exists(seg_path):
        return pd.DataFrame()

    fov_dir = os.path.join(img_dir, str(fov))
    polarity_path = os.path.join(fov_dir, f"{polarity_channel}{image_suffix}")
    texture_path = os.path.join(fov_dir, f"{texture_channel}{image_suffix}")
    corr1_path = os.path.join(fov_dir, f"{corr_channels[0]}{image_suffix}")
    corr2_path = os.path.join(fov_dir, f"{corr_channels[1]}{image_suffix}")

    required_paths = [polarity_path, texture_path, corr1_path, corr2_path]
    missing = [p for p in required_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing required image files for FOV {fov}: {missing}")

    seg_labels = imread(seg_path).astype(int)
    label_slices = _build_label_slices(seg_labels, padding=padding)
    polarity_img = imread(polarity_path)
    texture_img = imread(texture_path)
    corr_img1 = imread(corr1_path)
    corr_img2 = imread(corr2_path)

    fov_data = cell_table[cell_table[fov_key] == fov].copy()
    if fov_data.empty:
        return pd.DataFrame()

    rows = list(
        fov_data[[label_key, size_key, centroid_x_key, centroid_y_key]].itertuples(
            index=False,
            name=None,
        )
    )
    worker = lambda row: _process_single_cell(
        row=row,
        label_slices=label_slices,
        seg_labels=seg_labels,
        polarity_img=polarity_img,
        texture_img=texture_img,
        corr_img1=corr_img1,
        corr_img2=corr_img2,
    )

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        features = list(executor.map(worker, rows))

    results = []
    pcc_key = pcc_key or f"pcc_{corr_channels[0].lower()}_{corr_channels[1].lower()}"
    for feature_dict in features:
        if feature_dict is None:
            continue
        feature_dict["fov"] = fov
        feature_dict[pcc_key] = feature_dict.pop("channel_pcc")
        results.append(feature_dict)

    return pd.DataFrame(results)


def extract_cell_pixel_features(
    cell_table,
    seg_dir,
    img_dir,
    polarity_channel,
    texture_channel,
    corr_channels,
    fovs=None,
    label_key="label",
    fov_key="fov",
    size_key="cell_size",
    centroid_x_key="centroid-1",
    centroid_y_key="centroid-0",
    seg_suffix="_whole_cell.tiff",
    image_suffix=".tiff",
    padding=2,
    n_workers=12,
    output_path=None,
    per_fov_output_dir=None,
):
    """
    Extract per-cell pixel features across multiple FOVs.
    """
    if isinstance(cell_table, str):
        cell_table = pd.read_csv(cell_table)
    else:
        cell_table = cell_table.copy()

    _require_columns(
        cell_table,
        [label_key, fov_key, size_key, centroid_x_key, centroid_y_key],
    )

    if fovs is None:
        fovs = cell_table[fov_key].dropna().unique().tolist()

    if per_fov_output_dir is not None:
        os.makedirs(per_fov_output_dir, exist_ok=True)

    all_results = []
    for fov in fovs:
        fov_df = extract_cell_pixel_features_for_fov(
            fov=fov,
            cell_table=cell_table,
            seg_dir=seg_dir,
            img_dir=img_dir,
            polarity_channel=polarity_channel,
            texture_channel=texture_channel,
            corr_channels=corr_channels,
            label_key=label_key,
            fov_key=fov_key,
            size_key=size_key,
            centroid_x_key=centroid_x_key,
            centroid_y_key=centroid_y_key,
            seg_suffix=seg_suffix,
            image_suffix=image_suffix,
            padding=padding,
            n_workers=n_workers,
        )
        if not fov_df.empty:
            all_results.append(fov_df)
            if per_fov_output_dir is not None:
                fov_df.to_csv(
                    os.path.join(per_fov_output_dir, f"{fov}_pixel_features.csv"),
                    index=False,
                )

    if all_results:
        result = pd.concat(all_results, ignore_index=True)
    else:
        result = pd.DataFrame()

    if output_path is not None:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        result.to_csv(output_path, index=False)

    return result


class _XeniumMorphologyReader:
    """Small wrapper that keeps the TIFF and zarr store alive while reading crops."""

    def __init__(self, image_path, channel_index=0):
        import tifffile
        import zarr

        self.image_path = Path(image_path)
        self.channel_index = int(channel_index)
        self.tiff = tifffile.TiffFile(self.image_path)
        self.series = self.tiff.series[0]
        self.axes = self.series.axes
        self.store = self.tiff.aszarr(series=0)
        opened = zarr.open(self.store, mode="r")
        self.array = opened["0"] if hasattr(opened, "keys") else opened

    @property
    def shape(self):
        return self.array.shape

    def close(self):
        try:
            self.store.close()
        except Exception:
            pass
        self.tiff.close()

    def read_crop(self, y_min, y_max, x_min, x_max, z_projection="max"):
        axes = self.axes
        arr = self.array

        if arr.ndim == 2:
            return np.asarray(arr[y_min:y_max, x_min:x_max])

        slicer = []
        for axis in axes:
            if axis == "Y":
                slicer.append(slice(y_min, y_max))
            elif axis == "X":
                slicer.append(slice(x_min, x_max))
            elif axis == "Z":
                slicer.append(slice(None))
            elif axis == "C":
                slicer.append(self.channel_index)
            else:
                slicer.append(0)

        crop = np.asarray(arr[tuple(slicer)])
        if "Z" in axes:
            z_axis_original = axes.index("Z")
            z_axis = sum(1 for axis in axes[:z_axis_original] if axis in {"Z", "Y", "X"})
            if z_projection == "max":
                crop = crop.max(axis=z_axis)
            elif z_projection == "mean":
                crop = crop.mean(axis=z_axis)
            elif z_projection == "middle":
                crop = np.take(crop, crop.shape[z_axis] // 2, axis=z_axis)
            else:
                raise ValueError("z_projection must be one of: max, mean, middle")
        return np.asarray(crop)


def _xenium_pixel_size_um(outs_path):
    experiment_path = Path(outs_path) / "experiment.xenium"
    if not experiment_path.exists():
        raise FileNotFoundError(f"Missing experiment.xenium: {experiment_path}")
    meta = json.loads(experiment_path.read_text())
    pixel_size = meta.get("pixel_size")
    if pixel_size is None:
        raise ValueError(f"No pixel_size entry found in {experiment_path}")
    return float(pixel_size)


def _choose_xenium_morphology_image(outs_path, image_kind="auto", image_path=None):
    outs_path = Path(outs_path)
    if image_path is not None:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        resolved_kind = image_kind if image_kind != "auto" else "custom"
        return image_path, resolved_kind

    candidates = {
        "mip": outs_path / "morphology_mip.ome.tif",
        "focus": outs_path / "morphology_focus.ome.tif",
        "focus_dir0": outs_path / "morphology_focus" / "morphology_focus_0000.ome.tif",
        "zstack": outs_path / "morphology.ome.tif",
    }
    if image_kind != "auto":
        path = candidates.get(image_kind)
        if path is None:
            raise ValueError("image_kind must be one of: auto, mip, focus, focus_dir0, zstack")
        if not path.exists():
            raise FileNotFoundError(path)
        return path, image_kind

    for kind in ("mip", "focus", "focus_dir0", "zstack"):
        path = candidates[kind]
        if path.exists():
            return path, kind
    raise FileNotFoundError(f"No Xenium morphology OME-TIFF found in {outs_path}")


def _polygon_mask_from_micron_vertices(group, pixel_size_um, image_shape):
    x_px = group["vertex_x"].to_numpy(dtype=float) / pixel_size_um
    y_px = group["vertex_y"].to_numpy(dtype=float) / pixel_size_um
    if len(x_px) < 3:
        return None

    height, width = image_shape[-2], image_shape[-1]
    x_min = max(int(np.floor(np.nanmin(x_px))) - 1, 0)
    x_max = min(int(np.ceil(np.nanmax(x_px))) + 2, width)
    y_min = max(int(np.floor(np.nanmin(y_px))) - 1, 0)
    y_max = min(int(np.ceil(np.nanmax(y_px))) + 2, height)
    if x_max <= x_min or y_max <= y_min:
        return None

    rr, cc = draw_polygon(
        y_px - y_min,
        x_px - x_min,
        shape=(y_max - y_min, x_max - x_min),
    )
    if len(rr) == 0:
        return None

    mask = np.zeros((y_max - y_min, x_max - x_min), dtype=bool)
    mask[rr, cc] = True
    return mask, (y_min, y_max, x_min, x_max), x_px, y_px


def _extract_xenium_dapi_features_one(
    cell_id,
    group,
    reader,
    pixel_size_um,
    z_projection="max",
    compute_texture=True,
    compute_haralick=False,
    lacunarity_box_size=5,
):
    mask_info = _polygon_mask_from_micron_vertices(group, pixel_size_um, reader.shape)
    if mask_info is None:
        return None

    mask, bbox, x_px, y_px = mask_info
    y_min, y_max, x_min, x_max = bbox
    crop = reader.read_crop(y_min, y_max, x_min, x_max, z_projection=z_projection)
    if crop.shape != mask.shape:
        crop = np.squeeze(crop)
    if crop.shape != mask.shape or not np.any(mask):
        return None

    vals = np.asarray(crop[mask], dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None

    masked_image = np.zeros(crop.shape, dtype=float)
    masked_image[mask] = crop[mask]

    total_intensity = float(np.sum(vals))
    yy, xx = np.nonzero(mask)
    weights = np.asarray(crop[mask], dtype=float)
    if total_intensity > 0:
        intensity_x_px = float(np.sum((xx + x_min) * weights) / total_intensity)
        intensity_y_px = float(np.sum((yy + y_min) * weights) / total_intensity)
    else:
        intensity_x_px = np.nan
        intensity_y_px = np.nan

    geom_x_px = float(np.nanmean(x_px))
    geom_y_px = float(np.nanmean(y_px))
    equiv_diameter_um = 2.0 * np.sqrt((mask.sum() * pixel_size_um ** 2) / np.pi)
    centroid_shift_um = (
        np.sqrt((intensity_x_px - geom_x_px) ** 2 + (intensity_y_px - geom_y_px) ** 2) * pixel_size_um
        if np.isfinite(intensity_x_px) and np.isfinite(intensity_y_px)
        else np.nan
    )

    out = {
        "cell_id": str(cell_id),
        "dapi_n_pixels": int(vals.size),
        "dapi_area_um2": float(mask.sum() * pixel_size_um ** 2),
        "dapi_total_intensity": total_intensity,
        "dapi_mean": float(np.mean(vals)),
        "dapi_std": float(np.std(vals)),
        "dapi_min": float(np.min(vals)),
        "dapi_p10": float(np.percentile(vals, 10)),
        "dapi_p50": float(np.percentile(vals, 50)),
        "dapi_p90": float(np.percentile(vals, 90)),
        "dapi_max": float(np.max(vals)),
        "dapi_iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
        "dapi_nonzero_fraction": float(np.mean(vals > 0)),
        "dapi_intensity_centroid_x": intensity_x_px * pixel_size_um if np.isfinite(intensity_x_px) else np.nan,
        "dapi_intensity_centroid_y": intensity_y_px * pixel_size_um if np.isfinite(intensity_y_px) else np.nan,
        "dapi_centroid_shift_um": float(centroid_shift_um),
        "dapi_polarity_score": float(centroid_shift_um / equiv_diameter_um) if equiv_diameter_um > 0 else np.nan,
        "dapi_inertia": calculate_moment_of_inertia(masked_image, intensity_x_px - x_min, intensity_y_px - y_min)
        if np.isfinite(intensity_x_px) and np.isfinite(intensity_y_px)
        else np.nan,
    }

    if compute_texture:
        out["dapi_entropy"] = calculate_entropy(masked_image)
        out["dapi_lacunarity"] = calculate_lacunarity(masked_image, box_size=lacunarity_box_size)
    if compute_haralick and vals.size >= 4:
        out.update({f"dapi_{k}": v for k, v in calculate_haralick_features_rescaled(masked_image).items()})
    return out


def extract_xenium_dapi_features(
    outs_path,
    cell_ids=None,
    output_path=None,
    image_kind="auto",
    image_path=None,
    channel_index=0,
    z_projection="max",
    max_cells=None,
    random_state=42,
    compute_texture=True,
    compute_haralick=False,
    lacunarity_box_size=5,
    progress_every=10000,
):
    """Extract per-nucleus DAPI pixel features from 10x Xenium morphology images.

    Parameters
    ----------
    outs_path:
        Path to a Xenium ``outs`` directory containing ``nucleus_boundaries.parquet``,
        ``experiment.xenium``, and morphology OME-TIFF files.
    cell_ids:
        Optional iterable of Xenium cell IDs to restrict extraction. This is useful
        for epithelial-only pseudotime feature extraction.
    image_kind:
        ``auto`` prefers ``morphology_mip.ome.tif``, then root focus,
        then ``morphology_focus/morphology_focus_0000.ome.tif``, then z-stack.
        When ``image_path`` is provided, this is only recorded as source metadata.
    image_path:
        Optional explicit morphology OME-TIFF path. Use this for datasets where
        the verified DAPI image is not the file that ``image_kind='auto'`` would
        select.
    channel_index:
        Channel index used when the morphology image has a channel axis. The
        Xenium focus image often stores DAPI in channel 0.
    max_cells:
        Optional random pilot subset size.
    compute_haralick:
        Haralick features are useful but slower; keep False for first pilots.
    """

    outs_path = Path(outs_path)
    boundary_path = outs_path / "nucleus_boundaries.parquet"
    if not boundary_path.exists():
        raise FileNotFoundError(boundary_path)

    pixel_size_um = _xenium_pixel_size_um(outs_path)
    image_path, resolved_image_kind = _choose_xenium_morphology_image(
        outs_path,
        image_kind=image_kind,
        image_path=image_path,
    )

    boundaries = pd.read_parquet(boundary_path, columns=["cell_id", "vertex_x", "vertex_y"])
    boundaries["cell_id"] = boundaries["cell_id"].astype(str)
    if cell_ids is not None:
        cell_ids = pd.Index(cell_ids).astype(str)
        cell_id_set = set(cell_ids)
        boundaries = boundaries.loc[boundaries["cell_id"].isin(cell_id_set)].copy()

    unique_cell_ids = boundaries["cell_id"].drop_duplicates().to_numpy()
    if max_cells is not None and len(unique_cell_ids) > max_cells:
        rng = np.random.default_rng(random_state)
        keep = set(rng.choice(unique_cell_ids, size=int(max_cells), replace=False))
        boundaries = boundaries.loc[boundaries["cell_id"].isin(keep)].copy()

    rows = []
    reader = _XeniumMorphologyReader(image_path, channel_index=channel_index)
    try:
        n_groups = boundaries["cell_id"].nunique()
        for idx, (cell_id, group) in enumerate(boundaries.groupby("cell_id", sort=False), start=1):
            rec = _extract_xenium_dapi_features_one(
                cell_id=cell_id,
                group=group,
                reader=reader,
                pixel_size_um=pixel_size_um,
                z_projection=z_projection,
                compute_texture=compute_texture,
                compute_haralick=compute_haralick,
                lacunarity_box_size=lacunarity_box_size,
            )
            if rec is not None:
                rows.append(rec)
            if progress_every and idx % progress_every == 0:
                print(f"Processed {idx:,}/{n_groups:,} nuclei from {outs_path.name}")
    finally:
        reader.close()

    out = pd.DataFrame(rows)
    if not out.empty:
        out["xenium_dapi_image_kind"] = resolved_image_kind
        out["xenium_dapi_image_path"] = str(image_path)
        out["xenium_dapi_channel_index"] = int(channel_index)
        out["xenium_pixel_size_um"] = pixel_size_um

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, index=False)
    return out
