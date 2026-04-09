import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.ndimage import find_objects
from scipy.stats import pearsonr
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
