# Preprocessing (`sv.pp`)

Use `spatioev.pp` for data preparation, segmentation QC, coordinate checks,
and pixel feature extraction.

---

## Segmentation QC

::: spatioev.pp.qc.run_segmentation_qc

::: spatioev.pp.qc.generate_qc_summary

::: spatioev.pp.qc.filter_segmentation_errors

::: spatioev.pp.qc.compute_area_um2

::: spatioev.pp.qc.categorize_area

::: spatioev.pp.qc.categorize_nc_ratio

---

## Normalization & Feature Preparation

::: spatioev.pp.normalize.zscore_normalize

::: spatioev.pp.normalize.add_obs_from_var

::: spatioev.pp.normalize.add_zscore_obs_features

---

## Spatial Coordinate Preprocessing

::: spatioev.pp.spatial_prep.validate_spatial_coordinates

::: spatioev.pp.spatial_prep.compute_tissue_areas

::: spatioev.pp.spatial_prep.detect_edge_cells

::: spatioev.pp.spatial_prep.compute_convex_hull

::: spatioev.pp.spatial_prep.compute_convex_hull_area

::: spatioev.pp.spatial_prep.distance_to_convex_hull_boundary

---

## Pixel & Morphology Feature Extraction

::: spatioev.pp.pixel.extract_cell_pixel_features

::: spatioev.pp.pixel.extract_cell_pixel_features_for_fov

::: spatioev.pp.pixel.extract_xenium_dapi_features

::: spatioev.pp.pixel.calculate_haralick_features

::: spatioev.pp.pixel.calculate_haralick_features_rescaled

::: spatioev.pp.pixel.calculate_entropy

::: spatioev.pp.pixel.calculate_lacunarity

::: spatioev.pp.pixel.calculate_polarity_score

::: spatioev.pp.pixel.calculate_moment_of_inertia

::: spatioev.pp.pixel.calculate_channel_correlation

---

## Configuration

::: spatioev.config.QCConfig

::: spatioev.config.ClusteringConfig
