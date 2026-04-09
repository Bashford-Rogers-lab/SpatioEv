from . import qc
from . import plot

from .io.load import load_h5ad
from .preprocessing.normalize import (
    zscore_normalize,
    add_obs_from_var,
    add_zscore_obs_features,
)

from .phenotype.clustering import cluster_cells
from .phenotype.annotation import annotate_interactive, annotate_from_csv
from .phenotype.annotate_loop import annotate_refinements
from .phenotype.subset import subset_cells
from .phenotype.merge import merge_annotations, merge_refinements
from .phenotype.refine import refine_clusters

from .visualization.spatial import inspect_clusters
from .visualization.heatmap import plot_cluster_heatmap
from .visualization.overview import plot_refinement_umaps
from .visualization.spatial_scatter import spatial_scatter_plot

from .ml.svm import run_svm_phenotyping
from .ml.features import build_feature_matrix
from .ml.training import train_svm_classifier
from .ml.prediction import predict_svm
from .ml.svm_inspection import inspect_reassigned_cells, inspect_disagreements

from .spatial import (
    assign_tiles,
    compute_general_density, compute_phenotype_density,
    plot_density_heatmap, plot_phenotype_density_heatmap,
    phenotype_density_correlation, plot_density_correlation,
    compute_kde_density, plot_kde_density,
    compute_local_density_by_phenotype, compute_local_density_all_cells,
    plot_local_density_map,
    phenotype_interaction_density,
    plot_interaction_density,
    plot_interaction_overlay,
    plot_interaction_distribution,
    compute_radius_density,
    validate_spatial_coordinates,
    compute_convex_hull,
    compute_convex_hull_area,
    compute_tissue_areas,
    detect_edge_cells,

    ripleys_k,
    ripleys_curve,
    ripley_envelope,
    ripleys_k_by_image,
    ripleys_k_by_phenotype,
    cross_ripleys_k,
    cross_ripleys_k_by_phenotype,
    cross_ripleys_k_all_pairs,
    cross_ripleys_curve,
    cross_ripley_envelope,
    cross_ripleys_curve_by_phenotype,
    cross_ripley_envelope_by_phenotype,
    cross_ripley_permutation_envelope,
    ripley_local_counts_by_phenotype,
    cross_ripley_local_counts,
    ripley_interaction_scale,
    ripley_spatial_scales,
    morans_i,
    morans_i_by_image,
    local_morans_i,
    add_local_morans_i,
    classify_local_morans_i,
    add_local_morans_i_quadrants,
    cross_morans_i,
    cross_morans_i_by_image,
    cross_morans_i_permutation_test,
    cross_morans_i_by_image_permutation_test,
    local_cross_morans_i,
    add_local_cross_morans_i,
    classify_local_cross_morans_i,
    add_local_cross_morans_i_quadrants,
    summarize_target_features_around_source_cells,
    cross_morans_i_feature_matrix,
    add_local_cross_morans_i_between_phenotypes,
    
    plot_spatial_feature,
    plot_spatial_category,
    plot_niche_boundaries,
    plot_correlation_heatmap,
    estimate_density_adaptive_dbscan_params,
    estimate_spatial_component_params,
    cluster_spatial_niches,
    cluster_spatial_components,
    cluster_spatial_components_hdbscan,
    cluster_spatial_components_from_mask,
    build_niche_boundaries,
    buffer_niche_boundaries,
    assign_cells_to_niche_regions,
    summarize_niche_composition,
    add_niche_regions_to_obs,
    build_cell_graph,
    extract_niche_subgraph,
    extract_all_niche_subgraphs,
    summarize_niche_graph_features,
    build_niche_feature_table,
    build_niche_feature_table_batched,
    summarize_niche_surrounding_context,
    score_pdac_niche_pathology_modules,
    calculate_polarity_score,
    calculate_moment_of_inertia,
    calculate_haralick_features,
    calculate_entropy,
    calculate_lacunarity,
    calculate_channel_correlation,
    extract_cell_pixel_features_for_fov,
    extract_cell_pixel_features,
)
