from .general_density import (
    assign_tiles,
    compute_general_density, compute_phenotype_density,
    plot_density_heatmap, plot_phenotype_density_heatmap,
    phenotype_density_correlation, plot_density_correlation,
    compute_kde_density, plot_kde_density
)

from .local_density_KNN import (
    compute_local_density_by_phenotype, compute_local_density_all_cells,
    plot_local_density_map
)

from .interaction import (
    phenotype_interaction_density,
    plot_interaction_density,
    plot_interaction_overlay,
    plot_interaction_distribution
)

from .local_density_radius import compute_radius_density

from .preprocessing import (
    validate_spatial_coordinates,
    compute_convex_hull,
    compute_convex_hull_area,
    compute_tissue_areas,
    detect_edge_cells,
)

from .spatial_stats import (
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
    ripley_interaction_scale,
    ripley_spatial_scales,
    morans_i,
    morans_i_by_image,
    local_morans_i,
    add_local_morans_i,
    cross_morans_i,
    local_cross_morans_i
)

from .visualization import (
    plot_spatial_feature,
    plot_spatial_category,
    plot_niche_boundaries,
    plot_correlation_heatmap,
)

from .spatial_niche_boundaries import (
    estimate_density_adaptive_dbscan_params,
    cluster_spatial_niches,
    build_niche_boundaries,
    buffer_niche_boundaries,
    assign_cells_to_niche_regions,
    summarize_niche_composition,
    add_niche_regions_to_obs,
)
