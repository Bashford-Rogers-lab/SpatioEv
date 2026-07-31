# ============================================================================
# 07_xenium_05_spatialcellchat.R
#
# SpatialCellChat (CellChat v3) analysis of Xenium pancreas data
# stratified by niche pseudotime bins.
#
# Prerequisites
# -------------
# 1. Run Python notebook 07_xenium_04_transcriptional_trends.ipynb first
#    to export: spatialcellchat/niche_pseudotime_bins.csv,
#               spatialcellchat/{sample_id}_cell_meta.csv,
#               spatialcellchat/{sample_id}_counts_common_genes.csv.gz
#
# 2. Install CellChat v3 (SpatialCellChat) in R:
#    install.packages("devtools")
#    devtools::install_github("jinworks/CellChat")
#    # Dependencies (if not yet installed):
#    install.packages(c("Seurat","NMF","ggalluvial","ggplot2","dplyr","igraph"))
#
# Outputs
# -------
# spatialcellchat/cellchat_{sample_id}_bin{N}.rds  — per-bin CellChat objects
# spatialcellchat/lr_interaction_summary.csv        — L-R pair strengths per bin
# spatialcellchat/pathway_summary.csv               — signaling pathway strengths per bin
# ============================================================================

suppressPackageStartupMessages({
  library(CellChat)
  library(dplyr)
  library(ggplot2)
  library(data.table)
})

# ── Configuration ────────────────────────────────────────────────────────────
ROOT_DIR        <- "/Users/shihongwu/SpatioEv"
CELLCHAT_DIR    <- file.path(ROOT_DIR, "data", "xenium_pancreas_10x", "spatialcellchat")
FIGURE_DIR      <- file.path(ROOT_DIR, "data", "xenium_pancreas_10x", "figures")
dir.create(FIGURE_DIR,   showWarnings=FALSE, recursive=TRUE)

# Spatial communication radius (µm) — cells within this distance can communicate
# Xenium pixel size ≈ 0.2125 µm; 50 µm ≈ 235 pixels
SPATIAL_RADIUS_UM  <- 50

# Minimum cells per cell type to include in communication inference
MIN_CELLS_PER_TYPE <- 5

# Pseudotime bins to analyse (matching N_PT_BINS=5 in Python)
PT_BINS <- paste0("bin", 1:5)

# ── Load CellChat human database ─────────────────────────────────────────────
# CellChat v3 ships with human and mouse databases
CellChatDB <- CellChatDB.human

# Focus on secreted signalling (most relevant for tumour-stroma crosstalk)
# Options: "Secreted Signaling", "ECM-Receptor", "Cell-Cell Contact"
CellChatDB.use <- subsetDB(CellChatDB, search="Secreted Signaling", key="annotation")

# ── Helper: build one CellChat object for a given subset of cells ─────────────
build_cellchat <- function(counts_mat, meta_sub, sample_id, bin_id) {
  cat(sprintf("  Building CellChat for %s %s: %d cells\n",
              sample_id, bin_id, nrow(meta_sub)))

  # Filter to cells with valid spatial coordinates
  meta_sub <- meta_sub[!is.na(meta_sub$x) & !is.na(meta_sub$y), ]
  cell_ids  <- rownames(meta_sub)
  cell_ids  <- intersect(cell_ids, colnames(counts_mat))
  if (length(cell_ids) < 20) {
    cat("    Too few cells — skipping.\n")
    return(NULL)
  }

  X_sub    <- counts_mat[, cell_ids, drop=FALSE]
  meta_use <- meta_sub[cell_ids, , drop=FALSE]

  # Cell type labels (Tier_A)
  cell_types <- setNames(as.character(meta_use$Tier_A), cell_ids)
  cell_types[is.na(cell_types) | cell_types==""] <- "Unknown"

  # Check minimum cells per type
  type_counts <- table(cell_types)
  valid_types <- names(type_counts[type_counts >= MIN_CELLS_PER_TYPE])
  keep        <- cell_ids[cell_types[cell_ids] %in% valid_types]
  if (length(keep) < 20) {
    cat("    Too few cells with valid annotations — skipping.\n")
    return(NULL)
  }
  X_sub      <- X_sub[, keep, drop=FALSE]
  meta_use   <- meta_use[keep, , drop=FALSE]
  cell_types <- cell_types[keep]

  # Spatial coordinates (µm)
  spatial_locs <- data.frame(
    imagerow = meta_use$y,
    imagecol = meta_use$x,
    row.names = keep
  )

  # Create CellChat object
  cc <- createCellChat(
    object       = X_sub,
    meta         = data.frame(labels=cell_types, row.names=keep),
    group.by     = "labels",
    datatype     = "spatial",
    coordinates  = spatial_locs,
    spatial.factors = list(
      ratio       = 1,               # coordinates already in µm
      tol         = SPATIAL_RADIUS_UM
    )
  )
  cc@DB <- CellChatDB.use

  # Preprocessing
  cc <- subsetData(cc)
  cc <- identifyOverExpressedGenes(cc, do.fast = FALSE)  # presto not required
  cc <- identifyOverExpressedInteractions(cc)

  # Inference — use truncated mean (robust for sparse spatial data)
  # CellChat v3 uses contact.range (µm) instead of interaction.range
  # scale.distance: CellChat v3 requires the scaled minimum distance to be in [1,2].
  # The error message from a test run suggested a value slightly below 8.8 — use 8.
  cc <- computeCommunProb(cc, type="truncatedMean", trim=0.1,
                           distance.use=TRUE,
                           contact.range=SPATIAL_RADIUS_UM,
                           scale.distance=8)
  cc <- filterCommunication(cc, min.cells=MIN_CELLS_PER_TYPE)
  cc <- computeCommunProbPathway(cc)
  cc <- aggregateNet(cc)

  return(cc)
}

# ── Main loop: per-sample, per-pseudotime-bin analysis ───────────────────────
manifest <- read.csv(file.path(CELLCHAT_DIR, "export_manifest.csv"),
                     stringsAsFactors=FALSE)
niche_bins <- read.csv(file.path(CELLCHAT_DIR, "niche_pseudotime_bins.csv"),
                        stringsAsFactors=FALSE)

lr_rows      <- list()
pathway_rows <- list()

for (i in seq_len(nrow(manifest))) {
  sid        <- manifest$sample_id[i]
  count_path <- manifest$count_path[i]
  meta_path  <- manifest$meta_path[i]

  cat(sprintf("\n=== %s ===\n", sid))

  # Load count matrix (cells × genes) → transpose to genes × cells for CellChat
  # Prefer full-gene matrix (_all_genes) for richer CellChat L-R coverage;
  # fall back to common-gene matrix if full not yet exported.
  full_count_path <- sub("_counts_common_genes.csv.gz", "_counts_all_genes.csv.gz", count_path)
  if (file.exists(full_count_path)) {
    count_path <- full_count_path
    cat("  Using full-gene count matrix.\n")
  } else {
    cat("  WARNING: full-gene matrix not found; using 98-gene common matrix (poor L-R coverage).\n")
  }
  cat("  Loading count matrix...\n")
  counts_raw <- fread(count_path, header=TRUE, data.table=FALSE)
  rownames(counts_raw) <- counts_raw[[1]]
  counts_raw <- counts_raw[, -1, drop=FALSE]
  counts_mat <- t(as.matrix(counts_raw))   # genes × cells
  mode(counts_mat) <- "numeric"

  # Load cell metadata
  meta_all <- read.csv(meta_path, stringsAsFactors=FALSE)
  # row.names=1 is avoided because the index column name varies; handle manually
  if ("cell_id" %in% colnames(meta_all)) {
    rownames(meta_all) <- meta_all$cell_id
  } else {
    rownames(meta_all) <- as.character(seq_len(nrow(meta_all)))
  }

  # Assign pseudotime bin to each cell
  # Strategy A: direct join via niche_id if the column exists
  # Strategy B: spatial nearest-niche lookup via niche centroid coordinates
  niche_sid <- niche_bins[niche_bins$sample_id == sid, ]

  if ("pt_bin_cellchat" %in% colnames(meta_all)) {
    # Python already merged pseudotime bins into cell_meta.csv — use directly
    cat("  pt_bin_cellchat already present in cell metadata (from Python export).\n")
  } else if ("niche_id" %in% colnames(meta_all) && nrow(niche_sid) > 0) {
    cat("  Assigning pseudotime bins via niche_id join.\n")
    niche_map <- niche_sid[, c("xenium_ductal_epithelium_component", "pt_bin_cellchat")]
    meta_all  <- merge(meta_all, niche_map,
                       by.x="niche_id", by.y="xenium_ductal_epithelium_component",
                       all.x=TRUE)
    if ("cell_id" %in% colnames(meta_all)) rownames(meta_all) <- meta_all$cell_id
  } else {
    # Spatial nearest-niche assignment using centroid file
    centroid_path <- file.path(CELLCHAT_DIR, "niche_centroids_with_pseudotime.csv")
    if (file.exists(centroid_path) && all(c("x","y") %in% colnames(meta_all))) {
      cat("  Assigning pseudotime bins via spatial nearest-niche lookup.\n")
      centroids <- read.csv(centroid_path, stringsAsFactors=FALSE)
      centroids <- centroids[centroids$sample_id == sid &
                               !is.na(centroids$niche_x) & !is.na(centroids$niche_y) &
                               !is.na(centroids$pt_bin_cellchat), ]
      if (nrow(centroids) > 0) {
        cell_xy   <- as.matrix(meta_all[, c("x", "y")])
        niche_xy  <- as.matrix(centroids[, c("niche_x", "niche_y")])
        # For each cell find the nearest niche centroid
        nn_idx <- apply(cell_xy, 1, function(pt) {
          dists <- sqrt(rowSums((niche_xy - matrix(pt, nrow=nrow(niche_xy), ncol=2, byrow=TRUE))^2))
          which.min(dists)
        })
        meta_all$pt_bin_cellchat <- centroids$pt_bin_cellchat[nn_idx]
      } else {
        meta_all$pt_bin_cellchat <- NA_character_
      }
    } else {
      cat("  Warning: no niche_id and no centroid file — pt_bin_cellchat set to NA.\n")
      meta_all$pt_bin_cellchat <- NA_character_
    }
  }

  # Debug: show bin distribution after assignment
  if ("pt_bin_cellchat" %in% colnames(meta_all)) {
    cat("  pt_bin_cellchat distribution:\n")
    print(table(meta_all$pt_bin_cellchat, useNA="ifany"))
    cat("  Tier_A distribution:\n")
    print(table(meta_all$Tier_A, useNA="ifany"))
  }

  for (bin_id in PT_BINS) {
    # Get all cells in this pseudotime bin (ductal niches in this bin + their neighbours)
    # Strategy: include ALL cells whose niche is in this bin (or no niche assignment)
    niche_in_bin <- niche_sid$xenium_ductal_epithelium_component[
      !is.na(niche_sid$pt_bin_cellchat) & niche_sid$pt_bin_cellchat==bin_id
    ]
    if (length(niche_in_bin) < 3) {
      cat(sprintf("  %s: too few niches — skip\n", bin_id))
      next
    }

    # Include cells from niches in this bin + all cells within spatial radius of those niches
    # Simple approach: take all cells in the bin's niches
    meta_bin <- meta_all[!is.na(meta_all$pt_bin_cellchat) &
                          meta_all$pt_bin_cellchat == bin_id, , drop=FALSE]

    # Also include non-ductal cells that are spatial neighbours
    # (identified by proximity to ductal niche centroids)
    ductal_cells <- meta_bin[!is.na(meta_bin$Tier_A) &
                               meta_bin$Tier_A=="pancreatic ductal epithelium", ]

    if (nrow(ductal_cells) < MIN_CELLS_PER_TYPE) {
      cat(sprintf("  %s: too few ductal cells — skip\n", bin_id))
      next
    }

    # Find all non-ductal cells within SPATIAL_RADIUS_UM of any ductal cell
    non_ductal_all <- meta_all[is.na(meta_all$pt_bin_cellchat) |
                                 meta_all$pt_bin_cellchat != bin_id, ]
    if (nrow(non_ductal_all) > 0 && nrow(ductal_cells) > 0) {
      ductal_xy   <- as.matrix(ductal_cells[, c("x","y")])
      nonduc_xy   <- as.matrix(non_ductal_all[, c("x","y")])
      # vectorised distance — find neighbours within radius
      nbr_flags <- apply(nonduc_xy, 1, function(pt) {
        min(sqrt(rowSums((ductal_xy - matrix(pt, nrow=nrow(ductal_xy), ncol=2, byrow=TRUE))^2))) <= SPATIAL_RADIUS_UM
      })
      meta_neighbours <- non_ductal_all[nbr_flags, , drop=FALSE]
      # Cap at 50k non-ductal neighbours to avoid memory issues on large samples
      if (nrow(meta_neighbours) > 50000) {
        set.seed(42)
        meta_neighbours <- meta_neighbours[sample(nrow(meta_neighbours), 50000), , drop=FALSE]
      }
    } else {
      meta_neighbours <- data.frame()
    }

    meta_combined <- rbind(meta_bin, meta_neighbours)
    meta_combined <- meta_combined[!duplicated(rownames(meta_combined)), ]

    # Skip if this bin's RDS already exists (allows resuming interrupted runs)
    rds_path <- file.path(CELLCHAT_DIR, sprintf("cellchat_%s_%s.rds", sid, bin_id))
    if (file.exists(rds_path)) {
      cat(sprintf("  %s already exists — loading for extraction.\n", basename(rds_path)))
      cc <- readRDS(rds_path)
    } else {

    # Build CellChat object — wrapped in tryCatch to skip edge-case bins gracefully
    cc <- tryCatch(
      build_cellchat(counts_mat, meta_combined, sid, bin_id),
      error = function(e) {
        cat(sprintf("    ERROR in %s %s — skipping: %s\n", sid, bin_id, conditionMessage(e)))
        NULL
      }
    )
    if (is.null(cc)) next
    saveRDS(cc, rds_path)
    cat(sprintf("  Saved: %s\n", basename(rds_path)))
    }  # end else (not cached)

    # Extract L-R interaction table
    if (!is.null(cc@net$prob)) {
      lr_df <- subsetCommunication(cc)
      if (!is.null(lr_df) && nrow(lr_df) > 0) {
        lr_df$sample_id <- sid
        lr_df$pt_bin    <- bin_id
        lr_rows[[length(lr_rows)+1]] <- lr_df
      }
    }

    # Extract pathway summary
    if (!is.null(cc@netP$prob)) {
      pwy_df <- subsetCommunication(cc, slot.name="netP")
      if (!is.null(pwy_df) && nrow(pwy_df) > 0) {
        pwy_df$sample_id <- sid
        pwy_df$pt_bin    <- bin_id
        pathway_rows[[length(pathway_rows)+1]] <- pwy_df
      }
    }
  }
}

# ── Save summary tables ───────────────────────────────────────────────────────
if (length(lr_rows) > 0) {
  lr_all <- bind_rows(lr_rows)
  write.csv(lr_all, file.path(CELLCHAT_DIR, "lr_interaction_summary.csv"), row.names=FALSE)
  cat(sprintf("\nL-R summary: %d rows, %d unique L-R pairs\n",
              nrow(lr_all), length(unique(paste(lr_all$ligand, lr_all$receptor)))))
}

if (length(pathway_rows) > 0) {
  pwy_all <- bind_rows(pathway_rows)
  write.csv(pwy_all, file.path(CELLCHAT_DIR, "pathway_summary.csv"), row.names=FALSE)
  cat(sprintf("Pathway summary: %d rows, %d unique pathways\n",
              nrow(pwy_all), length(unique(pwy_all$pathway_name))))
}

cat("\n✓ SpatialCellChat analysis complete.\n")
cat(sprintf("Output directory: %s\n", CELLCHAT_DIR))
cat("Next step: open 07_xenium_06_cellchat_integration.ipynb\n")
