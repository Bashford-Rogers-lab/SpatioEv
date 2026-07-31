#!/usr/bin/env Rscript
#
# run_spatialcellchat_pdac_io_v1.R
#
# Run SpatialCellChat for pdac_io_v1 using the existing CSV export files.
#
# The meta CSV only has pt_bin_cellchat assigned for ductal epithelial cells.
# This script extends bin assignment to all cell types by spatial proximity:
# each non-ductal cell inherits the pt_bin of its nearest ductal cell
# (if within ASSIGN_RADIUS_UM).
#
# Input:
#   data/xenium_pancreas_10x/spatialcellchat/pdac_io_v1_cell_meta.csv
#   data/xenium_pancreas_10x/spatialcellchat/pdac_io_v1_counts_common_genes.csv.gz
#
# Output (same directory as existing samples):
#   cellchat_pdac_io_v1_bin{N}.rds    — one per pt_bin
#   pdac_io_v1_pathway_summary.csv    — pathway-level probabilities (append to pathway_summary.csv manually)
#   pdac_io_v1_lr_summary.csv         — LR-level table
#
# Run from project root:
#   Rscript scripts/run_spatialcellchat_pdac_io_v1.R

suppressPackageStartupMessages({
  library(Matrix)
  library(readr)
  library(dplyr)
  library(tibble)
  library(future)
})

if (!requireNamespace("SpatialCellChat", quietly = TRUE)) {
  stop(
    "SpatialCellChat is not installed.\n",
    "Install: devtools::install_github('jinworks/SpatialCellChat')"
  )
}
suppressPackageStartupMessages(library(SpatialCellChat))

options(stringsAsFactors = FALSE)
set.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# 0. Settings
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_ID        <- "pdac_io_v1"
DATA_DIR         <- "data/xenium_pancreas_10x/spatialcellchat"
META_CSV         <- file.path(DATA_DIR, paste0(SAMPLE_ID, "_cell_meta.csv"))
COUNTS_CSV_GZ    <- file.path(DATA_DIR, paste0(SAMPLE_ID, "_counts_common_genes.csv.gz"))

ASSIGN_RADIUS_UM <- 300      # max distance to inherit pt_bin from nearest ductal cell
INTERACTION_RANGE_UM <- 250  # CellChat spatial interaction range
CONTACT_RANGE_UM     <- 10   # CellChat contact range
MIN_CELLS_PER_BIN    <- 150  # skip bins with fewer total assigned cells
MIN_CELLS_PER_GROUP  <- 20   # CellChat minimum cells per cell type
MAX_CELLS_PER_BIN    <- 15000L  # downsample large bins (bin5 has 40k+ ductal cells)
FUTURE_WORKERS       <- 2L

BINS_TO_RUN <- c("bin2", "bin3", "bin4", "bin5")  # bin1 has 0 ductal cells

future::plan(multisession, workers = FUTURE_WORKERS)
options(future.globals.maxSize = 32 * 1024^3)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load metadata
# ─────────────────────────────────────────────────────────────────────────────

message("Loading metadata: ", META_CSV)
meta <- read_csv(META_CSV, show_col_types = FALSE)
message("  ", nrow(meta), " cells, columns: ", paste(names(meta), collapse = ", "))

# Require x, y, Tier_A
stopifnot(all(c("cell_id", "x", "y", "Tier_A", "pt_bin_cellchat") %in% names(meta)))

# ─────────────────────────────────────────────────────────────────────────────
# 2. Extend pt_bin to surrounding cells by spatial proximity
# ─────────────────────────────────────────────────────────────────────────────

message("Assigning surrounding cells to pt_bins by spatial proximity (radius = ", ASSIGN_RADIUS_UM, " µm)")

# Ductal cells with known bin
ductal_binned <- meta %>%
  filter(!is.na(pt_bin_cellchat)) %>%
  select(cell_id, x, y, pt_bin_cellchat)

# Non-ductal cells (or ductal without bin)
non_ductal <- meta %>%
  filter(is.na(pt_bin_cellchat)) %>%
  select(cell_id, x, y, Tier_A)

message("  Ductal cells with bin: ", nrow(ductal_binned))
message("  Cells needing assignment: ", nrow(non_ductal))

# Assign by nearest ductal cell within radius
# For speed, use vectorized distance with a chunked approach
assign_bin_by_proximity <- function(non_ductal_df, ductal_df, radius_um) {
  if (nrow(non_ductal_df) == 0 || nrow(ductal_df) == 0) {
    return(rep(NA_character_, nrow(non_ductal_df)))
  }

  nd_x <- non_ductal_df$x
  nd_y <- non_ductal_df$y
  d_x  <- ductal_df$x
  d_y  <- ductal_df$y
  d_bin <- ductal_df$pt_bin_cellchat

  # Process in chunks of 5000 to avoid memory explosion
  chunk_size <- 5000L
  n <- nrow(non_ductal_df)
  result <- character(n)

  for (start in seq(1, n, by = chunk_size)) {
    end    <- min(start + chunk_size - 1L, n)
    idx    <- start:end
    dx     <- outer(nd_x[idx], d_x, "-")
    dy     <- outer(nd_y[idx], d_y, "-")
    dist2  <- dx^2 + dy^2
    nearest_i <- max.col(-dist2)  # column index of minimum distance per row
    nearest_dist <- sqrt(dist2[cbind(seq_along(idx), nearest_i)])
    assigned_bin <- d_bin[nearest_i]
    assigned_bin[nearest_dist > radius_um] <- NA_character_
    result[idx] <- assigned_bin
  }
  result
}

non_ductal$pt_bin_cellchat <- assign_bin_by_proximity(non_ductal, ductal_binned, ASSIGN_RADIUS_UM)
message("  Assigned: ", sum(!is.na(non_ductal$pt_bin_cellchat)), " / ", nrow(non_ductal), " surrounding cells")

# Combine back
meta_extended <- bind_rows(
  meta %>% filter(!is.na(pt_bin_cellchat)),
  non_ductal %>% left_join(
    meta %>% select(cell_id, Tier_A, Tier_B, niche_id, xenium_pseudotime_norm),
    by = "cell_id"
  ) %>%
    filter(!is.na(pt_bin_cellchat))
) %>%
  filter(!is.na(Tier_A) & Tier_A != "")

message("  Extended meta: ", nrow(meta_extended), " total cells with bin assignment")
message("  Bin counts:\n",
        paste(capture.output(print(table(meta_extended$pt_bin_cellchat))), collapse = "\n"))

# ─────────────────────────────────────────────────────────────────────────────
# 3. Load expression matrix
# ─────────────────────────────────────────────────────────────────────────────

message("Loading expression matrix: ", COUNTS_CSV_GZ)
expr_df <- read_csv(COUNTS_CSV_GZ, show_col_types = FALSE)
# First column is cell_id (unnamed), rest are genes
cell_ids_in_expr <- expr_df[[1]]
gene_names       <- colnames(expr_df)[-1]

# Build sparse matrix (genes × cells), clip negatives to 0 for CellChat
expr_mat <- as.matrix(expr_df[, -1])
rownames(expr_mat) <- cell_ids_in_expr
expr_mat[expr_mat < 0] <- 0  # CellChat expects non-negative log-normalized values
expr_mat <- t(expr_mat)       # genes × cells
colnames(expr_mat) <- cell_ids_in_expr
rownames(expr_mat) <- gene_names
expr_mat <- Matrix(expr_mat, sparse = TRUE)
message("  Expression matrix: ", nrow(expr_mat), " genes × ", ncol(expr_mat), " cells")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Run CellChat per bin
# ─────────────────────────────────────────────────────────────────────────────

pathway_rows <- list()
lr_rows      <- list()

safe_subset_communication <- function(chat, ...) {
  tryCatch(
    subsetCommunication(chat, ...),
    error = function(e) {
      message("  subsetCommunication: ", conditionMessage(e))
      tibble()
    }
  )
}

for (bin_name in BINS_TO_RUN) {
  message("\n── Processing ", bin_name, " ──")

  bin_meta <- meta_extended %>% filter(pt_bin_cellchat == bin_name)
  message("  Total cells: ", nrow(bin_meta))

  if (nrow(bin_meta) < MIN_CELLS_PER_BIN) {
    message("  Skipping: fewer than ", MIN_CELLS_PER_BIN, " cells")
    next
  }

  # Require ≥ 2 cell types with ≥ MIN_CELLS_PER_GROUP cells
  group_counts <- table(bin_meta$Tier_A)
  keep_groups  <- names(group_counts[group_counts >= MIN_CELLS_PER_GROUP])
  if (length(keep_groups) < 2) {
    message("  Skipping: fewer than 2 cell types with ≥ ", MIN_CELLS_PER_GROUP, " cells")
    next
  }
  bin_meta <- bin_meta %>% filter(Tier_A %in% keep_groups)

  # Downsample if too large
  if (nrow(bin_meta) > MAX_CELLS_PER_BIN) {
    message("  Downsampling from ", nrow(bin_meta), " to ", MAX_CELLS_PER_BIN)
    idx <- unlist(lapply(split(seq_len(nrow(bin_meta)), bin_meta$Tier_A), function(i) {
      target <- max(1L, round(length(i) * MAX_CELLS_PER_BIN / nrow(bin_meta)))
      sample(i, min(length(i), target))
    }))
    bin_meta <- bin_meta[sort(idx), ]
  }

  # Align expression
  common_cells <- intersect(bin_meta$cell_id, colnames(expr_mat))
  if (length(common_cells) < MIN_CELLS_PER_BIN) {
    message("  Skipping: insufficient cells in expression matrix")
    next
  }
  bin_meta <- bin_meta %>% filter(cell_id %in% common_cells)
  bin_expr <- expr_mat[, bin_meta$cell_id, drop = FALSE]

  # Build CellChat inputs
  rownames(bin_meta) <- bin_meta$cell_id
  coordinates <- data.frame(
    x = bin_meta$x,
    y = bin_meta$y,
    row.names = bin_meta$cell_id
  )

  spatial.factors <- list(ratio = 1, tol = CONTACT_RANGE_UM / 2)

  tryCatch({
    chat <- createSpatialCellChat(
      object         = bin_expr,
      meta           = bin_meta,
      group.by       = "Tier_A",
      datatype       = "spatial",
      coordinates    = coordinates,
      spatial.factors = spatial.factors
    )

    chat@DB <- CellChatDB.human
    chat <- subsetData(chat)
    chat <- identifyOverExpressedGenes(chat)
    chat <- identifyOverExpressedInteractions(chat)
    chat <- computeCommunProb(
      chat,
      type             = "truncatedMean",
      trim             = 0.1,
      distance.use     = TRUE,
      interaction.range = INTERACTION_RANGE_UM,
      contact.dependent = TRUE,
      contact.range    = CONTACT_RANGE_UM
    )
    chat <- filterCommunication(chat, min.cells = MIN_CELLS_PER_GROUP)
    chat <- computeCommunProbPathway(chat)
    chat <- aggregateNet(chat)

    # Save RDS
    rds_path <- file.path(DATA_DIR, paste0("cellchat_", SAMPLE_ID, "_", bin_name, ".rds"))
    saveRDS(chat, rds_path)
    message("  Saved: ", rds_path)

    # Extract pathway summary
    pw_df <- safe_subset_communication(chat, slot.name = "netP")
    if (nrow(pw_df) > 0) {
      pw_df$sample_id <- SAMPLE_ID
      pw_df$pt_bin    <- bin_name
      pathway_rows[[bin_name]] <- pw_df
    }

    # Extract LR summary
    lr_df <- safe_subset_communication(chat)
    if (nrow(lr_df) > 0) {
      lr_df$sample_id <- SAMPLE_ID
      lr_df$pt_bin    <- bin_name
      lr_rows[[bin_name]] <- lr_df
    }

    message("  Pathways found: ", length(unique(pw_df$pathway_name)),
            "  LR pairs: ", nrow(lr_df))
  }, error = function(e) {
    message("  ERROR in ", bin_name, ": ", conditionMessage(e))
  })
}

# ─────────────────────────────────────────────────────────────────────────────
# 5. Write summary CSVs
# ─────────────────────────────────────────────────────────────────────────────

if (length(pathway_rows) > 0) {
  pw_all <- bind_rows(pathway_rows)
  out_pw <- file.path(DATA_DIR, paste0(SAMPLE_ID, "_pathway_summary.csv"))
  write_csv(pw_all, out_pw)
  message("\nSaved pathway summary: ", out_pw, " (", nrow(pw_all), " rows)")

  # Also append to the combined pathway_summary.csv
  combined_path <- file.path(DATA_DIR, "pathway_summary.csv")
  if (file.exists(combined_path)) {
    existing <- read_csv(combined_path, show_col_types = FALSE)
    # Remove any existing pdac_io_v1 rows to avoid duplicates
    existing <- existing %>% filter(sample_id != SAMPLE_ID)
    combined <- bind_rows(existing, pw_all %>% select(any_of(names(existing))))
    write_csv(combined, combined_path)
    message("Updated combined pathway_summary.csv: ", nrow(combined), " total rows")
  }
} else {
  message("\nNo pathway results produced — check CellChat output above.")
}

if (length(lr_rows) > 0) {
  lr_all <- bind_rows(lr_rows)
  out_lr <- file.path(DATA_DIR, paste0(SAMPLE_ID, "_lr_summary.csv"))
  write_csv(lr_all, out_lr)
  message("Saved LR summary: ", out_lr, " (", nrow(lr_all), " rows)")
}

future::plan(sequential)
message("\nDone.")
