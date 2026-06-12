#!/usr/bin/env Rscript

# Run SpatialCellChat on Xenium trajectory-state subsets exported by
# scripts/export_xenium_spatialcellchat_inputs.py.
#
# Example:
#   Rscript scripts/run_xenium_spatialcellchat.R \
#     --sample-id pdac_pancreas_v1 \
#     --input-dir data/xenium_pancreas_10x/spatialcellchat/input \
#     --output-dir data/xenium_pancreas_10x/spatialcellchat/results

suppressPackageStartupMessages({
  library(Matrix)
})

parse_cli_args <- function(args) {
  opt <- list(
    sample_id = "all",
    input_dir = "data/xenium_pancreas_10x/spatialcellchat/input",
    output_dir = "data/xenium_pancreas_10x/spatialcellchat/results",
    group_col = "Tier_A",
    state_col = "branch_time_state",
    min_cells_per_group = 20L,
    min_cells_per_state = 300L,
    max_cells_per_state = 12000L,
    interaction_range = 250,
    contact_range = 10,
    workers = 2L,
    force = FALSE
  )
  if (length(args) == 0) {
    return(opt)
  }
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (!startsWith(key, "--")) {
      stop("Unexpected positional argument: ", key)
    }
    key_name <- gsub("-", "_", sub("^--", "", key))
    if (key_name == "force") {
      opt[[key_name]] <- TRUE
      i <- i + 1
      next
    }
    if (i == length(args)) {
      stop("Missing value for argument: ", key)
    }
    value <- args[[i + 1]]
    if (!key_name %in% names(opt)) {
      stop("Unknown argument: ", key)
    }
    if (is.integer(opt[[key_name]])) {
      opt[[key_name]] <- as.integer(value)
    } else if (is.numeric(opt[[key_name]])) {
      opt[[key_name]] <- as.numeric(value)
    } else {
      opt[[key_name]] <- value
    }
    i <- i + 2
  }
  opt
}

opt <- parse_cli_args(commandArgs(trailingOnly = TRUE))

sample_ids <- c("pdac_pancreas_v1", "pdac_io_v1", "pdac_addon_v1", "normal_nondiseased_v1")
if (opt$sample_id != "all") {
  sample_ids <- strsplit(opt$sample_id, ",")[[1]]
}

if (!requireNamespace("SpatialCellChat", quietly = TRUE)) {
  stop(
    paste(
      "SpatialCellChat is not installed in this R library.",
      "Install with: devtools::install_github('jinworks/SpatialCellChat')"
    )
  )
}

suppressPackageStartupMessages(library(SpatialCellChat))

if (exists("setEnvironment", mode = "function")) {
  setEnvironment(workers = opt$workers, future.globals.maxSize = 100000 * 1024^2, conda_env = NULL)
}

read_sample_input <- function(sample_id) {
  sample_dir <- file.path(opt$input_dir, sample_id)
  metadata <- read.csv(gzfile(file.path(sample_dir, "metadata.csv.gz")), check.names = FALSE)
  genes <- read.csv(file.path(sample_dir, "genes.csv"), check.names = FALSE)
  cells <- read.csv(file.path(sample_dir, "cells.csv"), check.names = FALSE)
  mat <- readMM(gzfile(file.path(sample_dir, "expression_log1p_genes_by_cells.mtx.gz")))
  rownames(mat) <- genes$gene
  colnames(mat) <- cells$cell_id
  list(metadata = metadata, expression = mat)
}

safe_label <- function(x) {
  x <- gsub("[ /:]+", "_", x)
  x <- gsub("[^A-Za-z0-9_.-]", "_", x)
  x
}

downsample_state <- function(metadata, max_cells) {
  if (nrow(metadata) <= max_cells) {
    return(metadata)
  }
  split_idx <- split(seq_len(nrow(metadata)), metadata[[opt$group_col]])
  target_each <- max(1, ceiling(max_cells / max(1, length(split_idx))))
  keep <- unlist(lapply(split_idx, function(idx) {
    if (length(idx) <= target_each) {
      idx
    } else {
      sample(idx, target_each)
    }
  }), use.names = FALSE)
  metadata[sort(keep), , drop = FALSE]
}

run_one_state <- function(sample_id, state_name, metadata, expression, out_dir) {
  state_label <- ifelse(is.na(state_name), "sample_all", safe_label(state_name))
  output_csv <- file.path(out_dir, paste0(sample_id, "__", state_label, "__communication_lr.csv"))
  output_rds <- file.path(out_dir, paste0(sample_id, "__", state_label, "__cellchat.rds"))
  if (file.exists(output_csv) && !opt$force) {
    message("Using cached SpatialCellChat result: ", output_csv)
    return(NULL)
  }

  metadata <- metadata[!is.na(metadata[[opt$group_col]]) & metadata[[opt$group_col]] != "", , drop = FALSE]
  group_counts <- table(metadata[[opt$group_col]])
  keep_groups <- names(group_counts[group_counts >= opt$min_cells_per_group])
  metadata <- metadata[metadata[[opt$group_col]] %in% keep_groups, , drop = FALSE]
  if (nrow(metadata) < opt$min_cells_per_state || length(keep_groups) < 2) {
    message("Skipping ", sample_id, " / ", state_label, ": not enough cells/groups")
    return(NULL)
  }

  set.seed(42)
  metadata <- downsample_state(metadata, opt$max_cells_per_state)
  expression <- expression[, metadata$cell_id, drop = FALSE]
  coordinates <- metadata[, c("x", "y"), drop = FALSE]
  rownames(coordinates) <- metadata$cell_id
  rownames(metadata) <- metadata$cell_id

  spatial.factors <- list(ratio = 1, tol = opt$contact_range / 2)
  chat <- createSpatialCellChat(
    object = expression,
    meta = metadata,
    group.by = opt$group_col,
    datatype = "spatial",
    coordinates = coordinates,
    spatial.factors = spatial.factors
  )

  chat@DB <- CellChatDB.human
  chat <- subsetData(chat)
  chat <- identifyOverExpressedGenes(chat)
  chat <- identifyOverExpressedInteractions(chat)
  chat <- computeCommunProb(
    chat,
    type = "truncatedMean",
    trim = 0.1,
    distance.use = TRUE,
    interaction.range = opt$interaction_range,
    contact.dependent = TRUE,
    contact.range = opt$contact_range
  )
  chat <- filterCommunication(chat, min.cells = opt$min_cells_per_group)
  chat <- computeCommunProbPathway(chat)
  chat <- aggregateNet(chat)

  lr_df <- subsetCommunication(chat)
  if (nrow(lr_df) > 0) {
    lr_df$sample_id <- sample_id
    lr_df$branch_time_state <- ifelse(is.na(state_name), NA, state_name)
    lr_df$state_label <- state_label
    lr_df$n_cells_used <- nrow(metadata)
    write.csv(lr_df, output_csv, row.names = FALSE)
  } else {
    write.csv(data.frame(), output_csv, row.names = FALSE)
  }
  saveRDS(chat, output_rds)
  message("Saved ", output_csv)
  invisible(lr_df)
}

dir.create(opt$output_dir, recursive = TRUE, showWarnings = FALSE)

for (sample_id in sample_ids) {
  message("Reading ", sample_id)
  obj <- read_sample_input(sample_id)
  metadata <- obj$metadata
  expression <- obj$expression
  metadata <- metadata[metadata$cell_id %in% colnames(expression), , drop = FALSE]

  out_dir <- file.path(opt$output_dir, sample_id)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  run_one_state(sample_id, NA, metadata, expression, out_dir)
  state_values <- sort(unique(metadata[[opt$state_col]][!is.na(metadata[[opt$state_col]])]))
  for (state_name in state_values) {
    state_meta <- metadata[metadata[[opt$state_col]] == state_name, , drop = FALSE]
    run_one_state(sample_id, state_name, state_meta, expression, out_dir)
  }
}
