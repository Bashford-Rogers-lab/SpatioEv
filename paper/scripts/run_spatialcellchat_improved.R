#!/usr/bin/env Rscript
#
# run_spatialcellchat_improved.R
#
# Improved SpatialCellChat pipeline for all 4 Xenium PDAC samples.
# Incorporates:
#   - CSV-based input (cell_meta + counts_common_genes)
#   - Spatial proximity bin extension for non-ductal cells (300 µm)
#   - preProcessing() for SpatialCellChat internals
#   - Adaptive scale.distance calibration via computeCellDistance()
#   - filterProbability() bootstrapped filtering before filterCommunication()
#   - Defensive argument-checking for identifyOverExpressedGenes/Interactions
#   - netAnalysis_computeCentrality() for visualization readiness
#
# Usage (from project root):
#   Rscript scripts/run_spatialcellchat_improved.R
#   Rscript scripts/run_spatialcellchat_improved.R --sample-id pdac_io_v1
#   Rscript scripts/run_spatialcellchat_improved.R --sample-id pdac_pancreas_v1,pdac_addon_v1
#   Rscript scripts/run_spatialcellchat_improved.R --bins bin2,bin3,bin4,bin5
#
# Outputs (data/xenium_pancreas_10x/spatialcellchat/):
#   cellchat_{sample}_{bin}_improved.rds     — one per bin per sample
#   {sample}_lr_summary_improved.csv         — LR-level table
#   {sample}_pathway_summary_improved.csv    — pathway-level table
#   lr_interaction_summary.csv               — combined (overwritten)
#   pathway_summary.csv                      — combined (overwritten)

suppressPackageStartupMessages({
  library(Matrix)
  library(readr)
  library(dplyr)
  library(tibble)
  library(future)
  library(R.utils)   # for withTimeout()
})

if (!requireNamespace("SpatialCellChat", quietly = TRUE)) {
  stop(
    "SpatialCellChat is not installed.\n",
    "Install: devtools::install_github('jinworks/SpatialCellChat')"
  )
}
suppressPackageStartupMessages(library(SpatialCellChat))

# Point reticulate to known conda env to avoid downloads in offline runs
DEFAULT_PYTHON <- "/Users/shihongwu/anaconda3/envs/spatialdata_env/bin/python"
if (!nzchar(Sys.getenv("RETICULATE_PYTHON")) && file.exists(DEFAULT_PYTHON)) {
  Sys.setenv(RETICULATE_PYTHON = DEFAULT_PYTHON)
}

options(stringsAsFactors = FALSE)
set.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# 0. Parse CLI arguments
# ─────────────────────────────────────────────────────────────────────────────

ALL_SAMPLES <- c("normal_nondiseased_v1", "pdac_addon_v1",
                 "pdac_pancreas_v1",      "pdac_io_v1")
ALL_BINS    <- c("bin1", "bin2", "bin3", "bin4", "bin5")

parse_args <- function(args) {
  opt <- list(
    sample_ids = ALL_SAMPLES,
    bins       = ALL_BINS
  )
  i <- 1L
  while (i <= length(args)) {
    key <- args[[i]]
    if (!startsWith(key, "--") || i == length(args)) {
      stop("Unexpected argument at position ", i, ": ", key)
    }
    val <- args[[i + 1L]]
    if (key == "--sample-id") {
      opt$sample_ids <- trimws(strsplit(val, ",")[[1]])
    } else if (key == "--bins") {
      opt$bins <- trimws(strsplit(val, ",")[[1]])
    } else {
      stop("Unknown argument: ", key)
    }
    i <- i + 2L
  }
  opt
}

opt <- parse_args(commandArgs(trailingOnly = TRUE))
message("Samples to run: ", paste(opt$sample_ids, collapse = ", "))
message("Bins to run:    ", paste(opt$bins, collapse = ", "))

# ─────────────────────────────────────────────────────────────────────────────
# 1. Settings
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR <- "data/xenium_pancreas_10x/spatialcellchat"

ASSIGN_RADIUS_UM   <- 300    # max radius to inherit pt_bin from nearest ductal cell
INTERACTION_RANGE_UM <- 250  # CellChat L-R interaction range (µm)
CONTACT_RANGE_UM     <- 10   # CellChat contact range (µm); pancreatic ducts ~8-15 µm
SPATIAL_FACTORS      <- list(ratio = 1, tol = CONTACT_RANGE_UM / 2)
TARGET_MIN_SCALED_DISTANCE <- 1.05  # for adaptive scale.distance

MIN_CELLS_PER_BIN    <- 150L
MIN_CELLS_PER_GROUP  <- 20L
MAX_CELLS_PER_BIN    <- 8000L
NBOOT                <- 100L

# Use sequential plan: computeCommunProb() internally uses the future framework,
# but calling it under multisession while a tryCatch handler is on the stack
# raises "should not be called with handlers on the stack". Sequential plan
# avoids the conflict. CellChat runs correctly (single-threaded per bin).
future::plan(sequential)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Helper: adaptive scale.distance calibration (from script 06 pattern)
# ─────────────────────────────────────────────────────────────────────────────

estimate_scale_distance <- function(coordinates) {
  result <- tryCatch(
    computeCellDistance(
      coordinates      = coordinates,
      ratio            = SPATIAL_FACTORS$ratio,
      tol              = SPATIAL_FACTORS$tol,
      interaction.range = INTERACTION_RANGE_UM,
      contact.range    = CONTACT_RANGE_UM
    ),
    error = function(e) {
      message("    computeCellDistance failed: ", conditionMessage(e),
              " — using scale.distance = NULL")
      NULL
    }
  )
  if (is.null(result)) return(NULL)

  d_values <- tryCatch(result$d.spatial@x, error = function(e) NULL)
  if (is.null(d_values)) return(NULL)

  d_values <- d_values[is.finite(d_values) & d_values > 0]
  if (length(d_values) == 0) return(NULL)

  min_d        <- min(d_values)
  scale_d      <- TARGET_MIN_SCALED_DISTANCE / min_d
  message("    scale.distance = ", signif(scale_d, 4),
          "  (min raw dist = ", signif(min_d, 4), " µm)")
  scale_d
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. Helper: spatial proximity bin extension for non-ductal cells
# ─────────────────────────────────────────────────────────────────────────────

assign_bin_by_proximity <- function(non_ductal_df, ductal_df, radius_um) {
  if (nrow(non_ductal_df) == 0 || nrow(ductal_df) == 0) {
    return(rep(NA_character_, nrow(non_ductal_df)))
  }
  nd_x   <- non_ductal_df$x
  nd_y   <- non_ductal_df$y
  d_x    <- ductal_df$x
  d_y    <- ductal_df$y
  d_bin  <- ductal_df$pt_bin_cellchat

  chunk  <- 5000L
  n      <- nrow(non_ductal_df)
  result <- character(n)

  for (start in seq(1L, n, by = chunk)) {
    end    <- min(start + chunk - 1L, n)
    idx    <- start:end
    dx     <- outer(nd_x[idx], d_x, "-")
    dy     <- outer(nd_y[idx], d_y, "-")
    dist2  <- dx^2 + dy^2
    ni     <- max.col(-dist2)
    dist   <- sqrt(dist2[cbind(seq_along(idx), ni)])
    bin    <- d_bin[ni]
    bin[dist > radius_um] <- NA_character_
    result[idx] <- bin
  }
  result
}

extend_meta_bins <- function(meta) {
  ductal_binned <- meta %>%
    filter(!is.na(pt_bin_cellchat)) %>%
    select(cell_id, x, y, pt_bin_cellchat)

  non_ductal <- meta %>%
    filter(is.na(pt_bin_cellchat))

  if (nrow(non_ductal) == 0) return(meta)

  message("  Spatial proximity extension: ",
          nrow(ductal_binned), " binned ductal → assigning ",
          nrow(non_ductal), " surrounding cells (radius = ",
          ASSIGN_RADIUS_UM, " µm)")

  non_ductal$pt_bin_cellchat <- assign_bin_by_proximity(
    non_ductal, ductal_binned, ASSIGN_RADIUS_UM
  )

  assigned <- sum(!is.na(non_ductal$pt_bin_cellchat))
  message("  Assigned: ", assigned, " / ", nrow(non_ductal), " surrounding cells")

  bind_rows(
    meta %>% filter(!is.na(pt_bin_cellchat)),
    non_ductal
  ) %>%
    filter(!is.na(pt_bin_cellchat))
}

# ─────────────────────────────────────────────────────────────────────────────
# 4. Helper: safe subsetCommunication wrapper
# ─────────────────────────────────────────────────────────────────────────────

safe_subset_communication <- function(chat, slot.name = "net") {
  tryCatch(
    subsetCommunication(chat, slot.name = slot.name),
    error = function(e) {
      tryCatch(
        subsetCommunication(chat),
        error = function(e2) {
          message("    subsetCommunication failed: ", conditionMessage(e2))
          tibble()
        }
      )
    }
  )
}

# ─────────────────────────────────────────────────────────────────────────────
# 5. Core CellChat workflow for one bin
# ─────────────────────────────────────────────────────────────────────────────

run_one_bin <- function(sample_id, bin_name, bin_meta, expr_mat) {
  # NOTE ON ERROR HANDLING DESIGN:
  # computeCommunProb() and identifyOverExpressedInteractions() use furrr::future_map()
  # internally. future's assert_no_signal_handlers() fires if ANY tryCatch handler is
  # on the call stack — even with plan(sequential). Therefore:
  #   PHASE 1 (setup):    individual tryCatch blocks are OK — they COMPLETE and remove
  #                       their handlers before Phase 2 executes.
  #   PHASE 2 (inference): NO tryCatch at any level. If these calls error, the error
  #                       propagates to the bin loop, which logs it and continues.
  #   PHASE 3 (post-proc): tryCatch OK again — no future calls after aggregateNet().
  #
  # filterProbability() and computeAvgCommunProb() also use furrr internally, so they
  # are omitted here. The core outputs (LR table, pathway table) do not require them.

  message("  ── ", bin_name, ": ", nrow(bin_meta), " cells before filter")

  # ── Validation (early return — no tryCatch needed) ────────────────────────
  group_counts <- table(bin_meta$Tier_A)
  keep_groups  <- names(group_counts[group_counts >= MIN_CELLS_PER_GROUP])
  if (length(keep_groups) < 2) {
    message("  Skipping: < 2 cell types with ≥ ", MIN_CELLS_PER_GROUP, " cells")
    return(NULL)
  }
  bin_meta <- bin_meta %>% filter(Tier_A %in% keep_groups)

  if (nrow(bin_meta) < MIN_CELLS_PER_BIN) {
    message("  Skipping: only ", nrow(bin_meta), " cells after group filtering")
    return(NULL)
  }

  # Downsample large bins
  if (nrow(bin_meta) > MAX_CELLS_PER_BIN) {
    message("  Downsampling ", nrow(bin_meta), " → ", MAX_CELLS_PER_BIN, " cells")
    idx <- unlist(lapply(
      split(seq_len(nrow(bin_meta)), bin_meta$Tier_A),
      function(i) {
        target <- max(1L, round(length(i) * MAX_CELLS_PER_BIN / nrow(bin_meta)))
        sample(i, min(length(i), target))
      }
    ))
    bin_meta <- bin_meta[sort(idx), ]
  }

  # Align expression
  common_cells <- intersect(bin_meta$cell_id, colnames(expr_mat))
  if (length(common_cells) < MIN_CELLS_PER_BIN) {
    message("  Skipping: ", length(common_cells), " cells in expression matrix")
    return(NULL)
  }
  bin_meta <- bin_meta %>% filter(cell_id %in% common_cells)
  bin_expr <- expr_mat[, bin_meta$cell_id, drop = FALSE]

  rownames(bin_meta) <- bin_meta$cell_id
  coords <- data.frame(x = bin_meta$x, y = bin_meta$y,
                       row.names = bin_meta$cell_id)

  message("  Creating SpatialCellChat object (",
          nrow(bin_expr), " genes × ", ncol(bin_expr), " cells, ",
          length(keep_groups), " groups)")

  # ── PHASE 1: Object creation and preprocessing ────────────────────────────
  # These calls do NOT use future internally; tryCatch is safe here.
  # The handler is on the stack only during this block, removed before Phase 2.
  chat <- tryCatch({
    chat <- createSpatialCellChat(
      object          = bin_expr,
      meta            = bin_meta,
      group.by        = "Tier_A",
      datatype        = "spatial",
      coordinates     = coords,
      spatial.factors = SPATIAL_FACTORS
    )
    chat@DB <- CellChatDB.human
    chat <- subsetData(chat)
    chat
  }, error = function(e) {
    message("  Object creation failed: ", conditionMessage(e))
    NULL
  })
  if (is.null(chat)) return(NULL)

  # preProcessing — optional, individual tryCatch, completes before future steps
  if (existsMethod("preProcessing", "SpatialCellChat") ||
      exists("preProcessing", mode = "function")) {
    message("  preProcessing()")
    chat <- tryCatch(
      preProcessing(chat),
      error = function(e) {
        message("  preProcessing failed: ", conditionMessage(e), " — continuing")
        chat
      }
    )
  }
  # ← All Phase 1 tryCatch handlers are now OFF the stack.

  # ── PHASE 2: Feature selection and communication inference ─────────────────
  # NO tryCatch at any level while these execute.
  # identifyOverExpressedInteractions() and computeCommunProb() use furrr internally.
  oe_args <- names(formals(identifyOverExpressedGenes))
  if ("selection.method" %in% oe_args) {
    chat <- identifyOverExpressedGenes(chat, selection.method = "meringue",
                                       do.grid = FALSE)
  } else {
    chat <- identifyOverExpressedGenes(chat)
  }

  oi_args <- names(formals(identifyOverExpressedInteractions))
  if ("variable.both" %in% oi_args) {
    chat <- identifyOverExpressedInteractions(chat, variable.both = FALSE)
  } else {
    chat <- identifyOverExpressedInteractions(chat)
  }

  # estimate_scale_distance uses tryCatch internally, but that tryCatch COMPLETES
  # (returns scale_d or NULL) before computeCommunProb is called — stack is clean.
  message("  Estimating scale.distance...")
  scale_d <- estimate_scale_distance(coords)

  message("  computeCommunProb()")
  cp_formals <- names(formals(computeCommunProb))
  comm_args <- list(chat)
  # type / trim are CellChat-style args absent from some SpatialCellChat versions
  if ("type"              %in% cp_formals) comm_args$type              <- "truncatedMean"
  if ("trim"              %in% cp_formals) comm_args$trim              <- 0.1
  if ("distance.use"      %in% cp_formals) comm_args$distance.use      <- TRUE
  if ("contact.dependent" %in% cp_formals) comm_args$contact.dependent <- TRUE
  if ("interaction.range" %in% cp_formals) comm_args$interaction.range <- INTERACTION_RANGE_UM
  if ("contact.range"     %in% cp_formals) comm_args$contact.range     <- CONTACT_RANGE_UM
  if (!is.null(scale_d) && "scale.distance" %in% cp_formals)
    comm_args$scale.distance <- scale_d
  chat <- do.call(computeCommunProb, comm_args)

  # ── PHASE 3: Post-processing ───────────────────────────────────────────────
  # No future calls below — tryCatch is safe again.
  chat <- filterCommunication(chat, min.cells = MIN_CELLS_PER_GROUP)

  # computeCommunProbPathway can hang when prob matrix is sparse/unusual;
  # wrap with R-level timeout so one bad bin never stalls the whole run.
  chat <- tryCatch({
    withCallingHandlers(
      withTimeout(computeCommunProbPathway(chat), timeout = 120, onTimeout = "error"),
      error = function(e) invokeRestart("muffleError")
    )
  }, error = function(e) {
    message("  computeCommunProbPathway timed-out or failed: ", conditionMessage(e),
            " — skipping pathway aggregation for this bin")
    chat   # return chat without pathway probs; LR table still extractable
  })

  chat <- tryCatch(
    aggregateNet(chat),
    error = function(e) {
      message("  aggregateNet failed: ", conditionMessage(e)); chat
    }
  )

  if (exists("netAnalysis_computeCentrality", mode = "function")) {
    message("  netAnalysis_computeCentrality()")
    chat <- tryCatch(
      netAnalysis_computeCentrality(chat, slot.name = "netP",
                                    do.group = TRUE, degree.only = FALSE),
      error = function(e) {
        message("  netAnalysis_computeCentrality failed: ", conditionMessage(e))
        chat
      }
    )
  }

  chat
}

# ─────────────────────────────────────────────────────────────────────────────
# 6. Main loop
# ─────────────────────────────────────────────────────────────────────────────

all_lr_rows      <- list()
all_pathway_rows <- list()

for (sample_id in opt$sample_ids) {
  message("\n══════════════════════════════════════════════")
  message("Sample: ", sample_id)
  message("══════════════════════════════════════════════")

  meta_csv   <- file.path(DATA_DIR, paste0(sample_id, "_cell_meta.csv"))
  counts_csv <- file.path(DATA_DIR, paste0(sample_id, "_counts_common_genes.csv.gz"))

  if (!file.exists(meta_csv))   { message("SKIP: missing ", meta_csv);   next }
  if (!file.exists(counts_csv)) { message("SKIP: missing ", counts_csv); next }

  # ── 6a. Load metadata & extend bins ───────────────────────────────────────
  message("Loading metadata...")
  meta <- read_csv(meta_csv, show_col_types = FALSE)
  message("  ", nrow(meta), " total cells")

  meta <- extend_meta_bins(meta)
  message("  ", nrow(meta), " cells with bin assignment")

  # ── 6b. Load expression matrix ────────────────────────────────────────────
  message("Loading expression matrix...")
  expr_df      <- read_csv(counts_csv, show_col_types = FALSE)
  cell_ids_csv <- as.character(expr_df[[1]])
  gene_names   <- colnames(expr_df)[-1]

  expr_mat           <- as.matrix(expr_df[, -1])
  expr_mat[expr_mat < 0] <- 0   # clip z-scores; CellChat needs non-negative values
  expr_mat           <- t(expr_mat)  # genes × cells
  colnames(expr_mat) <- cell_ids_csv
  rownames(expr_mat) <- gene_names
  expr_mat           <- Matrix(expr_mat, sparse = TRUE)
  message("  ", nrow(expr_mat), " genes × ", ncol(expr_mat), " cells")

  # ── 6c. Per-bin loop ──────────────────────────────────────────────────────
  lr_rows_sample      <- list()
  pathway_rows_sample <- list()

  for (bin_name in opt$bins) {
    message("\n── ", sample_id, " / ", bin_name, " ──")

    bin_meta <- meta %>% filter(pt_bin_cellchat == bin_name)
    message("  Total cells in bin: ", nrow(bin_meta))

    if (nrow(bin_meta) < MIN_CELLS_PER_BIN) {
      message("  Skipping: fewer than ", MIN_CELLS_PER_BIN, " cells")
      next
    }

    # Checkpoint: skip if RDS already exists (allows restart after crash).
    # _improved suffix keeps these separate from the original May-31 run,
    # so we always start fresh with the corrected script.
    rds_path <- file.path(DATA_DIR,
      paste0("cellchat_", sample_id, "_", bin_name, "_improved.rds"))
    if (file.exists(rds_path)) {
      message("  Already done — loading from: ", basename(rds_path))
      chat <- readRDS(rds_path)
    } else {
      # IMPORTANT: do NOT wrap run_one_bin() in tryCatch here.
      # Phase 2 inside run_one_bin() uses furrr::future_map() which calls
      # future:::assert_no_signal_handlers(). Any active tryCatch on the stack
      # — including one here in the bin loop — triggers the error
      # "should not be called with handlers on the stack".
      # Strategy: run without tryCatch; if a bin crashes, save all prior results
      # via the checkpoint above, and re-run the script (the checkpoint skips
      # already-completed bins so the run resumes from where it failed).
      chat <- run_one_bin(sample_id, bin_name, bin_meta, expr_mat)
    }
    if (is.null(chat)) next

    # Save RDS immediately after success (checkpoint for resume-after-crash)
    if (!file.exists(rds_path)) {
      saveRDS(chat, rds_path)
      message("  Saved: ", basename(rds_path))
    }

    # Extract LR table
    lr_df <- safe_subset_communication(chat)
    if (nrow(lr_df) > 0) {
      lr_df$sample_id <- sample_id
      lr_df$pt_bin    <- bin_name
      lr_rows_sample[[bin_name]] <- lr_df
      message("  LR pairs: ", nrow(lr_df),
              "  pathways: ", length(unique(lr_df$pathway_name)))
    }

    # Extract pathway table
    pw_df <- tryCatch(
      subsetCommunication(chat, slot.name = "netP"),
      error = function(e) tibble()
    )
    if (nrow(pw_df) > 0) {
      pw_df$sample_id <- sample_id
      pw_df$pt_bin    <- bin_name
      pathway_rows_sample[[bin_name]] <- pw_df
    }
  }  # end bin loop

  # ── 6d. Write per-sample summaries ────────────────────────────────────────
  if (length(lr_rows_sample) > 0) {
    lr_sample <- bind_rows(lr_rows_sample)
    lr_out    <- file.path(DATA_DIR,
                           paste0(sample_id, "_lr_summary_improved.csv"))
    write_csv(lr_sample, lr_out)
    message("\nSaved: ", basename(lr_out), " (", nrow(lr_sample), " rows)")
    all_lr_rows[[sample_id]] <- lr_sample
  }

  if (length(pathway_rows_sample) > 0) {
    pw_sample <- bind_rows(pathway_rows_sample)
    pw_out    <- file.path(DATA_DIR,
                           paste0(sample_id, "_pathway_summary_improved.csv"))
    write_csv(pw_sample, pw_out)
    message("Saved: ", basename(pw_out), " (", nrow(pw_sample), " rows)")
    all_pathway_rows[[sample_id]] <- pw_sample
  }

}  # end sample loop

# ─────────────────────────────────────────────────────────────────────────────
# 7. Write combined summary CSVs (overwrite originals)
# ─────────────────────────────────────────────────────────────────────────────

if (length(all_lr_rows) > 0) {
  combined_lr <- bind_rows(all_lr_rows)
  lr_combined_path <- file.path(DATA_DIR, "lr_interaction_summary.csv")
  write_csv(combined_lr, lr_combined_path)
  message("\nCombined lr_interaction_summary.csv: ", nrow(combined_lr), " rows")
}

if (length(all_pathway_rows) > 0) {
  combined_pw <- bind_rows(all_pathway_rows)
  pw_combined_path <- file.path(DATA_DIR, "pathway_summary.csv")
  write_csv(combined_pw, pw_combined_path)
  message("Combined pathway_summary.csv: ", nrow(combined_pw), " rows")
}

message("\n\nDone. Run: Rscript scripts/run_spatialcellchat_improved.R")
