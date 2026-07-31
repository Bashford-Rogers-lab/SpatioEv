#!/usr/bin/env Rscript
#
# extract_cellchat_improved.R
#
# Extracts LR and pathway communication tables from *_improved.rds objects.
# Uses multiple fallback strategies because different SpatialCellChat versions
# store results in different slots.
#
# Output files (all written to DATA_DIR):
#   {sample}_lr_summary_improved.csv
#   {sample}_pathway_summary_improved.csv
#   lr_interaction_summary_improved.csv   (combined)
#   pathway_summary_improved.csv          (combined)
#
# Usage (from project root):
#   Rscript scripts/extract_cellchat_improved.R
#

suppressPackageStartupMessages({
  library(SpatialCellChat)
  library(dplyr)
  library(tibble)
  library(readr)
})

DATA_DIR <- "data/xenium_pancreas_10x/spatialcellchat"

ALL_SAMPLES <- c("normal_nondiseased_v1", "pdac_addon_v1",
                 "pdac_pancreas_v1",      "pdac_io_v1")
ALL_BINS    <- c("bin1", "bin2", "bin3", "bin4", "bin5")

# ── Strategy 1: subsetCommunication (standard API) ───────────────────────────
try_subset_communication <- function(chat, slot.name = "net") {
  df <- tryCatch(
    subsetCommunication(chat, slot.name = slot.name),
    error = function(e) {
      # Some versions don't have slot.name argument
      tryCatch(subsetCommunication(chat), error = function(e2) NULL)
    }
  )
  if (is.null(df) || nrow(df) == 0) return(tibble())
  as_tibble(df)
}

# ── Strategy 2: Read @LR$LRsig directly (populated by filterCommunication) ───
try_lr_slot <- function(chat) {
  tryCatch({
    df <- chat@LR$LRsig
    if (is.null(df) || nrow(df) == 0) return(tibble())
    as_tibble(df)
  }, error = function(e) tibble())
}

# ── Strategy 3: Flatten 3-D @net$prob array manually ─────────────────────────
try_net_prob <- function(chat) {
  tryCatch({
    prob <- chat@net$prob
    pval <- chat@net$pval
    if (is.null(prob) || length(dim(prob)) != 3) return(tibble())

    dn  <- dimnames(prob)
    cat_names  <- dn[[1]]   # source cell types
    lr_names   <- dn[[3]]   # LR pair names (if set)
    if (is.null(lr_names)) lr_names <- seq_len(dim(prob)[3])

    # Vectorised extraction of non-zero entries
    idx <- which(prob > 0, arr.ind = TRUE)
    if (nrow(idx) == 0) return(tibble())

    df <- tibble(
      source           = cat_names[idx[,1]],
      target           = cat_names[idx[,2]],
      interaction_name = lr_names[idx[,3]],
      prob             = prob[idx],
      pval             = if (!is.null(pval)) pval[idx] else NA_real_
    )

    # Annotate from DB if available
    if (!is.null(chat@DB) && !is.null(chat@DB$interaction)) {
      db_int  <- chat@DB$interaction
      matched <- match(df$interaction_name, rownames(db_int))
      df$pathway_name <- db_int$pathway_name[matched]
      df$annotation   <- db_int$annotation[matched]
    }
    df
  }, error = function(e) {
    message("    Strategy 3 (net$prob) failed: ", e$message)
    tibble()
  })
}

# ── Diagnostic: print slot summary for one object ─────────────────────────────
print_slot_summary <- function(chat, label) {
  cat("  [", label, "]\n")
  cat("    class:", class(chat), "\n")

  # @net
  cat("    @net  names:", paste(names(chat@net),  collapse=", "), "\n")
  if (!is.null(chat@net$prob))
    cat("    @net$prob dim:", paste(dim(chat@net$prob), collapse=" x "),
        "  non-zero:", sum(chat@net$prob > 0, na.rm=TRUE), "\n")
  if (!is.null(chat@net$df))
    cat("    @net$df rows:", nrow(chat@net$df), "\n")

  # @netP
  cat("    @netP names:", paste(names(chat@netP), collapse=", "), "\n")
  if (!is.null(chat@netP$prob))
    cat("    @netP$prob dim:", paste(dim(chat@netP$prob), collapse=" x "), "\n")

  # @LR
  cat("    @LR  names:", paste(names(chat@LR), collapse=", "), "\n")
  if (!is.null(chat@LR$LRsig))
    cat("    @LR$LRsig rows:", nrow(chat@LR$LRsig), "\n")
}

# ── Main loop ─────────────────────────────────────────────────────────────────
all_lr_rows      <- list()
all_pathway_rows <- list()

for (sample_id in ALL_SAMPLES) {
  cat("\n══ Sample:", sample_id, "\n")
  lr_rows_s  <- list()
  pw_rows_s  <- list()

  for (bin_name in ALL_BINS) {
    rds_path <- file.path(DATA_DIR,
      paste0("cellchat_", sample_id, "_", bin_name, "_improved.rds"))
    if (!file.exists(rds_path)) {
      cat("  [skip] missing:", basename(rds_path), "\n")
      next
    }

    chat <- tryCatch(readRDS(rds_path),
                     error = function(e) { cat("  ERROR loading:", e$message, "\n"); NULL })
    if (is.null(chat)) next

    print_slot_summary(chat, paste0(sample_id, "/", bin_name))

    # ── LR table (try in order) ───────────────────────────────────────────────
    lr_df <- try_subset_communication(chat, "net")
    if (nrow(lr_df) == 0) {
      cat("    subsetCommunication empty → trying @LR$LRsig\n")
      lr_df <- try_lr_slot(chat)
    }
    if (nrow(lr_df) == 0) {
      cat("    @LR$LRsig empty → trying @net$prob flatten\n")
      lr_df <- try_net_prob(chat)
    }

    if (nrow(lr_df) > 0) {
      lr_df$sample_id <- sample_id
      lr_df$pt_bin    <- bin_name
      lr_rows_s[[bin_name]] <- lr_df
      cat("    LR rows:", nrow(lr_df),
          " pathways:", length(unique(lr_df$pathway_name)), "\n")
    } else {
      cat("    !! No LR pairs extractable from this bin\n")
    }

    # ── Pathway table ─────────────────────────────────────────────────────────
    pw_df <- try_subset_communication(chat, "netP")
    if (nrow(pw_df) > 0) {
      pw_df$sample_id <- sample_id
      pw_df$pt_bin    <- bin_name
      pw_rows_s[[bin_name]] <- pw_df
    }

    rm(chat); gc(verbose=FALSE)
  }

  # Per-sample CSV
  if (length(lr_rows_s) > 0) {
    lr_s <- bind_rows(lr_rows_s)
    out  <- file.path(DATA_DIR, paste0(sample_id, "_lr_summary_improved.csv"))
    write_csv(lr_s, out)
    cat("  Saved:", basename(out), "(", nrow(lr_s), "rows)\n")
    all_lr_rows[[sample_id]] <- lr_s
  }
  if (length(pw_rows_s) > 0) {
    pw_s <- bind_rows(pw_rows_s)
    out  <- file.path(DATA_DIR, paste0(sample_id, "_pathway_summary_improved.csv"))
    write_csv(pw_s, out)
    cat("  Saved:", basename(out), "(", nrow(pw_s), "rows)\n")
    all_pathway_rows[[sample_id]] <- pw_s
  }
}

# Combined CSVs
if (length(all_lr_rows) > 0) {
  combined_lr <- bind_rows(all_lr_rows)
  out <- file.path(DATA_DIR, "lr_interaction_summary_improved.csv")
  write_csv(combined_lr, out)
  cat("\nCombined LR:", nrow(combined_lr), "rows\n")
  print(combined_lr %>% count(sample_id, pt_bin))
} else {
  cat("\n!! No LR rows extracted from any bin.\n")
  cat("   Check the slot diagnostics above.\n")
  cat("   The @net$prob fallback requires non-zero entries after filterCommunication.\n")
}

if (length(all_pathway_rows) > 0) {
  combined_pw <- bind_rows(all_pathway_rows)
  out <- file.path(DATA_DIR, "pathway_summary_improved.csv")
  write_csv(combined_pw, out)
  cat("Combined pathways:", nrow(combined_pw), "rows\n")
}

cat("\nDone.\n")
