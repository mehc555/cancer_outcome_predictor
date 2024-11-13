# Methylation processor (R/modules/data_handling/methylation_processor.R)

library(tidyverse)

#' Standardize vector handling NA values
#' @param x Numeric vector potentially containing NA values
#' @return Standardized vector with NAs preserved in their original positions
#' @keywords internal
standardize_with_na <- function(x) {
  if(all(is.na(x))) return(x)
  non_na <- !is.na(x)
  x_std <- x
  x_std[non_na] <- scale(x[non_na])[,1]
  return(x_std)
}

#' Process methylation data for deep learning input
#' @param cancer_type Character string specifying cancer type
#' @param base_dir Base directory containing the data
#' @return Matrix of processed methylation values
#' @export
process_methylation_data <- function(cancer_type, base_dir = "data/GDC_TCGA") {
  # Construct file paths
  input_file <- file.path(base_dir, cancer_type, "methylation.tsv")
  processed_dir <- create_processed_dir(cancer_type, base_dir)
  
  if (!file.exists(input_file)) {
    stop(sprintf("Methylation data file not found for cancer type %s", cancer_type))
  }
  
  # Read methylation data
  message("Reading methylation data...")
  methylation <- read_tsv(input_file, show_col_types = FALSE)
  
  # Record initial dimensions
  initial_probes <- ncol(methylation) - 1  # Subtract 1 for probe ID column
  initial_samples <- nrow(methylation)
  
  # Remove probes with all NA values
  message("Removing probes with all NA values...")
  na_probes <- methylation %>%
    column_to_rownames("Composite Element REF") %>%
    apply(1, function(x) all(is.na(x)))
  
  methylation_filtered <- methylation %>%
    filter(!na_probes)
  
  # Transform beta values to M-values
  message("Converting beta values to M-values...")
  meth_matrix <- methylation_filtered %>%
    column_to_rownames("Composite Element REF") %>%
    mutate(across(everything(), 
                 ~log2(.x/(1-.x)),
                 .names = "{.col}")) %>%
    # Handle infinite values that may result from beta values of 0 or 1
    mutate(across(everything(), 
                 ~replace(., is.infinite(.), sign(.)[is.infinite(.)] * 10)))
  
  # Perform Z-score standardization on M-values
  message("Performing Z-score standardization...")
  # Apply standardization to each probe (row), handling NAs
  meth_matrix_scaled <- meth_matrix %>%
    apply(1, standardize_with_na) %>%
    t() %>%
    as.data.frame()
  
  # Validate standardization before any rounding or transposition
  message("Validating standardization...")
  # Function to compute statistics excluding NAs
  compute_stats <- function(x) {
    if(all(is.na(x))) return(c(mean = NA, sd = NA))
    non_na_vals <- x[!is.na(x)]
    return(c(mean = mean(non_na_vals), sd = sd(non_na_vals)))
  }
  
  # Compute statistics for first probe
  first_probe_stats <- compute_stats(as.numeric(meth_matrix_scaled[1,]))
  message(sprintf("Validation - First probe statistics after standardization:"))
  message(sprintf("- Mean: %.6f (should be close to 0)", first_probe_stats["mean"]))
  message(sprintf("- SD: %.6f (should be close to 1)", first_probe_stats["sd"]))
  
  # Validate all probes before any transformation
  probe_stats <- meth_matrix_scaled %>%
    apply(1, compute_stats)  # Use rows since not yet transposed
  
  probe_means <- probe_stats[1,]
  probe_sds <- probe_stats[2,]
  
  # Remove NAs for summary statistics
  valid_means <- probe_means[!is.na(probe_means)]
  valid_sds <- probe_sds[!is.na(probe_sds)]
  
  message("\nValidation - All probes statistics (excluding NAs):")
  message(sprintf("Mean of probe means: %.6f", mean(valid_means)))
  message(sprintf("SD of probe means: %.6f", sd(valid_means)))
  message(sprintf("Mean of probe SDs: %.6f", mean(valid_sds)))
  message(sprintf("Range of probe means: [%.6f, %.6f]", min(valid_means), max(valid_means)))
  
  # Count NAs before transformation
  probes_with_nas <- sum(apply(meth_matrix_scaled, 1, function(x) any(is.na(x))))
  message(sprintf("Probes with some NAs: %d", probes_with_nas))
  
  # Round to 2 decimal places
  message("Rounding values to 2 decimal places...")
  meth_matrix_rounded <- round(meth_matrix_scaled, 2)
  
  # Transpose the matrix and prepare for output
  message("Transposing matrix (samples as rows, CpGs as columns)...")
  final_matrix <- meth_matrix_rounded %>%
    rownames_to_column("probe_id") %>%
    gather(key = "sample", value = "value", -probe_id) %>%
    spread(key = probe_id, value = value)
  
  # Save processed data
  output_file <- file.path(processed_dir, "methylation_processed.tsv")
  write_tsv(final_matrix, output_file)
  
  # Print processing summary
  message(sprintf("\nMethylation processing summary for %s:", cancer_type))
  message(sprintf("- Initial number of CpG probes: %d", initial_probes))
  message(sprintf("- Probes removed (all NA): %d", sum(na_probes)))
  message(sprintf("- Final number of CpG probes: %d", ncol(final_matrix) - 1))  # -1 for sample column
  message(sprintf("- Number of samples: %d", nrow(final_matrix)))
  message("- Processing steps completed:")
  message("  1. Removal of probes with all NA values")
  message("  2. Beta to M-value transformation")
  message("  3. Z-score standardization (computed using non-NA values)")
  message("  4. Rounding to 2 decimal places")
  message("  5. Matrix transposition (samples as rows, CpGs as columns)")
  message(sprintf("- Output saved to: %s", output_file))
  
  return(final_matrix)
}
