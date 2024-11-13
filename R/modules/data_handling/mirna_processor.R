# miRNA processor (R/modules/data_handling/mirna_processor.R)

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

#' Process miRNA expression data for deep learning input
#' @param cancer_type Character string specifying cancer type
#' @param min_expression Minimum expression threshold (default: 1)
#' @param min_samples Minimum number of samples a miRNA must be expressed in (default: 3)
#' @param base_dir Base directory containing the data
#' @return Matrix of processed miRNA expression values
#' @export
process_mirna_data <- function(cancer_type, 
                             min_expression = 1, 
                             min_samples = 3,
                             base_dir = "data/GDC_TCGA") {
  # Construct file paths
  input_file <- file.path(base_dir, cancer_type, "mirna.tsv")
  processed_dir <- create_processed_dir(cancer_type, base_dir)
  
  if (!file.exists(input_file)) {
    stop(sprintf("miRNA data file not found for cancer type %s", cancer_type))
  }
  
  # Read miRNA data
  message("Reading miRNA expression data...")
  mirna <- read.csv(input_file, row.names=1, sep="\t", check.names=F)
  
  # Record initial dimensions
  initial_mirnas <- nrow(mirna)
  initial_samples <- ncol(mirna) - 1  # Subtract miRNA_ID column
  
  # Extract miRNA names and expression data
  message("Extracting miRNA names and expression data...")
  mirna_names <- mirna$miRNA_ID
  expression_data <- mirna[, -1, drop=FALSE]  # Remove miRNA_ID column
  
  # Filter low-expression miRNAs
  message("Filtering low-expression miRNAs...")
  expressed_samples <- rowSums(expression_data >= min_expression, na.rm = TRUE)
  keep_mirnas <- expressed_samples >= min_samples
  
  # Filter the expression data
  expression_filtered <- expression_data[keep_mirnas, , drop = FALSE]
  mirna_names_filtered <- mirna_names[keep_mirnas]
  
  # Perform Z-score standardization
  message("Performing Z-score standardization...")
  expression_scaled <- t(apply(expression_filtered, 1, standardize_with_na))
  rownames(expression_scaled) <- mirna_names_filtered
  
  # Validate standardization
  message("Validating standardization...")
  first_mirna_stats <- compute_stats(expression_scaled[1,])
  message(sprintf("Validation - First miRNA statistics after standardization:"))
  message(sprintf("- Mean: %.6f (should be close to 0)", first_mirna_stats["mean"]))
  message(sprintf("- SD: %.6f (should be close to 1)", first_mirna_stats["sd"]))
  
  # Validate all miRNAs
  mirna_stats <- apply(expression_scaled, 1, compute_stats)
  
  message("\nValidation - All miRNAs statistics:")
  message(sprintf("Mean of miRNA means: %.6f", mean(mirna_stats[1,])))
  message(sprintf("SD of miRNA means: %.6f", sd(mirna_stats[1,])))
  message(sprintf("Mean of miRNA SDs: %.6f", mean(mirna_stats[2,])))
  
  # Round to 2 decimal places
  message("Rounding values to 2 decimal places...")
  expression_rounded <- round(expression_scaled, 2)
  
  # Create final data frame with samples as rows
  final_matrix <- as.data.frame(t(expression_rounded)) %>%
    rownames_to_column("sample_id")
  
  # Clean up sample names (replace dots with hyphens if needed)
  final_matrix$sample_id <- gsub("\\.", "-", final_matrix$sample_id)
  
  # Save processed data
  output_file <- file.path(processed_dir, "mirna_processed.tsv")
  write_tsv(final_matrix, output_file)
  
  # Print processing summary
  message(sprintf("\nmiRNA processing summary for %s:", cancer_type))
  message(sprintf("- Initial number of miRNAs: %d", initial_mirnas))
  message(sprintf("- miRNAs removed (low expression): %d", 
                 initial_mirnas - (ncol(final_matrix) - 1)))
  message(sprintf("- Final number of miRNAs: %d", ncol(final_matrix) - 1))
  message(sprintf("- Number of samples: %d", nrow(final_matrix)))
  message("- Processing steps completed:")
  message("  1. Filtering of low-expression miRNAs")
  message("  2. Z-score standardization")
  message("  3. Rounding to 2 decimal places")
  message("  4. Sample names cleaned (dots replaced with hyphens)")
  message(sprintf("- Output saved to: %s", output_file))
  
  return(final_matrix)
}

#' Helper function to compute basic statistics
#' @param x Numeric vector
#' @return Named vector with mean and sd
#' @keywords internal
compute_stats <- function(x) {
  if(all(is.na(x))) return(c(mean = NA, sd = NA))
  c(mean = mean(x, na.rm = TRUE), 
    sd = sd(x, na.rm = TRUE))
}

#' Create processed directory if it doesn't exist
#' @param cancer_type Cancer type
#' @param base_dir Base directory
#' @return Path to processed directory
#' @keywords internal
create_processed_dir <- function(cancer_type, base_dir) {
  processed_dir <- file.path(base_dir, cancer_type, "processed")
  if (!dir.exists(processed_dir)) {
    dir.create(processed_dir, recursive = TRUE)
  }
  return(processed_dir)
}
