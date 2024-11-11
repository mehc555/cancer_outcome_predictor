# R/modules/data_handling/expression_processor.R

library(tidyverse)

#' Create processed data directory
#' @param cancer_type Character string specifying cancer type
#' @param base_dir Base directory containing the data
#' @return Path to processed directory
#' @keywords internal
create_processed_dir <- function(cancer_type, base_dir = "data/GDC_TCGA") {
  processed_dir <- file.path(base_dir, cancer_type, "processed")
  if (!dir.exists(processed_dir)) {
    dir.create(processed_dir, recursive = TRUE)
  }
  return(processed_dir)
}

#' Process TPM expression data
#' @param cancer_type Character string specifying cancer type (e.g., "BRCA", "COAD", "LUAD")
#' @param min_tpm Minimum TPM value threshold for filtering genes (default: 1)
#' @param min_samples Minimum number of samples where gene should be expressed (default: 0.2)
#' @param base_dir Base directory containing the data, defaults to "data/GDC_TCGA"
#' @return Processed expression data frame
#' @export
process_expression_data <- function(cancer_type, min_tpm = 1, min_samples = 0.2, 
                                  base_dir = "data/GDC_TCGA") {
  # Construct file paths
  input_file <- file.path(base_dir, cancer_type, "star_tpm.tsv")
  processed_dir <- create_processed_dir(cancer_type, base_dir)
  output_file <- file.path(processed_dir, "expression_processed.tsv")
  
  if (!file.exists(input_file)) {
    stop(sprintf("Expression TPM data file not found for cancer type %s", cancer_type))
  }
  
  # Read expression data
  expression_data <- read_tsv(input_file, show_col_types = FALSE)
  
  # Record initial dimensions
  initial_genes <- nrow(expression_data)
  initial_samples <- ncol(expression_data) - 1  # Subtract 1 for Ensembl_ID column
  
  # Process the data
  expression_processed <- expression_data %>%
    # Remove version numbers from Ensembl IDs
    mutate(Ensembl_ID = sub("\\.[0-9]+$", "", Ensembl_ID)) %>%
    # Filter low-expression genes
    filter(rowSums(select(., -Ensembl_ID) >= min_tpm) >= 
             (min_samples * (ncol(.) - 1))) %>%
    # Log2 transform TPM values (adding small constant to avoid log(0))
    mutate(across(-Ensembl_ID, ~log2(.x + 0.1))) %>%
    # Remove genes with zero variance
    filter(apply(select(., -Ensembl_ID), 1, var) > 0)
  
  # Record filtered dimensions
  filtered_genes <- nrow(expression_processed)
  removed_genes <- initial_genes - filtered_genes
  
  # Validate the processed data
  validate_expression_data(expression_processed)
  
  # Save processed data
  write_tsv(expression_processed, output_file)
  
  # Print processing summary
  message(sprintf("Expression processing summary for %s:", cancer_type))
  message(sprintf("- Initial number of genes: %d", initial_genes))
  message(sprintf("- Removed %d genes (low expression/zero variance)", removed_genes))
  message(sprintf("- Retained %d genes", filtered_genes))
  message(sprintf("- Number of samples: %d", initial_samples))
  message("- Transformations applied: log2(TPM + 0.1)")
  
  return(expression_processed)
}

#' Validate expression data structure and content
#' @param expression_data Processed expression data frame
#' @return Logical indicating if validation passed (invisible)
#' @keywords internal
validate_expression_data <- function(expression_data) {
  # Check if data frame is empty
  if (nrow(expression_data) == 0) {
    stop("No genes remained after filtering")
  }
  
  # Check if Ensembl_ID column exists
  if (!"Ensembl_ID" %in% colnames(expression_data)) {
    stop("Ensembl_ID column not found in expression data")
  }
  
  # Check if Ensembl IDs are properly formatted (no version numbers)
  if (any(grepl("\\.[0-9]+$", expression_data$Ensembl_ID))) {
    stop("Some Ensembl IDs still contain version numbers")
  }
  
  # Check for duplicate Ensembl IDs
  if (any(duplicated(expression_data$Ensembl_ID))) {
    stop("Duplicate Ensembl IDs found after version number removal")
  }
  
  # Check if any non-numeric values in expression columns
  non_numeric_check <- expression_data %>%
    select(-Ensembl_ID) %>%
    sapply(function(x) all(is.numeric(x)))
  
  if (!all(non_numeric_check)) {
    stop("Non-numeric values found in expression data")
  }
  
  invisible(TRUE)
}
