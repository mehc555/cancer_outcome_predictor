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

#' Standardize expression matrix
#' @param expression_df Data frame of expression values (without Ensembl_ID column)
#' @return Standardized expression matrix
#' @keywords internal
standardize_expression <- function(expression_df) {
  # Convert to matrix
  expression_matrix <- as.matrix(expression_df)
  
  # Calculate mean and sd for each gene (row)
  gene_means <- rowMeans(expression_matrix, na.rm = TRUE)
  gene_sds <- apply(expression_matrix, 1, sd, na.rm = TRUE)
  
  # Replace zero standard deviations with 1 to avoid division by zero
  gene_sds[gene_sds == 0] <- 1
  
  # Standardize (z-score normalization)
  standardized_matrix <- sweep(expression_matrix, 1, gene_means, "-")
  standardized_matrix <- sweep(standardized_matrix, 1, gene_sds, "/")
  
  # Convert back to data frame with same column names
  standardized_df <- as.data.frame(standardized_matrix)
  colnames(standardized_df) <- colnames(expression_df)
  
  return(standardized_df)
}

#' Process TPM expression data
#' @param cancer_type Character string specifying cancer type (e.g., "BRCA", "COAD", "LUAD")
#' @param min_tpm Minimum TPM value threshold for filtering genes (default: 1)
#' @param min_samples Minimum number of samples where gene should be expressed (default: 0.2)
#' @param base_dir Base directory containing the data, defaults to "data/GDC_TCGA"
#' @return Processed expression data frame
#' @export
process_expression_data <- function(cancer_type, min_tpm = 1, min_samples = 3, 
                                  base_dir = "data/GDC_TCGA") {
  # Construct file paths
  input_file <- file.path(base_dir, cancer_type, "expression_tpm.tsv")
  processed_dir <- create_processed_dir(cancer_type, base_dir)
  output_file_log2 <- file.path(processed_dir, "expression_processed_log2.tsv")
  output_file_standardized <- file.path(processed_dir, "expression_processed_standardized.tsv")
  
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
    filter(rowSums(select(., -Ensembl_ID) >= min_tpm) >= min_samples) %>%
    # Log2 transform TPM values (adding small constant to avoid log(0))
    mutate(across(-Ensembl_ID, ~log2(.x + 0.1))) %>%
    # Remove genes with zero variance
    filter(apply(select(., -Ensembl_ID), 1, var) > 0)
  
  # Record filtered dimensions
  filtered_genes <- nrow(expression_processed)
  removed_genes <- initial_genes - filtered_genes
  
  # Validate the processed data
  validate_expression_data(expression_processed)
  
  # Separate Ensembl IDs and expression values
  ensembl_ids <- expression_processed$Ensembl_ID
  expression_values <- expression_processed %>% select(-Ensembl_ID)
  
  # Create standardized version
  standardized_values <- standardize_expression(expression_values)
  
  expression_values=round(expression_values, 3)
  standardized_values=round(standardized_values, 3)

  # Add Ensembl IDs back to both versions
  expression_processed <- bind_cols(
    tibble(Ensembl_ID = ensembl_ids),
    expression_values
  )
  
  expression_standardized <- bind_cols(
    tibble(Ensembl_ID = ensembl_ids),
    standardized_values
  )
  


  # Save both versions
  write_tsv(expression_processed, output_file_log2)
  write_tsv(expression_standardized, output_file_standardized)
  
  # Print processing summary
  message(sprintf("Expression processing summary for %s:", cancer_type))
  message(sprintf("- Initial number of genes: %d", initial_genes))
  message(sprintf("- Removed %d genes (low expression/zero variance)", removed_genes))
  message(sprintf("- Retained %d genes", filtered_genes))
  message(sprintf("- Number of samples: %d", initial_samples))
  message("- Transformations applied:")
  message("  1. log2(TPM + 0.1)")
  message("  2. Z-score standardization (saved separately)")
  
  # Return both versions in a list
  return(list(
    log2_transformed = expression_processed,
    standardized = expression_standardized
  ))
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
