# R/modules/data_handling/cnv_processor.R

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

#' Process copy number variation data
#' @param cancer_type Character string specifying cancer type (e.g., "BRCA", "COAD", "LUAD")
#' @param base_dir Base directory containing the data, defaults to "data/GDC_TCGA"
#' @return Processed CNV data frame
#' @export
process_cnv_data <- function(cancer_type, base_dir = "data/GDC_TCGA") {
  # Construct file paths
  input_file <- file.path(base_dir, cancer_type, "cnv_gene.tsv")
  processed_dir <- create_processed_dir(cancer_type, base_dir)
  output_file <- file.path(processed_dir, "cnv_genes.tsv")
  
  if (!file.exists(input_file)) {
    stop(sprintf("CNV data file not found for cancer type %s", cancer_type))
  }
  
  # Read and process CNV data
  cnv_data <- read_tsv(input_file, show_col_types = FALSE)
  
  # Record initial dimensions
  initial_genes <- nrow(cnv_data)
  
  # Process the data
  cnv_data <- cnv_data %>%
    # Remove version numbers from Ensembl IDs
    mutate(Ensembl_ID = sub("\\.[0-9]+$", "", Ensembl_ID)) %>%
    # Remove genes that are NA for all samples
    filter(rowSums(!is.na(select(., -Ensembl_ID))) > 0)
  
  # Record filtered dimensions
  filtered_genes <- nrow(cnv_data)
  removed_genes <- initial_genes - filtered_genes
  
  # Validate the processed data
  validate_cnv_data(cnv_data)
  
  # Save processed data
  write_tsv(cnv_data, output_file)
  
  # Print summary to console
  message(sprintf("CNV processing summary for %s:", cancer_type))
  message(sprintf("- Removed %d genes with all NA values", removed_genes))
  message(sprintf("- Retained %d genes with valid measurements", filtered_genes))
  message(sprintf("- Number of samples: %d", ncol(cnv_data) - 1))
  
  return(cnv_data)
}

#' Validate CNV data structure and content
#' @param cnv_data Processed CNV data frame
#' @return Logical indicating if validation passed (invisible)
#' @keywords internal
validate_cnv_data <- function(cnv_data) {
  # Check if data frame is empty after filtering
  if (nrow(cnv_data) == 0) {
    stop("No genes remained after filtering")
  }
  
  # Check if Ensembl_ID column exists
  if (!"Ensembl_ID" %in% colnames(cnv_data)) {
    stop("Ensembl_ID column not found in CNV data")
  }
  
  # Check if Ensembl IDs are properly formatted (no version numbers)
  if (any(grepl("\\.[0-9]+$", cnv_data$Ensembl_ID))) {
    stop("Some Ensembl IDs still contain version numbers")
  }
  
  # Check if any non-numeric values in copy number columns (excluding NAs)
  non_numeric_check <- cnv_data %>%
    select(-Ensembl_ID) %>%
    mutate(across(everything(), as.numeric)) %>%
    sapply(function(x) all(is.na(x) | is.numeric(x)))
  
  if (!all(non_numeric_check)) {
    stop("Non-numeric values found in copy number data")
  }
  
  # Check for duplicate Ensembl IDs after version removal
  if (any(duplicated(cnv_data$Ensembl_ID))) {
    stop("Duplicate Ensembl IDs found after version number removal")
  }
  
  invisible(TRUE)
}
