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

#' Process copy number variation data for both standard and DNN input formats
#' @param cancer_type Character string specifying cancer type (e.g., "BRCA", "COAD", "LUAD")
#' @param base_dir Base directory containing the data, defaults to "data/GDC_TCGA"
#' @param preprocessing_method One of "raw" (default), "binned", or "one_hot"
#' @param max_copy_number Integer specifying the maximum copy number to consider before binning (default: 8)
#' @return List containing both processed CNV data frames (standard and DNN format)
#' @export
process_cnv_data <- function(cancer_type, base_dir = "data/GDC_TCGA", 
                           preprocessing_method = "raw",
                           max_copy_number = 8) {
  # Validate preprocessing method
  valid_methods <- c("raw", "binned", "one_hot")
  if (!preprocessing_method %in% valid_methods) {
    stop(sprintf("preprocessing_method must be one of: %s", 
                paste(valid_methods, collapse = ", ")))
  }
  
  # Construct file paths
  input_file <- file.path(base_dir, cancer_type, "cnv_gene.tsv")
  processed_dir <- create_processed_dir(cancer_type, base_dir)
  standard_output <- file.path(processed_dir, "cnv_genes.tsv")
  dnn_output <- file.path(processed_dir, 
                         sprintf("cnv_genes_dnn_%s.tsv", preprocessing_method))
  
  if (!file.exists(input_file)) {
    stop(sprintf("CNV data file not found for cancer type %s", cancer_type))
  }
  
  # Read and process CNV data
  cnv_data <- read_tsv(input_file, show_col_types = FALSE)
  
  # Record initial dimensions
  initial_genes <- nrow(cnv_data)
  initial_samples <- ncol(cnv_data) - 1
  
  # Process the data - common steps
  cnv_data <- cnv_data %>%
    # Remove version numbers from Ensembl IDs
    mutate(Ensembl_ID = sub("\\.[0-9]+$", "", Ensembl_ID)) %>%
    # Remove genes that are NA for all samples
    filter(rowSums(!is.na(select(., -Ensembl_ID))) > 0)
  
  # Save standard format (genes as rows)
  write_tsv(cnv_data, standard_output)
  
  # Create DNN format (samples as rows)
  cnv_data_t <- cnv_data %>%
    column_to_rownames("Ensembl_ID") %>%
    t() %>%
    as.data.frame() %>%
    rownames_to_column("Sample_ID")
  
  # Apply preprocessing based on selected method
  cnv_processed <- switch(
    preprocessing_method,
    "raw" = preprocess_raw(cnv_data_t),
    "binned" = preprocess_binned(cnv_data_t, max_copy_number),
    "one_hot" = preprocess_one_hot(cnv_data_t, max_copy_number)
  )
  
  # Record dimensions
  filtered_genes <- ncol(cnv_processed) - 1  # subtract 1 for Sample_ID column
  removed_genes <- initial_genes - (if(preprocessing_method == "one_hot") 
                                  filtered_genes/max_copy_number else filtered_genes)
  
  # Validate both data formats
  validate_cnv_data_standard(cnv_data)
  validate_cnv_data_dnn(cnv_processed)
  
  # Save DNN format
  write_tsv(cnv_processed, dnn_output)
  
  # Print summary to console
  message("\nCNV processing summary for ", cancer_type, ":")
  message("\nStandard format (cnv_genes.tsv):")
  message(sprintf("- Genes: %d", nrow(cnv_data)))
  message(sprintf("- Samples: %d", ncol(cnv_data) - 1))
  
  message("\nDNN format (cnv_genes_dnn_", preprocessing_method, ".tsv):")
  message(sprintf("- Preprocessing method: %s", preprocessing_method))
  message(sprintf("- Samples: %d", nrow(cnv_processed)))
  message(sprintf("- Features: %d", ncol(cnv_processed) - 1))
  if (preprocessing_method == "binned") {
    message(sprintf("- Copy numbers binned at maximum of %d", max_copy_number))
  }
  
  # Return both formats in a list
  return(list(
    standard_format = cnv_data,
    dnn_format = cnv_processed
  ))
}

#' Preprocess raw CNV data (keep original values)
#' @param cnv_data Data frame containing CNV data
#' @return Processed CNV data frame
#' @keywords internal
preprocess_raw <- function(cnv_data) {
  # Replace any negative values with 0 (some datasets use -1 for deletions)
  cnv_data %>%
    mutate(across(-Sample_ID, ~ifelse(. < 0, 0, .)))
}

#' Preprocess CNV data with binning for high copy numbers
#' @param cnv_data Data frame containing CNV data
#' @param max_copy_number Maximum copy number before binning
#' @return Processed CNV data frame
#' @keywords internal
preprocess_binned <- function(cnv_data, max_copy_number) {
  cnv_data %>%
    mutate(across(-Sample_ID, function(x) {
      # Replace negative values with 0
      x[x < 0] <- 0
      # Bin high copy numbers
      x[x > max_copy_number] <- max_copy_number
      return(x)
    }))
}

#' One-hot encode CNV values
#' @param cnv_data Data frame containing CNV data
#' @param max_copy_number Maximum copy number before binning
#' @return One-hot encoded CNV data frame
#' @keywords internal
preprocess_one_hot <- function(cnv_data, max_copy_number) {
  # Keep Sample_ID column separate
  sample_ids <- cnv_data$Sample_ID
  
  # Function to one-hot encode a single column
  encode_column <- function(col_name, data) {
    # Bin values first
    values <- data[[col_name]]
    values[values < 0] <- 0
    values[values > max_copy_number] <- max_copy_number
    
    # Create one-hot encoding
    possible_values <- 0:max_copy_number
    result <- matrix(0, nrow = nrow(data), ncol = length(possible_values))
    colnames(result) <- paste0(col_name, "_", possible_values)
    
    for (i in seq_along(possible_values)) {
      result[, i] <- as.integer(values == possible_values[i])
    }
    return(as.data.frame(result))
  }
  
  # Apply one-hot encoding to each gene column
  encoded_data <- do.call(cbind, lapply(
    names(select(cnv_data, -Sample_ID)),
    function(col) encode_column(col, cnv_data)
  ))
  
  # Add back Sample_ID column
  encoded_data$Sample_ID <- sample_ids
  
  # Reorder columns to put Sample_ID first
  encoded_data <- encoded_data %>%
    select(Sample_ID, everything())
  
  return(encoded_data)
}

#' Validate standard CNV data structure and content
#' @param cnv_data Standard format CNV data frame
#' @return Logical indicating if validation passed (invisible)
#' @keywords internal
validate_cnv_data_standard <- function(cnv_data) {
  # Check if data frame is empty
  if (nrow(cnv_data) == 0) {
    stop("No genes found in the processed data")
  }
  
  # Check if Ensembl_ID column exists
  if (!"Ensembl_ID" %in% colnames(cnv_data)) {
    stop("Ensembl_ID column not found in CNV data")
  }
  
  # Check for duplicate Ensembl IDs
  if (any(duplicated(cnv_data$Ensembl_ID))) {
    stop("Duplicate Ensembl IDs found")
  }
  
  # Check if any non-numeric values in copy number columns (excluding NAs)
  non_numeric_check <- cnv_data %>%
    select(-Ensembl_ID) %>%
    mutate(across(everything(), as.numeric)) %>%
    sapply(function(x) all(is.na(x) | is.numeric(x)))
  
  if (!all(non_numeric_check)) {
    stop("Non-numeric values found in copy number data")
  }
  
  invisible(TRUE)
}

#' Validate DNN CNV data structure and content
#' @param cnv_data DNN format CNV data frame
#' @return Logical indicating if validation passed (invisible)
#' @keywords internal
validate_cnv_data_dnn <- function(cnv_data) {
  # Check if data frame is empty
  if (nrow(cnv_data) == 0) {
    stop("No samples found in the processed data")
  }
  
  # Check if Sample_ID column exists
  if (!"Sample_ID" %in% colnames(cnv_data)) {
    stop("Sample_ID column not found in CNV data")
  }
  
  # Check if any non-numeric values in feature columns (excluding NAs)
  non_numeric_check <- cnv_data %>%
    select(-Sample_ID) %>%
    mutate(across(everything(), as.numeric)) %>%
    sapply(function(x) all(is.na(x) | is.numeric(x)))
  
  if (!all(non_numeric_check)) {
    stop("Non-numeric values found in feature data")
  }
  
  # Check for duplicate sample IDs
  if (any(duplicated(cnv_data$Sample_ID))) {
    stop("Duplicate Sample IDs found")
  }
  
  invisible(TRUE)
}
