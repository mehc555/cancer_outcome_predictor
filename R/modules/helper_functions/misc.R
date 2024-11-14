# R/modules/helper_functions/misc.R

#' Convert processed data to torch datasets with proper missing value handling
#' @param processed_data List of processed data frames
#' @param config Model configuration
#' @return List of torch datasets containing both data and masks

create_torch_datasets <- function(processed_data, config) {
    torch_datasets <- list()
    
    for (modality in names(processed_data)) {
        if (!is.null(processed_data[[modality]])) {
            # Convert data frame to matrix and handle different modalities
            if (modality == "expression") {
                # Store feature names (gene IDs) before removing sample_id
                feature_names <- colnames(processed_data[[modality]])[-1]  # Remove sample_id column name
                torch_datasets[[paste0(modality, "_features")]] <- feature_names
                
                # Create data matrix without sample_id
                data_matrix <- as.matrix(processed_data[[modality]][, -1])  # Remove first column (sample_id)
                data_matrix <- matrix(as.numeric(data_matrix), 
                                    nrow = nrow(data_matrix),
                                    ncol = ncol(data_matrix))
                
            } else if (modality == "clinical") {
                # Select only numeric columns and convert to matrix
                numeric_cols <- sapply(processed_data[[modality]], is.numeric)
                data_matrix <- as.matrix(processed_data[[modality]][, numeric_cols])
                # Store column names for clinical data
                torch_datasets[[paste0(modality, "_features")]] <- 
                    colnames(processed_data[[modality]][, numeric_cols, drop=FALSE])
                
            } else if (modality == "cnv") {
                # Store feature names before removing sample_id
                feature_names <- colnames(processed_data[[modality]])
                if ("sample_id" %in% feature_names) {
                    feature_names <- feature_names[-which(feature_names == "sample_id")]
                }
                torch_datasets[[paste0(modality, "_features")]] <- feature_names
                
                # Remove Sample_ID column if present
                if ("sample_id" %in% colnames(processed_data[[modality]])) {
                    data_matrix <- as.matrix(processed_data[[modality]][, -which(colnames(processed_data[[modality]]) == "sample_id")])
                } else {
                    data_matrix <- as.matrix(processed_data[[modality]])
                }
                data_matrix <- matrix(as.numeric(data_matrix), 
                                    nrow = nrow(data_matrix),
                                    ncol = ncol(data_matrix))
                
            } else if (modality %in% c("mirna", "methylation", "mutations")) {
                # Store feature names before removing sample_id
                feature_names <- colnames(processed_data[[modality]])
                if ("sample_id" %in% feature_names) {
                    feature_names <- feature_names[-which(feature_names == "sample_id")]
                }
                torch_datasets[[paste0(modality, "_features")]] <- feature_names
                
                # Remove sample identifier column if present
                id_cols <- c("sample_id")
                cols_to_remove <- which(colnames(processed_data[[modality]]) %in% id_cols)
                if (length(cols_to_remove) > 0) {
                    data_matrix <- as.matrix(processed_data[[modality]][, -cols_to_remove])
                } else {
                    data_matrix <- as.matrix(processed_data[[modality]])
                }
                data_matrix <- matrix(as.numeric(data_matrix), 
                                    nrow = nrow(data_matrix),
                                    ncol = ncol(data_matrix))
            }
            
            # Debug print
            print(sprintf("Processing %s:", modality))
            print(sprintf("  Data dimensions: %d x %d", nrow(data_matrix), ncol(data_matrix)))
            print(sprintf("  Features count: %d", length(torch_datasets[[paste0(modality, "_features")]])))
            print(sprintf("  First few features: %s", paste(head(torch_datasets[[paste0(modality, "_features")]]), collapse=", ")))
            
            # Create mask for missing values
            mask_matrix <- !is.na(data_matrix)
            storage.mode(mask_matrix) <- "double"
            
            # Keep the original NA values in the data matrix
            storage.mode(data_matrix) <- "double"
            
            # Verify dimensions match features
            if (ncol(data_matrix) != length(torch_datasets[[paste0(modality, "_features")]])) {
                stop(sprintf("Mismatch in %s: data has %d columns but features list has %d items", 
                           modality, ncol(data_matrix), length(torch_datasets[[paste0(modality, "_features")]])))
            }
            
            # Create torch tensors with proper error handling
            tryCatch({
                # Create tensor for data (NAs will be propagated)
                torch_datasets[[modality]] <- torch_tensor(
                    data_matrix,
                    dtype = torch_float32()
                )
                
                # Create tensor for mask (1 for valid values, 0 for missing values)
                torch_datasets[[paste0(modality, "_mask")]] <- torch_tensor(
                    mask_matrix,
                    dtype = torch_float32()
                )
                
            }, error = function(e) {
                stop(sprintf("Error converting %s to torch tensor: %s", 
                           modality, e$message))
            })
        }
    }
    
    return(torch_datasets)
}

#' Check sample consistency across data modalities
#' @param data_files Named list of file paths for each data type
#' @param cancer_type Current cancer type being processed
#' @return List of data frames with consistent samples
validate_sample_consistency <- function(data_files, cancer_type) {
    # Read all data files
    data_list <- list()

    # Read each data type and extract sample IDs
    if (file.exists(data_files$mirna)) {
        data_list$mirna <- read_tsv(data_files$mirna, show_col_types = FALSE)
        data_list$mirna_samples <- data_list$mirna$sample_id
    }

    if (file.exists(data_files$methylation)) {
        data_list$methylation <- read_tsv(data_files$methylation, show_col_types = FALSE)
        data_list$methylation_samples <- data_list$methylation$sample
    }

    if (file.exists(data_files$mutations)) {
        data_list$mutations <- read_tsv(data_files$mutations, show_col_types = FALSE)
        data_list$mutations_samples <- data_list$mutations$sample
    }

    if (file.exists(data_files$expression)) {
        data_list$expression <- read_tsv(data_files$expression, show_col_types = FALSE)
        data_list$expression_samples <- data_list$expression$sample_id
    }

    if (file.exists(data_files$clinical)) {
        data_list$clinical <- read_tsv(data_files$clinical, show_col_types = FALSE)
        data_list$clinical_samples <- data_list$clinical$sample_id
    }

    if (file.exists(data_files$cnv)) {
        data_list$cnv <- read_tsv(data_files$cnv, show_col_types = FALSE)
        data_list$cnv_samples <- data_list$cnv$Sample_ID
    }

    # Get all sample ID vectors
    sample_lists <- list()
    for (data_type in c("mirna", "methylation", "mutations", "expression", "clinical", "cnv")) {
        sample_col <- paste0(data_type, "_samples")
        if (!is.null(data_list[[sample_col]])) {
            sample_lists[[data_type]] <- data_list[[sample_col]]
        }
    }

    # Find common samples across all available data types
    common_samples <- Reduce(intersect, sample_lists)

    # Report statistics
    message(sprintf("\nSample consistency report for %s:", cancer_type))
    message(sprintf("Common samples across all modalities: %d", length(common_samples)))
    for (data_type in names(sample_lists)) {
        message(sprintf("%s samples: %d", data_type, length(sample_lists[[data_type]])))
    }

}

