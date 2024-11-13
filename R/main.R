# R/main.R

# Source required modules
source("R/modules/data_handling/download_processor.R")
source("R/modules/data_handling/cnv_processor.R")
source("R/modules/data_handling/clinical_processor.R")
source("R/modules/data_handling/expression_processor.R")
source("R/modules/data_handling/mutation_processor.R")
source("R/modules/data_handling/methylation_processor.R")
source("R/modules/data_handling/mirna_processor.R")
set.seed(123)

#' Get common columns across all cancer types, excluding all-NA columns
#' @param cancer_types Vector of cancer types
#' @param base_dir Base directory containing the data
#' @return Vector of column names common to all cancer types with valid data
get_common_clinical_features <- function(cancer_types, base_dir = "data/GDC_TCGA") {
    # Read all clinical data files
    clinical_data_list <- lapply(cancer_types, function(cancer_type) {
        file_path <- file.path(base_dir, cancer_type, "clinical.tsv")
        if (!file.exists(file_path)) {
            stop(sprintf("Clinical data file not found for cancer type %s", cancer_type))
        }
        read_tsv(file_path, show_col_types = FALSE)
    })
    
    # Get column names for each cancer type
    column_lists <- lapply(clinical_data_list, colnames)
    
    # Find common columns
    common_columns <- Reduce(intersect, column_lists)
    
    # Check each common column for all-NA across all cancer types
    valid_columns <- common_columns[sapply(common_columns, function(col) {
        # Check if the column has any non-NA values in any cancer type
        any_valid <- sapply(clinical_data_list, function(data) {
            # Handle both NA and "not reported" as missing values
            values <- data[[col]]
            values[values %in% c("not reported", "Not Reported")] <- NA
            !all(is.na(values))
        })
        
        # Return TRUE if at least one cancer type has non-NA values
        any(any_valid)
    })]
    
    # Print removed columns for debugging
    removed_columns <- setdiff(common_columns, valid_columns)
    if (length(removed_columns) > 0) {
        message("Removed the following all-NA columns: ", 
                paste(removed_columns, collapse = ", "))
    }
    
    message("Found ", length(valid_columns), " common clinical features with valid data across all cancer types")
    return(valid_columns)
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
        # Expression data is transposed, get column names except first (Ensembl_ID)
        data_list$expression_samples <- colnames(data_list$expression)[-1]
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
    
    # Filter all data frames to include only common samples
    filtered_data <- list()
    
    if (!is.null(data_list$mirna)) {
        filtered_data$mirna <- data_list$mirna %>%
            filter(sample_id %in% common_samples)
    }
    
    if (!is.null(data_list$methylation)) {
        filtered_data$methylation <- data_list$methylation %>%
            filter(sample %in% common_samples)
    }
    
    if (!is.null(data_list$mutations)) {
        filtered_data$mutations <- data_list$mutations %>%
            filter(sample %in% common_samples)
    }
    
    if (!is.null(data_list$expression)) {
        filtered_data$expression <- data_list$expression %>%
            select(Ensembl_ID, all_of(common_samples))
    }
    
    if (!is.null(data_list$clinical)) {
        filtered_data$clinical <- data_list$clinical %>%
            filter(sample_id %in% common_samples)
    }
    
    if (!is.null(data_list$cnv)) {
        filtered_data$cnv <- data_list$cnv %>%
            filter(Sample_ID %in% common_samples)
    }
    
    # Verify all filtered datasets have the same number of samples
    sample_counts <- sapply(filtered_data, function(df) {
        if ("Ensembl_ID" %in% colnames(df)) {
            return(ncol(df) - 1)  # Subtract 1 for Ensembl_ID column
        } else {
            return(nrow(df))
        }
    })
    
    if (length(unique(sample_counts)) != 1) {
        stop(sprintf("Inconsistent sample counts after filtering in %s: %s", 
                    cancer_type, 
                    paste(names(sample_counts), sample_counts, sep="=", collapse=", ")))
    }
    
    return(filtered_data)
}

main <- function() {
    
    # Define cancer types
    cancer_types <- c("BRCA")
    base_dir <- "data/GDC_TCGA"
    
    # Get common clinical features
    message("\nIdentifying common clinical features...")
    common_clinical_features <- get_common_clinical_features(cancer_types)
    
    # Process data for each cancer type
    processed_data <- list()
    
    for (cancer_type in cancer_types) {
        message(sprintf("\nProcessing data for %s:", cancer_type))
        
        # Process CNV data
        message("\nProcessing CNV data...")
        #processed_data[[cancer_type]]$cnv <- process_cnv_data(cancer_type, preprocessing_method = "raw")
        
        # Process clinical data with common features
        message("\nProcessing clinical data...")
        #processed_data[[cancer_type]]$clinical <- process_clinical_data(cancer_type, 
        #                                                              common_features = common_clinical_features, 
        #                                                             impute = FALSE, 
        #                                                              impute_method = "missing_category")
        
        # Process expression data
        message("\nProcessing gene expression data...")
        #processed_data[[cancer_type]]$expression <- process_expression_data(cancer_type, 
        #                                                                  min_tpm = 1, 
        #                                                                  min_samples = 3)
        
        # Process mutation data
        message("\nProcessing mutation data...")
        #processed_data[[cancer_type]]$mutations <- process_mutation_data(cancer_type, 
        #                                                               min_freq = 0.01)
        
        message("\nProcessing methylation data...")
        #processed_data[[cancer_type]]$methylation <- process_methylation_data(cancer_type)
        
        message("\nProcessing mirna expression data...")
        #processed_data[[cancer_type]]$mirna <- process_mirna_data(cancer_type)
        
        # Define file paths for processed data
        data_files <- list(
            mirna = file.path(base_dir, cancer_type, "processed", "mirna_processed.tsv"),
            methylation = file.path(base_dir, cancer_type, "processed", "methylation_processed.tsv"),
            mutations = file.path(base_dir, cancer_type, "processed", "mutations_processed_hybrid.tsv"),
            expression = file.path(base_dir, cancer_type, "processed", "expression_processed_standardized.tsv"),
            clinical = file.path(base_dir, cancer_type, "processed", "clinical_processed_dnn.tsv"),
            cnv = file.path(base_dir, cancer_type, "processed", "cnv_genes_dnn_raw.tsv")
        )
        
        # Validate sample consistency and get filtered data
        message("\nValidating sample consistency across data modalities...")
        processed_data[[cancer_type]] <- validate_sample_consistency(data_files, cancer_type)
    }
    
    return(processed_data)
}

# Run the main function
main()
