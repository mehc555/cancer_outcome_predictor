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

main <- function() {
    # Define cancer types
    #cancer_types <- c("BRCA", "COAD", "LUAD")
    cancer_types <- c("BRCA")

    # Download TCGA data if needed
    # download_tcga_data() 

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
        #processed_data[[cancer_type]]$clinical <- process_clinical_data(cancer_type, common_features = common_clinical_features, impute = FALSE, impute_method = "missing_category")
        
        # Process expression data
        message("\nProcessing gene expression data...")
        #processed_data[[cancer_type]]$expression <- process_expression_data(cancer_type, min_tpm = 1, min_samples = 3)

	# Process mutation data
	message("\nProcessing mutation data...")
	#processed_data[[cancer_type]]$mutations <- process_mutation_data(cancer_type, min_freq = 0.01)
    
        message("\nProcessing methylation data...")
        #processed_data[[cancer_type]]$methylation <- process_methylation_data(cancer_type)

	message("\nProcessing mirna expression data...")
        processed_data[[cancer_type]]$mirna <- process_mirna_data(cancer_type)

    }
    
    return(processed_data)
}

# Run the main function
processed_data <- main()
