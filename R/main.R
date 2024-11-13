# R/main.R

# Source base setup
source("R/setup.R")
source("R/modules/models/torch_models.R")
source("R/modules/training/cv_engine.R")
#source("R/modules/training/train_utils.R")

# Source required modules
source("R/modules/data_handling/download_processor.R")
source("R/modules/data_handling/cnv_processor.R")
source("R/modules/data_handling/clinical_processor.R")
source("R/modules/data_handling/expression_processor.R")
source("R/modules/data_handling/mutation_processor.R")
source("R/modules/data_handling/methylation_processor.R")
source("R/modules/data_handling/mirna_processor.R")
set.seed(123)

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
        data_list$expression_samples <- data_list$sample_id
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
            filter(sample_id %in% common_samples)
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
            return(nrow(df))
    })
    
    if (length(unique(sample_counts)) != 1) {
        stop(sprintf("Inconsistent sample counts after filtering in %s: %s", 
                    cancer_type, 
                    paste(names(sample_counts), sample_counts, sep="=", collapse=", ")))
    }
    
    return(filtered_data)
}

#' Convert processed data to torch datasets with uniform handling
#' @param processed_data List of processed data frames
#' @param config Model configuration
#' @return List of torch datasets
create_torch_datasets <- function(processed_data, config) {
    torch_datasets <- list()
    
    for (modality in names(processed_data)) {
        if (!is.null(processed_data[[modality]])) {
            # Remove identifier columns if present
            id_cols <- c("sample_id", "sample", "Sample_ID")
            cols_to_remove <- which(colnames(processed_data[[modality]]) %in% id_cols)
            
            if (length(cols_to_remove) > 0) {
                data_matrix <- as.matrix(processed_data[[modality]][, -cols_to_remove])
            } else {
                data_matrix <- as.matrix(processed_data[[modality]])
            }
            
            # Convert to numeric matrix and handle NAs
            data_matrix <- matrix(as.numeric(data_matrix), 
                                nrow = nrow(data_matrix),
                                ncol = ncol(data_matrix))
            data_matrix[is.na(data_matrix)] <- 0
            storage.mode(data_matrix) <- "double"
            
            # Create torch tensor with error handling
            tryCatch({
                torch_datasets[[modality]] <- torch_tensor(
                    data_matrix,
                    dtype = torch_float32()
                )
                
                # Store feature names
                torch_datasets[[paste0(modality, "_features")]] <- colnames(data_matrix)
                
            }, error = function(e) {
                stop(sprintf("Error converting %s to torch tensor: %s", 
                           modality, e$message))
            })
        }
    }
    
    return(torch_datasets)
}

main <- function(download=FALSE) {
    # Initialize project and load config
    if (!exists("config")) {
        initialize_project()
        config <- load_config()
    }
    
    # Download data
    if(!dowload) {
    	download_tcga_data()
    }
    # Define cancer types
    cancer_types <- c("BRCA", "COAD", "LUAD")
    base_dir <- "data/GDC_TCGA"
    
    # Process data for each cancer type
    processed_data <- list()
    
    for (cancer_type in cancer_types) {
        message(sprintf("\nProcessing data for %s:", cancer_type))
        
        # Process CNV data
        message("\nProcessing CNV data...")
        processed_data[[cancer_type]]$cnv <- process_cnv_data(cancer_type, preprocessing_method = "raw")
        
        # Process clinical data with common features
        message("\nProcessing clinical data...")
        processed_data[[cancer_type]]$clinical <- process_clinical_data(cancer_type,  
                                                                      impute = FALSE, 
                                                                      impute_method = "missing_category")
        
        # Process expression data
        message("\nProcessing gene expression data...")
        processed_data[[cancer_type]]$expression <- process_expression_data(cancer_type, 
                                                                          min_tpm = 1, 
                                                                          min_samples = 3)
        
        # Process mutation data
        message("\nProcessing mutation data...")
        processed_data[[cancer_type]]$mutations <- process_mutation_data(cancer_type, 
                                                                       min_freq = 0.01)
        
        message("\nProcessing methylation data...")
        processed_data[[cancer_type]]$methylation <- process_methylation_data(cancer_type)
        
        message("\nProcessing mirna expression data...")
        processed_data[[cancer_type]]$mirna <- process_mirna_data(cancer_type)
        
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
        
        # Convert to torch datasets
        message("\nConverting data to torch format...")
        torch_datasets <- create_torch_datasets(
            processed_data[[cancer_type]], 
            config$model
        )
        
        # Run nested cross-validation with repeated validation sets
        message("\nStarting nested cross-validation...")
        cv_results <- run_nested_cv(
            datasets = torch_datasets,
            config = config,
            cancer_type = cancer_type
        )
        
        # Save results
        results_dir <- file.path(config$main$paths$results_dir, cancer_type)
        dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
        
        saveRDS(
            cv_results,
            file.path(results_dir, "cv_results.rds")
        )
        
        # Log completion
        logger::log_info("Completed processing for cancer type: {cancer_type}")
    }
    
    return(processed_data)
}

main(download=FALSE)

