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
                # Remove Ensembl_ID column and convert to numeric matrix
                data_matrix <- as.matrix(processed_data[[modality]][, -1])
                data_matrix <- matrix(as.numeric(data_matrix), 
                                    nrow = nrow(data_matrix),
                                    ncol = ncol(data_matrix))
                # Store gene IDs
                torch_datasets[[paste0(modality, "_features")]] <- 
                    processed_data[[modality]]$Ensembl_ID
                
            } else if (modality == "clinical") {
                # Select only numeric columns and convert to matrix
                numeric_cols <- sapply(processed_data[[modality]], is.numeric)
                data_matrix <- as.matrix(processed_data[[modality]][, numeric_cols])
                
            } else if (modality == "cnv") {
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
            
            # Create mask for missing values
            mask_matrix <- !is.na(data_matrix)
            storage.mode(mask_matrix) <- "double"
            
            # Keep the original NA values in the data matrix
            storage.mode(data_matrix) <- "double"
            
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
            
            # Add dimension names as attributes if needed
            if (modality != "expression") {  # Expression already has features stored
                torch_datasets[[paste0(modality, "_features")]] <- colnames(data_matrix)
            }
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
    if(!download) {
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
	# Change this later on to use the processed_data list instead
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
        validate_sample_consistency(data_files, cancer_type)
        
	# Write all input data to disk before training and harmonize sample_id column name
	
	input_dir <- file.path(base_dir, cancer_type, "input_data")
  	if (!dir.exists(input_dir)) {
    		dir.create(input_dir, recursive = TRUE)
  	}

        for (modality in names(processed_data[[cancer_type]])) {
		colnames(processed_data[[cancer_type]][[modality]])[1]="sample_id"
		print(paste0("Modality: ", modality))
		print(head(processed_data[[cancer_type]][[modality]][,1:4]))
		write.table(processed_data[[cancer_type]][[modality]], paste0(input_dir,"/",modality,".matrix.tsv"), quote=F, row.names=F, sep="\t")
	}

	# Convert to torch datasets
        message("\nConverting data to torch format...")
        torch_datasets <- create_torch_datasets(
            processed_data[[cancer_type]], 
            config$model
        )

	# Initialize model
	model <- MultiModalSurvivalModel(
    	modality_dims = config$model$architecture$modality_dims,
    	encoder_dims = config$model$architecture$encoder_dims,
    	fusion_dim = config$model$architecture$fusion_dim,
    	num_heads = config$model$architecture$num_heads,
    	dropout = config$model$architecture$dropout
	)

	# Run nested cross-validation with repeated validation sets
	message("\nStarting nested cross-validation...")
	cv_results <- run_nested_cv(
    	model = model,  # Add this line
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

