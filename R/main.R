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
source("R/modules/helper_functions/misc.R")
set.seed(123)


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
    	cancer_type = cancer_type,
        validation_pct = 0.3,
	test_pct = 0.3,
	seed = NULL # using seed defined at the top for now
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

