MultiModalDataset <- dataset(
    name = "MultiModalDataset",
    
    initialize = function(data) {
        # Input validation
        if (!is.list(data)) {
            stop("Input must be a list")
        }
        
        # Required modalities and their feature names
        required_prefixes <- c("clinical", "cnv", "expression", "mutations", "methylation", "mirna")
        
        # Initialize storage
        self$data <- list()
        self$features <- list()
        
        # Validation helper function
        validate_tensor <- function(tensor, name) {
            if (!inherits(tensor, "torch_tensor")) {
                stop(sprintf("Data for %s must be a torch tensor", name))
            }
            if (length(dim(tensor)) != 2) {
                stop(sprintf("Data for %s must be 2-dimensional", name))
            }
        }
        
        # Process feature names first
        for (modality in required_prefixes) {
            feature_name <- paste0(modality, "_features")
            if (!is.null(data[[feature_name]])) {
                # Convert to character vector and store
                self$features[[modality]] <- as.character(data[[feature_name]])
                cat(sprintf("Loaded %d features for %s\n", 
                          length(self$features[[modality]]), 
                          modality))
            }
        }
        
        # Process data tensors
        n_samples <- NULL
        for (modality in required_prefixes) {
            if (!is.null(data[[modality]])) {
                tryCatch({
                    validate_tensor(data[[modality]], modality)
                    
                    # Store tensor
                    self$data[[modality]] <- data[[modality]]
                    
                    # Set or verify number of samples
                    if (is.null(n_samples)) {
                        n_samples <- dim(data[[modality]])[1]
                    } else if (dim(data[[modality]])[1] != n_samples) {
                        stop(sprintf("Inconsistent number of samples in %s: expected %d, got %d",
                                   modality, n_samples, dim(data[[modality]])[1]))
                    }
                    
                    # Process mask if it exists
                    mask_name <- paste0(modality, "_mask")
                    if (!is.null(data[[mask_name]])) {
                        validate_tensor(data[[mask_name]], mask_name)
                        if (!all(dim(data[[mask_name]]) == dim(data[[modality]]))) {
                            stop(sprintf("Mask dimensions for %s don't match data dimensions", modality))
                        }
                        self$data[[mask_name]] <- data[[mask_name]]
                    }
                    
                    cat(sprintf("Loaded %s data: %s\n", 
                              modality,
                              paste(dim(data[[modality]]), collapse="x")))
                    
                }, error = function(e) {
                    warning(sprintf("Error processing %s: %s", modality, conditionMessage(e)))
                })
            }
        }
        
        # Final validation
        if (is.null(n_samples)) {
            stop("Could not determine number of samples from any modality")
        }
        
        # Store number of samples
        self$n_samples <- n_samples
        
        # Verify feature consistency
        for (modality in names(self$data)) {
            if (!grepl("_mask$", modality)) {  # Skip masks
                if (is.null(self$features[[modality]])) {
                    warning(sprintf("Missing features for modality %s", modality))
                } else if (length(self$features[[modality]]) != dim(self$data[[modality]])[2]) {
                    warning(sprintf("Feature count mismatch for %s: %d features vs %d columns",
                                  modality,
                                  length(self$features[[modality]]),
                                  dim(self$data[[modality]])[2]))
                }
            }
        }
        
        # Final summary
        cat("\nDataset initialized successfully:\n")
        cat(sprintf("- Total samples: %d\n", self$n_samples))
        cat("- Available modalities:\n")
        for (modality in required_prefixes) {
            if (!is.null(self$data[[modality]])) {
                n_features <- dim(self$data[[modality]])[2]
                n_available <- sum(!is.nan(self$data[[modality]][1,]$cpu()$numpy()))
                cat(sprintf("  * %s: %d features (%d available)\n",
                           modality,
                           n_features,
                           n_available))
            }
        }
    }
)
