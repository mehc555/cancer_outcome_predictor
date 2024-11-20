# R/modules/data_handling/feature_selection.R

library(dplyr)
library(tidyr)

#' Select features for each modality based on specified criteria
#' @param data List of modality data frames/matrices
#' @param n_features Named list of number of features to select for each modality
#' @param outcome_info List containing outcome configuration
#' @return List of selected feature names
select_multimodal_features <- function(data, n_features, outcome_info = NULL) {
    selected_features <- list()
    
    cat("\nFeature counts per modality:\n")
    print(n_features)
    
    # Process each modality
    for (modality in names(data)) {
        if (grepl("_mask$|_features$", modality)) next
        
        message(sprintf("\nSelecting features for %s modality...", modality))
        
        n_features_to_select <- n_features[[modality]]
        cat(sprintf("\nWill select %d features for %s\n", n_features_to_select, modality))
        
        # Convert data frame to matrix, ensuring numeric conversion
        if (is.data.frame(data[[modality]])) {
            # Keep sample IDs separate
            sample_ids <- data[[modality]][, 1]
            
            # Convert remaining columns to numeric matrix
            mat <- as.matrix(sapply(data[[modality]][, -1, drop = FALSE], function(x) {
                # Try to convert to numeric, handling factors
                if (is.factor(x)) {
                    as.numeric(as.character(x))
                } else {
                    as.numeric(x)
                }
            }))
            
            # Preserve column names
            colnames(mat) <- colnames(data[[modality]])[-1]
            rownames(mat) <- sample_ids
            
        } else {
            mat <- as.matrix(data[[modality]])
        }
        
        cat(sprintf("\nWorking matrix dimensions: %d x %d\n", nrow(mat), ncol(mat)))
        
        # For clinical modality, handle outcome variables
        if (modality == "clinical" && !is.null(outcome_info)) {
            if (outcome_info$type == "survival") {
                exclude_vars <- c(outcome_info$time_var, outcome_info$event_var)
            } else if (outcome_info$type == "binary") {
                exclude_vars <- outcome_info$var
            }
            cat(sprintf("\nExcluding outcome variable(s): %s\n", 
                       paste(exclude_vars, collapse=", ")))
            
            if (!is.null(colnames(mat))) {
                keep_cols <- !colnames(mat) %in% exclude_vars
                mat <- mat[, keep_cols, drop = FALSE]
                cat(sprintf("Matrix dimensions after removing outcomes: %d x %d\n", 
                           nrow(mat), ncol(mat)))
            }
        }
        
        # Debug: Check for non-numeric values
        if (any(!is.numeric(mat))) {
            warning(sprintf("Non-numeric values found in %s matrix after conversion", modality))
            print(str(mat[, !sapply(mat, is.numeric), drop = FALSE]))
        }
        
        # Select features based on modality
        if (!is.null(n_features_to_select)) {
            selected <- switch(
                modality,
                clinical = select_clinical_features(mat, n_features_to_select),
                cnv = select_cnv_features(mat, n_features_to_select),
                expression = select_variable_features(mat, n_features_to_select, "expression"),
                mutations = select_mutation_features(mat, n_features_to_select),
                methylation = select_variable_features(mat, n_features_to_select, "methylation"),
                mirna = select_variable_features(mat, n_features_to_select, "mirna"),
                NULL
            )
            
            selected_features[[modality]] <- selected
            
            if (!is.null(selected)) {
                message(sprintf("Selected %d features for %s", 
                              length(selected), modality))
            }
        }
    }
    
    # Verify outcome variables are not in selected clinical features
    if (!is.null(outcome_info) && !is.null(selected_features$clinical)) {
        if (outcome_info$type == "binary") {
            overlap <- outcome_info$var %in% selected_features$clinical
            if (any(overlap)) {
                stop(sprintf("Found outcome variable %s in selected clinical features!", outcome_info$var))
            }
        } else if (outcome_info$type == "survival") {
            overlap <- c(outcome_info$time_var, outcome_info$event_var) %in% selected_features$clinical
            if (any(overlap)) {
                stop(sprintf("Found outcome variables in selected clinical features: %s", 
                    paste(c(outcome_info$time_var, outcome_info$event_var)[overlap], collapse=", ")))
            }
        }
        message("Verified: No outcome variables present in selected clinical features")
    }
    
    return(selected_features)
}

#' Select CNV features based on frequency
#' @param mat Matrix of CNV data
#' @param n_features Number of features to select
#' @return Vector of selected feature names
select_cnv_features <- function(mat, n_features) {
    # Calculate CNV event frequency
    cnv_freq <- apply(mat, 2, function(x) {
        # Remove NA values
        x <- x[!is.na(x)]
        if (length(x) == 0) return(0)
        # Calculate frequency of non-normal copy numbers (!=2)
        mean(x != 2, na.rm = TRUE)
    })
    
    # Remove any NA frequencies
    cnv_freq[is.na(cnv_freq) | is.nan(cnv_freq)] <- 0
    
    # Select top features
    selected <- names(sort(cnv_freq, decreasing = TRUE))[1:min(n_features, length(cnv_freq))]
    return(selected)
}


#' Select clinical features based on variance and frequency
#' @param mat Matrix of clinical data
#' @param n_features Number of features to select
#' @return Vector of selected feature names
select_clinical_features <- function(mat, n_features) {
    if (is.null(colnames(mat))) {
        stop("Clinical matrix must have column names")
    }
    
    cat(sprintf("\nSelecting %d clinical features from %d available\n", 
                n_features, ncol(mat)))
    
    # Calculate feature scores
    feature_scores <- sapply(1:ncol(mat), function(i) {
        x <- mat[, i]
        # Remove NA values
        x <- x[!is.na(x)]
        if (length(x) == 0) return(0)
        
        # Check if binary
        unique_vals <- unique(x[!is.na(x)])
        if (length(unique_vals) <= 2 && all(unique_vals %in% c(0, 1))) {
            # Binary features - use frequency of 1s
            mean(x == 1, na.rm = TRUE)
        } else {
            # Numeric features - use variance
            var(x, na.rm = TRUE)
        }
    })
    
    names(feature_scores) <- colnames(mat)
    
    # Remove any NaN scores
    feature_scores[is.na(feature_scores) | is.nan(feature_scores)] <- 0
    
    # Select top features
    selected <- names(sort(feature_scores, decreasing = TRUE))[1:min(n_features, length(feature_scores))]
    return(selected)
}


#' Select features based on variance
#' @param mat Matrix of data
#' @param n_features Number of features to select
#' @param modality_name Name of modality (for logging)
#' @return Vector of selected feature names
select_variable_features <- function(mat, n_features, modality_name) {
    # Calculate variance for each feature
    feature_vars <- apply(mat, 2, function(x) {
        # Remove NA values
        x <- x[!is.na(x)]
        if (length(x) == 0) return(0)
        var(x, na.rm = TRUE)
    })

    # Remove features with NA or zero variance
    feature_vars[is.na(feature_vars) | is.nan(feature_vars) | feature_vars == 0] <- 0

    # Select top features
    selected <- names(sort(feature_vars, decreasing = TRUE))[1:min(n_features, length(feature_vars))]
    return(selected)
}

#' Select mutation features based on frequency
#' @param mat Matrix of mutation data
#' @param n_features Number of features to select
#' @return Vector of selected feature names
select_mutation_features <- function(mat, n_features) {
    # Calculate mutation frequency
    mutation_freq <- apply(mat, 2, function(x) {
        # Remove NA values
        x <- x[!is.na(x)]
        if (length(x) == 0) return(0)
        # Calculate frequency of mutations (!=0)
        mean(x != 0, na.rm = TRUE)
    })
    
    # Remove any NA frequencies
    mutation_freq[is.na(mutation_freq) | is.nan(mutation_freq)] <- 0
    
    # Select top features
    selected <- names(sort(mutation_freq, decreasing = TRUE))[1:min(n_features, length(mutation_freq))]
    return(selected)
}

#' Apply selected features to dataset
#' @param dataset MultiModalDataset object
#' @param selected_features List of selected feature names for each modality
#' @return Updated MultiModalDataset with only selected features

#' Apply selected features to dataset with special handling for clinical data
#' @param dataset MultiModalDataset object
#' @param selected_features List of selected feature names for each modality
#' @return Updated MultiModalDataset with only selected features

apply_feature_selection <- function(dataset, selected_features) {
    new_data <- list()
    new_features <- list()
    
    # Process each modality
    for (modality in names(selected_features)) {
        if (is.null(dataset$data[[modality]])) {
            message(sprintf("Skipping %s - no data found", modality))
            next
        }
        
        message(sprintf("\nProcessing %s modality", modality))
        message(sprintf("Data class: %s", class(dataset$data[[modality]])[1]))
        
        # Get feature indices (including sample_id)
        feature_cols <- c("sample_id", selected_features[[modality]])
        
        # Handle data frame input
        if (is.data.frame(dataset$data[[modality]])) {
            # Verify columns exist
            missing_cols <- setdiff(feature_cols, colnames(dataset$data[[modality]]))
            if (length(missing_cols) > 0) {
                warning(sprintf("Missing columns in %s: %s", 
                              modality, paste(missing_cols, collapse=", ")))
                next
            }
            
            # Select columns for data
            new_data[[modality]] <- dataset$data[[modality]][, feature_cols, drop=FALSE]
            
            # Handle corresponding mask
            mask_name <- paste0(modality, "_mask")
            if (!is.null(dataset$data[[mask_name]])) {
                # Get indices for selected features (excluding sample_id)
                mask_indices <- match(selected_features[[modality]], 
                                    colnames(dataset$data[[modality]])[-1])
                
                # Select corresponding mask columns
                new_data[[mask_name]] <- dataset$data[[mask_name]][, mask_indices, drop=FALSE]
                
                message(sprintf("Selected mask for %s: %.1f%% valid values", 
                              modality,
                              100 * mean(new_data[[mask_name]], na.rm=TRUE)))
            }
        } else if (inherits(dataset$data[[modality]], "torch_tensor")) {
            # Handle tensor input
            feature_indices <- match(selected_features[[modality]], 
                                   dataset$features[[modality]])
            
            # Add 1 for sample_id column
            feature_indices <- c(1, feature_indices + 1)
            
            # Select features from tensor
            torch_indices <- torch_tensor(feature_indices, dtype = torch_long())
            new_data[[modality]] <- torch_index_select(
                dataset$data[[modality]],
                dim = 2,
                index = torch_indices
            )
            
            # Handle corresponding mask
            mask_name <- paste0(modality, "_mask")
            if (!is.null(dataset$data[[mask_name]])) {
                new_data[[mask_name]] <- torch_index_select(
                    dataset$data[[mask_name]],
                    dim = 2,
                    index = torch_tensor(feature_indices[-1], dtype = torch_long())
                )
            }
        }
        
        # Store selected feature names
        new_features[[modality]] <- selected_features[[modality]]
        
        message(sprintf("Selected %d features for %s", 
                       length(selected_features[[modality]]), modality))
    }
    
    # Create new dataset
    new_dataset <- dataset(
        name = "MultiModalDataset",
        initialize = function() {
            self$data <- new_data
            self$features <- new_features
            self$n_samples <- dataset$n_samples
            self$sample_ids <- dataset$sample_ids
            self$unified_sample_ids <- dataset$unified_sample_ids
            self$sample_id_to_index <- dataset$sample_id_to_index
            self$outcome_info <- dataset$outcome_info
            self$outcomes <- dataset$outcomes
        },
        .getitem = dataset$.getitem,
        .length = function() self$n_samples,
        get_feature_names = dataset$get_feature_names,
        get_sample_ids = dataset$get_sample_ids
    )()
    
    return(new_dataset)
}


