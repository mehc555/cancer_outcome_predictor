# R/modules/data_handling/feature_selection.R

library(dplyr)
library(tidyr)

#' Select features for each modality based on specified criteria
#' @param data List of modality data frames/matrices
#' @param n_features Named list of number of features to select for each modality
#' @return List of selected feature names for each modality

select_multimodal_features <- function(data, n_features = list(
    cnv = 512,
    clinical = 64,
    expression = 512,
    mutations = 64,
    methylation = 512,
    mirna = 512
)) {
    selected_features <- list()
    
    # Process each modality
    for (modality in names(data)) {
        if (grepl("_mask$|_features$", modality)) next
        
        message(sprintf("Selecting features for %s modality...", modality))
        print(sprintf("Matrix dimensions for %s: %s", modality, 
                     paste(dim(as.matrix(data[[modality]]$cpu())), collapse="x")))
        print(sprintf("Number of features requested for %s: %s", modality, 
                     n_features[[modality]]))
        
        # Get the data matrix and convert tensor to matrix if needed
        if (inherits(data[[modality]], "torch_tensor")) {
            mat <- as.matrix(data[[modality]]$cpu())
            # Use parent object's features list
            feature_names <- train_data$features[[modality]]
            if (!is.null(feature_names)) {
                print(sprintf("Found %d feature names for %s", length(feature_names), modality))
                colnames(mat) <- feature_names
            } else {
                print(sprintf("No feature names found for %s", modality))
            }
        } else {
            mat <- as.matrix(data[[modality]])
        }
        
        # Select features based on modality type
        selected_features[[modality]] <- switch(
            modality,
            cnv = select_cnv_features(mat, n_features$cnv),
            clinical = select_clinical_features(mat, n_features$clinical),
            expression = select_variable_features(mat, n_features$expression, "expression"),
            mutations = select_mutation_features(mat, n_features$mutations),
            methylation = select_variable_features(mat, n_features$methylation, "methylation"),
            mirna = select_variable_features(mat, n_features$mirna, "mirna"),
            NULL
        )
        
        if (!is.null(selected_features[[modality]])) {
            message(sprintf("Selected %d features for %s", 
                          length(selected_features[[modality]]), 
                          modality))
        }
    }
    
    return(selected_features)
}

#' Select CNV features based on frequency
#' @param mat Matrix of CNV data
#' @param n_features Number of features to select
#' @return Vector of selected feature names

select_cnv_features <- function(mat, n_features) {
    # Calculate frequency of CNV events (any deviation from 2 copies), ignoring NAs
    cnv_freq <- apply(mat, 2, function(x) {
        # Remove NaN/NA values
        x_clean <- x[!is.na(x) & !is.nan(x)]
        if(length(x_clean) == 0) return(0)
        # Calculate frequency of non-normal copy numbers (!=2)
        mean(x_clean != 2)
    })
    
    # Remove any remaining NA frequencies (shouldn't happen but just in case)
    cnv_freq[is.na(cnv_freq)] <- 0
    
    # Select top features
    selected <- names(sort(cnv_freq, decreasing = TRUE))[1:min(n_features, length(cnv_freq))]
    return(selected)
}

#' Select clinical features based on variance and frequency
#' @param mat Matrix of clinical data
#' @param n_features Number of features to select
#' @return Vector of selected feature names
select_clinical_features <- function(mat, n_features) {
    # Calculate variance for numeric columns and frequency for categorical
    feature_scores <- apply(mat, 2, function(x) {
        if (all(x %in% c(0, 1))) {
            # Binary features - use frequency of 1s
            mean(x)
        } else {
            # Numeric features - use variance
            var(x, na.rm = TRUE)
        }
    })
    
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
    feature_vars <- apply(mat, 2, var, na.rm = TRUE)
    
    # Remove features with NA or zero variance
    feature_vars <- feature_vars[!is.na(feature_vars) & feature_vars > 0]
    
    # Select top features
    selected <- names(sort(feature_vars, decreasing = TRUE))[1:min(n_features, length(feature_vars))]
    return(selected)
}

#' Select mutation features based on frequency
#' @param mat Matrix of mutation data
#' @param n_features Number of features to select
#' @return Vector of selected feature names

select_mutation_features <- function(mat, n_features) {
    # Calculate mutation frequency, ignoring NAs
    mutation_freq <- apply(mat, 2, function(x) {
        # Remove NaN/NA values
        x_clean <- x[!is.na(x) & !is.nan(x)]
        if(length(x_clean) == 0) return(0)
        # Calculate frequency of mutations (!=0)
        mean(x_clean != 0)
    })
    
    # Remove any remaining NA frequencies
    mutation_freq[is.na(mutation_freq)] <- 0
    
    # Select top features
    selected <- names(sort(mutation_freq, decreasing = TRUE))[1:min(n_features, length(mutation_freq))]
    return(selected)
}

#' Apply selected features to dataset
#' @param dataset MultiModalDataset object
#' @param selected_features List of selected feature names for each modality
#' @return Updated MultiModalDataset with only selected features
apply_feature_selection <- function(dataset, selected_features) {
    # Create new data list
    new_data <- list()
    
    # Process each modality
    for (modality in names(selected_features)) {
        if (is.null(dataset$data[[modality]])) next
        
        # Get feature indices
        feature_indices <- match(selected_features[[modality]], 
                               dataset$features[[modality]])
        
        # Select features from data tensor
        new_data[[modality]] <- torch_index_select(
            dataset$data[[modality]],
            dim = 2,
            index = torch_tensor(feature_indices - 1, dtype = torch_long())
        )
        
        # Update feature names
        new_data[[paste0(modality, "_features")]] <- selected_features[[modality]]
        
        # Update masks if they exist
        mask_name <- paste0(modality, "_mask")
        if (!is.null(dataset$data[[mask_name]])) {
            new_data[[mask_name]] <- torch_index_select(
                dataset$data[[mask_name]],
                dim = 2,
                index = torch_tensor(feature_indices - 1, dtype = torch_long())
            )
        }
    }
    
    # Create new dataset
    new_dataset <- MultiModalDataset$new(new_data)
    return(new_dataset)
}
