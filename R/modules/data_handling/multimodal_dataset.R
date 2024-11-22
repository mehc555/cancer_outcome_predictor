library(torch)

MultiModalDataset <- dataset(
  name = "MultiModalDataset",
  
  initialize = function(data, outcome_info = NULL) {
    # Store initial data and setup
    self$data <- data
    self$features <- list()
    
    # Get reference samples from clinical data
    if (is.null(data$clinical)) {
      stop("Clinical data is required as the reference for sample alignment")
    }
    reference_samples <- as.character(data$clinical$sample_id)
    self$n_samples <- length(reference_samples)
    
    cat(sprintf("\nAligning all modalities to %d clinical samples...\n", self$n_samples))
    
    # Store features for each modality
    modalities <- c("clinical", "cnv", "expression", "mutations", "methylation", "mirna")
    
    # Align and pad each modality using vectorized operations
    for (modality in modalities) {
      if (!is.null(self$data[[modality]])) {
        # Store features
        self$features[[modality]] <- colnames(self$data[[modality]])[-1]
        
        # Get current data and samples
        current_data <- self$data[[modality]]
        current_samples <- as.character(current_data$sample_id)
        
        # Create new data matrix with NAs
        feature_cols <- self$features[[modality]]
        aligned_data <- matrix(NA, 
                             nrow = length(reference_samples), 
                             ncol = length(feature_cols))
        colnames(aligned_data) <- feature_cols
        
        # Find matching indices in one go
        match_idx <- match(reference_samples, current_samples)
        
        # Fill data in one operation
        valid_idx <- !is.na(match_idx)
        if (any(valid_idx)) {
          aligned_data[valid_idx,] <- as.matrix(current_data[match_idx[valid_idx], feature_cols])
        }
        
        # Create new dataframe with aligned data
        self$data[[modality]] <- data.frame(
          sample_id = reference_samples,
          aligned_data,
          stringsAsFactors = FALSE,
          check.names = FALSE
        )
        
        # Create mask as a matrix with row numbers preserved
        mask_matrix <- !is.na(aligned_data)
        rownames(mask_matrix) <- seq_len(nrow(mask_matrix))
        self$data[[paste0(modality, "_mask")]] <- mask_matrix
        
        # Print alignment summary
        n_complete <- sum(complete.cases(aligned_data))
        cat(sprintf("- %s: %d/%d complete samples (%.1f%%)\n", 
                   modality, n_complete, self$n_samples, 
                   100 * n_complete/self$n_samples))
      }
    }
    
    # Create sample index mapping
    self$sample_id_to_index <- setNames(seq_along(reference_samples), reference_samples)
    
    # Process outcomes if provided
    if (!is.null(outcome_info)) {
      if (outcome_info$type == "binary") {
        self$outcomes <- list(
          binary =as.numeric(self$data$clinical[[outcome_info$var]])
        )
      } else if (outcome_info$type == "survival") {
        self$outcomes <- list(
          time = as.numeric(self$data$clinical[[outcome_info$time_var]]),
          event = as.numeric(self$data$clinical[[outcome_info$event_var]])
        )
      }
    }
    
    # Print final dataset dimensions
    cat("\nDataset dimensions after alignment:\n")
    for (modality in modalities) {
      if (!is.null(self$data[[modality]])) {
        dimensions <- dim(self$data[[modality]])
        cat(sprintf("- %s: %s\n", modality, paste(dimensions, collapse=" x ")))
      }
    }
  },
  
  .getitem = function(index) {
    # Initialize data and mask lists
    data_list <- list()
    masks_list <- list()
    
    # Get current sample ID
    current_sample_id <- names(self$sample_id_to_index)[index]
    
    # Process each modality
    for (modality in names(self$data)) {
      if (!is.null(self$data[[modality]]) && !grepl("_mask$", modality)) {
        # Get data for current sample
        data_values <- as.matrix(self$data[[modality]][index, -1, drop = FALSE])
        
        # Convert to tensors
        data_list[[modality]] <- torch_tensor(
          data_values,
          dtype = torch_float32()
        )
        
        # Get mask for current sample (preserving row number)
        mask_values <- self$data[[paste0(modality, "_mask")]][index, , drop = FALSE]
        masks_list[[modality]] <- torch_tensor(
          mask_values,
          dtype = torch_float32()
        )
        
        # Replace NA values with 0 in data tensor
        data_list[[modality]][is.na(data_list[[modality]])] <- 0
      }
    }
    
    # Create batch
    batch <- list(
      sample_id = current_sample_id,
      data = data_list,
      masks = masks_list,
      features = self$features
    )
    
    # Add outcomes if they exist
    if (!is.null(self$outcomes)) {
      batch$outcomes <- lapply(self$outcomes, function(outcome) {
        outcome[index]
      })
    }
    
    return(batch)
  },
  
  .length = function() {
    self$n_samples
  }
)

#' Create torch datasets with proper outcome handling
#' @param data_list List of data matrices/dataframes for each modality
#' @param config Model configuration
#' @param outcome_info List specifying outcome variable(s)
#' @return MultiModalDataset object

create_torch_datasets <- function(data_list, config, outcome_info = NULL) {
    # Print initial dataset information
    cat("Creating torch datasets with the following data:\n")
    for (name in names(data_list)) {
        if (!is.null(data_list[[name]])) {
            cat(sprintf("- %s: %s (%s)\n",
                       name,
                       paste(dim(data_list[[name]]), collapse="x"),
                       class(data_list[[name]])[1]))
        }
    }
    
    # Create masks for each modality
    masked_data <- list()
    for (modality in names(data_list)) {
        if (!is.null(data_list[[modality]])) {
            # Store original data
            masked_data[[modality]] <- data_list[[modality]]
            
            # Create mask (1 for data, 0 for NA)
            mask <- !is.na(as.matrix(data_list[[modality]][,-1])) # exclude sample_id
            masked_data[[paste0(modality, "_mask")]] <- mask
            
            cat(sprintf("Created mask for %s: %d x %d with %.1f%% valid values\n",
                       modality,
                       nrow(mask),
                       ncol(mask),
                       100 * mean(mask, na.rm=TRUE)))
        }
    }
    
    # Create dataset with masks
    dataset <- MultiModalDataset(
        data = masked_data,
        outcome_info = outcome_info
    )
    
    return(dataset)
}


update_dimensions <- function(new_dims) {
  # Update modality dimensions
  self$modality_dims <- new_dims
  
  # Recreate encoders with new dimensions
  encoders_dict <- list()
  for (name in names(new_dims)) {
    # Calculate new encoder dimensions maintaining the same ratios
    orig_dims <- self$encoder_dims[[name]]
    dim_ratios <- orig_dims / self$modality_dims[[name]]
    new_encoder_dims <- ceiling(new_dims[[name]] * dim_ratios)
    
    # Create new encoder
    encoders_dict[[name]] <- EnhancedModalityEncoder(
      input_dim = new_dims[[name]],
      hidden_dims = new_encoder_dims,
      num_heads = self$num_heads,
      dropout = self$dropout
    )
  }
  
  # Update encoder dictionary
  self$encoders <- nn_module_dict(encoders_dict)
  
  # Update fusion module if needed
  final_encoder_dims <- sapply(new_encoder_dims, function(x) x[length(x)])
  self$fusion <- ModalityFusion(
    modality_dims = final_encoder_dims,
    fusion_dim = self$fusion_dim,
    num_heads = self$num_heads,
    dropout = self$dropout
  )
}
