library(torch)

MultiModalDataset <- dataset(
  name = "MultiModalDataset",
  
  initialize = function(data) {
    # Validate input data structure
    required_prefixes <- c("clinical", "cnv", "expression", "mutations", "methylation", "mirna")
    if (!all(sapply(required_prefixes, function(x) any(grepl(paste0("^", x), names(data)))))) {
      stop("Missing required data modalities")
    }
    
    # Store data
    self$data <- data
    self$features <- list()  # Store feature names for each modality
    self$sample_ids <- list()  # Store sample IDs for each modality
    
    # First pass: collect all sample IDs and features
    all_sample_ids <- list()
    for (name in names(self$data)) {
      if (!grepl("_features$|_mask$", name)) {
        all_sample_ids[[name]] <- as.character(self$data[[name]][,1])
        self$features[[name]] <- colnames(self$data[[name]])[-1]  # Exclude sample ID column
      }
    }
    
    # Create unified sample ID list
    self$unified_sample_ids <- unique(unlist(all_sample_ids))
    self$n_samples <- length(self$unified_sample_ids)
    
    # Create sample ID to index mapping
    self$sample_id_to_index <- setNames(seq_along(self$unified_sample_ids), self$unified_sample_ids)
    
    # Second pass: convert data to tensors with proper alignment
    for (name in names(self$data)) {
      # Skip feature names and masks
      if (grepl("_features$|_mask$", name)) {
        next
      }
      
      if (!inherits(self$data[[name]], "torch_tensor")) {
        tryCatch({
          # Get current modality's sample IDs
          current_sample_ids <- as.character(self$data[[name]][,1])
          self$sample_ids[[name]] <- current_sample_ids
          
          # Create data matrix
          mat_data <- as.matrix(self$data[[name]][,-1])  # Remove first column
          
          # Handle different data types
          if (!is.numeric(mat_data)) {
            mat_data <- apply(mat_data, 2, function(x) {
              if (is.character(x) || is.factor(x)) {
                as.numeric(as.factor(x)) - 1
              } else {
                as.numeric(x)
              }
            })
          }
          
          # Create full-size matrix with NaN for missing samples
          n_features <- ncol(mat_data)
          full_mat <- matrix(NaN, nrow=self$n_samples, ncol=n_features)
          
          # Fill in available data
          sample_indices <- self$sample_id_to_index[current_sample_ids]
          full_mat[sample_indices,] <- mat_data
          
          # Create corresponding mask (1 where data is present, 0 where missing)
          mask <- !is.na(full_mat)
          
          # Convert to tensors
          self$data[[name]] <- torch_tensor(full_mat, dtype=torch_float32())
          mask_name <- paste0(name, "_mask")
          self$data[[mask_name]] <- torch_tensor(mask, dtype=torch_float32())
          
        }, error = function(e) {
          warning(sprintf("Could not convert %s to tensor: %s", name, e$message))
          self$data[[name]] <- NULL
          self$features[[name]] <- NULL
          self$sample_ids[[name]] <- NULL
        })
      }
    }
    
    # Debug info
    cat("Dataset initialized with", self$n_samples, "unique samples\n")
    cat("Available modalities:\n")
    for (name in names(self$data)) {
      if (!grepl("_features$|_mask$", name) && !is.null(self$data[[name]])) {
        n_available <- length(self$sample_ids[[name]])  # Use original sample count
        cat(sprintf("- %s: %dx%d (%d features, %d samples available)\n", 
                   name, 
                   self$n_samples,
                   length(self$features[[name]]),
                   length(self$features[[name]]),
                   n_available))
      }
    }
  },
  
  .getitem = function(index) {
    # Get sample ID for this index
    sample_id <- self$unified_sample_ids[index]
    
    # Initialize containers for batch data and masks
    batch_data <- list()
    batch_masks <- list()
    
    # Get modality names (excluding _features and _mask suffixes)
    modality_names <- unique(gsub("(_features|_mask)$", "", names(self$data)))
    
    # Process each modality
    for (modality in modality_names) {
      # Skip feature lists
      if (grepl("_features$|_mask$", modality)) next
      
      # Get data tensor and mask
      data_name <- modality
      mask_name <- paste0(modality, "_mask")
      
      if (!is.null(self$data[[data_name]])) {
        # Extract single sample
        batch_data[[modality]] <- self$data[[data_name]][index,]
        batch_masks[[modality]] <- self$data[[mask_name]][index,]
      }
    }
    
    # Get target variables from clinical data
    target <- list(
      time = self$data$clinical[index, which(self$features$clinical == "survival_time")],
      event = self$data$clinical[index, which(self$features$clinical == "demographics_vital_status_alive")]
    )
    
    list(
      sample_id = sample_id,
      data = batch_data,
      masks = batch_masks,
      features = self$features,
      time = target$time,
      event = target$event
    )
  },
  
  .length = function() {
    self$n_samples
  },
  
  get_feature_names = function(modality) {
    if (!is.null(self$features[[modality]])) {
      return(self$features[[modality]])
    } else {
      warning(sprintf("No features found for modality: %s", modality))
      return(NULL)
    }
  },
  
  get_sample_ids = function(modality = NULL) {
    if (is.null(modality)) {
      return(self$unified_sample_ids)
    } else if (!is.null(self$sample_ids[[modality]])) {
      return(self$sample_ids[[modality]])
    }
    warning("No sample IDs found")
    return(NULL)
  }
)

# Helper function to create torch datasets
create_torch_datasets <- function(data_list, config) {
  cat("Creating torch datasets with the following data:\n")
  for (name in names(data_list)) {
    if (!is.null(data_list[[name]])) {
      cat(sprintf("- %s: %s (%s)\n", 
                 name, 
                 paste(dim(data_list[[name]]), collapse="x"),
                 class(data_list[[name]])[1]))
    }
  }
  
  dataset <- MultiModalDataset(data_list)
  return(dataset)
}

update_dimensions = function(new_dims) {
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
