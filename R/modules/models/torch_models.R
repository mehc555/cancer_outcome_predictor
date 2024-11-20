# R/modules/models/torch_models.R

library(torch)

# Simplified encoder block without attention
EncoderBlock <- nn_module(
  "EncoderBlock",
  initialize = function(input_dim, output_dim, dropout = 0.1) {
    self$layer <- nn_sequential(
      nn_linear(input_dim, output_dim),
      nn_layer_norm(output_dim, output_dim),
      nn_relu(),
      nn_dropout(dropout)
    )
  },
  
  forward = function(x) {
    # Replace NaNs with zeros
    x <- torch_where(
      torch_isnan(x),
      torch_zeros_like(x),
      x
    )
    self$layer(x)
  }
)

MLPBlock <- nn_module(
  "MLPBlock",
  initialize = function(dim, expansion_factor = 4, dropout = 0.1) {
    cat(sprintf("Initializing MLPBlock: dim=%d, expansion=%d\n", dim, expansion_factor))
    
    hidden_dim <- dim * expansion_factor
    self$net <- nn_sequential(
      nn_linear(dim, hidden_dim),
      nn_gelu(),
      nn_dropout(dropout),
      nn_linear(hidden_dim, dim),
      nn_dropout(dropout)
    )
    self$norm <- nn_layer_norm(dim, dim)
  },
  
  forward = function(x) {
    # Print detailed shape information
    cat(sprintf("\nMLPBlock forward:\n"))
    cat(sprintf("Input shape: %s\n", paste(x$size(), collapse=" x ")))
    
    residual <- x
    cat("Applying norm...\n")
    x <- self$norm(x)
    cat(sprintf("After norm shape: %s\n", paste(x$size(), collapse=" x ")))
    
    cat("Applying network...\n")
    x <- self$net(x)
    cat(sprintf("After net shape: %s\n", paste(x$size(), collapse=" x ")))
    
    cat("Adding residual...\n")
    x <- x + residual
    cat(sprintf("Final shape: %s\n", paste(x$size(), collapse=" x ")))
    x
  }
)

ModalityFusion <- nn_module(
  "ModalityFusion",
  initialize = function(modality_dims, fusion_dim, dropout = 0.1) {
    cat("\nInitializing ModalityFusion:\n")
    cat("Input dimensions per modality:\n")
    print(modality_dims)
    cat("Fusion dimension:", fusion_dim, "\n")

    # Create projections for each modality
    projections_dict <- list()
    for (name in names(modality_dims)) {
      cat(sprintf("Creating projection for %s: %d -> %d\n",
                 name, modality_dims[[name]], fusion_dim))

      # Since encoders output fusion_dim, no need for initial projection
      projections_dict[[name]] <- MLPBlock(fusion_dim, dropout = dropout)
    }

    self$modality_projections <- nn_module_dict(projections_dict)
    self$fusion_mlp <- MLPBlock(fusion_dim, dropout = dropout)
    self$fusion_norm <- nn_layer_norm(fusion_dim, fusion_dim)
    self$dropout <- nn_dropout(dropout)
    self$fusion_dim <- fusion_dim  # Store for reference
  },

  forward = function(modality_features, masks = NULL) {
    cat("\nModalityFusion forward pass:\n")
    projected_features <- list()
    valid_modality_count <- 0

    for (name in names(modality_features)) {
      if (!is.null(modality_features[[name]])) {
        cat(sprintf("\nProcessing %s:\n", name))
        cat(sprintf("Input shape: %s\n", paste(modality_features[[name]]$size(), collapse=" x ")))

        # Apply mask if available
        current_features <- modality_features[[name]]
        if (!is.null(masks) && !is.null(masks[[name]])) {
          current_features <- current_features * masks[[name]]
        }

        # Verify dimensions
        if (!all(current_features$size()[-1] == self$fusion_dim)) {
          stop(sprintf("Expected features of dimension %d for %s, but got %s",
                      self$fusion_dim, name, paste(current_features$size()[-1], collapse=" x ")))
        }

        projected <- self$modality_projections[[name]](current_features)
        cat(sprintf("Projected shape: %s\n", paste(projected$size(), collapse=" x ")))

        projected_features[[name]] <- projected
        valid_modality_count <- valid_modality_count + 1
      }
    }

    if (valid_modality_count == 0) {
      stop("No valid modalities found for fusion")
    }

    cat("\nStacking features...\n")
    # Convert list to tensor
    feature_names <- names(projected_features)
    feature_tensors <- lapply(feature_names, function(name) projected_features[[name]])
    feature_stack <- torch_stack(feature_tensors, dim = 2)
    cat(sprintf("Stacked shape: %s\n", paste(feature_stack$size(), collapse=" x ")))

    cat("\nAveraging features...\n")
    fused_features <- feature_stack$mean(dim = 2)
    cat(sprintf("Mean shape: %s\n", paste(fused_features$size(), collapse=" x ")))

    cat("\nFinal processing...\n")
    fused_features <- self$fusion_mlp(fused_features)
    fused_features <- self$fusion_norm(fused_features)
    fused_features <- self$dropout(fused_features)
    cat(sprintf("Final shape: %s\n", paste(fused_features$size(), collapse=" x ")))

    list(features = fused_features)
  }
)

# Modified ModalityEncoder to properly use masks
ModalityEncoder <- nn_module(
  "ModalityEncoder",
  initialize = function(input_dim, hidden_dims, dropout = 0.1) {
    self$layers <- nn_module_list()
    
    current_dim <- input_dim
    for (dim in hidden_dims) {
      self$layers$append(nn_sequential(
        nn_linear(current_dim, dim),
        nn_batch_norm1d(dim),
        nn_relu(),
        nn_dropout(dropout)
      ))
      current_dim <- dim
    }
  },
  
  forward = function(x, mask = NULL) {
    # Apply mask if provided
    if (!is.null(mask)) {
      x <- x * mask
    }
    
    # Replace NaN values with zeros
    x <- torch_where(torch_isnan(x), torch_zeros_like(x), x)
    
    for (layer in self$layers) {
      x <- layer(x)
    }
    x
  }
)


MultiModalSurvivalModel <- nn_module(
  "MultiModalSurvivalModel",
  initialize = function(modality_dims, encoder_dims, fusion_dim, 
                       dropout = 0.1, outcome_type = "binary") {
    cat("\nInitializing MultiModalSurvivalModel:\n")
    cat("Modality dimensions:\n")
    print(modality_dims)
    cat("Fusion dimension:", fusion_dim, "\n")
    
    self$outcome_type <- outcome_type
    self$modality_dims <- modality_dims
    self$fusion_dim <- fusion_dim
    self$dropout <- dropout
    
    # Create encoders for each modality
    encoders_dict <- list()
    modality_fusion_dims <- list()
    
    for (name in names(modality_dims)) {
      cat(sprintf("\nCreating encoder for %s:\n", name))
      cat(sprintf("Input dim: %d, Fusion dim: %d\n", modality_dims[[name]], fusion_dim))
      
      encoders_dict[[name]] <- ModalityEncoder(
        input_dim = modality_dims[[name]],
        hidden_dims = c(modality_dims[[name]], fusion_dim),
        dropout = dropout
      )
      modality_fusion_dims[[name]] <- fusion_dim
    }
    
    self$encoders <- nn_module_dict(encoders_dict)
    
    cat("\nCreating fusion module...\n")
    self$fusion <- ModalityFusion(
      modality_dims = modality_fusion_dims,
      fusion_dim = fusion_dim,
      dropout = dropout
    )
    
    cat("\nCreating prediction head...\n")
    self$prediction_head <- nn_sequential(
      nn_linear(fusion_dim, fusion_dim %/% 2),
      nn_layer_norm(fusion_dim %/% 2, fusion_dim %/% 2),
      nn_relu(),
      nn_dropout(dropout),
      nn_linear(fusion_dim %/% 2, 1)
    )
  },
  
  forward = function(x, masks = NULL) {
    # Process each modality with mask handling
    encoded_features <- list()
    modality_names <- names(self$modality_dims)
    
    cat("\nForward pass through MultiModalSurvivalModel:\n")
    cat("Processing modalities...\n")
    
    for (name in modality_names) {
      if (!is.null(x[[name]])) {
        cat(sprintf("\nProcessing %s:\n", name))
        cat(sprintf("Input shape: %s\n", paste(x[[name]]$size(), collapse=" x ")))
        
        # Get mask for current modality if available
        current_mask <- if (!is.null(masks) && !is.null(masks[[name]])) {
          cat(sprintf("Using mask for %s\n", name))
          masks[[name]]
        } else {
          cat(sprintf("No mask for %s\n", name))
          NULL
        }
        
        # Encode features
        encoded <- self$encoders[[name]](x[[name]], current_mask)
        cat(sprintf("Encoded %s shape: %s\n", name, paste(encoded$size(), collapse=" x ")))
        
        # Verify encoding dimension
        if (encoded$size(2) != self$fusion_dim) {
          stop(sprintf("Encoder output dimension mismatch for %s. Expected %d, got %d", 
                      name, self$fusion_dim, encoded$size(2)))
        }
        
        encoded_features[[name]] <- encoded
      } else {
        cat(sprintf("Skipping %s - no data\n", name))
      }
    }
    
    # Verify we have features to fuse
    if (length(encoded_features) == 0) {
      stop("No features to fuse - all modalities were empty")
    }
    
    cat("\nFusing modalities...\n")
    cat("Number of modalities to fuse:", length(encoded_features), "\n")
    for (name in names(encoded_features)) {
      cat(sprintf("%s shape: %s\n", name, 
                 paste(encoded_features[[name]]$size(), collapse=" x ")))
    }
    
    # Fuse modalities
    fusion_result <- self$fusion(encoded_features, masks)
    fused_features <- fusion_result$features
    
    cat(sprintf("\nFused features shape: %s\n", 
                paste(fused_features$size(), collapse=" x ")))
    
    # Generate predictions
    cat("\nGenerating predictions...\n")
    predictions <- self$prediction_head(fused_features)
    
    cat(sprintf("Predictions shape: %s\n", 
                paste(predictions$size(), collapse=" x ")))
    
    # Final NaN cleanup
    predictions <- torch_where(
      torch_isnan(predictions),
      torch_zeros_like(predictions),
      predictions
    )
    
    list(predictions = predictions)
  },
  
  create_copy = function() {
    cat("\nCreating model copy...\n")
    new_model <- MultiModalSurvivalModel(
      modality_dims = self$modality_dims,
      encoder_dims = NULL,  # Not used in current implementation
      fusion_dim = self$fusion_dim,
      dropout = self$dropout,
      outcome_type = self$outcome_type
    )
    new_model$load_state_dict(self$state_dict())
    return(new_model)
  }
)


# Helper function for debugging
print_tensor_info <- function(tensor, name) {
    cat(sprintf("%s shape: %s\n", name, paste(tensor$size(), collapse=" x ")))
}

# BCE loss with masking
masked_bce_loss <- function(predictions, targets, masks) {
  predictions <- torch_where(
    torch_isnan(predictions),
    torch_zeros_like(predictions),
    predictions
  )
  
  criterion <- nn_bce_with_logits_loss(reduction = 'none')
  loss <- criterion(predictions, targets)
  
  if (!is.null(masks)) {
    loss <- loss * masks
  }
  
  valid_loss <- loss[!torch_isnan(loss)]
  if (valid_loss$numel() > 0) {
    return(torch_mean(valid_loss))
  } else {
    return(torch_tensor(0.0))
  }
}
