# R/modules/models/torch_models.R

library(torch)

# First add the new EncoderBlock

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
  
  forward = function(x, mask = NULL) {
    # Replace NaNs with zeros using torch_where
    x <- torch_where(
      torch_isnan(x),
      torch_zeros_like(x),
      x
    )
    
    # Apply mask if provided
    if (!is.null(mask)) {
      x <- x * mask
    }
    
    self$layer(x)
  }
)



#' MLP block for feature transformation
#' @param dim Input dimension
#' @param expansion_factor Expansion factor for hidden dimension
#' @param dropout Dropout rate

# Updated MLPBlock for consistency
MLPBlock <- nn_module(
  "MLPBlock",
  initialize = function(dim, expansion_factor = 4, dropout = 0.1) {
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
    residual <- x
    x <- self$norm(x)
    x <- self$net(x)
    x + residual  # Residual connection
  }
)


#' Enhanced self-attention module with MLP
#' @param dim Input dimension
#' @param num_heads Number of attention heads
#' @param dropout Dropout rate

# Update the SelfAttention forward method
SelfAttention <- nn_module(
  "SelfAttention",
  initialize = function(dim, num_heads = 4, dropout = 0.1) {
    self$dim <- dim
    self$num_heads <- num_heads
    self$head_dim <- dim %/% num_heads
    self$scaling <- self$head_dim ^ (-0.5)
    
    self$query <- nn_linear(dim, dim)
    self$key <- nn_linear(dim, dim)
    self$value <- nn_linear(dim, dim)
    
    self$out_proj <- nn_linear(dim, dim)
    self$dropout <- nn_dropout(dropout)
    self$norm <- nn_layer_norm(dim, dim)
  },

  forward = function(x, mask = NULL, debug = FALSE) {
    # Get original dimensions and reshape if needed
    input_dims <- x$size()
    batch_size <- input_dims[1]
    is_2d <- length(input_dims) == 2
    
    if (is_2d) {
      # For 2D input, treat each feature as a sequence element
      seq_len <- input_dims[2] %/% self$dim
      x <- x$view(c(batch_size, seq_len, self$dim))
    } else {
      seq_len <- input_dims[2]
    }
    
    residual <- x
    x <- self$norm(x)
    
    # Project to Q, K, V
    q <- self$query(x)
    k <- self$key(x)
    v <- self$value(x)
    
    # Reshape for attention
    q <- q$view(c(batch_size, seq_len, self$num_heads, self$head_dim))$transpose(2, 3)
    k <- k$view(c(batch_size, seq_len, self$num_heads, self$head_dim))$transpose(2, 3)
    v <- v$view(c(batch_size, seq_len, self$num_heads, self$head_dim))$transpose(2, 3)
    
    # Compute attention
    attention_weights <- torch_matmul(q, k$transpose(3, 4)) * self$scaling
    
    if (!is.null(mask)) {
      mask <- mask$unsqueeze(1)$unsqueeze(2)
      attention_weights <- attention_weights$masked_fill(mask == 0, -Inf)
    }
    
    attention_weights <- nnf_softmax(attention_weights, dim = -1)
    attention_weights <- self$dropout(attention_weights)
    
    # Apply attention
    output <- torch_matmul(attention_weights, v)
    output <- output$transpose(2, 3)$contiguous()
    output <- output$view(c(batch_size, seq_len, self$dim))
    
    # Project and add residual
    output <- self$out_proj(output)
    output <- output + residual
    
    # Reshape back to 2D if input was 2D
    if (is_2d) {
      output <- output$view(c(batch_size, -1))
    }
    
    list(output = output, attention_weights = attention_weights)
  }
)


#'Enhanced modality encoder with self-attention and MLP
# Updated EnhancedModalityEncoder
EnhancedModalityEncoder <- nn_module(
  "EnhancedModalityEncoder",
  initialize = function(input_dim, hidden_dims, num_heads = 4, dropout = 0.1) {
    self$input_mlp <- MLPBlock(input_dim, dropout = dropout)
    self$layers <- nn_module_list()
    self$attention_layers <- nn_module_list()

    current_dim <- input_dim
    for (dim in hidden_dims) {
      if (current_dim != dim) {
        self$layers$append(nn_sequential(
          nn_linear(current_dim, dim),
          nn_layer_norm(dim, dim),
          nn_gelu(),
          nn_dropout(dropout)
        ))
      }
      
      self$attention_layers$append(SelfAttention(
        dim = dim,
        num_heads = num_heads,
        dropout = dropout
      ))
      
      current_dim <- dim
    }
  },

  forward = function(x, mask = NULL, debug = FALSE) {
    if (debug) {
      cat("\n=== EnhancedModalityEncoder Forward Pass ===\n")
      print_tensor_info(x, "input")
    }

    x <- self$input_mlp(x)
    attention_weights <- list()

    for (i in 1:length(self$layers)) {
      if (i <= length(self$layers)) {
        x <- self$layers[[i]](x)
        if (debug) print_tensor_info(x, sprintf("after layer %d", i))
      }

      attention_result <- self$attention_layers[[i]](x, mask, debug)
      x <- attention_result$output
      attention_weights[[i]] <- attention_result$attention_weights
      
      if (debug) print_tensor_info(x, sprintf("after attention layer %d", i))
    }

    list(
      output = x,
      attention_weights = attention_weights
    )
  }
)

# Update the ModalityFusion module to include MLP blocks
ModalityFusion <- nn_module(
  "ModalityFusion",
  initialize = function(modality_dims, fusion_dim, num_heads = 4, dropout = 0.1) {
    projections_dict <- list()
    for (name in names(modality_dims)) {
      projections_dict[[name]] <- nn_sequential(
        nn_linear(modality_dims[[name]], fusion_dim),
        MLPBlock(fusion_dim, dropout = dropout)
      )
    }

    self$modality_projections <- nn_module_dict(projections_dict)

    self$cross_attention <- MultiHeadAttention(
      query_dim = fusion_dim,
      key_dim = fusion_dim,
      num_heads = num_heads,
      dropout = dropout
    )

    self$fusion_mlp <- MLPBlock(fusion_dim, dropout = dropout)
    self$fusion_norm <- nn_layer_norm(fusion_dim, fusion_dim)
    self$dropout <- nn_dropout(dropout)
  },

  forward = function(modality_features, modality_masks = NULL) {
    # Project each modality to common space with MLP transformation
    projected_features <- list()
    for (name in names(modality_features)) {
      projected <- self$modality_projections[[name]](modality_features[[name]])
      projected_features[[name]] <- projected
    }

    # Concatenate all projected features
    feature_tensor <- torch_stack(projected_features, dim = 2)

    # Create combined mask if masks are provided
    if (!is.null(modality_masks)) {
      combined_mask <- torch_stack(modality_masks, dim = 2)
    } else {
      combined_mask <- NULL
    }

    # Apply cross-attention across modalities
    attention_result <- self$cross_attention(
      query = feature_tensor,
      key = feature_tensor,
      value = feature_tensor,
      key_mask = combined_mask
    )

    # Apply final MLP transformation
    fused_features <- self$fusion_mlp(attention_result$output)
    fused_features <- self$fusion_norm(fused_features)
    fused_features <- self$dropout(fused_features)

    list(
      features = fused_features,
      attention_weights = attention_result$attention_weights
    )
  }
)

#' Multi-head attention module for multi-modal data
#' @param query_dim Dimension of query tensor
#' @param key_dim Dimension of key tensor
#' @param num_heads Number of attention heads
#' @param dropout Dropout rate

MultiHeadAttention <- nn_module(
  "MultiHeadAttention",
  initialize = function(query_dim, key_dim, num_heads = 4, dropout = 0.1) {
    self$num_heads <- num_heads
    self$head_dim <- query_dim %/% num_heads
    self$scaling <- self$head_dim ^ (-0.5)
    
    self$q_proj <- nn_linear(query_dim, query_dim)
    self$k_proj <- nn_linear(key_dim, query_dim)
    self$v_proj <- nn_linear(key_dim, query_dim)
    self$out_proj <- nn_linear(query_dim, query_dim)
    
    self$dropout <- nn_dropout(dropout)
  },
  
  forward = function(query, key, value, key_mask = NULL) {
    batch_size <- query$size(1)
    
    # Print dimensions of input tensors
    cat("\nMultiHeadAttention input dimensions:")
    print_tensor_info(query, "query")
    print_tensor_info(key, "key")
    print_tensor_info(value, "value")
    
    # Project and reshape for attention
    q <- self$q_proj(query)
    k <- self$k_proj(key)
    v <- self$v_proj(value)
    
    cat("\nAfter projection dimensions:")
    print_tensor_info(q, "projected query")
    print_tensor_info(k, "projected key")
    print_tensor_info(v, "projected value")
    
    # Reshape for attention computation
    q <- q$view(c(batch_size, -1, self$num_heads, self$head_dim))$transpose(2, 3)
    k <- k$view(c(batch_size, -1, self$num_heads, self$head_dim))$transpose(2, 3)
    v <- v$view(c(batch_size, -1, self$num_heads, self$head_dim))$transpose(2, 3)
    
    cat("\nAfter reshape dimensions:")
    print_tensor_info(q, "reshaped query")
    print_tensor_info(k, "reshaped key")
    print_tensor_info(v, "reshaped value")
    
    # Compute attention
    attention_weights <- torch_matmul(q, k$transpose(3, 4)) * self$scaling
    
    cat("\nAttention weights dimensions:")
    print_tensor_info(attention_weights, "attention_weights")
    
    if (!is.null(key_mask)) {
      attention_weights <- attention_weights$masked_fill(key_mask == 0, -Inf)
    }
    
    attention_weights <- nnf_softmax(attention_weights, dim = -1)
    attention_weights <- self$dropout(attention_weights)
    
    output <- torch_matmul(attention_weights, v)
    cat("\nOutput dimensions:")
    print_tensor_info(output, "attention output")
    
    output <- output$transpose(2, 3)$contiguous()
    output <- output$view(c(batch_size, -1))
    output <- self$out_proj(output)
    
    cat("\nFinal output dimensions:")
    print_tensor_info(output, "final output")
    
    list(output = output, attention_weights = attention_weights)
  }
)


#' Modality-specific encoder
#' @param input_dim Input dimension
#' @param hidden_dims Vector of hidden layer dimensions
#' @param dropout Dropout rate
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
    if (!is.null(mask)) {
      x <- x * mask
    }
    
    for (layer in self$layers) {
      x <- layer(x)
    }
    x
  }
)

#' Complete multi-modal survival prediction model with self-attention
#' @param modality_dims Named list of input dimensions for each modality
#' @param encoder_dims Named list of hidden dimensions for each encoder
#' @param fusion_dim Dimension of fused representation
#' @param num_heads Number of attention heads
#' @param dropout Dropout rate

MultiModalSurvivalModel <- nn_module(
  "MultiModalSurvivalModel",
  initialize = function(modality_dims, encoder_dims, fusion_dim, num_heads = 4, dropout = 0.1) {
    self$modality_dims <- modality_dims
    self$fusion_dim <- fusion_dim
    
    # Create encoders for each modality
    encoders_dict <- list()
    for (name in names(modality_dims)) {
      encoders_dict[[name]] <- EncoderBlock(
        input_dim = modality_dims[[name]],
        output_dim = fusion_dim,
        dropout = dropout
      )
    }
    self$encoders <- nn_module_dict(encoders_dict)
    
    # Modified fusion layer
    self$fusion_layer <- nn_sequential(
      nn_linear(fusion_dim * length(modality_dims), fusion_dim * 2),
      nn_layer_norm(fusion_dim * 2, fusion_dim * 2),
      nn_relu(),
      nn_dropout(dropout),
      nn_linear(fusion_dim * 2, fusion_dim),
      nn_layer_norm(fusion_dim, fusion_dim),
      nn_relu(),
      nn_dropout(dropout)
    )
    
    # Prediction head
    self$prediction_head <- nn_sequential(
      nn_linear(fusion_dim, fusion_dim %/% 2),
      nn_layer_norm(fusion_dim %/% 2, fusion_dim %/% 2),
      nn_relu(),
      nn_dropout(dropout),
      nn_linear(fusion_dim %/% 2, 1)
    )
  },
  
  forward = function(x, masks = NULL) {
    # Process each modality
    encoded_features <- list()
    valid_counts <- torch_zeros(x[[1]]$size(1))
    
    for (name in names(self$modality_dims)) {
      if (!is.null(x[[name]])) {
        # Get or create mask
        current_mask <- if (!is.null(masks[[name]])) {
          masks[[name]]
        } else {
          torch_ones_like(x[[name]])
        }
        
        # Count valid features per sample
        valid_counts <- valid_counts + torch_sum(current_mask, dim = 2)$squeeze()
        
        # Encode features
        encoded <- self$encoders[[name]](x[[name]], current_mask)
        encoded_features[[length(encoded_features) + 1]] <- encoded
      }
    }
    
    # Concatenate features
    concatenated <- torch_cat(encoded_features, dim = 2)
    
    # Create attention mask for valid features
    valid_features_mask <- (valid_counts > 0)$unsqueeze(2)
    
    # Apply fusion with masked features
    fused <- self$fusion_layer(concatenated)
    fused <- fused * valid_features_mask
    
    # Generate predictions
    predictions <- self$prediction_head(fused)
    
    # Final NaN cleanup in predictions
    predictions <- torch_where(
      torch_isnan(predictions),
      torch_zeros_like(predictions),
      predictions
    )
    
    list(
      predictions = predictions,
      attention_weights = NULL
    )
  }

  ,

   create_copy = function() {
    # Create new model with same architecture
    new_model <- torch::nn_module(
        "MultiModalSurvivalModel",
        initialize = function() {
            self$modality_dims <- self$modality_dims
            self$encoder_dims <- self$encoder_dims
            self$fusion_dim <- self$fusion_dim
            self$num_heads <- self$num_heads
            self$dropout <- self$dropout

            # Initialize encoders
            encoders_dict <- list()
            for (name in names(self$modality_dims)) {
                encoders_dict[[name]] <- EnhancedModalityEncoder(
                    input_dim = self$modality_dims[[name]],
                    hidden_dims = self$encoder_dims[[name]],
                    num_heads = self$num_heads,
                    dropout = self$dropout
                )
            }
            self$encoders <- nn_module_dict(encoders_dict)

            # Initialize fusion
            final_encoder_dims <- sapply(self$encoder_dims, function(x) x[length(x)])
            self$fusion <- ModalityFusion(
                modality_dims = final_encoder_dims,
                fusion_dim = self$fusion_dim,
                num_heads = self$num_heads,
                dropout = self$dropout
            )

            # Initialize prediction head
            self$prediction_head <- nn_sequential(
                nn_linear(self$fusion_dim, self$fusion_dim %/% 2),
                nn_relu(),
                nn_dropout(self$dropout),
                nn_linear(self$fusion_dim %/% 2, 1)
            )
        }
    )()

    # Copy the state dict to the new model
    new_model$load_state_dict(self$state_dict())

    return(new_model)
 }

)

# Print for debug

print_tensor_info <- function(tensor, name) {
    cat(sprintf("%s shape: %s\n", name, paste(tensor$size(), collapse=" x ")))
}

debug_masked_tensor <- function(tensor, mask = NULL, name, full_stats = FALSE) {
  if (is.null(tensor)) {
    cat(sprintf("\n%s is NULL", name))
    return(FALSE)
  }

  tensor_cpu <- tensor$cpu()$numpy()

  # Mask statistics
  if (!is.null(mask)) {
    mask_cpu <- mask$cpu()$numpy()
    valid_elements <- tensor_cpu[mask_cpu == 1]
    masked_elements <- tensor_cpu[mask_cpu == 0]

    stats <- list(
      total_elements = length(tensor_cpu),
      masked_elements = sum(mask_cpu == 0),
      valid_elements = sum(mask_cpu == 1),
      valid_has_nan = any(is.nan(valid_elements)),
      masked_has_nan = any(is.nan(masked_elements)),
      valid_range = if(length(valid_elements) > 0) c(min(valid_elements, na.rm=TRUE),
                                                    max(valid_elements, na.rm=TRUE)) else c(NA, NA),
      valid_mean = if(length(valid_elements) > 0) mean(valid_elements, na.rm=TRUE) else NA
    )

    cat(sprintf("\n=== %s (Masked Tensor) ===", name))
    cat(sprintf("\nShape: %s", paste(tensor$size(), collapse=" x ")))
    cat(sprintf("\nValid elements: %d (%.1f%%)",
                stats$valid_elements,
                100 * stats$valid_elements/stats$total_elements))
    cat(sprintf("\nMasked elements: %d (%.1f%%)",
                stats$masked_elements,
                100 * stats$masked_elements/stats$total_elements))
    cat(sprintf("\nNaNs in valid elements: %s", stats$valid_has_nan))
    cat(sprintf("\nNaNs in masked elements: %s", stats$masked_has_nan))
    cat(sprintf("\nValid elements range: [%f, %f]", stats$valid_range[1], stats$valid_range[2]))
    cat(sprintf("\nValid elements mean: %f", stats$valid_mean))

  } else {
    # Original statistics for unmasked tensor
    stats <- list(
      has_nan = any(is.nan(tensor_cpu)),
      has_inf = any(is.infinite(tensor_cpu)),
      min = min(tensor_cpu, na.rm = TRUE),
      max = max(tensor_cpu, na.rm = TRUE),
      mean = mean(tensor_cpu, na.rm = TRUE),
      nan_count = sum(is.nan(tensor_cpu))
    )

    cat(sprintf("\n=== %s (Unmasked Tensor) ===", name))
    cat(sprintf("\nShape: %s", paste(tensor$size(), collapse=" x ")))
    cat(sprintf("\nHas NaN: %s (%d NaNs)", stats$has_nan, stats$nan_count))
    cat(sprintf("\nHas Inf: %s", stats$has_inf))
    cat(sprintf("\nRange: [%f, %f]", stats$min, stats$max))
    cat(sprintf("\nMean: %f", stats$mean))
  }

  if (full_stats && (stats$has_nan || (!is.null(mask) && stats$valid_has_nan))) {
    # Find positions of NaNs in valid (unmasked) regions
    if (!is.null(mask)) {
      nan_indices <- which(is.nan(tensor_cpu) & mask_cpu == 1, arr.ind = TRUE)
    } else {
      nan_indices <- which(is.nan(tensor_cpu), arr.ind = TRUE)
    }
    if (length(nan_indices) > 0) {
      cat(sprintf("\nFirst few NaN positions in valid regions: %s",
                  paste(head(apply(nan_indices, 1, paste, collapse=","), 5),
                        collapse=" | ")))
    }
  }

  invisible(stats)
}

masked_bce_loss <- function(predictions, targets, masks) {
  # Replace NaNs with zeros in predictions
  predictions <- torch_where(
    torch_isnan(predictions),
    torch_zeros_like(predictions),
    predictions
  )

  # Calculate BCE loss
  criterion <- nn_bce_with_logits_loss(reduction = 'none')
  loss <- criterion(predictions, targets)

  # Apply masks if provided
  if (!is.null(masks)) {
    loss <- loss * masks
  }

  # Calculate mean loss excluding NaNs
  valid_loss <- loss[!torch_isnan(loss)]
  if (valid_loss$numel() > 0) {
    return(torch_mean(valid_loss))
  } else {
    return(torch_tensor(0.0))
  }
}
