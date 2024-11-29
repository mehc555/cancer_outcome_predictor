library(torch)

# Helper Modules
MLPBlock <- nn_module(
  "MLPBlock",
  initialize = function(dim, expansion_factor = 4, dropout = 0.1) {
    hidden_dim <- dim * expansion_factor
    
    self$norm <- nn_layer_norm(dim)
    self$net <- nn_sequential(
      nn_linear(dim, hidden_dim),
      nn_gelu(),
      nn_dropout(dropout),
      nn_linear(hidden_dim, dim),
      nn_dropout(dropout)
    )
  },
  
  forward = function(x) {
    residual <- x
    x <- self$norm(x)
    x <- self$net(x)
    x <- x + residual
    return(x)
  }
)

# Attention Module
MultiHeadSelfAttention <- nn_module(
  "MultiHeadSelfAttention",
  initialize = function(dim, num_heads = 2, dropout = 0.1, pre_norm = TRUE) {
    self$num_heads <- num_heads
    self$head_dim <- dim %/% num_heads
    self$scale <- sqrt(self$head_dim)
    self$dim <- dim
    self$pre_norm <- pre_norm
    
    self$norm <- nn_layer_norm(dim)
    self$to_q <- nn_linear(dim, dim)
    self$to_k <- nn_linear(dim, dim)
    self$to_v <- nn_linear(dim, dim)
    
    self$to_out <- nn_sequential(
      nn_linear(dim, dim),
      nn_dropout(dropout)
    )
    
    self$dropout <- nn_dropout(dropout)
  },
  
  forward = function(x, mask = NULL) {
    batch_size <- x$size(1)
    
    # Store residual
    residual <- x
    
    # Apply normalization based on pre_norm flag
    if (self$pre_norm) {
      x <- self$norm(x)
    }
    
    # Linear projections
    q <- self$to_q(x)
    k <- self$to_k(x)
    v <- self$to_v(x)
    
    # Reshape tensors to include sequence length dimension
    q <- q$unsqueeze(2)
    k <- k$unsqueeze(2)
    v <- v$unsqueeze(2)
    
    # Split heads
    q <- q$view(c(batch_size, 1, self$num_heads, self$head_dim))$transpose(2, 3)
    k <- k$view(c(batch_size, 1, self$num_heads, self$head_dim))$transpose(2, 3)
    v <- v$view(c(batch_size, 1, self$num_heads, self$head_dim))$transpose(2, 3)
    
    # Calculate attention scores
    attn <- torch_matmul(q, k$transpose(3, 4)) / self$scale
    
    if (!is.null(mask)) {
      mask <- mask$unsqueeze(2)$unsqueeze(3)
      attn <- attn$masked_fill(mask == 0, -1e9)
    }
    
    attn <- nnf_softmax(attn, dim = -1)
    attn <- self$dropout(attn)
    
    # Apply attention to values
    out <- torch_matmul(attn, v)
    
    # Reshape back
    out <- out$transpose(2, 3)$contiguous()
    out <- out$view(c(batch_size, self$dim))
    
    # Final projection
    out <- self$to_out(out)
    
    # Apply normalization based on pre_norm flag
    if (!self$pre_norm) {
      out <- self$norm(out)
    }
    
    # Add residual
    out <- out + residual
    
    return(out)
  }
)

# Modality Encoder
ModalityEncoder <- nn_module(
  "ModalityEncoder",
  initialize = function(input_dim, hidden_dims, dropout = 0.1, attention_config = NULL) {
    self$layers <- nn_module_list()
    
    current_dim <- input_dim
    for (dim in hidden_dims) {
      layer <- nn_sequential(
        nn_linear(current_dim, dim),
        nn_batch_norm1d(dim),
        nn_relu(),
        nn_dropout(dropout)
      )
      self$layers$append(layer)
      current_dim <- dim
    }
    
    final_dim <- hidden_dims[length(hidden_dims)]
    
    # Only add attention if enabled in config
    if (!is.null(attention_config) && attention_config$enabled) {
      self$attention <- MultiHeadSelfAttention(
        dim = final_dim,
        num_heads = attention_config$num_heads,
        dropout = attention_config$dropout,
        pre_norm = attention_config$pre_norm
      )
    } else {
      self$attention <- NULL
    }
  },
  
  forward = function(x, mask = NULL) {
    x <- torch_where(torch_isnan(x), torch_zeros_like(x), x)
    
    for (i in 1:length(self$layers)) {
      x <- self$layers[[i]](x)
    }
    
    if (!is.null(self$attention)) {
      x <- self$attention(x, mask)
    }
    
    return(x)
  }
)

# Updated ModalityFusion with Global Attention
ModalityFusion <- nn_module(
  "ModalityFusion",
  initialize = function(modality_dims, fusion_dim, dropout = 0.1, attention_config = NULL) {
    self$fusion_dim <- fusion_dim
    self$modality_names <- names(modality_dims)
    
    # Create MLPs for each modality
    projections_dict <- list()
    for (name in self$modality_names) {
      projections_dict[[name]] <- MLPBlock(fusion_dim, dropout = dropout)
    }
    self$modality_projections <- nn_module_dict(projections_dict)
    
    # Add cross-attention if enabled
    if (!is.null(attention_config) && attention_config$cross_modality$enabled) {
      cross_attention_dict <- list()
      for (query_modality in self$modality_names) {
        for (key_modality in self$modality_names) {
          if (query_modality != key_modality) {
            module_name <- paste0(query_modality, "_to_", key_modality)
            cross_attention_dict[[module_name]] <- CrossModalityAttention(
              dim = fusion_dim,
              num_heads = attention_config$cross_modality$num_heads,
              dropout = attention_config$cross_modality$dropout
            )
          }
        }
      }
      self$cross_attention <- nn_module_dict(cross_attention_dict)
    } else {
      self$cross_attention <- NULL
    }
    
    # Add global attention if enabled
    if (!is.null(attention_config) && attention_config$global$enabled) {
      self$global_attention <- GlobalSelfAttention(
        dim = fusion_dim,
        num_heads = attention_config$global$num_heads,
        dropout = attention_config$global$dropout
      )
    } else {
      self$global_attention <- NULL
    }
    
    # Final fusion components
    self$fusion_norm <- nn_layer_norm(fusion_dim)
    self$fusion_mlp <- MLPBlock(fusion_dim, dropout = dropout)
  },
  
  forward = function(modality_features, masks = NULL) {
    # Project each modality to fusion space
    projected_features <- list()
    for (name in names(modality_features)) {
      if (!is.null(modality_features[[name]])) {
        projected <- self$modality_projections[[name]](modality_features[[name]])
        projected_features[[name]] <- projected
      }
    }
    
    # Apply cross-attention if enabled
    if (!is.null(self$cross_attention)) {
      attended_features <- projected_features
      
      for (query_modality in names(attended_features)) {
        cross_attended <- list()
        
        for (key_modality in names(attended_features)) {
          if (query_modality != key_modality) {
            module_name <- paste0(query_modality, "_to_", key_modality)
            
            attended <- self$cross_attention[[module_name]](
              query = attended_features[[query_modality]],
              key = attended_features[[key_modality]],
              value = attended_features[[key_modality]]
            )
            
            cross_attended[[key_modality]] <- attended
          }
        }
        
        if (length(cross_attended) > 0) {
          feature_tensors <- lapply(cross_attended, function(x) x)
          feature_stack <- torch_stack(feature_tensors, dim = 2)
          attended_features[[query_modality]] <- feature_stack$mean(dim = 2)
        }
      }
      
      projected_features <- attended_features
    }
    
    # Stack and combine features
    feature_tensors <- lapply(names(projected_features), function(name) {
      projected_features[[name]]
    })
    
    feature_stack <- torch_stack(feature_tensors, dim = 2)
    fused_features <- feature_stack$mean(dim = 2)
    
    # Apply fusion processing
    fused_features <- self$fusion_norm(fused_features)
    fused_features <- self$fusion_mlp(fused_features)
    
    # Apply global attention if enabled
    if (!is.null(self$global_attention)) {
      fused_features <- self$global_attention(fused_features)
    }
    
    return(list(features = fused_features))
  }
)


MultiModalSurvivalModel <- nn_module(
  "MultiModalSurvivalModel",
  initialize = function(modality_dims, encoder_dims, fusion_dim,
                       dropout = 0.1, attention_config = NULL,
                       outcome_type = "binary") {
    self$outcome_type <- outcome_type
    self$modality_dims <- modality_dims
    self$fusion_dim <- fusion_dim
    self$dropout <- dropout
    self$attention_config <- attention_config

    # Create encoders for each modality
    encoders_dict <- list()
    modality_fusion_dims <- list()

    for (name in names(modality_dims)) {
      hidden_dims <- c(modality_dims[[name]], fusion_dim)

      # Pass intra-modality attention config to encoders
      encoders_dict[[name]] <- ModalityEncoder(
        input_dim = modality_dims[[name]],
        hidden_dims = hidden_dims,
        dropout = dropout,
        attention_config = attention_config$intra_modality
      )
      modality_fusion_dims[[name]] <- fusion_dim
    }

    self$encoders <- nn_module_dict(encoders_dict)

    # Fusion module with both attention configurations
    self$fusion <- ModalityFusion(
      modality_dims = modality_fusion_dims,
      fusion_dim = fusion_dim,
      dropout = dropout,
      attention_config = attention_config  # Pass both intra and cross attention configs
    )

    # Prediction head
    self$prediction_head <- nn_sequential(
      nn_layer_norm(fusion_dim),
      nn_linear(fusion_dim, fusion_dim %/% 2),
      nn_relu(),
      nn_dropout(dropout),
      nn_linear(fusion_dim %/% 2, 1)
    )
  },

  forward = function(x, masks = NULL) {
    encoded_features <- list()
    modality_names <- names(self$modality_dims)

    for (name in modality_names) {
      if (!is.null(x[[name]])) {
        current_mask <- if (!is.null(masks) && !is.null(masks[[name]])) {
          masks[[name]]
        } else {
          NULL
        }

        encoded <- self$encoders[[name]](x[[name]], current_mask)
        encoded_features[[name]] <- encoded
      }
    }

    if (length(encoded_features) == 0) {
      stop("No features to fuse - all modalities were empty")
    }

    fusion_result <- self$fusion(encoded_features, masks)
    fused_features <- fusion_result$features

    predictions <- self$prediction_head(fused_features)
    predictions <- torch_where(
      torch_isnan(predictions),
      torch_zeros_like(predictions),
      predictions
    )

    return(list(predictions = predictions))
  },

  create_copy = function() {
    new_model <- MultiModalSurvivalModel(
      modality_dims = self$modality_dims,
      encoder_dims = NULL,  # Not used in current implementation
      fusion_dim = self$fusion_dim,
      dropout = self$dropout,
      attention_config = self$attention_config,
      outcome_type = self$outcome_type
    )
    new_model$load_state_dict(self$state_dict())
    return(new_model)
  }
)



# Loss function
compute_bce_loss <- function(predictions, targets) {
  predictions <- torch_where(
    torch_isnan(predictions),
    torch_zeros_like(predictions),
    predictions
  )
  
  valid_mask <- !torch_isnan(targets)
  n_valid <- valid_mask$sum()$item()
  
  if (n_valid > 0) {
    valid_predictions <- predictions[valid_mask]
    valid_targets <- targets[valid_mask]
    
    criterion <- nn_bce_with_logits_loss(reduction = 'mean')
    loss <- criterion(valid_predictions, valid_targets)
    
    return(loss)
  } else {
    return(torch_tensor(0.0))
  }
}

# Cross-Attention Module
CrossModalityAttention <- nn_module(
  "CrossModalityAttention",
  initialize = function(dim, num_heads = 2, dropout = 0.1) {
    self$num_heads <- num_heads
    self$head_dim <- floor(dim / num_heads)
    self$scale <- sqrt(self$head_dim)
    self$dim <- dim

    # Projections for cross-attention
    proj_dim <- self$head_dim * num_heads
    self$to_q <- nn_linear(dim, proj_dim)
    self$to_k <- nn_linear(dim, proj_dim)
    self$to_v <- nn_linear(dim, proj_dim)

    self$to_out <- nn_sequential(
      nn_linear(proj_dim, dim),
      nn_dropout(dropout)
    )

    self$norm <- nn_layer_norm(dim)
    self$dropout <- nn_dropout(dropout)
  },

  forward = function(query, key, value, mask = NULL) {
    batch_size <- query$size(1)

    # Store residual
    residual <- query

    # Linear projections
    q <- self$to_q(query)
    k <- self$to_k(key)
    v <- self$to_v(value)

    # Reshape for attention
    q <- q$view(c(batch_size, 1, self$num_heads, self$head_dim))
    k <- k$view(c(batch_size, 1, self$num_heads, self$head_dim))
    v <- v$view(c(batch_size, 1, self$num_heads, self$head_dim))

    # Transpose for attention
    q <- q$transpose(2, 3)
    k <- k$transpose(2, 3)
    v <- v$transpose(2, 3)

    # Calculate attention scores
    attn <- torch_matmul(q, k$transpose(3, 4)) / self$scale

    if (!is.null(mask)) {
      mask <- mask$unsqueeze(2)$unsqueeze(3)
      attn <- attn$masked_fill(mask == 0, -1e9)
    }

    attn <- nnf_softmax(attn, dim = -1)
    attn <- self$dropout(attn)

    # Apply attention to values
    out <- torch_matmul(attn, v)

    # Reshape back
    out <- out$transpose(2, 3)$contiguous()
    out <- out$view(c(batch_size, self$dim))

    # Final projection
    out <- self$to_out(out)
    out <- self$norm(out)

    # Add residual
    out <- out + residual

    return(out)
  }
)

# Global Attention Module
GlobalSelfAttention <- nn_module(
  "GlobalSelfAttention",
  initialize = function(dim, num_heads = 2, dropout = 0.1) {
    self$num_heads <- num_heads
    self$head_dim <- floor(dim / num_heads)
    self$scale <- sqrt(self$head_dim)
    self$dim <- dim

    # Projections for global attention
    proj_dim <- self$head_dim * num_heads
    self$to_q <- nn_linear(dim, proj_dim)
    self$to_k <- nn_linear(dim, proj_dim)
    self$to_v <- nn_linear(dim, proj_dim)

    self$to_out <- nn_sequential(
      nn_linear(proj_dim, dim),
      nn_dropout(dropout)
    )

    self$norm <- nn_layer_norm(dim)
    self$dropout <- nn_dropout(dropout)
  },

  forward = function(x, mask = NULL) {
    batch_size <- x$size(1)

    # Store residual
    residual <- x

    # Apply layer normalization first (pre-norm)
    x <- self$norm(x)

    # Linear projections
    q <- self$to_q(x)
    k <- self$to_k(x)
    v <- self$to_v(x)

    # Reshape for attention with sequence length of 1
    q <- q$view(c(batch_size, 1, self$num_heads, self$head_dim))
    k <- k$view(c(batch_size, 1, self$num_heads, self$head_dim))
    v <- v$view(c(batch_size, 1, self$num_heads, self$head_dim))

    # Transpose for attention
    q <- q$transpose(2, 3)
    k <- k$transpose(2, 3)
    v <- v$transpose(2, 3)

    # Calculate attention scores
    attn <- torch_matmul(q, k$transpose(3, 4)) / self$scale

    if (!is.null(mask)) {
      mask <- mask$unsqueeze(2)$unsqueeze(3)
      attn <- attn$masked_fill(mask == 0, -1e9)
    }

    attn <- nnf_softmax(attn, dim = -1)
    attn <- self$dropout(attn)

    # Apply attention to values
    out <- torch_matmul(attn, v)

    # Reshape back
    out <- out$transpose(2, 3)$contiguous()
    out <- out$view(c(batch_size, self$dim))

    # Final projection
    out <- self$to_out(out)

    # Add residual
    out <- out + residual

    return(out)
  }
)

