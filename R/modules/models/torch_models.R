# R/modules/models/torch_models.R

library(torch)

#' MLP block for feature transformation
#' @param dim Input dimension
#' @param expansion_factor Expansion factor for hidden dimension
#' @param dropout Dropout rate
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
    x + self$net(self$norm(x))  # Residual connection
  }
)

#' Enhanced self-attention module with MLP
#' @param dim Input dimension
#' @param num_heads Number of attention heads
#' @param dropout Dropout rate
SelfAttention <- nn_module(
  "SelfAttention",
  initialize = function(dim, num_heads = 4, dropout = 0.1) {
    self$num_heads <- num_heads
    self$head_dim <- dim %/% num_heads
    self$scaling <- self$head_dim ^ (-0.5)

    self$qkv_proj <- nn_linear(dim, dim * 3)
    self$out_proj <- nn_linear(dim, dim)

    self$dropout <- nn_dropout(dropout)
    self$norm1 <- nn_layer_norm(dim, dim)

    # Add MLP block
    self$mlp <- MLPBlock(dim, dropout = dropout)
    self$norm2 <- nn_layer_norm(dim, dim)
  },

  forward = function(x, mask = NULL) {
    # Store input for residual
    input_x <- x

    # First normalization
    x <- self$norm1(x)

    batch_size <- x$size(1)
    seq_len <- x$size(2)

    # Combined projection and split into Q, K, V
    qkv <- self$qkv_proj(x)
    qkv <- qkv$view(c(batch_size, seq_len, 3, self$num_heads, self$head_dim))
    qkv <- qkv$permute(c(3, 1, 4, 2, 5))

    q <- qkv[1]
    k <- qkv[2]
    v <- qkv[3]

    # Scaled dot-product attention
    attention_weights <- torch_matmul(q, k$transpose(4, 5)) * self$scaling

    if (!is.null(mask)) {
      mask <- mask$unsqueeze(2)
      attention_weights <- attention_weights$masked_fill(mask == 0, -Inf)
    }

    attention_weights <- nnf_softmax(attention_weights, dim = 5)
    attention_weights <- self$dropout(attention_weights)

    # Apply attention to values
    output <- torch_matmul(attention_weights, v)
    output <- output$transpose(2, 3)$contiguous()
    output <- output$view(c(batch_size, seq_len, -1))

    output <- self$out_proj(output)
    output <- self$dropout(output)

    # First residual connection
    output <- output + input_x

    # MLP block with residual
    output <- self$mlp(output)

    list(output = output, attention_weights = attention_weights)
  }
)

#'Enhanced modality encoder with self-attention and MLP

EnhancedModalityEncoder <- nn_module(
  "EnhancedModalityEncoder",
  initialize = function(input_dim, hidden_dims, num_heads = 4, dropout = 0.1) {
    self$input_mlp <- MLPBlock(input_dim, dropout = dropout)

    self$layers <- nn_module_list()
    self$attention_layers <- nn_module_list()

    current_dim <- input_dim
    for (dim in hidden_dims) {
      # Add projection if dimensions differ
      if (current_dim != dim) {
        self$layers$append(nn_sequential(
          nn_linear(current_dim, dim),
          nn_layer_norm(dim, dim),
          nn_gelu(),
          nn_dropout(dropout)
        ))
      }

      # Add self-attention layer
      self$attention_layers$append(SelfAttention(
        dim = dim,
        num_heads = num_heads,
        dropout = dropout
      ))

      current_dim <- dim
    }
  },

  forward = function(x, mask = NULL) {
    # Initial MLP transformation
    x <- self$input_mlp(x)

    attention_weights <- list()

    # Process through attention and MLP layers
    for (i in 1:length(self$layers)) {
      if (i <= length(self$layers)) {
        x <- self$layers[[i]](x)
      }

      attention_result <- self$attention_layers[[i]](x, mask)
      x <- attention_result$output
      attention_weights[[i]] <- attention_result$attention_weights
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
    
    # Linear projections and reshape for multi-head attention
    q <- self$q_proj(query)$view(c(batch_size, -1, self$num_heads, self$head_dim))$transpose(2, 3)
    k <- self$k_proj(key)$view(c(batch_size, -1, self$num_heads, self$head_dim))$transpose(2, 3)
    v <- self$v_proj(value)$view(c(batch_size, -1, self$num_heads, self$head_dim))$transpose(2, 3)
    
    # Scaled dot-product attention
    attention_weights <- torch_matmul(q, k$transpose(3, 4)) * self$scaling
    
    # Apply mask if provided
    if (!is.null(key_mask)) {
      mask <- key_mask$unsqueeze(2)$unsqueeze(3)
      attention_weights <- attention_weights$masked_fill(mask == 0, -Inf)
    }
    
    attention_weights <- nnf_softmax(attention_weights, dim = 4)
    attention_weights <- self$dropout(attention_weights)
    
    # Apply attention weights to values
    output <- torch_matmul(attention_weights, v)
    
    # Reshape and project output
    output <- output$transpose(2, 3)$contiguous()$view(c(batch_size, -1, query_dim))
    output <- self$out_proj(output)
    
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
    # Store the initialization parameters for copying
    self$modality_dims <- modality_dims
    self$encoder_dims <- encoder_dims
    self$fusion_dim <- fusion_dim
    self$num_heads <- num_heads
    self$dropout <- dropout

    # Initialize empty dictionary for encoders
    encoders_dict <- list()

    # Create enhanced encoders for each modality
    for (name in names(modality_dims)) {
      encoders_dict[[name]] <- EnhancedModalityEncoder(
        input_dim = modality_dims[[name]],
        hidden_dims = encoder_dims[[name]],
        num_heads = num_heads,
        dropout = dropout
      )
    }

    # Create module dictionary with the populated list
    self$encoders <- nn_module_dict(encoders_dict)

    # Create fusion module
    final_encoder_dims <- sapply(encoder_dims, function(x) x[length(x)])
    self$fusion <- ModalityFusion(
      modality_dims = final_encoder_dims,
      fusion_dim = fusion_dim,
      num_heads = num_heads,
      dropout = dropout
    )

    # Prediction head for survival
    self$prediction_head <- nn_sequential(
      nn_linear(fusion_dim, fusion_dim %/% 2),
      nn_relu(),
      nn_dropout(dropout),
      nn_linear(fusion_dim %/% 2, 1)
    )
  },

  forward = function(x, masks = NULL) {
    # Store all attention weights for interpretation
    all_attention_weights <- list(
      self_attention = list(),
      cross_attention = NULL
    )

    # Encode each modality
    encoded_features <- list()
    for (name in names(x)) {
      if (!endsWith(name, "_mask")) {
        current_mask <- if (!is.null(masks)) masks[[name]] else NULL
        encoder_result <- self$encoders[[name]](x[[name]], current_mask)
        encoded_features[[name]] <- encoder_result$output
        all_attention_weights$self_attention[[name]] <- encoder_result$attention_weights
      }
    }

    # Fuse modalities
    fusion_result <- self$fusion(encoded_features, masks)

    # Store cross-attention weights
    all_attention_weights$cross_attention <- fusion_result$attention_weights

    # Make prediction
    predictions <- self$prediction_head(fusion_result$features)

    list(
      predictions = predictions,
      attention_weights = all_attention_weights
    )
  },

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
