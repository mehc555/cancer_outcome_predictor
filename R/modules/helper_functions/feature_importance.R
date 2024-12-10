library(ggplot2)
library(reshape2)
library(gridExtra)
library(pheatmap)
library(RColorBrewer)

# Feature validation function
validate_features <- function(cv_results) {
  for (modality in names(cv_results$features)) {
    features <- cv_results$features[[modality]]
    
    # Check for NAs
    na_indices <- which(is.na(features))
    if (length(na_indices) > 0) {
      stop(sprintf(
        "Found NA features in %s modality at indices: %s\nTotal features: %d", 
        modality,
        paste(na_indices, collapse=", "),
        length(features)
      ))
    }
    
    # Check for empty strings
    empty_indices <- which(features == "")
    if (length(empty_indices) > 0) {
      stop(sprintf(
        "Found empty feature names in %s modality at indices: %s",
        modality,
        paste(empty_indices, collapse=", ")
      ))
    }
    
    # Check feature dimensions against model
    if (!is.null(cv_results$model$encoders$modules[[modality]])) {
      weight_dim <- dim(cv_results$model$encoders$modules[[modality]]$layers[[1]][[1]]$weight)[2]
      if (length(features) != weight_dim) {
        stop(sprintf(
          "Feature dimension mismatch in %s modality:\nFeatures length: %d\nWeight dimension: %d",
          modality,
          length(features),
          weight_dim
        ))
      }
    }
    
    cat(sprintf("Validated %s features: %d\n", modality, length(features)))
  }
}

# Feature importance analysis function
analyze_feature_importance <- function(model, selected_features, top_n = 20) {
  importance_by_modality <- list()
  
  for (modality in names(model$modality_dims)) {
    # Validate feature existence
    if (is.null(selected_features[[modality]])) {
      warning(sprintf("No features found for modality: %s", modality))
      next
    }
    
    # Get first layer weights
    weights <- tryCatch({
      model$encoders$modules[[modality]]$layers[[1]][[1]]$weight$cpu()
    }, error = function(e) {
      warning(sprintf("Could not extract weights for modality %s: %s", modality, e$message))
      return(NULL)
    })
    
    if (!is.null(weights)) {
      weights_matrix <- as.matrix(weights)
      importance_scores <- colMeans(abs(weights_matrix))
      
      features <- selected_features[[modality]]
      
      # Validate dimensions
      if (length(features) != length(importance_scores)) {
        stop(sprintf(
          "Dimension mismatch in %s: features (%d) != importance scores (%d)",
          modality, length(features), length(importance_scores)
        ))
      }
      
      importance_df <- data.frame(
        feature = features,
        importance = importance_scores,
        modality = modality
      )
      
      importance_df <- importance_df[order(-importance_df$importance), ]
      importance_by_modality[[modality]] <- importance_df
      
      cat(sprintf("\nTop %d %s features:\n", top_n, modality))
      print(head(importance_df[, c("feature", "importance")], top_n))
    }
  }
  
  return(importance_by_modality)
}

# Modified create_attention_matrix function
create_attention_matrix <- function(module, feature_names, source_mod = NULL, target_mod = NULL) {
  if (!is.null(source_mod) && any(is.na(source_mod))) stop("NA features found in source modality")
  if (!is.null(target_mod) && any(is.na(target_mod))) stop("NA features found in target modality")
  if (is.null(source_mod) && is.null(target_mod) && any(is.na(feature_names))) {
    stop("NA features found in feature list")
  }
  
  tryCatch({
    if (!is.null(module$to_q)) {
      # Get the query and key projection weights
      q_weights <- as.matrix(module$to_q$weight$detach()$cpu())
      k_weights <- if (!is.null(module$to_k)) {
        as.matrix(module$to_k$weight$detach()$cpu())
      } else {
        q_weights  # If to_k doesn't exist, use to_q weights
      }
      
      # Get dimensions
      hidden_dim <- nrow(q_weights)
      
      # For cross-modality attention, handle source and target features separately
      if (!is.null(source_mod) && !is.null(target_mod)) {
        # Create attention matrix for source-to-target attention only
        attention_matrix <- matrix(0, nrow = length(source_mod), ncol = length(target_mod))
        rownames(attention_matrix) <- source_mod
        colnames(attention_matrix) <- target_mod
        
        # Calculate attention scores
        if (ncol(q_weights) == hidden_dim) {
          temp_attention <- q_weights %*% t(k_weights)
          
          for (i in 1:length(source_mod)) {
            for (j in 1:length(target_mod)) {
              attention_matrix[i,j] <- temp_attention[
                ((i-1) %% hidden_dim) + 1, 
                ((j-1) %% hidden_dim) + 1
              ]
            }
          }
        } else {
          # Direct mapping if dimensions match
          attention_matrix <- q_weights[1:length(source_mod), ] %*% 
                            t(k_weights[1:length(target_mod), ])
        }
        
        return(attention_matrix)
      }
      
      # Original intra-modality logic
      n_features <- length(feature_names)
      attention_matrix <- matrix(0, nrow = n_features, ncol = n_features)
      rownames(attention_matrix) <- feature_names
      colnames(attention_matrix) <- feature_names
      
      if (ncol(q_weights) == hidden_dim) {
        temp_attention <- q_weights %*% t(k_weights)
        for (i in 1:n_features) {
          for (j in 1:n_features) {
            attention_matrix[i,j] <- temp_attention[
              ((i-1) %% hidden_dim) + 1, 
              ((j-1) %% hidden_dim) + 1
            ]
          }
        }
      } else {
        attention_matrix <- q_weights %*% t(k_weights)
      }
      
      return(attention_matrix)
    }
  }, error = function(e) {
    stop("Error in create_attention_matrix: ", e$message)
  })
  
  return(matrix(0, nrow = length(feature_names), ncol = length(feature_names)))
}

# Modified analyze_attention_patterns function
analyze_attention_patterns <- function(cv_results) {
  model <- cv_results$model
  features <- cv_results$features
  
  # Validate features before processing
  validate_features(cv_results)
  
  # Debug print dimensions for attention modules
  for (modality in names(model$encoders$modules)) {
    if (!is.null(model$encoders$modules[[modality]]$attention)) {
      weights <- model$encoders$modules[[modality]]$attention$to_q$weight
      cat(sprintf(
        "Attention weights dimension for %s: %s\n",
        modality,
        paste(dim(weights), collapse=" x ")
      ))
    }
  }
  
  attention_patterns <- list()
  
  # Analyze intra-modality attention
  intra_modality <- list()
  for (modality in names(features)) {
    message("Processing intra-modality attention for: ", modality)
    if (!is.null(model$encoders$modules[[modality]]$attention)) {
      intra_modality[[modality]] <- tryCatch({
        create_attention_matrix(
          model$encoders$modules[[modality]]$attention,
          features[[modality]]
        )
      }, error = function(e) {
        message("Error processing intra-modality attention for ", modality, ": ", e$message)
        return(NULL)
      })
    }
  }
  attention_patterns$intra_modality <- intra_modality
  
  # Process cross-modality attention
  if (!is.null(model$fusion) && !is.null(model$fusion$cross_attention)) {
    message("Processing cross-modality attention")
    cross_modality <- list()
    for (modality1 in names(features)) {
      for (modality2 in names(features)) {
        if (modality1 != modality2) {
          pair_name <- paste0(modality1, "_to_", modality2)
          if (!is.null(model$fusion$cross_attention$modules[[pair_name]])) {
            cross_modality[[pair_name]] <- tryCatch({
              create_attention_matrix(
                model$fusion$cross_attention$modules[[pair_name]],
                NULL,  # We don't need combined features anymore
                features[[modality1]],  # Source features
                features[[modality2]]   # Target features
              )
            }, error = function(e) {
              message("Error processing cross-modality attention for ", pair_name, ": ", e$message)
              return(NULL)
            })
          }
        }
      }
    }
    attention_patterns$cross_modality <- cross_modality
  }
  
  # Analyze global attention
  if (!is.null(model$fusion) && !is.null(model$fusion$global_attention)) {
    message("Processing global attention")
    all_features <- unlist(features)
    attention_patterns$global <- tryCatch({
      create_attention_matrix(
        model$fusion$global_attention,
        all_features
      )
    }, error = function(e) {
      message("Error processing global attention: ", e$message)
      return(NULL)
    })
  }
  
  return(attention_patterns)
}

# Basic visualization function
visualize_attention_patterns <- function(attention_patterns) {
  library(ggplot2)
  library(reshape2)
  
  plots <- list()
  
  create_heatmap <- function(matrix, title) {
    if (is.null(matrix)) return(NULL)
    
    df <- reshape2::melt(matrix)
    colnames(df) <- c("Feature1", "Feature2", "Weight")
    
    ggplot(df, aes(Feature1, Feature2, fill = Weight)) +
      geom_tile() +
      scale_fill_gradient2(low = "blue", mid = "white", high = "red",
                          midpoint = mean(df$Weight)) +
      theme_minimal() +
      theme(
        axis.text.x = element_text(angle = 45, hjust = 1, size = 6),
        axis.text.y = element_text(size = 6),
        axis.title = element_text(size = 8),
        plot.title = element_text(size = 10)
      ) +
      labs(title = title)
  }
  
  # Create visualizations for each type of attention
  if (!is.null(attention_patterns$intra_modality)) {
    for (modality in names(attention_patterns$intra_modality)) {
      plot <- create_heatmap(
        attention_patterns$intra_modality[[modality]],
        paste("Intra-modality Attention:", modality)
      )
      if (!is.null(plot)) plots[[paste0("intra_", modality)]] <- plot
    }
  }
  
  if (!is.null(attention_patterns$cross_modality)) {
    for (pair in names(attention_patterns$cross_modality)) {
      plot <- create_heatmap(
        attention_patterns$cross_modality[[pair]],
        paste("Cross-modality Attention:", pair)
      )
      if (!is.null(plot)) plots[[paste0("cross_", pair)]] <- plot
    }
  }
  
  if (!is.null(attention_patterns$global)) {
    plot <- create_heatmap(
      attention_patterns$global,
      "Global Attention Patterns"
    )
    if (!is.null(plot)) plots$global <- plot
  }
  
  return(plots)
}

# Summary function
summarize_attention_patterns <- function(attention_results, cv_results) {
  summary <- list()
  
  get_top_features <- function(matrix, feature_names, n = 10) {
    if (is.null(matrix)) return(NULL)
    
    # Validate inputs
    if (any(is.na(feature_names))) {
      stop("NA values found in feature names")
    }
    
    importance <- rowSums(abs(matrix))
    top_idx <- order(importance, decreasing = TRUE)[1:min(n, length(importance))]
    
    list(
      features = feature_names[top_idx],
      scores = importance[top_idx]
    )
  }
  
  # Summarize each type of attention
  if (!is.null(attention_results$intra_modality)) {
    summary$intra_modality <- list()
    for (modality in names(attention_results$intra_modality)) {
      summary$intra_modality[[modality]] <- get_top_features(
        attention_results$intra_modality[[modality]],
        cv_results$features[[modality]]
      )
    }
  }
  
  if (!is.null(attention_results$cross_modality)) {
    summary$cross_modality <- list()
    for (pair in names(attention_results$cross_modality)) {
      # Split the pair name to get source and target modalities
      modalities <- strsplit(pair, "_to_")[[1]]
      source_modality <- modalities[1]
      
      summary$cross_modality[[pair]] <- get_top_features(
        attention_results$cross_modality[[pair]],
        cv_results$features[[source_modality]]  # Only use source features
      )
    }
  }
  
  if (!is.null(attention_results$global)) {
    all_features <- unlist(cv_results$features)
    summary$global <- get_top_features(
      attention_results$global,
      all_features
    )
  }
  
  return(summary)
}


# Function to create attention heatmaps with appropriate feature name handling
create_attention_visualizations <- function(attention_results, cv_results, output_dir = "attention_plots") {
  # Create output directory if it doesn't exist
  if (!dir.exists(output_dir)) {
    dir.create(output_dir)
  }

  # Helper function to make row names unique with modality appending
  make_unique_names <- function(names, feature_modalities = NULL) {
    if (is.null(feature_modalities)) return(names)
    
    modified_names <- mapply(
      function(name, modality) {
        if (modality %in% c("expression", "cnv")) {
          sprintf("%s_%s", name, modality)
        } else {
          name
        }
      },
      names, feature_modalities,
      SIMPLIFY = TRUE
    )
    
    # Handle any remaining duplicates
    while(any(duplicated(modified_names))) {
      dup_idx <- which(duplicated(modified_names))
      modified_names[dup_idx] <- sprintf("%s_%d", modified_names[dup_idx], 
                                       sequence(tabulate(match(modified_names, unique(modified_names))))[dup_idx])
    }
    
    return(modified_names)
  }

  # Helper function to determine feature modality
  get_feature_modality <- function(feature, cv_results) {
    for (mod in names(cv_results$features)) {
      if (feature %in% cv_results$features[[mod]]) return(mod)
    }
    return("unknown")
  }

  # Helper function to create and save heatmap
  create_heatmap <- function(matrix, title, filename, annotation_col = NULL, 
                            annotation_row = NULL, feature_modalities = NULL) {
    if (is.null(matrix)) return(NULL)

    # Make row and column names unique with modality info
    if (!is.null(feature_modalities)) {
      rownames(matrix) <- make_unique_names(rownames(matrix), feature_modalities)
      if (is.null(annotation_col)) { # For intra-modality
        colnames(matrix) <- make_unique_names(colnames(matrix), feature_modalities)
      }
    }

    # Update annotation rownames to match matrix
    if (!is.null(annotation_row)) {
      rownames(annotation_row) <- rownames(matrix)
    }
    if (!is.null(annotation_col)) {
      rownames(annotation_col) <- colnames(matrix)
    }

    # Scale the matrix for better visualization
    scaled_matrix <- matrix / max(abs(matrix))

    # Create color palette
    colors <- colorRampPalette(rev(brewer.pal(11, "RdBu")))(100)

    # Create annotation colors
    ann_colors <- list(
      Modality = c(
        clinical = "#E41A1C",
        expression = "#377EB8",
        cnv = "#4DAF4A",
        mutations = "#984EA3",
        methylation = "#FF7F00",
        mirna = "#FFFF33"
      )
    )

    # Create heatmap
    pheatmap(
      scaled_matrix,
      main = title,
      color = colors,
      annotation_col = annotation_col,
      annotation_row = annotation_row,
      annotation_colors = ann_colors,
      show_rownames = TRUE,
      show_colnames = TRUE,
      fontsize_row = 6,
      fontsize_col = 6,
      filename = file.path(output_dir, paste0(filename, ".pdf")),
      width = 12,
      height = 8
    )
  }

  # Process intra-modality attention
  if (!is.null(attention_results$intra_modality)) {
    for (modality in names(attention_results$intra_modality)) {
      matrix <- attention_results$intra_modality[[modality]]
      if (!is.null(matrix)) {
        importance <- rowSums(abs(matrix))
        top_n <- min(30, nrow(matrix))
        top_idx <- order(importance, decreasing = TRUE)[1:top_n]
        
        top_matrix <- matrix[top_idx, top_idx]
        feature_modalities <- rep(modality, length(top_idx))
        
        annotation_row <- data.frame(
          Modality = rep(modality, length(top_idx)),
          row.names = rownames(top_matrix)
        )
        
        create_heatmap(
          top_matrix,
          paste("Intra-modality Attention:", modality),
          paste0("intra_", modality, "_heatmap"),
          annotation_row = annotation_row,
          feature_modalities = feature_modalities
        )
      }
    }
  }

  # Process cross-modality attention
  if (!is.null(attention_results$cross_modality)) {
    for (pair in names(attention_results$cross_modality)) {
      matrix <- attention_results$cross_modality[[pair]]
      if (!is.null(matrix)) {
        modalities <- strsplit(pair, "_to_")[[1]]
        source_mod <- modalities[1]
        target_mod <- modalities[2]
        
        # Get top features based on row sums (source features)
        importance <- rowSums(abs(matrix))
        source_top_n <- min(30, nrow(matrix))
        source_top_idx <- order(importance, decreasing = TRUE)[1:source_top_n]
        
        # Get top features based on column sums (target features)
        target_importance <- colSums(abs(matrix))
        target_top_n <- min(30, ncol(matrix))
        target_top_idx <- order(target_importance, decreasing = TRUE)[1:target_top_n]
        
        # Create submatrix with top features
        top_matrix <- matrix[source_top_idx, target_top_idx]
        
        # Create annotations
        annotation_row <- data.frame(
          Modality = rep(source_mod, length(source_top_idx)),
          row.names = rownames(top_matrix)
        )
        
        annotation_col <- data.frame(
          Modality = rep(target_mod, length(target_top_idx)),
          row.names = colnames(top_matrix)
        )
        
        create_heatmap(
          top_matrix,
          paste("Cross-modality Attention:", pair),
          paste0("cross_", pair, "_heatmap"),
          annotation_col = annotation_col,
          annotation_row = annotation_row
        )
      }
    }
  }

  # Process global attention
  if (!is.null(attention_results$global)) {
    importance <- rowSums(abs(attention_results$global))
    top_n <- min(50, length(importance))
    top_idx <- order(importance, decreasing = TRUE)[1:top_n]
    
    top_matrix <- attention_results$global[top_idx, top_idx]
    
    # Get feature modalities for the selected features
    feature_modalities <- sapply(rownames(top_matrix), function(feat) {
      get_feature_modality(feat, cv_results)
    })
    
    annotation_row <- data.frame(
      Modality = feature_modalities,
      row.names = rownames(top_matrix)
    )
    
    create_heatmap(
      top_matrix,
      "Global Attention Patterns",
      "global_attention_heatmap",
      annotation_row = annotation_row,
      feature_modalities = feature_modalities
    )
  }

  # Create importance barplots
  create_importance_barplot <- function(importance_vector, title, filename, feature_modalities) {
    if (length(importance_vector) == 0) return(NULL)
    
    top_n <- min(30, length(importance_vector))
    top_idx <- order(importance_vector, decreasing = TRUE)[1:top_n]
    
    plot_data <- data.frame(
      feature = make_unique_names(names(importance_vector)[top_idx], feature_modalities[top_idx]),
      importance = importance_vector[top_idx],
      modality = feature_modalities[top_idx]
    )
    
    p <- ggplot(plot_data, aes(x = reorder(feature, importance), y = importance, fill = modality)) +
      geom_bar(stat = "identity") +
      scale_fill_brewer(palette = "Set1") +
      coord_flip() +
      theme_minimal() +
      labs(
        title = title,
        x = "Feature",
        y = "Importance Score"
      ) +
      theme(
        axis.text.y = element_text(size = 8),
        plot.title = element_text(size = 12)
      )
    
    ggsave(
      file.path(output_dir, paste0(filename, ".pdf")),
      p,
      width = 10,
      height = 8
    )
  }

  # Create importance plots for each modality
  for (modality in names(attention_results$intra_modality)) {
    if (!is.null(attention_results$intra_modality[[modality]])) {
      importance <- rowSums(abs(attention_results$intra_modality[[modality]]))
      create_importance_barplot(
        importance,
        paste(modality, "Feature Importance"),
        paste0(modality, "_importance"),
        rep(modality, length(importance))
      )
    }
  }
}
