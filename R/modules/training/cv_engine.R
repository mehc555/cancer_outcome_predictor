# R/modules/training/cv_engine.R

library(future)
library(future.apply)
library(progressr)
library(survival)
library(glmnet)



#' Extract stratification variable from datasets
#' @param datasets Torch MultiModalDataset or list
#' @param outcome_type Either "binary" or "survival"
#' @return Vector for stratification
get_stratification_vector <- function(datasets, outcome_type = "binary", outcome_var=NULL) {
    # Handle both MultiModalDataset and list inputs
    if (inherits(datasets, "MultiModalDataset")) {
        clinical_data <- as.matrix(datasets$data$clinical)
        clinical_features <- datasets$features$clinical
    } 
    
    if (outcome_type == "binary") {
        event_col <- which(clinical_features == outcome_var)
        if (length(event_col) > 0) {
            return(clinical_data[, event_col])
        }
    } else if (outcome_type == "survival") {
        # For survival, stratify by event status
        event_col <- which(clinical_features == "event")
        if (length(event_col) > 0) {
            return(factor(clinical_data[, event_col]))
        }
    }
    return(NULL)
}


# Updated custom collate function with feature preservation

custom_collate <- function(batch) {
  data_tensors <- list()
  mask_tensors <- list()
  outcome_tensors <- list()
  
  # Store features from the first batch item
  features <- NULL
  if (length(batch) > 0 && !is.null(batch[[1]]$features)) {
    features <- batch[[1]]$features
  }
  
  # Combine all modalities from the batch
  for (item in batch) {
    # Handle data and masks as before
    for (modality in names(item$data)) {
      if (!modality %in% names(data_tensors)) {
        data_tensors[[modality]] <- list()
        mask_tensors[[modality]] <- list()
      }
      
      if (is.data.frame(item$data[[modality]]) || is.matrix(item$data[[modality]])) {
        tensor_data <- torch_tensor(
          as.matrix(item$data[[modality]]), 
          dtype = torch_float32()
        )
        data_tensors[[modality]][[length(data_tensors[[modality]]) + 1]] <- tensor_data
        
        mask <- torch_tensor(
          !is.na(as.matrix(item$data[[modality]])),
          dtype = torch_float32()
        )
        mask_tensors[[modality]][[length(mask_tensors[[modality]]) + 1]] <- mask
      } else if (inherits(item$data[[modality]], "torch_tensor")) {
        data_tensors[[modality]][[length(data_tensors[[modality]]) + 1]] <- item$data[[modality]]
        if (!is.null(item$masks[[modality]])) {
          mask_tensors[[modality]][[length(mask_tensors[[modality]]) + 1]] <- item$masks[[modality]]
        }
      }
    }
    
    # Handle outcomes
    if (!is.null(item$outcomes)) {
      for (outcome_name in names(item$outcomes)) {
        if (!outcome_name %in% names(outcome_tensors)) {
          outcome_tensors[[outcome_name]] <- list()
        }
        outcome_tensors[[outcome_name]][[length(outcome_tensors[[outcome_name]]) + 1]] <- 
          torch_tensor(item$outcomes[[outcome_name]], dtype = torch_float32())
      }
    }
  }
  
  # Combine tensors
  collated <- list(
    data = list(),
    masks = list(),
    features = features,
    outcomes = list()
  )
  
  # Stack data and mask tensors
  for (modality in names(data_tensors)) {
    if (length(data_tensors[[modality]]) > 0) {
      collated$data[[modality]] <- torch_stack(data_tensors[[modality]])$squeeze(2)
      collated$masks[[modality]] <- torch_stack(mask_tensors[[modality]])$squeeze(2)
      
      # Replace NaN values with 0
      collated$data[[modality]] <- torch_where(
        torch_isnan(collated$data[[modality]]),
        torch_zeros_like(collated$data[[modality]]),
        collated$data[[modality]]
      )
    }
  }
  
  # Stack outcome tensors
  for (outcome_name in names(outcome_tensors)) {
    if (length(outcome_tensors[[outcome_name]]) > 0) {
      collated$outcomes[[outcome_name]] <- torch_stack(outcome_tensors[[outcome_name]])
    }
  }
  
  # Add feature names
  if (!is.null(features)) {
    for (modality in names(features)) {
      feature_name <- paste0(modality, "_features")
      collated[[feature_name]] <- features[[modality]]
    }
  }
  
  return(collated)
}


# Updated inspect batch function to also show features
inspect_batch <- function(batch, verbose = TRUE) {
  if (verbose) cat("\n=== Inspecting Batch Structure ===\n")
  
  # Check basic batch structure
  cat("\nBatch components:", paste(names(batch), collapse=", "), "\n")
  
  # Inspect data tensors
  if (!is.null(batch$data)) {
    cat("\nData modalities:", paste(names(batch$data), collapse=", "), "\n")
    for (modality in names(batch$data)) {
      if (inherits(batch$data[[modality]], "torch_tensor")) {
        cat(sprintf("\n%s tensor shape: %s", 
                   modality, 
                   paste(batch$data[[modality]]$size(), collapse=" x ")))
        cat(sprintf("\n%s tensor type: %s", 
                   modality, 
                   batch$data[[modality]]$dtype))
        
        # Show first few feature names if available
        feature_name <- paste0(modality, "_features")
        if (!is.null(batch[[feature_name]])) {
          cat(sprintf("\n%s first features: %s", 
                     modality,
                     paste(head(batch[[feature_name]], 3), collapse=", ")))
        }
      } else {
        cat(sprintf("\nWarning: %s is not a tensor! Class: %s", 
                   modality, 
                   class(batch$data[[modality]])))
      }
    }
  }
  
  # Inspect masks
  if (!is.null(batch$masks)) {
    cat("\n\nMask modalities:", paste(names(batch$masks), collapse=", "), "\n")
    for (modality in names(batch$masks)) {
      if (inherits(batch$masks[[modality]], "torch_tensor")) {
        cat(sprintf("\n%s mask shape: %s", 
                   modality, 
                   paste(batch$masks[[modality]]$size(), collapse=" x ")))
      } else {
        cat(sprintf("\nWarning: %s mask is not a tensor! Class: %s", 
                   modality, 
                   class(batch$masks[[modality]])))
      }
    }
  }
  
  # Show available features if present
  if (!is.null(batch$features)) {
    cat("\n\nFeature information available for modalities:", 
        paste(names(batch$features), collapse=", "))
  }
  
  # Return TRUE if everything is tensors, FALSE otherwise
  all_tensors <- TRUE
  
  # Check data tensors
  if (!is.null(batch$data)) {
    for (modality in names(batch$data)) {
      if (!inherits(batch$data[[modality]], "torch_tensor")) {
        all_tensors <- FALSE
      }
    }
  }
  
  # Check masks
  if (!is.null(batch$masks)) {
    for (modality in names(batch$masks)) {
      if (!inherits(batch$masks[[modality]], "torch_tensor")) {
        all_tensors <- FALSE
      }
    }
  }
  
  return(all_tensors)
}

#' Train model with multi-modal data and feature selection
#' @param model Neural network model
#' @param train_data Training data
#' @param val_data Validation data
#' @param config Training configuration
#' @param outcome_type Either "binary" or "survival"
#' @return Trained model, training history, and selected features

train_model <- function(model, train_data, val_data, config, outcome_type = "binary", outcome_var = NULL, selected_features = NULL) {
 
 if (is.null(selected_features)) {
    message("Performing feature selection on training data...")
    selected_features <- select_multimodal_features(
      train_data$data,
      n_features = list(
          cnv = config$model$architecture$modality_dims$cnv,
          clinical = config$model$architecture$modality_dims$clinical,
          expression = config$model$architecture$modality_dims$expression,
          mutations = config$model$architecture$modality_dims$mutations,
          methylation = config$model$architecture$modality_dims$methylation,
          mirna = config$model$architecture$modality_dims$mirna
      ),
      outcome_info = list(
          type = "binary",
          var = outcome_var
      )
    )
  }


  train_data_selected <- apply_feature_selection(train_data, selected_features)
  val_data_selected <- apply_feature_selection(val_data, selected_features)
   
  train_loader <- dataloader(
    dataset = train_data_selected,
    batch_size = config$model$batch_size,
    shuffle = TRUE,
    collate_fn = custom_collate
  )
  
  val_loader <- dataloader(
    dataset = val_data_selected,
    batch_size = config$model$batch_size,
    shuffle = FALSE,
    collate_fn = custom_collate
  )
 
    # Initialize optimizer and scheduler
  optimizer <- optim_adam(
    model$parameters,
    lr = config$model$optimizer$lr,
    weight_decay = config$model$optimizer$weight_decay
  )
  
  scheduler <- lr_reduce_on_plateau(
    optimizer,
    patience = config$model$scheduler$patience,
    factor = config$model$scheduler$factor
  )
  



  # Training state variables
  best_val_loss <- Inf
  patience_counter <- 0
  best_model_state <- model$state_dict()
  training_history <- list()
  
  # Main training loop
  for (epoch in 1:config$model$max_epochs) {
    # === TRAINING PHASE ===
    model$train()  # Set model to training mode
    train_losses <- c()
    train_metrics <- list()

    #batch <- coro::collect(train_loader, 1)[[1]]
    #cat("Batch data dimensions:\n")
    #for (name in names(batch$data)) {
    #if (!is.null(batch$data[[name]])) {
    #cat(sprintf("%s: %s\n", name, paste(batch$data[[name]]$size(), collapse=" x ")))
    #}
    #}
  

    # Training loop - WITH gradients
    coro::loop(for (batch in train_loader) {
      optimizer$zero_grad()  # Zero out gradients
      
      # Forward pass
      output <- model(batch$data)
      target <- batch$outcomes$binary$unsqueeze(2)
      
      # Compute loss
      loss <- compute_bce_loss(output$predictions, target)
      
      # Backward pass and optimization
      loss$backward()  # Compute gradients
      optimizer$step() # Update weights
      
      # Store metrics
      train_losses <- c(train_losses, loss$item())
      batch_metrics <- calculate_binary_metrics(output$predictions, target)
      
      # Accumulate batch metrics
      for (metric_name in names(batch_metrics)) {
        if (is.null(train_metrics[[metric_name]])) {
          train_metrics[[metric_name]] <- c()
        }
        train_metrics[[metric_name]] <- c(
          train_metrics[[metric_name]], 
          batch_metrics[[metric_name]]
        )
      }
    })
  
    # === VALIDATION PHASE ===
   model$eval()  # Set model to evaluation mode
    val_losses <- c()
    val_metrics <- list()
    
    # Validation loop - WITHOUT gradients
    with_no_grad({  # Disable gradient computation
      coro::loop(for (batch in val_loader) {
        # Forward pass only
        output <- model(batch$data)
        target <- batch$outcomes$binary$unsqueeze(2)
        
        # Compute validation loss
        loss <- compute_bce_loss(output$predictions, target)
        
        # Store metrics
        val_losses <- c(val_losses, loss$item())
        batch_metrics <- calculate_binary_metrics(output$predictions, target)
        
        # Accumulate batch metrics
        for (metric_name in names(batch_metrics)) {
          if (is.null(val_metrics[[metric_name]])) {
            val_metrics[[metric_name]] <- c()
          }
          val_metrics[[metric_name]] <- c(
            val_metrics[[metric_name]], 
            batch_metrics[[metric_name]]
          )
        }
      })
    })
     
   
    # === EPOCH SUMMARY ===
    if (length(train_losses) > 0 && length(val_losses) > 0) {
      # Calculate epoch metrics
      epoch_metrics <- list(
        train_loss = mean(train_losses),
        val_loss = mean(val_losses)
      )

      # Function to safely calculate mean
      safe_mean <- function(x) {
        if (is.numeric(x)) {
          return(mean(x, na.rm = TRUE))
        } else if (is.list(x)) {
          return(x[[length(x)]])  # Take the last value for lists
        } else {
          return(NA)
        }
      }
    
      # Add other metrics
      numeric_metrics <- c("accuracy", "precision", "recall", "f1", "auc", "n_valid", "n_total")

      for (metric_name in numeric_metrics) {
        if (!is.null(train_metrics[[metric_name]])) {
          epoch_metrics[[paste0("train_", metric_name)]] <- safe_mean(train_metrics[[metric_name]])
        }
        if (!is.null(val_metrics[[metric_name]])) {
          epoch_metrics[[paste0("val_", metric_name)]] <- safe_mean(val_metrics[[metric_name]])
        }
      }
    
      # Handle class balance separately
      if (!is.null(train_metrics$class_balance)) {
        epoch_metrics$train_class_balance <- train_metrics$class_balance[[length(train_metrics$class_balance)]]
      }
      if (!is.null(val_metrics$class_balance)) {
        epoch_metrics$val_class_balance <- val_metrics$class_balance[[length(val_metrics$class_balance)]]
      }

      # Store history
      training_history[[epoch]] <- epoch_metrics

      # Learning rate scheduling
      scheduler$step(epoch_metrics$val_loss)

      # Early stopping check
      if (epoch_metrics$val_loss < best_val_loss - config$model$early_stopping$min_delta) {
        best_val_loss <- epoch_metrics$val_loss
        best_model_state <- model$state_dict()
        patience_counter <- 0
      } else {
        patience_counter <- patience_counter + 1
      }
    
      # Print progress (with default print_every if not in config)
      print_every <- if (!is.null(config$model$print_every)) {
        config$model$print_every
      } else {
        1  # default to printing every epoch
      }
    	
    
     if (epoch %% print_every == 0) {
    	message(sprintf(
      	"\nEpoch %d/%d", epoch, config$model$max_epochs
       	))
    	message(sprintf(
      	"Train - Loss: %.4f, Acc: %.4f, F1: %.4f",
      	epoch_metrics$train_loss,
      	epoch_metrics$train_accuracy,
      	epoch_metrics$train_f1
    	))
    	message(sprintf(
      	"Val   - Loss: %.4f, Acc: %.4f, F1: %.4f",
      	epoch_metrics$val_loss,
      	epoch_metrics$val_accuracy,
      	epoch_metrics$val_f1
    	))
     
    	# Print class balance with safety checks
     
    	train_balance <- get_class_balance(train_metrics)
    	val_balance <- get_class_balance(val_metrics)

    	if (!is.null(train_balance)) {
      	message(sprintf(
        "Train Class Balance - Pos: %d, Neg: %d",
        train_balance$positive,
        train_balance$negative
      	))
    	}
     
    	if (!is.null(val_balance)) {
      	message(sprintf(
        "Val Class Balance   - Pos: %d, Neg: %d",
        val_balance$positive,
        val_balance$negative
      	))
    	}
     }
    
	epoch_metrics$train_class_balance <- get_class_balance(train_metrics)
  	epoch_metrics$val_class_balance <- get_class_balance(val_metrics)
    }
  }
  # Load best model and return results
  model$load_state_dict(best_model_state)  
  list(
    model = model,
    history = training_history,
    best_val_loss = best_val_loss,
    selected_features = selected_features
  )
  
}



#' Compute Cox partial likelihood loss
#' @param predictions Model predictions
#' @param times Survival times
#' @param events Event indicators
#' @return Cox loss
compute_cox_loss <- function(predictions, times, events) {
  # Convert to CPU for calculations
  pred <- predictions$cpu()
  times <- times$cpu()
  events <- events$cpu()
  
  # Sort by time in descending order
  sorted_indices <- order(times, decreasing = TRUE)
  pred <- pred[sorted_indices]
  times <- times[sorted_indices]
  events <- events[sorted_indices]
  
  # Calculate Cox partial likelihood
  n_samples <- length(times)
  log_risk <- torch_zeros(n_samples)
  
  for (i in 1:n_samples) {
    if (events[i] == 1) {
      # Calculate risk set (patients who haven't experienced event at time i)
      risk_set <- pred[times >= times[i]]
      # Compute log partial likelihood
      log_risk[i] <- pred[i] - torch_log(torch_sum(torch_exp(risk_set)))
    }
  }
  
  # Return negative log partial likelihood
  -torch_mean(log_risk)
}

# Helper function for calculating binary metrics with masks

# Modified binary metrics calculation
calculate_binary_metrics <- function(predictions, targets, masks = NULL, threshold = 0.5) {
  # Handle NaNs in predictions
  predictions <- torch_where(
    torch_isnan(predictions),
    torch_zeros_like(predictions),
    predictions
  )
  
  # Convert predictions to probabilities
  probs <- torch_sigmoid(predictions)
  
  # Try different thresholds to find optimal one
  thresholds <- seq(0.1, 0.9, by = 0.1)
  metrics_by_threshold <- lapply(thresholds, function(thresh) {
    preds <- (probs > thresh)$to(dtype = torch_long())
    
    # Calculate metrics
    tp <- torch_sum(preds * targets)$item()
    fp <- torch_sum(preds * (1 - targets))$item()
    fn <- torch_sum((1 - preds) * targets)$item()
    tn <- torch_sum((1 - preds) * (1 - targets))$item()
    
    # Calculate F1
    precision <- if (tp + fp > 0) tp / (tp + fp) else 0
    recall <- if (tp + fn > 0) tp / (tp + fn) else 0
    f1 <- if (precision + recall > 0) 2 * (precision * recall) / (precision + recall) else 0
    
    list(
      threshold = thresh,
      f1 = f1,
      precision = precision,
      recall = recall
    )
  })
  
  # Find best threshold
  f1_scores <- sapply(metrics_by_threshold, function(x) x$f1)
  best_idx <- which.max(f1_scores)
  best_threshold <- thresholds[best_idx]
  
  # Use best threshold for final predictions
  preds <- (probs > best_threshold)$to(dtype = torch_long())
  
  # Calculate final metrics
  tp <- torch_sum(preds * targets)$item()
  fp <- torch_sum(preds * (1 - targets))$item()
  fn <- torch_sum((1 - preds) * targets)$item()
  tn <- torch_sum((1 - preds) * (1 - targets))$item()
  
  # Calculate class distribution
  n_positive <- torch_sum(targets)$item()
  n_negative <- length(targets) - n_positive
  
  accuracy <- (tp + tn) / (tp + fp + fn + tn)
  precision <- if (tp + fp > 0) tp / (tp + fp) else 0
  recall <- if (tp + fn > 0) tp / (tp + fn) else 0
  f1 <- if (precision + recall > 0) 2 * (precision * recall) / (precision + recall) else 0
  
  # Calculate balanced accuracy
  sensitivity <- recall
  specificity <- tn / (tn + fp)
  balanced_accuracy <- (sensitivity + specificity) / 2
  
  # Calculate AUC
  auc <- tryCatch({
    probs_cpu <- as.numeric(probs)
    targets_cpu <- as.numeric(targets)
    pROC::auc(pROC::roc(targets_cpu, probs_cpu, quiet = TRUE))
  }, error = function(e) {
    NA
  })
  
  list(
    accuracy = accuracy,
    balanced_accuracy = balanced_accuracy,
    precision = precision,
    recall = recall,
    specificity = specificity,
    f1 = f1,
    auc = auc,
    best_threshold = best_threshold,
    n_valid = length(targets),
    class_balance = list(
      positive = n_positive,
      negative = n_negative
    ),
    confusion_matrix = list(
      tp = tp,
      fp = fp,
      fn = fn,
      tn = tn
    )
  )
}

# Helper function to print detailed metrics
print_detailed_metrics <- function(metrics) {
  cat("\nDetailed Performance Metrics:\n")
  cat(sprintf("Accuracy: %.4f\n", metrics$accuracy))
  cat(sprintf("Balanced Accuracy: %.4f\n", metrics$balanced_accuracy))
  cat(sprintf("Precision: %.4f\n", metrics$precision))
  cat(sprintf("Recall (Sensitivity): %.4f\n", metrics$recall))
  cat(sprintf("Specificity: %.4f\n", metrics$specificity))
  cat(sprintf("F1 Score: %.4f\n", metrics$f1))
  cat(sprintf("AUC: %.4f\n", metrics$auc))
  cat(sprintf("Best Threshold: %.2f\n", metrics$best_threshold))
  
  cat("\nClass Balance:\n")
  cat(sprintf("Positive: %d (%.1f%%)\n", 
              metrics$class_balance$positive,
              100 * metrics$class_balance$positive/metrics$n_valid))
  cat(sprintf("Negative: %d (%.1f%%)\n", 
              metrics$class_balance$negative,
              100 * metrics$class_balance$negative/metrics$n_valid))
  
  cat("\nConfusion Matrix:\n")
  cat(sprintf("True Positives: %d\n", metrics$confusion_matrix$tp))
  cat(sprintf("False Positives: %d\n", metrics$confusion_matrix$fp))
  cat(sprintf("False Negatives: %d\n", metrics$confusion_matrix$fn))
  cat(sprintf("True Negatives: %d\n", metrics$confusion_matrix$tn))
}


#' Calculate metrics for survival prediction
#' @param predictions Model predictions
#' @param times Survival times
#' @param events Event indicators
#' @return List of metrics
calculate_survival_metrics <- function(predictions, times, events) {
  # Convert to CPU for calculations
  preds <- as.numeric(predictions$cpu())
  times <- as.numeric(times$cpu())
  events <- as.numeric(events$cpu())
  
  # Calculate C-index
  c_index <- survival::survConcordance(
    Surv(times, events) ~ preds
  )$concordance
  
  # Calculate integrated Brier score if possible
  ibs <- tryCatch({
    max_time <- max(times)
    time_points <- seq(0, max_time, length.out = 100)
    pred_survival <- exp(-exp(preds))
    
    brier_scores <- sapply(time_points, function(t) {
      # Calculate Brier score at time t
      actual_survival <- events == 0 | times > t
      mean((actual_survival - pred_survival)^2)
    })
    
    mean(brier_scores)
  }, error = function(e) NA)
  
  list(
    c_index = c_index,
    ibs = ibs
  )
}

#' Create data splits for nested cross-validation
#' @param n_samples Total number of samples
#' @param n_repeats Number of repetitions (R)
#' @param n_outer_folds Number of outer folds (K)
#' @param n_inner_folds Number of inner folds (M)
#' @param validation_pct Percentage for validation
#' @param test_pct Percentage for testing
#' @param stratify Optional vector for stratification
#' @param seed Optional seed for reproducibility
#' @return List of CV split indices
create_cv_splits <- function(n_samples, n_repeats, n_outer_folds, n_inner_folds, 
                           validation_pct = 0.3, test_pct = 0.3, stratify = NULL, seed = NULL) {
    if (!is.null(seed)) {
        set.seed(seed)
    }
    
    # Input validation
    if (n_samples < 1) stop("n_samples must be positive")
    if (n_repeats < 1) stop("n_repeats must be positive")
    if (n_outer_folds < 2) stop("n_outer_folds must be at least 2")
    if (n_inner_folds < 2) stop("n_inner_folds must be at least 2")
    if (validation_pct <= 0 || validation_pct >= 1) stop("validation_pct must be between 0 and 1")
    if (test_pct <= 0 || test_pct >= 1) stop("test_pct must be between 0 and 1")
    
    repeated_splits <- lapply(1:n_repeats, function(r) {
        # First split: Validation vs rest
        validation_size <- floor(n_samples * validation_pct)
        validation_indices <- if (!is.null(stratify)) {
            create_stratified_split(1:n_samples, stratify, validation_size)
        } else {
            sample(1:n_samples, validation_size)
        }
        
        # Get remaining indices
        remaining_indices <- setdiff(1:n_samples, validation_indices)
        remaining_samples <- length(remaining_indices)
        
        # Second split: Test vs Train from remaining
        test_size <- floor(remaining_samples * test_pct)
        #test_indices <- if (!is.null(stratify)) {
        #    create_stratified_split(remaining_indices, stratify, test_size)
        #} else {
        #    sample(remaining_indices, test_size)
        #}
        
        #train_indices <- setdiff(remaining_indices, test_indices)
        
        # Create outer folds - each uses full training set
        outer_folds <- create_stratified_folds(
            #train_indices = train_indices,
            train_indices=remaining_indices,
	    stratify = if (!is.null(stratify)) stratify[remaining_indices] else NULL,
            n_folds = n_outer_folds,
            test_pct = test_pct
        )
        
        # Add inner folds to each outer fold
        outer_folds <- lapply(seq_along(outer_folds), function(k) {
            inner_folds <- create_stratified_folds(
                train_indices = outer_folds[[k]]$train_idx,
                stratify = if (!is.null(stratify)) stratify[outer_folds[[k]]$train_idx] else NULL,
                n_folds = n_inner_folds,
                test_pct = test_pct
            )
            
            names(inner_folds) <- paste0("M-Fold-", 1:n_inner_folds)
            
            list(
                name = paste0("K-Fold-", k),
                train_idx = outer_folds[[k]]$train_idx,
                test_idx =  outer_folds[[k]]$test_idx,
                inner_folds = inner_folds
            )
        })
        
        names(outer_folds) <- paste0("K-Fold-", 1:n_outer_folds)
        
        # Print detailed structure for first repeat
        if (r == 1) {
            message("\n=== Initial Split ===")
            message(sprintf("Total samples: %d", n_samples))
            message(sprintf("Validation set: %d samples (%.1f%%)", 
                          validation_size, 100 * validation_size/n_samples))
            message(sprintf("Remaining samples: %d", remaining_samples))
            
            message("\n=== Second Split of Remaining Data ===")
            message(sprintf("Test set: %d samples (%.1f%% of remaining)", 
                          test_size, 100 * test_size/remaining_samples))
            message(sprintf("Training set: %d samples", remaining_samples-test_size))
            
            message("\n=== K-fold Structure ===")
            message(sprintf("Number of K-folds: %d", n_outer_folds))
            message(sprintf("Full training set (each K-fold): %d samples", 
                          length(outer_folds[[1]]$train_idx)))
            message(sprintf("Full test set (all K-folds): %d samples", 
                          length(outer_folds[[1]]$test_idx)))
            
            message("\n=== M-fold Structure (per K-fold) ===")
            inner_test_size <- floor(length(outer_folds[[1]]$train_idx) * test_pct)
            inner_train_size <- length(outer_folds[[1]]$train_idx) - inner_test_size
            message(sprintf("Number of M-folds per K-fold: %d", n_inner_folds))
            message(sprintf("Inner test size: %d samples (%.1f%% of training)", 
                          inner_test_size, 100 * test_pct))
            message(sprintf("Inner train size: %d samples", inner_train_size))
            
            if (!is.null(stratify)) {
                message("\n=== Stratification Balance ===")
                strata <- unique(stratify)
                for (split_name in c("Validation", "Test", "Training")) {
                    indices <- switch(split_name,
                                   "Validation" = validation_indices,
                                   "Test" = outer_folds[[1]]$test_idx,
                                   "Training" = outer_folds[[1]]$train_idx)
                    counts <- table(stratify[indices])
                    props <- prop.table(counts)
                    message(sprintf("\n%s set distribution:", split_name))
                    for (s in names(counts)) {
                        message(sprintf("  - %s: %.1f%% (%d samples)", 
                                      s, 100 * props[s], counts[s]))
                    }
                }
            }
        }
        
        list(
            iteration = r,
            validation = validation_indices,
            outer_splits = outer_folds
        )
    })
    
    attr(repeated_splits, "summary") <- list(
        n_samples = n_samples,
        n_repeats = n_repeats,
        n_outer_folds = n_outer_folds,
        n_inner_folds = n_inner_folds,
        stratified = !is.null(stratify),
        validation_pct = validation_pct,
        test_pct = test_pct
    )
    
    return(repeated_splits)
}


#' Create stratified split
#' @param indices Indices to split
#' @param stratify Stratification vector
#' @param size Size of split
#' @return Vector of indices for split
create_stratified_split <- function(indices, stratify, size) {
    strata <- unique(stratify)
    split_indices <- c()
    
    for (stratum in strata) {
        stratum_indices <- indices[stratify[indices] == stratum]
        stratum_size <- floor(size * length(stratum_indices) / length(indices))
        split_indices <- c(split_indices, sample(stratum_indices, stratum_size))
    }
    
    return(split_indices)
}

#' Create stratified folds for inner/outer splits
#' @param train_indices Training indices to split
#' @param stratify Stratification vector
#' @param n_folds Number of folds
#' @param test_pct Percentage for test set
#' @return List of folds with train and test indices

create_stratified_folds <- function(train_indices, stratify, n_folds, test_pct = 0.3) {
    folds <- lapply(1:n_folds, function(i) {
        # Calculate test size
	test_size <- floor(length(train_indices) * test_pct)
        if (!is.null(stratify)) {
            # Stratified sampling for test set
            strata <- unique(stratify)
            test_idx <- c()
            #print(strata)
            for (stratum in strata) {
                stratum_indices <- train_indices[stratify == stratum]
                stratum_size <- floor(test_size * length(stratum_indices) / length(train_indices))
                test_idx <- c(test_idx, sample(stratum_indices, stratum_size))
                #print(test_idx)
	    }
        } else {
            # Random sampling for test set
            test_idx <- sample(train_indices, test_size)
	}
        # Get training indices by excluding test indices
        fold_train_idx <- setdiff(train_indices, test_idx)
        
        list(
            train_idx = fold_train_idx,  # Use remaining indices for training
            test_idx = test_idx          # Sampled test set
        )
    })
    
    return(folds)
}

#' Memory-efficient dataset subsetting with proper tensor and outcome handling
#' @param datasets MultiModalDataset object
#' @param indices Indices to subset
#' @param batch_size Batch size for data loading
#' @return Subsetted MultiModalDataset

subset_datasets <- function(datasets, indices, batch_size = 32) {
    cat("1. Starting dataset subsetting\n")
    
    if (!inherits(datasets, "MultiModalDataset")) {
        stop("datasets must be a MultiModalDataset object")
    }
    
    cat("2. Creating new data list\n")
    subsetted_data <- list()
    
    cat("3. Processing data and masks for each modality\n")
    modalities <- c("clinical", "cnv", "expression", "mutations", "methylation", "mirna")
    for (modality in modalities) {
        if (!is.null(datasets$data[[modality]])) {
            cat(sprintf("   - Processing %s\n", modality))
            
            # Subset data
            subsetted_data[[modality]] <- datasets$data[[modality]][indices, , drop = FALSE]
            
            # Subset corresponding mask
            mask_name <- paste0(modality, "_mask")
            if (!is.null(datasets$data[[mask_name]])) {
                subsetted_data[[mask_name]] <- datasets$data[[mask_name]][indices, , drop = FALSE]
                
                cat(sprintf("     Maintained mask with %.1f%% valid values\n",
                          100 * mean(subsetted_data[[mask_name]], na.rm=TRUE)))
            }
        }
    }
    
    cat("4. Processing outcomes if present\n")
    # Handle outcomes
    subsetted_outcomes <- NULL
    if (!is.null(datasets$outcomes)) {
        subsetted_outcomes <- list()
        for (outcome_name in names(datasets$outcomes)) {
            original_outcome <- datasets$outcomes[[outcome_name]]
            if (!is.null(original_outcome)) {
                subsetted_outcomes[[outcome_name]] <- original_outcome[indices]
            }
        }
    }
    
    cat("5. Creating new dataset\n")
    new_dataset <- dataset(
        name = "MultiModalDataset",
        initialize = function() {
            self$data <- subsetted_data
            self$features <- datasets$features
            self$n_samples <- length(indices)
            self$sample_ids <- datasets$sample_ids
            self$unified_sample_ids <- datasets$unified_sample_ids[indices]
            self$sample_id_to_index <- datasets$sample_id_to_index
            self$outcome_info <- datasets$outcome_info
            self$outcomes <- subsetted_outcomes
        },
        .getitem = function(index) {
            # Initialize lists for data and masks
            data_list <- list()
            masks_list <- list()
            
            # Get current sample ID
            current_sample_id <- self$unified_sample_ids[index]
            
            # Process each modality
            for (modality in names(self$data)) {
                if (!grepl("_mask$", modality) && !is.null(self$data[[modality]])) {
                    # Get data for current sample
                    data_values <- as.matrix(self$data[[modality]][index, -1, drop = FALSE])
                    
                    # Convert to tensor
                    data_list[[modality]] <- torch_tensor(
                        data_values,
                        dtype = torch_float32()
                    )
                    
                    # Get corresponding mask
                    mask_name <- paste0(modality, "_mask")
                    if (!is.null(self$data[[mask_name]])) {
                        masks_list[[modality]] <- torch_tensor(
                            self$data[[mask_name]][index, , drop = FALSE],
                            dtype = torch_float32()
                        )
                    } else {
                        # Create mask from data if not stored
                        masks_list[[modality]] <- torch_tensor(
                            !is.na(data_values),
                            dtype = torch_float32()
                        )
                    }
                    
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
        .length = function() self$n_samples,
        get_feature_names = datasets$get_feature_names,
        get_sample_ids = datasets$get_sample_ids
    )()
    
    return(new_dataset)
}


#' Run nested cross-validation
#' @param model Neural network model
#' @param datasets List of torch datasets
#' @param config Configuration parameters
#' @param cancer_type Current cancer type
#' @param outcome_type Either "binary" or "survival"
#' @param validation_pct Percentage for validation
#' @param test_pct Percentage for testing
#' @param seed Optional seed for reproducibility
#' @param max_workers Maximum number of parallel workers
#' @param batch_size Batch size for data loading
#' @return List of results and models

run_nested_cv <- function(model, datasets, config, cancer_type, 
                         outcome_type = "binary",
                         validation_pct = 0.3, test_pct = 0.3, 
                         seed = NULL, max_workers = 2, batch_size = 32, outcome_var=NULL) {
    
    # Initial setup remains the same until outer fold processing
    gc()
    
    available_memory <- as.numeric(system("free -g | awk 'NR==2 {print $4}'", intern=TRUE))
    actual_workers <- max(1, min(max_workers, floor(available_memory / 1)))
    message(sprintf("Using %d workers based on available memory", actual_workers))
    
    message("\nVerifying initial dataset:")
    message(sprintf("- Total samples: %d", datasets$n_samples))
    message(sprintf("- Modalities: %s", paste(names(datasets$features), collapse=", ")))
    
    stratify <- as.numeric(get_stratification_vector(datasets, outcome_type, outcome_var))
    n_samples <- datasets$n_samples
    
    cv_splits <- create_cv_splits(
        n_samples = n_samples,
        n_repeats = config$cv_params$outer_repeats,
        n_outer_folds = config$cv_params$outer_folds,
        n_inner_folds = config$cv_params$inner_folds,
        validation_pct = validation_pct,
        test_pct = test_pct,
        stratify = stratify,
        seed = seed
    )
    
    validation_results <- validate_cv_splits(cv_splits)
    
    if (validation_results$summary$total_overlaps > 0) {
        warning(sprintf("Found %d overlapping indices across %d checks (%.2f%%). Maximum overlap: %.2f%%",
                       validation_results$summary$total_overlaps,
                       validation_results$summary$total_checks,
                       validation_results$summary$overlap_percentage,
                       validation_results$summary$max_overlap_percentage))
    }
    
    plan(multisession, workers = actual_workers)
    #future::plan(tweak(multisession, workers = actual_workers, rng = TRUE, future.seed=TRUE))

    results <- vector("list", length(cv_splits))
    fold_features <- list()
    

        # Process each repeat
    for (repeat_idx in 1:2) {
        message(sprintf("Processing repeat %d/%d", repeat_idx, length(cv_splits)))
        repeat_split <- cv_splits[[repeat_idx]]
        fold_features[[repeat_idx]] <- list()

        # Process outer folds in parallel
        outer_results <- future_lapply(seq_along(repeat_split$outer_splits),
                                     function(fold_idx) {
            model_copy <- model$create_copy()
            outer_split <- repeat_split$outer_splits[[fold_idx]]

            # Initialize feature storage for this fold
            fold_features[[repeat_idx]][[fold_idx]] <- list(
                outer = NULL,
                inner = list()
            )

            # Process inner folds to determine best hyperparameters
            inner_results <- lapply(seq_along(outer_split$inner_folds), function(inner_idx) {
                inner_model <- model$create_copy()

                # Create datasets for inner fold
                inner_train_data <- subset_datasets(datasets, outer_split$inner_folds[[inner_idx]]$train_idx, batch_size)
                inner_val_data <- subset_datasets(datasets, outer_split$inner_folds[[inner_idx]]$test_idx, batch_size)

                # Train inner model with feature selection
                trained_model <- train_model(
                    model = inner_model,
                    train_data = inner_train_data,
                    val_data = inner_val_data,  # Using inner validation set for early stopping
                    config = config,
                    outcome_type = outcome_type,
		    outcome_var = outcome_var
                )

                # Clean up

    		# Prepare validation data with correct features
		validation_data_selected <- prepare_for_evaluation(trained_model$model, inner_val_data, trained_model$selected_features)
		
		# Get validation performance
		val_results <- evaluate_model(trained_model$model, inner_val_data, outcome_type)
    
		#print_detailed_metrics(val_results$metrics)
		rm(inner_model, inner_train_data, inner_val_data)
                gc()

    		list(
		state_dict = trained_model$model$state_dict(),
        	val_metrics = val_results$metrics,
        	selected_features = trained_model$selected_features,
    		best_val_loss = trained_model$best_val_loss
		)
            })


	    # Find best performing inner model configuration
	    best_inner_idx <- which.min(sapply(inner_results, function(x) x$best_val_loss))
	    best_state_dict<- inner_results[[best_inner_idx]]$state_dict
	    best_features <- inner_results[[best_inner_idx]]$selected_features

	    # Train final model on outer fold using best configuration
	    outer_train_data <- subset_datasets(datasets, outer_split$train_idx, batch_size)
	    outer_val_data <- subset_datasets(datasets, outer_split$test_idx, batch_size)

	    # Initialize model with best configuration
	    model_copy <- model$create_copy()
	    model_copy$load_state_dict(best_state_dict)

	    # Train final model using best configuration from inner folds
	    final_model <- train_model(
    	    model = model_copy,
            train_data = outer_train_data,
            val_data = outer_val_data,
            config = config,
            outcome_type = outcome_type,
            outcome_var = outcome_var,
            selected_features = best_features  # Pass best features from inner folds
	    )

            # Prepare validation data with correct features

            validation_data_selected <- prepare_for_evaluation(final_model$model, outer_val_data, final_model$selected_features)
	
	    # Evaluate on the same validation set

	    test_results <- evaluate_model(final_model$model, outer_val_data, outcome_type)

            # Store selected features for outer fold

            fold_features[[repeat_idx]][[fold_idx]]$outer <- final_model$selected_features

            # Save intermediate results
            results_dir <- file.path(config$main$paths$results_dir, cancer_type)
            if (!dir.exists(results_dir)) dir.create(results_dir, recursive = TRUE)

            #saveRDS(
            #    list(
            #        model = final_model,
            #        features = fold_features[[repeat_idx]][[fold_idx]],
            #        test_results = test_results,
            #        inner_results = inner_results
            #    ),
            #    file.path(results_dir,
            #             sprintf("repeat_%d_fold_%d_results.rds",
            #                    repeat_idx, fold_idx))
            #)

            # Clean up
            rm(model_copy, actual_train_data, val_data, test_data)
            gc()

            list(
                model = final_model$model,
		state_dict = final_model$model$state_dict(),
                val_metrics = test_results$metrics,
		selected_features = final_model$selected_features,
		best_val_loss = final_model$best_val_loss
           )
        })

        # Evaluate on hold-out validation set
        validation_data <- subset_datasets(datasets, repeat_split$validation, batch_size)
        
	best_outer_idx <- which.min(sapply(outer_results, function(x) x$best_val_loss))
        best_outer_model <- outer_results[[best_outer_idx]]$model
	best_outer_state_dict<- outer_results[[best_outer_idx]]$state_dict
        best_outer_features <- outer_results[[best_outer_idx]]$selected_features
        best_outer_val_loss <- outer_results[[best_outer_idx]]$best_val_loss

        # Prepare validation data with correct features
        validation_data_selected <- prepare_for_evaluation(best_outer_model, validation_data, best_outer_features)

        # Now evaluate the model
        validation_results <- evaluate_model(best_outer_model, validation_data_selected, outcome_type)

	results[[repeat_idx]] <- list(
            repeat_idx = repeat_idx,
            outer_results = outer_results,
            validation_results = validation_results,
            best_model = best_outer_model,
            features = best_outer_features,
	    best_val_loss = best_outer_val_loss
        )

        # Save results
        saveRDS(
            results[[repeat_idx]],
            file.path(config$main$paths$results_dir, cancer_type,
                     sprintf("repeat_%d_results.rds", repeat_idx))
        )

        feature_summary <- analyze_feature_selection(fold_features[[repeat_idx]])
        saveRDS(
            feature_summary,
            file.path(config$main$paths$results_dir, cancer_type,
                     sprintf("repeat_%d_feature_summary.rds", repeat_idx))
        )
    }

    final_results <- aggregate_cv_results(results)
    #print_cv_results(final_results)

    final_model <- select_final_model(results)
    
    final_results$feature_analysis <- analyze_feature_selection()
    
    list(
        results = final_results,
        final_model = final_model,
        cv_splits = cv_splits,
        raw_results = results,
        split_validation = validation_results,
        fold_features = fold_features
    )
}


#' Analyze feature selection patterns across folds
#' @param fold_features Nested list of selected features
#' @return Analysis of feature selection consistency
analyze_feature_selection <- function(fold_features) {
    # Initialize storage for feature counts
    feature_counts <- list()
    
    # Process each modality separately
    modalities <- c("cnv", "clinical", "expression", "mutations", "methylation", "mirna")
    
    for (modality in modalities) {
        # Collect all features selected for this modality
        all_features <- list()
        
        # Process outer folds
        for (fold in fold_features) {
            if (!is.null(fold$outer[[modality]])) {
                all_features <- c(all_features, fold$outer[[modality]])
            }
            
            # Process inner folds
            for (inner in fold$inner) {
                if (!is.null(inner[[modality]])) {
                    all_features <- c(all_features, inner[[modality]])
                }
            }
        }
        
        # Calculate feature frequencies
        if (length(all_features) > 0) {
            feature_counts[[modality]] <- sort(table(unlist(all_features)), decreasing = TRUE)
        }
    }
    
    # Calculate summary statistics
    summary_stats <- lapply(feature_counts, function(counts) {
        list(
            n_unique_features = length(counts),
            top_features = names(head(counts, 10)),
            feature_frequencies = counts,
            consistency_score = mean(counts) / max(counts)
        )
    })
    
    list(
        feature_counts = feature_counts,
        summary_stats = summary_stats
    )
}

#' Select best model based on validation performance
#' @param outer_results List of outer fold results containing test performance
#' @return Best performing model
select_best_model <- function(outer_results) {
    # Extract test loss from each outer fold result
    performances <- sapply(outer_results, function(r) {
        r$best_val_loss  # Or another metric you prefer like 'f1', 'accuracy', etc.
    })
    
    # Find best performing model
    best_idx <- which.min(performances)
    
    # Return the best model
    outer_results[[best_idx]]$model
}


#' Select final model from all repeats
#' @param results List of results from all repeats
#' @return Best performing model
select_final_model <- function(results) {
    performances <- sapply(results, function(r) r$best_val_loss)
    best_repeat <- which.min(performances)
    results[[best_repeat]]$best_model
}


#' Aggregate CV results
#' @param results List of results from all repeats
#' @return Aggregated results summary with means and standard deviations
aggregate_cv_results <- function(results) {
    # Extract validation metrics from each repeat
    validation_metrics <- lapply(results, function(r) {
        metrics <- r$validation_results$metrics
        
        # Extract numeric metrics
        main_metrics <- list(
            accuracy = metrics$accuracy,
            balanced_accuracy = metrics$balanced_accuracy,
            precision = metrics$precision,
            recall = metrics$recall,
            specificity = metrics$specificity,
            f1 = metrics$f1,
            auc = as.numeric(metrics$auc)  # Convert AUC object to numeric
        )
        
        # Extract class balance
        class_balance <- list(
            positive_samples = metrics$class_balance$positive,
            negative_samples = metrics$class_balance$negative,
            pos_neg_ratio = metrics$class_balance$positive / metrics$class_balance$negative
        )
        
        # Extract confusion matrix
        confusion <- list(
            true_positives = metrics$confusion_matrix$tp,
            false_positives = metrics$confusion_matrix$fp,
            false_negatives = metrics$confusion_matrix$fn,
            true_negatives = metrics$confusion_matrix$tn
        )
        
        list(
            metrics = main_metrics,
            class_balance = class_balance,
            confusion = confusion
        )
    })
    
    # Function to compute summary statistics
    compute_summary <- function(values) {
        c(
            mean = mean(unlist(values), na.rm = TRUE),
            sd = sd(unlist(values), na.rm = TRUE),
            min = min(unlist(values), na.rm = TRUE),
            max = max(unlist(values), na.rm = TRUE)
        )
    }
    
    # Compute summaries for main metrics
    metric_names <- names(validation_metrics[[1]]$metrics)
    metrics_summary <- lapply(metric_names, function(metric) {
        values <- sapply(validation_metrics, function(x) x$metrics[[metric]])
        compute_summary(values)
    })
    names(metrics_summary) <- metric_names
    
    # Compute summaries for class balance
    balance_names <- names(validation_metrics[[1]]$class_balance)
    balance_summary <- lapply(balance_names, function(metric) {
        values <- sapply(validation_metrics, function(x) x$class_balance[[metric]])
        compute_summary(values)
    })
    names(balance_summary) <- balance_names
    
    # Compute summaries for confusion matrix
    confusion_names <- names(validation_metrics[[1]]$confusion)
    confusion_summary <- lapply(confusion_names, function(metric) {
        values <- sapply(validation_metrics, function(x) x$confusion[[metric]])
        compute_summary(values)
    })
    names(confusion_summary) <- confusion_names
    
    # Return organized results
    list(
        performance = list(
            metrics = metrics_summary,
            class_balance = balance_summary,
            confusion = confusion_summary
        ),
        n_repeats = length(results),
        n_samples = results[[1]]$validation_results$metrics$n_valid
    )
}

#' Print formatted CV results
#' @param results Aggregated CV results
#' @export
print_cv_results <- function(results) {
    cat("\n=== Cross-Validation Results ===")
    cat(sprintf("\nNumber of repeats: %d", results$n_repeats))
    cat(sprintf("\nNumber of samples: %d", results$n_samples))
    
    # Print main metrics
    cat("\n\n=== Performance Metrics ===")
    for(metric in names(results$performance$metrics)) {
        stats <- results$performance$metrics[[metric]]
        cat(sprintf("\n%s:", gsub("_", " ", toupper(metric))))
        cat(sprintf("\n  Mean ± SD: %.4f ± %.4f", stats["mean"], stats["sd"]))
        cat(sprintf("\n  Range: [%.4f, %.4f]", stats["min"], stats["max"]))
    }
    
    # Print class balance
    cat("\n\n=== Class Balance ===")
    for(metric in names(results$performance$class_balance)) {
        stats <- results$performance$class_balance[[metric]]
        cat(sprintf("\n%s:", gsub("_", " ", toupper(metric))))
        cat(sprintf("\n  Mean ± SD: %.2f ± %.2f", stats["mean"], stats["sd"]))
        cat(sprintf("\n  Range: [%.2f, %.2f]", stats["min"], stats["max"]))
    }
    
    # Print confusion matrix
    cat("\n\n=== Confusion Matrix Statistics ===")
    for(metric in names(results$performance$confusion)) {
        stats <- results$performance$confusion[[metric]]
        cat(sprintf("\n%s:", gsub("_", " ", toupper(metric))))
        cat(sprintf("\n  Mean ± SD: %.2f ± %.2f", stats["mean"], stats["sd"]))
        cat(sprintf("\n  Range: [%.2f, %.2f]", stats["min"], stats["max"]))
    }
    cat("\n")
}

#' Evaluate model performance
#' @param model Trained model
#' @param data Test data (MultiModalDataset)
#' @param outcome_type Either "binary" or "survival"
#' @return List of evaluation metrics and predictions
evaluate_model <- function(model, data, outcome_type = "binary") {
    model$eval()
    loader <- dataloader(dataset = data, batch_size = 32, shuffle = FALSE, collate_fn = custom_collate)
    
    predictions <- list()
    targets <- list()
    times <- list()
    events <- list()
    
    with_no_grad({
        coro::loop(for (batch in loader) {
            output <- model(batch$data)
            predictions[[length(predictions) + 1]] <- output$predictions$cpu()
            
            if (outcome_type == "binary") {
                # Extract binary outcome tensor
                if (!is.null(batch$outcomes$binary)) {
                    targets[[length(targets) + 1]] <- batch$outcomes$binary$unsqueeze(2)$cpu()
                } else {
                    stop("Binary outcomes not found in batch")
                }
            } else {
                times[[length(times) + 1]] <- batch$time$cpu()
                events[[length(events) + 1]] <- batch$event$cpu()
            }
        })
    })
    
    all_predictions <- torch_cat(predictions, dim = 1)
    
    if (outcome_type == "binary") {
        # Stack all target tensors
        all_targets <- torch_cat(targets, dim = 1)
        metrics <- calculate_binary_metrics(all_predictions, all_targets)
        
        results <- list(
            metrics = metrics,
            predictions = all_predictions,
            targets = all_targets
        )
    } else {
        all_times <- torch_cat(times, dim = 1)
        all_events <- torch_cat(events, dim = 1)
        metrics <- calculate_survival_metrics(all_predictions, all_times, all_events)
        
        results <- list(
            metrics = metrics,
            predictions = all_predictions,
            times = all_times,
            events = all_events
        )
    }
    
    results
}


#' Generate performance visualization
#' @param results CV results
#' @param outcome_type Either "binary" or "survival"
#' @return List of plots
generate_performance_visualization <- function(results, outcome_type = "binary") {
    library(ggplot2)
    library(tidyr)
    library(dplyr)

    plots <- list()

    # Convert results to data frame
    metrics_df <- do.call(rbind, lapply(1:length(results), function(i) {
        iteration_results <- results[[i]]

        # Extract validation metrics
        val_metrics <- as.data.frame(t(unlist(iteration_results$validation_results$metrics)))
        val_metrics$iteration <- i
        val_metrics$fold <- "validation"

        # Extract outer fold metrics
        outer_metrics <- do.call(rbind, lapply(1:length(iteration_results$outer_results),
                                             function(j) {
            fold_metrics <- as.data.frame(t(unlist(
                iteration_results$outer_results[[j]]$history[[
                    length(iteration_results$outer_results[[j]]$history)
                ]]
            )))
            fold_metrics$iteration <- i
            fold_metrics$fold <- paste0("fold_", j)
            fold_metrics
        }))

        rbind(val_metrics, outer_metrics)
    }))

    # Create performance plots based on outcome type
    if (outcome_type == "binary") {
        # ROC curve plot
        plots$roc <- ggplot(metrics_df, aes(x = 1 - specificity, y = sensitivity,
                                          color = fold)) +
            geom_line() +
            geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
            theme_minimal() +
            labs(title = "ROC Curves across CV Folds",
                 x = "False Positive Rate",
                 y = "True Positive Rate")

        # Metrics boxplot
        long_metrics <- metrics_df %>%
            pivot_longer(cols = c("accuracy", "precision", "recall", "f1", "auc"),
                        names_to = "metric", values_to = "value")

        plots$metrics <- ggplot(long_metrics, aes(x = metric, y = value)) +
            geom_boxplot() +
            theme_minimal() +
            labs(title = "Performance Metrics Distribution",
                 x = "Metric", y = "Value")

    } else {
        # Kaplan-Meier curves
        plots$km <- ggplot(metrics_df, aes(x = time, y = survival_prob,
                                         color = risk_group)) +
            geom_step() +
            facet_wrap(~iteration) +
            theme_minimal() +
            labs(title = "Kaplan-Meier Curves by Risk Group",
                 x = "Time", y = "Survival Probability")

        # C-index distribution
        plots$cindex <- ggplot(metrics_df, aes(x = fold, y = c_index)) +
            geom_boxplot() +
            theme_minimal() +
            labs(title = "C-index Distribution across Folds",
                 x = "Fold", y = "C-index")
    }

    plots
}

#' Save complete analysis results
#' @param results CV results
#' @param cancer_type Cancer type
#' @param config Configuration
#' @param output_dir Output directory

save_analysis_results <- function(results, cancer_type, config, output_dir) {
    dir.create(file.path(output_dir, cancer_type), recursive = TRUE, 
              showWarnings = FALSE)
    
    saveRDS(results$results, 
            file.path(output_dir, cancer_type, "performance_metrics.rds"))
    
    if (!is.null(results$feature_importance)) {
        saveRDS(results$feature_importance,
                file.path(output_dir, cancer_type, "feature_importance.rds"))
    }
    
    if (!is.null(results$plots)) {
        for (plot_name in names(results$plots)) {
            ggsave(
                filename = file.path(output_dir, cancer_type, 
                                   paste0(plot_name, ".pdf")),
                plot = results$plots[[plot_name]],
                width = 10,
                height = 8
            )
        }
    }
    
    saveRDS(config, file.path(output_dir, cancer_type, "config.rds"))
    
    report <- generate_summary_report(results, cancer_type, config)
    writeLines(report, file.path(output_dir, cancer_type, "summary_report.txt"))
}

#' Generate summary report
#' @param results CV results
#' @param cancer_type Cancer type
#' @param config Configuration
#' @return Text report
generate_summary_report <- function(results, cancer_type, config) {
    report <- c(
        "=== Analysis Summary Report ===\n",
        sprintf("Cancer Type: %s\n", cancer_type),
        sprintf("Date: %s\n", Sys.Date()),
        "\nConfiguration:",
        sprintf("- Outcome type: %s", config$model$outcome_type),
        sprintf("- Number of repeats: %d", config$cv_params$outer_repeats),
        sprintf("- Number of outer folds: %d", config$cv_params$outer_folds),
        sprintf("- Number of inner folds: %d", config$cv_params$inner_folds),
        "\nPerformance Summary:",
        sprintf("- Mean validation performance: %.3f", 
                mean(results$results$validation_summary$mean)),
        sprintf("- SD validation performance: %.3f", 
                mean(results$results$validation_summary$sd)),
        "\nTop Features:",
        paste("- ", names(head(sort(results$feature_importance, decreasing = TRUE), 10)),
              collapse = "\n"),
        "\nModel Architecture:",
        sprintf("- Number of parameters: %d", 
                sum(sapply(results$final_model$parameters, function(p) prod(p$size())))),
        "\nTraining Details:",
        sprintf("- Best epoch: %d", 
                which.min(results$final_model$history$val_loss)),
        sprintf("- Final learning rate: %.6f", 
                results$final_model$history$lr[length(results$final_model$history$lr)])
    )
    
    paste(report, collapse = "\n")
}

# Check if two sets of indices have any overlap
check_index_overlap <- function(set1, set2) {
  intersection <- intersect(set1, set2)
  overlap_pct <- length(intersection) / min(length(set1), length(set2)) * 100
  list(
    has_overlap = length(intersection) > 0,
    overlap_indices = intersection,
    overlap_percentage = overlap_pct
  )
}

# Validate all CV splits
validate_cv_splits <- function(cv_splits) {
  validation_results <- list()

  for (r in seq_along(cv_splits)) {
    repeat_split <- cv_splits[[r]]
    repeat_results <- list()

    # Check validation vs training/test
    for (k in seq_along(repeat_split$outer_splits)) {
      outer_split <- repeat_split$outer_splits[[k]]

      # Check validation vs training
      val_train_check <- check_index_overlap(
        repeat_split$validation,
        outer_split$train_idx
      )

      # Check validation vs test
      val_test_check <- check_index_overlap(
        repeat_split$validation,
        outer_split$test_idx
      )

      # Check inner folds
      inner_fold_results <- list()
      for (m in seq_along(outer_split$inner_folds)) {
        inner_fold <- outer_split$inner_folds[[m]]

        # Check inner train vs test
        inner_check <- check_index_overlap(
          inner_fold$train_idx,
          inner_fold$test_idx
        )

        inner_fold_results[[m]] <- list(
          fold = m,
          train_test_overlap = inner_check
        )
      }

      repeat_results[[k]] <- list(
        fold = k,
        validation_train_overlap = val_train_check,
        validation_test_overlap = val_test_check,
        inner_folds = inner_fold_results
      )
    }

    validation_results[[r]] <- list(
      repeatL = r,
      folds = repeat_results
    )
  }

  # Calculate summary statistics
  total_checks <- 0
  total_overlaps <- 0
  max_overlap_pct <- 0

  for (r in validation_results) {
    for (k in r$folds) {
      # Count validation checks
      total_checks <- total_checks + 2  # validation vs train/test
      if (k$validation_train_overlap$has_overlap) total_overlaps <- total_overlaps + 1
      if (k$validation_test_overlap$has_overlap) total_overlaps <- total_overlaps + 1
      max_overlap_pct <- max(max_overlap_pct,
                           k$validation_train_overlap$overlap_percentage,
                           k$validation_test_overlap$overlap_percentage)

      # Count inner fold checks
      for (m in k$inner_folds) {
        total_checks <- total_checks + 1
        if (m$train_test_overlap$has_overlap) total_overlaps <- total_overlaps + 1
        max_overlap_pct <- max(max_overlap_pct, m$train_test_overlap$overlap_percentage)
      }
    }
  }

  list(
    detailed_results = validation_results,
    summary = list(
      total_checks = total_checks,
      total_overlaps = total_overlaps,
      overlap_percentage = (total_overlaps / total_checks) * 100,
      max_overlap_percentage = max_overlap_pct
    )
  )
}

#' Validate sample consistency across data modalities
#' @param datasets MultiModalDataset or list of data files
#' @param cancer_type Cancer type identifier
#' @return TRUE if validation passes, stops with error if issues found
validate_sample_consistency <- function(datasets, cancer_type) {
    # Handle both MultiModalDataset and file path inputs
    if (inherits(datasets, "MultiModalDataset")) {
        # Get sample IDs from each modality
        sample_ids <- lapply(datasets$data, function(x) {
            if (inherits(x, "torch_tensor")) {
                # Assuming first column is sample ID
                as.character(x[,1]$cpu())
            } else {
                NULL
            }
        })

        # Remove NULL entries and feature name lists
        sample_ids <- sample_ids[!sapply(sample_ids, is.null)]
        sample_ids <- sample_ids[!grepl("_features$", names(sample_ids))]

    } else if (is.list(datasets) && all(sapply(datasets, is.character))) {
        # Handle file path input
        sample_ids <- lapply(datasets, function(file) {
            if (file.exists(file)) {
                df <- read.delim(file, nrows = 1)
                if ("sample_id" %in% colnames(df) || "Sample_ID" %in% colnames(df)) {
                    read.delim(file)[,1]
                } else {
                    NULL
                }
            } else {
                NULL
            }
        })
    } else {
        stop("Invalid input type for datasets")
    }

    # Perform consistency checks
    sample_ids <- sample_ids[!sapply(sample_ids, is.null)]
    if (length(sample_ids) == 0) {
        stop("No valid sample IDs found in any modality")
    }

    # Check for consistency across modalities
    reference_samples <- sample_ids[[1]]
    for (modality in names(sample_ids)[-1]) {
        if (!identical(sort(reference_samples), sort(sample_ids[[modality]]))) {
            stop(sprintf("Sample mismatch in %s modality for %s",
                        modality, cancer_type))
        }
    }

    return(TRUE)
}

# Function to safely calculate mean
  safe_mean <- function(x) {
    if (is.numeric(x)) {
      return(mean(x, na.rm = TRUE))
    } else if (is.list(x)) {
      return(x[[length(x)]])  # Take the last value for lists
    } else {
      return(NA)
    }
  }

  # Function to safely get class balance
  get_class_balance <- function(metrics) {
    if (is.null(metrics$class_balance)) return(NULL)
    if (is.list(metrics$class_balance)) {
      last_balance <- metrics$class_balance[[length(metrics$class_balance)]]
      if (is.list(last_balance)) {
        return(last_balance)
      }
    }
    return(NULL)
  }


#' Prepare validation data using model dimensions and selected features
#' @param model Trained model
#' @param validation_data Validation dataset
#' @param selected_features List of features by modality
#' @return Validation data with selected features
prepare_validation_features <- function(model, validation_data, selected_features) {
  cat("\nSelecting features for each modality:\n")
  selected_data <- validation_data
  
  for (modality in names(model$modality_dims)) {
    if (!is.null(validation_data$data[[modality]])) {
      current_features <- validation_data$features[[modality]]
      feature_indices <- match(selected_features[[modality]], current_features)
      
      if (any(is.na(feature_indices))) {
        missing_idx <- which(is.na(feature_indices))
        stop(sprintf("Missing %s features: %s", 
                    modality,
                    paste(selected_features[[modality]][missing_idx[1:min(5, length(missing_idx))]], 
                          collapse=", ")))
      }
      
      feature_indices <- c(1, feature_indices + 1)
      selected_data$data[[modality]] <- validation_data$data[[modality]][, feature_indices, drop = FALSE]
      
      mask_name <- paste0(modality, "_mask")
      if (!is.null(validation_data$data[[mask_name]])) {
        selected_data$data[[mask_name]] <- validation_data$data[[mask_name]][, feature_indices[-1], drop = FALSE]
      }
      
      cat(sprintf("%s: %d features\n", modality, length(selected_features[[modality]])))
    }
  }
  
  selected_data$features <- selected_features
  return(selected_data)
}


#' Prepare validation data using exact feature matching
#' @param model Trained model
#' @param validation_data Validation dataset
#' @param selected_features List of features by modality
#' @return Validation data with exactly matched features
prepare_validation_features <- function(model, validation_data, selected_features) {
  selected_data <- validation_data

  for (modality in names(selected_features)) {
    if (!is.null(validation_data$data[[modality]])) {
      feature_cols <- c("sample_id", selected_features[[modality]])
      available_cols <- colnames(validation_data$data[[modality]])
      missing_cols <- setdiff(feature_cols, available_cols)

      if (length(missing_cols) > 0) {
        stop(sprintf("Missing %s columns: %s",
                    modality,
                    paste(missing_cols[1:min(5, length(missing_cols))], collapse=", ")))
      }

      selected_data$data[[modality]] <- validation_data$data[[modality]][, feature_cols]
      selected_data$features[[modality]] <- selected_features[[modality]]

      cat(sprintf("%s: selected %d features\n", modality, length(selected_features[[modality]])))
    }
  }

  return(selected_data)
}


#' Create training validation split for outer fold
#' @param indices Vector of training indices
#' @param validation_split Proportion for validation (between 0 and 1)
#' @param stratify Optional stratification vector
#' @return List containing train and validation indices
create_train_val_split <- function(indices, validation_split, stratify = NULL) {
  n_total <- length(indices)
  n_val <- floor(n_total * validation_split)

  if (!is.null(stratify)) {
    # Get stratification values for these indices
    val_indices <- create_stratified_split(indices, stratify, n_val)
  } else {
    # Random sampling
    val_indices <- sample(indices, n_val)
  }

  train_indices <- setdiff(indices, val_indices)

  list(
    train = train_indices,
    validation = val_indices
  )
}
