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



#' Train model with multi-modal data and feature selection
#' @param model Neural network model
#' @param train_data Training data
#' @param val_data Validation data
#' @param config Training configuration
#' @param outcome_type Either "binary" or "survival"
#' @return Trained model, training history, and selected features

train_model <- function(model, train_data, config, outcome_type = "binary", outcome_var = NULL, selected_features = NULL) {
  if (is.null(selected_features)) {
    message("Performing feature selection on training data...")
    selected_features <- select_multimodal_features(
      train_data$data,
      n_features = config$model$architecture$modality_dims,
      outcome_info = list(
        type = outcome_type,
        var = outcome_var
      )
    )
   }
    train_data_selected <- apply_feature_selection(train_data, selected_features)
  
    train_loader <- dataloader(
        dataset = train_data_selected,
        batch_size = config$model$batch_size,
        shuffle = TRUE,
        collate_fn = custom_collate
    )

    optimizer <- optim_adam(
        model$parameters,
        lr = config$model$optimizer$lr,
        weight_decay = config$model$optimizer$weight_decay
    )

    training_history <- list()
    best_loss <- Inf
    patience_counter <- 0
    best_model_state <- model$state_dict()


    # Main training loop
    for (epoch in 1:config$model$max_epochs) {
        model$train()
        train_losses <- c()
        train_metrics <- list()

        # Training loop
        coro::loop(for (batch in train_loader) {
            optimizer$zero_grad()
            output <- model(batch$data)
            target <- batch$outcomes$binary$unsqueeze(2)
            loss <- compute_bce_loss(output$predictions, target)
            loss$backward()
            optimizer$step()
            train_losses <- c(train_losses, loss$item())
            batch_metrics <- calculate_binary_metrics(output$predictions, target)
            for (metric_name in names(batch_metrics)) {
                if (is.null(train_metrics[[metric_name]])) train_metrics[[metric_name]] <- c()
                train_metrics[[metric_name]] <- c(train_metrics[[metric_name]], batch_metrics[[metric_name]])
            }
        })

        if (length(train_losses) > 0) {
            current_loss <- mean(train_losses)
            epoch_metrics <- list(
                train_loss = current_loss
            )

            # Add other metrics
            for (metric_name in names(train_metrics)) {
                epoch_metrics[[paste0("train_", metric_name)]] <- safe_mean(train_metrics[[metric_name]])
            }

            training_history[[epoch]] <- epoch_metrics

            # Early stopping check
            if (current_loss < (best_loss - config$model$early_stopping$min_delta)) {
                best_loss <- current_loss
                best_model_state <- model$state_dict()
                patience_counter <- 0
            } else {
                patience_counter <- patience_counter + 1
            }

            if (epoch %% config$model$print_every == 0) {
                message(sprintf("Epoch %d/%d", epoch, config$model$max_epochs))
                cat(sprintf("Train - Loss: %.4f, Acc: %.4f, F1: %.4f, AUC: %.4f, Bal Acc: %.4f\n",
                    epoch_metrics$train_loss,
                    epoch_metrics$train_accuracy,
                    epoch_metrics$train_f1,
		    epoch_metrics$train_auc,
		    epoch_metrics$train_balanced_accuracy))
                cat(sprintf("Patience counter: %d/%d\n",
                    patience_counter,
                    config$model$early_stopping$patience))
            }

            # Check if we should stop
            if (patience_counter >= config$model$early_stopping$patience) {
                message(sprintf("\nEarly stopping triggered at epoch %d", epoch))
                break
            }
        }
    }

    # Load best model state
    model$load_state_dict(best_model_state)

    list(
        model = model,
        history = training_history,
        best_loss = best_loss,
        selected_features = selected_features,
    	config = config
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
    #cat("1. Starting dataset subsetting\n")
    
    if (!inherits(datasets, "MultiModalDataset")) {
        stop("datasets must be a MultiModalDataset object")
    }
    
    #cat("2. Creating new data list\n")
    subsetted_data <- list()
    
    #cat("3. Processing data and masks for each modality\n")
    modalities <- c("clinical", "cnv", "expression", "mutations", "methylation", "mirna")
    for (modality in modalities) {
        if (!is.null(datasets$data[[modality]])) {
            #cat(sprintf("   - Processing %s\n", modality))
            
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
    
    #cat("4. Processing outcomes if present\n")
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
    
    #cat("5. Creating new dataset\n")
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
                    #data_list[[modality]][is.na(data_list[[modality]])] <- 0
		    # With this safer approach:
		   if (inherits(data_list[[modality]], "torch_tensor")) {
  		   # Use torch's isnan function instead of R's is.na
  		data_list[[modality]] <- torch_where(
    			torch_isnan(data_list[[modality]]),
    			torch_zeros_like(data_list[[modality]]),
    			data_list[[modality]]
 	 		)
		   }
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

#' Initialize parallel processing environment
#' @param max_workers Maximum number of parallel workers
#' @return List containing worker configuration

setup_parallel_env <- function(max_workers) {
    gc()
    available_memory <- as.numeric(system("free -g | awk 'NR==2 {print $4}'", intern=TRUE))
    memory_per_worker <- floor(available_memory / (max_workers + 1))
    actual_workers <- max(1, min(max_workers, floor(available_memory / 2)))

    plan(multisession,
         workers = actual_workers,
         future.seed = TRUE)  # Add this line

    list(
        available_memory = available_memory,
        memory_per_worker = memory_per_worker,
        actual_workers = actual_workers
    )
}


#' Process inner cross-validation folds
#' @param inner_folds List of inner fold indices
#' @param model Base model to train
#' @param datasets Complete dataset
#' @param config Model configuration
#' @param params Additional parameters
#' @return List of inner fold results

process_inner_folds <- function(inner_folds, model, datasets, config, params) {
    lapply(seq_along(inner_folds), function(inner_idx) {
        message(sprintf("Processing inner fold: %d", inner_idx))
        inner_model <- model$create_copy()
        
        # Create datasets for inner fold
        inner_train_data <- subset_datasets(datasets, inner_folds[[inner_idx]]$train_idx, params$batch_size)
        inner_val_data <- subset_datasets(datasets, inner_folds[[inner_idx]]$test_idx, params$batch_size)
        
        # Train model
        trained_model <- train_model(
            model = inner_model,
            train_data = inner_train_data,
            config = config,
            outcome_type = params$outcome_type,
            outcome_var = params$outcome_var
        )
        
        # Evaluate model
        val_data_selected <- prepare_validation_features(
            trained_model$model, 
            inner_val_data, 
            trained_model$selected_features
        )
        val_results <- evaluate_model(trained_model$model, val_data_selected, params$outcome_type)

        # Cleanup
        rm(inner_model, inner_train_data, inner_val_data)
        gc()
        
        list(
            state_dict = trained_model$model$state_dict(),
            val_metrics = val_results$metrics,
            selected_features = trained_model$selected_features,
            best_loss = trained_model$best_loss
        )
    })
}

#' Process outer cross-validation fold
#' @param fold_idx Outer fold index
#' @param fold_data Outer fold data
#' @param model Base model
#' @param datasets Complete dataset
#' @param config Model configuration
#' @param params Additional parameters
#' @return List of outer fold results
process_outer_fold <- function(outer_fold_idx, fold_data, model, datasets, config, params) {
    message(sprintf("Processing outer fold: %d", outer_fold_idx))
    
    # Process inner folds
    inner_results <- process_inner_folds_with_hpo(
        fold_data$inner_folds, 
        model, 
        datasets, 
        config, 
        params
    )
    
    # Find best inner model
    best_inner_idx <- which.min(sapply(inner_results, function(x) x$best_loss))
    best_config <- inner_results[[best_inner_idx]]
    
    # Train final model
    outer_train_data <- subset_datasets(datasets, fold_data$train_idx, params$batch_size)
    outer_val_data <- subset_datasets(datasets, fold_data$test_idx, params$batch_size)
    
    outer_model <- model$create_copy()
    outer_model$load_state_dict(best_config$state_dict)
    
    final_model <- train_model(
        model = outer_model,
        train_data = outer_train_data,
        config = config,
        outcome_type = params$outcome_type,
        outcome_var = params$outcome_var,
        selected_features = best_config$selected_features
    )
    # Evaluate final model
    outer_data_selected <- prepare_validation_features(
        final_model$model, 
        outer_val_data,
        final_model$selected_features
    )

    test_results <- evaluate_model(final_model$model, outer_data_selected, params$outcome_type)
    
    # Cleanup
    rm(outer_model, outer_train_data, outer_val_data)
    gc()
    
    list(
        model = final_model$model,
        state_dict = final_model$model$state_dict(),
        val_metrics = test_results$metrics,
        selected_features = final_model$selected_features,
	best_loss = final_model$best_loss,
    	best_config = best_config$config
    )
}

#' Run nested cross-validation
#' @inheritParams [previous params remain the same]
run_nested_cv <- function(model, datasets, config, cancer_type, 
                         outcome_type = "binary",
                         validation_pct = 0.3, test_pct = 0.3, 
                         seed = NULL, max_workers = 2, batch_size = 32, 
                         outcome_var = NULL) {
    
    # Setup parallel environment
    parallel_config <- setup_parallel_env(max_workers)
    message(sprintf("\nParallel processing configuration:"))
    message(sprintf("- Available memory: %d GB", parallel_config$available_memory))
    message(sprintf("- Memory per worker: %d GB", parallel_config$memory_per_worker))
    message(sprintf("- Using %d workers", parallel_config$actual_workers))
    
    # Initialize cross-validation
    message("\nInitializing cross-validation:")
    message(sprintf("- Total samples: %d", datasets$n_samples))
    message(sprintf("- Modalities: %s", paste(names(datasets$features), collapse=", ")))
    
    # Create and validate CV splits
    stratify <- as.numeric(get_stratification_vector(datasets, outcome_type, outcome_var))
    cv_splits <- create_cv_splits(
        n_samples = datasets$n_samples,
        n_repeats = config$cv_params$outer_repeats,
        n_outer_folds = config$cv_params$outer_folds,
        n_inner_folds = config$cv_params$inner_folds,
        validation_pct = validation_pct,
        test_pct = test_pct,
        stratify = stratify,
        seed = seed
    )
    
    # Process repeats
    results <- vector("list", length(cv_splits))
    params <- list(
        outcome_type = outcome_type,
        outcome_var = outcome_var,
        batch_size = batch_size
    )
    
    for (repeat_idx in seq_along(cv_splits)) {
        message(sprintf("\nProcessing repeat %d/%d", repeat_idx, length(cv_splits)))
        
        # Process outer folds in parallel
        outer_results <- future_lapply(
            seq_along(cv_splits[[repeat_idx]]$outer_splits),
            function(outer_fold_idx) {
                process_outer_fold(
                    outer_fold_idx,
                    cv_splits[[repeat_idx]]$outer_splits[[outer_fold_idx]],
                    model,
                    datasets,
                    config,
                    params
                )
            }
        )
        
        # Process validation set
        best_outer_idx <- which.min(sapply(outer_results, function(x) x$best_loss))
        best_outer_model <- outer_results[[best_outer_idx]]
        
        validation_data <- subset_datasets(datasets, cv_splits[[repeat_idx]]$validation, batch_size)
        
	# Evaluate best outer model
    	
	validation_data_selected <- prepare_validation_features(
        best_outer_model$model,
        validation_data,
        best_outer_model$selected_features
    	)

	validation_results <- evaluate_model(best_outer_model$model, validation_data_selected, outcome_type)

        # Store results
        results[[repeat_idx]] <- list(
            repeat_idx = repeat_idx,
            outer_results = outer_results,
            validation_results = validation_results,
            best_model = best_outer_model$model,
            features = best_outer_model$selected_features,
            best_loss = best_outer_model$best_loss,
            validation_data = validation_data_selected
	)
                
    }

    # Aggregate and summarize results
    final_results <- aggregate_cv_results(results)
    print_cv_results(final_results)

	
    # Select best model
    best_repeat_idx <- which.min(sapply(results, function(r) r$best_loss))
    best_model_results <- list(
        results = final_results,
        model = results[[best_repeat_idx]]$best_model,
        cv_splits = cv_splits,
        features = results[[best_repeat_idx]]$features,
        validation_results = results[[best_repeat_idx]]$validation_results,
        best_loss = results[[best_repeat_idx]]$best_loss,
    	validation_data = results[[best_repeat_idx]]$validation_data
    )
    
    # Create results directory if it doesn't exist
    #results_dir <- file.path(config$experiment$output_dir, cancer_type)
    #if (!dir.exists(results_dir)) {
    #    dir.create(results_dir, recursive = TRUE)
    #    message(sprintf("Created results directory: %s", results_dir))
    #}
    
    # Save only the best model results
    #saveRDS(
    #    best_model_results,
    #    file.path(results_dir, "best_model_results.rds")
    #)
    #message(sprintf("Saved best model results to: %s", 
    #               file.path(results_dir, "best_model_results.rds")))
    
    return(best_model_results)
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
analyze_feature_importance <- function(model, selected_features, top_n = 20) {
  importance_by_modality <- list()
  # Extract weights from first layer of each encoder
  for (modality in names(model$modality_dims)) {
    # Get first layer weights
    weights <- model$encoders$modules[[modality]]$layers[[1]][[1]]$weight$cpu()
    weights_matrix <- as.matrix(weights)
    # Calculate importance scores (mean absolute weight)
    importance_scores <- colMeans(abs(weights_matrix))
    # Match with feature names
    features <- selected_features[[modality]]
    if (length(features) == length(importance_scores)) {
      importance_df <- data.frame(
        feature = features,
        importance = importance_scores,
        modality = modality
      )
      # Sort by importance
      importance_df <- importance_df[order(-importance_df$importance), ]
      importance_by_modality[[modality]] <- importance_df
    }
  }
  # Print top features for each modality
  for (modality in names(importance_by_modality)) {
    cat(sprintf("\nTop %d %s features:\n", top_n, modality))
    df <- head(importance_by_modality[[modality]], top_n)
    print(df[, c("feature", "importance")])
  }
  return(importance_by_modality)
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


