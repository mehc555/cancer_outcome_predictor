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
get_stratification_vector <- function(datasets, outcome_type = "binary") {
    # Handle both MultiModalDataset and list inputs
    if (inherits(datasets, "MultiModalDataset")) {
        clinical_data <- as.matrix(datasets$data$clinical$cpu())
        clinical_features <- datasets$data$clinical_features
    } else {
        clinical_data <- as.matrix(datasets$clinical$cpu())
        clinical_features <- datasets$clinical_features
    }
    
    if (outcome_type == "binary") {
        event_col <- which(clinical_features == "demographics_vital_status_alive")
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


#' Train model with multi-modal data
#' @param model Neural network model
#' @param train_data Training data
#' @param val_data Validation data
#' @param config Training configuration
#' @param outcome_type Either "binary" or "survival"
#' @return Trained model and training history
train_model <- function(model, train_data, val_data, config, outcome_type = "binary") {
  # Create data loaders
  train_loader <- dataloader(
    dataset = train_data,
    batch_size = config$model$batch_size,
    shuffle = TRUE
  )

  val_loader <- dataloader(
    dataset = val_data,
    batch_size = config$model$batch_size,
    shuffle = FALSE
  )

  # Initialize optimizer
  optimizer <- optim_adam(
    model$parameters,
    lr = config$model$optimizer$lr,
    weight_decay = config$model$optimizer$weight_decay
  )

  # Initialize scheduler
  scheduler <- lr_reduce_on_plateau(
    optimizer,
    patience = config$model$scheduler$patience,
    factor = config$model$scheduler$factor
  )

  # Initialize tracking variables
  best_val_loss <- Inf
  patience_counter <- 0
  best_model_state <- model$state_dict()
  training_history <- list()

  # Training loop
  for (epoch in 1:config$model$max_epochs) {
    # Training phase
    model$train()
    train_losses <- c()
    train_metrics <- list()

    coro::loop(for (batch in train_loader) {
      optimizer$zero_grad()

      # Forward pass
      output <- model(batch$data, batch$masks)
      
      # Compute loss based on outcome type
      if (outcome_type == "binary") {
        loss <- nn_bce_with_logits_loss()(output$predictions, batch$target)
      } else if (outcome_type == "survival") {
        # Cox partial likelihood loss
        loss <- compute_cox_loss(output$predictions, batch$time, batch$event)
      }

      # Backward pass and optimization
      loss$backward()
      optimizer$step()

      # Record loss
      train_losses <- c(train_losses, loss$item())

      # Calculate additional metrics
      if (outcome_type == "binary") {
        batch_metrics <- calculate_binary_metrics(output$predictions, batch$target)
      } else {
        batch_metrics <- calculate_survival_metrics(output$predictions, batch$time, batch$event)
      }
      
      # Accumulate metrics
      for (metric_name in names(batch_metrics)) {
        if (is.null(train_metrics[[metric_name]])) {
          train_metrics[[metric_name]] <- c()
        }
        train_metrics[[metric_name]] <- c(train_metrics[[metric_name]], batch_metrics[[metric_name]])
      }
    })

    # Validation phase
    model$eval()
    val_losses <- c()
    val_metrics <- list()

    with_no_grad({
      coro::loop(for (batch in val_loader) {
        output <- model(batch$data, batch$masks)
        
        # Compute validation loss
        if (outcome_type == "binary") {
          loss <- nn_bce_with_logits_loss()(output$predictions, batch$target)
        } else if (outcome_type == "survival") {
          loss <- compute_cox_loss(output$predictions, batch$time, batch$event)
        }
        
        val_losses <- c(val_losses, loss$item())

        # Calculate validation metrics
        if (outcome_type == "binary") {
          batch_metrics <- calculate_binary_metrics(output$predictions, batch$target)
        } else {
          batch_metrics <- calculate_survival_metrics(output$predictions, batch$time, batch$event)
        }
        
        # Accumulate metrics
        for (metric_name in names(batch_metrics)) {
          if (is.null(val_metrics[[metric_name]])) {
            val_metrics[[metric_name]] <- c()
          }
          val_metrics[[metric_name]] <- c(val_metrics[[metric_name]], batch_metrics[[metric_name]])
        }
      })
    })

    # Calculate epoch metrics
    epoch_metrics <- list(
      train_loss = mean(train_losses),
      val_loss = mean(val_losses)
    )
    
    # Add mean of accumulated metrics
    for (metric_name in names(train_metrics)) {
      epoch_metrics[[paste0("train_", metric_name)]] <- mean(train_metrics[[metric_name]])
      epoch_metrics[[paste0("val_", metric_name)]] <- mean(val_metrics[[metric_name]])
    }

    # Store training history
    training_history[[epoch]] <- epoch_metrics

    # Update scheduler
    scheduler$step(epoch_metrics$val_loss)

    # Early stopping check
    if (epoch_metrics$val_loss < best_val_loss - config$model$early_stopping$min_delta) {
      best_val_loss <- epoch_metrics$val_loss
      best_model_state <- model$state_dict()
      patience_counter <- 0
    } else {
      patience_counter <- patience_counter + 1
    }

    # Print progress
    if (epoch %% config$model$print_every == 0) {
      message(sprintf("Epoch %d/%d - Train Loss: %.4f - Val Loss: %.4f", 
                     epoch, config$model$max_epochs, 
                     epoch_metrics$train_loss, epoch_metrics$val_loss))
    }

    # Early stopping
    if (patience_counter >= config$model$early_stopping$patience) {
      message(sprintf("Early stopping triggered at epoch %d", epoch))
      break
    }
  }

  # Load best model state
  model$load_state_dict(best_model_state)

  return(list(
    model = model,
    history = training_history,
    best_val_loss = best_val_loss
  ))
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

#' Calculate metrics for binary classification
#' @param predictions Model predictions
#' @param targets True labels
#' @return List of metrics
calculate_binary_metrics <- function(predictions, targets) {
  # Convert predictions to probabilities
  probs <- torch_sigmoid(predictions)
  preds <- (probs > 0.5)$to(dtype = torch_long())
  
  # Convert to CPU for calculations
  preds <- preds$cpu()
  targets <- targets$cpu()
  
  # Calculate metrics
  tp <- torch_sum(preds * targets)$item()
  fp <- torch_sum(preds * (1 - targets))$item()
  fn <- torch_sum((1 - preds) * targets)$item()
  tn <- torch_sum((1 - preds) * (1 - targets))$item()
  
  # Calculate various metrics
  accuracy <- (tp + tn) / (tp + fp + fn + tn)
  precision <- if (tp + fp > 0) tp / (tp + fp) else 0
  recall <- if (tp + fn > 0) tp / (tp + fn) else 0
  f1 <- if (precision + recall > 0) 2 * (precision * recall) / (precision + recall) else 0
  
  # Calculate AUC-ROC if possible
  auc <- tryCatch({
    probs_cpu <- as.numeric(probs$cpu())
    targets_cpu <- as.numeric(targets$cpu())
    pROC::auc(pROC::roc(targets_cpu, probs_cpu, quiet = TRUE))
  }, error = function(e) NA)
  
  list(
    accuracy = accuracy,
    precision = precision,
    recall = recall,
    f1 = f1,
    auc = auc
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
        test_indices <- if (!is.null(stratify)) {
            create_stratified_split(remaining_indices, stratify[remaining_indices], test_size)
        } else {
            sample(remaining_indices, test_size)
        }
        
        train_indices <- setdiff(remaining_indices, test_indices)
        
        # Create outer folds - each uses full training set
        outer_folds <- create_stratified_folds(
            train_indices = train_indices,
            stratify = if (!is.null(stratify)) stratify[train_indices] else NULL,
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
                test_idx = test_indices,  # Use same test set for all K-folds
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
            message(sprintf("Training set: %d samples", length(train_indices)))
            
            message("\n=== K-fold Structure ===")
            message(sprintf("Number of K-folds: %d", n_outer_folds))
            message(sprintf("Full training set (each K-fold): %d samples", 
                          length(train_indices)))
            message(sprintf("Fixed test set (all K-folds): %d samples", 
                          length(test_indices)))
            
            message("\n=== M-fold Structure (per K-fold) ===")
            inner_test_size <- floor(length(train_indices) * test_pct)
            inner_train_size <- length(train_indices) - inner_test_size
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
                                   "Test" = test_indices,
                                   "Training" = train_indices)
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
            strata <- unique(stratify[train_indices])
            test_idx <- c()
            
            for (stratum in strata) {
                stratum_indices <- train_indices[stratify[train_indices] == stratum]
                stratum_size <- floor(test_size * length(stratum_indices) / length(train_indices))
                test_idx <- c(test_idx, sample(stratum_indices, stratum_size))
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


#' Memory-efficient dataset subsetting
#' @param datasets List of datasets
#' @param indices Indices to subset
#' @param batch_size Batch size for data loading
#' @return Subsetted datasets

#' Memory-efficient dataset subsetting
#' @param datasets List of datasets
#' @param indices Indices to subset
#' @param batch_size Batch size for data loading
#' @return Subsetted torch dataset
subset_datasets <- function(datasets, indices, batch_size = 32) {
    if (is.null(datasets)) stop("Null datasets provided")
    if (length(indices) == 0) stop("Empty indices provided")
    if (!is.list(datasets)) stop("Datasets must be a list")
    
    # Create a new list with the same names
    result <- vector("list", length(datasets))
    names(result) <- names(datasets)
    
    # Process each modality
    for (modality in names(datasets)) {
        data <- datasets[[modality]]
        if (is.null(data)) {
            result[[modality]] <- NULL
            next
        }
        
        # Check if it's a feature names list
        if (grepl("_features$", modality)) {
            result[[modality]] <- data  # Keep all feature names
            next
        }
        
        # For data and mask modalities
        if (inherits(data, "torch_tensor")) {
            # Convert indices to torch tensor for indexing
            torch_indices <- torch_tensor(indices, dtype=torch_long())
            
            # Subset the tensor
            result[[modality]] <- data[torch_indices,]
        } else {
            # For other data types (matrices, data frames)
            result[[modality]] <- data[indices,]
        }
    }
    
    # Create a MultiModalDataset from the subsetted data
    dataset <- MultiModalDataset(result)
    
    return(dataset)
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
                         validation_pct = 0.1, test_pct = 0.2, 
                         seed = NULL, max_workers = 2, batch_size = 32) {
    
    # Clear memory and set up parallel processing
    gc()
    options(future.globals.maxSize = 2000 * 1024^2)
    
    # Configure workers based on available memory
    available_memory <- as.numeric(system("free -g | awk 'NR==2 {print $4}'", intern=TRUE))
    actual_workers <- max(1, min(max_workers, floor(available_memory / 1)))
    message(sprintf("Using %d workers based on available memory", actual_workers))
    
    # Get stratification vector
    stratify <- get_stratification_vector(datasets, outcome_type)
    
    # Get total number of samples
    n_samples <- datasets$n_samples
    
    # Create CV splits
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
 
    # Validate splits
    validation_results <- validate_cv_splits(cv_splits)
    
    if (validation_results$summary$total_overlaps > 0) {
        warning(sprintf("Found %d overlapping indices across %d checks (%.2f%%). Maximum overlap: %.2f%%",
                       validation_results$summary$total_overlaps,
                       validation_results$summary$total_checks,
                       validation_results$summary$overlap_percentage,
                       validation_results$summary$max_overlap_percentage))
    }

    # Set up parallel processing
    plan(multisession, workers = actual_workers)
    
    # Process each repeat
    results <- vector("list", length(cv_splits))
    
    for (repeat_idx in seq_along(cv_splits)) {
        message(sprintf("Processing repeat %d/%d", repeat_idx, length(cv_splits)))
        
        repeat_split <- cv_splits[[repeat_idx]]
        
        # Process outer folds in parallel
        outer_results <- future_lapply(seq_along(repeat_split$outer_splits), 
                                     function(fold_idx) {
            model_copy <- model$create_copy()
            outer_split <- repeat_split$outer_splits[[fold_idx]]
            
            # Process inner folds
            inner_results <- lapply(outer_split$inner_folds, function(inner_fold) {
                inner_model <- model$create_copy()
                
                # Create datasets
                inner_train_data <- subset_datasets(datasets, inner_fold$train_idx, batch_size)
                inner_val_data <- subset_datasets(datasets, inner_fold$test_idx, batch_size)
                
                # Train and evaluate
                trained_model <- train_model(
                    model = inner_model,
                    train_data = inner_train_data,
                    val_data = inner_val_data,
                    config = config,
                    outcome_type = outcome_type
                )
                
                # Clean up
                rm(inner_model, inner_train_data, inner_val_data)
                gc()
                
                trained_model
            })
            
            # Train final model for outer fold
            outer_train_data <- subset_datasets(datasets, outer_split$train_idx, batch_size)
            outer_test_data <- subset_datasets(datasets, outer_split$test_idx, batch_size)
            
            final_model <- train_model(
                model = model_copy,
                train_data = outer_train_data,
                val_data = outer_test_data,
                config = config,
                outcome_type = outcome_type
            )
            
            # Save intermediate results
            results_dir <- file.path(config$main$paths$results_dir, cancer_type)
            saveRDS(
                final_model,
                file.path(results_dir, sprintf("repeat_%d_fold_%d_model.rds", 
                                             repeat_idx, fold_idx))
            )
            
            # Clean up
            rm(model_copy, outer_train_data, outer_test_data)
            gc()
            
            final_model
        })
        
        # Evaluate validation set
        validation_data <- subset_datasets(datasets, repeat_split$validation, batch_size)
        best_model <- select_best_model(outer_results)
        validation_results <- evaluate_model(best_model, validation_data, outcome_type)
        
        # Store results
        results[[repeat_idx]] <- list(
            repeat_idx = repeat_idx,
            outer_results = outer_results,
            validation_results = validation_results,
            best_model = best_model
        )
        
        # Save results
        saveRDS(
            results[[repeat_idx]],
            file.path(config$main$paths$results_dir, cancer_type,
                     sprintf("repeat_%d_results.rds", repeat_idx))
        )
    }
    
    # Aggregate results
    final_results <- aggregate_cv_results(results)
    final_model <- select_final_model(results)
    
    list(
        results = final_results,
        final_model = final_model,
        cv_splits = cv_splits,
        raw_results = results,
        split_validation = validation_results
    )

}

#' Select best model based on validation performance
#' @param models List of trained models
#' @return Best performing model
select_best_model <- function(models) {
    performances <- sapply(models, function(m) m$best_val_loss)
    best_idx <- which.min(performances)
    models[[best_idx]]$model
}

#' Select final model from all repeats
#' @param results List of results from all repeats
#' @return Best performing model
select_final_model <- function(results) {
    performances <- sapply(results, function(r) r$validation_results$metrics$primary_metric)
    best_repeat <- which.max(performances)
    results[[best_repeat]]$best_model
}

#' Aggregate CV results
#' @param results List of results from all repeats
#' @return Aggregated results summary
aggregate_cv_results <- function(results) {
    # Extract metrics
    validation_metrics <- lapply(results, function(r) r$validation_results$metrics)
    outer_metrics <- lapply(results, function(r) {
        lapply(r$outer_results, function(o) o$history[[length(o$history)]])
    })
    
    # Calculate summary statistics
    validation_summary <- list(
        mean = colMeans(do.call(rbind, validation_metrics)),
        sd = apply(do.call(rbind, validation_metrics), 2, sd)
    )
    
    outer_summary <- list(
        mean = colMeans(do.call(rbind, unlist(outer_metrics, recursive = FALSE))),
        sd = apply(do.call(rbind, unlist(outer_metrics, recursive = FALSE)), 2, sd)
    )
    
    list(
        validation_summary = validation_summary,
        outer_summary = outer_summary
    )
}

#' Evaluate model performance
#' @param model Trained model
#' @param data Test data (MultiModalDataset)
#' @param outcome_type Either "binary" or "survival"
#' @return List of evaluation metrics and predictions
evaluate_model <- function(model, data, outcome_type = "binary") {
    model$eval()
    loader <- dataloader(dataset = data, batch_size = 32, shuffle = FALSE)
    
    predictions <- list()
    targets <- list()
    times <- list()
    events <- list()
    attention_weights <- list()
    
    with_no_grad({
        coro::loop(for (batch in loader) {
            output <- model(batch$data, batch$masks)
            predictions[[length(predictions) + 1]] <- output$predictions$cpu()
            
            if (outcome_type == "binary") {
                targets[[length(targets) + 1]] <- batch$target$cpu()
            } else {
                times[[length(times) + 1]] <- batch$time$cpu()
                events[[length(events) + 1]] <- batch$event$cpu()
            }
            
            if (!is.null(output$attention_weights)) {
                attention_weights[[length(attention_weights) + 1]] <- output$attention_weights
            }
        })
    })
    
    # Concatenate results
    all_predictions <- torch_cat(predictions, dim = 1)
    
    if (outcome_type == "binary") {
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
    
    # Add attention weights if available
    if (length(attention_weights) > 0) {
        results$attention_weights <- attention_weights
    }
    
    results
}

#' Analyze feature importance using integrated gradients
#' @param model Trained model
#' @param data Reference data (MultiModalDataset)
#' @param n_steps Number of steps for path integral
#' @return Feature importance scores
analyze_feature_importance <- function(model, data, n_steps = 50) {
    model$eval()
    
    # Handle MultiModalDataset input
    if (inherits(data, "MultiModalDataset")) {
        baseline <- lapply(data$data, function(x) {
            if (inherits(x, "torch_tensor")) {
                torch_zeros_like(x)
            } else {
                NULL
            }
        })
        
        # Remove feature name lists from baseline
        baseline <- baseline[!grepl("_features$", names(baseline))]
    } else {
        baseline <- lapply(data, function(x) {
            if (inherits(x, "torch_tensor")) {
                torch_zeros_like(x)
            } else {
                NULL
            }
        })
    }
    
    # Calculate integrated gradients
    importance_scores <- list()
    
    with_no_grad({
        for (modality in names(baseline)) {
            if (!is.null(baseline[[modality]])) {
                # Create path points
                alphas <- seq(0, 1, length.out = n_steps)
                gradients <- torch_zeros_like(data$data[[modality]])
                
                for (alpha in alphas) {
                    # Interpolate between baseline and input
                    interp_input <- baseline[[modality]] + 
                        alpha * (data$data[[modality]] - baseline[[modality]])
                    interp_input$requires_grad_(TRUE)
                    
                    # Forward pass
                    output <- model(interp_input)
                    
                    # Calculate gradients
                    grad <- torch_autograd_grad(
                        output$predictions,
                        interp_input,
                        torch_ones_like(output$predictions)
                    )[[1]]
                    
                    gradients <- gradients + grad
                }
                
                # Calculate importance scores
                importance <- (data$data[[modality]] - baseline[[modality]]) * 
                    gradients / n_steps
                
                importance_scores[[modality]] <- importance$abs()$mean(dim = 1)
            }
        }
    })
    
    importance_scores
}

#' Analyze attention patterns
#' @param attention_weights List of attention weights
#' @param feature_names List of feature names by modality (from MultiModalDataset)
#' @return Analysis of attention patterns
analyze_attention_patterns <- function(attention_weights, dataset) {
    # Extract feature names from dataset
    if (inherits(dataset, "MultiModalDataset")) {
        feature_names <- lapply(names(dataset$data), function(name) {
            if (grepl("_features$", name)) {
                dataset$data[[name]]
            }
        })
        names(feature_names) <- gsub("_features$", "", names(feature_names))
    } else {
        feature_names <- dataset  # Assume direct feature names passed
    }
    
   if (!is.null(attention_weights$self_attention)) {
        attention_analysis$self_attention <- lapply(names(attention_weights$self_attention), 
                                                  function(modality) {
            weights <- attention_weights$self_attention[[modality]]
            
            # Average attention weights across heads and batches
            avg_weights <- torch_mean(weights, dim = c(1, 2))
            
            # Get top attended features
            top <- torch_topk(avg_weights, k = 10)
            top_values <- top[[1]]  # First element contains values
            top_indices <- top[[2]]  # Second element contains indices
            
            # Match with feature names
            top_features <- feature_names[[modality]][top_indices$cpu()$numpy()]
            
            list(
                modality = modality,
                top_features = top_features,
                attention_scores = top_values$cpu()$numpy()
            )
        })
    }
    
    # Analyze cross-attention patterns
    if (!is.null(attention_weights$cross_attention)) {
        # Average across heads and batches
        avg_cross_attention <- torch_mean(attention_weights$cross_attention, 
                                        dim = c(1, 2))
        
        # Get top cross-modal interactions
        top <- torch_topk(avg_cross_attention$view(-1), k = 10)
        top_values <- top[[1]]
        top_indices <- top[[2]]
        
        # Convert linear indices to 2D indices
        rows <- top_indices %/% avg_cross_attention$size(2)
        cols <- top_indices %% avg_cross_attention$size(2)
        
        # Match with modality names
        modality_names <- names(feature_names)
        cross_modal_interactions <- lapply(1:length(top_values), function(i) {
            list(
                from_modality = modality_names[rows[i] + 1],
                to_modality = modality_names[cols[i] + 1],
                attention_score = top_values[i]$cpu()$numpy()
            )
        })
        
        attention_analysis$cross_attention <- cross_modal_interactions
    }
    
    attention_analysis
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
    # Create output directory
    dir.create(file.path(output_dir, cancer_type), recursive = TRUE, 
              showWarnings = FALSE)
    
    # Save model performance metrics
    saveRDS(results$results, 
            file.path(output_dir, cancer_type, "performance_metrics.rds"))
    
    # Save feature importance analysis
    if (!is.null(results$feature_importance)) {
        saveRDS(results$feature_importance,
                file.path(output_dir, cancer_type, "feature_importance.rds"))
    }
    
    # Save attention analysis
    if (!is.null(results$attention_analysis)) {
        saveRDS(results$attention_analysis,
                file.path(output_dir, cancer_type, "attention_analysis.rds"))
    }
    
    # Save visualizations
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
    
    # Save configuration
    saveRDS(config, file.path(output_dir, cancer_type, "config.rds"))
    
    # Generate and save summary report
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
