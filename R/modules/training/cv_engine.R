# R/modules/training/cv_engine.R

library(future)
library(future.apply)
library(progressr)

#' Extract stratification variable from datasets
#' @param datasets List of torch datasets
#' @return Vector for stratification
get_stratification_vector <- function(datasets) {
  if (!is.null(datasets$clinical)) {
    # Extract the event information from clinical data
    clinical_data <- as.matrix(datasets$clinical)
    # Assuming the event column is present
    event_col <- which(colnames(datasets$clinical_features) == "demographics_vital_status_alive")
    if (length(event_col) > 0) {
      return(clinical_data[, event_col])
    }
  }
  return(NULL)
}

#' Train model with multi-modal data
#' @param model Neural network model
#' @param train_data Training data
#' @param val_data Validation data
#' @param config Training configuration
#' @return Trained model and training history
train_model <- function(model, train_data, val_data, config) {
  

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

  # Training loop
  best_val_loss <- Inf
  patience_counter <- 0

  for (epoch in 1:config$model$max_epochs) {
    # Training phase
    model$train()
    train_losses <- c()

    coro::loop(for (batch in train_loader) {
      optimizer$zero_grad()

      # Forward pass
      output <- model(batch$data, batch$masks)
      loss <- compute_loss(output$predictions, batch$target)

      # Backward pass
      loss$backward()
      optimizer$step()

      train_losses <- c(train_losses, loss$item())
    })

    # Validation phase
    model$eval()
    val_losses <- c()

    with_no_grad({
      coro::loop(for (batch in val_loader) {
        output <- model(batch$data, batch$masks)
        loss <- compute_loss(output$predictions, batch$target)
        val_losses <- c(val_losses, loss$item())
      })
    })

    # Calculate average losses
    avg_train_loss <- mean(train_losses)
    avg_val_loss <- mean(val_losses)

    # Update scheduler
    scheduler$step(avg_val_loss)

    # Early stopping check
    if (avg_val_loss < best_val_loss - config$model$early_stopping$min_delta) {
      best_val_loss <- avg_val_loss
      patience_counter <- 0
    } else {
      patience_counter <- patience_counter + 1
    }

    if (patience_counter >= config$model$early_stopping$patience) {
      break
    }
  }

  return(model)
}


#' Validate CV split indices
#' @param split_indices List of indices to validate
#' @param n_samples Total number of samples
#' @param name Name of the split for error messages
validate_indices <- function(split_indices, n_samples, name = "") {
    # Check if indices are numeric
    if (!is.numeric(split_indices)) {
        stop(sprintf("Invalid %s indices: must be numeric", name))
    }
    
    # Check for NA values
    if (any(is.na(split_indices))) {
        stop(sprintf("NA values found in %s indices", name))
    }
    
    # Check for duplicates
    if (any(duplicated(split_indices))) {
        stop(sprintf("Duplicate indices found in %s", name))
    }
    
    # Check range
    if (any(split_indices < 1) || any(split_indices > n_samples)) {
        stop(sprintf("%s indices out of range [1, %d]", name, n_samples))
    }
}

#' Check for overlap between sets of indices
#' @param set1 First set of indices
#' @param set2 Second set of indices
#' @param set1_name Name of first set for error messages
#' @param set2_name Name of second set for error messages
check_overlap <- function(set1, set2, set1_name, set2_name) {
    overlap <- intersect(set1, set2)
    if (length(overlap) > 0) {
        stop(sprintf("Overlap found between %s and %s: %s", 
                    set1_name, set2_name, 
                    paste(overlap, collapse=", ")))
    }
}

#' Create data splits for nested cross-validation with random sampling
#' @param n_samples Total number of samples
#' @param n_repeats Number of repetitions
#' @param n_outer_folds Number of outer folds 
#' @param n_inner_folds Number of inner folds
#' @param validation_pct Percentage of data for validation (0-1)
#' @param test_pct Percentage of remaining data for testing (0-1) 
#' @param stratify Optional vector for stratified sampling
#' @param seed Optional seed for reproducibility
#' @return List of CV split indices
create_cv_splits <- function(n_samples, n_repeats, n_outer_folds, n_inner_folds, 
                           validation_pct = 0.1, test_pct = 0.2, stratify = NULL, seed = NULL) {
    # Set seed if provided
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
    if (validation_pct + test_pct >= 1) stop("Sum of validation_pct and test_pct must be less than 1")
    
    # Calculate sizes
    validation_size <- floor(n_samples * validation_pct)
    
    # Create repeated splits
    repeated_splits <- lapply(1:n_repeats, function(r) {
        # For each repeat, randomly shuffle all indices
        all_indices <- sample(1:n_samples)
        
        # Take validation set from shuffled indices
        validation_indices <- if (!is.null(stratify)) {
            # Stratified sampling for validation
            strata <- unique(stratify)
            val_indices <- c()
            for (stratum in strata) {
                stratum_indices <- which(stratify == stratum)
                stratum_size <- floor(validation_size * length(stratum_indices) / n_samples)
                val_indices <- c(val_indices, 
                               sample(stratum_indices, stratum_size))
            }
            val_indices
        } else {
            # Random sampling for validation
            all_indices[1:validation_size]
        }
        
        # Get remaining indices for train+test
        remaining_indices <- setdiff(all_indices, validation_indices)
        remaining_indices <- sample(remaining_indices)  # Reshuffle remaining indices
        
        # Calculate test size for remaining data
        test_size_per_fold <- floor(length(remaining_indices) * test_pct / n_outer_folds)
        
        # Create outer folds
        outer_folds <- lapply(1:n_outer_folds, function(k) {
            # Randomly sample test indices for this fold
            available_indices <- remaining_indices
            
            test_idx <- if (!is.null(stratify)) {
                # Stratified sampling for test set
                strata <- unique(stratify[available_indices])
                test_indices <- c()
                for (stratum in strata) {
                    stratum_indices <- available_indices[stratify[available_indices] == stratum]
                    stratum_size <- floor(test_size_per_fold * 
                                        length(stratum_indices) / length(available_indices))
                    test_indices <- c(test_indices, 
                                    sample(stratum_indices, min(stratum_size, length(stratum_indices))))
                }
                test_indices
            } else {
                # Random sampling for test set
                sample(available_indices, test_size_per_fold)
            }
            
            # Get training indices (everything except test and validation)
            train_idx <- setdiff(remaining_indices, test_idx)
            
            # Create inner folds from training data
            inner_folds <- lapply(1:n_inner_folds, function(i) {
                # Randomly shuffle training indices for inner folds
                shuffled_train <- sample(train_idx)
                fold_size <- floor(length(train_idx) / n_inner_folds)
                start_idx <- (i-1) * fold_size + 1
                end_idx <- if(i == n_inner_folds) length(train_idx) else i * fold_size
                shuffled_train[start_idx:end_idx]
            })
            
            # Validate indices
            validate_indices(train_idx, n_samples, 
                           sprintf("Repeat %d Outer fold %d train", r, k))
            validate_indices(test_idx, n_samples, 
                           sprintf("Repeat %d Outer fold %d test", r, k))
            
            # Additional validation
            check_overlap(train_idx, test_idx, "train", "test")
            check_overlap(validation_indices, train_idx, "validation", "train")
            check_overlap(validation_indices, test_idx, "validation", "test")
            
            list(
                train_idx = train_idx,
                test_idx = test_idx,
                inner_folds = inner_folds
            )
        })
        
        list(
            repeat_num = r,
            validation = validation_indices,
            outer_splits = outer_folds
        )
    })
    
    # Add summary information
    attr(repeated_splits, "summary") <- list(
        n_samples = n_samples,
        n_repeats = n_repeats,
        n_outer_folds = n_outer_folds,
        n_inner_folds = n_inner_folds,
        stratified = !is.null(stratify),
        validation_pct = validation_pct,
        test_pct = test_pct,
        train_pct = 1 - (validation_pct + test_pct),
        seed = seed
    )
    
    return(repeated_splits)
}

#' Print summary of the CV structure with split percentages
#' @param cv_splits Cross-validation splits object
print_cv_structure <- function(cv_splits) {
    summary <- attr(cv_splits, "summary")
    cat("Cross-validation Structure Summary\n")
    cat("=================================\n")
    cat(sprintf("Total samples: %d\n", summary$n_samples))
    cat(sprintf("Number of repeats: %d\n", summary$n_repeats))
    cat(sprintf("Number of outer folds: %d\n", summary$n_outer_folds))
    cat(sprintf("Number of inner folds: %d\n", summary$n_inner_folds))
    cat(sprintf("Stratified: %s\n", ifelse(summary$stratified, "Yes", "No")))
    cat("\nSplit Ratios:\n")
    cat(sprintf("Validation: %.1f%%\n", summary$validation_pct * 100))
    cat(sprintf("Test: %.1f%%\n", summary$test_pct * 100))
    cat(sprintf("Train: %.1f%%\n", summary$train_pct * 100))
    
    cat("\nDetailed Structure:\n")
    for (r in seq_along(cv_splits)) {
        cat(sprintf("\nRepeat %d:\n", r))
        cat(sprintf("  Validation set size: %d\n", length(cv_splits[[r]]$validation)))
        
        for (k in seq_along(cv_splits[[r]]$outer_splits)) {
            outer <- cv_splits[[r]]$outer_splits[[k]]
            cat(sprintf("  Outer fold %d:\n", k))
            cat(sprintf("    Train size: %d\n", length(outer$train_idx)))
            cat(sprintf("    Test size: %d\n", length(outer$test_idx)))
            cat("    Inner fold sizes:", 
                paste(sapply(outer$inner_folds, length), collapse=", "), 
                "\n")
        }
    }
}

# Helper function to print summary of the CV structure
print_cv_structure <- function(cv_splits) {
    summary <- attr(cv_splits, "summary")
    cat("Cross-validation Structure Summary\n")
    cat("=================================\n")
    cat(sprintf("Total samples: %d\n", summary$n_samples))
    cat(sprintf("Number of repeats: %d\n", summary$n_repeats))
    cat(sprintf("Number of outer folds: %d\n", summary$n_outer_folds))
    cat(sprintf("Number of inner folds: %d\n", summary$n_inner_folds))
    cat(sprintf("Stratified: %s\n", ifelse(summary$stratified, "Yes", "No")))
    cat("\nDetailed Structure:\n")
    
    for (r in seq_along(cv_splits)) {
        cat(sprintf("\nRepeat %d:\n", r))
        cat(sprintf("  Validation set size: %d\n", length(cv_splits[[r]]$validation)))
        
        for (k in seq_along(cv_splits[[r]]$outer_splits)) {
            outer <- cv_splits[[r]]$outer_splits[[k]]
            cat(sprintf("  Outer fold %d:\n", k))
            cat(sprintf("    Train size: %d\n", length(outer$train_idx)))
            cat(sprintf("    Test size: %d\n", length(outer$test_idx)))
            cat("    Inner fold sizes:", 
                paste(sapply(outer$inner_folds, length), collapse=", "), 
                "\n")
        }
    }
}

#' Run nested cross-validation with random sampling
#' @param model Neural network model
#' @param datasets List of torch datasets
#' @param config Configuration parameters
#' @param cancer_type Current cancer type
#' @param validation_pct Percentage of data for validation (default 0.1)
#' @param test_pct Percentage of remaining data for testing (default 0.2)
#' @param seed Optional seed for reproducibility
#' @return List of results and models
run_nested_cv <- function(model, datasets, config, cancer_type, 
                         validation_pct = 0.1, test_pct = 0.2, seed = NULL) {
    
    # Find the event column index
    event_col_idx <- which(datasets$clinical_features == "demographics_vital_status_alive")
    if (length(event_col_idx) == 0) {
        stop("Could not find 'event' column in clinical features")
    }
    
    # Convert clinical tensor to R array and extract event information
    stratify <- if (!is.null(datasets$clinical)) {
        clinical_matrix <- as.array(datasets$clinical$cpu())
        clinical_matrix[, event_col_idx]
    } else {
        NULL
    }
    
    # Extract parameters from config
    n_repeats <- config$cv_params$outer_repeats
    n_outer_folds <- config$cv_params$outer_folds
    n_inner_folds <- config$cv_params$inner_folds
    
    # Get total number of samples
    n_samples <- dim(as.array(datasets[["clinical"]]$cpu()))[1]
    
    # Create all CV splits with random sampling
    cv_splits <- create_cv_splits(
        n_samples = n_samples,
        n_repeats = n_repeats,
        n_outer_folds = n_outer_folds,
        n_inner_folds = n_inner_folds,
        validation_pct = validation_pct,
        test_pct = test_pct,
        stratify = stratify,
        seed = seed
    )
    
    # Print CV structure for verification
    print_cv_structure(cv_splits)
    
    # Set up parallel processing
    plan(multisession, workers = config$parallel$workers)
    
    # Run repeated CV
    results <- future_lapply(seq_along(cv_splits), function(repeat_idx) {
        repeat_split <- cv_splits[[repeat_idx]]
        
        # Run outer CV
        outer_results <- lapply(seq_along(repeat_split$outer_splits), function(fold_idx) {
            outer_split <- repeat_split$outer_splits[[fold_idx]]
            
            # Run inner CV for hyperparameter tuning
            inner_results <- lapply(seq_along(outer_split$inner_folds), function(inner_fold_idx) {
                model_copy <- model$clone()
                inner_fold <- outer_split$inner_folds[[inner_fold_idx]]
                
                # Create training and validation datasets for this inner fold
                inner_train_data <- subset_datasets(datasets, inner_fold)
                inner_val_data <- subset_datasets(datasets, 
                                                setdiff(outer_split$train_idx, inner_fold))
                
                # Train model with current hyperparameters
                trained_model <- train_model(
                    model = model_copy,
                    train_data = inner_train_data,
                    val_data = inner_val_data,
                    config = config
                )
                
                # Evaluate on validation fold
                evaluate_model(trained_model, inner_val_data)
            })
            
            # Select best hyperparameters from inner CV
            best_params <- select_best_hyperparameters(inner_results)
            
            # Train final model for this outer fold
            model_copy <- model$clone()
            outer_train_data <- subset_datasets(datasets, outer_split$train_idx)
            outer_test_data <- subset_datasets(datasets, outer_split$test_idx)
            
            final_model <- train_model(
                model = model_copy,
                train_data = outer_train_data,
                val_data = outer_test_data,
                config = update_config(config, best_params)
            )
            
            # Evaluate on test set
            test_results <- evaluate_model(final_model, outer_test_data)
            
            list(
                fold = fold_idx,
                model = final_model,
                results = test_results,
                best_params = best_params,
                inner_results = inner_results
            )
        })
        
        # Evaluate best model from outer CV on validation set
        best_outer_model <- select_best_model(outer_results)
        validation_data <- subset_datasets(datasets, repeat_split$validation)
        validation_results <- evaluate_model(best_outer_model, validation_data)
        
        list(
            repeat_idx = repeat_idx,
            outer_results = outer_results,
            validation_results = validation_results,
            best_model = best_outer_model
        )
    })
    
    # Aggregate results across all repeats
    final_results <- aggregate_cv_results(results)
    
    # Select final best model
    final_model <- select_final_model(results)
    
    # Save CV structure
    results_dir <- file.path(config$main$paths$results_dir, cancer_type)
    dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
    
    saveRDS(
        cv_splits,
        file.path(results_dir, "cv_splits.rds")
    )
    
    return(list(
        results = final_results,
        final_model = final_model,
        cv_splits = cv_splits,
        raw_results = results  # Include raw results for detailed analysis if needed
    ))
}

#' Helper function to update configuration with new hyperparameters
#' @param config Original configuration
#' @param new_params New hyperparameters
#' @return Updated configuration
update_config <- function(config, new_params) {
    if (is.null(new_params)) return(config)
    
    # Deep copy of config
    new_config <- config
    
    # Update hyperparameters
    new_config$model$hyperparameters <- modifyList(
        new_config$model$hyperparameters,
        new_params
    )
    
    return(new_config)
}

#' Helper function to aggregate CV results
#' @param results List of results from all repeats
#' @return Aggregated results summary
aggregate_cv_results <- function(results) {
    # Extract metrics from all levels
    validation_metrics <- lapply(results, function(r) r$validation_results$metrics)
    outer_metrics <- lapply(results, function(r) {
        lapply(r$outer_results, function(o) o$results$metrics)
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
    
    # Collect hyperparameter statistics
    hyper_params <- lapply(results, function(r) {
        lapply(r$outer_results, function(o) o$best_params)
    })
    
    return(list(
        validation_summary = validation_summary,
        outer_summary = outer_summary,
        hyperparameter_summary = summarize_hyperparameters(hyper_params)
    ))
}

#' Helper function to summarize hyperparameter selections
#' @param hyper_params List of hyperparameters from all folds
#' @return Summary of hyperparameter selections
summarize_hyperparameters <- function(hyper_params) {
    # Flatten the nested list of hyperparameters
    flat_params <- unlist(hyper_params, recursive = FALSE)
    
    # For each hyperparameter, calculate statistics
    param_names <- unique(unlist(lapply(flat_params, names)))
    
    lapply(param_names, function(param) {
        values <- sapply(flat_params, function(p) p[[param]])
        if (is.numeric(values)) {
            list(
                mean = mean(values),
                sd = sd(values),
                median = median(values),
                min = min(values),
                max = max(values)
            )
        } else {
            # For categorical parameters, calculate frequencies
            table(values)
        }
    })
}

#' Subset datasets for CV splits
#' @param datasets List of datasets
#' @param indices Indices to subset
#' @return Subsetted datasets

subset_datasets <- function(datasets, indices) {
    if (is.null(datasets)) stop("Null datasets provided")
    if (length(indices) == 0) stop("Empty indices provided")
    
    tryCatch({
        lapply(datasets, function(dataset) {
            if (is.null(dataset)) return(NULL)
            if (inherits(dataset, "torch_tensor")) {
                dataset[indices, ]
            } else {
                dataset[indices, ]
            }
        })
    }, error = function(e) {
        stop(paste("Error subsetting datasets:", e$message))
    })
}


#' Select best hyperparameters from inner CV results
#' @param inner_results List of inner CV results
#' @return Best hyperparameter configuration
select_best_hyperparameters <- function(inner_results) {
    # Extract performance metrics
    performances <- sapply(inner_results, function(x) x$metric)
    
    # Select best configuration
    best_idx <- which.max(performances)
    inner_results[[best_idx]]$params
}

#' Update configuration with new hyperparameters
#' @param config Original configuration
#' @param new_params New hyperparameters
#' @return Updated configuration
update_config <- function(config, new_params) {
    # Deep copy of config
    new_config <- config
    
    # Update hyperparameters
    new_config$model$hyperparameters <- modifyList(
        new_config$model$hyperparameters,
        new_params
    )
    
    return(new_config)
}

#' Aggregate results from all CV repeats
#' @param results List of results from all repeats
#' @return Aggregated results
aggregate_cv_results <- function(results) {
    # Extract metrics from all levels
    metrics <- list(
        validation = lapply(results, function(r) r$validation_results$metrics),
        outer = lapply(results, function(r) {
            lapply(r$outer_results, function(o) o$results$metrics)
        })
    )
    
    # Calculate summary statistics
    summary_stats <- list(
        validation = list(
            mean = colMeans(do.call(rbind, metrics$validation)),
            sd = apply(do.call(rbind, metrics$validation), 2, sd)
        ),
        outer = list(
            mean = colMeans(do.call(rbind, unlist(metrics$outer, recursive = FALSE))),
            sd = apply(do.call(rbind, unlist(metrics$outer, recursive = FALSE)), 2, sd)
        )
    )
    
    return(list(
        metrics = metrics,
        summary = summary_stats
    ))
}

#' Select final model based on validation performance
#' @param results List of results from all repeats
#' @return Best performing model
select_final_model <- function(results) {
    # Extract validation performances
    performances <- sapply(results, function(r) r$validation_results$metrics$primary_metric)
    
    # Select best model
    best_repeat <- which.max(performances)
    results[[best_repeat]]$best_model
}

# Need to add evaluate_model function

evaluate_model <- function(model, data) {
    model$eval()
    loader <- dataloader(dataset = data, batch_size = 32, shuffle = FALSE)
    
    all_predictions <- list()
    all_targets <- list()
    
    with_no_grad({
        coro::loop(for (batch in loader) {
            output <- model(batch$data, batch$masks)
            all_predictions[[length(all_predictions) + 1]] <- output$predictions$cpu()
            all_targets[[length(all_targets) + 1]] <- batch$target$cpu()
        })
    })
    
    predictions <- torch_cat(all_predictions, dim = 1)
    targets <- torch_cat(all_targets, dim = 1)
    
    # Calculate metrics
    metrics <- list(
        primary_metric = calculate_primary_metric(predictions, targets),
        additional_metrics = calculate_additional_metrics(predictions, targets)
    )
    
    return(list(
        metrics = metrics,
        predictions = predictions,
        targets = targets
    ))
}

select_best_model <- function(outer_results) {
    if (length(outer_results) == 0) {
        stop("No outer results provided for model selection")
    }
    
    performances <- sapply(outer_results, function(x) {
        if (is.null(x$results$metrics$primary_metric)) {
            stop("Missing primary metric in results")
        }
        x$results$metrics$primary_metric
    })
    
    best_idx <- which.max(performances)
    if (length(best_idx) == 0) {
        stop("Could not determine best model")
    }
    
    return(outer_results[[best_idx]]$model)
}

