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

#' Create data splits for nested cross-validation with proper k-fold structure
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
                           validation_pct = 0.1, test_pct = 0.2, stratify = NULL, seed = NULL) {
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
    
    # Create R repeated splits
    repeated_splits <- lapply(1:n_repeats, function(r) {
        # Calculate split sizes
        validation_size <- floor(n_samples * validation_pct)
        remaining_samples <- n_samples - validation_size
        
        # First split: Validation vs rest
        all_indices <- 1:n_samples
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
            sample(all_indices, validation_size)
        }
        
        # Get remaining indices
        remaining_indices <- setdiff(all_indices, validation_indices)
        
        # Second split: Test vs Train from remaining
        test_size <- floor(remaining_samples * test_pct)
        test_indices <- sample(remaining_indices, test_size)
        train_indices <- setdiff(remaining_indices, test_indices)
        
        # Create K outer folds
        outer_folds <- lapply(1:n_outer_folds, function(k) {
            # For each K-fold, use all training data
            # Create M inner folds
            inner_folds <- lapply(1:n_inner_folds, function(m) {
                # For each M-fold:
                # 1. Use all training indices
                # 2. Randomly select test_pct for inner testing
                inner_test_size <- floor(length(train_indices) * test_pct)
                inner_test <- sample(train_indices, inner_test_size)
                inner_train <- setdiff(train_indices, inner_test)
                
                list(
                    train_idx = inner_train,
                    test_idx = inner_test
                )
            })
            
            names(inner_folds) <- paste0("K-Fold-", k, "-M-Fold-", 1:n_inner_folds)
            
            list(
                name = paste0("K-Fold-", k),
                train_idx = train_indices,  # Use full training set
                test_idx = test_indices,    # Same test set for all K folds
                inner_folds = inner_folds
            )
        })
        
        # Add names to outer folds
        names(outer_folds) <- paste0("K-Fold-", 1:n_outer_folds)
        
        # Print structure for first repeat
        if (r == 1) {
            message(sprintf("\nRepeat %d Structure:", r))
            message(sprintf("Total samples: %d", n_samples))
            message(sprintf("Validation set: %d samples (%.1f%%)", 
                          validation_size, 100 * validation_size/n_samples))
            message(sprintf("Test set: %d samples (%.1f%%)", 
                          test_size, 100 * test_size/remaining_samples))
            message(sprintf("Training set: %d samples", length(train_indices)))
            
            message("\nFold Structure:")
            message(sprintf("Outer: %d K-folds", n_outer_folds))
            for (k in seq_along(outer_folds)) {
                message(sprintf("  %s:", names(outer_folds)[k]))
                message(sprintf("    Train: %d samples", length(outer_folds[[k]]$train_idx)))
                message(sprintf("    Test: %d samples", length(outer_folds[[k]]$test_idx)))
                message("    Inner M-folds:")
                for (m in seq_along(outer_folds[[k]]$inner_folds)) {
                    message(sprintf("      %s: Train=%d, Test=%d samples", 
                                  names(outer_folds[[k]]$inner_folds)[m],
                                  length(outer_folds[[k]]$inner_folds[[m]]$train_idx),
                                  length(outer_folds[[k]]$inner_folds[[m]]$test_idx)))
                }
            }
        }
        
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
        train_pct = 1 - (validation_pct + test_pct)
    )
    
    return(repeated_splits)
}

#' Create random k folds with proper test/train structure
#' @param indices Vector of indices to split
#' @param k Number of folds
#' @param test_pct Percentage for test set
#' @param fold_prefix Prefix for fold labels
#' @param stratify_vec Optional stratification vector
create_random_kfold <- function(train_indices, test_indices, k, fold_prefix, stratify_vec = NULL) {
    # Test indices stay constant
    # Only split training indices into k folds
    n_train <- length(train_indices)
    fold_size <- floor(n_train / k)

    # Create k folds from training data
    folds <- vector("list", k)
    names(folds) <- paste0(fold_prefix, "-Fold-", 1:k)

    # Randomly shuffle training indices
    shuffled_train <- sample(train_indices)

    # Create k approximately equal-sized folds
    for (i in 1:k) {
        start_idx <- (i-1) * fold_size + 1
        if (i == k) {
            fold_train <- shuffled_train[start_idx:length(shuffled_train)]
        } else {
            end_idx <- i * fold_size
            fold_train <- shuffled_train[start_idx:end_idx]
        }

        folds[[i]] <- list(
            train = fold_train,
            test = test_indices  # Same test set for all folds
        )
    }

    return(folds)
}

#' Create inner M folds
#' @param train_indices Training indices to split
#' @param test_pct Percentage for test set
#' @param m Number of folds
create_inner_folds <- function(train_indices, test_pct, m) {
    n_train <- length(train_indices)
    n_test <- floor(n_train * test_pct)

    # For each fold, randomly select test set
    shuffled_indices <- sample(train_indices)
    test_indices <- shuffled_indices[1:n_test]
    inner_train_indices <- shuffled_indices[(n_test + 1):length(shuffled_indices)]

    fold_size <- floor(length(inner_train_indices) / m)
    inner_folds <- vector("list", m)

    for (i in 1:m) {
        start_idx <- (i-1) * fold_size + 1
        if (i == m) {
            fold_train <- inner_train_indices[start_idx:length(inner_train_indices)]
        } else {
            end_idx <- i * fold_size
            fold_train <- inner_train_indices[start_idx:end_idx]
        }

        inner_folds[[i]] <- list(
            train = fold_train,
            test = test_indices
        )
    }

    return(inner_folds)
}



#' Run nested cross-validation with memory optimization
#' @param model Neural network model
#' @param datasets List of torch datasets
#' @param config Configuration parameters
#' @param cancer_type Current cancer type
#' @param validation_pct Percentage for validation
#' @param test_pct Percentage for testing
#' @param seed Optional seed for reproducibility
#' @param max_workers Maximum number of parallel workers
#' @param batch_size Batch size for data loading
#' @return List of results and models
run_nested_cv <- function(model, datasets, config, cancer_type, 
                         validation_pct = 0.1, test_pct = 0.2, seed = NULL,
                         max_workers = 2, batch_size = 32) {
    
    # Clear memory at start
    gc()
    
    # Set memory limits for parallel processing
    options(future.globals.maxSize = 2000 * 1024^2)  # 2GB limit per worker
    
    # Limit number of parallel workers based on available memory
    available_memory <- as.numeric(system("free -g | awk 'NR==2 {print $4}'", intern=TRUE))
    suggested_workers <- min(max_workers, floor(available_memory / 4))  # Assume 4GB per worker
    actual_workers <- max(1, suggested_workers)  # Ensure at least 1 worker
    
    message(sprintf("Using %d workers based on available memory", actual_workers))
    
    # Find the event column index
    event_col_idx <- which(datasets$clinical_features == "demographics_vital_status_alive")
    if (length(event_col_idx) == 0) {
        stop("Could not find 'event' column in clinical features")
    }
    
    # Extract stratification information efficiently
    stratify <- if (!is.null(datasets$clinical)) {
        clinical_matrix <- as.array(datasets$clinical$cpu())
        result <- clinical_matrix[, event_col_idx]
        rm(clinical_matrix)
        gc()
        result
    } else {
        NULL
    }
    
    # Get total number of samples
    n_samples <- dim(as.array(datasets[["clinical"]]$cpu()))[1]
    
    # Extract parameters from config
    n_repeats <- config$cv_params$outer_repeats
    n_outer_folds <- config$cv_params$outer_folds
    n_inner_folds <- config$cv_params$inner_folds
    
    # Create CV splits
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
    
    
    # Set up parallel processing with memory limits
    plan(multisession, workers = actual_workers)
    
    # Function to clear torch CUDA cache if available
    clear_cuda_memory <- function() {
        if (torch::cuda_is_available()) {
            torch::cuda_empty_cache()
        }
        gc()
    }
    
    # Process each repeat sequentially to manage memory
    results <- vector("list", length(cv_splits))
    
    #for (repeat_idx in seq_along(cv_splits)) {
    for (repeat_idx in 1) {
    message(sprintf("Processing repeat %d/%d", repeat_idx, length(cv_splits)))
        
        repeat_split <- cv_splits[[repeat_idx]]
        
        # Process outer folds in parallel with memory management
        outer_results <- future_lapply(seq_along(repeat_split$outer_splits), 
                                     function(fold_idx) {
            # Clear memory at start of each fold
            clear_cuda_memory()
            
            outer_split <- repeat_split$outer_splits[[fold_idx]]
            
            # Process inner folds with memory management
            inner_results <- lapply(seq_along(outer_split$inner_folds), 
                                  function(inner_fold_idx) {
                # Clear memory before each inner fold
                clear_cuda_memory()
                
                model_copy <- model$clone()
                inner_fold <- outer_split$inner_folds[[inner_fold_idx]]
                
                # Create datasets with memory-efficient subset function
                inner_train_data <- subset_datasets(
                    datasets, inner_fold, 
                    batch_size = batch_size
                )
                inner_val_data <- subset_datasets(
                    datasets, 
                    setdiff(outer_split$train_idx, inner_fold),
                    batch_size = batch_size
                )
                
                # Train and evaluate
                trained_model <- train_model(
                    model = model_copy,
                    train_data = inner_train_data,
                    val_data = inner_val_data,
                    config = config,
                    batch_size = batch_size
                )
                
                result <- evaluate_model(trained_model, inner_val_data)
                
                # Clean up
                rm(model_copy, inner_train_data, inner_val_data, trained_model)
                clear_cuda_memory()
                
                result
            })
            
            # Select best hyperparameters
            best_params <- select_best_hyperparameters(inner_results)
            
            # Train final model for outer fold
            model_copy <- model$clone()
            outer_train_data <- subset_datasets(
                datasets, outer_split$train_idx,
                batch_size = batch_size
            )
            outer_test_data <- subset_datasets(
                datasets, outer_split$test_idx,
                batch_size = batch_size
            )
            
            final_model <- train_model(
                model = model_copy,
                train_data = outer_train_data,
                val_data = outer_test_data,
                config = update_config(config, best_params),
                batch_size = batch_size
            )
            
            # Evaluate and clean up
            test_results <- evaluate_model(final_model, outer_test_data)
            
            rm(model_copy, outer_train_data, outer_test_data)
            clear_cuda_memory()
            
            list(
                fold = fold_idx,
                model = final_model,
                results = test_results,
                best_params = best_params,
                inner_results = inner_results
            )
        })
        
        # Evaluate validation set
        best_outer_model <- select_best_model(outer_results)
        validation_data <- subset_datasets(
            datasets, repeat_split$validation,
            batch_size = batch_size
        )
        validation_results <- evaluate_model(best_outer_model, validation_data)
        
        # Store results for this repeat
        results[[repeat_idx]] <- list(
            repeat_idx = repeat_idx,
            outer_results = outer_results,
            validation_results = validation_results,
            best_model = best_outer_model
        )
        
        # Clean up after each repeat
        rm(best_outer_model, validation_data)
        clear_cuda_memory()
        
        # Save intermediate results
        saveRDS(
            results[[repeat_idx]],
            file.path(
                config$main$paths$results_dir,
                cancer_type,
                sprintf("repeat_%d_results.rds", repeat_idx)
            )
        )
    }
    
    # Aggregate results
    final_results <- aggregate_cv_results(results)
    
    # Select final best model
    final_model <- select_final_model(results)
    
    # Save final results
    results_dir <- file.path(config$main$paths$results_dir, cancer_type)
    dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
    
    saveRDS(cv_splits, file.path(results_dir, "cv_splits.rds"))
    
    return(list(
        results = final_results,
        final_model = final_model,
        cv_splits = cv_splits,
        raw_results = results
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

#' Memory-efficient dataset subsetting
#' @param datasets List of datasets
#' @param indices Indices to subset
#' @param batch_size Batch size for data loading
#' @return Subsetted datasets
subset_datasets <- function(datasets, indices, batch_size = 32) {
    if (is.null(datasets)) stop("Null datasets provided")
    if (length(indices) == 0) stop("Empty indices provided")
    
    # Process in batches if dataset is large
    if (length(indices) > 1000) {
        # Process in chunks
        chunk_size <- 1000
        chunks <- split(indices, ceiling(seq_along(indices)/chunk_size))
        
        result <- lapply(datasets, function(dataset) {
            if (is.null(dataset)) return(NULL)
            
            # Process each chunk
            combined_result <- NULL
            for (chunk in chunks) {
                chunk_result <- if (inherits(dataset, "torch_tensor")) {
                    dataset[chunk, ]
                } else {
                    dataset[chunk, ]
                }
                
                if (is.null(combined_result)) {
                    combined_result <- chunk_result
                } else {
                    combined_result <- rbind(combined_result, chunk_result)
                }
                
                # Clear memory after each chunk
                gc()
            }
            
            combined_result
        })
        
        return(result)
    } else {
        # Process small datasets directly
        return(lapply(datasets, function(dataset) {
            if (is.null(dataset)) return(NULL)
            if (inherits(dataset, "torch_tensor")) {
                dataset[indices, ]
            } else {
                dataset[indices, ]
            }
        }))
    }
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

