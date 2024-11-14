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


#' Create data splits for nested cross-validation
#' @param n_samples Total number of samples
#' @param n_repeats Number of repetitions for validation sets
#' @param n_outer_folds Number of outer folds
#' @param n_inner_folds Number of inner folds
#' @param stratify Vector of stratification labels (e.g., survival events)
#' @return List of indices for each split level
create_cv_splits <- function(n_samples, n_repeats, n_outer_folds, n_inner_folds, stratify = NULL) {
    # Create splits for repeated validation
    validation_splits <- lapply(1:n_repeats, function(r) {
        # If stratification is provided, use stratified sampling
        if (!is.null(stratify)) {
            fold_indices <- caret::createFolds(stratify, k = n_outer_folds, list = TRUE)
        } else {
            # Random sampling without stratification
            fold_indices <- split(sample(n_samples), rep(1:n_outer_folds, length.out = n_samples))
        }
        
        # Create validation and train+test sets
        list(
            validation = fold_indices[[1]],
            train_test = unlist(fold_indices[-1])
        )
    })
    
    # For each repeat, create outer and inner CV splits
    cv_splits <- lapply(validation_splits, function(repeat_split) {
        # Create outer folds from train+test data
        outer_splits <- if (!is.null(stratify)) {
            stratify_subset <- stratify[repeat_split$train_test]
            caret::createFolds(stratify_subset, k = n_outer_folds, list = TRUE)
        } else {
            split(sample(length(repeat_split$train_test)), 
                  rep(1:n_outer_folds, length.out = length(repeat_split$train_test)))
        }
        
        # For each outer fold, create inner folds
        inner_splits <- lapply(outer_splits, function(test_idx) {
            train_idx <- setdiff(seq_along(repeat_split$train_test), test_idx)
            
            # Create inner CV splits
            if (!is.null(stratify)) {
                stratify_inner <- stratify[repeat_split$train_test[train_idx]]
                inner_folds <- caret::createFolds(stratify_inner, k = n_inner_folds, list = TRUE)
            } else {
                inner_folds <- split(sample(length(train_idx)), 
                                   rep(1:n_inner_folds, length.out = length(train_idx)))
            }
            
            list(
                train_idx = train_idx,
                test_idx = test_idx,
                inner_folds = inner_folds
            )
        })
        
        list(
            validation = repeat_split$validation,
            outer_splits = inner_splits
        )
    })
    
    return(cv_splits)
}

#' Run nested cross-validation with repeated validation sets
#' @param datasets List of torch datasets
#' @param config Configuration parameters
#' @param cancer_type Current cancer type
#' @return List of results and models
run_nested_cv <- function(model, datasets, config, cancer_type) {


    # Extract stratification vector
    stratify <- get_stratification_vector(datasets)

    # Extract parameters from config
    n_repeats <- config$cv_params$outer_repeats
    n_outer_folds <- config$cv_params$outer_folds
    n_inner_folds <- config$cv_params$inner_folds
    
    # Get total number of samples
    n_samples <- nrow(datasets[[1]])
    
    # Get stratification vector if specified
    stratify <- if (!is.null(datasets$clinical)) {
        datasets$clinical$event  # Assuming 'event' is the stratification column
    } else {
        NULL
    }
    
    # Create all CV splits
    cv_splits <- create_cv_splits(
        n_samples = n_samples,
        n_repeats = n_repeats,
        n_outer_folds = n_outer_folds,
        n_inner_folds = n_inner_folds,
        stratify = stratify
    )
    
    # Initialize results storage
    results <- list()
    
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
                inner_fold <- outer_split$inner_folds[[inner_fold_idx]]
                
                # Create training and validation datasets for this inner fold
                inner_train_data <- subset_datasets(datasets, inner_fold)
                inner_val_data <- subset_datasets(datasets, 
                                                setdiff(outer_split$train_idx, inner_fold))
                
                # Train model with current hyperparameters
                model <- train_model(
                    train_data = inner_train_data,
                    val_data = inner_val_data,
                    config = config
                )
                
                # Evaluate on validation fold
                evaluate_model(model, inner_val_data)
            })
            
            # Select best hyperparameters from inner CV
            best_params <- select_best_hyperparameters(inner_results)
            
            # Train final model for this outer fold
            outer_train_data <- subset_datasets(datasets, outer_split$train_idx)
            outer_test_data <- subset_datasets(datasets, outer_split$test_idx)
            
            final_model <- train_model(
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
                best_params = best_params
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
    
    return(list(
        results = final_results,
        final_model = final_model,
        cv_splits = cv_splits
    ))
}

#' Subset datasets for CV splits
#' @param datasets List of datasets
#' @param indices Indices to subset
#' @return Subsetted datasets
subset_datasets <- function(datasets, indices) {
    lapply(datasets, function(dataset) {
        if (inherits(dataset, "torch_tensor")) {
            dataset[indices, ]
        } else {
            dataset[indices, ]
        }
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

