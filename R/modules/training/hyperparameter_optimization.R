# Generate random hyperparameters within reasonable ranges
generate_random_hyperparams <- function() {
    list(
        learning_rate = 10^(runif(1, min = -5, max = -2)),  # 1e-5 to 1e-2
        weight_decay = 10^(runif(1, min = -5, max = -3)),   # 1e-5 to 1e-3
        dropout = runif(1, min = 0.1, max = 0.5),
        batch_size = 2^sample(4:8, 1)  # 16 to 256 in powers of 2
    )
}

# Modified process_inner_folds with both model and config updates
process_inner_folds_with_hpo <- function(inner_folds, model, datasets, config, params) {
    lapply(seq_along(inner_folds), function(inner_idx) {
        message(sprintf("\nProcessing inner fold: %d", inner_idx))
        
        # Generate random hyperparameters for this fold
        hyperparams <- generate_random_hyperparams()


	message(sprintf("Using random hyperparameters:"))
        message(sprintf("- Learning rate: %e", hyperparams$learning_rate))
        message(sprintf("- Weight decay: %e", hyperparams$weight_decay))
        message(sprintf("- Dropout: %.2f", hyperparams$dropout))
        message(sprintf("- Batch size: %d", hyperparams$batch_size))

        # Update config
        trial_config <- config
        trial_config$model$optimizer$lr <- hyperparams$learning_rate
        trial_config$model$optimizer$weight_decay <- hyperparams$weight_decay
        trial_config$model$architecture$dropout <- hyperparams$dropout
        trial_config$model$batch_size <- hyperparams$batch_size

        # Create datasets for this fold
        inner_train_data <- subset_datasets(
            datasets,
            inner_folds[[inner_idx]]$train_idx,
            hyperparams$batch_size
        )
        inner_val_data <- subset_datasets(
            datasets,
            inner_folds[[inner_idx]]$test_idx,
            hyperparams$batch_size
        )

        # Create model copy
        inner_model <- model$create_copy()

        # Update dropout in model layers
        for (module in inner_model$modules) {
            if (inherits(module, "nn_dropout")) {
                module$p <- hyperparams$dropout
            }
        }

        # Train model with updated config (optimizer will be created in train_model)
        trained_model <- train_model(
            model = inner_model,
            train_data = inner_train_data,
            config = trial_config,
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

        # Store results
        result <- list(
            state_dict = trained_model$model$state_dict(),
            val_metrics = val_results$metrics,
            selected_features = trained_model$selected_features,
            best_loss = trained_model$best_loss,
            hyperparams = hyperparams,
            config = trial_config
        )

        # Cleanup
        rm(inner_model, trained_model, inner_train_data, inner_val_data)
        gc()

        return(result)

    })
}
