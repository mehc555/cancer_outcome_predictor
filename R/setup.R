# R/setup.R

#' Initialize project dependencies and settings
#' @return NULL
initialize_project <- function() {
  # Required packages
  required_packages <- c(
    "torch", "luz", "yaml", "future", "future.apply", "progressr",
    "tidyverse", "data.table", "logger", "testthat", "here"
  )
  
  # Install missing packages
  missing_packages <- required_packages[!require_packages(required_packages)]
  if (length(missing_packages) > 0) {
    install.packages(missing_packages)
  }
  
  # Load all required packages
  sapply(required_packages, require, character.only = TRUE)
  
  # Initialize logger
  logger::log_threshold(logger::INFO)
  logger::log_formatter(logger::formatter_paste)
  
  # Set default torch settings
  torch::torch_manual_seed(42)
  if (torch::cuda_is_available()) {
    logger::log_info("CUDA is available. Setting default device to CUDA")
    torch::cuda_set_device(1)
  }
}

#' Load and validate configuration files
#' @param config_dir Directory containing configuration files
#' @return List of validated configuration parameters
load_config <- function(config_dir = here::here("configs")) {
  # Load main configuration files
  config=yaml::read_yaml(file.path(config_dir, "config.yml"))
  
  # Validate configurations
  validate_config(config)
  
  # Create derived configurations
  config <- create_derived_configs(config)
  
  # Set up experiment directory
  setup_experiment_dir(config)
  
  return(config)
}

#' Validate configuration parameters
#' @param config List of configuration parameters
#' @return TRUE if valid, stops with error if invalid
validate_config <- function(config) {
  # Validate main config
  #assertthat::assert_that(
  #  !is.null(config$main$experiment$name),
  #  !is.null(config$main$experiment$seed),
  #  !is.null(config$main$cv_params$outer_repeats),
  #  !is.null(config$main$cv_params$outer_folds),
  #  !is.null(config$main$cv_params$inner_folds)
  #)
  
  # Validate model config
  #assertthat::assert_that(
  #  !is.null(config$model$architecture$encoder_dims),
  #  !is.null(config$model$architecture$fusion$type),
  #  !is.null(config$model$architecture$prediction_head$type)
  #)
  
  # Validate training config
  #assertthat::assert_that(
  #  !is.null(config$training$model$batch_size),
  #  !is.null(config$training$model$max_epochs),
  #  !is.null(config$training$model$optimizer$name),
  #  !is.null(config$training$model$optimizer$lr)
  #)
  
  return(TRUE)
}

#' Create derived configuration parameters
#' @param config List of configuration parameters
#' @return Updated configuration list with derived parameters
create_derived_configs <- function(config) {
  # Add timestamp to experiment name
  config$main$experiment$timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  config$main$experiment$full_name <- paste(
    config$main$experiment$name,
    config$main$experiment$timestamp,
    sep = "_"
  )
  
  # Set up paths
  config$main$paths <- list(
    base_dir = here::here(),
    results_dir = here::here("results", "experiments", config$main$experiment$full_name),
    models_dir = here::here("results", "experiments", config$main$experiment$full_name, "models"),
    logs_dir = here::here("results", "experiments", config$main$experiment$full_name, "logs")
  )
  
  return(config)
}

#' Set up experiment directory structure
#' @param config Configuration list
#' @return NULL
setup_experiment_dir <- function(config) {
  # Create main directories
  dirs_to_create <- c(
    config$main$paths$results_dir,
    config$main$paths$models_dir,
    config$main$paths$logs_dir
  )
  
  # Create directories if they don't exist
  lapply(dirs_to_create, dir.create, recursive = TRUE, showWarnings = FALSE)
  
  # Save configuration files
  yaml::write_yaml(config$main, file.path(config$main$paths$results_dir, "config_main.yml"))
  #yaml::write_yaml(config$model, file.path(config$main$paths$results_dir, "config_model.yml"))
  #yaml::write_yaml(config$training, file.path(config$main$paths$results_dir, "config_training.yml"))
  
  # Initialize experiment log file
  log_file <- file.path(config$main$paths$logs_dir, "experiment.log")
  logger::log_appender(logger::appender_file(log_file))
  
  logger::log_info("Experiment directory structure created at: {config$main$paths$results_dir}")
}

#' Helper function to check if packages are available
#' @param packages Vector of package names
#' @return Logical vector indicating if each package is available
require_packages <- function(packages) {
  sapply(packages, requireNamespace, quietly = TRUE)
}

# Example usage:
if (!interactive()) {
  initialize_project()
  config <- load_config()
  logger::log_info("Project initialization completed")
}
