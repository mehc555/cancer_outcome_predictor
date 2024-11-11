# Clinical processor (R/modules/data_handling/clinical_processor.R)

library(tidyverse)
library(recipes)

#' Create processed data directory
#' @param cancer_type Character string specifying cancer type
#' @param base_dir Base directory containing the data
#' @return Path to processed directory
#' @keywords internal
create_processed_dir <- function(cancer_type, base_dir = "data/GDC_TCGA") {
  processed_dir <- file.path(base_dir, cancer_type, "processed")
  if (!dir.exists(processed_dir)) {
    dir.create(processed_dir, recursive = TRUE)
  }
  return(processed_dir)
}

#' Process clinical data for deep neural network input
#' @param cancer_type Character string specifying cancer type
#' @param common_features Vector of column names common to all cancer types
#' @param impute Logical indicating whether to impute missing values (default: FALSE)
#' @param impute_method Character string specifying imputation method for categorical variables
#' @param base_dir Base directory containing the data
#' @return List containing processed data frames and preprocessing recipe
#' @export
process_clinical_data <- function(cancer_type, 
                                common_features, 
                                impute = FALSE,
                                impute_method = "missing_category",
                                base_dir = "data/GDC_TCGA") {
    
  # Validate inputs
  if (!impute_method %in% c("missing_category", "mode", "none")) {
    stop("impute_method must be one of: 'missing_category', 'mode', 'none'")
  }
  
  # Construct file paths
  input_file <- file.path(base_dir, cancer_type, "clinical.tsv")
  processed_dir <- create_processed_dir(cancer_type, base_dir)
  
  if (!file.exists(input_file)) {
    stop(sprintf("Clinical data file not found for cancer type %s", cancer_type))
  }
  
  # Read clinical data
  clinical_data <- read_tsv(input_file, show_col_types = FALSE)
  
  # Define essential clinical features from common features
  essential_features <- list(
    # Demographics
    age = "age_at_index.demographic",
    gender = "gender.demographic",
    race = "race.demographic",
    ethnicity = "ethnicity.demographic",
    
    # Disease characteristics
    stage = "ajcc_pathologic_stage.diagnoses",
    tumor_grade = "tumor_grade.diagnoses",
    
    # Survival information
    vital_status = "vital_status.demographic",
    days_to_death = "days_to_death.demographic",
    days_to_last_follow_up = "days_to_last_follow_up.diagnoses"
  )
  
  # Keep only essential features that are present in common_features
  available_features <- essential_features[essential_features %in% common_features]
  
  # Initial data processing
  clinical_processed <- clinical_data %>%
    select(sample_id = sample) 
  
  # Add available features
  for (new_name in names(available_features)) {
    old_name <- available_features[[new_name]]
    if (old_name %in% colnames(clinical_data)) {
      clinical_processed[[new_name]] <- clinical_data[[old_name]]
    }
  }
  
  # Get column types
  categorical_cols <- intersect(
    c("gender", "race", "ethnicity", "stage", "tumor_grade"),
    colnames(clinical_processed)
  )
  
  numeric_cols <- intersect(
    c("age"),
    colnames(clinical_processed)
  )
  
  # Handle survival information
  survival_cols <- c("vital_status", "days_to_death", "days_to_last_follow_up")
  if (all(survival_cols %in% colnames(clinical_processed))) {
    clinical_processed <- clinical_processed %>%
      mutate(
        vital_status = str_to_upper(vital_status),
        vital_status = case_when(
          vital_status %in% c("DEAD", "DECEASED", "EXPIRED") ~ "Dead",
          vital_status %in% c("ALIVE", "LIVING") ~ "Alive",
          TRUE ~ NA_character_
        ),
        days_to_death = as.numeric(days_to_death),
        days_to_last_follow_up = as.numeric(days_to_last_follow_up),
        event = case_when(
          vital_status == "Dead" ~ 1L,
          vital_status == "Alive" ~ 0L,
          TRUE ~ NA_integer_
        ),
        survival_time = case_when(
          event == 1 & !is.na(days_to_death) ~ abs(days_to_death),
          event == 0 & !is.na(days_to_last_follow_up) ~ abs(days_to_last_follow_up),
          TRUE ~ NA_real_
        )
      ) %>%
      select(-c(days_to_death, days_to_last_follow_up, vital_status))
    
    numeric_cols <- c(numeric_cols, "survival_time")
  }
  
  # Save pre-processed matrix before standardization/encoding
  clinical_formatted <- clinical_processed
  write_tsv(clinical_formatted, 
            file.path(processed_dir, "clinical_formatted.tsv"))
  
  # Create preprocessing recipe
  recipe_spec <- recipe(~ ., data = clinical_processed) %>%
    update_role(sample_id, new_role = "id") %>%
    step_string2factor(all_nominal_predictors())
  
  # Add imputation steps if requested
  if (impute) {
    if (length(numeric_cols) > 0) {
      recipe_spec <- recipe_spec %>%
        step_impute_median(all_of(numeric_cols))
    }
    
    if (length(categorical_cols) > 0) {
      if (impute_method == "missing_category") {
        recipe_spec <- recipe_spec %>%
          step_unknown(all_of(categorical_cols))
      } else if (impute_method == "mode") {
        recipe_spec <- recipe_spec %>%
          step_impute_mode(all_of(categorical_cols))
      }
    }
  }
  
  # Create preprocessing recipe
  
 recipe_spec <- recipe(~ ., data = clinical_processed) %>%
  # Remove identifier column from preprocessing
  update_role(sample_id, new_role = "id") %>%
  # First convert strings to factors
  step_string2factor(all_nominal_predictors()) %>%
  # Handle missing values before dummy encoding
  step_unknown(all_nominal_predictors(), new_level = "Missing") %>%
  # Remove zero variance predictors
  step_zv(all_predictors()) %>%
  # Create dummy variables
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  # Only normalize true continuous variables (age and survival_time)
  # All other variables are binary from one-hot encoding
  step_normalize(all_of(c("age", "survival_time")))


  # Prep the recipe and process the data
  recipe_trained <- prep(recipe_spec, training = clinical_processed, retain = TRUE)
  clinical_processed_dnn <- bake(recipe_trained, new_data = clinical_processed)

  # Ensure only one sample_id column exists and it's first
  clinical_processed_dnn <- clinical_processed_dnn %>%
    select(-starts_with("sample_id")) %>%  # Remove any existing sample_id columns
    bind_cols(select(clinical_processed, sample_id)) %>%  # Add the original sample_id
    select(sample_id, everything())  # Ensure sample_id is first
  
  # Calculate missing data summary
  missing_summary <- clinical_formatted %>%
    summarise(across(everything(), ~sum(is.na(.))/n())) %>%
    gather(variable, missing_proportion) %>%
    filter(variable != "sample_id") %>%
    arrange(desc(missing_proportion))
  
  # Save processed data and recipe
  write_tsv(clinical_processed_dnn, 
            file.path(processed_dir, "clinical_processed_dnn.tsv"))
  saveRDS(recipe_trained, 
          file.path(processed_dir, "preprocessing_recipe.rds"))
  
  # Print processing summary
  message(sprintf("\nClinical processing summary for %s:", cancer_type))
  message("\nPre-processed data shape:")
  message(sprintf("- Features: %d", ncol(clinical_formatted) - 1))
  message("- Feature names: ", 
          paste(setdiff(colnames(clinical_formatted), "sample_id"), collapse = ", "))
  
  message("\nPost-processed data shape:")
  message(sprintf("- One-hot encoded features: %d", ncol(clinical_processed_dnn) - 1))
  
  message("\nMissing data summary:")
  for(i in 1:nrow(missing_summary)) {
    if(missing_summary$missing_proportion[i] > 0) {
      message(sprintf("- %s: %.1f%% missing", 
                     missing_summary$variable[i], 
                     missing_summary$missing_proportion[i] * 100))
    }
  }
  
  message("\nPreprocessing steps applied:")
  message("- Categorical variables one-hot encoded")
  message("- Numeric variables standardized (mean=0, sd=1)")
  if (impute) {
    message("- Missing values handled:")
    message(sprintf("  * Numeric: median imputation"))
    message(sprintf("  * Categorical: %s", 
            switch(impute_method,
                   "missing_category" = "new category for missing values",
                   "mode" = "most frequent category",
                   "none" = "left as missing")))
  } else {
    message("- Missing values preserved (no imputation)")
  }
  
  # Return processed data, original formatted data, and recipe
  return(list(
    data_dnn = clinical_processed_dnn,
    data_formatted = clinical_formatted,
    recipe = recipe_trained,
    missing_summary = missing_summary
  ))
}

#' Validate clinical data structure and content
#' @param clinical_data Processed clinical data frame
#' @return Logical indicating if validation passed (invisible)
#' @keywords internal
validate_clinical_data <- function(clinical_data) {
  # Check if data frame is empty
  if (nrow(clinical_data) == 0) {
    stop("No samples remained after processing")
  }
  
  # Check for sample_id column
  if (!"sample_id" %in% colnames(clinical_data)) {
    stop("sample_id column not found in clinical data")
  }
  
  # Validate numeric features
  numeric_cols <- clinical_data %>%
    select(where(is.numeric)) %>%
    names()
  
  for (col in numeric_cols) {
    if (any(is.infinite(clinical_data[[col]]))) {
      stop(sprintf("Infinite values found in column: %s", col))
    }
  }
  
  # Validate that all features are either numeric or logical (for one-hot encoded)
  non_id_cols <- setdiff(names(clinical_data), "sample_id")
  invalid_cols <- non_id_cols[!sapply(clinical_data[non_id_cols], function(x) {
    is.numeric(x) || is.logical(x)
  })]
  
  if (length(invalid_cols) > 0) {
    stop(sprintf("Invalid column types found in: %s", 
                paste(invalid_cols, collapse = ", ")))
  }
  
  invisible(TRUE)
}
