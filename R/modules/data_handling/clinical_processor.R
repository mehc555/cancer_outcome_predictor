# Clinical processor (R/modules/data_handling/clinical_processor.R)

library(tidyverse)
library(recipes)


#' Clean feature names and values
#' @param x Character vector of names/values to clean
#' @return Cleaned character vector
#' @keywords internal
clean_names <- function(x) {
  x %>%
    str_replace_all("[[:space:]]+", "_") %>%
    str_replace_all("\\.", "_") %>%
    str_replace_all("__+", "_") %>%
    str_replace_all("^X_+|^X\\.", "") %>%
    str_replace_all("\\(|\\)", "") %>%
    str_to_lower()
}

#' Define cancer-specific feature mappings
#' @param cancer_type Character string specifying cancer type
#' @return List of feature mappings for specified cancer type
#' @keywords internal
get_cancer_features <- function(cancer_type) {
  # Common features across cancer types
  common_features <- list(
    demographics = c(
      "age" = "age_at_index.demographic",
      "gender" = "gender.demographic",
      "race" = "race.demographic",
      "ethnicity" = "ethnicity.demographic",
      "vital_status" = "vital_status.demographic",
      "days_to_death" = "days_to_death.demographic",
      "days_to_last_follow_up" = "days_to_last_follow_up.diagnoses"
    ),
    
    disease = c(
      "disease_type" = "disease_type",
      "age_at_diagnosis_days" = "age_at_diagnosis.diagnoses",
      "stage" = "ajcc_pathologic_stage.diagnoses",
      "tumor_grade" = "tumor_grade.diagnoses",
      "primary_diagnosis" = "primary_diagnosis.diagnoses",
      "prior_malignancy" = "prior_malignancy.diagnoses",
      "prior_treatment" = "prior_treatment.diagnoses",
      "tumor_size_t" = "ajcc_pathologic_t.diagnoses",
      "metastasis_m" = "ajcc_pathologic_m.diagnoses",
      "tumor_type" = "tissue.type.samples"
    ),
    
    treatment = c(
      "treatment_type" = "treatment_type.treatments.diagnoses",
      "treatment_or_therapy" = "treatment_or_therapy.treatments.diagnoses"
    )
  )
  
  # Cancer-specific feature mappings
  cancer_specific <- switch(
    cancer_type,
    "LUAD" = list(
      exposures = c(
        "cigarettes_per_day" = "cigarettes_per_day.exposures",
        "years_smoked" = "years_smoked.exposures",
        "pack_years_smoked" = "pack_years_smoked.exposures",
        "alcohol_history" = "alcohol_history.exposures"
      ),
      disease_specific = c(
        "site_of_resection" = "site_of_resection_or_biopsy.diagnoses",
        "primary_site" = "primary_site"
      )
    ),
    "COAD" = list(
      disease_specific = c(
        "site_of_resection" = "site_of_resection_or_biopsy.diagnoses",
        "primary_site" = "primary_site",
        "colon_location" = "tissue_or_organ_of_origin.diagnoses"
      )
    ),
    "BRCA" = list(
      disease_specific = c(
        "morphology" = "morphology.diagnoses",
        "primary_diagnosis" = "primary_diagnosis.diagnoses"
      )
    ),
    stop(sprintf("Unsupported cancer type: %s", cancer_type))
  )
  
  c(common_features, cancer_specific)
}

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

#' Process treatment information
#' @param df Data frame containing treatment information
#' @return Data frame with simplified treatment columns
#' @keywords internal
process_treatments <- function(df) {
  df %>%
    mutate(
      received_radiation = str_detect(treatment_type, "Radiation"),
      received_pharmaceutical = str_detect(treatment_type, "Pharmaceutical"),
      treatment_response = case_when(
        str_detect(treatment_or_therapy, "yes") ~ "responded",
        str_detect(treatment_or_therapy, "no") ~ "no_response",
        TRUE ~ "unknown"
      )
    ) %>%
    select(-treatment_type, -treatment_or_therapy)
}

#' Process clinical data for deep neural network input
#' @param cancer_type Character string specifying cancer type
#' @param impute Logical indicating whether to impute missing values (default: FALSE)
#' @param impute_method Character string specifying imputation method for categorical variables
#' @param base_dir Base directory containing the data
#' @return List containing processed data frames and preprocessing recipe
#' @export

process_clinical_data <- function(cancer_type,
                                impute = FALSE,
                                impute_method = "missing_category",
                                base_dir = "data/GDC_TCGA") {

  # Validate inputs
  if (!impute_method %in% c("missing_category", "mode", "none")) {
    stop("impute_method must be one of: 'missing_category', 'mode', 'none'")
  }

  # Get cancer-specific feature mappings
  feature_groups <- get_cancer_features(cancer_type)

  # Construct file paths
  input_file <- file.path(base_dir, cancer_type, "clinical.tsv")
  processed_dir <- create_processed_dir(cancer_type, base_dir)

  if (!file.exists(input_file)) {
    stop(sprintf("Clinical data file not found for cancer type %s", cancer_type))
  }

  # Read clinical data
  clinical_data <- read_tsv(input_file, show_col_types = FALSE)

  # Initial data processing with sample_id
  clinical_processed <- clinical_data %>%
    select(sample_id = sample)

  # Add available features
  all_features <- unlist(feature_groups)
  features_added <- character(0)

  for (new_name in names(all_features)) {
    old_name <- all_features[[new_name]]
    if (old_name %in% colnames(clinical_data)) {
      if (!all(is.na(clinical_data[[old_name]]))) {
        clinical_processed[[new_name]] <- clinical_data[[old_name]]
        features_added <- c(features_added, new_name)
      }
    }
  }

  # Clean column names
  names(clinical_processed) <- clean_names(names(clinical_processed))

  # Identify column types
  categorical_cols <- names(clinical_processed)[sapply(clinical_processed, is.character)]
  categorical_cols <- setdiff(categorical_cols, "sample_id")
  numeric_cols <- names(clinical_processed)[sapply(clinical_processed, is.numeric)]
  numeric_cols <- setdiff(numeric_cols, "sample_id")

  # Pre-process categorical variables
  if (length(categorical_cols) > 0) {
    clinical_processed <- clinical_processed %>%
      mutate(across(all_of(categorical_cols),
                   ~if_else(is.na(.) | . %in% c("not reported", "missing", ""),
                           "missing",
                           clean_names(.))))

    # Convert to factors with explicit missing level
    clinical_processed <- clinical_processed %>%
      mutate(across(all_of(categorical_cols),
                   ~factor(., levels = c(unique(.[. != "missing"]), "missing"))))
  }

  # Handle numeric variables
  if (length(numeric_cols) > 0) {
    clinical_processed <- clinical_processed %>%
      mutate(across(all_of(numeric_cols), as.numeric))
  }

  # Handle survival information
  if (all(c("vital_status", "days_to_death", "days_to_last_follow_up") %in% colnames(clinical_processed))) {
    clinical_processed <- clinical_processed %>%
      mutate(
        vital_status = str_to_lower(vital_status),
        vital_status = case_when(
          vital_status %in% c("dead", "deceased", "expired") ~ "dead",
          vital_status %in% c("alive", "living") ~ "alive",
          TRUE ~ "missing"
        ),
        days_to_death = as.numeric(days_to_death),
        days_to_last_follow_up = as.numeric(days_to_last_follow_up),
        event = case_when(
          vital_status == "dead" ~ 1L,
          vital_status == "alive" ~ 0L,
          TRUE ~ NA_integer_
        ),
        survival_time = coalesce(
          abs(days_to_death),
          abs(days_to_last_follow_up)
        )
      ) %>%
      select(-c(days_to_death, days_to_last_follow_up, vital_status))
  }

  # Special handling for morphology codes if present
  if ("morphology" %in% names(clinical_processed)) {
    clinical_processed <- clinical_processed %>%
      mutate(
        morphology = str_replace(morphology, "/.*$", ""),
        morphology = coalesce(morphology_map[morphology], morphology)
      )
  }

  # Save pre-processed data before encoding
  clinical_formatted <- clinical_processed
  write_tsv(clinical_formatted,
            file.path(processed_dir, "clinical_formatted.tsv"))

  # Create recipe
  recipe_spec <- recipe(~ ., data = clinical_processed) %>%
    update_role(sample_id, new_role = "id")

  # Add steps for categorical variables
  if (length(categorical_cols) > 0) {
    recipe_spec <- recipe_spec %>%
      step_other(all_nominal_predictors(), threshold = 0.01, other = "rare") %>%
      step_dummy(all_nominal_predictors(), one_hot = TRUE)
  }

  # Add steps for numeric variables
  if (length(numeric_cols) > 0) {
    recipe_spec <- recipe_spec %>%
      step_normalize(all_of(numeric_cols))
  }

  # Prep and bake
  recipe_trained <- prep(recipe_spec, training = clinical_processed)
  clinical_processed_dnn <- bake(recipe_trained, new_data = clinical_processed)

  # Ensure sample_id is first column
  clinical_processed_dnn <- clinical_processed_dnn %>%
    select(sample_id, everything())

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

  message("\nFeatures included by category:")
  for (group_name in names(feature_groups)) {
    features_in_group <- names(feature_groups[[group_name]])
    available_features <- intersect(features_in_group, features_added)

    if (length(available_features) > 0) {
      message(sprintf("\n%s:", tools::toTitleCase(group_name)))
      message(paste("  -", available_features, collapse = "\n"))
    }
  }

  message("\nData dimensions:")
  message(sprintf("- Raw features: %d", ncol(clinical_formatted) - 1))
  message(sprintf("- Processed features: %d", ncol(clinical_processed_dnn) - 1))

  message("\nMissing data summary:")
  for(i in 1:nrow(missing_summary)) {
    if(missing_summary$missing_proportion[i] > 0) {
      message(sprintf("- %s: %.1f%% missing",
                     missing_summary$variable[i],
                     missing_summary$missing_proportion[i] * 100))
    }
  }

  message("\nPreprocessing steps applied:")
  message("- Removed features with all missing values")
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

  # Return results
  return(data.frame(clinical_processed_dnn))
}
