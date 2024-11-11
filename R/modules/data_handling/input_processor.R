# R/modules/data_handling/input_processor.R

library(UCSCXenaTools)
library(dplyr)

download_multimodal_data <- function(cancer_types = c("BRCA", "COAD", "LUAD"), 
                                   base_dir = "data/GDC_TCGA") {
  
  # Define common data types based on dataset patterns
  data_patterns <- c(
    clinical = "clinical\\.tsv$",
    expression_counts = "star_counts\\.tsv$",
    expression_fpkm = "star_fpkm\\.tsv$",
    expression_fpkm_uq = "star_fpkm-uq\\.tsv$",
    expression_tpm = "star_tpm\\.tsv$",
    mutations = "somaticmutation_wxs\\.tsv$",
    methylation = "methylation27\\.tsv$",
    mirna = "mirna\\.tsv$",
    cnv_gene = "gene-level_ascat3\\.tsv$",
    cnv_allele = "allele_cnv_ascat3\\.tsv$"
  )
  
  # Create base directory if it doesn't exist
  if (!dir.exists(base_dir)) {
    dir.create(base_dir, recursive = TRUE)
  }
  
  # Function to create cancer type directory
  create_cancer_dir <- function(cancer_type) {
    cancer_dir <- file.path(base_dir, cancer_type)
    if (!dir.exists(cancer_dir)) {
      dir.create(cancer_dir, recursive = TRUE)
    }
    return(cancer_dir)
  }
  
  # Process each cancer type
  for (cancer in cancer_types) {
    message(sprintf("\nProcessing %s data...", cancer))
    cancer_dir <- create_cancer_dir(cancer)
    
    # Build dataset pattern for this cancer
    cancer_patterns <- paste0("TCGA-", cancer, ".", 
                            "(.*(", paste(data_patterns, collapse = "|"), "))")
    
    # Query all datasets
    xe <- XenaGenerate(subset = XenaHostNames == "gdcHub" & 
                      grepl(paste0("GDC TCGA.*", cancer), XenaCohorts)) %>%
      XenaFilter(filterDatasets = cancer_patterns)
    
    # Print available datasets before download
    message("Available datasets for ", cancer, ":")
    print(attributes(xe)$datasets)
    
    # Download data
    query <- XenaQuery(xe)
    XenaDownload(query, destdir = cancer_dir, trans_slash = TRUE)
    
    # Prepare clinical data specifically
    clinical_file <- file.path(cancer_dir, 
                             paste0("TCGA-", cancer, ".clinical.tsv.gz"))
    
    if (file.exists(clinical_file)) {
      clinical_df <- XenaPrepare(clinical_file)
      write.table(clinical_df,
                 file = file.path(cancer_dir, "clinical.tsv"),
                 sep = "\t",
                 row.names = FALSE,
                 col.names = TRUE,
                 quote = FALSE)
    }
    
    # Prepare and save other data types
    data_patterns=gsub("\\$","",data_patterns)

    for (data_type in names(data_patterns)) {
      pattern <- data_patterns[data_type]
      matching_files <- list.files(cancer_dir, pattern = pattern, 
                                 recursive = TRUE, full.names = TRUE)
      
      if (length(matching_files) > 0) {
        data_df <- XenaPrepare(matching_files[1])  # Take first match if multiple
        
        output_file <- file.path(cancer_dir, paste0(data_type, ".tsv"))
        write.table(data_df,
                   file = output_file,
                   sep = "\t",
                   row.names = FALSE,
                   col.names = TRUE,
                   quote = FALSE)
        
        message(sprintf("Processed %s data", data_type))
      } else {
        message(sprintf("Warning: No data found for %s", data_type))
      }
    }
  }
}

# Function to verify downloaded data
verify_data <- function(cancer_types = c("BRCA", "COAD", "LUAD"), 
                       base_dir = "data/GDC_TCGA") {
  expected_files <- paste0(c("clinical", "expression_counts", "expression_fpkm", 
                           "expression_fpkm_uq", "expression_tpm", "mutations",
                           "methylation", "mirna", "cnv_gene", "cnv_allele"), 
                           ".tsv")
  
  for (cancer in cancer_types) {
    cancer_dir <- file.path(base_dir, cancer)
    existing_files <- list.files(cancer_dir, pattern = "\\.tsv$")
    missing_files <- setdiff(expected_files, existing_files)
    
    message(sprintf("\nVerifying %s data:", cancer))
    message("Found files: ", paste(existing_files, collapse = ", "))
    if (length(missing_files) > 0) {
      message("Missing files: ", paste(missing_files, collapse = ", "))
    } else {
      message("All expected files present")
    }
    
    # Print dimensions and first few column names of each dataset
    for (file in existing_files) {
      data <- read.delim(file.path(cancer_dir, file))
      message(sprintf("\n%s:", file))
      message(sprintf("Dimensions: %d rows x %d columns", 
                     nrow(data), ncol(data)))
      message("First few columns: ", 
              paste(head(colnames(data), 5), collapse = ", "))
    }
  }
}

# Function to summarize data availability
summarize_availability <- function(cancer_types = c("BRCA", "COAD", "LUAD"),
                                 base_dir = "data/GDC_TCGA") {
  data_types <- c("clinical", "expression_counts", "expression_fpkm", 
                  "expression_fpkm_uq", "expression_tpm", "mutations",
                  "methylation", "mirna", "cnv_gene", "cnv_allele")
  
  availability <- matrix(FALSE, nrow = length(cancer_types), 
                        ncol = length(data_types),
                        dimnames = list(cancer_types, data_types))
  
  for (cancer in cancer_types) {
    cancer_dir <- file.path(base_dir, cancer)
    files <- list.files(cancer_dir, pattern = "\\.tsv$")
    files <- sub("\\.tsv$", "", files)
    availability[cancer, ] <- data_types %in% files
  }
  
  return(availability)
}

# Execution

download_tcga_data <- function() {
    cancer_types <- c("BRCA", "COAD", "LUAD")
    download_multimodal_data(cancer_types)
    verify_data(cancer_types)
    
    # Print availability summary
    message("\nData availability summary:")
    print(summarize_availability(cancer_types))
}

# Only run if script is run directly (not sourced)
if (sys.nframe() == 0) {
    download_tcga_data()
}
