# R/modules/helper_functions/misc.R

#' Check sample consistency across data modalities
#' @param data_files Named list of file paths for each data type
#' @param cancer_type Current cancer type being processed
#' @return List of data frames with consistent samples
validate_sample_consistency <- function(data_files, cancer_type) {
    # Read all data files
    data_list <- list()

    # Read each data type and extract sample IDs
    if (file.exists(data_files$mirna)) {
        data_list$mirna <- read_tsv(data_files$mirna, show_col_types = FALSE)
        data_list$mirna_samples <- data_list$mirna$sample_id
    }

    if (file.exists(data_files$methylation)) {
        data_list$methylation <- read_tsv(data_files$methylation, show_col_types = FALSE)
        data_list$methylation_samples <- data_list$methylation$sample
    }

    if (file.exists(data_files$mutations)) {
        data_list$mutations <- read_tsv(data_files$mutations, show_col_types = FALSE)
        data_list$mutations_samples <- data_list$mutations$sample
    }

    if (file.exists(data_files$expression)) {
        data_list$expression <- read_tsv(data_files$expression, show_col_types = FALSE)
        data_list$expression_samples <- data_list$expression$sample_id
    }

    if (file.exists(data_files$clinical)) {
        data_list$clinical <- read_tsv(data_files$clinical, show_col_types = FALSE)
        data_list$clinical_samples <- data_list$clinical$sample_id
    }

    if (file.exists(data_files$cnv)) {
        data_list$cnv <- read_tsv(data_files$cnv, show_col_types = FALSE)
        data_list$cnv_samples <- data_list$cnv$Sample_ID
    }

    # Get all sample ID vectors
    sample_lists <- list()
    for (data_type in c("mirna", "methylation", "mutations", "expression", "clinical", "cnv")) {
        sample_col <- paste0(data_type, "_samples")
        if (!is.null(data_list[[sample_col]])) {
            sample_lists[[data_type]] <- data_list[[sample_col]]
        }
    }

    # Find common samples across all available data types
    common_samples <- Reduce(intersect, sample_lists)

    # Report statistics
    message(sprintf("\nSample consistency report for %s:", cancer_type))
    message(sprintf("Common samples across all modalities: %d", length(common_samples)))
    for (data_type in names(sample_lists)) {
        message(sprintf("%s samples: %d", data_type, length(sample_lists[[data_type]])))
    }

}

