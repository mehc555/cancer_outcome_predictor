library(tidyverse)
library(httr)
library(jsonlite)

#' Fetch and cache cancer hotspots data
#' @param cache_dir Directory to store cached hotspot data
#' @return DataFrame of hotspot mutations
#' @keywords internal
fetch_hotspots <- function(cache_dir = "data/cache") {
  cache_file <- file.path(cache_dir, "cancer_hotspots.rds")
  
  # Check if cached data exists and is less than 1 month old
  if (file.exists(cache_file) &&
      difftime(Sys.time(), file.mtime(cache_file), units = "days") < 30) {
    return(readRDS(cache_file))
  }
  
  # Create cache directory if it doesn't exist
  if (!dir.exists(cache_dir)) {
    dir.create(cache_dir, recursive = TRUE)
  }
  
  # Fetch data from cancerhotspots.org
  message("Fetching hotspot data from cancerhotspots.org...")
  
  # Single residue hotspots
  response <- GET("https://www.cancerhotspots.org/api/hotspots/single")
  single_hotspots <- fromJSON(rawToChar(response$content), flatten = TRUE)
  
  # Process single hotspots data
  hotspots <- single_hotspots %>%
    as_tibble() %>%
    # Select base columns
    select(
      hugoSymbol,
      residue,
      tumorTypeCount,
      tumorCount,
      qValue,
      starts_with("variantAminoAcid.")
    ) %>%
    # Convert wide amino acid format to long
    pivot_longer(
      cols = starts_with("variantAminoAcid."),
      names_to = "variant",
      values_to = "count",
      names_prefix = "variantAminoAcid."
    ) %>%
    # Remove variants with no counts
    filter(!is.na(count) & count > 0) %>%
    # Clean up variant names
    mutate(
      variant = str_remove(variant, "^\\.|del$|dup$|ins.*$"),
      type = "single"
    ) %>%
    # Filter significant hotspots
    filter(qValue < 0.05) %>%
    # Create unique identifier for each hotspot
    mutate(
      hotspot_id = paste(hugoSymbol, residue, variant, sep = "_"),
      residue = as.character(residue)
    )
  
  # Cache the data
  saveRDS(hotspots, cache_file)
  
  message(sprintf("Found %d significant hotspot mutations across %d genes",
                 nrow(hotspots),
                 length(unique(hotspots$hugoSymbol))))
  
  # Print top hotspots for verification
  message("\nTop 5 hotspot mutations by tumor count:")
  hotspots %>%
    arrange(desc(tumorCount)) %>%
    select(hotspot_id, tumorCount, tumorTypeCount) %>%
    head(5) %>%
    print()
  
  return(hotspots)
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

#' Process mutation data for deep learning input
#' @param cancer_type Character string specifying cancer type
#' @param min_freq Minimum mutation frequency across samples to include gene (default: 0.01)
#' @param base_dir Base directory containing the data
#' @return List containing different mutation encoding matrices
#' @export
process_mutation_data <- function(cancer_type, min_freq = 0.01, 
                                base_dir = "data/GDC_TCGA") {
  # Construct file paths
  input_file <- file.path(base_dir, cancer_type, "mutations.tsv")
  processed_dir <- create_processed_dir(cancer_type, base_dir)
  
  if (!file.exists(input_file)) {
    stop(sprintf("Mutation data file not found for cancer type %s", cancer_type))
  }
  
  # Fetch hotspot data
  hotspots <- fetch_hotspots()
  
  # Read mutation data
  mutations <- read_tsv(input_file, show_col_types = FALSE)
  
  # Record initial dimensions
  initial_genes <- length(unique(mutations$gene))
  n_samples <- length(unique(mutations$sample))
  
  # Get unique samples and genes
  samples <- unique(mutations$sample)
  
  # Filter genes by mutation frequency
  gene_freq <- table(mutations$gene) / n_samples
  frequent_genes <- names(gene_freq)[gene_freq >= min_freq]
  
  # Create different encodings
  binary_matrix <- encode_binary_mutations(mutations, samples, frequent_genes)
  effect_matrix <- encode_effect_mutations(mutations, samples, frequent_genes)
  vaf_matrix <- encode_vaf_mutations(mutations, samples, frequent_genes)
  integrated_matrix <- encode_integrated_mutations(mutations, samples, frequent_genes)
  hotspot_matrix <- encode_hotspot_mutations(mutations, samples, frequent_genes, hotspots)
  
  # Create hybrid encoding
  hybrid_matrix <- create_hybrid_encoding(
    integrated_matrix,
    hotspot_matrix,
    hotspots
  )
  
  # Round numerical values to 3 decimal places
  matrices <- list(
    binary = binary_matrix,
    effect = effect_matrix,
    vaf = vaf_matrix,
    integrated = integrated_matrix,
    hotspot = hotspot_matrix,
    hybrid = hybrid_matrix
  )
  
  matrices <- lapply(matrices, function(mat) {
    mat %>% mutate(across(-sample, ~round(., 3)))
  })
  
  # Validate all matrices
  lapply(matrices, validate_mutation_data)
  
  # Save all versions
  file_suffixes <- c("binary", "effect", "vaf", "integrated", "hotspot", "hybrid")
  
  for (suffix in file_suffixes) {
    write_tsv(
      matrices[[suffix]],
      file.path(processed_dir, sprintf("mutations_processed_%s.tsv", suffix))
    )
  }
  
  # Print processing summary
  message(sprintf("Mutation processing summary for %s:", cancer_type))
  message(sprintf("- Initial number of genes: %d", initial_genes))
  message(sprintf("- Removed %d genes (frequency < %g%%)", 
                 initial_genes - length(frequent_genes), min_freq * 100))
  message(sprintf("- Retained %d genes", length(frequent_genes)))
  message(sprintf("- Number of samples: %d", n_samples))
  message(sprintf("- Number of hotspot mutations: %d", 
                 ncol(hotspot_matrix) - 1))  # -1 for sample column
  message("- Encodings generated:")
  message("  1. Binary (mutation presence/absence)")
  message("  2. Effect-based (functional impact weights)")
  message("  3. VAF-based (variant allele frequencies)")
  message("  4. Integrated (effect * VAF)")
  message("  5. Hotspot (position-specific mutations)")
  message("  6. Hybrid (integrated + hotspot features)")
  
  return(matrices)
}

#' Encode mutations using binary representation
#' @keywords internal
encode_binary_mutations <- function(mutations, samples, genes) {
  # Create sparse matrix of mutation presence/absence
  mutation_matrix <- matrix(0, nrow = length(samples), ncol = length(genes),
                          dimnames = list(samples, genes))
  
  # Fill matrix with 1s where mutations exist
  for (i in seq_len(nrow(mutations))) {
    if (mutations$gene[i] %in% genes) {
      mutation_matrix[mutations$sample[i], mutations$gene[i]] <- 1
    }
  }
  
  as_tibble(mutation_matrix, rownames = "sample")
}

#' Encode mutations using effect categories
#' @keywords internal
encode_effect_mutations <- function(mutations, samples, genes) {
  # Define effect categories and their weights
  effect_weights <- c(
    "frameshift" = 4,
    "nonsense" = 4,
    "splice" = 3,
    "missense" = 2,
    "synonymous" = 1,
    "other" = 1
  )
  
  # Create matrix
  mutation_matrix <- matrix(0, nrow = length(samples), ncol = length(genes),
                          dimnames = list(samples, genes))
  
  # Fill matrix with effect weights
  for (i in seq_len(nrow(mutations))) {
    if (mutations$gene[i] %in% genes) {
      effect <- mutations$effect[i]
      weight <- case_when(
        grepl("frameshift", effect) ~ effect_weights["frameshift"],
        grepl("stop_gained|nonsense", effect) ~ effect_weights["nonsense"],
        grepl("splice", effect) ~ effect_weights["splice"],
        grepl("missense", effect) ~ effect_weights["missense"],
        grepl("synonymous", effect) ~ effect_weights["synonymous"],
        TRUE ~ effect_weights["other"]
      )
      
      # Use maximum weight if multiple mutations exist
      current_weight <- mutation_matrix[mutations$sample[i], mutations$gene[i]]
      mutation_matrix[mutations$sample[i], mutations$gene[i]] <- max(current_weight, weight)
    }
  }
  
  as_tibble(mutation_matrix, rownames = "sample")
}

#' Encode mutations using variant allele frequency
#' @keywords internal
encode_vaf_mutations <- function(mutations, samples, genes) {
  # Create matrix
  mutation_matrix <- matrix(0, nrow = length(samples), ncol = length(genes),
                          dimnames = list(samples, genes))
  
  # Fill matrix with VAF values
  for (i in seq_len(nrow(mutations))) {
    if (mutations$gene[i] %in% genes) {
      # Use maximum VAF if multiple mutations exist
      current_vaf <- mutation_matrix[mutations$sample[i], mutations$gene[i]]
      mutation_matrix[mutations$sample[i], mutations$gene[i]] <- 
        max(current_vaf, mutations$dna_vaf[i])
    }
  }
  
  as_tibble(mutation_matrix, rownames = "sample")
}

#' Encode mutations using integrated approach (effect + VAF)
#' @keywords internal
encode_integrated_mutations <- function(mutations, samples, genes) {
  # Get effect-weighted matrix
  effect_matrix <- encode_effect_mutations(mutations, samples, genes)
  
  # Get VAF matrix
  vaf_matrix <- encode_vaf_mutations(mutations, samples, genes)
  
  # Combine matrices (multiply effect weight by VAF)
  mutation_matrix <- as.matrix(effect_matrix[-1]) * as.matrix(vaf_matrix[-1])
  
  # Convert to tibble and add sample column
  as_tibble(mutation_matrix) %>%
    bind_cols(sample = samples, .)
}

#' Encode hotspot mutations specifically
#' @keywords internal
encode_hotspot_mutations <- function(mutations, samples, genes, hotspots) {
  # Create matrix for hotspot mutations
  hotspot_features <- unique(hotspots$hotspot_id)
  mutation_matrix <- matrix(0, nrow = length(samples), 
                          ncol = length(hotspot_features),
                          dimnames = list(samples, hotspot_features))
  
  # Process each mutation
  for (i in seq_len(nrow(mutations))) {
    if (mutations$gene[i] %in% genes) {
      # Create potential hotspot ID
      hotspot_id <- paste(
        mutations$gene[i],
        sub("p\\.[A-Z]*(\\d+).*", "\\1", mutations$Amino_Acid_Change[i]),
        sub("p\\.[A-Z]*\\d+([A-Z]*)", "\\1", mutations$Amino_Acid_Change[i]),
        sep = "_"
      )
      
      # If this is a known hotspot, record its VAF
      if (hotspot_id %in% hotspot_features) {
        mutation_matrix[mutations$sample[i], hotspot_id] <- mutations$dna_vaf[i]
      }
    }
  }
  
  as_tibble(mutation_matrix, rownames = "sample")
}

#' Create hybrid encoding combining gene-level and hotspot features
#' @keywords internal
create_hybrid_encoding <- function(integrated_matrix, hotspot_matrix, hotspots) {
  # Get genes with hotspots
  hotspot_genes <- unique(hotspots$hugoSymbol)
  
  # Remove gene-level features for genes with hotspots
  gene_features <- integrated_matrix %>%
    select(-sample, -any_of(hotspot_genes))
  
  # Combine with hotspot features
  hybrid_matrix <- bind_cols(
    sample = integrated_matrix$sample,
    gene_features,
    select(hotspot_matrix, -sample)
  )
  
  return(hybrid_matrix)
}

#' Validate mutation data structure and content
#' @param mutation_data Processed mutation data frame
#' @return Logical indicating if validation passed (invisible)
#' @keywords internal
validate_mutation_data <- function(mutation_data) {
  # Check if data frame is empty
  if (nrow(mutation_data) == 0) {
    stop("No samples remained after processing")
  }
  
  # Check if sample column exists
  if (!"sample" %in% colnames(mutation_data)) {
    stop("Sample column not found in mutation data")
  }
  
  # Check for duplicate samples
  if (any(duplicated(mutation_data$sample))) {
    stop("Duplicate samples found in mutation data")
  }
  
  # Check if any non-numeric values in mutation columns
  non_numeric_check <- mutation_data %>%
    select(-sample) %>%
    sapply(function(x) all(is.numeric(x)))
  
  if (!all(non_numeric_check)) {
    stop("Non-numeric values found in mutation data")
  }
  
  invisible(TRUE)
}
