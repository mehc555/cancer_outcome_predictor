# R/modules/training/cv_engine.R
run_nested_cv <- function(data, config) {
  # Set up parallel backend
  future::plan(future::multiprocess)
  
  # Outer repeats
  results <- future_map(1:config$cv_params$outer_repeats, function(repeat_idx) {
    # Create validation split
    # Run nested CV
    # Return results
  })
  
  # Aggregate results
  aggregate_cv_results(results)
}
