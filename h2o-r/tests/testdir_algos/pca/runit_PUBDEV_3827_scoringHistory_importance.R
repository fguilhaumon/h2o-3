setwd(normalizePath(dirname(R.utils::commandArgs(asValues=TRUE)$"f")))
source("../../../scripts/h2o-r-test-setup.R")

# Test PCA on australia.csv
test.pca.pubdev3827 <- function() {
  Log.info("Importing australia.csv data...\n")

  ausPath = system.file("extdata", "australia.csv", package="h2o")
  australia.hex = h2o.importFile(path = ausPath)
  australia.pca <- h2o.prcomp(training_frame = australia.hex, transform="STANDARDIZE", k=3)
  sh_output = australia.pca@model$scoring_history   # length(sh_output) is 0 when outputs NULL
  im_output = australia.pca@model$importance

  
  australia2.pca <- h2o.prcomp(training_frame = australia.hex, transform="STANDARDIZE", k=3,
  pca_method="Randomized", compute_metrics=TRUE)
  australia2.pca@model$scoring_history
  australia2.pca@model$importance
}

doTest("PCA Test: PUBDEV-3827: Scoring history and importance of components", test.pca.pubdev3827)
