setwd(normalizePath(dirname(R.utils::commandArgs(asValues=TRUE)$"f")))
source("../../../scripts/h2o-r-test-setup.R")

# Test PCA on australia.csv
test.pca.pubdev3827 <- function() {
  Log.info("Importing australia.csv data...\n")
  ausPath = system.file("extdata", "australia.csv", package="h2o")
  australia.hex = h2o.importFile(path = ausPath)
  aus = as.data.frame(australia.hex)

  
  ranks = 3
  transform_chosen = "STANDARDIZE"  # R only support this transform.  Other transform options tested in Python already.
  Log.info("Building PCA model using R...")
  pcaR = prcomp(aus, center=TRUE, scale.=TRUE, rank.=ranks)
  
  Log.info("Building PCA model with GramSVD...")
  gramSVD.pca <- h2o.prcomp(training_frame = australia.hex, transform=transform_chosen, k=3)
  sh_output = gramSVD.pca@model$scoring_history   # length(sh_output) is 0 when outputs NULL
  im_output = gramSVD.pca@model$importance
  
  Log.info("Comparing R and GramSVD PCA models...")
  browser()
  isFlipped1 <- checkPCAModel(gramSVD.pca, pcaR, tolerance=1e-6, compare_all_importance=FALSE)

  Log.info("Building PCA model with Randomized...") 
  randomized.pca <- h2o.prcomp(training_frame = australia.hex, transform=transform_chosen, k=3,
  pca_method="Randomized", compute_metrics=TRUE)
  randomized.pca@model$scoring_history
  randomized.pca@model$importance
  Log.info("Comparing R and Randomized PCA models...")
  isFlipped1 <- checkPCAModel(randomized.pca, pcaR, tolerance=1e-6, compare_all_importance=FALSE)
  
  
  Log.info("Building PCA model with Power...") 
  power.pca <- h2o.prcomp(training_frame = australia.hex, transform=transform_chosen, k=3,
                               pca_method="Power", compute_metrics=TRUE)
  
  Log.info("Comparing R and Power PCA models...")
  isFlipped1 <- checkPCAModel(power.pca, pcaR, tolerance=1e-6, compare_all_importance=FALSE)
  
  Log.info("Building PCA model with GLRM...") 
  glrm.pca <- h2o.prcomp(training_frame = australia.hex, transform=transform_chosen, k=3,
                               pca_method="GLRM", compute_metrics=TRUE, use_all_factor_levels=TRUE)
  
  Log.info("Comparing R and GLRM PCA models...")
  isFlipped1 <- checkPCAModel(GLRM.pca, pcaR, tolerance=1e-6, compare_all_importance=FALSE)
}

doTest("PCA Test: PUBDEV-3827: Scoring history and importance of components", test.pca.pubdev3827)
