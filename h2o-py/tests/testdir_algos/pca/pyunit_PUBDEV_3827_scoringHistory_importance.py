from __future__ import print_function
from builtins import range
from random import randint
import sys
sys.path.insert(1,"../../../")
import h2o
from tests import pyunit_utils
from h2o.transforms.decomposition import H2OPCA

def pca_scoring_history_importance():
  """
  This test aims to check and make sure PCA returns the scoring history and importance which are
  reported missing for certain PCA mode.
  """
  transform_types = ["NONE", "STANDARDIZE", "NORMALIZE", "DEMEAN", "DESCALE"]
  transformN = transform_types[randint(0, len(transform_types)-1)]

  print("Importing australia.csv data...\n")
  australia = h2o.upload_file(pyunit_utils.locate("smalldata/extdata/australia.csv"))
  col_indices = list(range(0, australia.ncol))

  # checking out PCA with GramSVD
  print("@@@@@@  Building PCA with GramSVD...\n")
  gramSVD = H2OPCA(k=3, transform=transformN)
  gramSVD.train(x=col_indices, training_frame=australia)

  # check PCA with PCA set to Randomized
  print("@@@@@@  Building PCA with Randomized...\n")
  randomizedPCA = H2OPCA(k=3, transform=transformN, pca_method="Randomized", compute_metrics=True, seed=12345)
  randomizedPCA.train(x=col_indices, training_frame=australia)

  # compare singular values and stuff with GramSVD
  print("@@@@@@  Comparing eigenvalues, eigenvectors between GramSVD and Randomized...\n")
  assert pyunit_utils.equal_H2OTwoDimTable(gramSVD._model_json["output"]["importance"],
                                    randomizedPCA._model_json["output"]["importance"],
                                    ["Standard deviation", "Cumulative Proportion", "Cumulative Proportion"],
                                           tolerance=1e-5), "Randomized Eigenvalues are not comparable to GramSVD."
  # compare singular vectors
  assert pyunit_utils.equal_H2OTwoDimTable(gramSVD._model_json["output"]["eigenvectors"],
                                    randomizedPCA._model_json["output"]["eigenvectors"],
                                    randomizedPCA._model_json["output"]["names"], tolerance=1e-2),\
      "Randomized Eigenvectors are not comparable to GramSVD."

  # check PCA with PCA set to Power
  print("@@@@@@  Building PCA with Power...\n")
  powerPCA = H2OPCA(k=3, transform=transformN, pca_method="Power", compute_metrics=True)
  powerPCA.train(x=col_indices, training_frame=australia)

  # compare singular values and stuff with GramSVD
  print("@@@@@@  Comparing eigenvalues, eigenvectors between GramSVD and Power...\n")
  assert pyunit_utils.equal_H2OTwoDimTable(gramSVD._model_json["output"]["importance"],
                                           powerPCA._model_json["output"]["importance"],
                                           ["Standard deviation", "Cumulative Proportion", "Cumulative Proportion"]), \
      "Power Eigenvalues are not comparable to GramSVD."
  # compare singular vectors
  assert pyunit_utils.equal_H2OTwoDimTable(gramSVD._model_json["output"]["eigenvectors"],
                                           powerPCA._model_json["output"]["eigenvectors"],
                                           powerPCA._model_json["output"]["names"], tolerance=1e-5),\
      "Power Eigenvectors are not comparable to GramSVD."

  # check PCA with PCA set to GLRM
  print("@@@@@@  Building PCA with GLRM...\n")
  glrmPCA = H2OPCA(k=3, transform=transformN, pca_method="GLRM", compute_metrics=True, use_all_factor_levels=True,
                   seed=12345)
  glrmPCA.train(x=col_indices, training_frame=australia)

  # compare singular values and stuff with GramSVD
  print("@@@@@@  Comparing eigenvalues, eigenvectors between GramSVD and GLRM...\n")
  assert pyunit_utils.equal_H2OTwoDimTable(gramSVD._model_json["output"]["importance"],
                                    glrmPCA._model_json["output"]["importance"],
                                           ["Standard deviation", "Cumulative Proportion", "Cumulative Proportion"],
                                           tolerance=1e-3), "GLRM Eigenvalues are not comparable to GramSVD."
  # compare singular vectors
  assert pyunit_utils.equal_H2OTwoDimTable(gramSVD._model_json["output"]["eigenvectors"],
                                    glrmPCA._model_json["output"]["eigenvectors"],
                                    glrmPCA._model_json["output"]["names"], tolerance=1e-2), \
      "GLRM Eigenvectors are not comparable to GramSVD."



if __name__ == "__main__":
  pyunit_utils.standalone_test(pca_scoring_history_importance)
else:
  pca_scoring_history_importance()
