from dataclasses import dataclass
from typing import Optional, List

import numpy as np
from trees.params import Params
from trees.tree import Tree, fit_tree, eval_tree


@dataclass
class Model:
  trees: List[Tree]
  bucket_splits: List[np.ndarray]
  float_targets: bool
  mean: float

  def __str__(self, verbose = False):
    model_type = "Regression" if self.float_targets else "Classification"
    s = f'{model_type} model with {len(self.trees)} trees'
    if verbose:
      s += ':\n'
      s += '\n'.join(str(t) for t in self.trees)
    return s


BUCKET_COUNT = 256 # uint8 buckets
def choose_bucket_splits(
    X: np.ndarray,
    bucket_count=BUCKET_COUNT
) -> List[np.ndarray]:
  # returns a list of bins for each column
  # where bins is an array of right inclusive endpoints:
  #   (values in bucket 0) <= bins[0] < (values in bucket 1) <= bins[1] ...
  #
  # so that np.searchsorted(bins, vals) returns the buckets
  splits: List[np.ndarray] = []

  for col in range(X.shape[1]):
    uniqs = np.unique(X[:,col])

    if len(uniqs) <= bucket_count:
      # each value gets a bin
      bins = uniqs
    else:
      # assign a roughly equal amount of unique values to each bin
      # TODO weight by count
      bins = np.zeros(bucket_count, uniqs.dtype)
      for s in range(bucket_count):
        # average step by len(uniqs) // bucket_count
        #
        # right endpoint is inclusive, so -1
        #
        # so start with (s + 1) * (len(uniqs)) // bucket_count) - 1
        # but then divide by bucket count last so that the extras a distributed evenly
        idx = ((s+1) * len(uniqs)) // bucket_count - 1
        bins[s] = uniqs[idx]
    splits.append(bins)
  return splits


def apply_bucket_splits(
  X: np.ndarray,
  splits: List[np.ndarray]
) -> np.ndarray:
  ''' returns X bucketed into the splits '''
  assert X.ndim == 2
  assert X.shape[1] == len(splits)

  bucketed = np.zeros(X.shape, dtype=np.uint8)
  for col, bins in enumerate(splits):

    # TODO: decide how to handle different dtypes
    assert X.dtype == bins.dtype
    indices = np.searchsorted(bins, X[:, col])

    # in the training case, indices should always be < len(buckets)
    # in the prediction case, it's possible to see a new value outside the right endpoint
    # include those in the rightmost bucket
    bucketed[:, col] = np.minimum(len(bins) - 1, indices)
  return bucketed


def fit(
    X: np.ndarray,
    y: np.ndarray,
    params: Params
) -> Model:
  assert X.ndim == 2
  assert y.shape == (X.shape[0],)

  bucket_splits = choose_bucket_splits(X)
  X = apply_bucket_splits(X, bucket_splits)
  assert X.dtype == np.uint8

  float_targets = (y.dtype != np.bool)

  preds = np.mean(y)

  trees = []
  for t in range(params.tree_count):
    loss_gradient = preds - y
    target = -params.learning_rate * loss_gradient

    tree, new_preds = fit_tree(X, target, params)
    trees.append(tree)
    preds += new_preds

  return Model(trees, bucket_splits, float_targets, np.mean(y))


def predict(model: Model, X: np.ndarray) -> np.ndarray:
  assert X.ndim == 2
  X = apply_bucket_splits(X, model.bucket_splits)

  values = np.zeros(len(X)) + model.mean

  for tree in model.trees:
    values += eval_tree(tree, X)

  if model.float_targets:
    return values
  else:
    # we only handle binary classification so far
    # so the values are just probabilities
    return values >= 0.5

