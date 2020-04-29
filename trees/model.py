from dataclasses import dataclass
from typing import Optional, List

import numpy as np
from scipy.stats import chi2

from trees.params import Params
from trees.tree import Tree, fit_tree, eval_tree
from trees.c.tree import apply_bins as c_apply_bins
from trees.utils import timed


@dataclass
class Model:
  trees: List[Tree]
  targets_are_float: bool
  mean_y: float
  mean_xs: np.ndarray # TODO remove global X -= np.mean(X)

  def __str__(self, verbose = False):
    model_type = "Regression" if self.targets_are_float else "Classification"
    sizes = [t.node_count for t in self.trees]
    s = (f'{model_type} model with {len(self.trees)} trees with sizes '
      + f'min={min(sizes)} max={max(sizes)} mean={np.mean(sizes)}')
    if verbose:
      s += ':\n'
      s += '\n'.join(str(t) for t in self.trees)
    return s


BUCKET_COUNT = 256 # uint8 buckets
SAMPLE_COUNT = 256000
def choose_bins(
    X: np.ndarray,
    bucket_count=BUCKET_COUNT
) -> np.ndarray:
  # bins for each column
  # where bins is an array of right inclusive endpoints:
  #   (values in bucket 0) <= bins[0] < (values in bucket 1) <= bins[1] ...
  #
  #   ... bins[bucket_count-2] < (values in bucket bucket_count-1)
  #
  # the bins may include duplicates
  #
  rows, cols = X.shape

  if rows > SAMPLE_COUNT:
    sample = np.random.randint(rows, size=SAMPLE_COUNT, dtype=np.intp)
    X = X[sample, :]

  bins = np.zeros((cols, bucket_count-1), dtype=X.dtype)
  for c in range(cols):
    uniqs = np.unique(X[:, c])

    if len(uniqs) == 1:
      # print(f'Warning: feature {c} is always the same, with value {uniqs[0]}')
      continue

    # since splits are <= value, don't consider the highest value
    uniqs = uniqs[:-1]

    if len(uniqs) < bucket_count - 1:
      bins[c, :len(uniqs)] = uniqs
      bins[c, len(uniqs):] = uniqs[-1]
    else:
      # to make bucket_count buckets, there are bucket_count-1 separators
      # including both ends is bucket_count+1
      indices = np.linspace(0, len(uniqs), endpoint=False, num=bucket_count+1, dtype=np.intp)

      # now throw away both ends and keep just the separators
      indices = indices[1:-1]
      assert len(indices) == bucket_count-1

      bins[c] = uniqs[indices]

  assert bins.shape == (cols, bucket_count-1)
  assert bins.dtype == X.dtype

  # check bins are nondecreasing within each feature
  # by comparing all elements (skip the last) to their right neighbor (skip the first)
  assert np.all(bins[:,:-1] <= bins[:,1:])

  return bins


def apply_bins(
  X: np.ndarray,
  bins: np.ndarray
) -> np.ndarray:
  ''' returns X bucketed into the splits '''
  rows, cols = X.shape
  assert bins.shape == (cols, BUCKET_COUNT-1)
  assert X.dtype == np.float32
  assert bins.dtype == np.float32

  binned_X = np.zeros((cols, rows), dtype=np.uint8)

  c_apply_bins(X, bins, binned_X)

  return binned_X


def fit(
    X: np.ndarray,
    y: np.ndarray,
    params: Params
) -> Model:
  rows, feats = X.shape
  assert y.shape == (rows,)
  targets_are_float = (y.dtype != np.bool)

  X = X.astype(np.float32, copy=True) # TODO reduce copies
  mean_xs = np.mean(X, axis=0)
  X -= mean_xs
  XT_reg = X.T.copy()
  bins = choose_bins(X)
  XT_bin = apply_bins(X, bins)

  y = y.astype(np.double, copy=False)
  mean_y = np.mean(y)
  preds = np.full(rows, mean_y, dtype=np.double)

  assert XT_bin.dtype == np.uint8
  assert XT_reg.dtype == np.float32
  assert XT_bin.shape == XT_reg.shape

  trees = []
  for t in range(params.tree_count):
    target = params.learning_rate * (y - preds)

    tree, new_preds = fit_tree(XT_bin, XT_reg, target, bins, params)
    trees.append(tree)

    preds += new_preds

  return Model(trees, targets_are_float, mean_y, mean_xs), preds


def predict(model: Model, X: np.ndarray) -> np.ndarray:
  assert X.ndim == 2
  X = X.astype(np.float32, copy=False) - model.mean_xs

  values = np.full(len(X), model.mean_y, dtype=np.double)

  for tree in model.trees:
    values += eval_tree(tree, X)

  if model.targets_are_float:
    return values
  else:
    # we only handle binary classification so far
    # so the values are just probabilities
    return values >= 0.5

