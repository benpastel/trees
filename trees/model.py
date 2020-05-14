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
  mean: float

  def __str__(self, verbose = False):
    model_type = "Regression" if self.targets_are_float else "Classification"
    sizes = [t.node_count for t in self.trees]
    s = (f'{model_type} model with {len(self.trees)} trees with sizes '
      + f'min={min(sizes)} max={max(sizes)} mean={np.mean(sizes)}')
    if verbose:
      s += ':\n'
      s += '\n'.join(str(t) for t in self.trees)
    return s


def choose_bins(
    X: np.ndarray,
    bucket_count: int,
    sample_count: int
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

  if rows > sample_count:
    sample = np.random.randint(rows, size=sample_count, dtype=np.intp)
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
  splits = bins.shape[1]
  assert bins.ndim == 2
  assert bins.shape[0] == cols
  assert X.dtype == np.float32
  assert bins.dtype == np.float32

  bin_X = np.zeros((rows, cols), dtype=np.uint8)

  c_apply_bins(X, bins, bin_X)

  assert np.all(bin_X < splits+1)

  return bin_X


def fit(
    X: np.ndarray,
    y: np.ndarray,
    params: Params
) -> Model:
  rows, feats = X.shape
  assert y.shape == (rows,)
  assert 2 <= params.bucket_count <= 256
  assert 0 < params.bucket_sample_count
  assert 0 < params.trees_per_bucketing
  assert 0 < params.row_sample <= 1.0
  targets_are_float = (y.dtype != np.bool)

  X = X.astype(np.float32, copy=False)
  y = y.astype(np.double, copy=False)
  mean_y = np.mean(y)
  preds = np.full(rows, mean_y, dtype=np.double)

  bins = None
  bin_X = None

  trees = []
  for t in range(params.tree_count):
    if (t % params.trees_per_bucketing) == 0:
      bins = choose_bins(X, params.bucket_count, params.bucket_sample_count)
      bin_X = apply_bins(X, bins)

    in_sample = (np.random.rand(rows) <= params.row_sample)

    target = params.learning_rate * (y[in_sample] - preds[in_sample])

    tree, new_preds = fit_tree(bin_X[in_sample], target, bins, params)

    if tree.node_count == 1 and len(trees) > 1 and trees[-1].node_count == 1:
      # 2 trees with 1 node in a row
      # don't add the 2nd one, and stop early
      return Model(trees, targets_are_float, mean_y), preds

    trees.append(tree)
    preds[in_sample] += new_preds

  return Model(trees, targets_are_float, mean_y), preds


def predict(model: Model, X: np.ndarray) -> np.ndarray:
  assert X.ndim == 2
  X = X.astype(np.float32, copy=False)

  values = np.full(len(X), model.mean, dtype=np.double)

  for tree in model.trees:
    values += eval_tree(tree, X)

  if model.targets_are_float:
    return values
  else:
    # we only handle binary classification so far
    # so the values are just probabilities
    return values >= 0.5

