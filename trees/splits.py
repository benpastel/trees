from dataclasses import dataclass
from typing import Optional

import numpy as np

from trees.params import Params

@dataclass
class Split:
  column: int
  value: int


def variance(A: np.ndarray):
  mean = np.mean(A)
  return np.mean(A * A) - (mean * mean)


def choose_split(
    X: np.ndarray,
    y: np.ndarray,
    params: Params
) -> Optional[Split]:
  assert X.dtype == np.uint8
  assert y.ndim == 1

  if len(y) < params.min_leaf_size * 2:
    # we need at least min_leaf_size rows in both left child and right child
    return None

  orig_impurity = variance(y)
  best_split = None

  if orig_impurity <= params.extra_leaf_penalty:
    # cannot be improved by splitting
    return None

  # precalculate stats for each (feature, unique X value)
  counts = np.zeros((X.shape[1], 256), dtype=np.uint32)
  sums = np.zeros((X.shape[1], 256), dtype=np.float64)
  sum_sqs = np.zeros((X.shape[1], 256), dtype=np.float64)

  for f in range(X.shape[1]):
    vals = X[:, f]
    counts[f] = np.bincount(vals, minlength=256)
    sums[f] = np.bincount(vals, weights=y, minlength=256)
    sum_sqs[f] = np.bincount(vals, weights=(y * y), minlength=256)

  # choose the splitting value
  #
  # all of the vectorized ops are computed across the unique X values
  #
  # left side is inclusive and right side is exclusive
  # so for leftward cumulative stats, drop the last item
  # for rightward cumulative stats, drop the first item
  #
  # e.g. with only 1 feature:
  #
  #       counts = [[2, 0, 1, 3]]
  #  left_counts = [[2, 2, 3]]
  # right_counts = [[4, 4, 3]]
  #
  def left(A):
    return np.cumsum(A[:, :-1], axis=1)

  def right(A):
    return np.cumsum(A[:, -1:0:-1], axis=1)[:, ::-1]

  left_counts = left(counts)
  right_counts = right(counts)

  # only consider a value for splitting if:
  #   a training row has that value
  #   both sides of the split are large enough
  #
  # this also prevents division by 0 in the variance calculations
  ok = (counts[:,:-1] > 0) & (left_counts >= params.min_leaf_size) & (right_counts >= params.min_leaf_size)
  if not np.any(ok):
    return None

  # prevent dividing by 0
  # these scores will filtered by `ok` so they don't matter
  left_counts[left_counts == 0] = 1
  right_counts[right_counts == 0] = 1

  left_means = left(sums) / left_counts
  right_means = right(sums) / right_counts

  left_var = left(sum_sqs) / left_counts - (left_means * left_means)
  right_var = right(sum_sqs) / right_counts - (right_means * right_means)

  # weighted average of the variances.  TODO tweak
  scores = (left_var * left_counts + right_var * right_counts) / (2.0 * X.shape[0]) + params.extra_leaf_penalty

  # we can't filter by ok because it destroys the shape
  # so just set the scores too large to be used
  scores[~ok] = orig_impurity

  best_col, best_val = np.unravel_index(np.argmin(scores), scores.shape)

  assert scores[best_col, best_val] > -0.000000001, 'score should be positive, except rounding error'

  if scores[best_col, best_val] < orig_impurity:
    return Split(best_col, best_val)
  else:
    # couldn't decrease impurity by splitting
    return None

