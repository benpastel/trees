from dataclasses import dataclass
from typing import Optional

import numpy as np

from trees.params import Params

@dataclass 
class Split:
  column: int
  value: float
  left_idx: np.ndarray
  right_idx: np.ndarray

  def __str__(self):
    return (f'col: {self.column}, val: {self.value}, ' 
      + f'left_count: {len(self.left_idx)}, '
      + f'right_count: {len(self.right_idx)}')


def variance(A: np.ndarray):
  mean = np.mean(A)
  return np.mean(A * A) - (mean * mean)


def choose_split(
    idx: np.ndarray,
    X: np.ndarray, 
    y: np.ndarray,
    params: Params
) -> Optional[Split]:
  assert idx.dtype == np.intp
  assert idx.ndim == 1
  assert X.dtype == np.uint8

  if len(idx) < params.min_leaf_size * 2:
    # we need at least MIN_LEAF_SIZE points in both left child and right child
    return None

  min_impurity = variance(y[idx])
  best_split = None

  if min_impurity <= params.extra_leaf_penalty:
    # already perfect
    return None

  for col in range(X.shape[1]):
    vals = X[idx, col]
    ok_y = y[idx]

    # aggregate statistics on each unique X value
    counts = np.bincount(vals)
    sums = np.bincount(vals, weights=ok_y)
    sum_sqs = np.bincount(vals, weights=(ok_y * ok_y))

    # choose the splitting value  
    # 
    # all of the vectorized ops are over the unique X values
    # 
    # left side is inclusive and right side is exclusive
    # so for leftward stats, drop the last item
    # for rightward stats, drop the first item
    # 
    # e.g.:
    # 
    #       counts = [2, 0, 1, 3]
    #  left_counts = [2, 2, 3]       
    # right_counts = [4, 4, 3]
    # 
    left_counts = np.cumsum(counts[:-1])
    left_sums = np.cumsum(sums[:-1])
    left_sum_sqs = np.cumsum(sum_sqs[:-1])
    left_means = left_sums / left_counts
    left_mean_sqs = left_sum_sqs / left_counts
    left_var = left_mean_sqs - (left_means * left_means) 

    right_counts = np.cumsum(counts[-1:0:-1])[::-1]
    right_sums = np.cumsum(sums[-1:0:-1])[::-1]
    right_sum_sqs = np.cumsum(sum_sqs[-1:0:-1])[::-1]
    right_means = right_sums / right_counts
    right_mean_sqs = right_sum_sqs / right_counts
    right_var = right_mean_sqs - (right_means * right_means)

    scores = (left_var * left_counts + right_var * right_counts) / (2.0 * len(vals)) + params.extra_leaf_penalty

    can_split = (counts[:-1] > 0) & (left_counts >= params.min_leaf_size) & (right_counts >= params.min_leaf_size)

    if not np.any(can_split):
      continue

    impurity = np.min(scores[can_split])

    if impurity < min_impurity:
      min_impurity = impurity

      uniq_vals = np.arange(len(counts) - 1)
      best_val = uniq_vals[can_split][np.argmin(scores[can_split])]

      # TODO: delay the idx calculation to the end?
      best_split = Split(col, best_val, idx[vals <= best_val], idx[vals > best_val])


  if best_split is None:
    # couldn't decrease impurity by splitting
    return None

  return best_split
