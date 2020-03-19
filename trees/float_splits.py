from dataclasses import dataclass
from typing import Optional

import numpy as np

from trees.bool_splits import Split


def variance(A: np.ndarray):
  mean = np.mean(A)
  return np.mean(A * A) - (mean * mean)


EXTRA_LEAF_PENALTY = 0.0
def choose_float_split(
    idx: np.ndarray,
    X: np.ndarray, 
    y: np.ndarray, 
    min_leaf_size: int,
    extra_leaf_penalty: float = EXTRA_LEAF_PENALTY
) -> Optional[Split]:
  assert idx.dtype == np.intp
  assert idx.ndim == 1
  assert X.dtype == np.uint8

  if len(idx) < min_leaf_size * 2:
    # we need at least MIN_LEAF_SIZE points in both left child and right child
    return None

  min_impurity = variance(y[idx])
  best_split = None

  if min_impurity <= extra_leaf_penalty:
    # already perfect
    return None

  for col in range(X.shape[1]):
    vals = X[idx, col]
    order = np.argsort(vals)

    # left side is inclusive and right side is exclusive
    # so for leftward stats, drop the last item
    # for rightward stats, drop the first item
    # 
    # e.g.:
    # 
    # y[idx][order] = [0, 1, 2, 3]
    #         asc_y = [0, 1, 2]       
    #        desc_y = [3, 2, 1]
    # 
    asc_y = y[idx][order][:-1]
    desc_y = y[idx][order][-1:0:-1]

    left_totals = np.arange(len(asc_y)) + 1
    right_totals = left_totals[::-1]

    left_sums = np.cumsum(asc_y)
    right_sums = np.cumsum(desc_y)[::-1]
    left_sum_sqs = np.cumsum(asc_y * asc_y)
    right_sum_sqs = np.cumsum(desc_y * desc_y)[::-1]

    left_means = left_sums / left_totals
    right_means = right_sums / right_totals
    left_mean_sqs = left_sum_sqs / left_totals
    right_mean_sqs = right_sum_sqs / right_totals

    left_var = left_mean_sqs - (left_means * left_means) 
    right_var = right_mean_sqs - (right_means * right_means)

    scores = (left_var * left_totals + right_var * right_totals) / (2.0 * len(vals)) + extra_leaf_penalty

    # for left-inclusive splits, a valid place to split is one value before
    # the first time we see a unique value in the sorted array
    # 
    # e.g. 
    # ordered_vals: [2, 2, 2, 3, 3, 5]
    #     uniq_idx: [0, 3, 5]
    #    split_idx: [2, 4]
    # 
    ordered_vals = vals[order][:-1]
    _, uniq_idx = np.unique(ordered_vals, return_index=True)
    split_idx = uniq_idx[1:] - 1

    # TODO could use min_leaf_size start/stop offsets instead
    can_split = np.zeros(len(asc_y), dtype=bool)
    can_split[split_idx] = True
    can_split &= (left_totals >= min_leaf_size)
    can_split &= (right_totals >= min_leaf_size)
    if not np.any(can_split):
      continue

    impurity = np.min(scores)

    if impurity < min_impurity:
      min_impurity = impurity

      best_idx = np.argmin(scores[can_split])
      best_val = ordered_vals[can_split][best_idx]

      # TODO: should be possible to get left & right index by slicing w/ best_idx
      best_split = Split(col, best_val, idx[vals <= best_val], idx[vals > best_val])


  if best_split is None:
    # couldn't decrease impurity by splitting
    return None

  return best_split
