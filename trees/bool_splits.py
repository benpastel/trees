from dataclasses import dataclass
from typing import Optional

import numpy as np


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


def gini_impurity(A: np.ndarray) -> float:
  ''' 
  gini impurity in the boolean case:
    p_no * (1 - p_no) + p_yes * (1 - p_yes)
    2 * p_yes * p_no
    2 * (trues / total) * (falses / total)
    2 * trues * falses / total^2
  '''
  assert A.ndim == 1
  assert A.dtype == np.bool
  assert len(A) > 0

  trues = np.count_nonzero(A)
  falses = len(A) - trues
  return 2.0 * trues * falses / (len(A) * len(A))


EXTRA_LEAF_PENALTY = 0.0
def choose_bool_split(
    idx: np.ndarray,
    X: np.ndarray, 
    y: np.ndarray, 
    min_leaf_size: int,
    extra_leaf_penalty: float = EXTRA_LEAF_PENALTY
) -> Optional[Split]:
  assert idx.dtype == np.intp
  assert idx.ndim == 1
  assert X.dtype == np.uint8
  assert y.dtype == np.bool

  if len(idx) < min_leaf_size * 2:
    # we need at least MIN_LEAF_SIZE points in both left child and right child
    return None

  min_impurity = gini_impurity(y[idx])
  best_split = None

  if min_impurity <= extra_leaf_penalty:
    # already perfect
    return None

  for col in range(X.shape[1]):
    vals = X[idx, col]
    totals = np.bincount(vals)
    trues = np.bincount(vals[y[idx]], minlength=len(totals))
  
    left_totals = np.cumsum(totals)
    right_totals = np.cumsum(totals[::-1])[::-1]

    left_trues = np.cumsum(trues)
    right_trues = np.cumsum(trues[::-1])[::-1]

    for v in range(len(totals) - 1):
      # v is inclusive on left and exclusive on right, so add 1 to all right indices
      if totals[v] == 0 or left_totals[v] < min_leaf_size or right_totals[v+1] < min_leaf_size:
        continue

      left_false = left_totals[v] - left_trues[v]
      right_false = right_totals[v+1] - right_trues[v+1]

      gini_left = 2.0 * left_trues[v] * left_false / (left_totals[v] * left_totals[v])
      gini_right = 2.0 * right_trues[v+1] * right_false / (right_totals[v+1] * right_totals[v+1])

      impurity = (gini_left * left_totals[v] + gini_right * right_totals[v+1]) / (2.0 * totals[v]) + extra_leaf_penalty

      if impurity < min_impurity:
        min_impurity = impurity
        best_split = Split(col, v, idx[vals <= v], idx[vals > v])

  if best_split is None:
    # couldn't decrease impurity by splitting
    return None

  return best_split


