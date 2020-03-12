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


SPLITS_TO_CONSIDER = 256
def choose_split(
    idx: np.ndarray,
    X: np.ndarray, 
    y: np.ndarray, 
    min_leaf_size: int
) -> Optional[Split]:
  assert idx.dtype == np.intp
  assert idx.ndim == 1

  if len(idx) < min_leaf_size * 2:
    # we need at least MIN_LEAF_SIZE points in both left child and right child
    return None

  orig_impurity = gini_impurity(y[idx])
  if orig_impurity < 0.0000000001:
    # already perfect
    return None

  min_impurity = orig_impurity
  best_split = None

  for col in range(X.shape[1]):

    # try up to SPLITS_TO_CONSIDER unique values
    vals = X[idx, col]
    uniqs = np.unique(vals)
    stride = 1 + len(uniqs) // SPLITS_TO_CONSIDER

    for val in uniqs[::stride]:
      left_idx = idx[vals <= val]
      right_idx = idx[vals > val]

      if len(left_idx) < min_leaf_size or len(right_idx) < min_leaf_size:
        continue

      impurity = gini_impurity(y[left_idx]) + gini_impurity(y[right_idx])

      if impurity < min_impurity:
        min_impurity = impurity
        best_split = Split(col, val, left_idx, right_idx)

  if best_split is None:
    # couldn't decrease impurity by splitting
    return None

  return best_split


