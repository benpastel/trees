from dataclasses import dataclass
from typing import Optional

import numpy as np


MIN_LEAF_SIZE = 100


@dataclass 
class Split:
  column: int
  value: float
  left_mask: np.ndarray
  right_mask: np.ndarray

  def __str__(self):
    return (f'col: {self.column}, val: {self.value}, ' 
      + f'left_count: {np.count_nonzero(self.left_mask)}, '
      + f'right_count: {np.count_nonzero(self.right_mask)}')


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


def choose_split(
    mask: np.ndarray,
    X: np.ndarray, 
    y: np.ndarray, 
    feat_order: np.ndarray
) -> Optional[Split]:
  assert mask.dtype == np.bool
  assert mask.ndim == 1
  assert len(mask) == len(y)

  total = np.count_nonzero(mask)
  if total < MIN_LEAF_SIZE * 2:
    # we need at least MIN_LEAF_SIZE points in both left child and right child
    return None

  min_impurity = gini_impurity(y[mask])
  if min_impurity < 0.0000000001:
    # already perfect
    return None

  # exhaustively try every split and take the best one
  best_col = None
  best_val = None

  left_mask = np.zeros(X.shape[0], dtype=bool)
  right_mask = np.zeros(X.shape[0], dtype=bool)

  for col in range(X.shape[1]):

    # check every unique value in the simplest way
    # ignore the pre-sorted orders for now
    for val in np.unique(X[mask, col]):

      left_mask[:] = 0
      right_mask[:] = 0
      left_mask[mask] = (X[mask, col] <= val)
      right_mask[mask] = (X[mask, col] > val)

      left_count = np.count_nonzero(left_mask)
      right_count = np.count_nonzero(right_mask)

      if (left_count < MIN_LEAF_SIZE) or (right_count < MIN_LEAF_SIZE):
        continue

      impurity = gini_impurity(y[left_mask]) + gini_impurity(y[right_mask])

      if impurity < min_impurity:
        min_impurity = impurity
        best_col = col 
        best_val = val

  if best_col is None:
    # couldn't decrease impurity by splitting
    return None

  assert best_val is not None # convince mypy
  left_mask[:] = 0
  right_mask[:] = 0
  left_mask[mask] = (X[mask, best_col] <= best_val)
  right_mask[mask] = (X[mask, best_col] > best_val)
  split = Split(best_col, best_val, left_mask, right_mask)
  print(f'best split: {split}.  impurity: {impurity:.4f} => {min_impurity:.4f}')
  return split


