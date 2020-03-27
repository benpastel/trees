from dataclasses import dataclass
from typing import Optional

import numpy as np

from trees.params import Params
from trees.c.split import split as c_choose_split

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
  # check preconditions of the C function here
  # TODO also assert X,y are well-aligned C-style or whatever
  assert X.dtype == np.uint8
  rows, cols = X.shape
  assert y.shape == (rows,)
  assert rows > 0
  assert cols > 0
  assert params.min_leaf_size > 0

  if rows < params.min_leaf_size * 2:
    # we need at least min_leaf_size rows in both left child and right child
    return None

  orig_impurity = variance(y)
  best_split = None

  max_split_score = orig_impurity - params.extra_leaf_penalty;

  if max_split_score <= 0:
    # cannot be improved by splitting
    return None

  # TODO just return the tuple
  out = c_choose_split(X, y, max_split_score, params.min_leaf_size)
  return None if out is None else Split(out[0], out[1])



