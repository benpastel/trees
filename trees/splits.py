from dataclasses import dataclass
from typing import Optional

import numpy as np

from trees.params import Params
from trees.c.split import split as c_choose_split

@dataclass
class Split:
  column: int
  value: int
  score: float


def choose_split(
    X: np.ndarray,
    y: np.ndarray,
    parent_split_score: float,
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
  assert params.extra_leaf_penalty >= 0

  if rows < params.min_leaf_size * 2:
    # we need at least min_leaf_size rows in both left child and right child
    return None

  max_split_score = parent_split_score - params.extra_leaf_penalty

  out = c_choose_split(X, y, max_split_score, params.min_leaf_size)
  if out is None:
    return None

  split = Split(*out)
  split.score += params.extra_leaf_penalty
  return split


