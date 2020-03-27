import numpy as np

from trees.splits import choose_split
from trees.params import Params


# TODO add test case with EXACT output score
def test_choose_split():
  # this case is perfectly split by 2nd column
  X = np.array([
    [0, 0],
    [1, 2],
    [1, 0],
    [1, 1],
  ], dtype=np.uint8)
  y = np.array([1, 0, 1, 0])
  split = choose_split(
    X,
    y,
    100,
    Params(min_leaf_size=1, extra_leaf_penalty=0.0)
  )
  assert split is not None
  assert split.column == 1
  assert split.value == 0

  # this case has y ordered [0, 2, 2, 2, 9, 7] by X[:, 0]
  # so the best split is on value 2 into [0, 2, 2, 2], [9, 7]
  X = np.array([4, 1, 5, 0, 1, 2], dtype=np.uint8).reshape((-1, 1))
  y = np.array([9, 2, 7, 0, 2, 2])

  split = choose_split(
    X,
    y,
    np.var(y),
    Params(min_leaf_size=1, extra_leaf_penalty=0.0)
  )
  assert split is not None
  assert split.column == 0
  assert split.value == 2

  # same case, but at min_leaf_size = 3
  # we're forced to take value 1
  split = choose_split(
    X,
    y,
    np.var(y),
    Params(min_leaf_size=3, extra_leaf_penalty=0.0)
  )
  assert split is not None
  assert split.column == 0
  assert split.value == 1

  # at super high extra leaf penalty we never split
  split = choose_split(
    X,
    y,
    np.var(y),
    Params(min_leaf_size=1, extra_leaf_penalty=10.0)
  )
  assert split is None

  # at super low parent score we never split
  split = choose_split(
    X,
    y,
    0.1,
    Params(min_leaf_size=1, extra_leaf_penalty=0.0)
  )
  assert split is None
