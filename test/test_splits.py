import numpy as np

from trees.bool_splits import gini_impurity, choose_bool_split
from trees.float_splits import choose_float_split


def test_gini_impurity():
  #   p_no * (1 - p_no) + p_yes * (1 - p_yes)
  # = 2/5 * 3/5 + 3/5 * 2/5
  # = 12/25
  actual = gini_impurity(np.array([
    True,
    False,
    True,
    True,
    False
  ]))
  expected = 12.0 / 25.0
  assert abs(actual - expected) < 0.000001


def test_choose_bool_split():
  # this case is perfectly split by 2nd column
  X = np.array([
    [0, 0],
    [1, 2],
    [1, 0],
    [1, 1],
  ], dtype=np.uint8)
  y = np.array([
    True,
    False,
    True,
    False,
  ])
  split = choose_bool_split(
    np.arange(4, dtype=np.intp),
    X,
    y,
    min_leaf_size=1
  )
  assert split is not None
  assert split.column == 1
  assert split.value == 0


def test_choose_float_split():
  # this case is perfectly split by 2nd column
  X = np.array([
    [0, 0],
    [1, 2],
    [1, 0],
    [1, 1],
  ], dtype=np.uint8)
  y = np.array([1, 0, 1, 0])
  split = choose_float_split(
    np.arange(4, dtype=np.intp),
    X,
    y,
    min_leaf_size=1
  )
  assert split is not None
  assert split.column == 1
  assert split.value == 0

  # this case has y ordered [0, 2, 2, 2, 9, 7] by X[:, 0]
  # so the best split is on value 2 into [0, 2, 2, 2], [9, 7]
  X = np.array([4, 1, 5, 0, 1, 2], dtype=np.uint8).reshape((-1, 1))
  y = np.array([9, 2, 7, 0, 2, 2])

  split = choose_float_split(
    np.arange(6, dtype=np.intp),
    X,
    y,
    min_leaf_size=1
  )
  assert split is not None
  assert split.column == 0
  assert split.value == 2

  # same case, but at min_leaf_size = 3
  # we're forced to take value 1 
  split = choose_float_split(
    np.arange(6, dtype=np.intp),
    X,
    y,
    min_leaf_size=3
  )
  assert split is not None
  assert split.column == 0
  assert split.value == 1
  