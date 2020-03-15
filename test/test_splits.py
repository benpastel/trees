import numpy as np

from trees.splits import gini_impurity, choose_split


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


def test_choose_split():
  # this case is perfectly split by 2nd column
  X = np.array([
    [0, 0],
    [1, 2],
    [1, 0],
    [1, 1],
  ])
  y = np.array([
    True,
    False,
    True,
    False,
  ])
  split = choose_split(
    np.arange(4, dtype=np.intp),
    X,
    y,
    min_leaf_size=1
  )
  assert split is not None
  assert split.column == 1
  assert split.value == 0


  