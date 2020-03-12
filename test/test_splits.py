import numpy as np

from ..src.splits import gini_impurity


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

  X = np.array([
    [0],
    [0],

  ])

  y = np.array([
    True, 
    True,


  ])


  