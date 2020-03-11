import numpy as np

from ..splits import gini_purity


def test_gini_purity():
  #   1 - gini_impurity
  # = 1 - (p_no * (1 - p_no) + p_yes * (1 - p_yes))
  # = 1 - (2/5 * 3/5 + 3/5 * 2/5)
  # = 1 - 12/25
  # = 13/25
  actual = gini_purity(np.array([
    True,
    False,
    True,
    True,
    False
  ]))
  expected = 13.0 / 25.0
  assert abs(actual - expected) < 0.000001
  