import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from trees.model import choose_bins, apply_bins

def test_choose_bins():
  X = np.array([
    [0.0, 'b'],
    [2.0, 'b'],
    [1.0, 'a'],
    [5.0, 'a'],
  ], dtype=object)
  actual = choose_bins(X, bucket_count=2)
  assert actual.shape == (2, 1)
  assert_array_almost_equal(actual[0], [1.0])
  assert_array_equal(actual[1], ['a'])

def test_apply_bins():
  bins = np.zeros((1, 255), dtype=int) + 255
  bins[0, 0] = 1
  bins[0, 1] = 3
  X = np.array([
    [0],
    [1],
    [2],
    [3],
    [4]
  ])
  actual = apply_bins(X, bins)
  assert_array_equal(actual[:, 0], [0, 0, 1, 1, 2])