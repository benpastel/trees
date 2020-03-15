import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from trees.tree import choose_bucket_splits, apply_bucket_splits

def test_choose_bucket_splits():
  X = np.array([
    [0.0, 'b'],
    [2.0, 'b'],
    [1.0, 'a'],
    [5.0, 'a'],
  ], dtype=object)
  actual = choose_bucket_splits(X, bucket_count=2)
  assert len(actual) == 2
  assert_array_almost_equal(actual[0], [1.0, 5.0])
  assert_array_equal(actual[1], ['a', 'b'])

  # check that the bins work as expected with searchsorted 
  bins = actual[0]
  new_vals = np.arange(7)
  bucketed = np.searchsorted(bins, new_vals)
  assert_array_equal(bucketed, [0, 0, 1, 1, 1, 1, 2])

def test_apply_bucket_splits():
  splits = [np.array([1, 3])]
  X = np.array([
    [0],
    [1],
    [2],
    [3],
    [4]
  ])
  actual = apply_bucket_splits(X, splits)
  assert_array_equal(actual[:, 0], [0, 0, 1, 1, 1])