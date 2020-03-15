import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from trees.tree import choose_bucket_splits

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
