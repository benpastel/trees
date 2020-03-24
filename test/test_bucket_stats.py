import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from trees.c.bucket_stats import bucket_stats

def test_bucket_stats():
  X = np.array([
    [0, 3],
    [1, 3],
    [2, 3]
  ], dtype=np.uint8)
  y = np.array([1.0, 1.0, 10.0])

  counts = np.zeros((X.shape[1], 256), dtype=np.uint32)
  sums = np.zeros((X.shape[1], 256))
  sum_sqs = np.zeros((X.shape[1], 256))

  bucket_stats(X, y, counts, sums, sum_sqs)

  assert_array_equal(counts[0, 0:4], [1, 1, 1, 0])
  assert_array_equal(counts[1, 0:4], [0, 0, 0, 3])
  assert np.all(counts[:, 4:] == 0)

  assert_array_almost_equal(sums[0, 0:4], [1.0, 1.0, 10.0,  0.0])
  assert_array_almost_equal(sums[1, 0:4], [0.0, 0.0,  0.0, 12.0])
  assert np.all(sums[:, 4:] == 0.0)

  assert_array_almost_equal(sum_sqs[0, 0:4], [1.0, 1.0, 100.0,   0.0])
  assert_array_almost_equal(sum_sqs[1, 0:4], [0.0, 0.0,   0.0, 102.0])
  assert np.all(sum_sqs[:, 4:] == 0.0)