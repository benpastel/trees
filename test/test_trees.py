import numpy as np

from numpy.testing import assert_array_equal, assert_array_almost_equal

from trees.tree import fit_tree, eval_tree
from trees.params import Params


def test_fit_tree():
  # expect a single split:
  # col 1 <= val 2
  X = np.array([
    [0, 2],
    [1, 6],
    [1, 2],
    [1, 7],
  ], dtype=np.uint8)
  y = np.array([1, 0, 1, 0])
  tree, preds = fit_tree(
    X,
    y,
    Params(min_leaf_size=1, extra_leaf_penalty=0.0)
  )
  print(tree)
  assert tree.node_count == 3
  assert_array_equal(tree.left_children, [1, 0, 0])
  assert_array_equal(tree.right_children, [2, 0, 0])
  assert_array_equal(tree.split_cols, [1, 0, 0])
  assert_array_equal(tree.split_vals, [2, 0, 0])
  assert_array_almost_equal(tree.node_means, [0.5, 1.0, 0.0])

  #   <=5
  #    |  \
  #    |   \
  #   <=2  [100, 100, 100]
  #    |  \
  #    |   \
  #    |  [9, 7]
  #    |
  # [0, 2, 2, 2]
  #
  X = np.array([4, 1, 5, 0, 1, 2, 100, 100, 100], dtype=np.uint8).reshape((-1, 1))
  y = np.array([9, 2, 7, 0, 2, 2, 100, 100, 100])

  tree, preds = fit_tree(
    X,
    y,
    Params(min_leaf_size=1, extra_leaf_penalty=2.0)
  )
  print(tree)
  assert tree.node_count == 5
  assert_array_equal(tree.left_children, [1, 3, 0, 0, 0])
  assert_array_equal(tree.right_children, [2, 4, 0, 0, 0])
  assert_array_equal(tree.split_cols, [0, 0, 0, 0, 0])
  assert_array_equal(tree.split_vals, [5, 2, 0, 0, 0])
  assert_array_almost_equal(tree.node_means, [322/9, 22/6, 100, 1.5, 8])

  # at min_leaf_size = 3, we're forced to take value 1 on the last split
  #   <=5
  #    |  \
  #    |   \
  #   <=1  [100, 100, 100]
  #    |  \
  #    |   \
  #    |  [2, 9, 7]
  #    |
  # [0, 2, 2]
  #
  tree, preds = fit_tree(
    X,
    y,
    Params(min_leaf_size=3, extra_leaf_penalty=2.0)
  )
  print(tree)
  assert tree.node_count == 5
  assert_array_equal(tree.left_children, [1, 3, 0, 0, 0])
  assert_array_equal(tree.right_children, [2, 4, 0, 0, 0])
  assert_array_equal(tree.split_cols, [0, 0, 0, 0, 0])
  assert_array_equal(tree.split_vals, [5, 1, 0, 0, 0])
  assert_array_almost_equal(tree.node_means, [322/9, 22/6, 100, 4/3, 6])

  # at high extra leaf penalty, only the first split is worth it
  #   <=5
  #    |  \
  #    |   \
  #    |  [100,100,100]
  #    |
  #[0,2,2,2,9,7]
  #
  tree, preds = fit_tree(
    X,
    y,
    Params(min_leaf_size=1, extra_leaf_penalty=10.0)
  )
  print(tree)
  assert tree.node_count == 3
  assert_array_equal(tree.left_children, [1, 0, 0])
  assert_array_equal(tree.right_children, [2, 0, 0])
  assert_array_equal(tree.split_cols, [0, 0, 0])
  assert_array_equal(tree.split_vals, [5, 0, 0])
  assert_array_almost_equal(tree.node_means, [322/9, 22/6, 100])

