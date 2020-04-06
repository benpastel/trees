import numpy as np
from scipy.stats import chi2

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
    Params()
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
  #    |  [100, 100, 100]
  #    |
  # [0, 2, 2, 2, 9, 7]
  #
  X = np.array([4, 1, 5, 0, 1, 2, 100, 100, 100], dtype=np.uint8).reshape((-1, 1))
  y = np.array([9, 2, 7, 0, 2, 2, 100, 100, 100])
  tree, preds = fit_tree(
    X,
    y,
    Params(smooth_factor=1.0)
  )
  print(tree)
  assert tree.node_count == 3
  # assert_array_equal(tree.left_children, [1, 0, 0])
  # assert_array_equal(tree.right_children, [2, 0, 0])
  # assert_array_equal(tree.split_cols, [0, 0, 0])
  # assert_array_equal(tree.split_vals, [5, 0, 0])
  # assert_array_almost_equal(tree.node_means, [322/9, 22/6, 100])

  #   <=5
  #    |  \
  #    |   \
  #   <=2  [20, 20, 20]
  #    |  \
  #    |   \
  #    |  [9, 7]
  #    |
  # [0, 2, 2, 2]
  #
  X = np.array([4, 1, 5, 0, 1, 2, 100, 100, 100], dtype=np.uint8).reshape((-1, 1))
  y = np.array([9, 2, 7, 0, 2, 2, 20, 20, 20])
  tree, preds = fit_tree(
    X,
    y,
    Params(smooth_factor=1.0)
  )
  print(tree)
  assert tree.node_count == 5
  assert_array_equal(tree.left_children, [1, 3, 0, 0, 0])
  assert_array_equal(tree.right_children, [2, 4, 0, 0, 0])
  assert_array_equal(tree.split_cols, [0, 0, 0, 0, 0])
  assert_array_equal(tree.split_vals, [5, 2, 0, 0, 0])
  assert_array_almost_equal(tree.node_means, [82/9, 22/6, 20, 1.5, 8])

  #            <=5
  #           /  \
  #          /    \
  #        <=2   [20, 20, 20]
  #        /  \
  #       /    \
  #      /      \
  #     /        \
  #   <=0         <=4
  #   /  \        / \
  # [0] [2,2]   [9] [7]
  #
  X = np.array([4, 1, 5, 0, 1, 2, 100, 100, 100], dtype=np.uint8).reshape((-1, 1))
  y = np.array([9, 2, 7, 0, 2, 2, 20, 20, 20])
  tree, preds = fit_tree(
    X,
    y,
    Params(smooth_factor=0.0)
  )
  print(tree)
  assert tree.node_count == 9
  assert_array_equal(tree.left_children,  [1, 3, 0, 5, 7, 0, 0, 0, 0])
  assert_array_equal(tree.right_children, [2, 4, 0, 6, 8, 0, 0, 0, 0])
  assert_array_equal(tree.split_cols,     [0, 0, 0, 0, 0, 0, 0, 0, 0])
  assert_array_equal(tree.split_vals,     [5, 2, 0, 0, 4, 0, 0, 0, 0])
  assert_array_almost_equal(tree.node_means, [82/9, 22/6, 20, 1.5, 8, 0, 2, 9, 7])

