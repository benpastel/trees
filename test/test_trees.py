import numpy as np
import pprint

from numpy.testing import assert_array_equal, assert_array_almost_equal

from trees.tree import fit_tree, eval_tree
from trees.params import Params

# TODO test eval


def test_fit_tree():
  pp = pprint.PrettyPrinter(indent=4)

  # test X values & columns are tracked properly
  #
  # perfect split on 2nd column
  #
  #      (1, 3)
  #     /   |   \
  # [1,1] [2,2] [0,0]
  #
  X = np.array([
    [0, 2],
    [1, 1],
    [2, 4],
    [3, 3],
    [4, 0],
    [5, 5],
  ], dtype=np.uint8)
  y = np.array([2, 1, 0, 2, 1, 0], dtype=np.double)
  tree = fit_tree(
    X,
    y,
    Params(smooth_factor=1.0)
  )
  pp.pprint(tree.__dict__)
  assert tree.node_count == 4
  assert_array_equal(tree.left_children,     [1, 0, 0, 0])
  assert_array_equal(tree.mid_children,      [2, 0, 0, 0])
  assert_array_equal(tree.right_children,    [3, 0, 0, 0])
  assert_array_equal(tree.split_cols,        [1, 0, 0, 0])
  assert_array_equal(tree.split_lo_vals,     [1, 0, 0, 0])
  assert_array_equal(tree.split_hi_vals,     [3, 0, 0, 0])
  assert_array_almost_equal(tree.node_means, [1, 1, 2, 0])

  # from here on use 1 column X, X = y, for simplicity
  #            (3,9)
  #           /  |  \
  #         /    |    \
  #       /      |      \
  # [0,2,2,3]   [9,7]  [100,100,100]
  #
  X = np.array([9, 2, 7, 0, 2, 100, 3, 100, 100], dtype=np.uint8).reshape((-1, 1))
  y = np.array([9, 2, 7, 0, 2, 100, 3, 100, 100], dtype=np.double)
  tree = fit_tree(
    X,
    y,
    Params(smooth_factor=1.0)
  )
  pp.pprint(tree.__dict__)
  assert tree.node_count == 4
  assert_array_equal(tree.left_children,  [1, 0, 0, 0])
  assert_array_equal(tree.mid_children,   [2, 0, 0, 0])
  assert_array_equal(tree.right_children, [3, 0, 0, 0])
  assert_array_equal(tree.split_cols,     [0, 0, 0, 0])
  assert_array_equal(tree.split_lo_vals,  [3, 0, 0, 0])
  assert_array_equal(tree.split_hi_vals,  [9, 0, 0, 0])
  assert_array_almost_equal(tree.node_means, [np.mean(y), 7/4, 8, 100])

  #             (3, 9)
  #             /  |  \
  #           /    |    \
  #         /      |      \
  #      (0,2)   [7, 9]  [20,20,20]
  #     /  |  \
  #  [0] [2,2] [3]
  #
  X = np.array([9, 2, 7, 0, 2, 20, 3, 20, 20], dtype=np.uint8).reshape((-1, 1))
  y = np.array([9, 2, 7, 0, 2, 20, 3, 20, 20], dtype=np.double)
  tree = fit_tree(
    X,
    y,
    Params(smooth_factor=0.0)
  )
  pp.pprint(tree.__dict__)
  assert tree.node_count == 7
  assert_array_equal(tree.left_children,  [1, 4, 0, 0, 0, 0, 0])
  assert_array_equal(tree.mid_children,   [2, 5, 0, 0, 0, 0, 0])
  assert_array_equal(tree.right_children, [3, 6, 0, 0, 0, 0, 0])
  assert_array_equal(tree.split_cols,     [0, 0, 0, 0, 0, 0, 0])
  assert_array_equal(tree.split_lo_vals,  [3, 0, 0, 0, 0, 0, 0])
  assert_array_equal(tree.split_hi_vals,  [9, 2, 0, 0, 0, 0, 0])
  assert_array_almost_equal(tree.node_means, [np.mean(y), 7/4, 8, 20, 0, 2, 3])

