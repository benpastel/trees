import numpy as np
import pprint

from numpy.testing import assert_array_equal, assert_array_almost_equal

from trees.tree import fit_tree, eval_tree, Tree
from trees.params import Params


def test_fit_tree():
  pp = pprint.PrettyPrinter(indent=4)

  # case: perfect split on 2nd column
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
  ])
  y = np.array([2, 1, 0, 2, 1, 0], dtype=np.double)
  bins = np.array([
    np.arange(255, dtype=np.float32),
    10.0 * np.arange(255, dtype=np.float32)
  ])

  tree, preds = fit_tree(
    X.T.astype(np.uint8),
    X.T.astype(np.float32),
    y,
    bins,
    Params(smooth_factor=1.0)
  )
  pp.pprint(tree.__dict__)
  assert tree.node_count == 4
  assert_array_equal(tree.left_children,        [ 1, 0, 0, 0])
  assert_array_equal(tree.mid_children,         [ 2, 0, 0, 0])
  assert_array_equal(tree.right_children,       [ 3, 0, 0, 0])
  assert_array_equal(tree.split_cols,           [ 1, 0, 0, 0])
  assert_array_almost_equal(tree.split_lo_vals, [10, 0, 0, 0])
  assert_array_almost_equal(tree.split_hi_vals, [30, 0, 0, 0])
  assert_array_almost_equal(tree.coefs,         [ 0, 0, 0, 0])
  assert_array_almost_equal(tree.intercepts,   [ 1, 0, 1, -1])
  assert_array_almost_equal(preds, y)

  # case: perfect regression
  # y = 2x + 10
  # TODO remove global X -= np.mean(X)
  X = np.array([9, 2, 7, 0, 2, 100, -3, 100, 100]).reshape((-1, 1))
  y = np.array([28, 14, 24, 10, 14, 210, 4, 210, 210], dtype=np.double) + 10
  bins = np.arange(255, dtype=np.float32).reshape((1, -1))
  tree, preds = fit_tree(
    X.T.astype(np.uint8),
    X.T.astype(np.float32),
    y,
    bins,
    Params(smooth_factor=1.0)
  )
  pp.pprint(tree.__dict__)
  assert tree.node_count == 2
  assert_array_equal(tree.left_children,         [1, 0])
  assert_array_equal(tree.mid_children,          [0, 0])
  assert_array_equal(tree.right_children,        [0, 0])
  assert_array_equal(tree.split_cols,            [1, 0])
  assert_array_almost_equal(tree.split_lo_vals,  [0, 0])
  assert_array_almost_equal(tree.split_hi_vals,  [0, 0])
  assert_array_almost_equal(tree.coefs,          [2, 0])
  assert_array_almost_equal(tree.intercepts,     [10, 0])
  assert_array_almost_equal(preds, y)

  # case: smooth_factor prevents splitting further
  #
  #            (3,7)
  #           /  |  \
  #         /    |    \
  #       /      |      \
  # [0,2,2,3]   [6,7]  [100,100,100,100]
  #
  X = np.array([7, 2, 6, 0, 2, 50, 3, 50, 150, 150]).reshape((-1, 1))
  y = np.array([7, 2, 6, 0, 2, 100, 3, 100, 100, 100], dtype=np.double)
  bins = np.arange(255, dtype=np.float32).reshape((1, -1))
  tree, preds = fit_tree(
    X.T.astype(np.uint8),
    X.T.astype(np.float32),
    y,
    bins,
    Params(smooth_factor=1.0)
  )
  pp.pprint(tree.__dict__)
  assert tree.node_count == 4
  assert_array_equal(tree.left_children,         [1, 0, 0, 0])
  assert_array_equal(tree.mid_children,          [2, 0, 0, 0])
  assert_array_equal(tree.right_children,        [3, 0, 0, 0])
  assert_array_equal(tree.split_cols,            [0, 0, 0, 0])
  assert_array_almost_equal(tree.split_lo_vals,  [3, 0, 0, 0])
  assert_array_almost_equal(tree.split_hi_vals,  [7, 0, 0, 0])
  assert_array_almost_equal(tree.coefs,          [0, 0, 0, 0])
  assert_array_almost_equal(tree.intercepts, [42, -40.25, -35.5, 58])
  assert_array_almost_equal(preds, [6.5, 1.75, 6.5, 1.75, 1.75, 100, 1.75, 100, 100, 100])

  # case: mixed splitting & regression
  #
  #              +10
  #             (4, 7)
  #          /     |    \
  #        /       |      \
  #      /         |        \
  #   +1x-10    -3.5    [20,20,20,20]
  #     |        /  |   \
  # [0,1,2,4]   6 empty  7
  #
  X = np.array([8, 4, 6, 0, 1, 15, 2, 15, 25, 25]).reshape((-1, 1))
  y = np.array([7, 4, 6, 0, 1, 20, 2, 20, 20, 20], dtype=np.double)
  bins = np.arange(255, dtype=np.float32).reshape((1, -1))
  tree, preds = fit_tree(
    X.T.astype(np.uint8),
    X.T.astype(np.float32),
    y,
    bins,
    Params(smooth_factor=0.0)
  )
  pp.pprint(tree.__dict__)
  assert tree.node_count == 8
  assert_array_equal(tree.left_children,         [1, 4, 5, 0, 0, 0, 0, 0])
  assert_array_equal(tree.mid_children,          [2, 0, 6, 0, 0, 0, 0, 0])
  assert_array_equal(tree.right_children,        [3, 0, 7, 0, 0, 0, 0, 0])
  assert_array_equal(tree.split_cols,            [0, 0, 0, 0, 0, 0, 0, 0])
  assert_array_almost_equal(tree.split_lo_vals,  [4, 0, 5, 0, 0, 0, 0, 0])
  assert_array_almost_equal(tree.split_hi_vals,  [7, 0, 5, 0, 0, 0, 0, 0])
  assert_array_almost_equal(tree.coefs,          [0, 1, 0, 0, 0, 0, 0, 0])
  assert_array_almost_equal(tree.intercepts, [10, -10, -3.5, 10, -1.75, 0.25, 1.25])
  assert_array_almost_equal(preds, y)


def test_eval_tree():
  #         +2
  #     x_1: 1,100
  #    /    |      \
  #   /     |       \
  #  +1     -2     +2x_1
  #    x_0: 10,10     \
  #      /   |   \     \
  #    +3   +0   +2    +9
  #
  tree = Tree(
    node_count = 8,
    split_cols     = np.array([1,   0,  0,  1, 0, 0, 0, 0], dtype=np.uint64),
    split_lo_vals  = np.array([1,   0, 10,  0, 0, 0, 0, 0], dtype=np.float32),
    split_hi_vals  = np.array([100, 0, 10,  0, 0, 0, 0, 0], dtype=np.float32),
    coefs          = np.array([0,   0,  0,  2, 0, 0, 0, 0], dtype=np.float32),
    left_children  = np.array([1,   0,  4,  7, 0, 0, 0, 0], dtype=np.uint16),
    mid_children   = np.array([2,   0,  5,  0, 0, 0, 0, 0], dtype=np.uint16),
    right_children = np.array([3,   0,  6,  0, 0, 0, 0, 0], dtype=np.uint16),
    intercepts     = np.array([2,   1,  -2, 0, 3, 0, 2, 9], dtype=np.float32),
  )

  X = np.array([
    [10, 10],
    [11, 10],
    [10, 1],
    [10, 101],
  ], dtype=np.float32)
  assert_array_almost_equal(
    eval_tree(tree, X),
    np.array([
      3,
      2,
      3,
      213
    ]))





