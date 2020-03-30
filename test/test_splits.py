import numpy as np

from numpy.testing import assert_array_equal, assert_array_almost_equal

from trees.tree import fit_tree, eval_tree
from trees.params import Params


def test_trees():
  # expect a single split:
  # col 1 <= val 1
  X = np.array([
    [0, 2],
    [1, 6],
    [1, 2],
    [1, 7],
  ], dtype=np.uint8)
  y = np.array([1, 0, 1, 0])
  tree = fit_tree(
    X,
    y,
    Params(min_leaf_size=1, extra_leaf_penalty=0.0)
  )
  assert tree.node_count == 3
  assert_array_equals(tree.left_children, [1, 0, 0])
  assert_array_equals(tree.right_children, [2, 0, 0])
  assert_array_equals(tree.split_cols, [1, 0, 0])
  assert_array_equals(tree.split_vals, [2, 0, 0])
  assert_array_almost_equal(tree.node_means, [0.0, 1.0, 0.0])

  # # this case has y ordered [0, 2, 2, 2, 9, 7] by X[:, 0]
  # # so the best split is on value 2 into [0, 2, 2, 2], [9, 7]
  # X = np.array([4, 1, 5, 0, 1, 2], dtype=np.uint8).reshape((-1, 1))
  # y = np.array([9, 2, 7, 0, 2, 2])

  # split = choose_split(
  #   X,
  #   y,
  #   np.var(y),
  #   Params(min_leaf_size=1, extra_leaf_penalty=0.0)
  # )
  # assert split is not None
  # assert split.column == 0
  # assert split.value == 2

  # # same case, but at min_leaf_size = 3
  # # we're forced to take value 1
  # split = choose_split(
  #   X,
  #   y,
  #   np.var(y),
  #   Params(min_leaf_size=3, extra_leaf_penalty=0.0)
  # )
  # assert split is not None
  # assert split.column == 0
  # assert split.value == 1

  # # at super high extra leaf penalty we never split
  # split = choose_split(
  #   X,
  #   y,
  #   np.var(y),
  #   Params(min_leaf_size=1, extra_leaf_penalty=10.0)
  # )
  # assert split is None

