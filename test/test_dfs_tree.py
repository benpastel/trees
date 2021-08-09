import numpy as np
import pprint

from numpy.testing import assert_array_equal, assert_array_almost_equal

from trees.dfs_tree import fit_tree, eval_tree, Tree, variance
from trees.params import Params


def test_variance():

  assert variance(8, 4, 2) == 0

def test_fit_tree_simple():
  pp = pprint.PrettyPrinter(indent=4)

  # perfect split on 2nd column
  #
  #    (c=1, v=2)
  #     /      \
  # {2,2}     {1,1}
  #
  X = np.array([
    [0, 2],
    [1, 4],
    [2, 1],
    [3, 3],
  ], dtype=np.uint8)
  y = np.array([2, 1, 2, 1], dtype=np.double)
  bins = np.array([
    np.arange(255, dtype=np.float32),
    np.arange(255, dtype=np.float32)
  ])
  # TODO include nontrivial bins

  tree, preds = fit_tree(
    X,
    y,
    bins,
    Params(bucket_count=256)
  )
  pp.pprint(tree.__dict__)
  assert tree.node_count == 3
  assert_array_equal(tree.left_children,  [ 1, 0, 0])
  assert_array_equal(tree.right_children, [ 2, 0, 0])
  assert_array_equal(tree.split_cols[0],  1)
  assert_array_almost_equal(tree.split_vals[0], 2)
  assert_array_almost_equal(tree.node_means, [ 0, 2, 1])
  assert_array_almost_equal(preds, [2, 1, 2, 1])


def test_fit_tree_order():
  pp = pprint.PrettyPrinter(indent=4)
  #
  #         n=0: (c=0,v=9)
  #             /     \
  #  n=1: (c=0, v=3)   \
  #       /     |       \
  # {0,2,2,3}  {9,7}   {100,100,100}
  #    n=3      n=4       n=2
  #
  X = np.array([9, 2, 7, 0, 2, 100, 3, 100, 100], dtype=np.uint8).reshape((-1, 1))
  y = np.array([9, 2, 7, 0, 2, 100, 3, 100, 100], dtype=np.double)
  bins = np.arange(255, dtype=np.float32).reshape((1, -1))
  tree, preds = fit_tree(
    X,
    y,
    bins,
    Params(max_nodes=5, bucket_count=256)
  )
  pp.pprint(tree.__dict__)
  assert tree.node_count == 5
  assert_array_equal(tree.left_children,     [1, 3, 0, 0, 0])
  assert_array_equal(tree.right_children,    [2, 4, 0, 0, 0])
  assert_array_equal(tree.split_cols,        [0, 0, 0, 0, 0])
  assert_array_almost_equal(tree.split_vals, [9, 0, 3, 0, 0])
  assert_array_almost_equal(tree.node_means, [0, 0, 100, 7/4, 8])
  assert_array_almost_equal(preds, [8, 7/4, 8, 7/4, 7/4, 100, 7/4, 100, 100])


def test_eval_tree():
  #     n=0: (c=1: v=100)
  #       /      \
  #      /        \
  #    mean=1     (c=0: v=10)
  #              /        \
  #            mean=3     mean=2
  #
  tree = Tree(
    node_count = 5,
    split_cols = np.array([1,   0,  0, 0, 0], dtype=np.uint64),
    split_vals = np.array([100, 0, 10, 0, 0], dtype=np.float32),
    left_children  = np.array([1, 0, 3, 0, 0], dtype=np.uint16),
    right_children = np.array([2, 0, 4, 0, 0], dtype=np.uint16),
    node_means     = np.array([0, 1, 0, 3, 2], dtype=np.double),
  )

  X = np.array([
    [10, 100],
    [10, 101],
    [11, 101],
  ], dtype=np.float32)
  assert_array_almost_equal(
    eval_tree(tree, X),
    np.array([
      1,
      3,
      2,
    ]))

