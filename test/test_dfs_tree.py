import numpy as np
import pprint

from numpy.testing import assert_array_equal, assert_array_almost_equal

from trees.dfs_tree import fit_tree, eval_tree, Tree
from trees.params import Params

from trees.c.dfs_tree import (
  update_node_splits as c_update_node_splits,
  copy_smaller as c_copy_smaller,
)

from test.python_stubs import update_node_splits as py_update_node_splits, variance

def test_c_update_nodes():
  max_nodes = 2
  cols = 2
  vals = 256

  hist_counts = np.random.randint(10, size=(max_nodes, cols, vals), dtype=np.uint32)
  hist_sums = np.random.random(size=(max_nodes, cols, vals)).astype(np.float64)
  hist_sum_sqs = np.random.random(size=(max_nodes, cols, vals)).astype(np.float64)

  py_node_gains = np.full(max_nodes, -np.inf, dtype=np.float64)
  py_split_cols = np.zeros(max_nodes, dtype=np.uint64)
  py_split_bins = np.zeros(max_nodes, dtype=np.uint8)

  c_node_gains = np.full(max_nodes, -np.inf, dtype=np.float64)
  c_split_cols = np.zeros(max_nodes, dtype=np.uint64)
  c_split_bins = np.zeros(max_nodes, dtype=np.uint8)

  py_update_node_splits(hist_counts, hist_sums, hist_sum_sqs, py_node_gains, py_split_cols, py_split_bins, 0)
  c_update_node_splits(hist_counts, hist_sums, hist_sum_sqs, c_node_gains, c_split_cols, c_split_bins, 0)

  assert_array_almost_equal(py_node_gains, c_node_gains)
  assert_array_equal(py_split_cols, c_split_cols)
  assert_array_equal(py_split_bins, c_split_bins)


def test_c_copy_smaller():
  # TODO: also big test against stub to check multithreaded version

  # inputs
  split_c = 1
  split_val = 4
  is_left = True;
  parent_X = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8],
  ], dtype = np.uint8)
  parent_y = np.array([
    10,
    11,
    12,
    13,
  ], dtype = np.double)
  parent_indices = np.array([
    5,
    6,
    7,
    8,
  ], dtype = np.uint64);

  # input & output
  parent_is_removed = np.array([
    True,
    False,
    False,
    False,
  ], dtype = np.bool)

  # outputs
  child_X = np.zeros((1, 2), dtype=np.uint8)
  child_y = np.zeros((1,), dtype=np.double)
  child_indices = np.zeros((1,), dtype=np.uint64)

  c_copy_smaller(
    split_c,
    split_val,
    parent_X,
    parent_y,
    parent_indices,
    parent_is_removed,
    child_X,
    child_y,
    child_indices,
    is_left
  )
  assert_array_equal(parent_is_removed, np.array([
    True,
    True,
    False,
    False,
  ], dtype = np.bool))
  assert_array_equal(child_X, np.array([[3, 4]], dtype = np.uint8))
  assert_array_almost_equal(child_y, np.array([11], dtype = np.double))
  assert_array_equal(child_indices, np.array([6], dtype = np.uint64))


def test_fit_tree_simple():
  pp = pprint.PrettyPrinter(indent=4)

  # perfect split on 2nd column
  #
  #    (c=1, bin=2, v=20)
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

  # the bins in the 2nd column are
  bins = np.array([
    [0, 1, 2, 3, 4],
    [0, 10, 20, 30, 40]
  ], dtype=np.float32)

  tree, preds = fit_tree(
    X,
    y,
    bins,
    Params(bucket_count=6)
  )
  pp.pprint(tree.__dict__)
  assert tree.node_count == 3
  assert_array_equal(tree.left_children,     [1,  0, 0])
  assert_array_equal(tree.split_cols,        [1,  0, 0])
  assert_array_almost_equal(tree.split_vals, [20, 0, 0])
  assert_array_almost_equal(tree.node_means, [0,  2, 1])
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
    Params(dfs_max_nodes=5, bucket_count=256)
  )
  pp.pprint(tree.__dict__)
  assert tree.node_count == 5
  assert_array_equal(tree.left_children,     [1, 3, 0, 0, 0])
  assert_array_equal(tree.split_cols,        [0, 0, 0, 0, 0])
  assert_array_almost_equal(tree.split_vals, [9, 3, 0, 0, 0])
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


