import numpy as np
import pprint

from numpy.testing import assert_array_equal, assert_array_almost_equal

from trees.dfs_tree import fit_tree, eval_tree, Tree
from trees.params import Params

from trees.c.dfs_tree import update_node_splits as c_update_node_splits
from test.python_stubs import update_node_splits as py_update_node_splits, variance

def test_c_update_nodes():
  max_nodes = 2
  cols = 2
  vals = 256

  hist_counts = np.random.randint(10, size=(cols, vals), dtype=np.uint32)
  hist_sums = np.random.random(size=(cols, vals)).astype(np.float32)
  hist_sum_sqs = np.random.random(size=(cols, vals)).astype(np.float32)

  py_node_gains = np.full(max_nodes, -np.inf, dtype=np.float32)
  py_split_cols = np.zeros(max_nodes, dtype=np.uint32)
  py_split_bins = np.zeros(max_nodes, dtype=np.uint8)

  c_node_gains = np.full(max_nodes, -np.inf, dtype=np.float32)
  c_split_cols = np.zeros(max_nodes, dtype=np.uint32)
  c_split_bins = np.zeros(max_nodes, dtype=np.uint8)

  py_update_node_splits(hist_counts, hist_sums, hist_sum_sqs, py_node_gains, py_split_cols, py_split_bins, 0)
  c_update_node_splits(hist_counts, hist_sums, hist_sum_sqs, c_node_gains, c_split_cols, c_split_bins, 0)

  assert_array_almost_equal(py_node_gains, c_node_gains, decimal=3)
  assert_array_equal(py_split_cols, c_split_cols)
  assert_array_equal(py_split_bins, c_split_bins)


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
  y = np.array([2, 1, 2, 1], dtype=np.float32)
  preds = np.zeros(4, dtype=np.float32)

  # the bins in the 2nd column are
  bins = np.array([
    [0, 1, 2, 3, 4],
    [0, 10, 20, 30, 40]
  ], dtype=np.float32)

  tree = fit_tree(
    X,
    y,
    bins,
    Params(bucket_count=6),
    preds
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
  y = np.array([9, 2, 7, 0, 2, 100, 3, 100, 100], dtype=np.float32)
  bins = np.arange(255, dtype=np.float32).reshape((1, -1))
  preds = np.zeros(len(y), dtype=np.float32)
  tree = fit_tree(
    X,
    y,
    bins,
    Params(dfs_max_nodes=5, bucket_count=256),
    preds
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
    split_cols = np.array([1,   0,  0, 0, 0], dtype=np.uint32),
    split_vals = np.array([100, 0, 10, 0, 0], dtype=np.float32),
    left_children  = np.array([1, 0, 3, 0, 0], dtype=np.uint16),
    node_means     = np.array([0, 1, 0, 3, 2], dtype=np.float32),
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


