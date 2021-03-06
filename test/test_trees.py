import numpy as np
import pprint

from numpy.testing import assert_array_equal, assert_array_almost_equal

from trees.tree import fit_tree, eval_tree, Tree
from trees.params import Params


def test_fit_tree():
  pp = pprint.PrettyPrinter(indent=4)

  # test X values, columns, bins are tracked properly
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
  bins = np.array([
    np.arange(255, dtype=np.float32),
    10.0 * np.arange(255, dtype=np.float32)
  ])

  tree, preds = fit_tree(
    X,
    y,
    bins,
    Params(smooth_factor=1.0, third_split_penalty=0.0, bucket_count=256, weight_smooth_factor=0.0)
  )
  pp.pprint(tree.__dict__)
  assert tree.node_count == 4
  assert_array_equal(tree.left_children,        [ 1, 0, 0, 0])
  assert_array_equal(tree.mid_children,         [ 2, 0, 0, 0])
  assert_array_equal(tree.right_children,       [ 3, 0, 0, 0])
  assert_array_equal(tree.split_cols,           [ 1, 0, 0, 0])
  assert_array_almost_equal(tree.split_lo_vals, [10, 0, 0, 0])
  assert_array_almost_equal(tree.split_hi_vals, [30, 0, 0, 0])
  assert_array_almost_equal(tree.node_means,    [ 0, 1, 2, 0])
  assert_array_almost_equal(preds, [2, 1, 0, 2, 1, 0])

  #            (3,9)
  #           /  |  \
  #         /    |    \
  #       /      |      \
  # [0,2,2,3]   [9,7]  [100,100,100]
  #     #3       #1       #2
  #
  X = np.array([9, 2, 7, 0, 2, 100, 3, 100, 100], dtype=np.uint8).reshape((-1, 1))
  y = np.array([9, 2, 7, 0, 2, 100, 3, 100, 100], dtype=np.double)
  bins = np.arange(255, dtype=np.float32).reshape((1, -1))
  tree, preds = fit_tree(
    X,
    y,
    bins,
    Params(smooth_factor=1.0, third_split_penalty=0.01, bucket_count=256, weight_smooth_factor=0.0)
  )
  pp.pprint(tree.__dict__)
  assert tree.node_count == 4
  assert_array_equal(tree.left_children,         [3, 0, 0, 0])
  assert_array_equal(tree.mid_children,          [1, 0, 0, 0])
  assert_array_equal(tree.right_children,        [2, 0, 0, 0])
  assert_array_equal(tree.split_cols,            [0, 0, 0, 0])
  assert_array_almost_equal(tree.split_lo_vals,  [3, 0, 0, 0])
  assert_array_almost_equal(tree.split_hi_vals,  [9, 0, 0, 0])
  assert_array_almost_equal(tree.node_means, [0, 8, 100, 7/4])
  assert_array_almost_equal(preds, [8, 7/4, 8, 7/4, 7/4, 100, 7/4, 100, 100])

  # same as above, but with weight_smooth_factor=2.0, it's like adding 2 values to each leaf with y=0
  tree, preds = fit_tree(
    X,
    y,
    bins,
    Params(smooth_factor=1.0, third_split_penalty=0.01, bucket_count=256, weight_smooth_factor=2.0)
  )
  assert tree.node_count == 4
  assert_array_almost_equal(tree.node_means, [0, 4, 60, 7/6])
  assert_array_almost_equal(preds, [4, 7/6, 4, 7/6, 7/6, 60, 7/6, 60, 60])

  #             (9,9)
  #           /   |  \
  #         /     |    \
  #       /       |      \
  # [0,2,2,3,9,7] []  [100,100,100]
  #       #3      #1        #2
  X = np.array([9, 2, 7, 0, 2, 100, 3, 100, 100], dtype=np.uint8).reshape((-1, 1))
  y = np.array([9, 2, 7, 0, 2, 100, 3, 100, 100], dtype=np.double)
  bins = np.arange(255, dtype=np.float32).reshape((1, -1))
  tree, preds = fit_tree(
    X,
    y,
    bins,
    Params(smooth_factor=1.0, third_split_penalty=10.0, bucket_count=256, weight_smooth_factor=0.0)
  )
  pp.pprint(tree.__dict__)
  assert tree.node_count == 4
  assert_array_equal(tree.left_children,         [3, 0, 0, 0])
  assert_array_equal(tree.mid_children,          [1, 0, 0, 0])
  assert_array_equal(tree.right_children,        [2, 0, 0, 0])
  assert_array_equal(tree.split_cols,            [0, 0, 0, 0])
  assert_array_almost_equal(tree.split_lo_vals,  [9, 0, 0, 0])
  assert_array_almost_equal(tree.split_hi_vals,  [9, 0, 0, 0])
  assert_array_almost_equal(tree.node_means, [0, 0, 100, 23/6])
  assert_array_almost_equal(preds, [23/6, 23/6, 23/6, 23/6, 23/6, 100, 23/6, 100, 100])


  #             (3, 9)
  #          /    |    \
  #        /      |      \
  #      /        |        \
  #   (0,2)#3   (7,7)#1    [20,20,20]#2
  #  /  |  \    /  |   \
  # 0 [2,2] 3  7 empty  9
  # #7 #9  #8  #4  #5  #6
  #
  X = np.array([9, 2, 7, 0, 2, 20, 3, 20, 20], dtype=np.uint8).reshape((-1, 1))
  y = np.array([9, 2, 7, 0, 2, 20, 3, 20, 20], dtype=np.double)
  bins = np.arange(255, dtype=np.float32).reshape((1, -1))
  tree, preds = fit_tree(
    X,
    y,
    bins,
    Params(smooth_factor=0.0, bucket_count=256, weight_smooth_factor=0.0)
  )
  pp.pprint(tree.__dict__)
  assert tree.node_count == 10
  assert_array_equal(tree.left_children,         [3, 4, 0, 7, 0, 0, 0, 0, 0, 0])
  assert_array_equal(tree.mid_children,          [1, 5, 0, 9, 0, 0, 0, 0, 0, 0])
  assert_array_equal(tree.right_children,        [2, 6, 0, 8, 0, 0, 0, 0, 0, 0])
  assert_array_equal(tree.split_cols,            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
  assert_array_almost_equal(tree.split_lo_vals,  [3, 7, 0, 0, 0, 0, 0, 0, 0, 0])
  assert_array_almost_equal(tree.split_hi_vals,  [9, 7, 0, 2, 0, 0, 0, 0, 0, 0])
  assert_array_almost_equal(tree.node_means,     [0, 0, 20, 0, 7, 0, 9, 0, 3, 2])
  assert_array_almost_equal(preds, [9, 2, 7, 0, 2, 20, 3, 20, 20])


def test_eval_tree():
  #     1:(1,100)
  #   /    |      \
  #  /     |       \
  # 1   0:(10,10)   9
  #      /   |   \
  #     3   empty 2
  #
  tree = Tree(
    node_count = 7,
    split_cols     = np.array([1,   0,  0, 0, 0, 0, 0], dtype=np.uint64),
    split_lo_vals  = np.array([1,   0, 10, 0, 0, 0, 0], dtype=np.float32),
    split_hi_vals  = np.array([100, 0, 10, 0, 0, 0, 0], dtype=np.float32),
    left_children  = np.array([1,   0,  4, 0, 0, 0, 0], dtype=np.uint16),
    mid_children   = np.array([2,   0,  5, 0, 0, 0, 0], dtype=np.uint16),
    right_children = np.array([3,   0,  6, 0, 0, 0, 0], dtype=np.uint16),
    node_means     = np.array([0,   1,  0, 9, 3, 0, 2], dtype=np.double),
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
      1,
      9
    ]))





