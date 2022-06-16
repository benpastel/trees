from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np

from trees.params import Params
from trees.c.dfs_tree import (
  update_histograms as c_update_histograms,
  update_node_splits as c_update_node_splits,
  update_memberships as c_update_memberships,
  eval_tree as c_eval_tree,
  copy_smaller as c_copy_smaller,
)

# A binary tree that grows best-node-first, like LightGBM
# currently has no regularization and is all in python
@dataclass
class Tree:
  node_count: int
  split_cols: np.ndarray
  split_vals: np.ndarray

  # index of the left child of this node
  # right child is always left child + 1
  # left_children == 0 means it's a leaf
  left_children: np.ndarray
  node_means: np.ndarray


# TODO: move more node attributes into here
@dataclass
class Node:
  # subset of X in node
  X: np.ndarray

  # subset of y in node
  y: np.ndarray

  # array of indices into the original X & y for each row in node
  indices: np.ndarray

  # tombstone mask marking X, y, and indices as removed
  is_removed: np.ndarray


_VERBOSE = False


def fit_tree(
    X: np.ndarray,
    y: np.ndarray,
    bins: np.ndarray,
    params: Params
) -> Tuple[Tree, np.ndarray]:
  rows, cols = X.shape
  assert X.dtype == np.uint8
  assert y.dtype == np.double
  assert y.shape == (rows,)
  assert bins.shape == (cols, params.bucket_count-1)
  assert bins.dtype == np.float32
  assert 2 <= params.bucket_count <= 256, 'buckets must fit in uint8'
  assert 0 < rows < 2**32-1, 'rows must fit in uint32'
  assert 0 < cols < 2**32-1, 'cols must fit in uint32'
  max_nodes = params.dfs_max_nodes
  assert 0 < max_nodes < 2**16-1, 'nodes must fit in uint16'

  split_cols = np.zeros(max_nodes, dtype=np.uint64)
  split_bins = np.zeros(max_nodes, dtype=np.uint8)
  left_children = np.zeros(max_nodes, dtype=np.uint16)

  # nodes are stored:
  # node_id => Node
  # the root node (0) has all rows
  all_indices = np.arange(rows, dtype=np.uint64)
  nothing_removed = np.zeros(rows, dtype=bool)
  nodes = {0: Node(X, y, all_indices, nothing_removed)}

  # node stats:
  #   - count of rows in the node
  #   - best gain from splitting at this node
  node_counts = np.zeros(max_nodes, dtype=np.uint64)
  node_gains = np.full(max_nodes, -np.inf, dtype=np.float64)

  # histograms
  # node, col, val => stat where X[c] == val in this node
  #   - count of rows
  #   - sum of y
  #   - sum of y^2
  # TODO why did I make the counts uint32? should be uint64 right?
  # TODO just use a node_idx => array dict
  hist_counts = np.zeros((max_nodes, cols, params.bucket_count), dtype=np.uint32)
  hist_sums = np.zeros((max_nodes, cols, params.bucket_count), dtype=np.float64)
  hist_sum_sqs = np.zeros((max_nodes, cols, params.bucket_count), dtype=np.float64)

  # root count
  node_counts[0] = rows

  # root histograms
  # TODO: memberships different
  c_update_histograms(X, y, hist_counts, hist_sums, hist_sum_sqs, 0)

  # root node
  c_update_node_splits(
    hist_counts,
    hist_sums,
    hist_sum_sqs,
    node_gains,
    split_cols,
    split_bins,
    0,
  )

  # grow the tree
  node_count = 1
  while node_count + 2 <= max_nodes:
    # can add 2 children and not have too many nodes

    # choose the node & column with the lowest score
    # can split leaves with enough data
    can_split_node = (node_counts >= 2) & (left_children == 0)
    node_gains[~can_split_node] = -np.inf

    if node_gains.max() <= 0:
      # can't improve anymore
      # TODO plus a splitting penalty
      break

    split_n = int(np.argmax(node_gains))
    split_c = int(split_cols[split_n])
    split_bin = split_bins[split_n]

    if _VERBOSE:
      print(f"best gain: {node_gains.max()}")
      print(f"split: {split_n, split_c, split_bin}")

    # make the split
    left_children[split_n] = left_n = node_count
    right_n = node_count + 1
    node_count += 2

    # find the number of rows in each child
    # split bin or less goes left
    left_count = node_counts[left_n] = hist_counts[split_n, split_c, :split_bin+1].sum()
    right_count = node_counts[right_n] = node_counts[split_n] - node_counts[left_n]

    # TODO probably worth parameterizing on smaller child & larger child
    parent = nodes[split_n]
    if left_count < right_count:
      # initialize an empty node of the correct size;
      # copy_smaller will fill it in
      left = nodes[left_n] = Node(
        X = np.zeros((left_count, cols), dtype=X.dtype),
        y = np.zeros(left_count, dtype=y.dtype),
        indices = np.zeros(left_count, dtype=np.uint64),
        is_removed = np.zeros(left_count, dtype=bool)
      )

      # fill in the new node & mark removed in old node
      c_copy_smaller(
        split_c,
        split_bin,
        parent.X,
        parent.y,
        parent.indices,
        parent.is_removed,
        left.X,
        left.y,
        left.indices,
        True
      )
      # the right node is now the parent, with left rows marked as removed
      nodes[right_n] = nodes[split_n]
      del nodes[split_n]

      # update left histograms
      # TODO: eventually do during partition
      c_update_histograms(left.X, left.y, hist_counts, hist_sums, hist_sum_sqs, left_n)

      # find right histograms via subtraction
      hist_counts[right_n] = hist_counts[split_n] - hist_counts[left_n]
      hist_sums[right_n] = hist_sums[split_n] - hist_sums[left_n]
      hist_sum_sqs[right_n] = hist_sum_sqs[split_n] - hist_sum_sqs[left_n]

    else:
      # initialize an empty node of the correct size;
      # copy_smaller will fill it in
      right = nodes[right_n] = Node(
        X = np.zeros((right_count, cols), dtype=X.dtype),
        y = np.zeros(right_count, dtype=y.dtype),
        indices = np.zeros(right_count, dtype=np.uint64),
        is_removed = np.zeros(right_count, dtype=bool)
      )

      # fill in the new node, mark removed in old node, count left histogram
      c_copy_smaller(
        split_c,
        split_bin,
        parent.X,
        parent.y,
        parent.indices,
        parent.is_removed,
        right.X,
        right.y,
        right.indices,
        False
      )
      # the left node is now the parent, with right rows marked as removed
      nodes[left_n] = nodes[split_n]
      del nodes[split_n]

      # update right histograms
      # TODO: eventually do during partition
      c_update_histograms(right.X, right.y, hist_counts, hist_sums, hist_sum_sqs, right_n)

      # update left histograms via subtraction
      hist_counts[left_n] = hist_counts[split_n] - hist_counts[right_n]
      hist_sums[left_n] = hist_sums[split_n] - hist_sums[right_n]
      hist_sum_sqs[left_n] = hist_sum_sqs[split_n] - hist_sum_sqs[right_n]

    # find the best splits for each new node
    c_update_node_splits(
      hist_counts,
      hist_sums,
      hist_sum_sqs,
      node_gains,
      split_cols,
      split_bins,
      left_n,
    )
    c_update_node_splits(
      hist_counts,
      hist_sums,
      hist_sum_sqs,
      node_gains,
      split_cols,
      split_bins,
      right_n,
    )

  # finished growing the tree

  # any node remaining in membership is a leaf
  # prediction for each row is the mean of the node the row is in
  node_means = np.zeros(node_count)
  preds = np.zeros(rows, dtype=np.float32)

  for n, node in nodes.items():
    node_means[n] = np.mean(node.y[~node.is_removed])
    leaf_members = node.indices[~node.is_removed]
    preds[leaf_members] = node_means[n]

  # convert the splits from binned uint8 values => original float32 values
  split_vals = np.zeros(node_count, dtype=np.float32)
  for n in range(node_count):
    split_vals[n] = bins[split_cols[n], split_bins[n]]

  # truncate down to the number of nodes we actually used
  split_cols = split_cols[:node_count]
  left_children = left_children[:node_count]

  # zero out split_vals at leaves
  # it doesn't matter what they are, but this makes it easier to assert what the
  # tree looks like in the unit test
  is_leaf = (left_children == 0)
  split_vals[is_leaf] = 0

  return Tree(
    node_count,
    split_cols,
    split_vals,
    left_children,
    node_means,
  ), preds


def eval_tree(tree: Tree, X: np.ndarray) -> np.ndarray:
  # this will be the python wrapper for the c function
  assert X.dtype == np.float32
  rows, feats = X.shape

  values = np.zeros(rows, dtype=np.double)
  c_eval_tree(
    X,
    tree.split_cols,
    tree.split_vals,
    tree.left_children,
    tree.node_means,
    values)
  return values
