import os
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np

from trees.params import Params
from trees.c.dfs_tree import (
  update_histograms as c_update_histograms,
  update_node_splits as c_update_node_splits,
  update_members as c_update_members,
  eval_tree as c_eval_tree,
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

# set VERBOSE=1 when debugging
_VERBOSE = bool(os.environ.get('VERBOSE', '0'))

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

  # node stats:
  #   - count of rows in the node
  #   - best gain from splitting at this node
  node_counts = np.zeros(max_nodes, dtype=np.uint64)
  node_gains = np.full(max_nodes, -np.inf, dtype=np.float64)

  # root count
  node_counts[0] = rows

  # members contains the indices of all rows in X
  # start_members points the start of each node's members
  # so members[start_members[n]: start_members[n]+node_counts[n]] is the rows in n
  # initially, all rows belong the root (0)
  # TODO: figure out how to pass that slice directly into C
  members = np.arange(rows, dtype=np.uint64)
  start_members = {0: 0}

  # histograms
  # node, col, val => stat where X[c] == val in this node
  #   - count of rows
  #   - sum of y
  #   - sum of y^2
  # TODO why did I make the counts uint32? should be uint64 right?
  hist_counts = np.zeros((max_nodes, cols, params.bucket_count), dtype=np.uint32)
  hist_sums = np.zeros((max_nodes, cols, params.bucket_count), dtype=np.float64)
  hist_sum_sqs = np.zeros((max_nodes, cols, params.bucket_count), dtype=np.float64)

  # root histograms
  c_update_histograms(members, X, y, hist_counts, hist_sums, hist_sum_sqs, 0, 0, rows)

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

    # make the split
    left_children[split_n] = left_child = node_count
    right_child = node_count + 1
    node_count += 2

    # find the number of rows in each child
    # split bin or less goes left
    node_counts[left_child] = hist_counts[split_n, split_c, :split_bin+1].sum()
    node_counts[right_child] = node_counts[split_n] - node_counts[left_child]

    if _VERBOSE:
      print(f"best gain: {node_gains.max()}")
      print(f"split: {split_n, split_c, split_bin}")
      print(f"{node_counts[left_child]} going left, {node_counts[right_child]} going right")
    assert node_counts[left_child] >= 1 and node_counts[right_child] >= 1

    # swap around the parent members so the left members are on the left
    # and the right members are on the right
    if _VERBOSE:
      print(f"members before: {members}")
      print(f"parent members: {members[start_members[split_n]:start_members[split_n]+int(node_counts[split_n])]}")
    c_update_members(
      X,
      members,
      start_members[split_n],
      int(node_counts[left_child]),
      int(node_counts[right_child]),
      split_c,
      split_bin,
    )
    start_members[left_child] = start_members[split_n]
    start_members[right_child] = start_members[split_n] + int(node_counts[left_child])
    if _VERBOSE:
      print(f"members after: {members}")
      print(f"left members: {members[start_members[left_child]:start_members[left_child]+int(node_counts[left_child])]}")
      print(f"right members: {members[start_members[right_child]:start_members[right_child]+int(node_counts[right_child])]}")

    # update histograms
    if node_counts[left_child] < node_counts[right_child]:
      # calculate left
      c_update_histograms(
        members, X, y,
        hist_counts, hist_sums, hist_sum_sqs,
        left_child, start_members[left_child], int(node_counts[left_child]))

      # find right via subtraction
      hist_counts[right_child] = hist_counts[split_n] - hist_counts[left_child]
      hist_sums[right_child] = hist_sums[split_n] - hist_sums[left_child]
      hist_sum_sqs[right_child] = hist_sum_sqs[split_n] - hist_sum_sqs[left_child]
    else:
      # calculate right
      c_update_histograms(
        members, X, y,
        hist_counts, hist_sums, hist_sum_sqs,
        right_child, start_members[right_child], int(node_counts[right_child]))

      # find left via subtraction
      hist_counts[left_child] = hist_counts[split_n] - hist_counts[right_child]
      hist_sums[left_child] = hist_sums[split_n] - hist_sums[right_child]
      hist_sum_sqs[left_child] = hist_sum_sqs[split_n] - hist_sum_sqs[right_child]

    # find the best splits for each new node
    c_update_node_splits(
      hist_counts,
      hist_sums,
      hist_sum_sqs,
      node_gains,
      split_cols,
      split_bins,
      left_child,
    )

    c_update_node_splits(
      hist_counts,
      hist_sums,
      hist_sum_sqs,
      node_gains,
      split_cols,
      split_bins,
      right_child,
    )

  # finished growing the tree

  # any node remaining in membership is a leaf
  # prediction for each row is the mean of the node the row is in
  node_means = np.zeros(node_count)
  preds = np.zeros(rows, dtype=np.float32)
  for n, start in start_members.items():
    leaf_members = members[start:start + int(node_counts[n])]
    node_means[n] = np.mean(y[leaf_members])
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
