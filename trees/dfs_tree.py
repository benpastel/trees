from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np

from trees.params import Params
from trees.c.dfs_tree import (
  update_histograms as c_update_histograms,
  update_node_splits as c_update_node_splits,
  update_memberships as c_update_memberships,
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


_VERBOSE = False


def fit_tree(
    X: np.ndarray,
    y: np.ndarray,
    bins: np.ndarray,
    params: Params
) -> Tuple[Tree, np.ndarray]:
  rows, cols = X.shape
  assert X.dtype == np.uint8
  assert y.dtype == np.float32
  assert y.shape == (rows,)
  assert bins.shape == (cols, params.bucket_count-1)
  assert bins.dtype == np.float32
  assert 2 <= params.bucket_count <= 256, 'buckets must fit in uint8'
  assert 0 < rows < 2**32-1, 'rows must fit in uint32'
  assert 0 < cols < 2**32-1, 'cols must fit in uint32'
  max_nodes = params.dfs_max_nodes
  assert 0 < max_nodes < 2**16-1, 'nodes must fit in uint16'

  split_cols = np.zeros(max_nodes, dtype=np.uint32)
  split_bins = np.zeros(max_nodes, dtype=np.uint8)
  left_children = np.zeros(max_nodes, dtype=np.uint16)

  # node => array of rows belonging to it
  # initially all rows belong to root (0)
  memberships = {0: np.arange(rows, dtype=np.uint32)}

  # node stats:
  #   - count of rows in the node
  #   - best gain from splitting at this node
  node_counts = np.zeros(max_nodes, dtype=np.uint32)
  node_gains = np.full(max_nodes, -np.inf, dtype=np.float32)

  # histograms
  # node => array of (col, val) => stat where X[c] == val in this node
  #   - count of rows
  #   - sum of y
  #   - sum of y^2
  hist_counts = {0: np.zeros((cols, params.bucket_count), dtype=np.uint32)}
  hist_sums = {0: np.zeros((cols, params.bucket_count), dtype=np.float32)}
  hist_sum_sqs = {0: np.zeros((cols, params.bucket_count), dtype=np.float32)}

  # root count
  node_counts[0] = rows

  # root histograms
  c_update_histograms(memberships[0], X, y, hist_counts[0], hist_sums[0], hist_sum_sqs[0])

  # root splits
  c_update_node_splits(
    hist_counts[0],
    hist_sums[0],
    hist_sum_sqs[0],
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
    node_counts[left_n] = hist_counts[split_n][split_c, :split_bin+1].sum()
    node_counts[right_n] = node_counts[split_n] - node_counts[left_n]

    # allocate new membership arrays
    memberships[left_n] = np.zeros(node_counts[left_n], dtype=np.uint32)
    memberships[right_n] = np.zeros(node_counts[right_n], dtype=np.uint32)
    c_update_memberships(
      X,
      memberships[split_n],
      memberships[left_n],
      memberships[right_n],
      split_c,
      split_bin,
    )
    del memberships[split_n]

    # update histograms
    # TODO parameterize smaller / larger
    if node_counts[left_n] < node_counts[right_n]:
      small_n = left_n
      large_n = right_n
    else:
      small_n = right_n
      large_n = left_n

    # calculate smaller
    hist_counts[small_n] = np.zeros((cols, params.bucket_count), dtype=np.uint32)
    hist_sums[small_n] = np.zeros((cols, params.bucket_count), dtype=np.float32)
    hist_sum_sqs[small_n] = np.zeros((cols, params.bucket_count), dtype=np.float32)
    c_update_histograms(memberships[small_n], X, y, hist_counts[small_n], hist_sums[small_n], hist_sum_sqs[small_n])

    # find larger via subtraction
    # reuse the parent node histogram arrays (split_n) since we're done with them
    hist_counts[split_n] -= hist_counts[small_n]
    hist_sums[split_n] -= hist_sums[small_n]
    hist_sum_sqs[split_n] -= hist_sum_sqs[small_n]

    hist_counts[large_n] = hist_counts.pop(split_n)
    hist_sums[large_n] = hist_sums.pop(split_n)
    hist_sum_sqs[large_n] = hist_sum_sqs.pop(split_n)

    # find the best splits for each new node
    c_update_node_splits(
      hist_counts[left_n],
      hist_sums[left_n],
      hist_sum_sqs[left_n],
      node_gains,
      split_cols,
      split_bins,
      left_n,
    )

    c_update_node_splits(
      hist_counts[right_n],
      hist_sums[right_n],
      hist_sum_sqs[right_n],
      node_gains,
      split_cols,
      split_bins,
      right_n,
    )

  # finished growing the tree

  # any node remaining in membership is a leaf
  # prediction for each row is the mean of the node the row is in
  node_means = np.zeros(node_count, dtype=np.float32)
  preds = np.zeros(rows, dtype=np.float32)
  for n, leaf_members in memberships.items():
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
  assert X.dtype == np.float32
  rows, feats = X.shape

  values = np.zeros(rows, dtype=np.float32)
  c_eval_tree(
    X,
    tree.split_cols,
    tree.split_vals,
    tree.left_children,
    tree.node_means,
    values)
  return values
