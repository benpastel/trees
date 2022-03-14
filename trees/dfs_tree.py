from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np

from trees.params import Params
from trees.c.dfs_tree import (
  update_histograms as c_update_histograms,
  update_node_splits as c_update_node_splits,
  update_memberships as c_update_memberships,
  eval_tree as c_eval_tree,
  choose_bin
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
  assert y.dtype == np.double
  assert y.shape == (rows,)
  assert bins.shape == (cols, params.bucket_count-1)
  assert bins.dtype == np.float32
  assert 2 <= params.bucket_count <= 256, 'buckets must fit in uint8'
  assert 0 < rows < 2**32-1, 'rows must fit in uint32'
  assert 0 < cols < 2**32-1, 'cols must fit in uint32'
  max_nodes = params.dfs_max_nodes
  assert 0 < max_nodes < 2**16-1, 'nodes must fit in uint16'

  sample_count = 2**16

  X_T = X.T.copy()
  if rows > sample_count:
    sample = np.random.choice(np.arange(rows, dtype=np.intp), size=sample_count)
    sample_X = X[sample, :]
    sample_y = y[sample]
  else:
    sample_X = X
    sample_y = y

  sample_rows = len(sample_X)

  split_cols = np.zeros(max_nodes, dtype=np.uint64)
  split_bins = np.zeros(max_nodes, dtype=np.uint8)
  left_children = np.zeros(max_nodes, dtype=np.uint16)

  # node => array of rows belonging to it
  # initially all rows belong to root (0)
  memberships = {0: np.arange(rows, dtype=np.uint64)}
  sample_memberships = {0: np.arange(sample_rows, dtype=np.uint64)}

  # node stats:
  #   - count of rows in the node
  #   - best gain from splitting at this node
  node_counts = np.zeros(max_nodes, dtype=np.uint64)
  sample_node_counts = np.zeros(max_nodes, dtype=np.uint64)
  node_gains = np.full(max_nodes, -np.inf, dtype=np.float64)

  # histograms
  # node, col, val => stat where sample_X[c] == val in this node
  #   - count of rows
  #   - sum of y
  #   - sum of y^2
  # TODO why did I make the counts uint32? should be uint64 right?
  sample_hist_counts = np.zeros((max_nodes, cols, params.bucket_count), dtype=np.uint32)
  sample_hist_sums = np.zeros((max_nodes, cols, params.bucket_count), dtype=np.float64)
  sample_hist_sum_sqs = np.zeros((max_nodes, cols, params.bucket_count), dtype=np.float64)

  # root count
  # node_counts[0] = rows
  sample_node_counts[0] = sample_rows

  # root histograms
  c_update_histograms(sample_memberships[0], sample_X, sample_y, sample_hist_counts, sample_hist_sums, sample_hist_sum_sqs, 0)

  # root node
  c_update_node_splits(
    sample_hist_counts,
    sample_hist_sums,
    sample_hist_sum_sqs,
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
    can_split_node = (sample_node_counts >= 2) & (left_children == 0)
    node_gains[~can_split_node] = -np.inf


    if node_gains.max() <= 0:
      # can't improve anymore
      # TODO plus a splitting penalty
      break

    split_n = int(np.argmax(node_gains))
    split_c = int(split_cols[split_n])
    sample_split_bin = split_bins[split_n]
    # ignore the sample split bin, and recalculate from the full dataset
    in_node = memberships[split_n]

    split_bin, left_count = choose_bin(X_T[split_c, in_node], y[in_node])

    if _VERBOSE:
      print(f"best gain: {node_gains.max()}")
      print(f"split: {split_n, split_c, sample_split_bin, split_bin}")

    # make the split
    left_children[split_n] = left_child = node_count
    right_child = node_count + 1
    node_count += 2

    # find the number of rows in each child
    # split bin or less goes left
    sample_node_counts[left_child] = sample_hist_counts[split_n, split_c, :split_bin+1].sum()
    sample_node_counts[right_child] = sample_node_counts[split_n] - sample_node_counts[left_child]
    node_counts[left_child] = left_count
    node_counts[right_child] = node_counts[split_n] - left_count

    # allocate new membership arrays
    sample_memberships[left_child] = np.zeros(sample_node_counts[left_child], dtype=np.uint64)
    sample_memberships[right_child] = np.zeros(sample_node_counts[right_child], dtype=np.uint64)
    c_update_memberships(
      sample_X,
      sample_memberships[split_n],
      sample_memberships[left_child],
      sample_memberships[right_child],
      split_c,
      split_bin,
    )
    del sample_memberships[split_n]

    # update histograms
    if sample_node_counts[left_child] < sample_node_counts[right_child]:
      # calculate left
      c_update_histograms(sample_memberships[left_child], sample_X, sample_y, sample_hist_counts, sample_hist_sums, sample_hist_sum_sqs, left_child)

      # find right via subtraction
      sample_hist_counts[right_child] = sample_hist_counts[split_n] - sample_hist_counts[left_child]
      sample_hist_sums[right_child] = sample_hist_sums[split_n] - sample_hist_sums[left_child]
      sample_hist_sum_sqs[right_child] = sample_hist_sum_sqs[split_n] - sample_hist_sum_sqs[left_child]
    else:
      # calculate right
      c_update_histograms(sample_memberships[right_child], sample_X, sample_y, sample_hist_counts, sample_hist_sums, sample_hist_sum_sqs, right_child)

      # find left via subtraction
      sample_hist_counts[left_child] = sample_hist_counts[split_n] - sample_hist_counts[right_child]
      sample_hist_sums[left_child] = sample_hist_sums[split_n] - sample_hist_sums[right_child]
      sample_hist_sum_sqs[left_child] = sample_hist_sum_sqs[split_n] - sample_hist_sum_sqs[right_child]

    # find the best splits for each new node
    c_update_node_splits(
      sample_hist_counts,
      sample_hist_sums,
      sample_hist_sum_sqs,
      node_gains,
      split_cols,
      split_bins,
      left_child,
    )

    c_update_node_splits(
      sample_hist_counts,
      sample_hist_sums,
      sample_hist_sum_sqs,
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
  for n, leaf_members in sample_memberships.items():
    node_means[n] = np.mean(sample_y[leaf_members])
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
