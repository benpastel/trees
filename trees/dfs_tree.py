from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

import numpy as np

from trees.params import Params, DEBUG_STATS
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

def _init(max_nodes: int, rows: int, cols: int, params: Params) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[int, np.ndarray],
    np.ndarray,
    np.ndarray,
    Dict[int, np.ndarray],
    Dict[int, np.ndarray],
    Dict[int, np.ndarray]
]:
  # TODO for profiling; maybe remove

  split_cols = np.zeros(max_nodes, dtype=np.uint32)
  split_bins = np.zeros(max_nodes, dtype=np.uint8)
  left_children = np.zeros(max_nodes, dtype=np.uint16)

  # node => array of rows belonging to it
  # initially all rows belong to root (0)
  # TODO since we special-case the root, maybe remove this?
  memberships = {0: np.arange(rows, dtype=np.uint32)}

  # node stats:
  #   - count of rows in the node
  #   - best gain from splitting at this node
  node_counts = np.empty(max_nodes, dtype=np.uint32)
  node_gains = np.full(max_nodes, -np.inf, dtype=np.float32)

  # histograms
  # node => array of (col, val) => stat where X[c] == val in this node
  #   - count of rows
  #   - sum of y
  #   - sum of y^2
  hist_counts = {0: np.zeros((cols, params.bucket_count), dtype=np.uint32)}
  hist_sums = {0: np.zeros((cols, params.bucket_count), dtype=np.float32)}
  hist_sum_sqs = {0: np.zeros((cols, params.bucket_count), dtype=np.float32)}

  return (
    split_cols,
    split_bins,
    left_children,
    memberships,
    node_counts,
    node_gains,
    hist_counts,
    hist_sums,
    hist_sum_sqs,
  )



def _set_root_stats(
    node_counts: np.ndarray,
    rows: int,
    memberships: Dict[int, np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    hist_counts: Dict[int, np.ndarray],
    hist_sums: Dict[int, np.ndarray],
    hist_sum_sqs: Dict[int, np.ndarray],
    node_gains: np.ndarray,
    split_cols: np.ndarray,
    split_bins: np.ndarray,
  ) -> None:
  # TODO for profiling; maybe remove

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

def _choose_split(
  node_counts: np.ndarray,
  left_children: np.ndarray,
  node_gains: np.ndarray,
  split_cols: np.ndarray,
  split_bins: np.ndarray
) -> Optional[Tuple[int, int, int]]:
  # TODO for profiling; maybe remove
  # choose the node & column with the lowest score
  # can split leaves with enough data
  can_split_node = (node_counts >= 2) & (left_children == 0)
  node_gains[~can_split_node] = -np.inf

  if node_gains.max() <= 0:
    # can't improve anymore
    # TODO plus a splitting penalty
    return None

  split_n = int(np.argmax(node_gains))
  split_c = int(split_cols[split_n])
  split_bin = split_bins[split_n]

  if _VERBOSE:
    print(f"best gain: {node_gains.max()}")
    print(f"split: {split_n, split_c, split_bin}")

  return split_n, split_c, split_bin


def _set_child_counts(
  node_counts: np.ndarray,
  hist_counts: Dict[int, np.ndarray],
  split_n: int,
  split_c: int,
  split_bin: int,
  left_n: int,
  right_n: int
) -> None:
  # TODO for profiling; maybe remove
  # find the number of rows in each child
  # split bin or less goes left
  node_counts[left_n] = hist_counts[split_n][split_c, :split_bin+1].sum()
  node_counts[right_n] = node_counts[split_n] - node_counts[left_n]


def _update_memberships(
  memberships: Dict[int, np.ndarray],
  node_counts: np.ndarray,
  X: np.ndarray,
  left_n: int,
  right_n: int,
  split_c: int,
  split_bin: int,
  split_n: int
) -> None:
  # TODO for profiling; maybe remove
  # allocate new membership arrays
  # TODO: tombstone the larger array sometimes instead of copying?
  # try shuffling the array instead of allocating new ones?
  memberships[left_n] = np.empty(node_counts[left_n], dtype=np.uint32)
  memberships[right_n] = np.empty(node_counts[right_n], dtype=np.uint32)
  c_update_memberships(
    X,
    memberships[split_n],
    memberships[left_n],
    memberships[right_n],
    split_c,
    split_bin,
  )
  del memberships[split_n]


def _unbin(
    node_count: int,
    split_cols: np.ndarray,
    split_bins: np.ndarray,
    bins: np.ndarray,
) -> np.ndarray:
  # TODO for profiling; maybe remove

  # convert the splits from binned uint8 values => original float32 values
  split_vals = np.empty(node_count, dtype=np.float32)
  for n in range(node_count):
    split_vals[n] = bins[split_cols[n], split_bins[n]]

  return split_vals

def _update_histograms(
    node_counts: np.ndarray,
    hist_counts: Dict[int, np.ndarray],
    hist_sums: Dict[int, np.ndarray],
    hist_sum_sqs: Dict[int, np.ndarray],
    memberships: Dict[int, np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    left_n: int,
    right_n: int,
    split_n: int,
    cols: int,
    params: Params,
  ) -> Tuple[int, int]:
  # TODO for profiling; maybe remove
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

  return small_n, large_n


def _means_and_preds(
    node_count: int,
    memberships: Dict[int, np.ndarray],
    hist_sums: Dict[int, np.ndarray],
    hist_counts: Dict[int, np.ndarray],
    preds: np.ndarray
) -> np.ndarray:
  # TODO for profiling; maybe remove
  # any node remaining in membership is a leaf
  # prediction for each row is the mean of the node the row is in
  node_means = np.zeros(node_count, dtype=np.float32)
  for n, leaf_members in memberships.items():
    # mean y is sum/count in any histogram
    # so use feature 0
    node_means[n] = np.sum(hist_sums[n][0]) / np.sum(hist_counts[n][0])
    preds[leaf_members] += node_means[n]
  return node_means


def _truncate_and_zero(
  split_cols: np.ndarray,
  left_children: np.ndarray,
  split_vals: np.ndarray,
  node_count: int
) -> Tuple[np.ndarray, np.ndarray]:
  # TODO for profiling; maybe remove
  # truncate down to the number of nodes we actually used
  split_cols = split_cols[:node_count]
  left_children = left_children[:node_count]

  # zero out split_vals at leaves
  # it doesn't matter what they are, but this makes it easier to assert what the
  # tree looks like in the unit test
  is_leaf = (left_children == 0)
  split_vals[is_leaf] = 0
  return split_cols, left_children

def fit_tree(
    X: np.ndarray,
    y: np.ndarray,
    bins: np.ndarray,
    params: Params,
    preds: np.ndarray
) -> Tree:
  rows, cols = X.shape
  assert X.dtype == np.uint8
  assert y.dtype == np.float32
  assert y.shape == (rows,)
  assert bins.shape == (cols, params.bucket_count-1)
  assert bins.dtype == np.float32
  assert 2 <= params.bucket_count <= 256, 'buckets must fit in uint8'
  max_nodes = params.dfs_max_nodes
  assert 0 < max_nodes < 2**16-1, 'nodes must fit in uint16'

  # TODO promote X indices to uint64 to avoid this?
  assert 0 < rows * cols< 2**32-1, 'rows * cols must fit in uint32'

  (
    split_cols,
    split_bins,
    left_children,
    memberships,
    node_counts,
    node_gains,
    hist_counts,
    hist_sums,
    hist_sum_sqs,
  ) = _init(max_nodes, rows, cols, params)

  _set_root_stats(
    node_counts,
    rows,
    memberships,
    X,
    y,
    hist_counts,
    hist_sums,
    hist_sum_sqs,
    node_gains,
    split_cols,
    split_bins,
  )

  if DEBUG_STATS:
    small_fractions = []
    small_counts = []
    large_counts = []

  # grow the tree
  node_count = 1
  while node_count + 2 <= max_nodes:
    # can add 2 children and not have too many nodes

    split = _choose_split(
      node_counts,
      left_children,
      node_gains,
      split_cols,
      split_bins
    )
    if split is None:
      break

    # make the split
    split_n, split_c, split_bin = split
    left_children[split_n] = left_n = node_count
    right_n = node_count + 1
    node_count += 2
    _set_child_counts(
      node_counts,
      hist_counts,
      split_n,
      split_c,
      split_bin,
      left_n,
      right_n
    )

    _update_memberships(
      memberships,
      node_counts,
      X,
      left_n,
      right_n,
      split_c,
      split_bin,
      split_n
    )

    small_n, large_n = _update_histograms(
      node_counts,
      hist_counts,
      hist_sums,
      hist_sum_sqs,
      memberships,
      X,
      y,
      left_n,
      right_n,
      split_n,
      cols,
      params
    )

    if DEBUG_STATS:
      small_counts.append(node_counts[small_n])
      large_counts.append(node_counts[large_n])
      small_fractions.append(node_counts[small_n] / node_counts[large_n])

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
  node_means = _means_and_preds(
    node_count,
    memberships,
    hist_sums,
    hist_counts,
    preds
  )

  split_vals = _unbin(
    node_count,
    split_cols,
    split_bins,
    bins,
  )

  split_cols, left_children = _truncate_and_zero(
    split_cols,
    left_children,
    split_vals,
    node_count
  )

  if DEBUG_STATS:
    print(f"""
      {np.bincount(split_cols)=}
      {node_means=}
      {small_fractions=}
      {small_counts=}
      {large_counts=}
    """)

  return Tree(
    node_count,
    split_cols,
    split_vals,
    left_children,
    node_means,
  )


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
