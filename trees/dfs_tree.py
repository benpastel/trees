from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np

from trees.params import Params
from trees.c.tree import build_dfs_tree

from trees.bfs_tree import Tree, eval_tree

# evaluation is the same as bfs_tree
# TODO move to shared location
assert eval_tree


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
  assert 0 <= params.smooth_factor
  assert 0 <= params.weight_smooth_factor
  assert 0 <= params.third_split_penalty
  assert 2 <= params.bucket_count <= 256, 'buckets must fit in uint8'
  assert 0 < rows < 2**32-1, 'rows must fit in uint32'
  assert 0 < cols < 2**32-1, 'cols must fit in uint32'

  # ignore bfs params and use fewer max nodes
  max_nodes = 64
  assert 0 < max_nodes < 2**16-1, 'nodes must fit in uint16'

  split_cols = np.zeros(max_nodes, dtype=np.uint64)
  split_lo_bins = np.zeros(max_nodes, dtype=np.uint8)
  split_hi_bins = np.zeros(max_nodes, dtype=np.uint8)
  left_children = np.zeros(max_nodes, dtype=np.uint16)
  mid_children = np.zeros(max_nodes, dtype=np.uint16)
  right_children = np.zeros(max_nodes, dtype=np.uint16)
  node_means = np.zeros(max_nodes, dtype=np.double)
  preds = np.zeros(rows, dtype=np.double)

  # row => node it belongs to
  memberships = np.zeros(rows, dtype=np.uint16)

  hists = init_histograms(X, y, max_nodes)

  for node_count in range(max_nodes):
    split_n, split_c = choose_split_column(hists)

    split_lo, split_hi = choose_split_values(hists, split_n, split_c)

    # TODO update node metadata (children, parents) here
    split_cols[split_n] = split_c
    split_lo_bins[split_n] = split_lo
    split_hi_bins[split_n] = split_hi

    new_mean = split(preds, memberships, some_other_metadata)

    # TODO update hists here


  # convert the splits from binned uint8 values => original float32 values
  split_lo_vals = np.zeros(node_count, dtype=np.float32)
  split_hi_vals = np.zeros(node_count, dtype=np.float32)
  for n in range(node_count):
    split_lo_vals[n] = bins[split_cols[n], split_lo_bins[n]]
    split_hi_vals[n] = bins[split_cols[n], split_hi_bins[n]]

  # filter down to the number of nodes we actually used
  return Tree(
    node_count,
    split_cols[:node_count],
    split_lo_vals,
    split_hi_vals,
    left_children[:node_count],
    mid_children[:node_count],
    right_children[:node_count],
    node_means[:node_count]
  ), preds

