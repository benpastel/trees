from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np

from trees.params import Params
from trees.c.bfs_tree import build_tree, eval_tree as c_eval_tree

@dataclass
class Tree:
  node_count: int
  split_cols: np.ndarray
  split_lo_vals: np.ndarray
  split_hi_vals: np.ndarray
  left_children: np.ndarray
  mid_children: np.ndarray
  right_children: np.ndarray
  node_means: np.ndarray

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

  # check if depth constraint imposes a tighter max_nodes
  max_nodes_from_depth = np.sum(3**np.arange(params.max_depth))
  max_nodes = min(params.max_nodes, max_nodes_from_depth)
  assert 0 < max_nodes < 2**16, 'max_nodes must fit in uint16'
  assert 0 < cols * max_nodes * params.bucket_count < 2**64-1, 'histograms indices must fit in uint64'

  # output arrays for c function
  # pre-allocated to the max number of nodes
  split_cols = np.zeros(max_nodes, dtype=np.uint64)
  split_lo_bins = np.zeros(max_nodes, dtype=np.uint8)
  split_hi_bins = np.zeros(max_nodes, dtype=np.uint8)
  left_children = np.zeros(max_nodes, dtype=np.uint16)
  mid_children = np.zeros(max_nodes, dtype=np.uint16)
  right_children = np.zeros(max_nodes, dtype=np.uint16)
  node_means = np.zeros(max_nodes, dtype=np.double)
  preds = np.zeros(rows, dtype=np.double)

  node_count = build_tree(
    X,
    y,
    split_cols,
    split_lo_bins,
    split_hi_bins,
    left_children,
    mid_children,
    right_children,
    node_means,
    preds,
    params.smooth_factor,
    params.weight_smooth_factor,
    params.max_depth,
    params.third_split_penalty,
    params.bucket_count)

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


def eval_tree(tree: Tree, X: np.ndarray) -> np.ndarray:
  assert X.dtype == np.float32
  assert X.ndim == 2
  rows, feats = X.shape

  values = np.zeros(rows, dtype=np.double)
  c_eval_tree(
    X,
    tree.split_cols,
    tree.split_lo_vals,
    tree.split_hi_vals,
    tree.left_children,
    tree.mid_children,
    tree.right_children,
    tree.node_means,
    values)
  return values

