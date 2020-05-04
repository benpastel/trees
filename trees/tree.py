from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np

from trees.params import Params
from trees.c.tree import build_tree, eval_tree as c_eval_tree

@dataclass
class Tree:
  node_count: int
  split_cols: np.ndarray
  split_vals: np.ndarray
  children: np.ndarray
  node_means: np.ndarray

def fit_tree(
    XT: np.ndarray,
    y: np.ndarray,
    bins: np.ndarray,
    params: Params
) -> Tuple[Tree, np.ndarray]:
  feats, rows = XT.shape
  assert XT.dtype == np.uint8
  assert y.dtype == np.double
  assert y.shape == (rows,)
  assert bins.shape == (feats, 255)
  assert bins.dtype == np.float32
  assert 2 <= params.branch_count <= 4
  assert 0 <= params.smooth_factor

  # check if depth constraint imposes a tighter max_nodes
  max_nodes_from_depth = np.sum(params.branch_count**np.arange(params.max_depth))
  max_nodes = min(params.max_nodes, max_nodes_from_depth)
  assert 0 < max_nodes < 2**16

  # output arrays for c function
  # pre-allocated to the max number of nodes
  split_cols = np.zeros(max_nodes, dtype=np.uint64)
  split_bins = np.zeros((max_nodes, params.branch_count - 1), dtype=np.uint8)
  children = np.zeros((max_nodes, params.branch_count), dtype=np.uint16)
  node_means = np.zeros(max_nodes, dtype=np.double)
  preds = np.zeros(rows, dtype=np.double)

  node_count = build_tree(
    XT,
    y,
    split_cols,
    split_bins,
    children,
    node_means,
    preds,
    params.smooth_factor,
    params.max_depth)

  # convert the splits from binned uint8 values => original float32 values
  split_vals = np.zeros((node_count, params.branch_count - 1), dtype=np.float32)

  for n in range(node_count):
    for b in range(params.branch_count-1): # TODO vectorize over b
      split_vals[n, b] = bins[split_cols[n], split_bins[n, b]]

  # filter down to the number of nodes we actually used
  return Tree(
    node_count,
    split_cols[:node_count],
    split_vals,
    children[:node_count,:],
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
    tree.split_vals,
    tree.children,
    tree.node_means,
    values)
  return values

