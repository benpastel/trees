from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np

from trees.params import Params
from trees.c.build_tree import build_tree, eval_tree as c_eval_tree

@dataclass
class Tree:
  node_count: int
  split_cols: np.ndarray
  split_vals: np.ndarray
  left_children: np.ndarray
  right_children: np.ndarray
  node_means: np.ndarray

def fit_tree(
    X: np.ndarray,
    y: np.ndarray,
    params: Params
) -> Tuple[Tree, np.ndarray]:
  assert X.dtype == np.uint8
  assert X.ndim == 2
  assert y.ndim == 1
  rows, feats = X.shape
  assert y.shape == (rows,)
  y = y.astype(np.double)
  assert 0 < params.max_nodes < 2**16
  assert 0 < params.min_leaf_size < 2**32
  assert 0 <= params.extra_leaf_penalty

  # output arrays for c function
  # pre-allocated to the max number of nodes
  split_cols = np.zeros((params.max_nodes,), dtype=np.uint64)
  split_vals = np.zeros((params.max_nodes,), dtype=np.uint8)
  left_children = np.zeros((params.max_nodes,), dtype=np.uint16)
  right_children = np.zeros((params.max_nodes,), dtype=np.uint16)
  node_means = np.zeros((params.max_nodes,), dtype=np.double)

  node_count = build_tree(
    X,
    y,
    split_cols,
    split_vals,
    left_children,
    right_children,
    node_means,
    params.extra_leaf_penalty,
    params.min_leaf_size)

  # filter down to the number of nodes we actually used
  tree = Tree(
    node_count,
    split_cols[:node_count],
    split_vals[:node_count],
    left_children[:node_count],
    right_children[:node_count],
    node_means[:node_count]
  )
  vals = eval_tree(tree, X)
  return tree, vals


def eval_tree(tree: Tree, X: np.ndarray) -> np.ndarray:
  assert X.dtype == np.uint8
  assert X.ndim == 2
  rows, feats = X.shape

  values = np.zeros(rows, dtype=np.double)
  c_eval_tree(
    X,
    tree.split_cols,
    tree.split_vals,
    tree.left_children,
    tree.right_children,
    tree.node_means,
    values)
  return values

