from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np

from trees.params import Params
from trees.c.build_tree import build_tree

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

  # output arrays for c function
  # pre-allocated to the max number of nodes
  # TODO assert arrays are nice
  split_cols = np.zeros((params.max_nodes,), dtype=np.uint64)
  split_vals = np.zeros((params.max_nodes,), dtype=np.uint8)
  left_children = np.zeros((params.max_nodes,), dtype=np.uint16)
  right_children = np.zeros((params.max_nodes,), dtype=np.uint16)
  node_means = np.zeros((params.max_nodes,), dtype=np.double)

  node_count = build_tree(
    X.copy(), # TODO
    y.copy(),
    split_cols,
    split_vals,
    left_children,
    right_children,
    node_means,
    params.extra_leaf_penalty,
    params.min_leaf_size)

  # filter down to the number of nodes we actually used
  return Tree(
    node_count,
    split_cols[:node_count],
    split_vals[:node_count],
    left_children[:node_count],
    right_children[:node_count],
    node_means[:node_count]
  )


def eval_tree(tree: Tree, X: np.ndarray) -> np.ndarray:
  assert X.dtype == np.uint8
  assert X.ndim == 2
  rows, feats = X.shape

  # TODO move to C

  # DFS through nodes so that we know the parent indices when we reach the child
  open_nodes = [0]
  open_indices = [np.arange(rows, dtype=np.intp)]
  values = np.zeros(rows)
  value_set_count = np.zeros(rows, dtype=int)

  while len(open_nodes) > 0:
    n = open_nodes.pop()
    idx = open_indices.pop()

    if tree.left_children[n] == 0 and tree.right_children[n] == 0:
      # leaf
      values[idx] = node.value
      value_set_count[idx] += 1
    else:
      is_left = (X[idx, tree.split_cols[n]] <= tree.split_vals[n])

      left_idx = idx[is_left]
      right_idx = idx[~is_left]

      open_nodes += [tree.left_children[n], node.right_children[n]]
      open_indices += [left_idx, right_idx]

  assert np.all(value_set_count == 1)
  return values

