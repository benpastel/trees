from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np

from trees.params import Params
from trees.splits import Split, choose_split


class Node:
  # indices in the input data contained in this node
  def __init__(self, depth: int, count: int, parent: Optional['Node'] = None):
    self.depth = depth
    self.count = count
    self.parent = parent

    self.split: Optional[Split] = None
    self.left_child: Optional['Node'] = None
    self.right_child: Optional['Node'] = None
    self.value: Optional[float] = None

  def __str__(self, level = 0):
    ''' recursively print the tree '''
    indent = '    ' * level
    if self.split is None:
      # leaf
      return f'{indent}value: {self.value:.2f}, count: {self.count}\n'
    else:
      # non-leaf
      return (f'{indent}feature {self.split.column} at <= {self.split.value}:\n'
        + self.left_child.__str__(level + 1)
        + self.right_child.__str__(level + 1))


@dataclass
class Tree:
  root: Node
  nodes: List[Node]

  def __str__(self):
    leaf_count = sum([1 for node in self.nodes if node.split is None])
    return f'tree with {len(self.nodes)} nodes and {leaf_count} leaves:\n{self.root}'


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

  all_indices = np.arange(rows, dtype=np.intp)
  root = Node(1, rows)
  all_nodes = [root]

  # for DFS splitting
  # indices show which rows are inside the corresponding node
  open_nodes = [root]
  open_indices = [all_indices]

  preds = np.zeros_like(y)

  while len(open_nodes) > 0:
    node = open_nodes.pop()
    idx = open_indices.pop()
    X_in_node = X[idx, :]
    y_in_node = y[idx]

    if node.parent is None:
      parent_score = np.var(y)
    else:
      parent_score = node.parent.split.score

    if node.depth == params.max_depth or \
        (split := choose_split(X_in_node, y_in_node, parent_score, params)) is None:
      # leaf
      node.value = np.mean(y_in_node)
      preds[idx] = node.value
    else:
      # not leaf
      node.split = split
      is_left = (X_in_node[:, split.column] <= split.value)

      left_idx = idx[is_left]
      right_idx = idx[~is_left]
      node.left_child = Node(node.depth + 1, len(left_idx), parent=node)
      node.right_child = Node(node.depth + 1, len(right_idx), parent=node)

      all_nodes += [node.left_child, node.right_child]
      open_nodes += [node.left_child, node.right_child]
      open_indices += [left_idx, right_idx]

  return Tree(root, all_nodes), preds


def eval_tree(tree: Tree, X: np.ndarray) -> np.ndarray:
  assert X.dtype == np.uint8
  assert X.ndim == 2

  # DFS through nodes so that we know the parent indices when we reach the child
  open_nodes = [tree.root]
  open_indices = [np.arange(X.shape[0], dtype=np.intp)]
  values = np.zeros(len(X))
  while len(open_nodes) > 0:
    node = open_nodes.pop()
    idx = open_indices.pop()

    if node.split is not None:
      is_left = (X[idx, node.split.column] <= node.split.value)

      left_idx = idx[is_left]
      right_idx = idx[~is_left]

      open_nodes += [node.left_child, node.right_child]
      open_indices += [left_idx, right_idx]
    else:
      values[idx] = node.value

  return values

