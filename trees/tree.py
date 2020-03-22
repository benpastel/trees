from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np

from trees.params import Params
from trees.splits import Split, choose_split


class Node:  
  # indices in the input data contained in this node
  def __init__(self, depth: int, count: int):
    self.depth = depth
    self.count = count

    # stay None for leaves
    self.split: Optional[Split] = None
    self.left_child: Optional['Node'] = None
    self.right_child: Optional['Node'] = None

    # non-none for leaves
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
  assert y.shape == (X.shape[0],) 

  all_indices = np.arange(X.shape[0], dtype=np.intp)
  root = Node(depth=1, count=X.shape[0])
  all_nodes = [root]

  # for BFS splitting
  # indices show which rows are inside the corresponding node
  open_nodes = [root]
  open_indices = [all_indices]

  preds = np.zeros_like(y)
  pred_count = np.zeros(len(preds), dtype=int) # TODO remove

  while len(open_nodes) > 0:
    node = open_nodes.pop()
    idx = open_indices.pop()
    X_in_node = X[idx, :]
    y_in_node = y[idx]

    if node.depth == params.max_depth or (split := choose_split(X_in_node, y_in_node, params)) is None:
      # leaf
      node.value = np.mean(y_in_node)
      preds[idx] = node.value
      pred_count[idx] += 1
    else:
      # not leaf
      node.split = split
      left_idx = idx[X_in_node[:, split.column] <= split.value]
      right_idx = idx[X_in_node[:, split.column] > split.value]
      node.left_child = Node(node.depth + 1, len(left_idx))
      node.right_child = Node(node.depth + 1, len(right_idx))
      
      all_nodes += [node.left_child, node.right_child]
      open_nodes += [node.left_child, node.right_child]
      open_indices += [left_idx, right_idx]

  assert np.all(pred_count == 1)
  return Tree(root, all_nodes), preds


def eval_tree(tree: Tree, X: np.ndarray) -> np.ndarray:
  assert X.dtype == np.uint8
  assert X.ndim == 2

  # BFS through nodes so that we know the parent indices when we reach the child
  open_nodes = [tree.root]
  open_indices = [np.arange(X.shape[0], dtype=np.intp)]
  values = np.zeros(len(X))
  while len(open_nodes) > 0:
    node = open_nodes.pop()
    idx = open_indices.pop()

    if node.split is not None:
      vals = X[idx, node.split.column]

      left_idx = idx[vals <= node.split.value]
      right_idx = idx[vals > node.split.value]

      open_nodes += [node.left_child, node.right_child]
      open_indices += [left_idx, right_idx]
    else:
      values[idx] = node.value

  return values

