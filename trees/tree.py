from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np

from trees.params import Params
from trees.splits import Split, choose_split


class Node:  
  # indices in the input data contained in this node
  def __init__(self, idx: np.ndarray, depth: int):
    self.idx = idx
    self.depth = depth

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
      return f'{indent}value: {self.value:.2f}, count: {len(self.idx)}\n'
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
  root = Node(all_indices, depth=1)
  open_nodes = [root]
  all_nodes = [root]

  preds = np.zeros_like(y)
  pred_count = np.zeros(len(preds), dtype=int) # TODO remove

  while len(open_nodes) > 0:
    node = open_nodes.pop()

    if node.depth == params.max_depth or (split := choose_split(node.idx, X, y, params)) is None:
      # leaf
      node.value = np.mean(y[node.idx])
      preds[node.idx] = node.value
      pred_count[node.idx] += 1
    else:
      # not leaf
      node.split = split
      node.left_child = Node(split.left_idx, node.depth + 1)
      node.right_child = Node(split.right_idx, node.depth + 1)
      open_nodes += [node.left_child, node.right_child]
      all_nodes += [node.left_child, node.right_child]

  assert np.all(pred_count == 1)
  return Tree(root, all_nodes), preds


def eval_tree(tree: Tree, X: np.ndarray) -> np.ndarray:
  assert X.dtype == np.uint8
  assert X.ndim == 2

  # BFS through nodes so that we know the parent indices when we reach the child
  indices = [[] for _ in tree.nodes]

  # represent all nodes by their position in tree.nodes
  # so that we can use the same position fo the indices
  r = tree.nodes.index(tree.root)
  open_nodes = [r]
  indices[r] = np.arange(X.shape[0], dtype=np.intp)

  values = np.zeros(len(X))

  while len(open_nodes) > 0:
    n = open_nodes.pop()
    node = tree.nodes[n]
    idx = indices[n]

    if node.split is not None:
      vals = X[idx, node.split.column]

      left_n = tree.nodes.index(node.left_child)
      right_n = tree.nodes.index(node.right_child)

      indices[left_n] = idx[vals <= node.split.value]
      indices[right_n] = idx[vals > node.split.value]

      open_nodes += [left_n, right_n]
    else:
      values[idx] = node.value

  return values

