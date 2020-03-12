from dataclasses import dataclass
from typing import Optional

import numpy as np

from splits import Split, choose_split


@dataclass
class Node:
  # boolean mask over input data
  # true where data is in leaf
  mask: np.ndarray

  # stay None for leaves
  split: Optional[Split] = None
  left_child: Optional['Node'] = None
  right_child: Optional['Node'] = None

  # non-none for leaves
  value: Optional[float] = None

  def __str__(self, level = 0):
    ''' recursively print the tree '''
    indent = '    ' * level
    if self.split is None:
      # leaf
      return f'{indent}value: {self.value}, count: {np.count_nonzero(self.mask)}\n'
    else:
      # non-leaf
      return (f'{indent}feature {self.split.column} at <= {self.split.value}:\n'
        + self.left_child.__str__(level + 1)
        + self.right_child.__str__(level + 1))


@dataclass
class Model:
  root: Node

  def __str__(self):
    return f'model with nodes:\n{str(self.root)}'



def fit(X: np.ndarray, y: np.ndarray) -> Model:
  assert X.ndim == 2
  assert y.shape == (X.shape[0],)

  # start with binary classification over floats only
  X = X.astype(float)
  y = y.astype(bool)

  # sort each feature
  feat_order = X.argsort(axis=0)

  root = Node(np.ones(X.shape[0], dtype=bool))
  open_nodes = [root]

  while len(open_nodes) > 0:
    node = open_nodes.pop()
    print(f'considering: {node}')

    split = choose_split(node.mask, X, y, feat_order)

    if split is None:
      # leaf
      node.value = np.mean(y[node.mask])
    else:
      # not a leaf
      node.split = split
      node.left_child = Node(split.left_mask)
      node.right_child = Node(split.right_mask)
      open_nodes.append(node.left_child)
      open_nodes.append(node.right_child)

  return Model(root)
