from dataclasses import dataclass
from typing import Optional

import numpy as np

from splits import Split, choose_split


@dataclass
class Node:
  # indices in the input data contained in this node
  idx: np.ndarray

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
      return f'{indent}value: {self.value}, count: {len(self.idx)}\n'
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



def fit(X: np.ndarray, y: np.ndarray, min_leaf_size: int) -> Model:
  assert X.ndim == 2
  assert y.shape == (X.shape[0],)

  # start with binary classification over floats only
  X = X.astype(float)
  y = y.astype(bool)

  all_indices = np.arange(X.shape[0], dtype=np.intp)
  root = Node(all_indices)
  open_nodes = [root]

  while len(open_nodes) > 0:
    node = open_nodes.pop()

    split = choose_split(node.idx, X, y, min_leaf_size)

    if split is None:
      # leaf
      node.value = np.mean(y[node.idx])
    else:
      # not a leaf
      node.split = split
      node.left_child = Node(split.left_idx)
      node.right_child = Node(split.right_idx)
      open_nodes.append(node.left_child)
      open_nodes.append(node.right_child)

  return Model(root)


def predict(model: Model, X: np.ndarray) -> np.ndarray:
  assert X.ndim == 2
  probs = np.zeros(len(X))

  for i in range(X.shape[0]):
    node = model.root
    while node.split is not None:
      if X[i, node.split.column] <= node.split.value:
        node = node.left_child
      else:
        node = node.right_child
    probs[i] = node.value

  return probs >= 0.5



