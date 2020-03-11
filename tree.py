from dataclasses import dataclass
from typing import Optional

import numpy as np

MIN_LEAF_SIZE = 100

@dataclass 
class Split:
  column: int
  value: float
  left_child: 'Node'
  right_child: 'Node'


@dataclass
class Node:
  # boolean mask over input data
  # true where data is in leaf
  mask: np.ndarray

  # stays None for leaves
  split: Optional[Split] = None

  # non-none for leaves
  value: Optional[float] = None

  def __str__(self, level = 0):
    ''' recursively print the tree '''
    indent = '  ' * level
    if self.split is None:
      # leaf
      assert self.value is not None
      return f'{indent}value: {self.value:.4f}\n'
    else:
      # non-leaf
      return (f'{indent}[feature {self.split.column}] < {self.split.value}:\n'
        + __str__(self.split.left_child, level + 1)
        + f'{indent}{self.split.value} <= [feature {self.split.column}]:\n'
        + __str__(self.split.left_child, level + 1))


@dataclass
class Model:
  root: Node

  def __str__(self):
    return str(self.root)


def choose_split(
    node: Node, 
    X: np.ndarray, 
    y: np.ndarray, 
    feat_order: np.ndarray
) -> Optional[Node]:
  total = np.count_nonzero(node.mask)

  if total < MIN_LEAF_SIZE * 2:
    # we need at least MIN_LEAF_SIZE points in both left child and right child
    return None

  trues = np.count_nonzero(y[node.mask])
  falses = total - trues

  total_gini = 
  return None


def fit(X: np.ndarray, y: np.ndarray) -> Model:
  assert X.ndim == 2
  assert y.shape == (X.shape[0],)

  # start with binary classification over floats only
  X = X.astype(float)
  y = y.astype(bool)

  # sort each feature
  feat_order = X.argsort(axis=0)

  root = Node(mask = np.ones(X.shape[0], dtype=bool))
  open_nodes = [root]

  while len(open_nodes) > 0:
    node = open_nodes.pop()
    split = choose_split(node, X, y, feat_order)

    if split is None:
      # leaf
      node.value = np.mean(X[node.mask])
    else:
      # not a leaf
      open_nodes.append(split.left_child)
      open_nodes.append(split.right_child)

  return Model(root)












  










