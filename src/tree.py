import numpy as np

from common import Model, Node
from splits import choose_split


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












  










