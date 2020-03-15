from dataclasses import dataclass
from typing import Optional, List

import numpy as np

from trees.splits import Split, choose_split


@dataclass
class Node:
  # indices in the input data contained in this node
  idx: np.ndarray
  depth: int

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
  node_count: int
  leaf_count: int
  bucket_splits: List[np.ndarray]

  def __str__(self):
    return f'model with {self.node_count} nodes and {self.leaf_count} leaves'

BUCKET_COUNT = 256 # uint8 buckets
def choose_bucket_splits(
    X: np.ndarray, 
    bucket_count=BUCKET_COUNT
) -> List[np.ndarray]:
  # returns a list of bins for each column
  # where bins is an array of right inclusive endpoints:
  #   (values in bucket 0) <= bins[0] < (values in bucket 1) <= bins[1] ...
  # 
  # so that np.searchsorted(bins, vals) returns the buckets
  splits: List[np.ndarray] = []

  for col in range(X.shape[1]):
    uniqs = np.unique(X[:,col])

    if len(uniqs) <= bucket_count:
      # each value gets a bin
      bins = uniqs
    else:
      # assign a roughly equal amount of unique values to each bin
      # TODO weight by count
      bins = np.zeros(bucket_count, uniqs.dtype)
      for s in range(bucket_count):
        # average step by len(uniqs) // bucket_count
        # 
        # right endpoint is inclusive, so -1 
        # 
        # so start with (s + 1) * (len(uniqs)) // bucket_count) - 1
        # but then divide by bucket count last so that the extras a distributed evenly
        idx = ((s+1) * len(uniqs)) // bucket_count - 1
        bins[s] = uniqs[idx]
    splits.append(bins)
  return splits


def apply_bucket_splits(
  X: np.ndarray,
  splits: List[np.ndarray]
) -> np.ndarray:
  ''' returns X bucketed into the splits '''
  assert X.ndim == 2
  assert X.shape[1] == len(splits)

  bucketed = np.zeros(X.shape, dtype=np.uint8)
  for col, bins in enumerate(splits):

    # TODO: decide how to handle different dtypes
    assert X.dtype == bins.dtype 
    indices = np.searchsorted(bins, X[:, col])

    # in the training case, indices should always be < len(buckets)
    # in the prediction case, it's possible to see a new value outside the right endpoint
    # include those in the rightmost bucket
    bucketed[:, col] = np.minimum(len(bins) - 1, indices)
  return bucketed


MAX_DEPTH = 6
def fit(
    X: np.ndarray, 
    y: np.ndarray, 
    min_leaf_size: int, 
    max_depth: int = MAX_DEPTH
) -> Model:
  assert X.ndim == 2
  assert y.shape == (X.shape[0],)

  bucket_splits = choose_bucket_splits(X)
  X = apply_bucket_splits(X, bucket_splits)
  assert X.dtype == np.uint8

  # start with binary classification only
  y = y.astype(bool)

  all_indices = np.arange(X.shape[0], dtype=np.intp)
  root = Node(all_indices, depth=1)
  node_count = 1
  leaf_count = 0
  open_nodes = [root]

  while len(open_nodes) > 0:
    node = open_nodes.pop()

    if node.depth == max_depth or (split := choose_split(node.idx, X, y, min_leaf_size)) is None:
      # leaf
      node.value = np.mean(y[node.idx])
      leaf_count += 1
    else:
      # not a leaf
      node.split = split
      node.left_child = Node(split.left_idx, node.depth + 1)
      node.right_child = Node(split.right_idx, node.depth + 1)
      open_nodes.append(node.left_child)
      open_nodes.append(node.right_child)
      node_count += 2

  return Model(root, node_count, leaf_count, bucket_splits)


def predict(model: Model, X: np.ndarray) -> np.ndarray:
  assert X.ndim == 2
  X = apply_bucket_splits(X, model.bucket_splits)

  probs = np.zeros(len(X))

  for i in range(len(X)):
    node = model.root
    while node.split is not None:
      if X[i, node.split.column] <= node.split.value:
        node = node.left_child
      else:
        node = node.right_child
    probs[i] = node.value

  return probs >= 0.5



