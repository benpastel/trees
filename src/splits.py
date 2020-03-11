from typing import Optional

import numpy as np

from common import Model, Node


MIN_LEAF_SIZE = 100


def gini_purity(A: np.ndarray) -> float:
  ''' 
  gini purity in the boolean case:
    1 - p_no * (1 - p_no) + p_yes * (1 - p_yes)
    1 - 2 * p_yes * p_no
    1 - 2 * (trues / total) * (falses / total)
    1 - 2 * trues * falses / total^2
  '''
  assert A.ndim == 1
  assert A.dtype == np.bool
  assert len(A) > 0

  trues = np.count_nonzero(A)
  falses = len(A) - trues

  return 1.0 - 2.0 * trues * falses / (len(A) * len(A))


def choose_split(
    node: Node, 
    X: np.ndarray, 
    y: np.ndarray, 
    feat_order: np.ndarray
) -> Optional[Node]:
  if np.count_nonzero(node.mask) < MIN_LEAF_SIZE * 2:
    # we need at least MIN_LEAF_SIZE points in both left child and right child
    return None

  total_purity = gini_purity(y[node.mask])
  print(f'purity: {total_purity}')

  return None