from dataclasses import dataclass

from typing import Optional

import numpy as np

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
