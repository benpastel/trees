from dataclasses import dataclass


@dataclass
class Params:
  smooth_factor: float = 100.0
  max_nodes: int = 3280
  max_depth: int = 8
  tree_count: int = 10
  learning_rate: float = 0.3

