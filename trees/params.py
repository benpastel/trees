from dataclasses import dataclass


@dataclass
class Params:
  smooth_factor: float = 1.0
  max_nodes: int = 40
  tree_count: int = 10
  learning_rate: float = 0.3

