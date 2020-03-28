from dataclasses import dataclass


@dataclass
class Params:
  min_leaf_size: int = 10
  extra_leaf_penalty: float = 0.0
  max_depth: int = 6
  tree_count: int = 10
  learning_rate: float = 0.3
