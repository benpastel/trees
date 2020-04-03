from dataclasses import dataclass


@dataclass
class Params:
  confidence: float = 0.8
  max_nodes: int = 128
  tree_count: int = 10
  learning_rate: float = 0.3

