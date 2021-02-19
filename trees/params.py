from dataclasses import dataclass


@dataclass
class Params:
  smooth_factor: float = 4.0
  weight_smooth_factor: float = 4.0
  max_nodes: int = 10000
  max_depth: int = 10
  tree_count: int = 10
  learning_rate: float = 0.5
  third_split_penalty: float = 4.0
  bucket_count: int = 256
  bucket_sample_count: int = 10000
  trees_per_bucketing: int = 2

