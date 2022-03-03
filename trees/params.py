from dataclasses import dataclass

@dataclass
class Params:
  use_bfs_tree: bool = True

  # Model params
  tree_count: int = 10
  learning_rate: float = 0.3
  bucket_count: int = 64
  bucket_sample_count: int = 10000
  trees_per_bucketing: int = 2

  # BFS tree parameters
  bfs_smooth_factor: float = 100.0
  bfs_weight_smooth_factor: float = 100.0
  bfs_max_nodes: int = 3280
  bfs_max_depth: int = 8
  bfs_third_split_penalty: float = 4.0

  # DFS tree parameters
  dfs_max_nodes: int = 64 # lightgbm defaults to 31 max leaves, let's try that?



