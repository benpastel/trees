from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np

from trees.params import Params
from trees.c.tree import build_dfs_tree

from trees.bfs_tree import Tree, eval_tree

# evaluation is the same as bfs_tree
assert eval_tree

def calc_histograms(
    n: int,
    memberships: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    hist_counts: np.ndarray,
    hist_sums: np.ndarray,
    hist_sum_sqs: np.ndarray
  ) -> None:
  # will be C
  nodes, cols, vals = hist_counts.shape
  rows, _ = X.shape
  assert hist_counts.shape == hist_sums.shape == hist_sum_sqs.shape
  assert cols == X.shape[1]
  assert (rows,) == y.shape
  assert memberships.shape == (rows,)

  X_in_node = X[memberships == n]

  for c in range(cols):
    for v in range(vals):
      matching_rows = (X_in_node[:, c] == v)
      hist_counts[0][c][v] = np.count_nonzero(matching_rows)
      hist_sums[0][c][v] = np.sum(y[matching_rows])
      hist_sum_sqs[0][c][v] = np.sum(y[matching_rows] * y[matching_rows])

def choose_split(
  counts: np.ndarray,
  sums: np.ndarray,
  sum_sqs: np.ndarray,
  penalty: float,
  third_split_penalty: float
) -> Tuple[float, int, int]:
  '''
  return (score, lo_split_val, hi_split_val)

  python-only interface
  '''

  assert len(counts.shape) == 1
  assert counts.shape == sums.shape == sum_sqs.shape
  vals, = counts.shape

  total_count = counts.sum()
  total_sum = sums.sum()
  total_sum_sq = sum_sqs.sum()

  left_count = 0
  left_sum = 0
  left_sum_sq = 0

  best_score = np.inf
  best_lo = -1
  best_hi = -1

  for lo in range(vals-2):
    left_count += counts[lo]
    left_sum += sums[lo]
    left_sum_sq += sum_sqs[lo]

    # force non-empty left split
    if left_count == 0: continue

    left_var = left_sum_sq - (left_sum * left_sum / left_count)

    mid_count = 0
    mid_sum = 0
    mid_sum_sq = 0

    for hi in range(lo, vals-1):
      if (hi > lo):
        # middle split is nonempty
        # penalize it by a factor
        split_penalty = penalty + third_split_penalty * penalty

        mid_count += counts[hi]
        mid_sum += sums[hi]
        mid_sum_sq += sum_sqs[hi]
      else:
        # middle split is empty
        split_penalty = penalty

      right_count = total_count - left_count - mid_count
      right_sum = total_sum - left_sum - mid_sum
      right_sum_sq = total_sum_sq - left_sum_sq - mid_sum_sq

      # force non-empty right split
      if (right_count == 0): break

      # score is weighted average of splits' variance
      mid_var = 0 if mid_count == 0 else mid_sum_sq - (mid_sum * mid_sum / mid_count)
      right_var = right_sum_sq - (right_sum * right_sum / right_count)
      score = (left_var + mid_var + right_var + split_penalty) / total_count

      if score < best_score:
        best_score = float(score)
        best_lo = lo
        best_hi = hi

  assert best_lo >= 0
  return best_score, best_lo, best_hi


def calc_node_scores(
    n: int,
    counts: np.ndarray,
    sums: np.ndarray,
    sum_sqs: np.ndarray,
    scores: np.ndarray,
    penalty: float,
    third_split_penalty: float
  ) -> None:
  # will be in C
  nodes, cols, vals = counts.shape
  assert scores.shape == (nodes, cols)
  assert counts.shape == sums.shape == sum_sqs.shape
  assert 0 <= n < nodes

  for c in range(cols):
    score, _, _ = choose_split(
      counts[n,c],
      sums[n,c],
      sum_sqs[n,c],
      penalty,
      third_split_penalty
    )
    scores[n, c] = score


def update_memberships(
    c: int,
    parent: int,
    left_child: int,
    mid_child: int,
    right_child: int,
    lo: int,
    hi: int,
    X: np.ndarray,
    memberships: np.ndarray
) -> None:
  rows, cols = X.shape
  assert memberships.shape == (rows,)
  assert left_child < right_child
  assert mid_child == 0 or (left_child < mid_child < right_child)
  assert lo <= hi
  assert 0 <= c < cols

  split_vals = X[memberships == parent, c]

  new_vals = np.zeros(len(split_vals), dtype=memberships.dtype)
  new_vals[split_vals <= lo] = left_child
  new_vals[lo < split_vals <= hi] = mid_child
  new_vals[hi < split_vals] = right_child

  memberships[memberships == parent] = new_vals


def subtract_histograms(
    target_node: int,
    parent_node: int,
    source_node_1: int,
    source_node_2: int,
    counts: np.ndarray,
    sums: np.ndarray,
    sum_sqs: np.ndarray
  ) -> None:
  nodes, cols, vals = counts.shape
  assert counts.shape == sums.shape == sum_sqs.shape
  assert 0 <= parent_node < target_node < nodes
  assert parent_node < source_node_1 < nodes
  assert parent_node < source_node_2 < nodes
  counts[target_node] = counts[parent_node] - counts[source_node_1] - counts[source_node_2]
  sums[target_node] = sums[parent_node] - sums[source_node_1] - sums[source_node_2]
  sum_sqs[target_node] = sum_sqs[parent_node] - sum_sqs[source_node_1] - sum_sqs[source_node_2]


def fit_tree(
    X: np.ndarray,
    y: np.ndarray,
    bins: np.ndarray,
    params: Params
) -> Tuple[Tree, np.ndarray]:
  rows, cols = X.shape
  assert X.dtype == np.uint8
  assert y.dtype == np.double
  assert y.shape == (rows,)
  assert bins.shape == (cols, params.bucket_count-1)
  assert bins.dtype == np.float32
  assert 0 <= params.smooth_factor
  assert 0 <= params.weight_smooth_factor
  assert 0 <= params.third_split_penalty
  assert 2 <= params.bucket_count <= 256, 'buckets must fit in uint8'
  assert 0 < rows < 2**32-1, 'rows must fit in uint32'
  assert 0 < cols < 2**32-1, 'cols must fit in uint32'

  # ignore bfs params and use fewer max nodes
  max_nodes = 64
  assert 0 < max_nodes < 2**16-1, 'nodes must fit in uint16'

  split_cols = np.zeros(max_nodes, dtype=np.uint64)
  split_lo_bins = np.zeros(max_nodes, dtype=np.uint8)
  split_hi_bins = np.zeros(max_nodes, dtype=np.uint8)
  left_children = np.zeros(max_nodes, dtype=np.uint16)
  mid_children = np.zeros(max_nodes, dtype=np.uint16)
  right_children = np.zeros(max_nodes, dtype=np.uint16)
  node_counts = np.zeros(max_nodes, dtype=np.uint64)

  # row => node it belongs to
  # initially belong to root (0)
  memberships = np.zeros(rows, dtype=np.uint16)

  # how much variance reduction would we gain by splitting at this node & column?
  scores = np.zeros((max_nodes, cols), dtype=np.float64)

  # histograms
  # node, col, val => stat where X[c] == val in this node
  #   - count of rows
  #   - sum of y
  #   - sum of y^2
  hist_counts = np.zeros((max_nodes, cols, params.bucket_count), dtype=np.uint32)
  hist_sums = np.zeros((max_nodes, cols, params.bucket_count), dtype=np.float64)
  hist_sum_sqs = np.zeros((max_nodes, cols, params.bucket_count), dtype=np.float64)

  # root histogram
  calc_histograms(0, memberships, X, y, hist_counts, hist_sums, hist_sum_sqs)

  # root count & variance
  assert hist_counts[0,:,:].sum() == rows
  node_counts[0] = rows
  root_sum = hist_sums[0,:,:].sum()
  root_sum_sqs = hist_sum_sqs[0,:,:].sum()
  root_var = (root_sum_sqs / rows) - (root_sum / rows) * (root_sum / rows);
  penalty = root_var * params.smooth_factor

  # find the score for splitting the root at each column
  calc_node_scores(
    0,
    hist_counts,
    hist_sums,
    hist_sum_sqs,
    scores,
    params.third_split_penalty,
    penalty
  )

  node_count = 1
  while node_count + 3 <= max_nodes:
    # can add 3 children and not have too many nodes

    # choose the node & column with the highest score
    # can split leaves with enough data
    can_split_node = (node_counts > 2) & (left_children != 0)
    scores[~can_split_node] = np.inf
    split_n, split_c = np.unravel_index(np.argmin(scores), scores.shape)

    # now choose which values to split on for that node & column
    # recompute for now because of the sampling idea
    score, split_lo, split_hi = choose_split(
      hist_counts[split_n, split_c],
      hist_sums[split_n, split_c],
      hist_sum_sqs[split_n, split_c],
      penalty,
      params.third_split_penalty
    )

    if score >= root_var:
      break
      # TODO parent score

    # make the split
    # update node metadata
    split_cols[split_n] = split_c
    split_lo_bins[split_n] = split_lo
    split_hi_bins[split_n] = split_hi

    if split_lo == split_hi:
      left_children[split_n] = node_count
      right_children[split_n] = node_count + 1
      node_count += 2
    else:
      left_children[split_n] = node_count
      mid_children[split_n] = node_count + 1
      right_children[split_n] = node_count + 2
      node_count += 3

    # split the data
    # by updating which row belongs to which node
    update_memberships(
      split_c,
      split_n,
      left_children[split_n],
      mid_children[split_n],
      right_children[split_n],
      split_lo,
      split_hi,
      X,
      memberships
    )

    # update counts
    # TODO variance too here
    node_counts[left_children[split_n]] = np.count_nonzero(memberships == left_children[split_n])
    node_counts[mid_children[split_n]] = np.count_nonzero(memberships == mid_children[split_n])
    node_counts[right_children[split_n]] = np.count_nonzero(memberships == right_children[split_n])

    # update histograms
    # smallest 2 splits are calculated
    # then largest is via subtraction
    if (node_counts[right_children[split_n]] > node_counts[left_children[split_n]] and
      node_counts[right_children[split_n]] > node_counts[mid_children[split_n]]):
      calc_histograms(left_children[split_n], memberships, X, y, hist_counts, hist_sums, hist_sum_sqs)
      if split_lo != split_hi:
        calc_histograms(mid_children[split_n], memberships, X, y, hist_counts, hist_sums, hist_sum_sqs)
      subtract_histograms(right_children[split_n], split_n, left_children[split_n], mid_children[split_n], hist_counts, hist_sums, hist_sum_sqs)
    elif (split_lo == split_hi or
      (node_counts[left_children[split_n]] > node_counts[mid_children[split_n]] and
      node_counts[left_children[split_n]] > node_counts[right_children[split_n]])):
      calc_histograms(right_children[split_n], memberships, X, y, hist_counts, hist_sums, hist_sum_sqs)
      if split_lo != split_hi:
        calc_histograms(mid_children[split_n], memberships, X, y, hist_counts, hist_sums, hist_sum_sqs)
      subtract_histograms(left_children[split_n], split_n, right_children[split_n], mid_children[split_n], hist_counts, hist_sums, hist_sum_sqs)
    else:
      calc_histograms(left_children[split_n], memberships, X, y, hist_counts, hist_sums, hist_sum_sqs)
      calc_histograms(right_children[split_n], memberships, X, y, hist_counts, hist_sums, hist_sum_sqs)
      subtract_histograms(mid_children[split_n], split_n, left_children[split_n], right_children[split_n], hist_counts, hist_sums, hist_sum_sqs)

  # prediction for each row is the mean of the node the row is in
  node_means = np.zeros(node_count)
  for n in range(node_count):
    node_means[n] = np.mean(y[memberships == n])
  preds = node_means[memberships]

  # convert the splits from binned uint8 values => original float32 values
  split_lo_vals = np.zeros(node_count, dtype=np.float32)
  split_hi_vals = np.zeros(node_count, dtype=np.float32)
  for n in range(node_count):
    split_lo_vals[n] = bins[split_cols[n], split_lo_bins[n]]
    split_hi_vals[n] = bins[split_cols[n], split_hi_bins[n]]

  # filter down to the number of nodes we actually used
  return Tree(
    node_count,
    split_cols[:node_count],
    split_lo_vals,
    split_hi_vals,
    left_children[:node_count],
    mid_children[:node_count],
    right_children[:node_count],
    node_means
  ), preds

