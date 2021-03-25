from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np

from trees.params import Params
from trees.c.tree import build_dfs_tree

from trees.bfs_tree import Tree, eval_tree

# evaluation is the same as bfs_tree
# TODO move to shared location
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
  nodes, cols, vals = hist_counts.shape
  rows, _ = X.shape
  assert hist_counts.shape == hist_sums.shape == hist_sum_sqs.shape
  assert cols == X.shape[1]
  assert (rows,) == y.shape
  assert memberships.shape = (rows,)

  X_in_node = X[memberships == n]

  for c in range(cols):
    for v in range(vals):
      matching_rows = (X_in_node[:, c] == v)
      hist_counts[0][c][v] = np.count_nonzero(matching_rows)
      hist_sums[0][c][v] = np.sum(y[matching_rows])
      hist_sum_sqs[0][c][v] = np.sum(y[matching_rows] * y[matching_rows])


def calc_node_scores(
    n: int,
    counts: np.ndarray,
    sums: np.ndarray,
    sum_sqs: np.ndarray,
    scores: np.ndarray,
    penalty: float,
    third_split_penalty: float
  ) -> None:
  nodes, cols, vals = counts.shape
  assert scores.shape == (nodes, cols)
  assert counts.shape == sums.shape == sum_sqs.shape
  assert n < nodes

  for c in range(cols):
    total_count = counts[n,c,:].sum()
    total_sum = sums[n,c,:].sum()
    total_sum_sqs = sum_sqs[n,c,:].sum()

    left_count = 0
    left_sum = 0
    left_sum_sqs = 0

    for lo in range(vals-2):
      left_count += counts[n,c,lo]
      left_sum += sums[n,c,lo]
      left_sum_sqs += sum_sqs[n,c,lo]

      # force non-empty left split
      if left_count == 0: continue

      left_var = left_sum_sqs - (left_sum * left_sum / left_count)

      mid_count = 0
      mid_sum = 0
      mid_sum_sqs = 0

      for hi in range(lo, vals-1):
        if (hi > lo):
          # middle split is nonempty
          # penalize it by a factor
          split_penalty = penalty + third_split_penalty * penalty

          mid_count += counts[n,c,hi]
          mid_sum += sums[n,c,hi]
          mid_sum_sq += sum_sqs[n,c,hi]
        else:
          # middle split is empty
          split_penalty = penalty

        right_count = total_count - left_count - mid_count
        right_sum = total_sum - left_sum - mid_sum
        right_sum_sq = total_sum_sqs - left_sum_sq - mid_sum_sq

        # force non-empty right split
        if (right_count == 0): break

        # score is weighted average of splits' variance
        mid_var = (mid_count == 0) ? 0 : mid_sum_sq - (mid_sum * mid_sum / mid_count)
        right_var = right_sum_sq - (right_sum * right_sum / right_count)
        scores[n, c] = (left_var + mid_var + right_var + split_penalty) / total_count


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
  node_means = np.zeros(max_nodes, dtype=np.float64)

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

  calc_histograms(0, memberships, X, y, hist_counts, hist_sums, hist_sum_sqs)

  assert hist_counts[0,:,:].sum() == rows

  root_sum = hist_sums[0,:,:].sum()
  root_sum_sqs = hist_sum_sqs[0,:,:].sum()
  root_var = (root_sum_sq / rows) - (root_sum / rows) * (root_sum / rows);
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

  for node_count in range(max_nodes):
    # can split leaves with enough data
    can_split_node = (node_counts > 1) & (left_children != 0)
    scores[~can_split_node] = np.inf # TODO max float
    split_n, split_c = argmax(scores[can_split_node]) # ravel

    # TODO LEFT OFF HERE (and didn't test anything)

    split_lo, split_hi, score = best_split_values(hists, split_n, split_c)

    # TODO update node metadata (children, parents) here
    split_cols[split_n] = split_c
    split_lo_bins[split_n] = split_lo
    split_hi_bins[split_n] = split_hi

    new_mean = split(X, preds, memberships)

    # TODO update hists here


  # prediction for each row is the mean of the node the row is in
  # TODO de-bias
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
    node_means[:node_count]
  ), preds

