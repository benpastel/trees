# python versions of c functions, for testing
import numpy as np
from typing import Tuple, Optional
from trees.dfs_tree import _VERBOSE

def update_memberships(
  X: np.ndarray,
  parent_members: np.ndarray,
  left_members: np.ndarray,
  right_members: np.ndarray,
  split_c: int,
  split_bin: int,
) -> None:
  is_left = (X[parent_members, split_c] <= split_bin)
  left_members[:] = parent_members[is_left]
  right_members[:] = parent_members[~is_left]


def variance(
    sum_sqs: float,
    sum_: float,
    count: float
  ) -> float:
  assert count > 0
  # TODO bessel's correction?
  return (sum_sqs - (sum_ * sum_) / count) / count


def calc_gain(
  left_var: float,
  right_var: float,
  parent_var: float,
  left_count: int,
  right_count: int,
  parent_count: int
) -> float:
  assert left_count + right_count == parent_count
  # how good is a potential split?
  #
  # TODO add penalty back in
  #
  # we'll start with the reduction in train MSE:
  #   higher is better
  #   above 0 means we made an improvement
  #   below 0 means we got worse
  #
  # predictions at leaves are E(X), so train MSE is sum((X - E(X))^2)
  #  = (variance at each leaf) * (number of examples in that leaf)
  #
  # variance is E(X - E(X))^2
  # E(X) will be the
  # i.e. (change in variance) * (number of rows that change applies to)

  old_mse = parent_var * parent_count

  new_mse = left_var * left_count + right_var * right_count

  return old_mse - new_mse


def _best_col_split(
  counts: np.ndarray,
  sums: np.ndarray,
  sum_sqs: np.ndarray,
) -> Tuple[float, Optional[int]]:
  '''
  return
    (score, split_bin),
    or (inf, None) if there is no valid split

  python-only helper function that runs on a single column
  the c function runs over all columns
  '''

  assert len(counts.shape) == 1
  assert counts.shape == sums.shape == sum_sqs.shape
  bins, = counts.shape

  total_count = counts.sum()
  total_sum = sums.sum()
  total_sum_sq = sum_sqs.sum()

  if total_count < 2:
    return -np.inf, None

  parent_var = variance(total_sum_sq, total_sum, total_count)
  if _VERBOSE:
    print(f"  parent:{parent_var}")

  left_count = 0
  left_sum = 0
  left_sum_sq = 0

  best_gain = -np.inf
  best_bin = None

  for b in range(bins-1):
    left_count += counts[b]
    left_sum += sums[b]
    left_sum_sq += sum_sqs[b]

    # force non-empty left split
    if left_count == 0 : continue

    right_count = total_count - left_count
    right_sum = total_sum - left_sum
    right_sum_sq = total_sum_sq - left_sum_sq

    # force non-empty right split
    if (right_count == 0): break

    left_var = variance(left_sum_sq, left_sum, left_count)
    right_var = variance(right_sum_sq, right_sum, right_count)
    gain = calc_gain(left_var, right_var, parent_var, left_count, right_count, total_count)
    if _VERBOSE:
      print(f"  b={b}: {left_var} ({left_sum_sq}, {left_sum}, {left_count}), {right_var} ({right_sum_sq}, {right_sum}, {right_count}), {gain}")

    if gain > best_gain:
      best_gain = float(gain)
      best_bin = b

  return best_gain, best_bin



def update_node_splits(
    hist_counts: np.ndarray,
    hist_sums: np.ndarray,
    hist_sum_sqs: np.ndarray,

    node_gains: np.ndarray,
    split_cols: np.ndarray,
    split_bins: np.ndarray,

    n: int,
  ) -> None:
  # set:
  #   split_cols[n] to the best column to split this node on
  #   split_bins[n] to the value to split that column
  #   node_gains[n] to the gain from taking the split
  #
  cols, vals = hist_counts.shape
  nodes = len(node_gains)
  assert hist_counts.shape == hist_sums.shape == hist_sum_sqs.shape
  assert node_gains.shape == split_cols.shape == split_bins.shape == (nodes,)
  assert 0 <= n < nodes
  assert 0 < cols

  for c in range(cols):
    if _VERBOSE:
      print(f"c={c}")
    gain, split_bin = _best_col_split(
      hist_counts[c],
      hist_sums[c],
      hist_sum_sqs[c],
    )
    if gain > node_gains[n]:
      node_gains[n] = gain
      split_cols[n] = c
      split_bins[n] = split_bin


def like_c_eval_tree(
  X: np.ndarray,
  split_cols: np.ndarray,
  split_vals: np.ndarray,
  left_children: np.ndarray,
  node_means: np.ndarray,
  out_vals: np.ndarray
) -> None:
  rows, feats = X.shape

  for r in range(rows):

    # start at the root
    n = 0

    # find the leaf for this row
    # TODO: change representation to assume right is left + 1
    # so we can just choose whether to add that
    while left_children[n]:

      # right child nodes are always one higher than left
      go_right = (X[r, split_cols[n]] > split_vals[n])

      n = left_children[n] + go_right

    # the predicted value is the mean of the leaf we ended up in
    out_vals[r] = node_means[n]
