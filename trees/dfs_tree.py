from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np

from trees.params import Params


def variance(
    sum_sqs: float,
    sum_: float,
    count: float
  ) -> float:
  assert count > 0
  # TODO will need to inline
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
  # TODO will need to inlined
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


# unlike the trinary tree we use in BFS, this is a normal binary tree
@dataclass
class Tree:
  node_count: int
  split_cols: np.ndarray
  split_vals: np.ndarray
  left_children: np.ndarray
  right_children: np.ndarray
  node_means: np.ndarray

def update_histograms(
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
  y_in_node = y[memberships == n]

  for c in range(cols):
    for v in range(vals):
      matching_y = y_in_node[X_in_node[:, c] == v]
      hist_counts[n,c,v] = len(matching_y)
      hist_sums[n,c,v] = matching_y.sum()
      hist_sum_sqs[n,c,v] = np.sum(matching_y * matching_y)


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
  the eventual c function will run over all columns
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
    print(f"  b={b}: {left_var} ({left_sum_sq}, {left_sum}, {left_count}), {right_var} ({right_sum_sq}, {right_sum}, {right_count}), {gain}")

    if gain > best_gain:
      best_gain = float(gain)
      best_bin = b

  return best_gain, best_bin


def update_node_splits(
    n: int,

    hist_counts: np.ndarray,
    hist_sums: np.ndarray,
    hist_sum_sqs: np.ndarray,

    node_gains: np.ndarray,
    split_cols: np.ndarray,
    split_bins: np.ndarray

  ) -> None:
  # set split_cols[n] to the best column to split this node on
  # and node_gains[n] to the gain from taking that split
  #
  # will be in C
  nodes, cols, vals = hist_counts.shape
  assert hist_counts.shape == hist_sums.shape == hist_sum_sqs.shape
  assert node_gains.shape == split_cols.shape == split_bins.shape == (nodes,)
  assert 0 <= n < nodes
  assert 0 < cols

  for c in range(cols):
    print(f"c={c}")
    gain, split_bin = _best_col_split(
      hist_counts[n,c],
      hist_sums[n,c],
      hist_sum_sqs[n,c],
    )
    if gain > node_gains[n]:
      node_gains[n] = gain
      split_cols[n] = c
      split_bins[n] = split_bin


def update_memberships_and_counts(
    c: int,
    parent: int,
    left_child: int,
    right_child: int,
    split_val: int,
    X: np.ndarray,
    memberships: np.ndarray,
    node_counts: np.ndarray
) -> None:
  # will be in C
  rows, cols = X.shape
  assert memberships.shape == (rows,)
  assert left_child < right_child
  assert 0 <= c < cols

  split_vals = X[memberships == parent, c]

  new_memberships = np.full(len(split_vals), right_child, dtype=memberships.dtype)
  new_memberships[split_vals <= split_val] = left_child

  memberships[memberships == parent] = new_memberships

  node_counts[left_child] = np.count_nonzero(new_memberships == left_child)
  node_counts[right_child] = np.count_nonzero(new_memberships == right_child)


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
  assert 2 <= params.bucket_count <= 256, 'buckets must fit in uint8'
  assert 0 < rows < 2**32-1, 'rows must fit in uint32'
  assert 0 < cols < 2**32-1, 'cols must fit in uint32'

  # TODO split DFS and BFS specific Params off
  # currenly only respects max nodes, and we want a different default value
  max_nodes = params.max_nodes
  assert 0 < max_nodes < 2**16-1, 'nodes must fit in uint16'

  split_cols = np.zeros(max_nodes, dtype=np.uint64)
  split_bins = np.zeros(max_nodes, dtype=np.uint8)
  left_children = np.zeros(max_nodes, dtype=np.uint16)
  right_children = np.zeros(max_nodes, dtype=np.uint16)

  # row => node it belongs to
  # initially belong to root (0)
  memberships = np.zeros(rows, dtype=np.uint16)

  # node stats:
  #   - count of rows in the node
  #   - best gain from splitting at this node
  node_counts = np.zeros(max_nodes, dtype=np.uint64)
  node_gains = np.full(max_nodes, -np.inf, dtype=np.float64)

  # histograms
  # node, col, val => stat where X[c] == val in this node
  #   - count of rows
  #   - sum of y
  #   - sum of y^2
  hist_counts = np.zeros((max_nodes, cols, params.bucket_count), dtype=np.uint32)
  hist_sums = np.zeros((max_nodes, cols, params.bucket_count), dtype=np.float64)
  hist_sum_sqs = np.zeros((max_nodes, cols, params.bucket_count), dtype=np.float64)

  # root count
  node_counts[0] = rows

  # root histograms
  update_histograms(0, memberships, X, y, hist_counts, hist_sums, hist_sum_sqs)

  # root node
  update_node_splits(
    0,
    hist_counts,
    hist_sums,
    hist_sum_sqs,
    node_gains,
    split_cols,
    split_bins,
  )

  # grow the tree
  node_count = 1
  while node_count + 2 <= max_nodes:
    # can add 2 children and not have too many nodes

    # choose the node & column with the lowest score
    # can split leaves with enough data
    can_split_node = (node_counts >= 2) & (left_children == 0)
    node_gains[~can_split_node] = -np.inf

    print(f"best gain: {node_gains.max()}")

    if node_gains.max() <= 0:
      # can't improve anymore
      # TODO plus a splitting penalty
      break

    split_n = np.argmax(node_gains)
    split_c = split_cols[split_n]
    split_bin = split_bins[split_n]

    print(f"split: {split_n, split_c, split_bin}")

    # make the split
    left_children[split_n] = left_child = node_count
    right_children[split_n] = right_child = node_count + 1
    node_count += 2
    update_memberships_and_counts(
      int(split_c),
      int(split_n),
      left_child,
      right_child,
      split_bin,
      X,
      memberships,
      node_counts,
    )

    # update histograms
    if node_counts[left_child] < node_counts[right_child]:
      # calculate left
      update_histograms(left_child, memberships, X, y, hist_counts, hist_sums, hist_sum_sqs)

      # find right via subtraction
      hist_counts[right_child] = hist_counts[split_n] - hist_counts[left_child]
      hist_sums[right_child] = hist_sums[split_n] - hist_sums[left_child]
      hist_sum_sqs[right_child] = hist_sum_sqs[split_n] - hist_sum_sqs[left_child]
    else:
      # calculate right
      update_histograms(right_child, memberships, X, y, hist_counts, hist_sums, hist_sum_sqs)

      # find left via subtraction
      hist_counts[left_child] = hist_counts[split_n] - hist_counts[right_child]
      hist_sums[left_child] = hist_sums[split_n] - hist_sums[right_child]
      hist_sum_sqs[left_child] = hist_sum_sqs[split_n] - hist_sum_sqs[right_child]

    # find the best splits for each new node
    update_node_splits(
      left_child,
      hist_counts,
      hist_sums,
      hist_sum_sqs,
      node_gains,
      split_cols,
      split_bins,
    )

    update_node_splits(
      right_child,
      hist_counts,
      hist_sums,
      hist_sum_sqs,
      node_gains,
      split_cols,
      split_bins,
    )


  # finished growing the tree

  # prediction for each row is the mean of the node the row is in
  node_means = np.zeros(node_count)
  for n in range(node_count):
    if np.any(memberships == n):
      node_means[n] = np.mean(y[memberships == n])
  preds = node_means[memberships]

  # convert the splits from binned uint8 values => original float32 values
  split_vals = np.zeros(node_count, dtype=np.float32)
  for n in range(node_count):
    split_vals[n] = bins[split_cols[n], split_bins[n]]

  # filter down to the number of nodes we actually used
  return Tree(
    node_count,
    split_cols[:node_count],
    split_vals,
    left_children[:node_count],
    right_children[:node_count],
    node_means
  ), preds


def c_eval_tree(
  X: np.ndarray,
  split_cols: np.ndarray,
  split_vals: np.ndarray,
  left_children: np.ndarray,
  right_children: np.ndarray,
  node_means: np.ndarray,
  out_vals: np.ndarray
) -> None:
  # this will be in C
  rows, feats = X.shape

  for r in range(rows):

    # start at the root
    n = 0

    # find the leaf for this row
    # TODO: change representation to assume right is left + 1
    # so we can just choose whether to add that
    while left_children[n]:
      if X[r, split_cols[n]] <= split_vals[n]:
        # go left
        n = left_children[n]
      else:
        # go right
        n = right_children[n]

    # the predicted value is the mean of the leaf we ended up in
    out_vals[r] = node_means[n]


def eval_tree(tree: Tree, X: np.ndarray) -> np.ndarray:
  # this will be the python wrapper for the c function
  assert X.dtype == np.float32
  rows, feats = X.shape

  values = np.zeros(rows, dtype=np.double)
  c_eval_tree(
    X,
    tree.split_cols,
    tree.split_vals,
    tree.left_children,
    tree.right_children,
    tree.node_means,
    values)
  return values
