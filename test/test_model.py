import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from trees.model import fit, predict, Model
from trees.params import Params
import pprint

def test_model_with_bfs():
  pp = pprint.PrettyPrinter(indent=4)

  X = np.random.random((100, 3)).astype(np.float32)
  y = np.random.random(100).astype(np.float32)

  model, preds_from_fit1 = fit(X, y, Params(use_bfs_tree=True, tree_count=1, learning_rate=1.0, bfs_smooth_factor=0.0, bucket_count=256))
  preds_from_predict = predict(model, X)
  assert_array_almost_equal(preds_from_fit1, preds_from_predict)

  model, preds_from_fit2 = fit(X, y, Params(use_bfs_tree=True, tree_count=1, learning_rate=1.0, bfs_smooth_factor=0.0, bucket_count=2))
  preds_from_predict = predict(model, X)
  assert_array_almost_equal(preds_from_fit2, preds_from_predict)

  model, preds_from_fit10 = fit(X, y, Params(use_bfs_tree=True, tree_count=10, learning_rate=1.0, bfs_smooth_factor=0.0, bucket_count=256))
  preds_from_predict = predict(model, X)
  assert_array_almost_equal(preds_from_fit10, preds_from_predict)

  loss_1 = np.sum(np.abs(y - preds_from_fit1))
  loss_10 = np.sum(np.abs(y - preds_from_fit10))
  assert loss_1 > loss_10 - 0.0001, "training loss should decrease when we add more trees"

  loss_2 = np.sum(np.abs(y - preds_from_fit2))
  assert loss_1 < loss_2, "training loss should increase with fewer buckets"


def test_model_with_dfs():
  pp = pprint.PrettyPrinter(indent=4)

  X = np.random.random((100, 3)).astype(np.float32)
  y = np.random.random(100).astype(np.float32)

  model, preds_from_fit1 = fit(X, y, Params(use_bfs_tree=False, dfs_max_nodes=3, tree_count=1, learning_rate=1.0, bucket_count=256))
  preds_from_predict = predict(model, X)
  assert_array_almost_equal(preds_from_fit1, preds_from_predict)

  model, preds_from_fit2 = fit(X, y, Params(use_bfs_tree=False, dfs_max_nodes=3, tree_count=1, learning_rate=1.0, bucket_count=2))
  preds_from_predict = predict(model, X)
  assert_array_almost_equal(preds_from_fit2, preds_from_predict)

  model, preds_from_fit10 = fit(X, y, Params(use_bfs_tree=False, dfs_max_nodes=3, tree_count=10, learning_rate=1.0, bucket_count=256))
  preds_from_predict = predict(model, X)
  assert_array_almost_equal(preds_from_fit10, preds_from_predict)

  model, preds_from_fit3 = fit(X, y, Params(use_bfs_tree=False, dfs_max_nodes=64, tree_count=1, learning_rate=1.0, bucket_count=256))
  preds_from_predict = predict(model, X)
  assert_array_almost_equal(preds_from_fit3, preds_from_predict)

  loss_1 = np.sum(np.abs(y - preds_from_fit1))
  loss_2 = np.sum(np.abs(y - preds_from_fit2))
  loss_10 = np.sum(np.abs(y - preds_from_fit10))
  loss_3 = np.sum(np.abs(y - preds_from_fit3))
  assert loss_1 > loss_3 - 0.0001, "training loss should decrease when we add more nodes"
  assert loss_1 < loss_2 + 0.0001, "training loss should increase with fewer buckets"
  assert loss_1 > loss_10 - 0.0001, "training loss should decrease with more trees"







