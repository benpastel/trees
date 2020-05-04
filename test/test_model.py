import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from trees.model import fit, predict, Model
from trees.params import Params
import pprint

def test_model():
  pp = pprint.PrettyPrinter(indent=4)

  X = np.random.random((100, 3))
  y = np.random.random(100)

  model, preds_from_fit1 = fit(X, y, Params(tree_count=1, learning_rate=1.0, smooth_factor=0.0))
  preds_from_predict = predict(model, X)
  assert_array_almost_equal(preds_from_fit1, preds_from_predict)

  model, preds_from_fit10 = fit(X, y, Params(tree_count=10, learning_rate=1.0, smooth_factor=0.0))
  preds_from_predict = predict(model, X)
  assert_array_almost_equal(preds_from_fit10, preds_from_predict)

  # training loss should decrease when we add more trees
  loss_1 = np.sum(np.abs(y - preds_from_fit1))
  loss_10 = np.sum(np.abs(y - preds_from_fit10))
  assert loss_1 > loss_10










