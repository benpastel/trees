from typing import Tuple

import xgboost as xgb
import numpy as np
import pandas as pd

from tree import fit, predict
from utils import timed

import cProfile

def load_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
  frame = pd.read_csv(path)
  y = frame['TARGET'].values.astype(np.bool) 

  # for now we just take the numeric data
  X = np.zeros((len(frame), 104))
  c = 0
  cols_to_skip = [
    'SK_ID_CURR', # ids
    'TARGET'      # y
  ]
  for name, contents in frame.iteritems():
    if np.issubdtype(contents.dtype, np.number) and name not in cols_to_skip:
      X[:, c] = contents.values
      c += 1
  assert c == 104
  X[np.isnan(X)] = -1
  return X, y


if __name__ == '__main__':
  with timed('loading data...'):
    # split a validation set
    all_X, all_y = load_data('data/credit/application_train.csv')


    valid_count = len(all_X) // 5 # 20%
    valid_set = np.random.choice(np.arange(len(all_X)), size=valid_count, replace=False)
    is_valid = np.zeros(len(all_X), dtype=bool)
    is_valid[valid_set] = True
    train_X = all_X[~is_valid]
    train_y = all_y[~is_valid]
    valid_X = all_X[is_valid]
    valid_y = all_y[is_valid]
  print(f'{train_X.shape=} {valid_X.shape=}')

  with timed('fit XGB'):
    model = xgb.XGBClassifier(n_estimators = 1)
    model.fit(train_X, train_y)

  with timed('predict XGB'):
    train_preds = model.predict(train_X)
    valid_preds = model.predict(valid_X)

  print(f'''XGBoost on credit default, single tree:
    train accuracy: {100.0 * np.mean(train_preds == train_y):.2f}%
    test accuracy: {100.0 * np.mean(valid_preds == valid_y):.2f}%
  ''')

  min_leaf_size = 100

  with timed(f'Fit tree with min_leaf_size={min_leaf_size}: '):
    # with cProfile.Profile() as pr:
    model = fit(train_X, train_y, min_leaf_size=min_leaf_size)
  # pr.print_stats()

  with timed('predict tree'):
    train_preds = predict(model, train_X)
    valid_preds = predict(model, valid_X)

  print(f'''Tree on credit default, min_leaf_size = {min_leaf_size}:
    train accuracy: {100.0 * np.mean(train_preds == train_y):.2f}%
    test accuracy: {100.0 * np.mean(valid_preds == valid_y):.2f}%
  ''')