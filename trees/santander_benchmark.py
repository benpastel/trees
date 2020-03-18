from typing import Tuple

import xgboost as xgb
import numpy as np
import pandas as pd

from trees.tree import fit, predict
from trees.utils import timed, stats


def regression_stats(preds, trues):
  assert preds.shape == trues.shape
  mse = np.mean((preds - trues)  * (preds - trues))
  mae = np.mean(np.abs(preds - trues))
  return f'MSE: {mse:.1f}, MAE: {mae:.1f}'


def load_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
  frame = pd.read_csv(path)

  y = frame['target'].values

  X = np.zeros((len(frame), 4991))
  c = 0
  cols_to_skip = [
    'ID',
    'target'
  ]
  for name, contents in frame.iteritems():
    if name not in cols_to_skip:
      X[:, c] = contents.values
      c += 1
  assert c == 4991
  X[np.isnan(X)] = -1
  return X, y


if __name__ == '__main__':
  with timed('loading data...'):
    # split a validation set
    all_X, all_y = load_data('data/santander/train.csv')
    valid_count = len(all_X) // 10 # 10%
    valid_set = np.random.choice(np.arange(len(all_X)), size=valid_count, replace=False)
    is_valid = np.zeros(len(all_X), dtype=bool)
    is_valid[valid_set] = True
    train_X = all_X[~is_valid]
    train_y = all_y[~is_valid]
    valid_X = all_X[is_valid]
    valid_y = all_y[is_valid]
  print(f'{train_X.shape=} {valid_X.shape=}')

  xg_kwargs = {'n_estimators': 1, 'tree_method': 'hist'}
  print(f'xgboost with args: {xg_kwargs}')

  with timed('  fit'):
    model = xgb.XGBRegressor(**xg_kwargs)
    model.fit(train_X, train_y)

  with timed('  predict'):
    train_preds = model.predict(train_X)
    valid_preds = model.predict(valid_X)

  print(f'''
    train_preds: min={np.min(train_preds):.1f}, max={np.max(train_preds):.1f}, mean={np.mean(train_preds):.1f}
    train: {regression_stats(train_preds, train_y)}
    valid: {regression_stats(valid_preds, valid_y)}
  ''')

  tree_params = {'min_leaf_size': 1, 'extra_leaf_penalty': 0.0, 'max_depth': 6}

  print(f'Classification tree with params {tree_params=}')
  with timed(f'  fit: '):
    model = fit(train_X, train_y, **tree_params)
  print(model)

  with timed('  predict:'):
    train_preds = predict(model, train_X)
    valid_preds = predict(model, valid_X)

  print(f''':
    train_preds: min={np.min(train_preds):.1f}, max={np.max(train_preds):.1f}, mean={np.mean(train_preds):.1f}
    train: {regression_stats(train_preds, train_y)}
    valid: {regression_stats(valid_preds, valid_y)}
  ''')




