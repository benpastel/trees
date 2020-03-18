from typing import Tuple

import xgboost as xgb
import numpy as np
import pandas as pd

from trees.tree import fit, predict
from trees.utils import timed, stats

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

  xg_kwargs = {
    'n_estimators': 10,
    'eta': 0.1,
    'tree_method': 'auto',
    'colsample_bytree': 0.9,
    'subsample': 0.9,
    'scale_pos_weight': 11.0
  }
  print(f'xgboost with args: {xg_kwargs}')

  with timed('  fit'):
    model = xgb.XGBClassifier(**xg_kwargs)
    model.fit(train_X, train_y.astype(int))

  with timed('  predict'):
    train_preds = model.predict(train_X)
    valid_preds = model.predict(valid_X)

  print(f'''  on credit default:
    train_preds bincount: {np.bincount(train_preds)}
    train: {stats(train_preds, train_y)}
    valid: {stats(valid_preds, valid_y)}
  ''')

  min_leaf_size = 100
  print(f'Classification tree with {min_leaf_size=}')
  with timed(f'  fit: '):
    model = fit(train_X, train_y.astype(bool), min_leaf_size=min_leaf_size)
  print(model)

  with timed('  predict:'):
    train_preds = predict(model, train_X)
    valid_preds = predict(model, valid_X)

  print(f'''  on credit default:
    train_preds bincount: {np.bincount(train_preds)}
    train: {stats(train_preds, train_y)}
    valid: {stats(valid_preds, valid_y)}
  ''')

  print(f'Classification tree with {min_leaf_size=}')
  with timed(f'  fit: '):
    model = fit(train_X, train_y.astype(float), min_leaf_size=min_leaf_size)
  print(model)

  with timed('  predict:'):
    train_preds = predict(model, train_X)
    valid_preds = predict(model, valid_X)

  print(f'''  on credit default:
    train_preds bincount: {np.bincount(train_preds)}
    train: {stats(train_preds, train_y)}
    valid: {stats(valid_preds, valid_y)}
  ''')




