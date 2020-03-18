import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_svmlight_file

from trees.utils import timed, binary_stats, regression_stats
from trees.tree import fit, predict


def print_stats(
    train_preds, train_trues, 
    valid_preds, valid_trues,
    is_regression):
  if is_regression:
    print(f'''
      train preds: min={np.min(train_preds):.1f}, max={np.max(train_preds):.1f}, mean={np.mean(train_preds):.1f}
      train: {regression_stats(train_preds, train_y)}
      valid: {regression_stats(valid_preds, valid_y)}
    ''')
  else:
    print(f'''
      train preds bincount: {np.bincount(train_preds)}
      train: {binary_stats(train_preds, train_y)}
      valid: {binary_stats(valid_preds, valid_y)}
    ''')


def split(all_X, all_y):
  assert all_X.ndim == 2
  assert all_y.ndim == 1
  assert all_X.shape[0] == len(all_y)
  valid_count = len(all_X) // 5 # 20%
  valid_idx = np.random.choice(np.arange(len(all_X)), size=valid_count, replace=False)
  is_valid = np.zeros(len(all_X), dtype=bool)
  is_valid[valid_idx] = True
  return all_X[~is_valid], all_y[~is_valid], all_X[is_valid], all_y[is_valid]


def load_agaricus():
  X_svm, y = load_svmlight_file('data/agaricus/agaricus.txt.train')
  return X_svm.toarray(), y.astype(np.bool)


def load_credit():
  frame = pd.read_csv('data/credit/application_train.csv')
  y = frame['TARGET'].values.astype(np.bool) 

  # for now we just take the numeric data
  X = np.zeros((len(frame), 104))
  c = 0
  cols_to_skip = [
    'SK_ID_CURR',
    'TARGET'
  ]
  for name, contents in frame.iteritems():
    if np.issubdtype(contents.dtype, np.number) and name not in cols_to_skip:
      X[:, c] = contents.values
      c += 1
  assert c == 104
  X[np.isnan(X)] = -1
  return X, y


def load_santander():
  frame = pd.read_csv('data/santander/train.csv')
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
  benchmark_names = [
    'Agaricus', 
    'Home Credit Default Risk', 
    'Santander Value', 
  ]

  load_data_functions = [
    load_agaricus,
    load_credit,
    load_santander
  ]

  xgboost_args = [
    {'n_estimators': 1, 'tree_method': 'exact'},
    {'n_estimators': 1, 'tree_method': 'hist'},
    {'n_estimators': 1, 'tree_method': 'hist'},
  ]

  tree_args = [
    {'min_leaf_size': 10, 'max_depth': 6, 'extra_leaf_penalty': 0.0},
    {'min_leaf_size': 10, 'max_depth': 6, 'extra_leaf_penalty': 0.0},
    {'min_leaf_size': 10, 'max_depth': 6, 'extra_leaf_penalty': 0.0},
  ]

  for b, name in enumerate(benchmark_names):
    print(f'\n\n{name}:\n')

    with timed('loading data...'):
      X, y = load_data_functions[b]()
      train_X, train_y, valid_X, valid_y = split(X, y)
      print(f'X.shape: train {train_X.shape}, valid {valid_X.shape}')

    # only handle regression or binary classification cases so far
    is_regression = (train_y.dtype != np.bool)

    if is_regression:
      print(f'regression targets with min={np.min(y):.1f}, max={np.max(y):.1f}, mean={np.mean(y):.1f}')
    else:
      print(f'binary classification with {np.count_nonzero(y)} true and {np.count_nonzero(~y)} false')

    with timed(f'train & predict xgboost with: {xgboost_args[b]}'):
      if is_regression:
        model = xgb.XGBRegressor(**xgboost_args[b])
      else:
        model = xgb.XGBClassifier(**xgboost_args[b])

      model.fit(train_X, train_y)
      train_preds = model.predict(train_X)
      valid_preds = model.predict(valid_X)

    print_stats(train_preds, train_y, valid_preds, valid_y, is_regression)

    with timed(f'train & predict our tree with {tree_args[b]}'):
      model = fit(train_X, train_y, **tree_args[b])

      train_preds = predict(model, train_X)
      valid_preds = predict(model, valid_X)

    print_stats(train_preds, train_y, valid_preds, valid_y, is_regression)

