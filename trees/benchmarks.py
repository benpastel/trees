import os
import gc
import sys
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.datasets import load_svmlight_file

from trees.utils import timed, profiled, binary_stats, regression_stats
from trees.model import fit, predict, Params


def print_stats(
    train_preds, train_trues,
    valid_preds, valid_trues,
    is_regression):
  if is_regression:
    print(f'''
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


def load_house_prices():
  frame = pd.read_csv('data/houses/train.csv')
  y = frame['SalePrice'].values

  X = np.zeros((len(frame), 79))
  c = 0
  cols_to_skip = [
    'Id',
    'SalePrice'
  ]
  for name, contents in frame.iteritems():
    if name in cols_to_skip:
      continue

    if not np.issubdtype(contents.dtype, np.number):
      # xgboost needs a number encoding
      # some of these columns mix strings and numbers
      _, inverse = np.unique([str(v) for v in contents.values], return_inverse=True)
      X[:, c] = inverse
    else:
      X[:, c] = contents.values

    c += 1
  assert c == 79
  X[np.isnan(X)] = -1
  return X, y


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


M5_CACHE = 'data/m5/cache.npz'
def load_m5():
  if os.path.isfile(M5_CACHE):
    print('loading from cache')
    cached = np.load(M5_CACHE)
    return cached['X'], cached['y']

  with timed('loading data from csv...'):
    frame = pd.read_csv('data/m5/sales_train_validation.csv')

    # input format:
    #   one row per item, one column per day
    items = 30490
    days = 1912
    sales = np.zeros((items, days), dtype=int)
    for d in range(days):
      sales[:, d] = frame[f'd_{d+1}'].values

    y = sales[:, 365:].flatten()

    # start with a very simple timeseries prediction setup
    #
    # skip the first 365 days, then predict each day based on:
    #   the previous day
    #   the previous 7 days
    #   the previous 28 days
    #   the previous 365 days
    target_days = days - 365
    rows = items * target_days

    # ordinals for the categorical variables
    ordinals = {}
    for c, col in enumerate(['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']):
      _, inverse = np.unique(frame[col], return_inverse=True)
      ordinals[c] = inverse

    # this axes order is efficient for writing
    # approximate with uint16 for efficiency
    # scale floats before the uint16 truncation:
    #
    #   np.max(sales) == 763 so 80 is a safe scale for mean
    #   np.std is harder to predict, so use 10 to be more cautious
    feats = 19
    X = np.zeros((target_days, feats, items), dtype=np.uint16)
    sales_T = sales.T
    for t in range(target_days):
      if (t % 100) == 0:
        print(f'{t}/{target_days}')
      d = t + 365

      # 1 feature: previous day
      X[t, 0] = sales_T[d-1]

      # 5 features: previous 7 days
      X[t, 1] = sales_T[d-7]
      X[t, 2] = np.mean(sales_T[d-7:d], axis=0) * 80
      X[t, 3] = np.min(sales_T[d-7:d], axis=0)
      X[t, 4] = np.max(sales_T[d-7:d], axis=0)
      X[t, 5] = np.std(sales_T[d-7:d], axis=0) * 10

      # 4 features: previous 28 days
      # (min is almost always going to be 0)
      X[t, 6] = sales_T[d-28]
      X[t, 7] = np.mean(sales_T[d-28:d], axis=0) * 80
      X[t, 8] = np.max(sales_T[d-28:d], axis=0)
      X[t, 9] = np.std(sales_T[d-28:d], axis=0) * 10

      # 5 features: previous 365 days
      # (min is almost always going to be 0)
      X[t, 10] = sales_T[d-365]
      X[t, 11] = np.mean(sales_T[d-365:d], axis=0) * 80
      X[t, 12] = np.max(sales_T[d-365:d], axis=0)
      X[t, 13] = np.std(sales_T[d-365:d], axis=0) * 10

      # 5 features: ordinals for the categorical variables
      X[t, 14] = ordinals[0]
      X[t, 15] = ordinals[1]
      X[t, 16] = ordinals[2]
      X[t, 17] = ordinals[3]
      X[t, 18] = ordinals[4]

    # (target_days, feats, items) => (items, target_days, feats)
    X = np.moveaxis(X, -1, 0)

    # (items, target_days, feats) => (rows, feats)
    X = X.reshape((items * target_days, feats))

  with timed('saving...'):
    np.savez_compressed(M5_CACHE, X=X, y=y)
  return X, y

GRUPO_CACHE = 'data/grupo/cache.npz'
def load_grupo():
  if os.path.isfile(GRUPO_CACHE):
    print('loading from cache')
    cached = np.load(GRUPO_CACHE)
    return cached['X'], cached['y']

  frame = pd.read_csv('data/grupo/train.csv', usecols=[0, 1, 2, 3, 4, 5, 10], dtype=np.uint32)

  # weeks are 3-9:
  #   train: 6-8
  #   valid: 9
  #
  # features:
  #   over 5 categories (agency, canal, ruta, cliente, producto)
  #     over 5 stats (count, sum, mean, sum_sqs, var)
  #       over 3 periods (lag 1 week, lag 2 weeks, lag 3 weeks)
  #
  # for 5 * 5 * 3 = 75 features total
  # aggregating them efficiently is a little tricky
  feats = 75
  train_valid_count = np.count_nonzero(frame['Semana'].values >= 6)
  X = np.zeros((train_valid_count, feats), dtype=np.float32)

  categories = ['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID', 'Producto_ID']
  stat_names = ['counts', 'sums', 'sum_sqs', 'means', 'vars']
  lags = [1, 2, 3]

  for c, category in enumerate(categories):
    # dense-encode the column values as their indices into the unique array
    uniqs, encoded = np.unique(frame[category].values, return_inverse=True)

    with timed(f'Aggregating stats for {category}...'):
      # key is week + stat name
      # value is array of stats for each unique value
      stats = {}
      for w in [3, 4, 5, 6, 7, 8]:
        is_week = (frame['Semana'].values == w)

        values = encoded[is_week]
        targets = frame['Demanda_uni_equil'].values[is_week]

        counts = np.bincount(values, minlength=len(encoded))
        sums = np.bincount(values, weights=targets, minlength=len(encoded))
        sum_sqs = np.bincount(values, weights=(targets * targets), minlength=len(encoded))

        stats[f'{w}_counts'] = counts
        stats[f'{w}_sums'] = sums
        stats[f'{w}_sum_sqs'] = sum_sqs

        safe_counts = np.maximum(1, counts) # for division

        means = sums / safe_counts
        stats[f'{w}_means'] = means
        stats[f'{w}_vars']= (sum_sqs / safe_counts) - (means * means)

    with timed(f'Writing features for {category}...'):
      for w in [6, 7, 8, 9]:

        # frame is sorted by week,
        # so X is just frame shifted because we dropped weeks 3,4,5
        X_is_week = (frame['Semana'].values[-train_valid_count:] == w)
        X_codes = encoded[-train_valid_count:][X_is_week]

        per_cat_feats = len(lags) * len(stat_names)
        per_cat_feat = 0
        for lag in lags:
          for stat_name in stat_names:
            aggs = stats[f'{w-lag}_{stat_name}']
            f = c * per_cat_feats + per_cat_feat
            X[X_is_week, f] = aggs[X_codes]
            per_cat_feat += 1

        assert per_cat_feat == per_cat_feats

  y = frame['Demanda_uni_equil'].values[-train_valid_count:].astype(np.uint16)
  with timed('saving...'):
    np.savez_compressed(GRUPO_CACHE, X=X, y=y)
  return X, y


if __name__ == '__main__':
  np.random.seed(0)

  if 'TREE_COUNT' not in os.environ:
    raise ValueError('Expected env variable TREE_COUNT')
  tree_count = int(os.environ['TREE_COUNT'])
  print(f'\n\nTREE_COUNT={tree_count}')

  # name => function that loads data and returns (X, y)
  benchmarks = {
    # 'Agaricus':            load_agaricus,
    'House Prices':        load_house_prices,
    'Home Credit Default': load_credit,
    'Santander Value':     load_santander,
    'M5':                  load_m5,
    'Grupo':               load_grupo,
  }

  xgboost_args = {'n_estimators': tree_count, 'tree_method': 'hist'}
  lgb_args =  {'n_estimators': tree_count}

  for name, load_data_fn in benchmarks.items():
    print(f'\n\n{name}:\n')

    # split the time-series according to time
    X, y = load_data_fn()
    train_X, train_y, valid_X, valid_y = split(X, y)
    del X
    del y
    gc.collect()
    print(f'X.shape: train {train_X.shape}, valid {valid_X.shape}')

    # only handle regression or binary classification cases so far
    is_regression = (train_y.dtype != np.bool_)

    if is_regression:
      print(f'regression targets with min={np.min(train_y):.1f}, max={np.max(train_y):.1f}, mean={np.mean(train_y):.1f}')
    else:
      print(f'binary classification with {np.count_nonzero(train_y)} true and {np.count_nonzero(~train_y)} false')

    with timed('\ntrain DFS tree ...'):
      # with profiled():
      model: Any = None
      model, _ = fit(train_X, train_y, Params(use_bfs_tree=False, tree_count=tree_count))
    print(model.__str__(verbose=False))

    with timed(f'  predict DFS tree...'):
      # with profiled():
      train_preds = predict(model, train_X)
      valid_preds = predict(model, valid_X)
    print_stats(train_preds, train_y, valid_preds, valid_y, is_regression)
    del model
    del train_preds
    del valid_preds
    gc.collect()

    with timed(f'train xgboost with: {xgboost_args}...'):
      if is_regression:
        model = xgb.XGBRegressor(**xgboost_args)
      else:
        model = xgb.XGBClassifier(**xgboost_args)
      model.fit(train_X, train_y)

    with timed(f'predict xgboost...'):
      train_preds = model.predict(train_X)
      valid_preds = model.predict(valid_X)
    print_stats(train_preds, train_y, valid_preds, valid_y, is_regression)
    del model
    del train_preds
    del valid_preds
    gc.collect()

    with timed(f'train lightgbm with: {lgb_args}...'):
      if is_regression:
        model = lgb.LGBMRegressor(**lgb_args)
      else:
        model = lgb.LGBMClassifier(**lgb_args)
      model.fit(train_X, train_y)

    with timed(f'predict lightgbm...'):
      train_preds = model.predict(train_X)
      valid_preds = model.predict(valid_X)
    print_stats(train_preds, train_y, valid_preds, valid_y, is_regression)
    del model
    del train_preds
    del valid_preds
    gc.collect()

    # with timed('\ntrain BFS tree ...'):
    #   # with profiled():
    #   model, _ = fit(train_X, train_y, Params(use_bfs_tree=True, tree_count=tree_count))
    # print(model.__str__(verbose=False))

    # with timed(f'  predict BFS tree...'):
    #   # with profiled():
    #   train_preds = predict(model, train_X)
    #   valid_preds = predict(model, valid_X)
    # print_stats(train_preds, train_y, valid_preds, valid_y, is_regression)
    # del model
    # del train_preds
    # del valid_preds
    # gc.collect()


