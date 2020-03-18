import xgboost as xgb
import numpy as np

from sklearn.datasets import load_svmlight_file

from trees.tree import fit, predict
from trees.utils import stats


if __name__ == '__main__':
  train_X, train_y = load_svmlight_file('data/agaricus.txt.train')
  valid_X, valid_y = load_svmlight_file('data/agaricus.txt.test')

  train_X = train_X.toarray()
  valid_X = valid_X.toarray()

  model = xgb.XGBClassifier(n_estimators = 1)
  model.fit(train_X, train_y)

  train_preds = model.predict(train_X)
  valid_preds = model.predict(valid_X)

  print(f'''XGBoost on agaricus, single tree:
    train accuracy: {100.0 * np.mean(train_preds == train_y):.2f}%
    valid accuracy: {100.0 * np.mean(valid_preds == valid_y):.2f}%
  ''')

  min_leaf_size = 10
  print(f'Classification tree with {min_leaf_size=}')
  model = fit(train_X, train_y.astype(bool), min_leaf_size=min_leaf_size)
  print(model)
  train_preds = predict(model, train_X)
  valid_preds = predict(model, valid_X)

  print(f'''  on agaricus:
    train_preds bincount: {np.bincount(train_preds)}
    train: {stats(train_preds, train_y)}
    valid: {stats(valid_preds, valid_y)}
  ''')

  print(f'Classification tree with {min_leaf_size=}')
  model = fit(train_X, train_y.astype(float), min_leaf_size=min_leaf_size)
  print(model)
  train_preds = predict(model, train_X)
  valid_preds = predict(model, valid_X)

  print(f'''  on agaricus:
    train_preds bincount: {np.bincount(train_preds)}
    train: {stats(train_preds, train_y)}
    valid: {stats(valid_preds, valid_y)}
  ''')
