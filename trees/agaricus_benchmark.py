import xgboost as xgb
import numpy as np

from sklearn.datasets import load_svmlight_file

from trees.tree import fit, predict


if __name__ == '__main__':
  train_X, train_y = load_svmlight_file('data/agaricus.txt.train')
  test_X, test_y = load_svmlight_file('data/agaricus.txt.test')

  train_X = train_X.toarray()
  test_X = test_X.toarray()

  model = xgb.XGBClassifier(n_estimators = 1)
  model.fit(train_X, train_y)

  train_preds = model.predict(train_X)
  test_preds = model.predict(test_X)

  print(f'''XGBoost on agaricus, single tree:
    train accuracy: {100.0 * np.mean(train_preds == train_y):.2f}%
    test accuracy: {100.0 * np.mean(test_preds == test_y):.2f}%
  ''')

  for min_leaf_size in [1, 10, 100, 1000]: 
    model = fit(train_X, train_y, min_leaf_size=min_leaf_size)
    train_preds = predict(model, train_X)
    test_preds = predict(model, test_X)
    print(f'''Tree on agaricus, min_leaf_size = {min_leaf_size}:
      train accuracy: {100.0 * np.mean(train_preds == train_y):.2f}%
      test accuracy: {100.0 * np.mean(test_preds == test_y):.2f}%
      model: {model}
    ''')
