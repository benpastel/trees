import xgboost as xgb
import numpy as np

from sklearn.datasets import load_svmlight_file



if __name__ == '__main__':
  train_X, train_y = load_svmlight_file('../data/agaricus.txt.train')
  test_X, test_y = load_svmlight_file('../data/agaricus.txt.test')

  train_X = train_X.toarray()
  test_X = test_X.toarray()

  model = xgb.XGBClassifier()
  model.fit(train_X, train_y)

  train_preds = model.predict(train_X)
  test_preds = model.predict(test_X)

  print(f'''XGBoost on agaricus, default settings:
    train accuracy: {100.0 * np.mean(train_preds == train_y):.2f}%
    test accuracy: {100.0 * np.mean(test_preds == test_y):.2f}%
  ''')