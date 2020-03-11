import numpy as np

from sklearn.datasets import load_svmlight_file

from tree import fit

if __name__ == '__main__':
  train_X, train_y = load_svmlight_file('data/agaricus.txt.train')
  test_X, test_y = load_svmlight_file('data/agaricus.txt.test')
  train_X = train_X.toarray()
  test_X = test_X.toarray()

  model = fit(train_X, train_y)
  print(str(model))