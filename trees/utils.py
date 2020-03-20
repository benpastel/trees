from time import time
from contextlib import contextmanager

import numpy as np


@contextmanager
def timed(msg: str):
  print(msg)
  start = time()
  yield
  stop = time()
  print(f'({stop - start:.1f}s)')


def percent(num: int, denom: int) -> str:
  denom = max(1, denom)
  return f'{100.0 * num / denom:.2f}%'


def binary_stats(preds: np.ndarray, trues: np.ndarray) -> str:
  assert preds.shape == trues.shape
  assert trues.ndim == 1
  accuracy = percent(np.count_nonzero(preds == trues), len(trues))
  hits = np.count_nonzero((preds == trues) & (trues != 0))
  precision = percent(hits, np.count_nonzero(preds))
  recall = percent(hits, np.count_nonzero(trues))
  return f'accuracy = {accuracy}, precision = {precision}, recall = {recall}'


def regression_stats(preds: np.ndarray, trues: np.ndarray) -> str:
  assert preds.shape == trues.shape
  assert trues.ndim == 1
  mse = np.mean((preds - trues)  * (preds - trues))
  mae = np.mean(np.abs(preds - trues))
  if mse > 1000000:
    return f'log(MSE): {np.log(mse):.1f}, MAE: {mae:.1f}'
  else:
    return f'MSE: {mse:.1f}, MAE: {mae:.1f}'
