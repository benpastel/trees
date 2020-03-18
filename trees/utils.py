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


def stats(preds: np.ndarray, trues: np.ndarray) -> str:
  assert preds.shape == trues.shape
  assert trues.ndim == 1
  acc = percent(np.count_nonzero(preds == trues), len(trues))
  hits = np.count_nonzero((preds == trues) & trues)
  precision = percent(hits, np.count_nonzero(preds))
  recall = percent(hits, np.count_nonzero(trues))
  return f'accuracy = {acc}, precision = {precision}, recall = {recall}'