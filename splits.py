import numpy as np

def gini_purity(A: np.ndarray) -> float:
  ''' 
  gini purity in the boolean case:
    1 - p_no * (1 - p_no) + p_yes * (1 - p_yes)
    1 - 2 * p_yes * p_no
    1 - 2 * (trues / total) * (falses / total)
    1 - 2 * trues * falses / total^2
  '''
  assert A.ndim == 1
  assert A.dtype == np.bool
  assert len(A) > 0

  trues = np.count_nonzero(A)
  falses = len(A) - trues

  return 1.0 - 2.0 * trues * falses / (len(A) * len(A))
