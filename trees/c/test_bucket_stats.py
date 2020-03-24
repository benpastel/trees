import numpy as np

from trees.c.bucket_stats import bucket_stats


X = np.arange(10)
y = np.arange(10)
counts = np.zeros(10, dtype=np.uint32)

bucket_stats(X, y, counts)

print('ok dokie')
print(counts)
