import numpy as np

arr = np.array([np.nan, 0, 2, np.nan, 4, 6, 8, 10])
print(arr.shape)
arr[arr<5] = -1.0
print(arr)