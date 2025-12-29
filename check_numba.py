import numba
print(f"Numba version: {numba.__version__}")
from numba import njit, prange
import numpy as np

@njit(parallel=True)
def test_numba(x):
    y = np.zeros_like(x)
    for i in prange(x.shape[0]):
        y[i] = x[i] * 2
    return y

x = np.ones(10)
y = test_numba(x)
print("Numba test passed")
