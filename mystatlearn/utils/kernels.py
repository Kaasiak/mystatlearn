import numpy as np

def linear(x, y, axis=None):
    if axis:
        return x @ y.T
    else:
        return x @ y

def rbf(x, y, axis=None):
    if axis:
        return np.exp(-0.5 * np.linalg.norm(x - y, axis=axis)**2)
    else:
        return np.exp(-0.5 * np.linalg.norm(x - y)**2)