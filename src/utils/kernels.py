import numpy as np

def linear(x, y, param, axis=None):
    if axis:
        return x @ y.T
    else:
        return x @ y

def rbf(x, y, sigma=1, axis=None):
    if axis:
        return np.exp(- 0.5 * (np.linalg.norm(x - y, axis=axis) / sigma)**2)
    else:
        return np.exp(-0.5 * (np.linalg.norm(x - y) / sigma)**2)

def poly(x, y, d=3, axis=None):
    if axis:
        return (x @ y.T)**d
    else:
        return (x @ y)**d

def sigmoid(x, y, gamma, axis=None):
    if axis:
        return np.tanh(x @ y.T * gamma)
    else:
        return np.tanh(x @ y * gamma)

def gaussian(x):
    return 1 / (np.sqrt(2 * np.pi)) * np.exp(- x**2 / 2)

def epanechnikov(x):
    k = 3 / (4 * np.sqrt(5)) * (1 - x**2 / 5)
    k[np.abs(x) > np.sqrt(5)] = 0
    return k

def triangular(x):
    k = 1 / np.sqrt(6) * (1 - np.abs(x) / np.sqrt(6))
    k[np.abs(x) > np.sqrt(6)] = 0
    return k

def uniform(x):
    k = np.zeros(len(x))
    k[np.abs(x) <= np.sqrt(3)] = 1 / (2 * np.sqrt(3))
    return k