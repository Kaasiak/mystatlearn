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