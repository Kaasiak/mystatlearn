import numpy as np
from mystatlearn.utils.kernels import gaussian, epanechnikov, uniform, triangular

class KDE():
    def __init__(self, kernel='gaussian', bandwidth=1):
        self.x = None
        self.h = bandwidth
        if kernel == 'gaussian':
            self.kernel = gaussian
        elif kernel == 'epanechnikov':
            self.kernel = epanechnikov
        elif kernel == 'uniform':
            self.kernel = uniform
        elif kernel == 'triangular':
            self.kernel = triangular
        
    def fit(self, x):
        self.x = x

    def density(self, x):
        return np.vectorize(self._density)(x)

    def _density(self, x):
        return np.sum(self.kernel((x - self.x) / self.h)) / (len(self.x) * self.h)