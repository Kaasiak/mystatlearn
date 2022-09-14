import numpy as np

class LinearResgression():

    def __init__(self, penalty=None, C=None):
        self.train_X = None
        self.train_y = None
        self.beta = None
        self.penalty = penalty
        self.C = C
    
    def fit(self, X, y, add_intercept=False):
        if add_intercept:
            X = self._add_intercept(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)    
        self.train_X = X
        self.train_y = y
        if self.penalty is None:
            self.Q, self.R = np.linalg.qr(X)
            self.beta = np.linalg.inv(self.R) @ self.Q.T @ y
        if self.penalty == 'L2':
            self.beta = np.linalg.inv(
                X.T @ X + self.C * np.eye(X.shape[1])) @ X.T @ y

    def transform(self):
        if self.penalty is None:
            y_hat = self.Q @ self.Q.T @ self.train_y
        else:
            y_hat = self.X @ self.beta
        return y_hat
    
    def predict(self, X, add_intercept=False):
        if add_intercept:
            X = self._add_intercept(X)
        return X @ self.beta
    
    def fit_trasform(self, X, y, **kwargs):
        self.fit(X, y, **kwargs)
        return self.transform()
    
    def _add_intercept(self, X):
        return  np.column_stack([
                np.repeat(1, len(X)),
                X
        ])
    