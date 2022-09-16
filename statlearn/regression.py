import numpy as np

class Regressor:
    def __init__(self):
        self.train_X = None
        self.train_y = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the model to the training data X and training target values y.

        :param X: training data
        :param y: training target values
        """
        self.train_X = X
        self.train_y = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...
    
    def transform(self) -> np.ndarray:
        """
        Returns the fitted labels on the training data.
        """
        return self.predict(self.train_X)
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fits the model to the training data X, and trainig target values y,
        and returns the fitted values.
    
        :param X: training data
        :param y: training target values
        """
        self.fit(X, y)
        return self.transform()


class LinearResgression(Regressor):

    def __init__(self, penalty: str=None, C: float=None):
        """
        Initializes the the LinearRegression model.

        :param penalty: None=OLS model, 'L2'=Ridge Regression.
        :param C: the value of the penalty parameter.
        """
        self.train_X = None
        self.train_y = None
        self.beta = None
        self.penalty = penalty
        self.C = C
    
    def fit(self, X: np.ndarray, y: np.ndarray, add_intercept:bool=False):
        """
        Fits the LinearRegression model to the training data X
        and target values y.

        :param X: the training data
        :param y: the training target values
        :param add_intercept: if True, adds a column of 1s, default=Fale.
        """
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
        """
        Returns the fitted labels on the training data.
        """ 
        if self.penalty is None:
            y_hat = self.Q @ self.Q.T @ self.train_y
        else:
            y_hat = self.X @ self.beta
        return y_hat
    
    def predict(self, X, add_intercept=False):
        """
        Returns the vector of predicted value for data X.

        :param X: values for prediction
        :param add_intercept: if True, adds a column of 1s, default=Fale.
        """
        if add_intercept:
            X = self._add_intercept(X)
        return X @ self.beta
    
    def fit_trasform(self, X, y, **kwargs):
        """
        Fits the model to the training data X, and trainig target values y,
        and returns the fitted values.
    
        :param X: training data
        :param y: training target values
        """
        self.fit(X, y, **kwargs)
        return self.transform()
    
    def _add_intercept(self, X):
        """
        Adds a column of 1s to the design matrix X.
        """
        return  np.column_stack([
                np.repeat(1, len(X)),
                X
        ])
    