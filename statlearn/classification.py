from multiprocessing.sharedctypes import Value
import numpy as np

class Classifier:
    def __init__(self):
        self.train_X = None
        self.train_y = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.train_X = X
        self.train_y = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...
    
    def transform(self) -> np.ndarray:
        return self.predict(self.train_X)
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform()
        

class LDA(Classifier):
    def __init__(self):
        self.n_classes = None
        self.n_dim = None
        self.n_obs = None
        self.params = dict()
        self.deltas = None
        self.train_X = None
        self.train_y = None

    def fit(
            self, 
            X: np.ndarray, 
            y: np.ndarray, 
            probas: np.ndarray=None
        ) -> None:
        """
        Fits the LDA classifier.
        
        :param X: training data
        :param y: training target labels
        :param probas: class probabilities. If None, 
            class probabilities estimated from the data, deafult probas=None.
        """
        self.n_classes = len(np.unique(y))
        self.n_dim = X.shape[1]
        self.n_obs = X.shape[0]
        self.params['mu'] = dict()
        self.probas = np.zeros(self.n_classes)
        self.train_X = X
        self.train_y = y
        Sigma = 0
        for i in range(self.n_classes):
            x = X[y == i, :]
            self.params['mu'][i] = np.mean(x, axis=0)
            x = x - self.params['mu'][i]
            Sigma += x.T @ x
            if probas is None:
                self.probas[i] = len(x) / self.n_obs
        self.params['Sigma'] = Sigma / (self.n_obs - self.n_classes)
        self.params['U'], self.params['D'], _ = np.linalg.svd(
            self.params['Sigma'])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the labels for data X.

        :param X: the design matrix.
        """
        self.deltas = np.zeros((X.shape[0], self.n_classes))
        M = np.row_stack(list(self.params['mu'].values()))
        # orthogonalise
        sqrt_inv_D = np.diag(np.sqrt(1 / self.params['D']))
        X_star = X @ self.params['U'] @ sqrt_inv_D
        M_star = M @ self.params['U'] @ sqrt_inv_D
        for i in range(self.n_classes):
            self.deltas[:, i] = self._get_discriminant(
                X_star, 
                M_star[i, :], 
                self.probas[i]
            )
        return np.argmax(self.deltas, axis=1)
    
    def _get_discriminant(
            self, 
            X_star: np.ndarray, 
            mu_star: np.ndarray, 
            pi: float
        ) -> float:
        """
        Returns the value of the descriminant function for a class with mean mu.

        :param X_star: training data in the orthogonal coordinates.
        :param mu_star: class mean vector in the orthogonal space
        :pi: class probability
        """
        delta =  (
            X_star @ mu_star - 0.5 * mu_star.T @ mu_star
            + np.log(pi)
        )
        return delta

    def _find_line_params(
            self, 
            mu1: np.ndarray, 
            mu2:np.ndarray, 
            Sigma: np.ndarray, 
            pi1: float, 
            pi2: float
        ) -> tuple:
        """
        Returns the slope and intercept coefficients for the linear boundary
        separating two classes.

        :param mu1: mean vector of class 1
        :param mu2: mean vector of class 2
        :param Sigma: pooled covariance matrix
        :param pi1: probability of class 1
        :param pi2: probability of class 2
        """
        prec_mat = np.linalg.inv(Sigma)
        m = (mu1 + mu2) / 2
        w = prec_mat @ (mu2 - mu1)
        k = np.log(pi2 / pi1)
        beta =  - w[0] / w[1]
        intercept =  -k + m[1] - m[0] * beta
        return beta, intercept

    def get_boundary(
            self, 
            type: str, 
            class1: int=None, 
            class2: int=None, 
            equal_prob: bool=False
        ) -> tuple:
        """
        Returns the slope and intercept coefficients for the linear boundary
        separating the classes.

        :param type: If 'one-to-one', returns the boundary between 
            class1 and class12. If 'one-vs-all', returns the boundary between
            class1 and all remaining classes.
        :param class1: the label of class1
        :param class2: the label of class2 (only for type='one-to-one')
        :param equal_prob: If True, class probas for type='one-vs-all' are equal.
            default=False.
        """
        if type == 'one-to-one':
            mu1 = self.params['mu'][class1]
            mu2 = self.params['mu'][class2]
            Sigma = self.params['Sigma']
            pi2 = self.probas[class1]
            pi1 = self.probas[class2]
        elif type == 'one-vs-all':
            mu1 = self.params['mu'][class1]
            x1 = self.train_X[self.train_y == class1, :]
            x2 = self.train_X[self.train_y != class1, :]
            mu2 = np.mean(x2, axis=0)
            Sigma = (
                (x1 - mu1).T @ (x1 - mu1)
                + (x2 - mu2).T @ (x2 - mu2)
            ) / (self.n_obs - 2)
            if equal_prob:
                pi1 = pi2 = 0.5
            else:
                pi1 = len(x1) / self.n_obs
                pi2 = len(x2) / self.n_obs
        else:
            raise(ValueError(
                "Boundary type must be one of: 'one-to-one', 'one-vs-all'."
            ))
        return self._find_line_params(mu1, mu2, Sigma, pi1, pi2)


class QDA(Classifier):
    def __init__(self):
        self.n_classes = None
        self.n_dim = None
        self.n_obs = None
        self.params = dict()
        self.deltas = None
        self.train_X = None
        self.train_y = None

    def fit(self, X, y, probas=None):
        """
        Fits the QDAclassifier.
        
        :param X: training data
        :param y: training target labels
        :param probas: class probabilities. If None, 
            class probabilities estimated from the data, deafult probas=None.
        """
        self.n_classes = len(np.unique(y))
        self.n_dim = X.shape[1]
        self.n_obs = X.shape[0]
        self.params['Sigma'] = dict()
        self.params['mu'] = dict()
        self.params['U'] = dict()
        self.params['D'] = dict()
        self.probas = np.zeros(self.n_classes)
        self.train_X = X
        self.train_y = y
        for i in range(self.n_classes):
            x = X[y == i, :]
            self.params['mu'][i] = np.mean(x, axis=0)
            self.params['Sigma'][i] = np.cov(x.T)
            # Diagonalise the covariance matrix
            self.params['U'][i], self.params['D'][i], _ = np.linalg.svd(
                self.params['Sigma'][i]
            )
            if probas is None:
                self.probas[i] = len(x) / self.n_obs
    
    def predict(self, X: np.ndarray):
        """
        Predicts the labels for data X.

        :param X: the design matrix.
        """
        self.deltas = np.zeros((X.shape[0], self.n_classes))
       
        for i in range(self.n_classes):
            self.deltas[:, i] = self._get_discriminant(
                X, 
                self.params['mu'][i], 
                self.params['U'][i], 
                self.params['D'][i], 
                self.probas[i]
            )
        return np.argmax(self.deltas, axis=1)

    def _get_discriminant(
            self, 
            X: np.ndarray, 
            mu: np.ndarray, 
            U: np.ndarray, 
            D: np.ndarray, 
            pi: float
        ) -> float:
        """
        Returns the value of the descriminant function for a class with mean mu.

        :param X: training data
        :param mu: class mean vector
        :param U: corrdinate change matrix
        :param D: diagonalised matrix X
        :pi: class probability
        """
        inv_D = np.diag(1 / D)
        def _compute_delta(x):
            return (
                -0.5 * np.log(D).sum()
                -0.5 * (U @ (x - mu)).T  @ inv_D @ U @ (x - mu) 
                + np.log(pi)
            )
        delta = np.apply_along_axis(_compute_delta, 1, X)
        return delta
