import numpy as np
from cvxopt import matrix, solvers
from mystatlearn.utils import kernels

class Classifier:
    def __init__(self):
        self.train_X = None
        self.train_y = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the model to the training data X and training labels y.

        :param X: training data
        :param y: training labels
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
        Fits the model to the training data X, and trainig labels y,
        and returns the fitted values.
    
        :param X: training data
        :param y: training labels
        """
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


class SVM(Classifier):

    def __init__(self, margin:str ='soft', C=100):
        super().__init__()
        self.C = C
        self.margin = margin
    
    def fit(self, X, y):
        self.train_y = y
        self.train_X = X
        n = len(X)
        # setup th QP
        P = matrix((X * y) @ (X * y).T, tc='d')
        q = matrix(-np.ones(n), tc='d')
        A = matrix(y.T, tc='d')
        b = matrix(0, tc='d')
        if self.margin == 'soft':
            G = matrix(np.concatenate([
                np.eye(n), -np.eye(n)]), tc='d')
            h = matrix(np.concatenate([
                np.repeat(self.C / (2 * n), n), np.zeros(n)]), tc='d')
        else:
            G = matrix(-np.eye(n), tc='d')
            h = matrix(np.zeros(n), tc='d')
        # solve for Lagrange multipliers
        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, A, b)
        self.lambdas = np.array(sol['x'])
        # find the support vecotrs
        if self.margin == 'hard':
            support = (self.lambdas > 1e-5)
        else:
            support = (self.lambdas > 1e-5) & (self.lambdas < self.C / (2 * n))
        support = support.flatten()
        # get the hyperplane and the bias
        self.w = (self.train_X * y).T @ self.lambdas
        self.bias = np.mean(
            y[support] - X[support] @ self.w.reshape(-1,1)
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
            return np.sign(X @ self.w + self.bias)


class KernelSVM(Classifier):
    def __init__(self, margin:str ='soft', C=100, kernel='linear'):
        super().__init__()
        if kernel == 'linear':
            self.kernel = kernels.linear
        elif kernel == 'rbf':
            self.kernel = kernels.rbf
        self.C = C
        self.margin = margin
    
    def fit(self, X, y):
        self.train_y = y
        self.train_X = X
        n = len(X)
        # setup the QP
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self.kernel(X[i, :], X[j, :])
        P = matrix(np.outer(y, y) * K, tc='d')
        q = matrix(-np.ones(n), tc='d')
        A = matrix(y.T, tc='d')
        b = matrix(0, tc='d')
        if self.margin == 'soft':
            G = matrix(np.vstack([
                np.eye(n), -np.eye(n)]), tc='d')
            h = matrix(np.concatenate([
                np.repeat(self.C / (2 * n), n), np.zeros(n)]), tc='d')
        else:
            G = matrix(-np.eye(n), tc='d')
            h = matrix(np.zeros(n), tc='d')
        # solve for lagrange multipliers
        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, A, b)
        self.lambdas = np.array(sol['x'])

        # get the vectors inside the margin
        self.margin_vectors_idx = (self.lambdas > 1e-5).flatten()
        self.margin_lambdas = self.lambdas[self.margin_vectors_idx]
        self.margin_vectors = self.train_X[self.margin_vectors_idx]
        self.margin_labels = self.train_y[self.margin_vectors_idx]

        # get one support vector
        self.support_idx = np.argwhere(
            self.margin_lambdas.flatten() == self.margin_lambdas.min())[0][0]

        # Find the bias using the first support vector
        K0 = self.kernel(self.margin_vectors[self.support_idx], self.margin_vectors, axis=1)
        self.bias = (
            self.margin_labels[self.support_idx] 
            -  (self.margin_lambdas * self.margin_labels).T @ K0
        )

    def predict(self, X):
        m = len(X)
        n = len(self.margin_lambdas)
        K = np.zeros((n, m))
        for j in range(m):
            K[:, j] = self.kernel(
                self.margin_vectors, X[j, ], axis=1
            )
        return np.sign(
            K.T @ (self.margin_lambdas * self.margin_labels) 
            + self.bias
        )