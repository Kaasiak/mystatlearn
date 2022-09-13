import numpy as np

class LDA:
    def __init__(self):
        self.n_classes = None
        self.n_dim = None
        self.n_obs = None
        self.params = dict()
        self.deltas = None
        self.train_X = None
        self.train_y = None

    def fit(self, X, y, probas=None):
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
            Sigma += (x.transpose()).dot(x)
            if probas is None:
                self.probas[i] = len(x) / self.n_obs
        self.params['Sigma'] = Sigma / (self.n_obs - self.n_classes)
        self.params['U'], self.params['D'], _ = np.linalg.svd(
            self.params['Sigma'])
    
    def transform(self, X):
        self.deltas = np.zeros((X.shape[0], self.n_classes))
        M = np.row_stack(list(self.params['mu'].values()))
        # orthogonalise
        sqrt_inv_D = np.diag(np.sqrt(1 / self.params['D']))
        X_star = X.dot(self.params['U']).dot(sqrt_inv_D)
        M_star = M.dot(self.params['U']).dot(sqrt_inv_D)
        for i in range(self.n_classes):
            self.deltas[:, i] = self._get_discriminant(
                X_star, 
                M_star[i, :], 
                self.probas[i]
            )
        return np.argmax(self.deltas, axis=1)
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X):
        return self.transform(X)

    def _get_discriminant(self, X_star, mu_star, pi):
        delta =  (
            np.dot(X_star, mu_star) - 0.5 * np.dot(mu_star.T, mu_star)
            + np.log(pi)
        )
        return delta

    def _find_line_params(self, mu1, mu2, Sigma, pi1, pi2):
        prec_mat = np.linalg.inv(Sigma)
        m = (mu1 + mu2) / 2
        w = prec_mat.dot(mu2 - mu1)
        k = np.log(pi2 / pi1)
        beta =  - w[0] / w[1]
        intercept =  -k + m[1] - m[0] * beta
        return beta, intercept

    def get_boundary(self, type, class1=None, class2=None, equal_prob=False):
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
                (x1 - mu1).transpose().dot(x1 - mu1)
                + (x2 - mu2).transpose().dot(x2 - mu2)
            ) / (self.n_obs - 2)
            if equal_prob:
                pi1 = pi2 = 0.5
            else:
                pi1 = len(x1) / self.n_obs
                pi2 = len(x2) / self.n_obs

        return self._find_line_params(mu1, mu2, Sigma, pi1, pi2)


class QDA:
    def __init__(self):
        self.n_classes = None
        self.n_dim = None
        self.n_obs = None
        self.params = dict()
        self.deltas = None
        self.train_X = None
        self.train_y = None

    def fit(self, X, y, probas=None):
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

        Sigma = 0
        for i in range(self.n_classes):
            x = X[y == i, :]
            self.params['mu'][i] = np.mean(x, axis=0)
            self.params['Sigma'][i] = np.cov(x.T)
            self.params['U'][i], self.params['D'][i], _ = np.linalg.svd(
                self.params['Sigma'][i]
            )
            if probas is None:
                self.probas[i] = len(x) / self.n_obs
    
    def transform(self, X):
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
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X):
        return self.transform(X)

    def _get_discriminant(self, X, mu, U, D, pi):
        inv_D = np.diag(1 / D)
        def _compute_delta(x):
            return (
                -0.5 * np.log(D).sum()
                -0.5 * (np.dot(U, (x - mu))).T.dot(inv_D).dot(
                    np.dot(U, (x - mu))) + np.log(pi)
            )
        delta = np.apply_along_axis(_compute_delta, 1, X)
        return delta
