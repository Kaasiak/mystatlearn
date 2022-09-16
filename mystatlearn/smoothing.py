import numpy as np
from mystatlearn.regression import LinearResgression

class BSpline():

    def __init__(self, p, knots=None, type=None, n=None):
        self.p = p
        if  (type is not None) and (n is not None):
            self.knots = self.get_knots(n, p, type)
        elif knots is not None:
            self.knots = knots
        else:
            raise(TypeError(
                "Either the knot values or the type and number of control",
                 + "points must be provided."
            ))
    
    def get_spline(
            self, 
            X: np.ndarray, 
            controls: np.ndarray, 
        ) -> np.ndarray:
        """
        Returns the values of the B-spline curve at X.

        :param X: the array of x values
        :param knots: the array of knots
        :param controls: the array of contol points
        :params p: the degree of the B-spline curve
        """
        n = len(controls)
        p = self.p
        knots = self.knots
        if n != len(knots) - (p + 1):
            raise(ValueError(
                "Number of control points doesn't match the nunmber of knots."
                + f" For {len(knots)} knots, expected" 
                + f" {len(knots) - (p + 1)} controls."
            ))
        if (np.max(X) >= knots[n]) or (np.max(X) < knots[p]):
            raise(ValueError(
                f"Values of X are outside of the support " 
                + f"[{knots[p]}, {knots[n]})"
            ))
        self.controls = controls
        y = np.empty(len(X))
        for i in range(0, len(knots) - 1):
            index = (X >= knots[i]) & (X < knots[i + 1])
            xi = X[index]
            if xi.size > 0:
                y[index] = self._deBoor(xi, i)
        return y
    
    def fit(
            self, 
            X: np.ndarray, 
            y: np.ndarray, 
        ) -> None:
        """
        Fits and B-spline curve to data (X, y) with given knots and degree.

        :param X: the array of x values
        :param y: the array of y values
        :param knots: the array of knots
        :param p: the degree of the B-spline curve
        """
        p = self.p
        knots = self.knots
        n = len(knots) - (p + 1)
        m = len(X)
        if (np.max(X) >= knots[n]) or (np.max(X) < knots[p]):
            raise(ValueError(
                f"Values of X are outside of the support " 
                + f"[{knots[p]}, {knots[n]})"
            ))
        A = np.zeros((m, n))
        for i in range(0, m):
            for j in range(0, n):
                    A[i][j] = self._bspline_basis(X[i], j, p)
        self.A = A      
        self.cf = LinearResgression()
        self.cf.fit(A, y)
        self.controls = self.cf.beta
    
    def transform(self):
        y_pred = self.cf.transform()
        return y_pred
    
    def predict(self, X):
        return self.get_spline(X, self.controls)
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform()
            
    def _deBoor(
            self, 
            x: float, 
            k: int, 
        ) -> float:
        """
        Calculated the value of the B-spline curve at x 
        for which x belongs to the knot interval [k, k + 1].

        : param x: The x value to be evaluated
        : param k: the index of the knot interval
        """
        t = self.knots
        p = self.p
        d = [self.controls[j + k - p] for j in range(0, p + 1)]
        d = np.repeat(
            [self.controls[j + k - p] for j in range(0, p + 1)], 
            len(x)
        ).reshape(p + 1, -1).astype('float32')
        for r in range(1, p + 1):
            for j in range(p, r - 1, -1):
                alpha = (
                    (x - t[j + k - p]) 
                    / (t[j + 1 + k - r] - t[j + k - p])
                )
                d[j, :] = (1.0 - alpha) * d[j - 1, :] + alpha * d[j, :]
        return d[p]
    
    def get_knots(self, n: int, p: int, type: str) -> np.ndarray:
        """
        Returns an array of knots with for n control points for the
        uniform and period knot distribution.

        :param n: the number of control points
        :param p: the degree of the B-spline curve
        :param type: one of 'uniform', 'periodic'
        """
        knots = np.zeros(n + p + 1)
        if type == 'uniform':
            for i in range(n + p  + 1):
                if i <= p:
                    knots[i] = 0
                elif i < n:
                    knots[i] = (i - p) / (n - p)
                else:
                    knots[i] = 1
        elif type == 'periodic':
            for i in range(n + p + 1):
                knots[i] = (i - p) / (n - p)
        else:
            raise(ValueError(
                "Supported types are: 'uniform', 'periodic'"
            ))
        return knots
    
    def _bspline_basis(self, x: float, k: int, p: int) -> float:
        """
        Evaluated the value of the B-spline basis function
        at the i-th knot interval and p-th degree.

        :param x: the value x to be evaluated
        :param k: the index of the interval
        :param p: the degree of the curve
        """
        t = self.knots
        if p == 0:
            if (t[k] <= x) and (x < t[k + 1]):
                return 1
            else: 
                return 0
        else:
            if t[k + p] == t[k]:
                a = 0
            else:
                a = (x - t[k]) / (t[k + p] - t[k])
            if t[k + p + 1] == t[k + 1]:
                b = 0
            else:
                b = (t[k + p + 1] - x) / (t[k + p + 1] - t[k + 1])
            return (
                a * self._bspline_basis(x, k, p - 1)
                + b * self._bspline_basis(x, k + 1, p - 1)
            )

    
