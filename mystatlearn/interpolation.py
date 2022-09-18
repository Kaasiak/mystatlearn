import numpy as np

class BSpline():
    """
    Generation of B-splines basis functions.

    References
    -----
    Implementation based on:
    [1] -  https://en.wikipedia.org/w/index.php?title=De_Boor%27s_algorithm&oldid=1073433593
    """
    def __init__(
            self, p: int, 
            knots: np.ndarray=None, 
            controls: np.ndarray=None, 
            type: str=None, 
            n: int=None
        ):
        """
        Initialises the B-spline object.

        :param p: the degree of the B-spline curves
        :param knots: array of knots
        :param controls: array of control points
        :param type: if knots is None, one of: 'periodic', 'uniform'
        :param n: if knots is None, the number of knots
        """
        self.p = p
        self.controls = controls
        if  (type is not None) and (n is not None):
            self.knots = self.get_knots(n, p, type)
        elif knots is not None:
            self.knots = knots
        else:
            raise(TypeError(
                "Either the knot values or the type and number of control",
                 + "points must be provided."
            ))
    
    def interpolate(
            self, 
            X: np.ndarray, 
        ) -> np.ndarray:
        """
        Returns the values of the B-spline curve at X.

        :param X: the array of x values
        :param knots: the array of knots
        :params p: the degree of the B-spline curve
        """
        n = len(self.controls)
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
        y = np.empty(len(X))
        for i in range(0, len(knots) - 1):
            index = (X >= knots[i]) & (X < knots[i + 1])
            xi = X[index]
            if xi.size > 0:
                y[index] = self._deBoor(xi, i)
        return y
                
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
    
    def bspline_basis(self, x: float, k: int, p: int) -> float:
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
                a * self.bspline_basis(x, k, p - 1)
                + b * self.bspline_basis(x, k + 1, p - 1)
            )

    
class CubicSpline():

    def __init__(self):
        self.X = None
        self.y = None
        self.coef = None

    def fit (self, X, y):
        self.X = X
        self.y = y
        n = len(X) - 1
        a = y.copy()
        b = np.zeros(n)
        d = np.zeros(n)
        h = np.diff(X)
        alpha = np.zeros(n)
        alpha[1:] = (
            (3 / h[1:])  * (a[2:] - a[1:-1]) 
            - (3 / h[:-1])  * (a[1:-1] - a[:-2])
        )
        c = np.zeros(n + 1)
        l = np.zeros(n + 1)
        mu = np.zeros(n + 1)
        z = np.zeros(n + 1)
        l[0] =  1
        for i in range(1, n):
            l[i] = 2 * (X[i + 1] - X[i - 1]) - h[i - 1] * mu[i - 1]
            mu[i] = h[i] / l[i]
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]
        l[n] = 1
        c[n] = z[n] = 0
        for j in range(n - 1, -1, -1):
            c[j] = z[j] - mu[j] * c[j + 1]
            b[j] = (
                (a[j + 1] - a[j]) / h[j] 
                - (h[j] * (c[j + 1] + 2 * c[j])) / 3
            )
            d[j] = (c[j+ 1] - c[j]) / (3 * h[j])
        self.coef = (a, b, c, d)

    def predict(self, x):
        y_pred = np.zeros(len(x))
        for j in range(0, len(self.X) - 1):
            index = (x >= self.X[j]) & (x <= self.X[j + 1])
            if x[index].size > 0 :
                y_pred[index] = self._get_j_spline(x[index], j)
        return y_pred

    def _get_j_spline(self, x, j):
        (a, b, c, d) = self.coef
        S = (
            a[j] + b[j] * (x - self.X[j]) + c[j] * (x - self.X[j]) ** 2 
            + d[j] * (x - self.X[j]) ** 3
        )
        return S


    