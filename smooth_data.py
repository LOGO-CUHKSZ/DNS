import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin

def smooth_sample(sample_t: np.ndarray, sample_x: np.ndarray, min_t: float, max_t: float, n_knots: int) -> np.ndarray:
    sample = [
        get_natural_cubic_spline_model(sample_t, sample_x[:, i], min_t, max_t, n_knots).predict(sample_t)
        for i in range(sample_x.shape[-1])
    ]
    return np.stack(sample).T

def get_natural_cubic_spline_model(x, y, minval=None, maxval=None, n_knots=None, knots=None):
    """
    Get a natural cubic spline model for the data.

    For the knots, give (a) `knots` (as an array) or (b) minval, maxval and n_knots.

    If the knots are not directly specified, the resulting knots are equally
    space within the *interior* of (max, min).  That is, the endpoints are
    *not* included as knots.

    Parameters
    ----------
    x: np.array of float
        The input data
    y: np.array of float
        The outpur data
    minval: float 
        Minimum of interval containing the knots.
    maxval: float 
        Maximum of the interval containing the knots.
    n_knots: positive integer 
        The number of knots to create.
    knots: array or list of floats 
        The knots.

    Returns
    --------
    model: a model object
        The returned model will have following method:
        - predict(x):
            x is a numpy array. This will return the predicted y-values.
    """

    if knots:
        spline = SmoothNaturalCubicSpline(knots=knots)
    else:
        spline = SmoothNaturalCubicSpline(max=maxval, min=minval, n_knots=n_knots)

    p = Pipeline([
        ('nat_cubic', spline),
        ('regression', LinearRegression(fit_intercept=True))
    ])

    p.fit(x, y)

    return p

class AbstractSpline(BaseEstimator, TransformerMixin):
    """Base class for all spline basis expansions."""

    def __init__(self, max=None, min=None, n_knots=None, n_params=None, knots=None):
        if knots is None:
            if not n_knots:
                n_knots = self._compute_n_knots(n_params)
            knots = np.linspace(min, max, num=(n_knots + 2))[1:-1]
            max, min = np.max(knots), np.min(knots)
        self.knots = np.asarray(knots)

    @property
    def n_knots(self):
        return len(self.knots)

    def fit(self, *args, **kwargs):
        return self

class SmoothNaturalCubicSpline(AbstractSpline):
    """Apply a natural cubic basis expansion to an array.
    The features created with this basis expansion can be used to fit a
    piecewise cubic function under the constraint that the fitted curve is
    linear *outside* the range of the knots..  The fitted curve is continuously
    differentiable to the second order at all of the knots.
    This transformer can be created in two ways:
      - By specifying the maximum, minimum, and number of knots.
      - By specifying the cutpoints directly.  

    If the knots are not directly specified, the resulting knots are equally
    space within the *interior* of (max, min).  That is, the endpoints are
    *not* included as knots.
    Parameters
    ----------
    min: float 
        Minimum of interval containing the knots.
    max: float 
        Maximum of the interval containing the knots.
    n_knots: positive integer 
        The number of knots to create.
    knots: array or list of floats 
        The knots.
    """

    def _compute_n_knots(self, n_params):
        return n_params

    @property
    def n_params(self):
        return self.n_knots - 1

    def transform(self, X, **transform_params):
        X_spl = self._transform_array(X)
        if isinstance(X, pd.Series):
            col_names = self._make_names(X)
            X_spl = pd.DataFrame(X_spl, columns=col_names, index=X.index)
        return X_spl

    def _make_names(self, X):
        first_name = "{}_spline_linear".format(X.name)
        rest_names = ["{}_spline_{}".format(X.name, idx)
                      for idx in range(self.n_knots - 2)]
        return [first_name] + rest_names

    def _transform_array(self, X, **transform_params):
        X = X.squeeze()
        try:
            X_spl = np.zeros((X.shape[0], self.n_knots - 1))
        except IndexError: # For arrays with only one element
            X_spl = np.zeros((1, self.n_knots - 1))
        X_spl[:, 0] = X.squeeze()

        def d(knot_idx, x):
            def ppart(t): return np.maximum(0, t)

            def cube(t): return t*t*t
            numerator = (cube(ppart(x - self.knots[knot_idx]))
                         - cube(ppart(x - self.knots[self.n_knots - 1])))
            denominator = self.knots[self.n_knots - 1] - self.knots[knot_idx]
            return numerator / denominator

        for i in range(0, self.n_knots - 2):
            X_spl[:, i+1] = (d(i, X) - d(self.n_knots - 2, X)).squeeze()
        return X_spl


KNOTS = 20


org_folder = 'spring'
data_folder = 'smooth_spring'
os.makedirs(data_folder, exist_ok=True)


edges_train = np.load(os.path.join(org_folder, 'edges_train.npy'))
edges_test = np.load(os.path.join(org_folder, 'edges_test.npy'))
time_train = np.load(os.path.join(org_folder, 'time_train.npy'))
time_test = np.load(os.path.join(org_folder, 'time_test.npy'))
np.save(os.path.join(data_folder, 'edges_train.npy'), edges_train)
np.save(os.path.join(data_folder, 'edges_test.npy'), edges_test)
np.save(os.path.join(data_folder, 'time_train.npy'), time_train)
np.save(os.path.join(data_folder, 'time_test.npy'), time_test)

t_max = time_train.shape[1] - 1


loc_train = np.load(os.path.join(org_folder, 'loc_train.npy'))
loc_train = loc_train.reshape(*loc_train.shape[:2], -1)
loc_test = np.load(os.path.join(org_folder, 'loc_test.npy'))
loc_test = loc_test.reshape(*loc_test.shape[:2], -1)

loc_train = np.stack([smooth_sample(
    time_train[i],
    loc_train[i],
    0, t_max, KNOTS
    ) for i in tqdm(range(len(time_train)))
])
loc_test = np.stack([smooth_sample(
    time_test[i],
    loc_test[i],
    0, t_max, KNOTS
    ) for i in tqdm(range(len(time_test)))
])

np.save(os.path.join(data_folder, 'loc_train.npy'), loc_train)
np.save(os.path.join(data_folder, 'loc_test.npy'), loc_test)


vel_train = np.load(os.path.join(org_folder, 'vel_train.npy'))
vel_train = vel_train.reshape(*vel_train.shape[:2], -1)
vel_test = np.load(os.path.join(org_folder, 'vel_test.npy'))
vel_test = vel_test.reshape(*vel_test.shape[:2], -1)

vel_train = np.stack([smooth_sample(time_train[i], vel_train[i], 0, t_max, KNOTS) for i in tqdm(range(len(time_train)))])
vel_test = np.stack([smooth_sample(time_test[i], vel_test[i], 0, t_max, KNOTS) for i in tqdm(range(len(time_test)))])

np.save(os.path.join(data_folder, 'vel_train.npy'), vel_train)
np.save(os.path.join(data_folder, 'vel_test.npy'), vel_test)



