import numpy

import copy
import math
import numpy
import pandas as pd

# ===================================== HELPERS =============================
from sklearn.model_selection import GridSearchCV

from src import settings


def shuffle_x_y(x, y, additional=None):
    assert isinstance(x, numpy.ndarray), print(f"type x: {type(x)}")
    assert isinstance(y, numpy.ndarray), print(f"type x: {type(x)}")
    if additional is not None:
        assert isinstance(additional, numpy.ndarray), print(f"type x: {type(additional)}")

    x = copy.deepcopy(x)
    y = copy.deepcopy(y)

    random_perm = numpy.random.permutation(x.shape[0])
    x_shuffled = x[random_perm]
    y_shuffled = y[random_perm]

    assert x.shape == x_shuffled.shape
    assert y.shape == y_shuffled.shape

    if additional is not None:
        return x_shuffled, y_shuffled, additional[random_perm]
    return x_shuffled, y_shuffled


def copy_and_numpify(x):
    x = copy.deepcopy(x)
    if isinstance(x, pd.DataFrame):
        return x.values
    elif isinstance(x, numpy.ndarray):
        return x
    else:
        raise TypeError("Is neither DF nor numpy array")


def _attack_helper(x, y, regressor, pred_this=None):
    assert isinstance(x, numpy.ndarray)
    assert isinstance(y, numpy.ndarray)

    # mu not used
    x = copy.deepcopy(x)

    r = regressor()
    r.fit(x, y)
    pred = r.predict(x) if pred_this is None else r.predict(pred_this)
    return pred


def _get_sampled_x(x, eps):
    mu = numpy.mean(x, axis=0)
    cov = numpy.cov(x.T)
    n_p = int(eps * x.shape[0])
    return numpy.random.multivariate_normal(mean=mu, cov=cov, size=n_p)


def _poison_flip(x, y, n_poisoned, regressor=None, best_relax=1):
    # regressor is just here to have unified api with statP
    del regressor
    if n_poisoned < 1:
        return numpy.zeros(shape=(0, x.shape[1])), numpy.zeros(shape=(0,))

    tmax = numpy.max(y)
    tmin = numpy.min(y)

    # find out if there is more potential shifting the decision surface uniformly towards the max or min
    upper_max_abs_error = numpy.abs(y - tmax)
    lower_max_abs_error = numpy.abs(y - tmin)

    direction = ['up' if upper_max_abs_error[i] > lower_max_abs_error[i] else 'down' for i in range(y.shape[0])]

    # get x_p (subset of true x)
    delta = [upper_max_abs_error[i] if direction[i] == 'up' else lower_max_abs_error[i] for i in range(y.shape[0])]

    # select those who are 1) farthest
    assert best_relax >= 1
    best = numpy.argsort(delta)[-n_poisoned * best_relax:]
    best = numpy.random.choice(best, size=n_poisoned, replace=False)
    x_p = x[best]

    # get y_p
    # target = [tmin if (numpy.max(y) - numpy.min(y)) / 2 < yyy else tmax for yyy in y[best]]
    target = [tmin if direction[i] == 'down' else tmax for i in best]
    y_p = numpy.asarray(target)
    return x_p, y_p


def _poison_statP(x, y, n_poisoned, regressor):
    if n_poisoned < 1:
        return numpy.zeros(shape=(0, x.shape[1])), numpy.zeros(shape=(0,))

    # find best regressor
    g = GridSearchCV(estimator=regressor[0](), param_grid=regressor[1], refit=False,
                     cv=3, scoring='neg_mean_squared_error', iid=False)
    g.fit(x, y)
    best_regressor = lambda: g.estimator.__class__(**g.best_params_)
    del regressor

    for i in range(x.shape[1]):
        if numpy.unique(x[:, i]).size < 2:
            # print(f"Found constant variable in column {i}, adding small noise.")
            x[:, i] = x[:, i] + numpy.random.normal(loc=0, scale=1e-4, size=x[:, i].shape)

    mu = numpy.mean(x, axis=0)
    cov = numpy.cov(x.T)
    assert not numpy.any(numpy.isnan(cov))

    tmax = numpy.max(y)
    tmin = numpy.min(y)
    assert 0 - 1e-6 <= tmin, f'Non MinMax not implemented (y_min = {tmin})'
    assert 1 + 1e-6 >= tmax, f'Non MinMax not implemented, (y_max = {tmax})'

    x_p = numpy.random.multivariate_normal(mean=mu, cov=cov, size=n_poisoned)
    x_p = numpy.clip(numpy.round(x_p),
                     a_min=settings.MinMaxSettings.minmax_min, a_max=settings.MinMaxSettings.minmax_max)
    pred_x_p = _attack_helper(x=x, y=y, regressor=best_regressor, pred_this=x_p)
    y_p = numpy.round(1 - pred_x_p)

    # return
    return x_p, y_p
