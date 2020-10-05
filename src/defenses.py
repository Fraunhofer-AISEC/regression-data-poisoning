import copy

import numpy
import pandas


def defense_trim(x, y, regressor, eps_hat, tol, return_meta=False):
    assert 0 <= eps_hat < 1
    n = numpy.floor(x.shape[0] / (1 + eps_hat)).astype(int)  # number of non-poisoned points

    # initial fit and estimation of best hyper parameters
    current_data_indices = numpy.random.choice(x.shape[0], n, replace=False)
    r = regressor()
    r.fit(x[current_data_indices], y[current_data_indices])
    train_squared_error = numpy.square(r.predict(x)-y)

    # if eps == 0, no use in starting defense loop
    if eps_hat == 0.0:
        return (x, y, numpy.mean(train_squared_error)) if return_meta else (x, y)

    # main loop: Iteratively
    maxi = 20
    for i in range(maxi):
        # get those indexes which have lowest error on current model
        current_data_indices = numpy.argsort(train_squared_error)[:n]
        rr = regressor()
        rr.fit(x[current_data_indices], y[current_data_indices])  # fit on them
        new_train_squared_error = numpy.square(rr.predict(x)-y)  # re-check residuals with respect to new regressor
        delta_mse = numpy.mean(numpy.abs(train_squared_error - new_train_squared_error))
        train_squared_error = new_train_squared_error
        # print(i, "--", numpy.mean(train_squared_error[current_data_indices]))
        if delta_mse < tol:
            break

    # some sanity asserts, then return
    xc, yc = x[current_data_indices], y[current_data_indices]
    assert xc.shape[0] > 0
    assert yc.shape[0] > 0
    if not return_meta:
        return xc, yc
    else:
        return xc, yc, numpy.mean(numpy.square(rr.predict(xc) - yc))


def defense_both_trim_and_itrim(x, y, regressor, eps_hat, treshold_def=1e-3, tol=1e-4):
    """
    Defend a given data set. Since iTrim calls Trim, we group these defenses together in a single file here.
    :param x:
    :param y:
    :param regressor:
    :param eps_hat:
    :param treshold_def:    The threshold for itrim, which we use to establish the 'knick'
    :param tol:             A parameter for trim, to improve runtime (early stopping)
    :return:
    """

    # If estimated degree of poison is 0, do nothing
    if eps_hat == 0:
        tmp = defense_trim(x, y, regressor, eps_hat=0, tol=tol)
        return {'defense_itrim': copy.deepcopy(tmp), 'defense_trim': copy.deepcopy(tmp), 'eps_est_itrim': 0}

    # Prepare some indermediate results variable
    # We save iTrim results into dict, we 1) safe computation, and 2) do not have variance between trim and iTrim runs
    res = dict()
    df = pandas.DataFrame()

    # main iTrim loop
    for i in numpy.arange(start=0, stop=eps_hat+0.02, step=0.02):
        xc, yc, train_error = defense_trim(x, y, regressor, eps_hat=i, return_meta=True, tol=tol)
        res[i] = (xc, yc)
        df = df.append({'eps_hat': i, 'train_error': train_error}, ignore_index=True)

    df['error_shifted'] = df['train_error'].shift(periods=-1, fill_value=0)
    df['delta'] = df['train_error'] - df['error_shifted']

    # This code here can be commented in to get debug information
    # df.plot(x="eps_hat", y='train_error', figsize=[6, 4], legend=False)
    # import matplotlib.pyplot as plt
    # plt.xticks(df['eps_hat'])
    # plt.ylabel('Train MSE')
    # plt.savefig("/home/mueller/Desktop/fig2.png")
    # plt.show()
    # ExpHelper.ExpHelper.pretty_print(df)
    # import random
    # df.to_csv("visualisation/csvs/train_error.csv")

    df.set_index('eps_hat')

    # apply iTrim threshold selection
    below_thres = df[df['delta'].apply(abs) < treshold_def]
    if below_thres.shape[0] == 0:
        estimated_eps = eps_hat
    else:
        estimated_eps = numpy.min(below_thres['eps_hat'].values)
    # print("est eps", estimated_eps)
    return {'defense_itrim': res[estimated_eps], 'defense_trim': res[eps_hat], "eps_est_itrim": estimated_eps}

