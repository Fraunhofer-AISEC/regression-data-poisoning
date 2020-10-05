import logging

import tensorflow as tf
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, ElasticNet, Lasso, HuberRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from src.datasets import load_warfarin, load_loan, load_armes_housing, load_github_regression, \
    load_from_keel_repo


class MinMaxSettings:
    use_minmax = bool(1)
    minmax_min = 0
    minmax_max = 1
    width = minmax_max - minmax_min
    assert width > 0


def attacker_objective(ytrue, ypred):
    return tf.square(ytrue - ypred)


store_path = './out/csvs/out.csv'

no_attack_run = bool(0)  # <-- set this to True the first time you run the project.
do_not_load_from_disk = bool(0)
do_debug_visualisation = bool(0)

# Set the max and min number of points for the data sets in the empirical evaluation
nmin = 1000
nmax = 10000

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(levelname)-8s %(message)s')

# =====================================================================================================================
# List the used data sets here
datasets = [
       (load_warfarin, 'warfarin'),
       (load_loan, 'loan'),
       (load_armes_housing, 'armesHousing'),
       (lambda: load_github_regression("Abalone.csv", "Rings"), "rings"),
       (lambda: load_github_regression("boston.csv", "HousValue"), "boston"),
       (lambda: load_github_regression("Accel.csv", "acceleration"), "accel"),
       (lambda: load_github_regression("availPwr.csv", "available.power"), "availPwr"),
       (lambda: load_github_regression("bank8FM.csv", "rej"), "bank8fm"),
       (lambda: load_github_regression("cpuSm.csv", "usr"), "cpu"),
       (lambda: load_github_regression("fuelCons.csv", "fuel.consumption.country"), "fuelCons"),
       (lambda: load_github_regression("heat.csv", "heat"), "heat"),
       (lambda: load_github_regression("maxTorque.csv", "maximal.torque"), "torque"),
   ] + load_from_keel_repo()
datasets = [x for x in datasets if x[0]().shape[0] > nmin]

print(f'Using {len(datasets)} data sets with at least {nmin} instances')

# =====================================================================================================================
# Alpha controls regularisation error. if MinMax scale is activated, the regularisation becomes stronger
alpha_ = [1e-8, 1e-7, 1e-5, 1e-3, 1e-1, 1e-0, 1e1]
run_settings = {
    'run': [0],
    'dataset_tuple': datasets,
    'regressor': [
        (Lasso, {'alpha': alpha_}),
        (ElasticNet, {'alpha': alpha_}),
        (Ridge, {'alpha': alpha_}),
        (HuberRegressor, {'epsilon': [1.35, 1.2]}),  # where 'close' points are L2, and outliers are L1 penalized
        (MLPRegressor, {'hidden_layer_sizes': [(20,), (20, 20)], 'alpha': [1e-3], 'max_iter': [40]}),
        (SVR, {'kernel': ['rbf'], 'gamma': ['scale'], 'C': [0.1, 1, 10]}),
        (KernelRidge, {'kernel': ['rbf'], 'alpha': alpha_}),
    ],
    # The true degree of poisoning
    'eps': [
        0.0,
        0.02,
        0.04,
        0.06,
        0.08,
        0.1
    ],
    # The estimated degree of poisoning
    "eps_hat": [
        # Currently used as eps_hat for trim, and ignored by the rest
        # None,  # If None, mu := eps
        0.14
    ]
}
