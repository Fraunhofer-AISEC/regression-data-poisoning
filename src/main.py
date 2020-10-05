# ==============================================================================
# This file contains the entry point to the main experiment:
# It will iterate over all combinations of regressors / data sets, poison them using Flip/StatP,
# then clean using Trim/iTrim, and finally report the results.
# ==============================================================================

import logging

import numpy
import os
import pandas as pd
import warnings
from os.path import abspath
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler

from src import settings
from src.ExpHelper.ExpHelper import ExpHelper
from src.attacks import _poison_flip, shuffle_x_y, _poison_statP
from src.defenses import defense_both_trim_and_itrim
from src.settings import datasets, run_settings

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def df_minmax(df):
    """
    Apply MinMax operation to a given dataframe (both features and target)
    """
    # return df
    if settings.MinMaxSettings.use_minmax:
        s = MinMaxScaler(feature_range=(settings.MinMaxSettings.minmax_min, settings.MinMaxSettings.minmax_max))
        v = s.fit_transform(df.values)
        return pd.DataFrame(v, columns=df.columns)
    else:
        return df


def outer_attack_loop(all_args_tuple):
    """
    This is the main loop which handles the logic for calling the attacks_deprecated and defenses.
    Called in parallel by ExpHelper
    """
    # load values from input args
    dataset_tuple, regressor, eps, eps_hat, run = all_args_tuple
    regressor_name = str(regressor[0]()).split('(')[0]
    dat, data_name = dataset_tuple
    logging.debug(f'Run: {run}, Eps: {eps}, Dataset: {data_name}, Regressor: {regressor_name}, eps_hat: {eps_hat}')
    dat = dat()

    # subsample, and split into train-test and substitute data set
    target_n = min(settings.nmax, dat.shape[0])
    dat_subsamples = ExpHelper.do_or_load_intermediates('./tmp',
                                                        [run, data_name, regressor_name],  # shoud not depend on eps!
                                                        lambda: dat.sample(n=target_n, replace=False).reset_index(
                                                            drop=True),
                                                        'base_data.dill')
    assert not numpy.any(dat_subsamples.isnull()), "Data has Null values"
    assert not numpy.any(dat_subsamples.isna()), "Data has NaN values"
    del dat
    # how much of the full data is to be used for train-test. Rest goes to substitute
    frac_true = 0.75
    dat_true = dat_subsamples[:int(frac_true * target_n)]
    dat_substitue = dat_subsamples[:int(frac_true * target_n)]

    # minmax
    dat_true = df_minmax(dat_true)
    dat_substitue = df_minmax(dat_substitue)

    # setup train/test split
    test_size = 0.2
    x_train, x_test, y_train, y_test = train_test_split(dat_true.drop('target', axis=1).values,
                                                        dat_true.loc[:, 'target'].values,
                                                        test_size=test_size)
    x_substitue, y_substitue = dat_substitue.drop('target', axis=1).values, dat_substitue.loc[:, 'target'].values

    # set up result dictionary
    res_tmp = {'data': data_name, 'data_shape': dat_true.shape, 'eps': eps, "eps_hat": eps_hat, 'run': run,
               'regressor': regressor_name}

    # Stop if we only want to create train_test_data
    if settings.no_attack_run:
        return res_tmp

    # Calculating the number of poison samples
    n_poisoned = int(eps*x_train.shape[0])

    # ========= ATTACKS ===========
    for attack in [
        _poison_flip,
        _poison_statP,
    ]:
        # check and execute attack
        assert attack.__name__.startswith('_poison'), f'Attack {attack.__name__} does not start with "_poison". '

        # get poisoned data using attack
        x_poisoned, y_poisoned = attack(x_substitue, y_substitue, n_poisoned=n_poisoned, regressor=regressor)

        # assert correctness
        assert 0 < x_poisoned.shape[0] or n_poisoned == 0, \
            f'Attack {attack.__name__} returns no poison samples for dataset {data_name}.'
        assert 0 < y_poisoned.shape[0] or n_poisoned == 0, \
            f'Attack {attack, __name__} returns no poison samples for dataset {data_name}.'

        # concat poison to clean data
        x_both, y_both, y_is_poisoned = shuffle_x_y(
            numpy.concatenate((x_train, x_poisoned)),
            numpy.concatenate((y_train, y_poisoned)),
            numpy.concatenate((numpy.zeros(y_train.shape), numpy.ones(y_poisoned.shape))))

        # do grid search ONCE on poisoned data!
        g = GridSearchCV(estimator=regressor[0](), param_grid=regressor[1], refit=False,
                         cv=3, scoring='neg_mean_squared_error', iid=False)
        g.fit(x_both, y_both)
        best_regressor = lambda: g.estimator.__class__(**g.best_params_)

        # get model
        poisoned_regressor = best_regressor()
        poisoned_regressor.fit(x_both, y_both)

        # predict the test set
        pred = poisoned_regressor.predict(x_test)
        attack_mse = numpy.average(numpy.square(pred - y_test))

        # save to result dataframe, then delete variables
        res_tmp[f"_mse_{attack.__name__}"] = attack_mse
        del poisoned_regressor, pred, attack_mse

        # =======================================================================================
        # ======================================== DEFENSE ======================================
        # =======================================================================================
        # if we attacked using the 'flip' attack, don't defend. We only eval defense for Flip since it's stronger
        if "statp" in attack.__name__.lower():
            continue
        trim_and_itrim_dict = defense_both_trim_and_itrim(x_both, y_both,
                                                          regressor=best_regressor,
                                                          eps_hat=eps if eps_hat is None else eps_hat)

        # for trim and iTrim
        for defense_variant in trim_and_itrim_dict.keys():
            if not defense_variant.startswith("defense"):
                continue

            # get defended data x, y
            x_defended, y_defended = trim_and_itrim_dict[defense_variant]
            assert 0 < x_defended.shape[0], f'Defense removes all data points, dataset {data_name}.'
            assert 0 < y_defended.shape[0], f'Defense removes all data points, dataset {data_name}.'

            # evaluate and save defense's result
            x_defended, y_defended = shuffle_x_y(x_defended, y_defended)
            defended_regressor = best_regressor()
            defended_regressor.fit(x_defended, y_defended)
            pred = defended_regressor.predict(x_test)
            defense_mse = numpy.average(numpy.square(pred - y_test))
            res_tmp[f"_mse_{attack.__name__}/{defense_variant}"] = defense_mse

        # Comment in for debug information
        # if numpy.abs(eps - trim_and_itrim_dict['eps_est_itrim']) > 0.001:
        #     print(" est_trim:", trim_and_itrim_dict['eps_est_itrim'], "true:", eps, data_name)

        # take note how well itrim estimated eps - perfect delta is zero, larger than zero is overestimation, etc.
        res_tmp[f"_mse_{attack.__name__}/epshat-eps"] = trim_and_itrim_dict['eps_est_itrim'] - eps

    # Lastly, compute clean MSE
    # ============ CLEAN ===============
    clean_regressor = best_regressor()
    clean_regressor.fit(x_train, y_train)
    pred = clean_regressor.predict(x_test)
    clean_mse = numpy.average(numpy.square(pred - y_test))
    res_tmp['_mse_clean'] = clean_mse
    del clean_regressor, pred

    # Return
    return res_tmp


if __name__ == "__main__":
    logging.info(f'Number of datasets: {len(datasets)}')
    logging.info(f'Running attacks_deprecated: {not settings.no_attack_run}')
    if settings.do_not_load_from_disk:
        logging.warning('Currently set to force new data generation.')

    logging.info(f'Running attacks_deprecated and defenses and storing in {abspath(settings.store_path)}')
    e = ExpHelper(name='e0', settings=run_settings)
    res = e.parallelize(outer_attack_loop, ['dataset_tuple', 'regressor', 'eps', "eps_hat", 'run'],
                        cpus=1 if settings.no_attack_run else min(15, os.cpu_count()*0.2))
    df = pd.DataFrame(res)
    df.to_csv(settings.store_path, index=False)
    logging.info("Written to " + settings.store_path)
    if settings.no_attack_run:
        print("ATTENTION: This run has created the train/test splits, but not executed attacks and defenses"
              "Change settings.no_attack_run to False and restart main!")

