import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

from src.ExpHelper.ExpHelper import ExpHelper

x_ticks = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]


def plot_trim_diff_eps_estimate(path_test):
    """
    Plots the results of using trim with different estimates of $\hat{\epsilon}$.
    :return:
    """
    # Preparing the data frame for plotting
    df = pd.read_csv(path_test)
    df = df[(df['data'] == 'loan') | (df['data'] == 'warfarin') | (df['data'] == 'armesHousing')]
    df = df.groupby(by=["eps_hat", 'regressor']).median()
    df_new = pd.DataFrame()
    for i in df.index:
        df_new.loc[i[0], i[1]] = df.loc[i, '_mse__poison_flip/defense_trim']

    # Plotting the figure with all regressors
    df_new.plot(figsize=[6, 4])
    plt.xticks(x_ticks)
    plt.ylabel("Test MSE")
    # plt.ylim(0, 0.02)
    plt.xlabel('$\^\epsilon$')
    plt.tight_layout()
    plt.savefig("./out/all_effect_trim_eps_estimate.png")
    plt.show()

    # Plotting only the median over all regressors
    df_median = df_new.median(axis=1)
    df_median.plot(label='Median over all regressors', legend=True, figsize=(6, 4))
    plt.xticks(x_ticks)
    plt.ylabel("Test MSE")
    # plt.ylim(0, 0.02)
    plt.xlabel('$\^\epsilon$')
    plt.tight_layout()
    plt.savefig("./out/median_effect_trim_eps_estimate.png")
    plt.show()


def plot_train_test_mse_trim(path_test, path_train, which="warfarin"):
    """
    Plots a figure containing the train and test error with different estimates of $\hat{\epsilon}$.
    :return:
    """
    # Preparing the test data set
    df = pd.read_csv(path_test)
    df = df[(df['data'] == which)]
    df = df.groupby(by=["eps_hat", 'regressor']).median()
    df_new = pd.DataFrame()
    for i in df.index:
        df_new.loc[i[0], i[1]] = df.loc[i, '_mse__poison_flip/defense_trim']

    # Plotting the test error
    df_new = df_new["KernelRidge"]
    df_new.plot(label='Test MSE', legend=True, figsize=[6, 4])

    # Plotting train error
    df_train = pd.read_csv(path_train, index_col='eps_hat')
    df_train["train_error"].plot(label='Train MSE', legend=True, figsize=[6, 4], style=["-.", "--"])

    # Defining general plot stuff and storing everything
    plt.xticks(x_ticks)
    plt.ylabel("MSE")
    plt.ylim([0, 0.02])
    plt.xlabel('$\^\epsilon$')
    plt.tight_layout()
    plt.savefig(f"./out/{which}_trim_eps_estimate.png")
    plt.show()


def plot_statp_on_nonlinear():
    """
    Plots the test mse of the statP attack for non-linear regressors KernelRidge, MLP, SVR.
    :return:
    """
    # Preparing the data frame for plotting
    df = pd.read_csv('./out/csvs/statP.csv')
    df = df[(df['regressor'] == 'KernelRidge') | (df['regressor'] == 'MLPRegressor') | (df['regressor'] == 'SVR')]
    df = df.groupby(by=['eps', 'regressor']).median()
    df_new = pd.DataFrame()
    for i in df.index:
        df_new.loc[i[0], i[1]] = df.loc[i, '_mse__poison_statP']

    # Plotting everything
    df_new.plot(figsize=[6, 4])
    print(tabulate(df_new, headers="keys", tablefmt="psql"))
    plt.xticks(x_ticks)
    plt.ylabel("Test MSE")
    plt.xlabel('$\epsilon$')
    plt.ylim(0, 0.025)
    plt.tight_layout()
    plt.savefig("./out/statp_on_nonlinear.png")
    plt.show()


def plot_flip_results(path):
    """
    Plots the results of the flip attack. One image for median, one for all regressors individually.
    :return:
    """
    # Preparing the data frame for plotting
    df = pd.read_csv(path)
    df = df.groupby(by=['eps', 'regressor']).median()
    df_new = pd.DataFrame()
    for i in df.index:
        df_new.loc[i[0], i[1]] = df.loc[i, '_mse__poison_flip']

    # Plotting the figure with all regressors
    df_new.plot(figsize=[6, 4])
    plt.xticks(x_ticks)
    plt.ylabel("Test MSE")
    plt.ylim(0, 0.025)
    plt.xlabel('$\epsilon$')
    plt.tight_layout()
    plt.savefig("./out/all_reg_flip_results.png")
    plt.show()

    # Plotting only the median over all regressors
    df_median = df_new.median(axis=1)
    df_median.plot(label='Median over all regressors', legend=True, figsize=(6, 4))
    plt.xticks(x_ticks)
    plt.ylabel("Test MSE")
    plt.ylim(0, 0.025)
    plt.xlabel('$\epsilon$')
    plt.tight_layout()
    plt.savefig("./out/median_flip_results.png")
    plt.show()


def plot_trim_itrim_results(path):
    """
    Plots the results of applying trim and itrim against flip.
    For each defense, it generates one plot with all regressors.
    Also it generates a plot containing the median of trim/clean and itrim/clean.
    Lastly, it generates plots trim/itrim. Both, median only and all regressors.
    :return:
    """
    # make the table
    for divisor, exp_name in [
        ('_mse__poison_statP', 'statP'),
        ('_mse__poison_flip', 'Flip'),
        ('_mse__poison_flip/defense_trim', "Trim"),
        ('_mse__poison_flip/defense_itrim', "iTrim"),
    ]:
        df = pd.read_csv(path)
        # df = df[df['regressor'] == "SVR"]
        df = df.groupby(by=['data', 'eps']).median()
        df = df.drop('run', axis=1)
        df.to_latex("./out/all_results.tex")
        df_2 = pd.DataFrame()
        for i in df.index:
            df_2.loc[i[0], i[1]] = df.loc[i, divisor] / df.loc[i, '_mse_clean']
        df_2 = df_2.T.reset_index()
        df_2 = df_2.rename(columns={'index': 'eps'})
        print(tabulate(df_2, headers="keys", tablefmt="psql"))
        ax = plt.gca()

        df_2_mean = df_2.set_index('eps').mean(axis=1).reset_index().rename(columns={0: "mean"})
        df_2.plot(x='eps', figsize=[12, 8], ax=ax)
        df_2['mean'] = df_2_mean['mean']
        df_2.plot(x='eps', ax=ax, y='mean', linewidth=4, color='black', linestyle='-.')
        # df_2_mean.plot(x='eps', y='mean', ax=ax, linewidth=7, legend=None)
        # df_2_mean.plot(label="mean")
        plt.legend(loc='upper right')
        plt.xticks(x_ticks)
        plt.ylabel(f"{exp_name}/Clean")
        plt.ylim(1, 3 if 'defense' in divisor else 10)
        plt.xlabel('$\epsilon$')
        plt.tight_layout()
        plt.savefig(f"./out/all_regressors_{exp_name}.png")
        plt.show()
    del df, df_2

    # making the plots
    # Preparing the base data
    df = pd.read_csv(path)
    df = df.groupby(by=['eps', 'regressor']).median()

    # Creating Data Frame for trim plotting
    df_trim = pd.DataFrame()
    for i in df.index:
        df_trim.loc[i[0], i[1]] = df.loc[i, '_mse__poison_flip/defense_trim'] / df.loc[i, '_mse_clean']

    # Plotting the figure for trim with all regressors
    df_trim.plot(figsize=[6, 4])
    plt.xticks(x_ticks)
    plt.ylabel("Test MSE Trim/Clean")
    plt.ylim(1, 3)
    plt.xlabel('$\epsilon$')
    plt.tight_layout()
    plt.savefig("./out/all_reg_trim_results.png")
    plt.show()

    # Plotting only the median over all regressors for trim
    df_trim_median = df_trim.median(axis=1)
    df_trim_median.plot(label='Median Trim', legend=True, figsize=(6, 4))

    # Creating data frame for plotting itrim
    df_itrim = pd.DataFrame()
    for i in df.index:
        df_itrim.loc[i[0], i[1]] = df.loc[i, '_mse__poison_flip/defense_itrim'] / df.loc[i, '_mse_clean']

    # Plotting only the median over all regressors for itrim and finalizing figure
    df_itrim_median = df_itrim.median(axis=1)
    df_itrim_median.plot(label='Median iTrim', legend=True, figsize=(6, 4), style=["-.", "--"])
    plt.xticks(x_ticks)
    plt.ylabel("Test MSE Defense/Clean")
    # plt.ylim(1.2, 1.7)
    plt.xlabel('$\epsilon$')
    plt.tight_layout()
    plt.savefig("./out/median_trim_itrim_results.png")
    plt.show()

    # Plotting the figure with all regressors for itrim
    df_itrim.plot(figsize=[6, 4])
    plt.xticks(x_ticks)
    plt.ylabel("Test MSE iTrim/Clean")
    plt.ylim(1, 3)
    plt.xlabel('$\epsilon$')
    plt.tight_layout()
    plt.savefig("./out/all_reg_itrim_results.png")
    plt.show()

    # Creating data frame for plotting trim/itrim
    df_trim_itrim = df_trim / df_itrim

    # Plotting the figure with all regressors for trim/itrim
    df_trim_itrim.plot(figsize=[6, 4])
    plt.xticks(x_ticks)
    plt.ylabel("Test MSE Trim/iTrim")
    # plt.ylim(0.8, 1.7)
    plt.xlabel('$\epsilon$')
    plt.tight_layout()
    plt.savefig("./out/all_reg_trim_vs_itrim_results.png")
    plt.show()

    # Plotting only the median over all regressors for trim/itrim
    df_trim_itrim_median = df_trim_median / df_itrim_median
    ax = df_trim_itrim_median.plot(label='Median Trim/iTrim', legend=True, figsize=(6, 4))
    plt.xticks(x_ticks)
    plt.ylabel("Test MSE Trim/iTrim")
    # plt.ylim(0.9, 1.2)
    ax.axhline(y=1, color='black')
    plt.xlabel('$\epsilon$')
    plt.tight_layout()
    plt.savefig("./out/median_trim_vs_itrim_results.png")
    plt.show()


def count_data_sets(path):
    df = pd.read_csv(path)['data']
    print(f"{df.unique().shape} data sets")


def get_errors_in_itrim_epshat_estimate(path):
    # Let's count how good the eps_hat of itrim is
    df = pd.read_csv(path)['_mse__poison_flip/epshat-eps']
    print(df.value_counts())


def make_dataset_table():
    """
    Create a table with all the data sets used
    """
    from src.settings import datasets
    from src import settings
    datasets = [x for x in datasets if x[0]().shape[0] > settings.nmin]
    res = pd.DataFrame()
    for d in datasets:
        ds = d[0]()
        res = res.append({'Name': d[1], 'n': int(ds.shape[0]), 'features': int(ds.shape[1] - 1),
                          # 'target domain': f'[{np.min(ds.loc[:, "target"]):.1f}, {np.max(ds.loc[:, "target"]):.1f}]'
                          },
                         ignore_index=True)

    res = res.sort_values('Name').reset_index(drop=True)
    res['features'] = res['features'].round(0)
    res['n'] = res['n'].round(0)

    print(tabulate(res, headers="keys", tablefmt="psql"))
    res.to_latex('./out/used_datasets.tex')


if __name__ == "__main__":
    # Generally:
    # eps = poison rate, eps_hat = estimated poison rate
    # - test_error.csv has a fixed eps, varies over eps_hat,
    # and only load, warfarin and hours data set
    # - full_runs have a fixed eps_hat (0.2), varying eps, and all data sets
    # path_full_run = './src/visualisation/csvs/epsmax010/nov_24_5runs_epshat_014.csv'
    path_full_run = './out/csvs/Feb_05_2020.csv'
    get_errors_in_itrim_epshat_estimate(path_full_run)
    count_data_sets(path_full_run)
    plot_trim_itrim_results(path_full_run)
    plot_flip_results(path_full_run)
    plot_statp_on_nonlinear()
    plot_trim_diff_eps_estimate(path_test='./out/csvs/test_error_eps_004.csv')
    for w in ["warfarin", "loan", "armesHousing"]:
        plot_train_test_mse_trim(path_test='./out/csvs/test_error_eps_004.csv',
                                 path_train='./out/csvs/train_error_eps_004.csv',
                                 which=w)
    make_dataset_table()
    # main()
