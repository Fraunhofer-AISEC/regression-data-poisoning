from sys import exit

import numpy as np
import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate

from src import settings
from src.attacks import shuffle_x_y, _poison_flip
from src.datasets import load_warfarin
from src.defenses import defense_both_trim_and_itrim
from src.settings import run_settings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def main():
    """
    This function creates the results tables for the warfarin use case from the paper
    """
    load = bool(0)

    # we may reuse an existing data set by setting the load variable to True
    if load:
        result_df = pd.read_csv('./out/csvs/warfarin_results.csv', index_col=0)
        result_df = result_df.groupby(by=["Regressor"]).mean()
        result_df['MAE P/C'] = result_df['MAE P'] / result_df['MAE C']
        result_df['MAE D/C'] = result_df['MAE D'] / result_df['MAE C']
        result_df['MSE P/C'] = result_df['MSE P'] / result_df['MSE C']
        result_df['Dec W20 P/C'] = (1 - result_df['Within 20% P'] / result_df['Within 20% C']) * 100
        result_df['Dec W20 D/C'] = (1 - result_df['Within 20% D'] / result_df['Within 20% C']) * 100
        result_df.loc['Median', :] = result_df.median(axis=0)
        print(tabulate(result_df, headers="keys", tablefmt="psql"))
        result_df = result_df.drop(columns=['Within 20% C', 'Within 20% P', 'Within 20% D', 'MSE C', 'MSE P', 'MSE D',
                                            'MSE P/C', 'Run']).round(2)
        tex_file_more = open('./out/csvs/warfarin_results_more.tex', 'w')
        result_df.to_latex(buf=tex_file_more)
        tex_file_more.close()
        print(tabulate(result_df, headers="keys", tablefmt="psql"))
        result_df = result_df.drop(columns=['MAE D', 'MAE D/C', 'Dec W20 D/C']).round(2)
        print(tabulate(result_df, headers="keys", tablefmt="psql"))
        tex_file = open('./out/csvs/warfarin_results.tex', 'w')
        result_df.to_latex(buf=tex_file)
        tex_file.close()
        exit(0)

    # Else, redo the computation.
    # Step 1: Load the warfarin data set
    data = load_warfarin()
    frac_true = 0.75

    # Step 2: Split of a substitute data set
    dat_true = data[:int(frac_true * data.shape[0])]
    dat_substitue = data[:int(frac_true * data.shape[0])]

    # Step 3: Apply Preprocessing
    scaler_true = MinMaxScaler(feature_range=(settings.MinMaxSettings.minmax_min, settings.MinMaxSettings.minmax_max))
    scaler_sub = MinMaxScaler(feature_range=(settings.MinMaxSettings.minmax_min, settings.MinMaxSettings.minmax_max))

    dat_true = scaler_true.fit_transform(dat_true)
    dat_substitue = scaler_sub.fit_transform(dat_substitue)

    # Step 4: Train/Test split
    test_size = 0.2
    x_train, x_test, y_train, y_test = train_test_split(dat_true[:, :-1], dat_true[:, -1],
                                                        test_size=test_size, random_state=42)
    x_substitue, y_substitue = dat_substitue[:, :-1], dat_substitue[:, -1]

    test_full = np.concatenate((x_test, y_test.reshape((-1, 1))), axis=1)
    test_target = scaler_true.inverse_transform(test_full)[:, -1]

    eps = 0.02
    n_poisoned = int(eps * x_train.shape[0])

    # Step 5: Poisong the data using Flip Attack
    x_poisoned, y_poisoned = _poison_flip(x_substitue, y_substitue, n_poisoned)
    data_poisoned = np.concatenate((x_poisoned, y_poisoned.reshape((-1, 1))), axis=1)
    data_poisoned = scaler_true.transform(scaler_sub.inverse_transform(data_poisoned))
    x_poisoned = data_poisoned[:, :-1]
    y_poisoned = data_poisoned[:, -1]

    # concat the poison data to the training data
    x_both, y_both, y_is_poisoned = shuffle_x_y(
        np.concatenate((x_train, x_poisoned)),
        np.concatenate((y_train, y_poisoned)),
        np.concatenate((np.zeros(y_train.shape), np.ones(y_poisoned.shape))))

    # Delete intermediate results
    del x_poisoned, y_poisoned, x_substitue, y_substitue, dat_true, dat_substitue

    # Create a result data frame
    result_df = pd.DataFrame(columns=["Regressor", "Run", "MAE C", "MAE P", "MAE D", "MSE C", "MSE P", "MSE D",
                                      "Within 20% C", "Within 20% P", "Within 20% D"])
    result_df.index.name = 'Model'
    outside_border = 0.2

    # For each regressor, evaluate how poisoning affects the warfarin data set
    for regressor in run_settings['regressor']:
        regressor_name = regressor[0]().__class__.__name__

        # Average over five runs
        runs = 5
        for run in range(runs):
            print(f"Evaluating {regressor_name} on Warfarin data set, run {run}/{runs-1}")
            result_dict = {'Regressor': regressor_name, 'Run': run}

            # ============= Create Clean Results ================ #
            grid_clean = GridSearchCV(estimator=regressor[0](), param_grid=regressor[1], refit=True,
                                      cv=3, scoring='neg_mean_squared_error', iid=False)
            grid_clean.fit(x_train, y_train)
            pred_clean = grid_clean.predict(x_test)
            pred_clean_full = np.concatenate((x_test, pred_clean.reshape((-1, 1))), axis=1)

            pred_clean_target = scaler_true.inverse_transform(pred_clean_full)[:, -1]

            clean_in_border_mask = np.logical_and((1 - outside_border) * test_target <= pred_clean_target,
                                                  pred_clean_target <= (1 + outside_border) * test_target)
            result_dict["Within 20% C"] = np.sum(clean_in_border_mask) / clean_in_border_mask.shape[0]

            diff_clean = np.abs(pred_clean_target - test_target)

            mse_clean = np.mean(np.square(diff_clean))

            # ========== Create Poisoning Results ========= #
            grid_poisoned = GridSearchCV(estimator=regressor[0](), param_grid=regressor[1], refit=True,
                                         cv=3, scoring='neg_mean_squared_error', iid=False)
            grid_poisoned.fit(x_both, y_both)
            pred_poisoned = grid_poisoned.predict(x_test)
            pred_poisoned_full = np.concatenate((x_test, pred_poisoned.reshape((-1, 1))), axis=1)

            pred_poisoned_target = scaler_true.inverse_transform(pred_poisoned_full)[:, -1]

            poisoned_in_border_mask = np.logical_and((1 - outside_border) * test_target <= pred_poisoned_target,
                                                     pred_poisoned_target <= (1 + outside_border) * test_target)
            result_dict["Within 20% P"] = np.sum(poisoned_in_border_mask) / poisoned_in_border_mask.shape[0]

            diff_poisoned = np.abs(pred_poisoned_target - test_target)

            mse_poisoned = np.mean(np.square(diff_poisoned))

            # ========== Create Defense Results ========= #
            grid = lambda: GridSearchCV(estimator=regressor[0](), param_grid=regressor[1], refit=True,
                                        cv=3, scoring='neg_mean_squared_error', iid=False)

            x_defended, y_defended = defense_both_trim_and_itrim(x_both, y_both, grid, eps_hat=0.30)["defense_itrim"]

            grid_defended = GridSearchCV(estimator=regressor[0](), param_grid=regressor[1], refit=True,
                                         cv=3, scoring='neg_mean_squared_error', iid=False)
            grid_defended.fit(x_defended, y_defended)
            pred_defended = grid_defended.predict(x_test)
            pred_defended_full = np.concatenate((x_test, pred_defended.reshape((-1, 1))), axis=1)

            pred_defended_target = scaler_true.inverse_transform(pred_defended_full)[:, -1]

            defended_in_border_mask = np.logical_and((1 - outside_border) * test_target <= pred_defended_target,
                                                     pred_defended_target <= (1 + outside_border) * test_target)
            result_dict["Within 20% D"] = np.sum(defended_in_border_mask) / defended_in_border_mask.shape[0]

            diff_defended = np.abs(pred_defended_target - test_target)

            mse_defended = np.mean(np.square(diff_defended))

            # # ============== Plot everything ============== #
            #
            # plt.hlines(1, 0, 100)
            #
            # diff_quot = diff_poisoned / diff_clean
            # plt.plot(np.linspace(0, 100, diff_quot.shape[0]), np.sort(diff_quot),
            #      label="quotient")  # , np.linspace(0, 100, diff_clean.shape[0]), label="clean")
            #
            # plt.legend()
            # plt.title(label=regressor_name)
            #
            # # plt.scatter(test_df, diff_df)
            # plt.ylim(0, 20)
            # # plt.xlim(-75, 50)
            # plt.show()
            #
            # plt.scatter(diff_clean, diff_poisoned)
            # plt.plot(np.linspace(0, 50, 100), np.linspace(0, 50, 100))
            # plt.xlim(0, 50)
            # plt.ylim(0, 50)
            # plt.title(label=regressor_name)
            # plt.show()
            #
            # plt.scatter(np.linspace(0, 100, diff_clean.shape[0]), diff_clean, label="clean")
            # plt.scatter(np.linspace(0, 100, diff_clean.shape[0]), diff_poisoned, label="poisoned")
            # plt.legend()
            # plt.title(label=regressor_name)
            # plt.show()
            #
            # # mask = diff_poisoned > diff_clean

            # ============= Store Remaining Stuff in DF ============ #
            # result_df.loc[regressor_name, f"Within 20% P/C"] = result_df.loc[regressor_name, f"Within 20% P"]/result_df.loc[regressor_name, f"Within 20% C"]
            # result_df.loc[regressor_name, f"Within 20% D/C"] = result_df.loc[regressor_name, f"Within 20% D"]/result_df.loc[regressor_name, f"Within 20% C"]

            result_dict["MSE C"] = mse_clean
            result_dict["MSE P"] = mse_poisoned
            result_dict["MSE D"] = mse_defended
            result_dict["MAE C"] = np.mean(diff_clean)
            result_dict["MAE P"] = np.mean(diff_poisoned)
            result_dict["MAE D"] = np.mean(diff_defended)

            # result_df.loc[regressor_name, "MSE C"] = mse_clean
            # result_df.loc[regressor_name, "MSE P"] = mse_poisoned
            # result_df.loc[regressor_name, "MSE D"] = mse_defended
            # result_df.loc[regressor_name, "MSE P/C"] = mse_poisoned / mse_clean
            # result_df.loc[regressor_name, "MSE D/C"] = mse_defended / mse_clean
            # result_df.loc[regressor_name, "MAE C"] = np.mean(diff_clean)
            # result_df.loc[regressor_name, "MAE P"] = np.mean(diff_poisoned)
            # result_df.loc[regressor_name, "MAE D"] = np.mean(diff_defended)
            # result_df.loc[regressor_name, "MAE P/C"] = result_df.loc[regressor_name, "MAE P"]/result_df.loc[regressor_name, "MAE C"]
            # result_df.loc[regressor_name, "Test size"] = diff_poisoned.shape[0]
            # result_df.loc[regressor_name, "Greater Error Poisoned"] = np.sum(mask)
            # result_df.loc[regressor_name, "Average GEP Inc"] = np.mean(diff_poisoned[mask] - diff_clean[mask])
            # result_df.loc[regressor_name, "Average GEP"] = np.mean(diff_poisoned[mask])
            # result_df.loc[regressor_name, "Average GEP Clean"] = np.mean(diff_clean[mask])
            # result_df.loc[regressor_name, "Lower Error Poisoned"] = np.sum(~mask)
            # result_df.loc[regressor_name, "Average LEP Inc"] = np.mean(diff_poisoned[~mask] - diff_clean[~mask])
            # result_df.loc[regressor_name, "Average LEP"] = np.mean(diff_poisoned[~mask])
            # result_df.loc[regressor_name, "Average LEP Clean"] = np.mean(diff_clean[~mask])
            result_df = result_df.append(result_dict, ignore_index=True)

    # Pretty Print results
    print(tabulate(result_df, headers="keys", tablefmt="psql"))

    # Save results
    result_df.to_csv('./out/csvs/warfarin_results.csv')


if __name__ == '__main__':
    main()
