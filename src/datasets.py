import numpy
from io import StringIO
import os
import tarfile
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd
from sklearn.datasets import make_regression
from sklearn.preprocessing import OneHotEncoder


def helper_one_hot(df, target_col):
    # remove rows where target col is nan
    df.dropna(subset=[target_col], inplace=True, axis='rows')
    df = df.reset_index(drop=True)

    string_columns = [c for c in df.columns if df.loc[:, c].dtype == object and c != target_col]  # strings
    numerical_columns = [c for c in df.columns if df.loc[:, c].dtype != object and c != target_col]  # numericals
    target_column = [target_col]  # target

    # remove string rows where too many diff values
    t = df[string_columns].nunique()
    string_columns = [c for c in string_columns if t[c] < 15]

    # handle NaN values for strings and numericals seperately
    df_strings = df[string_columns].fillna('na')
    df_numerical = df[numerical_columns].fillna(0)
    y = df[target_column]
    assert not numpy.any(y.isna())
    assert not numpy.any(y.isnull())

    # one_hot encoding and dataframe construction
    df_strings_onehot = OneHotEncoder().fit_transform(df_strings.values).toarray()
    del df_strings
    x = numpy.concatenate((df_strings_onehot, df_numerical), axis=1)
    df = pd.DataFrame(x)
    assert not numpy.any(df.isna())
    assert not numpy.any(df.isnull())
    df.loc[:, 'target'] = y
    return df


def load_loan():
    df = pd.read_csv('./res/poisoning_datasets/loan/load_0.csv').drop(columns=['Unnamed: 0', 'id', 'member_id'])
    return helper_one_hot(df, target_col='int_rate')


def load_armes_housing():
    train = pd.read_csv('./res/poisoning_datasets/armes_housing/train.csv').drop(columns=['Id'])
    # comparison paper uses only train set
    return helper_one_hot(train, target_col='SalePrice')


def load_warfarin():
    train = pd.read_csv('./res/poisoning_datasets/warfarin/warfarin.csv', sep="\t")
    train.drop(inplace=True, columns=['PharmGKB Subject ID', 'PharmGKB Sample ID', 'Project Site',
                                      'Comments regarding Project Site Dataset',
                                      'INR on Reported Therapeutic Dose of Warfarin'])
    train = train.reset_index(drop=True)
    return helper_one_hot(train, target_col='Therapeutic Dose of Warfarin')


def load_keel_single(src_folder, directory, filename):
    def f():
        with open(os.path.join(src_folder, directory, filename)) as file_raw:
            try:
                data = [x for x in file_raw.readlines() if not x.startswith("@") and not x.startswith(".DS_Store")]
                io = StringIO("".join(data))
                df = pd.read_csv(io, sep=",", header=None)
                target_col = df.columns[-1]
                df["target"] = df[target_col]
                df.drop(target_col, inplace=True, axis=1)
                # print(df.head())
                return df
            except UnicodeDecodeError:
                print("failed reading " + os.path.join(src_folder, directory, filename))
                return pd.DataFrame()

    return f


def load_from_keel_repo(src_folder="./res/keel"):
    pds = []
    for directory in os.listdir(src_folder):
        for filename in [x for x in os.listdir(os.path.join(src_folder, directory)) if not "-" in x]:
            if "" in filename:
                dd = load_keel_single(src_folder, directory, filename)
                pds.append((dd, filename))
    print('Remove from keel where "-" in name.')
    return pds


def make_unbal_uniform(a, b, n, perc=0.8, side_pos=True, use_linspace=False):
    f = numpy.linspace if use_linspace else numpy.random.uniform
    y = f(a, b, int((1 - perc) * n))
    tposa = a if side_pos else (b - a) / 2 + a
    y_add = numpy.asarray([tposa, ] * int(n * perc))
    x = pd.DataFrame()
    x.loc[:, "target"] = numpy.concatenate((y, y_add))
    return x


def make_regression_pd(**kwargs):
    x, y = make_regression(**kwargs)
    x = pd.DataFrame(x)
    x.loc[:, "target"] = y
    return x


def make_with_fun_2d(num, low=-2, high=2, fun=lambda x: x, noise_strength=1e-1):
    # create feats + target
    snum = int(numpy.sqrt(num))
    x = numpy.linspace(low, high, snum) + numpy.random.normal(0, 1e-3, snum)
    y = numpy.linspace(low, high, snum) + numpy.random.normal(0, 1e-3, snum)

    vx, vy = numpy.meshgrid(x, y)
    stacked = numpy.stack((vx, vy), 2)
    xy = stacked.reshape(-1, stacked.shape[-1])
    df = pd.DataFrame(xy)
    df.loc[:, 'target'] = df.apply(func=fun, axis=1)
    df.loc[:, 'target'] = df.loc[:, 'target'] + numpy.random.normal(0, noise_strength, df.shape[0]) * df.loc[:,
                                                                                                      'target']

    return df


def make_normal_dummy(mu, sigma, num, eps_frac=0.1):
    x_true = numpy.random.normal(mu, sigma, num)
    x_noise = x_true + numpy.random.normal(0, sigma * eps_frac, num)
    x = pd.DataFrame(x_noise)
    x.loc[:, "target"] = x_true
    return x


def load_github_regression(name, target_column):
    maybe_download_dataset(file_name=f"./res/github_regression/{name}",
                           url=f"https://raw.githubusercontent.com/paobranco/Imbalanced-Regression-DataSets/master/CSV_data/",
                           url_file_name=name)
    df = pd.read_csv(f"./res/github_regression/{name}")
    assert target_column in df.columns, f"{target_column} not in dataframe's columns {df.columns}"

    df = df.dropna(axis=0)
    target = df[target_column]
    df = df.drop(target_column, axis=1)
    # to one hot
    df = pd.get_dummies(df)
    df.loc[:, "target"] = target

    return df.sample(frac=1).reset_index(drop=True)


def maybe_download_dataset(file_name: str, url: str, url_file_name: str, extract_file: str = None) -> None:
    """
    Checks if a dataset of name file_name exists and tries to download it from url otherwise.
    :param file_name:       str, Name of the file of the dataset.
    :param url:             str, URL from which the dataset will be downloaded if it does not exist.
    :param url_file_name:   str, Name of the file in the URL.
    :param extract_file:    str, In case we download a zip file, we extract this file from the zip
    :return:                None
    """
    if not os.path.isfile(file_name):
        dataset_directory = os.path.dirname(file_name)

        # Ensure the directory exists
        if not os.path.exists(dataset_directory):
            os.makedirs(dataset_directory)

        # Download content from url
        response = urlopen(url + url_file_name)

        # Extract data, if we have a tar file
        if url_file_name.endswith("tar.gz") or url_file_name.endswith(".tgz"):
            intermediate_file_path = os.path.join(dataset_directory, url_file_name)
            with open(intermediate_file_path, 'wb') as intermediate_file:
                intermediate_file.write(response.read())
            tar = tarfile.open(intermediate_file_path, "r:gz")
            tar.extractall(dataset_directory)
            tar.close()
            os.remove(intermediate_file_path)
        # Extract da if we have a zip file
        elif url_file_name.endswith('.zip'):
            intermediate_file_path = os.path.join(dataset_directory, url_file_name)
            with open(intermediate_file_path, 'wb') as intermediate_file:
                intermediate_file.write(response.read())
            zip = ZipFile(intermediate_file_path, 'r')
            zip.extract(extract_file, path=dataset_directory)
            os.rename(os.path.join(dataset_directory, extract_file), file_name)
            os.remove(intermediate_file_path)
        # Otherwise we assume we already downloaded the required file
        else:
            with open(file_name, 'wb') as file:
                file.write(response.read())


if __name__ == "__main__":
    # load_github_regression("boston.csv", "HousValue")
    # load_github_regression("a1.csv", "a1")
    # load_github_regression("a2.csv", "a2")
    # load_github_regression("a3.csv", "a3")
    # load_github_regression("a4.csv", "a4")
    # load_github_regression("a6.csv", "a6")
    # load_github_regression("a7.csv", "a7")
    # load_github_regression("Abalone.csv", "Rings")
    load_from_keel_repo()
