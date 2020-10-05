import time

import copy
import itertools

import math
import numpy
import os
from tabulate import tabulate

import logging
import dill
import pathos as pathos
from tqdm import tqdm

from src.settings import do_not_load_from_disk

logging.basicConfig(
    # filename='HISTORYlistener.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


class ExpHelper:
    def __init__(self, name, settings: dict):
        self.name = name
        self.settings = settings

    def parallelize(self, function, on_settings_keys: list, cpus):

        def logfun(*args):
            numpy.random.seed(int(os.getpid() + time.time()))
            return function(*args)

        for s in on_settings_keys:
            assert s in self.settings.keys(), f"{s} not in settings (keys: {' '.join(self.settings.keys())})"

        # set up tasks
        p = pathos.multiprocessing.ProcessPool(math.ceil(cpus))
        total_tasks = itertools.product(*[self.settings[i] for i in on_settings_keys])
        num_total_tasks = len(list(copy.deepcopy(total_tasks)))  # iterator can be iterated over only once - thus copy

        # get res
        # Exponential moving average smoothing factor for speed estimates
        # Ranges from 0 (average speed) to 1 (current/instantaneous speed) [default: 0.3].
        full_res = list(tqdm(p.uimap(logfun, total_tasks), total=num_total_tasks, smoothing=0))
        logging.info('Done.\n')
        return full_res

    @staticmethod
    def do_or_load_intermediates(base_path, list_of_properties, func, fname):

        def custom_path(max_len=None):
            max_len = len(list_of_properties) if max_len is None else max_len
            return [str(i) for i in list_of_properties][:max_len]
        path_of_file = os.path.join(base_path, *custom_path(), fname)
        if os.path.exists(path_of_file) and not do_not_load_from_disk:
            logging.debug(f'Data found in {path_of_file}. Loading old data.')
            with open(path_of_file, 'rb') as df:
                return dill.load(df)

        if do_not_load_from_disk:
            logging.debug("Forced new data generation / not loading from disk.")
        else:
            logging.debug(f'No data found in {path_of_file}. Creating new data.')
        # if not exists:
        res = func()
        for i in range(len(list_of_properties) + 1):
            p = os.path.join(base_path, *custom_path(i))
            os.makedirs(p, exist_ok=True)
        with open(path_of_file, 'wb') as df:
            dill.dump(res, df)
        return res


    @staticmethod
    def do_or_load_main_runs(base_path, list_of_properties, func, args: dict):

        def custom_path(max_len=None):
            max_len = len(list_of_properties) if max_len is None else max_len
            return [f"{j}={args[j]}" for j in list_of_properties[:max_len]]

        fname = 'store.dill'
        path_of_file = os.path.join(base_path, *custom_path(), fname)
        if os.path.exists(path_of_file):
            with open(path_of_file, 'rb') as df:
                return dill.load(df)

        # if not exists:
        res = func(**args)
        for i in range(len(list_of_properties) + 1):
            p = os.path.join(base_path, *custom_path(i))
            os.makedirs(p, exist_ok=True)
        with open(path_of_file, 'wb') as df:
            dill.dump(res, df)

    @staticmethod
    def pretty_print(df):
        print(tabulate(df, headers="keys", tablefmt="psql"))
