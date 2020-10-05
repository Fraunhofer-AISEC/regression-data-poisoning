from src import settings
from src.settings import datasets

datasets = [x for x in datasets if x[0]().shape[0] > settings.nmin]

import pandas as pd
from tabulate import tabulate
import numpy as np

