import traceback

from gari_function_definitions import *
from function_definitions import *
from argparser import *

import numpy as np
import pylab as plt


data_file_name = args.initial_connectivity_file
if ".npz" not in data_file_name:
    data_file_name += ".npz"

data = np.load(data_file_name)

print(data['exception'])
