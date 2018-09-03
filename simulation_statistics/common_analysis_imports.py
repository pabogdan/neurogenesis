from __future__ import division, print_function
from collections import Iterable
import traceback
import numpy as np
import matplotlib.pyplot as plt
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import animation, rc, colors
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import scipy
import scipy.stats as stats
from glob import glob
from pprint import pprint as pp
from analysis_functions_definitions import *
from argparser import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
from brian2.units import *
import os
import ntpath

import matplotlib as mlib

from spinn_utilities.progress_bar import ProgressBar

mlib.rcParams.update({'font.size': 24})
mlib.rcParams.update({'errorbar.capsize': 5})
mlib.rcParams.update({'figure.autolayout': True})
