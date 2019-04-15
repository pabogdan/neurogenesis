import argparse

# Current defaults for [App: Motion detection]
# as of 08.04.2019

# Default values
DEFAULT_CHUNK_SIZE = 200  # ms
DEFAULT_W_MIN = 0
DEFAULT_W_MAX = .2
DEFAULT_CLASSES = [0, 90]
DEFAULT_CLASS_DEV = 2  # +- 2 * 5 degreess
DEFAULT_NO_ITERATIONS = DEFAULT_CHUNK_SIZE * (100 * len(DEFAULT_CLASSES))
DEFAULT_T_RECORD = 200000
DEFAULT_P_CONNECT = .2  # 20%
DEFAULT_RUNS = 1
DEFAULT_TESTING_NO_ITERATIONS_PER_CLASS = 200

DEFAULT_B = 1.2
DEFAULT_TAU_MINUS = 60  # ms
DEFAULT_TAU_PLUS = 20  # ms
DEFAULT_TAU_REFRAC = 2.0  # ms
DEFAULT_A_PLUS = 0.1
DEFAULT_A_MINUS = (DEFAULT_A_PLUS * DEFAULT_TAU_PLUS * DEFAULT_B) \
                  / float(DEFAULT_TAU_MINUS)

# REWIRING STUFF
DEFAULT_S_MAX = 128
DEFAULT_SIGMA_STIM = 2
# these shouldn't really be used here due to layer shape mismatch
# rewiring run without distance dependence
DEFAULT_SIGMA_FORM_LAT = 1
DEFAULT_SIGMA_FORM_FF = 1

# Default probabilities

DEFAULT_P_FORM_LATERAL = 1
DEFAULT_P_FORM_FORWARD = 0.16
DEFAULT_P_ELIM_DEP = 0.0245
DEFAULT_P_ELIM_POT = 1.36 * (10 ** -5)  # 10 times lower than usual

DEFAULT_F_REW = 10 ** 4

# Default flags
DEFAULT_REWIRING_FLAG = False
DEFAULT_MNIST_FLAG = False
DEFAULT_LATERAL_INHIBITION = False
# Organise things into folders

DEFAULT_SIM_FOLDER = "./"

# Argument parser
parser = argparse.ArgumentParser(
    description='Readout module for topographic maps generated by SpiNNaker '
                'in order to perform classification. '
                '[WARNING] This module assumes a specific architecture (1 '
                'exc and 1 inh target layers each with 3 input projections)')
flags = parser.add_argument_group('flag arguments')
mutex_required_args = parser.add_mutually_exclusive_group(
    required=True)
mutex_required_args.add_argument('--min_supervised',
                                 help="label is communicated to readout "
                                      "neurons "
                                      "connection weights set to w_min "
                                      "-- [default {}]".format(DEFAULT_W_MIN),
                                 action="store_true")

mutex_required_args.add_argument('--max_supervised',
                                 help="label is communicated to readout "
                                      "neurons. initial "
                                      "connection weights set to w_max"
                                      "-- [default {}]".format(DEFAULT_W_MAX),
                                 action="store_true")

mutex_required_args.add_argument('--unsupervised',
                                 help="label is NOT communicated to "
                                      "readout neurons. "
                                      "initial connection "
                                      "weights set to w_max",
                                 action="store_true")

parser.add_argument('path', help='path of input .npz archive defining '
                                 'connectivity', nargs='*')

parser.add_argument('-o', '--output', type=str,
                    help="name of the numpy archive storing simulation results",
                    dest='filename')

parser.add_argument('--suffix', type=str,
                    help="add a recognisable suffix to the filename",
                    dest='suffix')

parser.add_argument('--b', type=float,
                    default=DEFAULT_B,
                    help='ration between area under depression curve and '
                         'area under potentiation curve'
                         ' -- [default {}]'.format(DEFAULT_B))

parser.add_argument('--tau_minus', type=float,
                    default=DEFAULT_TAU_MINUS,
                    help='time constant for depression'
                         ' -- [default {}]'.format(DEFAULT_TAU_MINUS))

parser.add_argument('--tau_plus', type=float,
                    default=DEFAULT_TAU_PLUS,
                    help='time constant for potentiation'
                         ' -- [default {}]'.format(DEFAULT_TAU_PLUS))

parser.add_argument('--a_plus', type=float,
                    default=DEFAULT_A_PLUS,
                    help='potentiation learning rate'
                         ' -- [default {}]'.format(DEFAULT_A_PLUS))

parser.add_argument('--w_max', type=float,
                    default=DEFAULT_W_MAX,
                    help='maximum weight'
                         ' -- [default {}]'.format(DEFAULT_W_MAX))

parser.add_argument('--w_min', type=float,
                    default=DEFAULT_W_MIN,
                    help='minimum weight'
                         ' -- [default {}]'.format(DEFAULT_W_MIN))

parser.add_argument('--p_connect', type=float,
                    default=DEFAULT_P_CONNECT,
                    help='readout neurons have, on average, p_connect '
                         'connectivity (0<=p_connect<=1)'
                         ' -- [default {}]'.format(DEFAULT_P_CONNECT))

parser.add_argument('--plot', help="display plots",
                    action="store_true")

flags.add_argument('--snapshots',
                   help="store snapshot information "
                        "(connectivity + weights + delays)",
                   action="store_true")

parser.add_argument('--label_time_offset',
                    help="time offset of label presentation within a time bin "
                         "-- [default {}]".format([DEFAULT_CHUNK_SIZE - 1]),
                    type=int, nargs="+", default=[DEFAULT_CHUNK_SIZE - 1],
                    dest='label_time_offset'
                    )

flags.add_argument('--rewiring',
                   help="flag to allow readout neurons to rewire "
                        "-- [default {}]".format(DEFAULT_REWIRING_FLAG),
                   action="store_true")

flags.add_argument('--wta_readout',
                   help="flag to enable WTA / strong lateral connectivity in "
                        "readout layer "
                        "-- [default {}]".format(DEFAULT_LATERAL_INHIBITION),
                   action="store_true")

flags.add_argument('--mnist',
                   help="flag used to enable inputting MNIST digits "
                        "(as opposed to moving bars) "
                        "-- [default {}]".format(DEFAULT_MNIST_FLAG),
                   action="store_true")

parser.add_argument('--no_iterations', type=int,
                    default=DEFAULT_NO_ITERATIONS, dest='no_iterations',
                    help='total number of iterations (or time steps) for '
                         'the simulation (technically, ms)'
                         ' -- [default {}]'.format(DEFAULT_NO_ITERATIONS))

parser.add_argument('--no_iterations_per_class', type=int,
                    default=DEFAULT_TESTING_NO_ITERATIONS_PER_CLASS,
                    dest='testing_no_iterations_per_class',
                    help='total number of iterations (pattern presentations) for '
                         'each class'
                         ' -- [default {}]'.format(DEFAULT_TESTING_NO_ITERATIONS_PER_CLASS))

parser.add_argument('--classes',
                    help="[App: Motion detection] Network will be trained "
                         "using a random succession of these classes"
                         " -- [default {}]".format(DEFAULT_CLASSES),
                    type=int, nargs="+", default=DEFAULT_CLASSES,
                    dest='classes'
                    )

parser.add_argument('--t_record', type=int,
                    default=DEFAULT_T_RECORD, dest='t_record',
                    help='time between retrieval of recordings (ms)'
                         ' -- [default {}]'.format(DEFAULT_T_RECORD))

parser.add_argument('--testing_t_record', type=int,
                    default=DEFAULT_T_RECORD, dest='testing_t_record',
                    help='time between retrieval of recordings (ms)'
                         ' -- [default {}]'.format(DEFAULT_T_RECORD))

parser.add_argument('--runs', type=int, nargs="+",
                    default=DEFAULT_RUNS,
                    help='how many times to run the training + testing '
                         'experiments -- [default {}]'.format(DEFAULT_RUNS))

parser.add_argument('--sim_dir', type=str,
                    default=DEFAULT_SIM_FOLDER,
                    help='folder in which to save simulation results'
                         ' -- [default {}]'.format(DEFAULT_SIM_FOLDER))

# Rewiring arguments
parser.add_argument('--s_max', type=int,
                    default=DEFAULT_S_MAX, dest='s_max',
                    help='maximum synaptic capacity'
                         ' -- [default {}]'.format(DEFAULT_S_MAX))

parser.add_argument('--sigma_form_ff', type=float,
                    default=DEFAULT_SIGMA_FORM_FF, dest='sigma_form_ff',
                    help='spread of feedforward formations'
                         ' -- [default {}]'.format(DEFAULT_SIGMA_FORM_FF))

parser.add_argument('--sigma_form_lat', type=float,
                    default=DEFAULT_SIGMA_FORM_LAT, dest='sigma_form_lat',
                    help='spread of lateral formations'
                         ' -- [default {}]'.format(DEFAULT_SIGMA_FORM_LAT))

parser.add_argument('--p_elim_pot', type=float,
                    default=DEFAULT_P_ELIM_POT, dest='p_elim_pot',
                    help='probability of eliminating potentiated synapses'
                         ' -- [default {}]'.format(DEFAULT_P_ELIM_POT))

parser.add_argument('--p_form_forward', type=float,
                    default=DEFAULT_P_FORM_FORWARD, dest='p_form_forward',
                    help='probability of forming feedforward synapses'
                         ' -- [default {}]'.format(DEFAULT_P_FORM_FORWARD))

parser.add_argument('--p_form_lateral', type=float,
                    default=DEFAULT_P_FORM_LATERAL, dest='p_form_lateral',
                    help='probability of forming lateral synapses'
                         ' -- [default {}]'.format(DEFAULT_P_FORM_LATERAL))

parser.add_argument('--p_elim_dep', type=float,
                    default=DEFAULT_P_ELIM_DEP, dest='p_elim_dep',
                    help='probability of eliminating depressed synapses'
                         ' -- [default {}]'.format(DEFAULT_P_ELIM_DEP))

parser.add_argument('--f_rew', type=float,
                    default=DEFAULT_F_REW, dest='f_rew',
                    help='frequency of rewire attempts (Hz)')

parser.add_argument('--test_jitter',
                    help="[App: Motion detection] jitter the input by +-1 ms",
                    action="store_true")


parser.add_argument('--test_class_with_deviation',
                    help="[App: Motion detection] jitter the input by +-1 ms",
                    action="store_true")

parser.add_argument('--test_class_dev',
                    help="[App: Motion detection] Network will be tested "
                         "using a random succession of these classes"
                         " -- [default {}]".format(DEFAULT_CLASS_DEV),
                    type=int, nargs=1, default=DEFAULT_CLASS_DEV
                    )


parser.add_argument('--no-cache',
                    help="force simulation re-run without "
                         "using cached "
                         "information"
                         " -- [default {}]".format(False),
                    action="store_true", dest="no_cache")

args = parser.parse_args()
