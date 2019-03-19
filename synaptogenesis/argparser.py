import argparse

# Current defaults for [App: Motion detection]
# as of 5.09.2018
# Previous version:
#   topographic map formation B.O.T -> 30.07.2018

# Constants
CASE_CORR_AND_REW = 1
CASE_CORR_NO_REW = 2
CASE_REW_NO_CORR = 3

SSP = 1
SSA = 2

DEFAULT_TAU_REFRAC = 5.0
DEFAULT_F_PEAK = 152.8
DEFAULT_F_BASE = 5
DEFAULT_NO_ITERATIONS = 2400000
DEFAULT_TESTING_ITERATIONS = 1200000
DEFAULT_T_RECORD = 100000
DEFAULT_T_STIM = 20
DEFAULT_S_MAX = 96
DEFAULT_F_MEAN = 20
DEFAULT_F_REW = 10 ** 4
DEFAULT_LAT_INH = False
DEFAULT_G_MAX = 0.2

DEFAULT_N = 32

DEFAULT_DELAY = 1

DEFAULT_SPIKE_SOURCE = SSP
DEFAULT_B = 1.2
DEFAULT_T_MINUS = 64

DEFAULT_SIGMA_STIM = 2
DEFAULT_SIGMA_FORM_LAT = 5
DEFAULT_SIGMA_FORM_FF = 5

# Default probabilities

DEFAULT_P_FORM_LATERAL = 1
DEFAULT_P_FORM_FORWARD = 0.16
DEFAULT_P_ELIM_DEP = 0.0245
DEFAULT_P_ELIM_POT = 1.36 * (10 ** -4)

# Different input types
GAUSSIAN_INPUT = 1
POINTY_INPUT = 2
SCALED_POINTY_INPUT = 3
SQUARE_INPUT = 4

# Types of lesions / insult / developmental starting conditions
NO_LESION = 0
RANDOM_CONNECTIVITY_LESION = 1
ONE_TO_ONE_LESION = 2

# Enable latero-lateral interaction
DEFAULT_LAT_LAT_CONN = False

# Topology configuration
DEFAULT_TOPOLOGY = 3
DEFAULT_DELAY_TYPE = True

DEFAULT_FPS = 200
DEFAULT_CHUNK_SIZE = 200  # ms

DEFAULT_JITTER = False

DEFAULT_COPLANAR_UPPER_DELAY = 3  # ms, exclusive

parser = argparse.ArgumentParser(
    description='Test for topographic map formation using STDP and '
                'synaptic rewiring on SpiNNaker.',
    formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-c", '--case', type=int,
                    choices=[CASE_CORR_AND_REW, CASE_CORR_NO_REW,
                             CASE_REW_NO_CORR],
                    default=CASE_CORR_AND_REW, dest='case',
                    help='an integer controlling the experimental setup'
                         ' -- [default {}]'.format(CASE_CORR_AND_REW))

parser.add_argument('--testing', type=str, dest='testing',
                    help='put the network in testing mode. provide .npz'
                         'archive with connectivity information to commence'
                         'testing')

parser.add_argument("-l", '--lesion', type=int,
                    choices=[NO_LESION, RANDOM_CONNECTIVITY_LESION,
                             ONE_TO_ONE_LESION],
                    default=NO_LESION, dest='lesion',
                    help='what type of lesion to do (none, random, 1:1, '
                         'all:all)'
                         ' -- [default {}]'.format(NO_LESION))

parser.add_argument('--p_elim_dep', type=float,
                    default=DEFAULT_P_ELIM_DEP, dest='p_elim_dep',
                    help='probability of eliminating depressed synapses'
                         ' -- [default {}]'.format(DEFAULT_P_ELIM_DEP))

parser.add_argument('--chunk', type=int,
                    default=None, dest='chunk_size',
                    help='length of presentation of a pattern (in ms)'
                         ' -- [default {}]'.format(DEFAULT_CHUNK_SIZE))

parser.add_argument('--g_max', type=float,
                    default=DEFAULT_G_MAX, dest='g_max',
                    help='Maximum synaptic weight'
                         ' -- [default {}]'.format(DEFAULT_G_MAX))

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

parser.add_argument('--tau_refract', type=float,
                    default=DEFAULT_TAU_REFRAC, dest='tau_refrac',
                    help='refractory time constant (ms)'
                         ' -- [default {}]'.format(DEFAULT_TAU_REFRAC))

parser.add_argument('--sigma_stim', type=float,
                    default=DEFAULT_SIGMA_STIM, dest='sigma_stim',
                    help='[App: Topographic map formation] stimulus spread'
                         ' -- [default {}]'.format(DEFAULT_SIGMA_STIM))

parser.add_argument('--sigma_form_lat', type=float,
                    default=DEFAULT_SIGMA_FORM_LAT, dest='sigma_form_lat',
                    help='spread of lateral formations'
                         ' -- [default {}]'.format(DEFAULT_SIGMA_FORM_LAT))

parser.add_argument('--sigma_form_ff', type=float,
                    default=DEFAULT_SIGMA_FORM_FF, dest='sigma_form_ff',
                    help='spread of feedforward formations'
                         ' -- [default {}]'.format(DEFAULT_SIGMA_FORM_FF))

parser.add_argument('-n', '--n', type=int,
                    default=DEFAULT_N, dest='n',
                    help='size of one edge of the layer'
                         ' -- [default {}]'.format(DEFAULT_N))

parser.add_argument('--t_record', type=int,
                    default=DEFAULT_T_RECORD, dest='t_record',
                    help='time between retrieval of recordings (ms)'
                         ' -- [default {}]'.format(DEFAULT_T_RECORD))

parser.add_argument('--lat_inh', action="store_true",
                    dest='lateral_inhibition',
                    help='enable lateral inhibition'
                         ' -- [default {}]'.format(DEFAULT_LAT_INH))

parser.add_argument('--t_stim', type=int,
                    default=DEFAULT_T_STIM, dest='t_stim',
                    help='time between stimulus location change (ms)')

parser.add_argument('--f_peak', type=float,
                    default=DEFAULT_F_PEAK, dest='f_peak',
                    help='peak input spike rate (Hz)')

parser.add_argument('--f_base', type=float,
                    default=DEFAULT_F_BASE, dest='f_base',
                    help='base input spike rate (Hz)')

parser.add_argument('--f_rew', type=float,
                    default=DEFAULT_F_REW, dest='f_rew',
                    help='frequency of rewire attempts (Hz)')

parser.add_argument('--f_mean', type=float,
                    default=DEFAULT_F_MEAN, dest='f_mean',
                    help='input spike rate (Hz) used with case 3')

parser.add_argument('--s_max', type=int,
                    default=DEFAULT_S_MAX, dest='s_max',
                    help='maximum synaptic capacity'
                         ' -- [default {}]'.format(DEFAULT_S_MAX))

parser.add_argument('--b', type=float,
                    default=DEFAULT_B, dest='b',
                    help='ration between area under depression curve and '
                         'area under potentiation curve'
                         ' -- [default {}]'.format(DEFAULT_B))

parser.add_argument('--t_minus', type=int,
                    default=DEFAULT_T_MINUS, dest='t_minus',
                    help='time constant for depression'
                         ' -- [default {}]'.format(DEFAULT_T_MINUS))

parser.add_argument('--delay', type=int,
                    default=DEFAULT_DELAY,
                    help='delay_distribution (in ms) applied to '
                         'spikes in the network')

parser.add_argument('--no_iterations', type=int,
                    default=DEFAULT_NO_ITERATIONS, dest='no_iterations',
                    help='total number of iterations (or time steps) for '
                         'the simulation (technically, ms)'
                         ' -- [default {}]'.format(DEFAULT_NO_ITERATIONS))

parser.add_argument('--testing_iterations', type=int,
                    default=DEFAULT_TESTING_ITERATIONS,
                    dest='testing_iterations',
                    help='total number of testing iterations (or time steps) '
                         'for '
                         'the simulation (technically, ms)'
                         ' -- [default {}]'.format(DEFAULT_TESTING_ITERATIONS))

parser.add_argument('--plot', help="display plots",
                    action="store_true")

parser.add_argument('--record_source',
                    help="record spikes generated by the source layer",
                    action="store_true")


parser.add_argument('--record_exc_v',
                    help="record voltage for the excitatory layer",
                    action="store_true")

parser.add_argument('--random_partner',
                    help="select a random partner for rewiring rather than "
                         "the last neuron to have spiked",
                    action="store_true")

parser.add_argument('-o', '--output', type=str,
                    help="name of the numpy archive "
                         "storing simulation results",
                    dest='filename')

parser.add_argument('-i', '--input', type=str,
                    help="name of the numpy archive storing "
                         "initial connectivity for the simulation",
                    dest='initial_connectivity_file')

parser.add_argument('--input_type', type=int,
                    choices=[GAUSSIAN_INPUT, POINTY_INPUT,
                             SCALED_POINTY_INPUT, SQUARE_INPUT],
                    default=GAUSSIAN_INPUT, dest='input_type',
                    help='[App: Topographic map formation] what type of '
                         'input shape to use (gaussian, pointy, '
                         'scaled pointy, square)'
                         ' -- [default {}]'.format(GAUSSIAN_INPUT))

parser.add_argument('--random_input',
                    help="instead of input a digit"
                         " input noise at prescribed f_mean",
                    action="store_true")

parser.add_argument('--no_lateral_conn',
                    help="[App: MNIST]  run experiment without "
                         "lateral "
                         "connectivity",
                    action="store_true")

parser.add_argument('--constant_delay',
                    help="[App: Motion detection] constant delay_distribution",
                    action="store_true")

parser.add_argument('--lat_lat_conn',
                    help="run experiment with latero-lateral "
                         "connectivity", default=DEFAULT_LAT_LAT_CONN,
                    action="store_true")

parser.add_argument('--topology',
                    help="[App: Motion detection] Modifies the "
                         "architecture of the network (0. constant lateral "
                         "inhibition, 1. no lateral inhibition, "
                         "2. learned lateral inhibition, 3. lat inh now also "
                         "sees the input)"
                         " -- [default {}]".format(DEFAULT_TOPOLOGY),
                    type=int, default=DEFAULT_TOPOLOGY,
                    choices=[0, 1, 2, 3],
                    dest='topology')

parser.add_argument('-ta', '--training_angles',
                    help="[App: Motion detection] Network will be trained "
                         "using a random succession of these angles"
                         " -- [default {}]".format([0]),
                    type=int, nargs="+", default=[0],
                    dest='training_angles'
                    )

parser.add_argument('--all_angles',
                    help="[App: Motion detection] Network will be trained "
                         "using a random succession of all angles",
                    action="store_true",
                    dest='all_angles'
                    )

parser.add_argument('--fps',
                    help="[App: Motion detection] Bar speed across "
                         "receptive field (default is 200)",
                    type=int, default=DEFAULT_FPS,
                    dest='fps')

parser.add_argument('--record_inh',
                    help="record spikes generated by the inhibitory layer",
                    action="store_false")

parser.add_argument('--jitter',
                    help="[App: Motion detection] jitter the input by +-1 ms",
                    action="store_true")

parser.add_argument('--common_rewiring_seed',
                    help="[App: Motion detection] common rewiring shared "
                         "seed means post neurons are selected in tandem",
                    action="store_true")

# flag to force simulation re-run
parser.add_argument('--no-cache',
                    help="force simulation re-run without "
                         "using cached "
                         "information"
                         " -- [default {}]".format(False),
                    action="store_true", dest="no_cache")

parser.add_argument('--fsi',
                    help="inhibitory cells have dynamics akin to Fast Spiking Interneurons (FSI)"
                         " -- [default {}]".format(False),
                    action="store_true", dest="fsi")

# f_rew exc + inh needed to see if behaviour of populations matches observations in Nowke2018
parser.add_argument('--f_rew_exc',
                    help="[App: Motion detection] Rate of rewiring (Hz) attempts for the EXC population",
                    type=float, default=DEFAULT_F_REW)

parser.add_argument('--f_rew_inh',
                    help="[App: Motion detection] Rate of rewiring (Hz) attempts for the INH population",
                    type=float, default=DEFAULT_F_REW)


parser.add_argument('--no_off_polarity', help="disable off polarity for moving bar",
                    action="store_true")


parser.add_argument('--stationary_input', help="disable off polarity for moving bar",
                    action="store_true")


parser.add_argument('--mnist_input', help="input MNIST, not moving bars",
                    action="store_true")

parser.add_argument('--coplanar',
                    help="Target layers are now coplanar. Affects delay between them"
                    " -- [default {}]".format(None),
                    type=int, default=None)

parser.add_argument('--invert_polarities', help="flip polarities for moving bar",
                    action="store_true")

args = parser.parse_args()
