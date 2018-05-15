import argparse

# Constants
CASE_CORR_AND_REW = 1
CASE_CORR_NO_REW = 2
CASE_REW_NO_CORR = 3

SSP = 1
SSA = 2

DEFAULT_TAU_REFRAC = 5.0
DEFAULT_F_PEAK = 152.8
DEFAULT_NO_INTERATIONS = 300000
DEFAULT_T_RECORD = DEFAULT_NO_INTERATIONS
DEFAULT_T_STIM = 20
DEFAULT_S_MAX = 32
DEFAULT_F_MEAN = 20
DEFAULT_F_REW = 10**4
DEFAULT_LAT_INH = False

DEFAULT_N = 16

DEFAULT_DELAY = 1

DEFAULT_SPIKE_SOURCE = SSP
DEFAULT_B = 1.2
DEFAULT_T_MINUS = 64

DEFAULT_SIGMA_STIM = 2
DEFAULT_SIGMA_FORM_LAT = 1
DEFAULT_SIGMA_FORM_FF = 2.5

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


parser = argparse.ArgumentParser(
    description='Test for topographic map formation using STDP and '
                'synaptic rewiring on SpiNNaker.',
    formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-c", '--case', type=int,
                    choices=[CASE_CORR_AND_REW, CASE_CORR_NO_REW,
                             CASE_REW_NO_CORR],
                    default=CASE_CORR_AND_REW, dest='case',
                    help='an integer controlling the experimental setup')

parser.add_argument('--testing', type=str, dest='testing',
                    help='put the network in testing mode. provide .npz'
                         'archive with connectivity information to commence'
                         'testing')

parser.add_argument("-l", '--lesion', type=int,
                    choices=[NO_LESION, RANDOM_CONNECTIVITY_LESION,
                             ONE_TO_ONE_LESION],
                    default=NO_LESION, dest='lesion',
                    help='what type of lesion to do (none, random, 1:1, '
                         'all:all)')

parser.add_argument('--p_elim_dep', type=float,
                    default=DEFAULT_P_ELIM_DEP, dest='p_elim_dep',
                    help='probability of eliminating depressed synapses')

parser.add_argument('--p_elim_pot', type=float,
                    default=DEFAULT_P_ELIM_POT, dest='p_elim_pot',
                    help='probability of eliminating potentiated synapses')

parser.add_argument('--p_form_forward', type=float,
                    default=DEFAULT_P_FORM_FORWARD, dest='p_form_forward',
                    help='probability of forming feedforward synapses')

parser.add_argument('--p_form_lateral', type=float,
                    default=DEFAULT_P_FORM_LATERAL, dest='p_form_lateral',
                    help='probability of forming lateral synapses')

parser.add_argument('--tau_refract', type=float,
                    default=DEFAULT_TAU_REFRAC, dest='tau_refrac',
                    help='refractory time constant (ms)')

parser.add_argument('--sigma_stim', type=float,
                    default=DEFAULT_SIGMA_STIM, dest='sigma_stim',
                    help='stimulus spread')

parser.add_argument('--sigma_form_lat', type=float,
                    default=DEFAULT_SIGMA_FORM_LAT, dest='sigma_form_lat',
                    help='spread of lateral formations')

parser.add_argument('--sigma_form_ff', type=float,
                    default=DEFAULT_SIGMA_FORM_FF, dest='sigma_form_ff',
                    help='spread of feedforward formations')

parser.add_argument('-n','--n', type=int,
                    default=DEFAULT_N, dest='n',
                    help='size of one edge of the layer (default 16)')

parser.add_argument('--t_record', type=int,
                    default=DEFAULT_T_RECORD, dest='t_record',
                    help='time between retrieval of recordings (ms)')

parser.add_argument('--lat_inh', action="store_true",
                    dest='lateral_inhibition',
                    help='enable lateral inhibition')

parser.add_argument('--t_stim', type=int,
                    default=DEFAULT_T_STIM, dest='t_stim',
                    help='time between stimulus location change (ms)')

parser.add_argument('--f_peak', type=float,
                    default=DEFAULT_F_PEAK, dest='f_peak',
                    help='peak input spike rate (Hz)')

parser.add_argument('--f_rew', type=float,
                    default=DEFAULT_F_REW, dest='f_rew',
                    help='frequency of rewire attempts (Hz)')

parser.add_argument('--f_mean', type=float,
                    default=DEFAULT_F_MEAN, dest='f_mean',
                    help='input spike rate (Hz) used with case 3')

parser.add_argument('--s_max', type=int,
                    default=DEFAULT_S_MAX, dest='s_max',
                    help='maximum synaptic capacity')

parser.add_argument('--b', type=float,
                    default=DEFAULT_B, dest='b',
                    help='ration between area under depression curve and area under potentiation curve')

parser.add_argument('--t_minus', type=int,
                    default=DEFAULT_T_MINUS, dest='t_minus',
                    help='time constant for depression')

parser.add_argument('--delay', type=int,
                    default=DEFAULT_DELAY, dest='delay',
                    help='delay (in ms) applied to spikes in the network')

parser.add_argument('--no_iterations', type=int,
                    default=DEFAULT_NO_INTERATIONS, dest='no_iterations',
                    help='total number of iterations (or time steps) for the simulation (technically, ms)')

parser.add_argument('--plot', help="display plots",
                    action="store_true")

parser.add_argument('--record_source',
                    help="record spikes generated by the source layer",
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
                    help='what type of input shape to use (gaussian, pointy, '
                         'scaled pointy, square)')

parser.add_argument('--random_input',
                    help="instead of input a digit"
                         " input noise at prescribed f_mean",
                    action="store_true")

parser.add_argument('--no_lateral_conn',
                    help="run experiment without lateral "
                         "connectivity",
                    action="store_true")

parser.add_argument('--lat_lat_conn',
                    help="run experiment with latero-lateral "
                         "connectivity", default=DEFAULT_LAT_LAT_CONN,
                    action="store_true")

args = parser.parse_args()
