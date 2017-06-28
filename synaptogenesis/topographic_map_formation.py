"""
Test for topographic map formation using STDP and synaptic rewiring.

http://hdl.handle.net/1842/3997
"""
# Imports
import numpy as np
import pylab

from spinn_utilities.progress_bar import ProgressBar
import time
from pacman.model.constraints.placer_constraints.placer_chip_and_core_constraint import \
    PlacerChipAndCoreConstraint
import spynnaker7.pyNN as sim

import argparse

# Constants
case = None

CASE_CORR_AND_REW = 1
CASE_CORR_NO_REW = 2
CASE_REW_NO_CORR = 3

parser = argparse.ArgumentParser(
    description='Process arguments for testing topographic'
                'map formation model on SpiNNaker.')
parser.add_argument('--case', type=int, nargs='+',
                    default=CASE_CORR_AND_REW, dest='case',
                    help='an integer controlling the experimental setup')

args = parser.parse_args()
try:
    case = args.case[0]
except TypeError as e:
    case = CASE_CORR_AND_REW
    print "Defaulted to case ", case, ", i.e. input correlations and rewiring on"

# SpiNNaker setup
start_time = pylab.datetime.datetime.now()

sim.setup(timestep=1.0, min_delay=1.0, max_delay=10)
sim.set_number_of_neurons_per_core("IF_curr_exp", 50)
sim.set_number_of_neurons_per_core("IF_cond_exp", 256 // 13)
sim.set_number_of_neurons_per_core("SpikeSourcePoisson", 256 // 8)


# +-------------------------------------------------------------------+
# | Function definitions                                              |
# +-------------------------------------------------------------------+

# Periodic boundaries
# https://github.com/pabogdan/neurogenesis/blob/master/notebooks/neurogenesis-in-numbers/Periodic%20boundaries.ipynb
def distance(x0, x1, grid=np.asarray([16, 16]), type='euclidian'):
    x0 = np.asarray(x0)
    x1 = np.asarray(x1)
    delta = np.abs(x0 - x1)
    delta = np.where(delta > grid * .5, delta - grid, delta)

    if type == 'manhattan':
        return np.abs(delta).sum(axis=-1)
    return np.sqrt((delta ** 2).sum(axis=-1))


# Poisson spike source as from list spike source
# https://github.com/project-rig/pynn_spinnaker_bcpnn/blob/master/examples/modular_attractor/network.py#L115-L148
def poisson_generator(rate, t_start, t_stop):
    n = int((t_stop - t_start) / 1000.0 * rate)
    number = np.ceil(n + 3 * np.sqrt(n))
    if number < 100:
        number = min(5 + np.ceil(2 * n), 100)

    if number > 0:
        isi = np.random.exponential(1.0 / rate, int(number)) * 1000.0
        if number > 1:
            spikes = np.add.accumulate(isi)
        else:
            spikes = isi
    else:
        spikes = np.array([])

    spikes += t_start
    i = np.searchsorted(spikes, t_stop)

    extra_spikes = []
    if len(spikes) == i:
        # ISI buf overrun

        t_last = spikes[-1] + np.random.exponential(1.0 / rate, 1)[0] * 1000.0

        while (t_last < t_stop):
            extra_spikes.append(t_last)
            t_last += np.random.exponential(1.0 / rate, 1)[0] * 1000.0

            spikes = np.concatenate((spikes, extra_spikes))
    else:
        spikes = np.resize(spikes, (i,))

    # Return spike times, rounded to millisecond boundaries
    return [round(x) for x in spikes]


def generate_rates(s, grid):
    '''
    Function that generates an array the same shape as the input layer so that
    each cell has a value corresponding to the firing rate for the neuron
    at that position.
    '''
    _rates = np.zeros(grid)
    for x in range(grid[0]):
        for y in range(grid[1]):
            _d = distance(s, (x, y), grid)
            _rates[x, y] = f_base + f_peak * np.e ** (
                -_d / (2 * (sigma_stim ** 2)))
    return _rates


def formation_rule(potential_pre, post, sigma, p_form):
    d = distance(potential_pre, post)
    r = np.random.rand()
    p = p_form * (np.e ** (-(d ** 2 / (2 * (sigma ** 2)))))
    if r < p:
        return True
    return False


# Initial connectivity

def generate_initial_connectivity(s, existing_pre, connections, sigma, p, msg):
    # print "|", 256 // 4 * "-", "|"
    # print "|",
    pbar = ProgressBar(total_number_of_things_to_do=N_layer,
                       string_describing_what_being_progressed=msg)
    for postsynaptic_neuron_index in range(N_layer):
        # if postsynaptic_neuron_index % 8 == 0:
        #     print "=",
        pbar.update()
        post = (postsynaptic_neuron_index // n, postsynaptic_neuron_index % n)
        while s[postsynaptic_neuron_index] < s_max:
            potential_pre_index = np.random.randint(0, N_layer)
            pre = (potential_pre_index // n, potential_pre_index % n)
            if potential_pre_index not in existing_pre[
                postsynaptic_neuron_index]:
                if formation_rule(pre, post, sigma, p):
                    s[postsynaptic_neuron_index] += 1
                    existing_pre[postsynaptic_neuron_index].append(
                        potential_pre_index)
                    connections.append((potential_pre_index,
                                        postsynaptic_neuron_index, g_max, 1))
                    # print " |"


# +-------------------------------------------------------------------+
# | General Parameters                                                |
# +-------------------------------------------------------------------+

# Population parameters
model = sim.IF_cond_exp

# Membrane
v_rest = -70  # mV
e_ext = 0  # V
v_thr = -54  # mV
g_max = 0.2
tau_m = 20  # ms
tau_ex = 5  # ms

cell_params = {'cm': 20.0,  # nF
               'i_offset': 0.0,
               'tau_m': 20.0,
               'tau_refrac': 5.0,
               'tau_syn_E': 5.0,
               'tau_syn_I': 5.0,
               'v_reset': -70.0,
               'v_rest': -70.0,
               'v_thresh': -50.0,
               'e_rev_E': 0.,
               'e_rev_I': -80.
               }

# +-------------------------------------------------------------------+
# | Rewiring Parameters                                               |
# +-------------------------------------------------------------------+
no_iterations = 300000 # 3000000 # 3,000,000 iterations
simtime = no_iterations
# Wiring
n = 16
N_layer = n ** 2
S = (n, n)
# S = (256, 1)
grid = np.asarray(S)

s = (n // 2, n // 2)
s_max = 16
sigma_form_forward = 2.5
sigma_form_lateral = 1
p_form_lateral = 1
p_form_forward = 0.16
p_elim_dep = 0.0245
p_elim_pot = 1.36 * np.e ** -4
f_rew = 10 ** 4  # Hz

# Inputs
f_mean = 20  # Hz
f_base = 5  # Hz
f_peak = 152  # 152.8  # Hz
sigma_stim = 2  # 2
t_stim = 20  # 20  # ms
t_record = 1000 # ms
# t_stim = no_iterations
# t_record = no_iterations

# STDP
a_plus = 0.1
b = 1.2
tau_plus = 20.  # ms
tau_minus = 64.  # ms
a_minus = (a_plus * tau_plus * b) / tau_minus

# Reporting

sim_params = {'g_max': g_max,
              't_stim': t_stim,
              'simtime': simtime,
              'f_base': f_base,
              'f_peak': f_peak,
              'sigma_stim': sigma_stim,
              't_record': t_record}

# +-------------------------------------------------------------------+
# | Initial network setup                                             |
# +-------------------------------------------------------------------+
# Need to setup the moving input

# generate all rates
if case == CASE_REW_NO_CORR:
    rates = np.ones(grid) * f_mean
else:
    rates = generate_rates(np.random.randint(0, 16, size=2), grid)
source_pop = sim.Population(N_layer,
                            sim.SpikeSourcePoisson,
                            {'rate': rates.ravel(),
                             'start': 0,
                             'duration': simtime
                             }, label="Poisson spike source")

ff_s = np.zeros(N_layer)
lat_s = np.zeros(N_layer)

existing_pre_ff = []
existing_pre_lat = []
for _ in range(N_layer):
    existing_pre_ff.append([])
    existing_pre_lat.append([])

init_ff_connections = []
init_lat_connections = []
generate_initial_connectivity(
    ff_s, existing_pre_ff, init_ff_connections,
    sigma_form_forward, p_form_forward,
    "\nGenerating initial feedforward connectivity...")
generate_initial_connectivity(
    lat_s, existing_pre_lat, init_lat_connections,
    sigma_form_lateral, p_form_lateral,
    "\nGenerating initial lateral connectivity...")
print "\n"

# Neuron populations
target_pop = sim.Population(N_layer, model, cell_params, label="TARGET_POP")
target_pop.set_constraint(PlacerChipAndCoreConstraint(0, 1))
# Connections
# Plastic Connections between pre_pop and post_pop

stdp_model = sim.STDPMechanism(
    timing_dependence=sim.SpikePairRule(tau_plus=tau_plus,
                                        tau_minus=tau_minus),
    weight_dependence=sim.AdditiveWeightDependence(w_min=0, w_max=g_max,
                                                   # A_plus=0.02, A_minus=0.02
                                                   A_plus=a_plus,
                                                   A_minus=a_minus)
)
if case == CASE_CORR_AND_REW or case == CASE_REW_NO_CORR:
    structure_model_w_stdp = sim.StructuralMechanism(stdp_model=stdp_model,
                                                     weight=g_max,
                                                     s_max=s_max * 2,
                                                     grid=grid)
elif case == CASE_CORR_NO_REW:
    structure_model_w_stdp = stdp_model

# structure_model_w_stdp = sim.StructuralMechanism(weight=g_max, s_max=s_max)

ff_projection = sim.Projection(
    source_pop, target_pop,
    sim.FromListConnector(init_ff_connections),
    synapse_dynamics=sim.SynapseDynamics(slow=structure_model_w_stdp),
    label="plastic_ff_projection"
)

lat_projection = sim.Projection(
    target_pop, target_pop,
    sim.FromListConnector(init_lat_connections),
    synapse_dynamics=sim.SynapseDynamics(slow=structure_model_w_stdp),
    label="plastic_lat_projection"
)

# +-------------------------------------------------------------------+
# | Simulation and results                                            |
# +-------------------------------------------------------------------+

# Record neurons' potentials
# target_pop.record_v()

# Record spikes
source_pop.record()
target_pop.record()

# Run simulation
pre_spikes = None
post_spikes = None

pre_sources = []
pre_targets = []
pre_weights = []
pre_delays = []

post_sources = []
post_targets = []
post_weights = []
post_delays = []

rates_history = np.zeros((16, 16, simtime // t_stim))

print "Starting the sim"
for run in range(simtime // t_stim):
    print "run", run + 1, "of", simtime // t_stim
    sim.run(t_stim)
    if args.case == CASE_REW_NO_CORR:
        rates = np.ones(grid) * f_mean
    else:
        rates = generate_rates(np.random.randint(0, 16, size=2), grid)
    source_pop.set("rate", rates.ravel())

    rates_history[:, :, run] = rates
    # Retrieve data

    if run * t_stim % t_record == 0:
        pre_weights.append(
            np.array(ff_projection._get_synaptic_data(False, 'weight')))
        post_weights.append(
            np.array(lat_projection._get_synaptic_data(False, 'weight')))


pre_spikes = source_pop.getSpikes(compatible_output=True)
post_spikes = target_pop.getSpikes(compatible_output=True)

# sim.run(simtime)

# print("Weights:", plastic_projection.getWeights())
end_time = pylab.datetime.datetime.now()
total_time = end_time - start_time

print "Total time elapsed -- " + str(total_time)


def plot_spikes(spikes, title):
    if spikes is not None:
        f, ax1 = pylab.subplots(1, 1, figsize=(16, 8))
        ax1.set_xlim((0, simtime))
        ax1.scatter([i[1] for i in spikes], [i[0] for i in spikes], s=.2)
        ax1.set_xlabel('Time/ms')
        ax1.set_ylabel('spikes')
        ax1.set_title(title)

    else:
        print "No spikes received"


init_ff_conn_network = np.ones((256, 256)) * np.nan
init_lat_conn_network = np.ones((256, 256)) * np.nan
for source, target, weight, delay in init_ff_connections:
    init_ff_conn_network[int(source), int(target)] = weight
for source, target, weight, delay in init_lat_connections:
    init_lat_conn_network[int(source), int(target)] = weight

suffix = time.strftime("_%H%M%S_%d%m%Y")
np.savez("structural_results_stdp" + suffix, pre_spikes=pre_spikes,
         post_spikes=post_spikes,
         init_ff_connections=init_ff_conn_network,
         init_lat_connections=init_lat_conn_network,
         pre_weights=pre_weights,
         post_weights=post_weights,
         simtime=simtime, rate_trace=rates_history, sim_params=sim_params,
         total_time=total_time)

# https://stackoverflow.com/questions/36809437/dynamic-marker-colour-in-matplotlib
# pretty cool effect

plot_spikes(pre_spikes, "Source layer spikes")
pylab.show()
plot_spikes(post_spikes, "Target layer spikes")
pylab.show()


f, (ax1, ax2) = pylab.subplots(1, 2, figsize=(16, 8))
i = ax1.matshow(np.nan_to_num(pre_weights[-1].reshape(256, 256)))
i2 = ax2.matshow(np.nan_to_num(post_weights[-1].reshape(256, 256)))
ax1.grid(visible=False)
ax1.set_title("Feedforward connectivity matrix", fontsize=16)
ax2.set_title("Lateral connectivity matrix", fontsize=16)
cbar_ax = f.add_axes([.91, 0.155, 0.025, 0.72])
cbar = f.colorbar(i2, cax=cbar_ax)
cbar.set_label("Synaptic conductance - $G_{syn}$", fontsize=16)
pylab.show()

f, (ax1, ax2) = pylab.subplots(1, 2, figsize=(16, 8))
i = ax1.matshow(
    np.nan_to_num(pre_weights[-1].reshape(256, 256)) - np.nan_to_num(
        init_ff_conn_network))
i2 = ax2.matshow(
    np.nan_to_num(post_weights[-1].reshape(256, 256)) - np.nan_to_num(
        init_lat_conn_network))
ax1.grid(visible=False)
ax1.set_title("Diff- Feedforward connectivity matrix", fontsize=16)
ax2.set_title("Diff- Lateral connectivity matrix", fontsize=16)
cbar_ax = f.add_axes([.91, 0.155, 0.025, 0.72])
cbar = f.colorbar(i2, cax=cbar_ax)
cbar.set_label("Synaptic conductance - $G_{syn}$", fontsize=16)
pylab.show()
# End simulation on SpiNNaker
sim.end()
print "Total time elapsed -- " + str(total_time)
