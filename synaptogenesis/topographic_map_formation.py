"""
Test for topographic map formation using STDP and synaptic rewiring.

http://hdl.handle.net/1842/3997
"""
# Imports
import numpy as np
import pylab

from pacman.model.constraints.placer_constraints.placer_chip_and_core_constraint import PlacerChipAndCoreConstraint

import spynnaker7.pyNN as sim

# SpiNNaker setup
sim.setup(timestep=1.0, min_delay=1.0, max_delay=10)
sim.set_number_of_neurons_per_core("IF_curr_exp", 50)
sim.set_number_of_neurons_per_core("IF_cond_exp", 25)
sim.set_number_of_neurons_per_core("SpikeSourcePoisson", 100)


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
            _rates[x, y] = f_base + f_peak * np.e ** (-_d / (2 * (sigma_stim ** 2)))
    return _rates


def generate_spikes(rates):
    spikes = np.zeros((16, 16, 21))
    for x in range(16):
        for y in range(16):
            spikes_times_xy = poisson_generator(rates[x, y], 0, 20)
            for time in spikes_times_xy:
                spikes[x, y, int(time)] = 1
    return spikes


def formation_rule(potential_pre, post, sigma, p_form):
    d = distance(potential_pre, post)
    r = np.random.rand()
    p = p_form * (np.e ** (-(d ** 2 / (2 * (sigma ** 2)))))
    if r < p:
        return True
    return False

# Initial connectivity

def generate_initial_connectivity(s, existing_pre, connections, sigma, p):
    print "|", 256 // 4 * "-", "|"
    print "|",
    for postsynaptic_neuron_index in range(N_layer):
        if postsynaptic_neuron_index % 8 == 0:
            print "=",
        post = (postsynaptic_neuron_index // n, postsynaptic_neuron_index % n)
        while s[postsynaptic_neuron_index] < s_max:
            potential_pre_index = np.random.randint(0, N_layer)
            pre = (potential_pre_index // n, potential_pre_index % n)
            if potential_pre_index not in existing_pre[postsynaptic_neuron_index]:
                if formation_rule(pre, post, sigma, p):
                    s[postsynaptic_neuron_index] += 1
                    existing_pre[postsynaptic_neuron_index].append(potential_pre_index)
                    connections.append((potential_pre_index, postsynaptic_neuron_index, g_max, 1))
    print " |"

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
no_iterations = 60000
simtime = no_iterations
# Wiring
n = 16
N_layer = n ** 2
S = (n, n)
grid = np.asarray(S)
g_max = 0.2

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
f_mean = 5  # Hz
f_base = 5  # Hz
f_peak = 5 #152.8  # Hz
sigma_stim = 3#2
t_stim = 1000 #20  # ms

# STDP
a_plus = 0.1
b = 1.2
tau_plus = 20.  # ms
tau_minus = 64.  # ms
a_minus = (a_plus * tau_plus * b) / tau_minus

# +-------------------------------------------------------------------+
# | Initial network setup                                             |
# +-------------------------------------------------------------------+
# Need to setup the moving input

# generate all rates

rates = generate_rates((n // 2, n // 2), grid)
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
print "| Generating initial feedforward connectivity..."
generate_initial_connectivity(ff_s, existing_pre_ff, init_ff_connections, sigma_form_forward, p_form_forward)
print "| Generating initial lateral     connectivity..."
generate_initial_connectivity(lat_s, existing_pre_lat, init_lat_connections, sigma_form_lateral, p_form_lateral)

# Neuron populations
target_pop = sim.Population(N_layer, model, cell_params, label="TARGET_POP")
target_pop.set_constraint(PlacerChipAndCoreConstraint(0, 1))
# Connections
# Plastic Connections between pre_pop and post_pop
stdp_model = sim.STDPMechanism(
    timing_dependence=sim.SpikePairRule(tau_plus=tau_plus, tau_minus=tau_minus),
    weight_dependence=sim.AdditiveWeightDependence(w_min=0, w_max=g_max,
                                                   # A_plus=0.02, A_minus=0.02
                                                   A_plus=a_plus, A_minus=a_minus)
)

structure_model_w_stdp = sim.StructuralMechanism(stdp_model=stdp_model, weight=g_max, s_max=s_max)
# structure_model_w_stdp = sim.StructuralMechanism(weight=g_max, s_max=s_max)

ff_projection = sim.Projection(
    source_pop, target_pop,
    sim.FromListConnector(init_ff_connections),
    # sim.FixedNumberPreConnector(16, weights=0.2),
    # sim.FixedProbabilityConnector(0),  # TODO change to a FromListConnector
    synapse_dynamics=sim.SynapseDynamics(slow=structure_model_w_stdp),
    label="plastic_ff_projection"
)

lat_projection = sim.Projection(
    target_pop, target_pop,
    # sim.FromListConnector(init_lat_connections),
    # sim.FixedNumberPreConnector(16, weights=0.2),
    sim.FixedProbabilityConnector(0.3, weights=0),  # TODO change to a FromListConnector
    # synapse_dynamics=sim.SynapseDynamics(slow=structure_model_w_stdp),
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
# for run in range(simtime//t_stim):
#     rates = generate_rates(np.random.randint(0, 16, size=2), grid)
#     source_pop = sim.Population(N_layer,
#                                 sim.SpikeSourcePoisson,
#                                 {'rate': rates.ravel(),
#                                  'start': run * t_stim,
#                                  'duration': (run + 1)* t_stim
#                                  }, label="Poisson spike source")
#     sim.run(t_stim)
sim.run(simtime)

# print("Weights:", plastic_projection.getWeights())


def plot_spikes(spikes, title):
    if spikes is not None:
        pylab.figure()
        pylab.xlim((0, simtime))
        pylab.plot([i[1] for i in spikes], [i[0] for i in spikes], ".")
        pylab.xlabel('Time/ms')
        pylab.ylabel('spikes')
        pylab.title(title)

    else:
        print "No spikes received"


pre_spikes = source_pop.getSpikes(compatible_output=True)
post_spikes = target_pop.getSpikes(compatible_output=True)

pre_sources = np.asarray([ff_projection._get_synaptic_data(True, 'source')]).T
pre_targets = np.asarray([ff_projection._get_synaptic_data(True, 'target')]).T
pre_weights = np.asarray([ff_projection._get_synaptic_data(True, 'weight')]).T
# sanity check
pre_delays = np.asarray([ff_projection._get_synaptic_data(True, 'delay')]).T

ff_proj = np.concatenate((pre_sources, pre_targets, pre_weights, pre_delays), axis=1)

post_sources = np.asarray([lat_projection._get_synaptic_data(True, 'source')]).T
post_targets = np.asarray([lat_projection._get_synaptic_data(True, 'target')]).T
post_weights = np.asarray([lat_projection._get_synaptic_data(True, 'weight')]).T
# sanity check
post_delays = np.asarray([lat_projection._get_synaptic_data(True, 'delay')]).T

lat_proj = np.concatenate((post_sources, post_targets, post_weights, post_delays), axis=1)

import time

suffix = time.strftime("_%H%M%S_%d%m%Y")
np.savez("structural_results_stdp" + suffix, pre_spikes=pre_spikes, post_spikes=post_spikes,
         ff_projection_w=ff_proj, lat_projection_w=lat_proj, simtime=simtime)

# https://stackoverflow.com/questions/36809437/dynamic-marker-colour-in-matplotlib
# pretty cool effect
plot_spikes(post_spikes, "Target layer spikes")
pylab.show()

connectivity_matrix = np.zeros((256, 256))
for source, target, weight, delay in ff_proj:
    assert delay == 1
    connectivity_matrix[int(source), int(target)] = weight
lat_connectivity_matrix = np.zeros((256, 256))
for source, target, weight, delay in lat_proj:
    assert delay == 1
    lat_connectivity_matrix[int(source), int(target)] = weight

init_ff_conn_network = np.zeros((256, 256))
init_lat_conn_network = np.zeros((256, 256))
for source, target, weight, delay in init_ff_connections:
    init_ff_conn_network[int(source), int(target)] = weight
for source, target, weight, delay in init_lat_connections:
    init_lat_conn_network[int(source), int(target)] = weight

f, (ax1, ax2) = pylab.subplots(1, 2, figsize=(16, 8))
i = ax1.matshow(connectivity_matrix)
i2 = ax2.matshow(lat_connectivity_matrix)
ax1.grid(visible=False)
ax1.set_title("Feedforward connectivity matrix", fontsize=16)
ax2.set_title("Lateral connectivity matrix", fontsize=16)
cbar_ax = f.add_axes([.91, 0.155, 0.025, 0.72])
cbar = f.colorbar(i2, cax=cbar_ax)
cbar.set_label("Synaptic conductance - $G_{syn}$", fontsize=16)
pylab.show()


f, (ax1, ax2) = pylab.subplots(1, 2, figsize=(16, 8))
i = ax1.matshow(connectivity_matrix - init_ff_conn_network)
i2 = ax2.matshow(lat_connectivity_matrix - init_lat_conn_network)
ax1.grid(visible=False)
ax1.set_title("Diff- Feedforward connectivity matrix", fontsize=16)
ax2.set_title("Diff- Lateral connectivity matrix", fontsize=16)
cbar_ax = f.add_axes([.91, 0.155, 0.025, 0.72])
cbar = f.colorbar(i2, cax=cbar_ax)
cbar.set_label("Synaptic conductance - $G_{syn}$", fontsize=16)
pylab.show()
# End simulation on SpiNNaker
sim.end()
