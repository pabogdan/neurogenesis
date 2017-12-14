"""
Test for ocular preferrence in topographic maps using STDP and synaptic rewiring.

http://hdl.handle.net/1842/3997
"""
# Imports
import numpy as np
import pylab as plt
import time
from pacman.model.constraints.placer_constraints.placer_chip_and_core_constraint import \
    PlacerChipAndCoreConstraint
import spynnaker7.pyNN as sim

from function_definitions import *
from argparser import *

case = args.case
print "Case", case, "selected!"

# SpiNNaker setup
start_time = plt.datetime.datetime.now()

sim.setup(timestep=1.0, min_delay=1.0, max_delay=10)
sim.set_number_of_neurons_per_core("IF_curr_exp", 50)
sim.set_number_of_neurons_per_core("IF_cond_exp", 256 // 10)
sim.set_number_of_neurons_per_core("SpikeSourcePoisson", 256 // 13)
sim.set_number_of_neurons_per_core("SpikeSourcePoissonVariable", 256 // 13)

# +-------------------------------------------------------------------+
# | General Parameters                                                |
# +-------------------------------------------------------------------+

# Population parameters
model = sim.IF_cond_exp

# Membrane
v_rest = -70  # mV
e_ext = 0  # V
v_thr = -54  # mV
g_max = 0.1
tau_m = 20  # ms
tau_ex = 5  # ms

cell_params = {'cm': 20.0,  # nF
               'i_offset': 0.0,
               'tau_m': 20.0,
               'tau_refrac': args.tau_refrac,
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
no_iterations = args.no_iterations  # 300000 # 3000000 # 3,000,000 iterations
simtime = no_iterations
# Wiring
n = args.n
N_layer = n ** 2
S = (n, n)
# S = (256, 1)
grid = np.asarray(S)

s_max = args.s_max
sigma_form_forward = 1.99
sigma_form_lateral = 2.49
p_form_lateral = args.p_form_lateral
p_form_forward = args.p_form_forward
p_elim_dep = 0.01
p_elim_pot = 5.56 * (10.**-5)
f_rew = args.f_rew  # 10 ** 4  # Hz

# Inputs
f_mean = args.f_mean  # Hz
f_base = 5  # Hz
f_peak = 152.8 * 2  # 152.8  # Hz
sigma_stim = args.sigma_stim  # 2
t_stim = args.t_stim  # 20  # ms
t_record = args.t_record  # ms

# STDP
a_plus = 0.1
b = args.b
tau_plus = 20.  # ms
tau_minus = args.t_minus  # ms
a_minus = (a_plus * tau_plus * b) / tau_minus

# Reporting

sim_params = {'g_max': g_max,
              't_stim': t_stim,
              'simtime': simtime,
              'f_base': f_base,
              'f_peak': f_peak,
              'sigma_stim': sigma_stim,
              't_record': t_record,
              'cell_params': cell_params,
              'case': args.case,
              'grid': grid,
              's_max': s_max,
              'sigma_form_forward': sigma_form_forward,
              'sigma_form_lateral': sigma_form_lateral,
              'p_form_lateral': p_form_lateral,
              'p_form_forward': p_form_forward,
              'p_elim_dep': p_elim_dep,
              'p_elim_pot': p_elim_pot,
              'f_rew': f_rew,
              'lateral_inhibition': args.lateral_inhibition,
              'delay': args.delay,
              'b': b,
              't_minus': tau_minus,
              't_plus': tau_plus,
              'tau_refrac': args.tau_refrac,
              'a_minus': a_minus,
              'a_plus': a_plus
              }

if args.gaussian_input:
    gen_rate = generate_gaussian_input_rates
else:
    gen_rate = generate_rates

# +-------------------------------------------------------------------+
# | Initial network setup                                             |
# +-------------------------------------------------------------------+
# Need to setup the moving input
one_row = np.asarray(np.arange(16) % 2, dtype=bool)

binoc_positions = np.asarray([one_row if i % 2 == 0 else np.logical_not(one_row) for i in range(16)])

left_positions = np.where(binoc_positions==0)
right_positions = np.where(binoc_positions==1)

positions = [left_positions, right_positions]

if case == CASE_REW_NO_CORR:
    raise NotImplementedError

elif case == CASE_CORR_AND_REW or case == CASE_CORR_NO_REW:

    rates = np.empty((simtime // t_stim, grid[0], grid[1]))
    for rate_id in range(simtime // t_stim):
        rand_offset = np.random.randint(0, N_layer//2)
        stim_position = (positions[rate_id%2][0][rand_offset], positions[rate_id%2][1][rand_offset])
        assert binoc_positions[stim_position] == rate_id%2
        r = gen_rate(stim_position,
                     f_base=f_base,
                     grid=grid,
                     f_peak=f_peak,
                     sigma_stim=sigma_stim)

        rates[rate_id, :, :] = r
    rates = rates.reshape(simtime // t_stim, N_layer)

    source_pop = sim.Population(N_layer,
                                sim.SpikeSourcePoissonVariable,
                                {'rate': rates,
                                 'start': 100,
                                 'duration': simtime,
                                 'rate_interval_duration': t_stim
                                 }, label="Variable-rate Poisson spike source")

ff_s = np.zeros(N_layer, dtype=np.uint)
lat_s = np.zeros(N_layer, dtype=np.uint)

init_ff_connections = []
init_lat_connections = []

if args.initial_connectivity_file is None:
    raise NotImplementedError
else:
    if "npz" in args.initial_connectivity_file:
        initial_connectivity = np.load(args.initial_connectivity_file)
    else:
        import scipy.io as io

        initial_connectivity = io.loadmat(args.initial_connectivity_file)
    conn = initial_connectivity['ConnPostToPre'] - 1
    weight = initial_connectivity['WeightPostToPre']

    for target in range(conn.shape[1]):
        for index in range(conn.shape[0]):
            if conn[index, target] >= 0:
                if conn[index, target] < N_layer:
                    init_ff_connections.append(
                        (conn[index, target], target,
                         weight[index, target], 1))
                else:
                    init_lat_connections.append(
                        (conn[index, target] - N_layer, target,
                         weight[index, target], 1))

# Neuron populations
target_pop = sim.Population(N_layer, model, cell_params, label="TARGET_POP")
# Putting this populations on chip 0 1 makes it easier to copy the provenance
# data somewhere else
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
    structure_model_w_stdp = sim.StructuralMechanism(
        stdp_model=stdp_model,
        weight=g_max,
        delay=args.delay,
        s_max=s_max,
        grid=grid,
        f_rew=f_rew,
        lateral_inhibition=args.lateral_inhibition,
        random_partner=args.random_partner,
        p_elim_dep=p_elim_dep,
        p_elim_pot=p_elim_pot,
        sigma_form_forward=sigma_form_forward,
        sigma_form_lateral=sigma_form_lateral,
        p_form_forward=p_form_forward,
        p_form_lateral=p_form_lateral
    )
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
    label="plastic_lat_projection",
    target="inhibitory" if args.lateral_inhibition else "excitatory"
)

# +-------------------------------------------------------------------+
# | Simulation and results                                            |
# +-------------------------------------------------------------------+

# Record neurons' potentials
# target_pop.record_v()

# Record spikes
# if case == CASE_REW_NO_CORR:
if args.record_source:
    source_pop.record()
target_pop.record()

# Run simulation
pre_spikes = []
post_spikes = []

pre_sources = []
pre_targets = []
pre_weights = []
pre_delays = []

post_sources = []
post_targets = []
post_weights = []
post_delays = []

# rates_history = np.zeros((16, 16, simtime // t_stim))
e = None
print "Starting the sim"

no_runs = simtime // t_record
run_duration = t_record

try:
    for current_run in range(no_runs):
        print "run", current_run + 1, "of", no_runs
        sim.run(run_duration)

        if (current_run + 1) * run_duration % t_record == 0:
            pre_weights.append(
                np.array([
                    ff_projection._get_synaptic_data(True, 'source'),
                    ff_projection._get_synaptic_data(True, 'target'),
                    ff_projection._get_synaptic_data(True, 'weight'),
                    ff_projection._get_synaptic_data(True, 'delay')]).T)
            post_weights.append(
                np.array([
                    lat_projection._get_synaptic_data(True, 'source'),
                    lat_projection._get_synaptic_data(True, 'target'),
                    lat_projection._get_synaptic_data(True, 'weight'),
                    lat_projection._get_synaptic_data(True, 'delay')]).T)
    if args.record_source:
        pre_spikes = source_pop.getSpikes(compatible_output=True)
    else:
        pre_spikes = []
    post_spikes = target_pop.getSpikes(compatible_output=True)
    # End simulation on SpiNNaker
    sim.end()
except Exception as e:
    print e
# print("Weights:", plastic_projection.getWeights())
end_time = plt.datetime.datetime.now()
total_time = end_time - start_time

pre_spikes = np.asarray(pre_spikes)
post_spikes = np.asarray(post_spikes)

print "Total time elapsed -- " + str(total_time)

suffix = end_time.strftime("_%H%M%S_%d%m%Y")

if args.filename:
    filename = args.filename
else:
    filename = "ocular_preference_results" + str(suffix)

total_target_neuron_mean_spike_rate = \
    post_spikes.shape[0] / float(simtime) * 1000. / N_layer

np.savez(filename, pre_spikes=pre_spikes,
         post_spikes=post_spikes,
         init_ff_connections=init_ff_connections,
         init_lat_connections=init_lat_connections,
         ff_connections=pre_weights,
         lat_connections=post_weights,
         final_pre_weights=pre_weights[-1],
         final_post_weights=post_weights[-1],
         simtime=simtime,
         sim_params=sim_params,
         total_time=total_time,
         mean_firing_rate=total_target_neuron_mean_spike_rate,
         exception=e,
         insult=args.insult)

# Plotting
if args.plot and e is None:
    init_ff_conn_network = np.ones((256, 256)) * np.nan
    init_lat_conn_network = np.ones((256, 256)) * np.nan
    for source, target, weight, delay in init_ff_connections:
        if np.isnan(init_ff_conn_network[int(source), int(target)]):
            init_ff_conn_network[int(source), int(target)] = weight
        else:
            init_ff_conn_network[int(source), int(target)] += weight
    for source, target, weight, delay in init_lat_connections:
        if np.isnan(init_lat_conn_network[int(source), int(target)]):
            init_lat_conn_network[int(source), int(target)] = weight
        else:
            init_lat_conn_network[int(source), int(target)] += weight


    def plot_spikes(spikes, title):
        if spikes is not None and len(spikes) > 0:
            f, ax1 = plt.subplots(1, 1, figsize=(16, 8))
            ax1.set_xlim((0, simtime))
            ax1.scatter([i[1] for i in spikes], [i[0] for i in spikes], s=.2)
            ax1.set_xlabel('Time/ms')
            ax1.set_ylabel('spikes')
            ax1.set_title(title)

        else:
            print "No spikes received"


    plot_spikes(pre_spikes, "Source layer spikes")
    plt.show()
    plot_spikes(post_spikes, "Target layer spikes")
    plt.show()

    final_ff_conn_network = np.ones((256, 256)) * np.nan
    final_lat_conn_network = np.ones((256, 256)) * np.nan
    for source, target, weight, delay in pre_weights[-1]:
        if np.isnan(final_ff_conn_network[int(source), int(target)]):
            final_ff_conn_network[int(source), int(target)] = weight
        else:
            final_ff_conn_network[int(source), int(target)] += weight
        assert delay == args.delay

    for source, target, weight, delay in post_weights[-1]:
        if np.isnan(final_lat_conn_network[int(source), int(target)]):
            final_lat_conn_network[int(source), int(target)] = weight
        else:
            final_lat_conn_network[int(source), int(target)] += weight
        assert delay == args.delay

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    i = ax1.matshow(np.nan_to_num(final_ff_conn_network))
    i2 = ax2.matshow(np.nan_to_num(final_lat_conn_network))
    ax1.grid(visible=False)
    ax1.set_title("Feedforward connectivity matrix", fontsize=16)
    ax2.set_title("Lateral connectivity matrix", fontsize=16)
    cbar_ax = f.add_axes([.91, 0.155, 0.025, 0.72])
    cbar = f.colorbar(i2, cax=cbar_ax)
    cbar.set_label("Synaptic conductance - $G_{syn}$", fontsize=16)
    plt.show()

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    i = ax1.matshow(
        np.nan_to_num(final_ff_conn_network) - np.nan_to_num(
            init_ff_conn_network))
    i2 = ax2.matshow(
        np.nan_to_num(final_lat_conn_network) - np.nan_to_num(
            init_lat_conn_network))
    ax1.grid(visible=False)
    ax1.set_title("Diff- Feedforward connectivity matrix", fontsize=16)
    ax2.set_title("Diff- Lateral connectivity matrix", fontsize=16)
    cbar_ax = f.add_axes([.91, 0.155, 0.025, 0.72])
    cbar = f.colorbar(i2, cax=cbar_ax)
    cbar.set_label("Synaptic conductance - $G_{syn}$", fontsize=16)
    plt.show()

print "Results in", filename
print "Total time elapsed -- " + str(total_time)
