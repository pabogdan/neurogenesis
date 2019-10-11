"""
Test for topographic map formation using STDP and synaptic rewiring.

http://hdl.handle.net/1842/3997
"""
# Imports
import traceback

from synaptogenesis.function_definitions import *
from synaptogenesis.argparser import *

import numpy as np
import pylab as plt

import spynnaker8 as sim
from spynnaker8.extra_models import SpikeSourcePoissonVariable

case = args.case
print("Case", case, "selected!")

# SpiNNaker setup
start_time = plt.datetime.datetime.now()

sim.setup(timestep=0.1, min_delay=0.1, max_delay=1)
sim.set_number_of_neurons_per_core(sim.IF_cond_exp, 32)
sim.set_number_of_neurons_per_core(sim.SpikeSourcePoisson, 16)
sim.set_number_of_neurons_per_core(SpikeSourcePoissonVariable, 16)

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

s_max = args.s_max // 2
sigma_form_forward = args.sigma_form_ff
sigma_form_lateral = args.sigma_form_lat
p_form_lateral = args.p_form_lateral
p_form_forward = args.p_form_forward
p_elim_dep = args.p_elim_dep
p_elim_pot = args.p_elim_pot
f_rew = args.f_rew  # 10 ** 4  # Hz

# Inputs
f_mean = args.f_mean  # Hz
f_base = 5  # Hz
f_peak = args.f_peak  # 152.8  # Hz
sigma_stim = args.sigma_stim  # 2
t_stim = args.t_stim  # 20  # ms
t_record = args.t_record  # ms

# STDP
a_plus = 0.1
b = args.b
tau_plus = 20.  # ms
tau_minus = args.t_minus  # ms
a_minus = (a_plus * tau_plus * b) / tau_minus
# a_minus = 0.0375

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
              'a_plus': a_plus,
              'input_type': args.input_type,
              'random_partner': args.random_partner,
              'lesion': args.lesion
              }

if args.input_type == GAUSSIAN_INPUT:
    print("Gaussian input")
    gen_rate = generate_gaussian_input_rates
elif args.input_type == POINTY_INPUT:
    print("Pointy input")
    gen_rate = generate_rates
elif args.input_type == SCALED_POINTY_INPUT:
    print("Scaled pointy input")
    gen_rate = generate_scaled_pointy_rates
elif args.input_type == SQUARE_INPUT:
    print("Square input")
    gen_rate = generate_square_rates

# +-------------------------------------------------------------------+
# | Initial network setup                                             |
# +-------------------------------------------------------------------+
# Need to setup the moving input

number_of_slots = int(simtime / t_stim)
range_of_slots = np.arange(number_of_slots)
slots_starts = np.ones((N_layer, number_of_slots)) * (range_of_slots * t_stim)
durations = np.ones((N_layer, number_of_slots)) * t_stim

if case == CASE_REW_NO_CORR:
    rates = np.ones(grid) * f_mean
    source_pop = sim.Population(N_layer,
                                sim.SpikeSourcePoisson,
                                {'rate': rates.ravel(),
                                 'start': 100,
                                 'duration': simtime
                                 }, label="Poisson spike source")
elif case == CASE_CORR_AND_REW or case == CASE_CORR_NO_REW:

    rates = np.empty((grid[0], grid[1], number_of_slots))
    for rate_id in range(number_of_slots):
        r = gen_rate(np.random.randint(0, n, size=2),
                     f_base=f_base,
                     grid=grid,
                     f_peak=args.f_peak,
                     sigma_stim=sigma_stim)
        rates[:, :, rate_id] = r
    rates = rates.reshape(N_layer, number_of_slots)

    source_pop = sim.Population(N_layer,
                                SpikeSourcePoissonVariable,
                                {'rates': rates,
                                 'starts': slots_starts,
                                 'durations': durations
                                 }, label="Variable-rate Poisson spike source")

ff_s = np.zeros(N_layer, dtype=np.uint)
lat_s = np.zeros(N_layer, dtype=np.uint)

init_ff_connections = []
init_lat_connections = []

if args.initial_connectivity_file is None:
    init_ff_connections = generate_initial_connectivity(
        sigma_form_forward, p_form_forward,
        "\nGenerating initial feedforward connectivity...",
        N_layer=N_layer, n=n, s_max=s_max, g_max=g_max, delay=args.delay)
    init_lat_connections = generate_initial_connectivity(
        sigma_form_lateral, p_form_lateral,
        "\nGenerating initial lateral connectivity...",
        N_layer=N_layer, n=n, s_max=s_max, g_max=g_max, delay=args.delay)
    print("\n")
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
# Connections
# Plastic Connections between pre_pop and post_pop

stdp_model = sim.STDPMechanism(
    timing_dependence=sim.SpikePairRule(tau_plus=tau_plus, tau_minus=tau_minus, A_plus=a_plus, A_minus=a_minus),
    weight_dependence=sim.AdditiveWeightDependence(w_min=0, w_max=g_max)
)

if case == CASE_CORR_AND_REW or case == CASE_REW_NO_CORR:
    structure_model_w_stdp = sim.StructuralMechanismSTDP(
        stdp_model=stdp_model,  # wrap around the STDP model to use
        weight=g_max,  # initial weight
        delay=args.delay,  # synaptic delay
        s_max=s_max * 2,  # synaptic capacity
        grid=grid,  # grid description
        f_rew=f_rew,  # rate of rewiring
        lateral_inhibition=args.lateral_inhibition,  # lateral connections are always inhibitory
        random_partner=args.random_partner,  # form connections to a random partner, not L2S
        p_elim_dep=p_elim_dep,  # probability of eliminating a depressed synapse
        p_elim_pot=p_elim_pot,  # probability of eliminating a potentiated synapse
        sigma_form_forward=sigma_form_forward,  # spread of feadforward receptive field
        sigma_form_lateral=sigma_form_lateral,  # spread of lateral receptive field
        p_form_forward=p_form_forward,  # feedforward formation probability
        p_form_lateral=p_form_lateral  # lateral formation probability
    )
elif case == CASE_CORR_NO_REW:
    structure_model_w_stdp = stdp_model

# structure_model_w_stdp = sim.StructuralMechanism(weight=g_max, s_max=s_max)

if not args.lesion:
    print("No insults")
    ff_projection = sim.Projection(
        source_pop, target_pop,
        sim.FromListConnector(init_ff_connections),
        synapse_type=structure_model_w_stdp,
        label="plastic_ff_projection"
    )

    lat_projection = sim.Projection(
        target_pop, target_pop,
        sim.FromListConnector(init_lat_connections),
        synapse_type=structure_model_w_stdp,
        label="plastic_lat_projection",
        receptor_type="inhibitory" if args.lateral_inhibition else "excitatory"
    )
elif args.lesion == ONE_TO_ONE_LESION:
    # ff_pos = range(len(init_ff_connections))
    # lat_pos = range(len(init_lat_connections))
    # subsample_ff = np.random.choice(ff_pos, 10)
    # subsample_lat = np.random.choice(lat_pos, 10)
    # init_ff_connections = np.asarray(init_ff_connections)
    # init_lat_connections = np.asarray(init_lat_connections)
    print("Insulted network")

    # ff_prob_conn = [(i, j, g_max, args.delay) for i in range(N_layer) for j in range(N_layer) if np.random.rand() < .05]
    # lat_prob_conn = [(i, j, g_max, args.delay) for i in range(N_layer) for j in range(N_layer) if np.random.rand() < .05]
    #
    # init_ff_connections = ff_prob_conn
    # init_lat_connections = lat_prob_conn

    # one_to_one_conn = [(i, i, g_max, args.delay) for i in range(N_layer)]
    one_to_one_conn = []
    ff_projection = sim.Projection(
        source_pop, target_pop,
        # sim.FromListConnector(one_to_one_conn),
        # sim.FromListConnector(ff_prob_conn),
        # sim.OneToOneConnector(weights=g_max, delays=args.delay),
        # sim.FromListConnector(init_ff_connections[subsample_ff]),
        sim.FixedProbabilityConnector(p_connect=0.),
        synapse_type=structure_model_w_stdp,
        label="plastic_ff_projection"
    )

    lat_projection = sim.Projection(
        target_pop, target_pop,
        # sim.FromListConnector(one_to_one_conn),
        # sim.FromListConnector(lat_prob_conn),
        # sim.OneToOneConnector(weights=g_max, delays=args.delay),
        # sim.FromListConnector(init_lat_connections[subsample_lat]),
        sim.FixedProbabilityConnector(p_connect=0.0),
        synapse_type=structure_model_w_stdp,
        label="plastic_lat_projection",
        receptor_type="inhibitory" if args.lateral_inhibition else "excitatory"
    )

    init_ff_connections = one_to_one_conn
    init_lat_connections = one_to_one_conn

    # init_ff_connections = init_ff_connections[subsample_ff]
    # init_lat_connections = init_lat_connections[subsample_lat]

elif args.lesion == RANDOM_CONNECTIVITY_LESION:
    # ff_pos = range(len(init_ff_connections))
    # lat_pos = range(len(init_lat_connections))
    # subsample_ff = np.random.choice(ff_pos, 10)
    # subsample_lat = np.random.choice(lat_pos, 10)
    # init_ff_connections = np.asarray(init_ff_connections)
    # init_lat_connections = np.asarray(init_lat_connections)
    print("Insulted network")

    ff_prob_conn = [(i, j, g_max, args.delay) for i in range(N_layer) for j in
                    range(N_layer) if np.random.rand() < .05]
    lat_prob_conn = [(i, j, g_max, args.delay) for i in range(N_layer) for j in
                     range(N_layer) if np.random.rand() < .05]

    init_ff_connections = ff_prob_conn
    init_lat_connections = lat_prob_conn

    ff_projection = sim.Projection(
        source_pop, target_pop,
        sim.FromListConnector(ff_prob_conn),
        synapse_type=structure_model_w_stdp,
        label="plastic_ff_projection"
    )

    lat_projection = sim.Projection(
        target_pop, target_pop,
        sim.FromListConnector(lat_prob_conn),
        synapse_type=structure_model_w_stdp,
        label="plastic_lat_projection",
        receptor_type="inhibitory" if args.lateral_inhibition else "excitatory"
    )

    # init_ff_connections = one_to_one_conn
    # init_lat_connections = one_to_one_conn
    #
    # init_ff_connections = init_ff_connections[subsample_ff]
    # init_lat_connections = init_lat_connections[subsample_lat]

# +-------------------------------------------------------------------+
# | Simulation and results                                            |
# +-------------------------------------------------------------------+

# Record neurons' potentials
# target_pop.record_v()

# Record spikes
# if case == CASE_REW_NO_CORR:
if args.record_source:
    source_pop.record(['spikes'])
target_pop.record(['spikes'])

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
print("Starting the sim")

no_runs = simtime // t_record
run_duration = t_record

try:
    for current_run in range(no_runs):
        print("run", current_run + 1, "of", no_runs)
        sim.run(run_duration)

        if (current_run + 1) * run_duration % t_record == 0:
            # TODO using public method results in stupid data structure
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
        pre_spikes = source_pop.spinnaker_get_data('spikes')
    else:
        pre_spikes = []
    post_spikes = target_pop.spinnaker_get_data('spikes')
    # End simulation on SpiNNaker
    sim.end()
except Exception as e:
    # print(e)
    traceback.print_exc()
# print("Weights:", plastic_projection.getWeights())
end_time = plt.datetime.datetime.now()
total_time = end_time - start_time

pre_spikes = np.asarray(pre_spikes)
post_spikes = np.asarray(post_spikes)

print("Total time elapsed -- " + str(total_time))

suffix = end_time.strftime("_%H%M%S_%d%m%Y")

if args.filename:
    filename = args.filename
else:
    filename = "pynn8_topographic_map_results" + str(suffix)

total_target_neuron_mean_spike_rate = \
    post_spikes.shape[0] / float(simtime) * 1000. / N_layer

np.savez_compressed(
    filename, pre_spikes=pre_spikes,
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
    insult=args.lesion,
    input_type=args.input_type)

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
            print("No spikes received")


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

print("Results in", filename)
print("Total time elapsed -- " + str(total_time))
