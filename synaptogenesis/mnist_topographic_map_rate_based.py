import numpy as np
import traceback
import pylab as plt
import spynnaker7.pyNN as sim
from function_definitions import *
from argparser import *
import sys

# SpiNNaker setup
start_time = plt.datetime.datetime.now()

sim.setup(timestep=1.0, min_delay=1.0, max_delay=10)
sim.set_number_of_neurons_per_core("IF_curr_exp", 50)
sim.set_number_of_neurons_per_core("IF_cond_exp", 50)
sim.set_number_of_neurons_per_core("SpikeSourcePoisson", 256)
sim.set_number_of_neurons_per_core("SpikeSourcePoissonVariable", 128)
sim.set_number_of_neurons_per_core("SpikeSourceArray", 256)
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
               'tau_syn_I': 15.0,
               'v_reset': -70.0,
               'v_rest': -70.0,
               'v_thresh': -50.0,
               'e_rev_E': 0.,
               'e_rev_I': -80.
               }
# Check for cached versions

filename = None
adjusted_name = None
if args.filename:
    filename = args.filename
elif args.testing:
    filename = "testing_" + args.testing

if filename:
    if ".npz" in filename:
        adjusted_name = filename
    else:
        adjusted_name = filename + ".npz"
if adjusted_name and os.path.isfile(adjusted_name) and not args.no_cache:
    print("Simulation has been run before & Cached version of results "
          "exists!")
    sys.exit()


# +-------------------------------------------------------------------+
# | Rewiring Parameters                                               |
# +-------------------------------------------------------------------+
no_iterations = args.no_iterations  # iterations
simtime = no_iterations
# Wiring
n = 28
N_layer = n ** 2
S = (n, n)
grid = np.asarray(S)

s_max = args.s_max
sigma_form_forward = args.sigma_form_ff
sigma_form_lateral = args.sigma_form_lat
p_form_lateral = args.p_form_lateral
p_form_forward = args.p_form_forward
p_elim_dep = args.p_elim_dep
p_elim_pot = args.p_elim_pot
f_rew = 10 ** 4  # Hz

# Inputs
f_mean = args.f_mean  # Hz
f_base = 5  # Hz
f_peak = 60  # Hz
sigma_stim = 2  # 2
t_stim = 200  # ms
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
              'f_mean': f_mean,
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
# +-------------------------------------------------------------------+
# | Initial network setup                                             |
# +-------------------------------------------------------------------+


# Connections

stdp_model = sim.STDPMechanism(
    timing_dependence=sim.SpikePairRule(tau_plus=tau_plus,
                                        tau_minus=tau_minus),
    weight_dependence=sim.AdditiveWeightDependence(w_min=0, w_max=g_max,
                                                   A_plus=a_plus,
                                                   A_minus=a_minus)
)

if args.case == CASE_CORR_AND_REW:
    structure_model_w_stdp = sim.StructuralMechanismSTDP(
        stdp_model=stdp_model,
        weight=g_max,
        s_max=s_max,
        grid=grid, f_rew=f_rew,
        lateral_inhibition=args.lateral_inhibition,
        random_partner=args.random_partner,
        p_elim_dep=p_elim_dep,
        p_elim_pot=p_elim_pot,
        sigma_form_forward=sigma_form_forward,
        sigma_form_lateral=sigma_form_lateral,
        p_form_forward=p_form_forward,
        p_form_lateral=p_form_lateral)
elif args.case == CASE_REW_NO_CORR or args.case == CASE_CORR_NO_REW:
    structure_model_w_stdp = sim.StructuralMechanismStatic(
        weight=g_max,
        s_max=s_max,
        grid=grid, f_rew=f_rew,
        lateral_inhibition=args.lateral_inhibition,
        random_partner=args.random_partner,
        p_elim_dep=p_elim_dep,
        p_elim_pot=p_elim_pot,
        sigma_form_forward=sigma_form_forward,
        sigma_form_lateral=sigma_form_lateral,
        p_form_forward=p_form_forward,
        p_form_lateral=p_form_lateral)
else:
    raise ValueError("I don't know what {} is supposed to do.".format(args.case))
# if not testing (i.e. training) construct 10 sources + 10 targets
# grouped into 2 columns
# For each source VRPSS load mnist rates from file
# Use the same initial connectivity for all sets of Populations
randomised_testing_numbers = None

number_of_slots = int(simtime / t_stim)
range_of_slots = np.arange(number_of_slots)
slots_starts = np.ones((N_layer, number_of_slots)) * (range_of_slots * t_stim)
durations = np.ones((N_layer, number_of_slots)) * t_stim

if not args.testing:

    source_column = []
    ff_connections = []
    target_column = []
    lat_connections = []

    init_ff_connections = [(i, j, g_max, args.delay) for i in
                           range(N_layer)
                           for j in range(N_layer) if
                           np.random.rand() < .01]

    init_lat_connections = [(i, j, g_max, args.delay) for i in
                            range(N_layer)
                            for j in range(N_layer) if
                            np.random.rand() < .01]

    for number in range(10):
        # TODO Make input folder variable so that you can swap between averaged and CS
        rates_on, rates_off = load_mnist_rates('mnist_input_rates/averaged/',
                                               number, min_noise=f_mean / 4.,
                                               max_noise=f_mean / 4.,
                                               mean_rate=f_mean)

        # TODO randomise input and allow for arbitrary simulation durations
        # Input population
        source_column.append(
            sim.Population(
                N_layer,
                sim.SpikeSourcePoissonVariable,
                {'rates': rates_on[0:simtime // t_stim, :, :].reshape(simtime // t_stim, N_layer).T,
                 'starts': slots_starts,
                 'durations': durations
                 },
                label="Variable-rate Poisson spike source # " +
                      str(number))
        )

        # Neuron populations
        target_column.append(
            sim.Population(N_layer, model, cell_params,
                           label="TARGET_POP # " + str(number))
        )

        # Setting up connectivity
        ff_connections.append(
            sim.Projection(
                source_column[number], target_column[number],
                sim.FromListConnector(init_ff_connections),
                synapse_dynamics=sim.SynapseDynamics(
                    slow=structure_model_w_stdp),
                label="plastic_ff_projection"
            )
        )

        if args.case != CASE_CORR_NO_REW:
            lat_connections.append(
                sim.Projection(
                    target_column[number], target_column[number],
                    sim.FromListConnector(init_lat_connections),
                    synapse_dynamics=sim.SynapseDynamics(
                        slow=structure_model_w_stdp),
                    label="plastic_lat_projection",
                    target="inhibitory" if args.lateral_inhibition
                    else "excitatory"
                )
            )
    if args.lat_lat_conn:
        for number_pre in range(10):
            for number_post in range(10):
                sim.Projection(
                    target_column[number_pre], target_column[number_post],
                    sim.OneToOneConnector(g_max, args.delay),
                    label="lat_lat_projection"
                          + str(number_pre) + "" + str(number_post),
                    target="inhibitory"
                )

else:
    # Testing mode is activated.
    # 1. Retrieve connectivity for each pair of populations
    # 2. Create a single VRPSS for testing
    # 3. Create a target column
    # 4. Set up connectivity between the source and targets
    # 5. Run for simtime, showing each digit for a 10th of the time, but
    # shuffled randomly. Keep a track of the digits shown!
    init_ff_connections = None
    init_lat_connections = None

    source_column = []
    ff_connections = []
    target_column = []
    lat_connections = []

    if ".npz" in args.testing:
        testing_data = np.load(args.testing)
    else:
        testing_data = np.load(args.testing + ".npz")

    trained_ff_connectivity = testing_data['ff_connections'][-10:]
    trained_lat_connectivity = testing_data['lat_connections'][-10:]

    if not args.random_input:
        randomised_testing_numbers = np.random.randint(0, 10,
                                                       simtime // t_stim)

        # load all rates
        rates = []
        for number in range(10):
            rates_on, rates_off = load_mnist_rates(
                'mnist_input_rates/testing/',
                number, min_noise=f_mean / 4.,
                max_noise=f_mean / 4.,
                mean_rate=f_mean)

            rates.append(rates_on)
    testing_rates = np.empty((simtime // t_stim, grid[0], grid[1]))
    for index in np.arange(testing_rates.shape[0]):
        if not args.random_input:
            testing_rates[index, :, :] = \
                rates[
                    randomised_testing_numbers
                    [index]][np.random.randint(0, rates[randomised_testing_numbers[index]].shape[0]), :, :]
        else:
            break
    if not args.random_input:
        source_pop = sim.Population(
            N_layer,
            sim.SpikeSourcePoissonVariable,
            {
                'rates': testing_rates.reshape(simtime // t_stim, N_layer).T,
                'starts': slots_starts,
                'durations': durations
            },
            label="VRPSS for testing")
    else:
        source_pop = sim.Population(
            N_layer,
            sim.SpikeSourcePoisson,
            {'rate': f_mean,
             'start': 0,
             'duration': simtime,
             },
            label="PSS for testing")

    source_column.append(source_pop)
    for number in range(10):
        # Neuron populations
        target_column.append(
            sim.Population(N_layer, model, cell_params,
                           label="TARGET_POP # " + str(number))
        )

        ff_connections.append(
            sim.Projection(
                source_pop, target_column[number],
                sim.FromListConnector(trained_ff_connectivity[number]),
                label="ff_projection " + str(number)
            )
        )
        if args.case != CASE_CORR_NO_REW:
            lat_connections.append(
                sim.Projection(
                    target_column[number], target_column[number],
                    sim.FromListConnector(
                        trained_lat_connectivity[number]),
                    label="lat_projection " + str(number),
                    target="inhibitory" if args.lateral_inhibition
                    else "excitatory"
                )
            )
    if args.lat_lat_conn:
        for number_pre in range(10):
            for number_post in range(10):
                sim.Projection(
                    target_column[number_pre], target_column[number_post],
                    sim.OneToOneConnector(g_max, args.delay),
                    label="lat_lat_projection"
                          + str(number_pre) + "" + str(number_post),
                    target="inhibitory"
                )

if args.record_source:
    for source_pop in source_column:
        source_pop.record()
for target_pop in target_column:
    target_pop.record()

# Run simulation
pre_spikes = []
post_spikes = []

pre_weights = []
post_weights = []

no_runs = simtime // t_record
run_duration = t_record
e = None
print("Starting the sim")

try:
    for current_run in range(no_runs):
        print("run", current_run + 1, "of", no_runs)
        sim.run(run_duration)

        # Retrieve data if training
        if not args.testing:
            for ff_projection in ff_connections:
                pre_weights.append(
                    np.array([
                        ff_projection._get_synaptic_data(True, 'source'),
                        ff_projection._get_synaptic_data(True, 'target'),
                        ff_projection._get_synaptic_data(True, 'weight'),
                        ff_projection._get_synaptic_data(True, 'delay')]).T)
            if args.case != CASE_CORR_NO_REW:
                for lat_projection in lat_connections:
                    post_weights.append(
                        np.array([
                            lat_projection._get_synaptic_data(True, 'source'),
                            lat_projection._get_synaptic_data(True, 'target'),
                            lat_projection._get_synaptic_data(True, 'weight'),
                            lat_projection._get_synaptic_data(True, 'delay')]).T)

    if args.record_source:
        for source_pop in source_column:
            pre_spikes.append(source_pop.getSpikes(compatible_output=True))
    for target_pop in target_column:
        post_spikes.append(target_pop.getSpikes(compatible_output=True))
    # End simulation on SpiNNaker
    sim.end()
except Exception as e:
    # print(e)
    traceback.print_exc()

end_time = plt.datetime.datetime.now()
total_time = end_time - start_time

print("Total time elapsed -- " + str(total_time))

suffix = end_time.strftime("_%H%M%S_%d%m%Y")

if args.filename:
    filename = args.filename
elif args.testing:
    filename = "testing_" + args.testing
else:
    filename = "mnist_topographic_map_rate_results" + str(suffix)

if e:
    filename = "error_" + filename

np.savez_compressed(filename,
                    pre_spikes=pre_spikes,
                    post_spikes=post_spikes,
                    init_ff_connections=init_ff_connections,
                    init_lat_connections=init_lat_connections,
                    ff_connections=pre_weights,
                    lat_connections=post_weights,
                    final_pre_weights=pre_weights[-10:],
                    final_post_weights=post_weights[-10:],
                    simtime=simtime,
                    sim_params=sim_params,
                    total_time=total_time,
                    testing_numbers=randomised_testing_numbers,
                    testing_file=args.testing, random_input=args.random_input,
                    exception=None)

print("Results in", filename)
print("Total time elapsed -- " + str(total_time))
