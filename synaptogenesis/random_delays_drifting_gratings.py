"""
Spatio-temporal Selectivity arising from Synaptic Rewiring Procedure

Author: Petrut A. Bogdan

"""
# Imports
import traceback

from gari_function_definitions import *
from function_definitions import *
from argparser import *
import numpy as np
import pylab as plt
import spynnaker7.pyNN as sim
import sys

case = args.case
print("Case", case, "selected!")

# SpiNNaker setup
start_time = plt.datetime.datetime.now()

sim.setup(timestep=1.0, min_delay=1.0, max_delay=15)
sim.set_number_of_neurons_per_core("IF_curr_exp", 50)
sim.set_number_of_neurons_per_core("IF_cond_exp", 50)
sim.set_number_of_neurons_per_core("SpikeSourcePoisson", 256)
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
g_max = args.g_max
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

fsi_cell_params = {'cm': 10.0,  # nF
                   'i_offset': 0.0,
                   'tau_m': 10.0,
                   'tau_refrac': 1.0,
                   # KEEP AN EYE OUT FOR THIS!!!!<-------------
                   'tau_syn_E': 5.0,
                   'tau_syn_I': 5.0,
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

# Different input types
BAR_MOVING = 1
BAR_STATIC = 2
MNIST_MOVING = 3
MNIST_STATIC = 4

input_type = args.input_type

# +-------------------------------------------------------------------+
# | Rewiring Parameters                                               |
# +-------------------------------------------------------------------+
if args.testing:
    no_iterations = args.testing_iterations
else:
    no_iterations = args.no_iterations

simtime = no_iterations
# Wiring
n = args.n
N_layer = n ** 2
S = (n, n)
# S = (256, 1)
grid = np.asarray(S)

s_max = args.s_max
sigma_form_forward = args.sigma_form_ff
sigma_form_lateral = args.sigma_form_lat
p_form_lateral = args.p_form_lateral
p_form_forward = args.p_form_forward
p_elim_dep = args.p_elim_dep * 10.
p_elim_pot = args.p_elim_pot / 10.
f_rew = args.f_rew  # 10 ** 4  # Hz

# Inputs
f_mean = args.f_mean  # Hz
f_base = args.f_base  # Hz
f_peak = args.f_peak  # 152.8  # Hz
sigma_stim = args.sigma_stim  # 2
t_stim = args.t_stim  # 20  # ms
t_record = args.t_record if args.t_record <= args.no_iterations else \
    args.no_iterations
# ms

local_connection_delay_dist = [1, 3]

# STDP
a_plus = 0.1
b = args.b
tau_plus = 20.  # ms
tau_minus = args.t_minus  # ms
a_minus = (a_plus * tau_plus * b) / tau_minus
# a_minus = 0.0375


if args.constant_delay:
    delay_interval = [1, 1]
else:
    delay_interval = [1, 16]

input_grating_fname = None

inh_sources = []
inh_targets = []
inh_weights = []
inh_delays = []

inh_inh_sources = []
inh_inh_targets = []
inh_inh_weights = []
inh_inh_delays = []

exh_sources = []
exh_targets = []
exh_weights = []
exh_delays = []

on_inh_weights = []
off_inh_weights = []
noise_inh_weights = []

if args.training_angles and not args.all_angles:
    training_angles = args.training_angles
elif args.training_angles == [0] and args.all_angles:
    training_angles = np.arange(0, 360, 5)
else:
    raise AttributeError("Can't have both a selection of angles and all "
                         "angles at the same time!")

if not args.chunk_size:
    if n == 32:
        chunk = 200
    elif n == 64:
        chunk = 400
    else:
        raise AttributeError("What do I do for the specified grid size?")
else:
    chunk = args.chunk_size

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
              'f_rew_exc': args.f_rew_exc,
              'f_rew_inh': args.f_rew_inh,
              'lateral_inhibition': args.lateral_inhibition,
              'delay': args.delay,
              'b': b,
              't_minus': tau_minus,
              't_plus': tau_plus,
              'tau_refrac': args.tau_refrac,
              'a_minus': a_minus,
              'a_plus': a_plus,
              'input_type': input_type,
              'random_partner': args.random_partner,
              'lesion': args.lesion,
              'delay_interval': delay_interval,
              'topology': args.topology,
              'constant_delay': args.constant_delay,
              'training_angles': training_angles,
              'argparser': vars(args),
              'chunk': chunk,
              'no_off_polarity': args.no_off_polarity,
              'coplanar': args.coplanar,
              'record_exc_v': args.record_exc_v
              }

if args.input_type == GAUSSIAN_INPUT:
    print("Drifting grating")
    gen_rate = split_in_spikes
else:
    print("Gaussian input")
    gen_rate = generate_gaussian_input_rates

# +-------------------------------------------------------------------+
# | Initial network setup                                             |
# +-------------------------------------------------------------------+
# Need to setup the moving input

actual_angles = []
number_of_slots = int(simtime / chunk)
range_of_slots = np.arange(number_of_slots)
slots_starts = np.ones((N_layer, number_of_slots)) * (range_of_slots * chunk)
durations = np.ones((N_layer, number_of_slots)) * chunk

if not args.testing:
    # TRAINING REGIME!
    # the following loads ALL of the spikes in Memory! This is expensive for long simulations
    # if input_type == MNIST_STATIC:
    #     on_rates = {}
    #     off_rates = {}
    #     for number in training_angles:
    #         rates_on, rates_off = load_mnist_rates('mnist_input_rates/averaged/',
    #                                                number,
    #                                                # min_noise=f_mean / 4., max_noise=f_mean / 4.,
    #                                                min_noise=0, max_noise=0,
    #                                                mean_rate=f_mean)
    #         rates_on = rates_on.reshape(rates_on.shape[0], N_layer).T
    #         on_rates[number] = rates_on
    #         rates_off = rates_off.reshape(rates_off.shape[0], N_layer).T
    #         off_rates[number] = rates_off
    #     randomised_testing_numbers = np.random.choice(training_angles, number_of_slots, replace=True)
    # elif input_type == BAR_STATIC:
    #     aa, final_on_gratings, final_off_gratings = \
    #         generate_bar_input(no_iterations, chunk, N_layer,
    #                            angles=training_angles)
    #     actual_angles.append(aa)
    #
    # else:
    aa, final_on_gratings, final_off_gratings = \
        generate_bar_input(no_iterations, chunk, N_layer,
                           angles=training_angles)
    actual_angles.append(aa)

    # Add +-1 ms to all times in input
    if args.jitter:
        final_on_gratings, final_off_gratings = jitter_the_input(
            final_on_gratings, final_off_gratings)
    source_pop = sim.Population(N_layer,
                                sim.SpikeSourceArray,
                                {'spike_times': final_on_gratings
                                 }, label="Moving grating on population")
    if args.no_off_polarity:
        print("No off polarity will be injected into the system.")
        source_pop_off = sim.Population(N_layer,
                                        sim.SpikeSourceArray,
                                        {'spike_times': []},
                                        label="(No) Moving grating off population")
    else:
        source_pop_off = sim.Population(N_layer,
                                        sim.SpikeSourceArray,
                                        {'spike_times': final_off_gratings
                                         }, label="Moving grating off population")

    if np.isclose(f_base, 0):
        print("No noise will be injected into the system.")
        noise_pop = sim.Population(N_layer,
                                   sim.SpikeSourceArray,
                                   {'spike_times': []},
                                   label="(No) Noise population")
    else:
        noise_pop = sim.Population(N_layer,
                                   sim.SpikeSourcePoisson,
                                   {'rate': f_base,
                                    'start': 0,
                                    'duration': simtime},
                                   label="Noise population")
else:
    # TESTING REGIME!
    input_grating_fname = "compressed_spiking_moving_bar_input/" \
                          "spiking_moving_bar_motif_bank_simtime_" \
                          "{}x{}_{}s.npz".format(n, n, no_iterations // 1000)
    data = np.load(input_grating_fname)
    try:
        actual_angles = data['actual_angles']
    except:
        print("Can't load actual angles. Did the name change?")
    on_spikes = data['on_spikes']
    final_on_gratings = on_spikes

    off_spikes = data['off_spikes']
    final_off_gratings = off_spikes

    # Add +-1 ms to all times in input
    if args.jitter:
        final_on_gratings, final_off_gratings = jitter_the_input(
            on_spikes, off_spikes)

    source_pop = sim.Population(N_layer,
                                sim.SpikeSourceArray,
                                {'spike_times': final_on_gratings
                                 }, label="Moving grating on population")

    if args.no_off_polarity:
        print("No off polarity will be injected into the system.")
        source_pop_off = sim.Population(N_layer,
                                        sim.SpikeSourceArray,
                                        {'spike_times': []},
                                        label="(No) Moving grating off population")
    else:
        source_pop_off = sim.Population(N_layer,
                                        sim.SpikeSourceArray,
                                        {'spike_times': final_off_gratings
                                         },
                                        label="Moving grating off population")
    if np.isclose(f_base, 0):
        print("No noise will be injected into the system.")
        noise_pop = sim.Population(N_layer,
                                   sim.SpikeSourceArray,
                                   {'spike_times': []},
                                   label="(No) Noise population")
    else:
        noise_pop = sim.Population(N_layer,
                                   sim.SpikeSourcePoisson,
                                   {'rate': f_base,
                                    'start': 0,
                                    'duration': simtime},
                                   label="Noise population")

ff_s = np.zeros(N_layer, dtype=np.uint)
lat_s = np.zeros(N_layer, dtype=np.uint)

init_ff_connections = []
init_lat_connections = []
# Neuron populations
target_pop = sim.Population(N_layer, model, cell_params, label="TARGET_POP")
if args.topology != 1:
    inh_pop = sim.Population(N_layer, model, cell_params,
                             label="INH_POP")
# Putting this populations on chip 0 1 makes it easier to copy the provenance
# data somewhere else
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
    structure_model_w_stdp = sim.StructuralMechanismSTDP(
        stdp_model=stdp_model,
        weight=g_max,
        delay=delay_interval,
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
    if args.common_rewiring_seed:
        inh_structure_model_w_stdp = structure_model_w_stdp
    else:
        inh_structure_model_w_stdp = sim.StructuralMechanismSTDP(
            stdp_model=stdp_model,
            weight=g_max,
            delay=delay_interval,
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

if not args.testing:
    print("No insults")
    ff_projection = sim.Projection(
        source_pop, target_pop,
        sim.FixedProbabilityConnector(0.),
        # sim.FromListConnector(init_ff_connections),
        synapse_dynamics=sim.SynapseDynamics(slow=structure_model_w_stdp),
        label="plastic_ff_projection"
    )

    ff_off_projection = sim.Projection(
        source_pop_off, target_pop,
        sim.FixedProbabilityConnector(0.),
        synapse_dynamics=sim.SynapseDynamics(slow=structure_model_w_stdp),
        label="ff_off_projection"
    )

    noise_projection = sim.Projection(
        noise_pop, target_pop,
        sim.FixedProbabilityConnector(0.),
        synapse_dynamics=sim.SynapseDynamics(slow=structure_model_w_stdp),
        label="noise_projection"
    )

    lat_projection = sim.Projection(
        target_pop, target_pop,

        sim.FixedProbabilityConnector(0.),
        # sim.FromListConnector(init_lat_connections),
        synapse_dynamics=sim.SynapseDynamics(slow=structure_model_w_stdp),
        label="plastic_lat_projection",
        target="inhibitory" if args.lateral_inhibition else "excitatory"
    )
    if args.coplanar:
        structure_model_w_stdp.set_projection_parameter(
            lat_projection,
            sim.StructuralMechanismSTDP.connectivity_exception_param.delay,
            local_connection_delay_dist)
    if args.topology == 0:
        inh_weights = generate_initial_connectivity(5, p_form_forward,
                                                    "inh ff weights ...",
                                                    N_layer=N_layer, n=n,
                                                    s_max=16, g_max=.1,
                                                    delay=1.)
        inh_inh_weights = generate_initial_connectivity(5, p_form_forward,
                                                        "inh ff weights ...",
                                                        N_layer=N_layer, n=n,
                                                        s_max=16, g_max=.1,
                                                        delay=1.)
        exh_weights = generate_initial_connectivity(5, p_form_forward,
                                                    "inh ff weights ...",
                                                    N_layer=N_layer, n=n,
                                                    s_max=16, g_max=.1,
                                                    delay=1.)
        inh_projection = sim.Projection(
            inh_pop, target_pop,
            sim.FromListConnector(inh_weights),
            label="static_inh_lat_projection",
            target="inhibitory"
        )
        inh_inh_projection = sim.Projection(
            inh_pop, inh_pop,
            sim.FromListConnector(inh_inh_weights),
            label="static_inh_inh_projection",
            target="inhibitory"
        )
        if args.coplanar:
            structure_model_w_stdp.set_projection_parameter(
                inh_inh_projection,
                sim.StructuralMechanismSTDP.connectivity_exception_param.delay,
                local_connection_delay_dist)
        exh_projection = sim.Projection(
            target_pop, inh_pop,
            sim.FromListConnector(exh_weights),
            label="static_exh_lat_projection",
            target="excitatory"
        )

    if args.topology == 2 or args.topology == 3:
        inh_projection = sim.Projection(
            inh_pop, target_pop,
            sim.FixedProbabilityConnector(0.),
            synapse_dynamics=sim.SynapseDynamics(slow=structure_model_w_stdp),
            label="plastic_inh_lat_projection",
            target="inhibitory"
        )
        if args.coplanar:
            structure_model_w_stdp.set_projection_parameter(
                inh_projection,
                sim.StructuralMechanismSTDP.connectivity_exception_param.delay,
                local_connection_delay_dist)
        inh_inh_projection = sim.Projection(
            inh_pop, inh_pop,
            sim.FixedProbabilityConnector(0.),
            synapse_dynamics=sim.SynapseDynamics(slow=inh_structure_model_w_stdp),
            label="plastic_inh_inh_projection",
            target="inhibitory"
        )
        if args.coplanar:
            structure_model_w_stdp.set_projection_parameter(
                inh_inh_projection,
                sim.StructuralMechanismSTDP.connectivity_exception_param.delay,
                local_connection_delay_dist)
        exh_projection = sim.Projection(
            target_pop, inh_pop,
            sim.FixedProbabilityConnector(0.),
            synapse_dynamics=sim.SynapseDynamics(slow=inh_structure_model_w_stdp),
            label="plastic_exh_lat_projection",
            target="excitatory"
        )
        if args.coplanar:
            structure_model_w_stdp.set_projection_parameter(
                exh_projection,
                sim.StructuralMechanismSTDP.connectivity_exception_param.delay,
                local_connection_delay_dist)
    if args.topology == 3:
        ff_inh_projection = sim.Projection(
            source_pop, inh_pop,
            sim.FixedProbabilityConnector(0.),
            synapse_dynamics=sim.SynapseDynamics(slow=inh_structure_model_w_stdp),
            label="plastic_ff_inh_projection"
        )

        ff_off_inh_projection = sim.Projection(
            source_pop_off, inh_pop,
            sim.FixedProbabilityConnector(0.),
            synapse_dynamics=sim.SynapseDynamics(slow=inh_structure_model_w_stdp),
            label="ff_off_inh_projection"
        )

        noise_inh_projection = sim.Projection(
            noise_pop, inh_pop,
            sim.FixedProbabilityConnector(0.),
            synapse_dynamics=sim.SynapseDynamics(slow=inh_structure_model_w_stdp),
            label="noise_inh_projection"
        )

else:
    data_file_name = args.testing
    if ".npz" not in args.testing:
        data_file_name += ".npz"
    testing_data = np.load(data_file_name)
    trained_ff_on_connectivity = testing_data['ff_connections'][-1]
    trained_ff_off_connectivity = testing_data['ff_off_connections'][-1]
    trained_lat_connectivity = testing_data['lat_connections'][-1]
    trained_noise_connectivity = testing_data['noise_connections'][-1]
    if args.topology != 1:
        trained_inh_lat_connectivity = testing_data['inh_connections']
        trained_exh_lat_connectivity = testing_data['exh_connections']
        trained_inh_inh_connectivity = testing_data['inh_inh_connections']
    if args.topology == 3:
        trained_on_inh_connectivity = testing_data['on_inh_connections']
        trained_off_inh_connectivity = testing_data['off_inh_connections']
        trained_noise_inh_connectivity = testing_data['noise_inh_connections']

    print("TESTING PHASE")
    ff_projection = sim.Projection(
        source_pop, target_pop,
        sim.FromListConnector(trained_ff_on_connectivity),
        label="plastic_ff_projection"
    )

    if trained_noise_connectivity.size == 0:
        ff_off_projection = sim.Projection(
            source_pop_off, target_pop,
            sim.FixedProbabilityConnector(0),
            label="ff_off_projection"
        )
    else:
        ff_off_projection = sim.Projection(
            source_pop_off, target_pop,
            sim.FromListConnector(trained_ff_off_connectivity),
            label="ff_off_projection"
        )

    if trained_noise_connectivity.size == 0:
        noise_projection = sim.Projection(
            noise_pop, target_pop,
            sim.FixedProbabilityConnector(0),
            label="noise_projection"
        )
    else:
        noise_projection = sim.Projection(
            noise_pop, target_pop,
            sim.FromListConnector(trained_noise_connectivity),
            label="noise_projection"
        )

    lat_projection = sim.Projection(
        target_pop, target_pop,
        sim.FromListConnector(trained_lat_connectivity),
        label="plastic_lat_projection",
        target="inhibitory" if args.lateral_inhibition else "excitatory"
    )

    if args.topology != 1:
        inh_projection = sim.Projection(
            inh_pop, target_pop,
            sim.FromListConnector(trained_inh_lat_connectivity),
            label="plastic_inh_lat_projection",
            target="inhibitory"
        )
        inh_inh_projection = sim.Projection(
            inh_pop, inh_pop,
            sim.FromListConnector(trained_inh_inh_connectivity),
            label="plastic_inh_inh_projection",
            target="inhibitory"
        )
        exh_projection = sim.Projection(
            target_pop, inh_pop,
            sim.FromListConnector(trained_exh_lat_connectivity),
            label="plastic_exh_lat_projection",
            target="excitatory"
        )
    if args.topology == 3:
        ff_inh_projection = sim.Projection(
            source_pop, inh_pop,
            sim.FromListConnector(trained_on_inh_connectivity),
            label="plastic_ff_inh_projection"
        )

        ff_off_inh_projection = sim.Projection(
            source_pop_off, inh_pop,
            sim.FromListConnector(trained_off_inh_connectivity),
            label="ff_off_inh_projection"
        )
        if trained_noise_inh_connectivity.size == 0:
            noise_inh_projection = sim.Projection(
                noise_pop, inh_pop,
                sim.FixedProbabilityConnector(0),
                label="noise_inh_projection"
            )
        else:
            noise_inh_projection = sim.Projection(
                noise_pop, inh_pop,
                sim.FromListConnector(trained_noise_inh_connectivity),
                label="noise_inh_projection"
            )

# +-------------------------------------------------------------------+
# | Simulation and results                                            |
# +-------------------------------------------------------------------+

# Record neurons' potentials
if args.record_exc_v:
    target_pop.record_v()

# Record spikes
# if case == CASE_REW_NO_CORR:
if args.record_source:
    source_pop.record()
    source_pop_off.record()
    noise_pop.record()

if args.topology != 1 and args.record_inh:
    inh_pop.record()

if args.testing:
    target_pop.record()

# Run simulation
pre_spikes = []
pre_off_spikes = []
pre_noise_spikes = []
post_spikes = []
inh_post_spikes = []

pre_sources = []
pre_targets = []
pre_weights = []
pre_delays = []

pre_off_sources = []
pre_off_targets = []
pre_off_weights = []
pre_off_delays = []

noise_sources = []
noise_targets = []
noise_weights = []
noise_delays = []

post_sources = []
post_targets = []
post_weights = []
post_delays = []

exc_v_recording = []

# rates_history = np.zeros((16, 16, simtime // t_stim))
e = None
print("Starting the sim")

no_runs = simtime // t_record
run_duration = t_record

try:
    for current_run in range(no_runs):
        print("run", current_run + 1, "of", no_runs)
        sim.run(run_duration)

        # generate spikes depending on whether we're training or testing
        # load data
        # if not args.testing:
        #     aa, final_on_gratings, final_off_gratings = \
        #         generate_bar_input(t_record, chunk, N_layer,
        #                            angles=training_angles,
        #                            offset=current_run * run_duration)
        #
        #     actual_angles.append(aa)
        #     # Add +-1 ms to all times in input
        #     if args.jitter:
        #         final_on_gratings, final_off_gratings = jitter_the_input(
        #             final_on_gratings, final_off_gratings)
        #     source_pop.tset("spike_times", final_on_gratings)
        #     source_pop_off.tset("spike_times", final_off_gratings)

    if not args.testing:
        pre_weights.append(
            np.array([
                ff_projection._get_synaptic_data(True, 'source'),
                ff_projection._get_synaptic_data(True, 'target'),
                ff_projection._get_synaptic_data(True, 'weight'),
                ff_projection._get_synaptic_data(True, 'delay')]).T)
        pre_off_weights.append(
            np.array([
                ff_off_projection._get_synaptic_data(True, 'source'),
                ff_off_projection._get_synaptic_data(True, 'target'),
                ff_off_projection._get_synaptic_data(True, 'weight'),
                ff_off_projection._get_synaptic_data(True, 'delay')]).T)

        noise_weights.append(
            np.array([
                noise_projection._get_synaptic_data(True, 'source'),
                noise_projection._get_synaptic_data(True, 'target'),
                noise_projection._get_synaptic_data(True, 'weight'),
                noise_projection._get_synaptic_data(True, 'delay')]).T)

        post_weights.append(
            np.array([
                lat_projection._get_synaptic_data(True, 'source'),
                lat_projection._get_synaptic_data(True, 'target'),
                lat_projection._get_synaptic_data(True, 'weight'),
                lat_projection._get_synaptic_data(True, 'delay')]).T)

        if args.topology == 2 or args.topology == 3:
            inh_weights = \
                np.array([
                    inh_projection._get_synaptic_data(True, 'source'),
                    inh_projection._get_synaptic_data(True, 'target'),
                    inh_projection._get_synaptic_data(True, 'weight'),
                    inh_projection._get_synaptic_data(True, 'delay')]).T

            inh_inh_weights = \
                np.array([
                    inh_inh_projection._get_synaptic_data(True, 'source'),
                    inh_inh_projection._get_synaptic_data(True, 'target'),
                    inh_inh_projection._get_synaptic_data(True, 'weight'),
                    inh_inh_projection._get_synaptic_data(True, 'delay')]).T

            exh_weights = \
                np.array([
                    exh_projection._get_synaptic_data(True, 'source'),
                    exh_projection._get_synaptic_data(True, 'target'),
                    exh_projection._get_synaptic_data(True, 'weight'),
                    exh_projection._get_synaptic_data(True, 'delay')]).T
        if args.topology == 3:
            on_inh_weights = \
                np.array([
                    ff_inh_projection._get_synaptic_data(True, 'source'),
                    ff_inh_projection._get_synaptic_data(True, 'target'),
                    ff_inh_projection._get_synaptic_data(True, 'weight'),
                    ff_inh_projection._get_synaptic_data(True, 'delay')]).T
            off_inh_weights = \
                np.array([
                    ff_off_inh_projection._get_synaptic_data(True, 'source'),
                    ff_off_inh_projection._get_synaptic_data(True, 'target'),
                    ff_off_inh_projection._get_synaptic_data(True, 'weight'),
                    ff_off_inh_projection._get_synaptic_data(True, 'delay')]).T

            noise_inh_weights = \
                np.array([
                    noise_inh_projection._get_synaptic_data(True, 'source'),
                    noise_inh_projection._get_synaptic_data(True, 'target'),
                    noise_inh_projection._get_synaptic_data(True, 'weight'),
                    noise_inh_projection._get_synaptic_data(True, 'delay')]).T

    if args.record_source:
        pre_spikes = source_pop.getSpikes(compatible_output=True)
        pre_off_spikes = source_pop_off.getSpikes(compatible_output=True)
        pre_noise_spikes = noise_pop.getSpikes(compatible_output=True)
    else:
        pre_spikes = []

    if args.testing:
        post_spikes = target_pop.getSpikes(compatible_output=True)
    else:
        post_spikes = []

    if args.topology != 1 and args.record_inh and args.testing:
        inh_post_spikes = inh_pop.getSpikes(compatible_output=True)
    else:
        inh_post_spikes = []
    # End simulation on SpiNNaker
    if args.record_exc_v:
        exc_v_recording = target_pop.get_v(compatible_output=True)
    sim.end()
except Exception as e:
    # print(e)
    traceback.print_exc()
# print("Weights:", plastic_projection.getWeights())
end_time = plt.datetime.datetime.now()
total_time = end_time - start_time

pre_spikes = np.asarray(pre_spikes)
post_spikes = np.asarray(post_spikes)
inh_post_spikes = np.asarray(inh_post_spikes)

print("Total time elapsed -- " + str(total_time))

suffix = end_time.strftime("_%H%M%S_%d%m%Y")

if args.filename:
    filename = args.filename
elif args.testing:
    filename = "testing_" + args.testing
else:
    filename = "drifting_grating_topographic_map_results" + str(suffix)

if e:
    filename = "error_" + filename
# total_target_neuron_mean_spike_rate = \
#     post_spikes.shape[0] / float(simtime) * 1000. / N_layer

np.savez_compressed(filename,
                    # Source Spike recordings
                    pre_spikes=pre_spikes,
                    pre_off_spikes=pre_off_spikes,
                    pre_noise_spikes=pre_noise_spikes,

                    # Post-synaptic (target population) Spike recordings
                    post_spikes=post_spikes,
                    inh_post_spikes=inh_post_spikes,

                    init_ff_connections=init_ff_connections,
                    init_lat_connections=init_lat_connections,

                    ff_connections=pre_weights,
                    lat_connections=post_weights,
                    ff_off_connections=pre_off_weights,
                    noise_connections=noise_weights,

                    on_inh_connections=on_inh_weights,
                    off_inh_connections=off_inh_weights,
                    noise_inh_connections=noise_inh_weights,

                    inh_connections=inh_weights,
                    inh_inh_connections=inh_inh_weights,
                    exh_connections=exh_weights,

                    simtime=simtime,
                    sim_params=sim_params,
                    total_time=total_time,
                    mean_firing_rate=None,
                    exception=str(e),
                    insult=args.lesion,
                    input_type=args.input_type,
                    testing=args.testing,
                    # final_on_gratings=final_on_gratings,
                    # final_off_gratings=final_off_gratings,
                    input_grating_fname=input_grating_fname,
                    actual_angles=actual_angles,

                    topology=args.topology,
                    training_angles=training_angles,

                    exc_v_recording=exc_v_recording
                    )

print("Results in", filename)
print("Total time elapsed -- " + str(total_time))
