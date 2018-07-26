"""
Test for topographic map formation using STDP and synaptic rewiring.

http://hdl.handle.net/1842/3997
"""
# Imports
import traceback

from gari_function_definitions import *
from function_definitions import *
from argparser import *

import numpy as np
import pylab as plt

import spynnaker7.pyNN as sim

case = args.case
print("Case", case, "selected!")

# SpiNNaker setup
start_time = plt.datetime.datetime.now()

sim.setup(timestep=1.0, min_delay=1.0, max_delay=10)
sim.set_number_of_neurons_per_core("IF_curr_exp", 50)
sim.set_number_of_neurons_per_core("IF_cond_exp", 50)
# sim.set_number_of_neurons_per_core("SpikeSourcePoisson", 256 // 13)
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
sigma_form_forward = args.sigma_form_ff
sigma_form_lateral = args.sigma_form_lat
p_form_lateral = args.p_form_lateral
p_form_forward = args.p_form_forward
p_elim_dep = args.p_elim_dep * 10.
p_elim_pot = args.p_elim_pot / 10.
f_rew = args.f_rew  # 10 ** 4  # Hz

# Inputs
f_mean = args.f_mean  # Hz
f_base = 5  # Hz
f_peak = args.f_peak  # 152.8  # Hz
sigma_stim = args.sigma_stim  # 2
t_stim = args.t_stim  # 20  # ms
t_record = args.t_record if args.t_record <= args.no_iterations else \
    args.no_iterations
# ms

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
              'delay': args.delay_distribution,
              'b': b,
              't_minus': tau_minus,
              't_plus': tau_plus,
              'tau_refrac': args.tau_refrac,
              'a_minus': a_minus,
              'a_plus': a_plus,
              'input_type': args.input_type,
              'random_partner': args.random_partner,
              'lesion': args.lesion,
              'delay_interval': delay_interval
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

training_actual_angles = []

if not args.testing:
    training_actual_angles, final_on_gratings, final_off_gratings = \
        generate_bar_input(simtime, 200, N_layer, angles=args.training_angles)

    # Add +-1 ms to all times in input
    # final_on_gratings = []
    # for row in on_gratings:
    #     row = np.asarray(row)
    #     final_on_gratings.append(row + np.random.randint(-1, 2,
    #                                                      size=row.shape))
    #
    # final_off_gratings = []
    # for row in off_gratings:
    #     row = np.asarray(row)
    #     final_off_gratings.append(row + np.random.randint(-1, 2,
    #                                                       size=row.shape))

    source_pop = sim.Population(N_layer,
                                sim.SpikeSourceArray,
                                {'spike_times': final_on_gratings
                                 }, label="Moving grating on population")

    source_pop_off = sim.Population(N_layer,
                                    sim.SpikeSourceArray,
                                    {'spike_times': final_off_gratings
                                     }, label="Moving grating off population")

    noise_pop = sim.Population(N_layer,
                               sim.SpikeSourcePoisson,
                               {'rate': f_base,
                                'start': 0,
                                'duration': simtime},
                               label="Noise population")
else:
    input_grating_fname = "spiking_moving_bar_input/" \
                          "spiking_moving_bar_motif_bank_simtime_{" \
                          "}s.npz".format(no_iterations // 1000)
    data = np.load(input_grating_fname)
    on_spikes = data['on_spikes']
    final_on_gratings = []
    for row in on_spikes:
        row = np.asarray(row)
        final_on_gratings.append(row + np.random.randint(-1, 2,
                                                         size=row.shape))

    final_off_gratings = []
    off_spikes = data['off_spikes']
    for row in off_spikes:
        row = np.asarray(row)
        final_off_gratings.append(row + np.random.randint(-1, 2,
                                                          size=row.shape))

    source_pop = sim.Population(N_layer,
                                sim.SpikeSourceArray,
                                {'spike_times': final_on_gratings
                                 }, label="Moving grating on population")

    source_pop_off = sim.Population(N_layer,
                                    sim.SpikeSourceArray,
                                    {'spike_times': final_off_gratings
                                     },
                                    label="Moving grating off population")
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
if args.topology == 0:
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
        exh_projection = sim.Projection(
            target_pop, inh_pop,
            sim.FromListConnector(exh_weights),
            label="static_exh_lat_projection",
            target="excitatory"
        )

    if args.topology == 2:
        inh_projection = sim.Projection(
            inh_pop, target_pop,
            sim.FixedProbabilityConnector(0.),
            synapse_dynamics=sim.SynapseDynamics(slow=structure_model_w_stdp),
            label="plastic_inh_lat_projection",
            target="inhibitory"
        )
        inh_inh_projection = sim.Projection(
            inh_pop, inh_pop,
            sim.FixedProbabilityConnector(0.),
            synapse_dynamics=sim.SynapseDynamics(slow=structure_model_w_stdp),
            label="plastic_inh_inh_projection",
            target="inhibitory"
        )
        exh_projection = sim.Projection(
            target_pop, inh_pop,
            sim.FixedProbabilityConnector(0.),
            synapse_dynamics=sim.SynapseDynamics(slow=structure_model_w_stdp),
            label="plastic_exh_lat_projection",
            target="excitatory"
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
    if args.topology == 0 or args.topology == 2:
        trained_inh_lat_connectivity = testing_data['inh_connections']
        trained_exh_lat_connectivity = testing_data['exh_connections']
        trained_inh_inh_connectivity = testing_data['inh_inh_connections']
    print("TESTING PHASE")
    ff_projection = sim.Projection(
        source_pop, target_pop,
        sim.FromListConnector(trained_ff_on_connectivity),
        label="plastic_ff_projection"
    )

    ff_off_projection = sim.Projection(
        source_pop_off, target_pop,
        sim.FromListConnector(trained_ff_off_connectivity),
        label="ff_off_projection"
    )

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

    if args.topology == 0 or args.topology == 2:
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

# rates_history = np.zeros((16, 16, simtime // t_stim))
e = None
print("Starting the sim")

no_runs = simtime // t_record
run_duration = t_record

try:
    for current_run in range(no_runs):
        print("run", current_run + 1, "of", no_runs)
        sim.run(run_duration)

    if not args.testing:
        pre_weights.append(
            np.array([
                ff_projection._get_synaptic_data(True, 'source'),
                ff_projection._get_synaptic_data(True, 'target'),
                ff_projection._get_synaptic_data(True, 'weight'),
                ff_projection._get_synaptic_data(True,
                                                 'delay')]).T)
        pre_off_weights.append(
            np.array([
                ff_off_projection._get_synaptic_data(True, 'source'),
                ff_off_projection._get_synaptic_data(True, 'target'),
                ff_off_projection._get_synaptic_data(True, 'weight'),
                ff_off_projection._get_synaptic_data(True,
                                                     'delay')]).T)

        noise_weights.append(
            np.array([
                noise_projection._get_synaptic_data(True, 'source'),
                noise_projection._get_synaptic_data(True, 'target'),
                noise_projection._get_synaptic_data(True, 'weight'),
                noise_projection._get_synaptic_data(True,
                                                    'delay')]).T)

        post_weights.append(
            np.array([
                lat_projection._get_synaptic_data(True, 'source'),
                lat_projection._get_synaptic_data(True, 'target'),
                lat_projection._get_synaptic_data(True, 'weight'),
                lat_projection._get_synaptic_data(True,
                                                  'delay')]).T)

        if args.topology == 2:
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
    if args.record_source:
        pre_spikes = source_pop.getSpikes(compatible_output=True)
    else:
        pre_spikes = []
    post_spikes = target_pop.getSpikes(compatible_output=True)
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
elif args.testing:
    filename = "testing_" + args.testing
else:
    filename = "drifting_grating_topographic_map_results" + str(suffix)

# total_target_neuron_mean_spike_rate = \
#     post_spikes.shape[0] / float(simtime) * 1000. / N_layer

np.savez(filename, pre_spikes=pre_spikes,
         post_spikes=post_spikes,
         init_ff_connections=init_ff_connections,
         init_lat_connections=init_lat_connections,
         ff_connections=pre_weights,
         lat_connections=post_weights,
         ff_off_connections=pre_off_weights,
         noise_connections=noise_weights,
         inh_connections=inh_weights,
         inh_inh_connections=inh_inh_weights,
         exh_connections=exh_weights,
         simtime=simtime,
         sim_params=sim_params,
         total_time=total_time,
         mean_firing_rate=None,
         exception=e,
         insult=args.lesion,
         input_type=args.input_type,
         testing=args.testing,
         final_on_gratings=final_on_gratings,
         final_off_gratings=final_off_gratings,
         input_grating_fname=input_grating_fname,
         training_actual_angles=training_actual_angles
         )

print("Results in", filename)
print("Total time elapsed -- " + str(total_time))
