import numpy as np
import pylab as plt
import time
from pacman.model.constraints.placer_constraints.placer_chip_and_core_constraint import \
    PlacerChipAndCoreConstraint
import spynnaker7.pyNN as sim

from function_definitions import *
from argparser import *

# SpiNNaker setup
start_time = plt.datetime.datetime.now()


sim.setup(timestep=1.0, min_delay=1.0, max_delay=10)
sim.set_number_of_neurons_per_core("IF_cond_exp", 256 // 10)
sim.set_number_of_neurons_per_core("SpikeSourcePoisson", 256 // 5)
sim.set_number_of_neurons_per_core("SpikeSourceArray", 256 // 5)
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
n = 16
N_layer = n ** 2
# S = (n, n)
S = (1, 256)
grid = np.asarray(S)

s_max = 32
sigma_form_forward = 5
sigma_form_lateral = 1
p_form_lateral = 1
p_form_forward = 0.4
p_elim_dep = 0.0245
p_elim_pot = p_elim_dep # 1.36 * (10 ** -3)
f_rew = 10 ** 4 # Hz

# Inputs
f_mean = args.f_mean  # Hz
f_base = 5  # Hz
f_peak = 60  # Hz
sigma_stim = 10  # 2
t_stim = args.t_stim  # 20  # ms
t_record = args.t_record  # ms

# STDP
a_plus = 0.1
b = 1.0
tau_plus = 20.  # ms
tau_minus = 10.  # ms
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
              'lateral_inhibition':args.lateral_inhibition,
              'b':b
              }

# +-------------------------------------------------------------------+
# | Initial network setup                                             |
# +-------------------------------------------------------------------+




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

if args.case == CASE_CORR_AND_REW:
    structure_model_w_stdp = sim.StructuralMechanism(stdp_model=stdp_model,
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
elif args.case == CASE_CORR_NO_REW:
    structure_model_w_stdp = stdp_model
elif args.case == CASE_REW_NO_CORR:
    structure_model_w_stdp = sim.StructuralMechanism(weight=g_max,
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

# structure_model_w_stdp = sim.StructuralMechanism(weight=g_max, s_max=s_max)
# rates = generate_multimodal_gaussian_rates([[1, 256//4], [1, (3*256)//4]], grid, sigma_stim=2.5, f_peak=args.f_peak)
rates = generate_gaussian_input_rates([1, 256//4], grid, sigma_stim=sigma_stim, f_peak=f_peak)
source_pop = sim.Population(N_layer,
                            sim.SpikeSourcePoisson,
                            {'rate': rates.ravel(),
                             'start': 0,
                             'duration': simtime
                             }, label="Poisson spike source")

ff_prob_conn = [(i, j, g_max, args.delay) for i in range(N_layer) for j in range(N_layer) if np.random.rand() < .05]

ff_projection = sim.Projection(
    source_pop, target_pop,
    # sim.FixedProbabilityConnector(p_connect=.1, weights=g_max),
    sim.FromListConnector(ff_prob_conn),
    synapse_dynamics=sim.SynapseDynamics(slow=structure_model_w_stdp),
    label="plastic_ff_projection"
)


if args.record_source:
    source_pop.record()
target_pop.record()

# Run simulation
pre_spikes = []
post_spikes = []

pre_weights = []
post_weights = []


no_runs = simtime // t_record
run_duration = t_record

for current_run in range(no_runs):
    print "run", current_run + 1, "of", no_runs
    sim.run(run_duration)

    # Retrieve data
    pre_weights.append(
        np.array([
            ff_projection._get_synaptic_data(True, 'source'),
            ff_projection._get_synaptic_data(True, 'target'),
            ff_projection._get_synaptic_data(True, 'weight'),
            ff_projection._get_synaptic_data(True, 'delay')]).T)

if args.record_source:
    pre_spikes = source_pop.getSpikes(compatible_output=True)
post_spikes = target_pop.getSpikes(compatible_output=True)
# End simulation on SpiNNaker
sim.end()

end_time = plt.datetime.datetime.now()
total_time = end_time - start_time

pre_spikes = np.asarray(pre_spikes)
post_spikes = np.asarray(post_spikes)

print "Total time elapsed -- " + str(total_time)

suffix = end_time.strftime("_%H%M%S_%d%m%Y")

if args.filename:
    filename = args.filename
else:
    filename = "1d_topographic_map_results" + str(suffix)

total_target_neuron_mean_spike_rate = \
    post_spikes.shape[0] / float(simtime) * 1000. / N_layer

np.savez(filename, pre_spikes=pre_spikes,
         post_spikes=post_spikes,
         init_ff_connections=ff_prob_conn,
         init_lat_connections=None,
         ff_connections=pre_weights,
         lat_connections=None,
         final_pre_weights=pre_weights[-1],
         final_post_weights=None,
         simtime=simtime,
         sim_params=sim_params,
         total_time=total_time,
         mean_firing_rate=total_target_neuron_mean_spike_rate,
         exception=None)

if args.plot:
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
print "Results in", filename
print "Total time elapsed -- " + str(total_time)