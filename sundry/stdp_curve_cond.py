import pylab as plt

try:
    import pyNN.spiNNaker as sim
except Exception as e:
    import spynnaker7.pyNN as sim

import numpy as np

# ------------------------------------------------------------------
# This example uses the sPyNNaker implementation of pair-based STDP
# To reproduce the eponymous STDP curve first
# Plotted by Bi and Poo (1998)
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Common parameters
# ------------------------------------------------------------------
start_datetime = plt.datetime.datetime.now()
time_between_pairs = 1400
num_pairs = 60
start_w = 0.005
delta_t = [-100, -60, -40, -30, -20, -10, -1, 0, 1, 10, 20, 30, 40, 60, 80, 100, 120, 140, 160, 180, 220, 260]
# delta_t = np.linspace(-100, 260, 20)
start_time = 200
# mad = True
mad = 3

# Original values
w_min = 0.0
w_max = 0.0175

tau_plus = 16.7
tau_minus = 33.7
A_plus = 0.005
A_minus = 0.005

# Values used in my simulations

# b = 1.2
#
# A_plus = 0.1
tau_plus = 20.  # ms
tau_minus = 64.  # ms
#
# w_max = 0.2
# A_minus = (A_plus * tau_plus * b) / tau_minus

# Population parameters
model = sim.IF_cond_exp

# original cell_params
cell_params = {'cm': 0.25,  # nF
               'i_offset': 0.0,
               'tau_m': 10.0,
               'tau_refrac': 2.0,
               'tau_syn_E': 2.5,
               'tau_syn_I': 2.5,
               'v_reset': -70.0,
               'v_rest': -65.0,
               'v_thresh': -55.4
               }

# # cell_params used in examples
# cell_params = {'cm': 20.0,  # nF
#                'i_offset': 0.0,
#                'tau_m': 20.0,
#                'tau_refrac': 5.0,
#                'tau_syn_E': 5.0,
#                'tau_syn_I': 5.0,
#                'v_reset': -70.0,
#                'v_rest': -70.0,
#                'v_thresh': -50.0,
#                'e_rev_E': 0.,
#                'e_rev_I': -80.
#                }


# SpiNNaker setup
sim.setup(timestep=1.0, min_delay=1.0, max_delay=10.0)

# -------------------------------------------------------------------
# Experiment loop
# -------------------------------------------------------------------
projections = []
sim_time = 0
for t in delta_t:
    # Calculate phase of input spike trains
    # If M.A.D., take into account dendritic delay
    if mad == 1:
        # Pre after post
        if t > 0:
            post_phase = 0
            pre_phase = t + 1
        # Post after pre
        else:
            post_phase = -t
            pre_phase = 1
    # Otherwise, take into account axonal delay
    elif mad == 2:
        # Pre after post
        if t > 0:
            post_phase = 1
            pre_phase = t
        # Post after pre
        else:
            post_phase = 1 - t
            pre_phase = 0
    elif mad == 3:
        # Pre after post
        if t > 0:
            post_phase = 0
            pre_phase = t
        # Post after pre
        else:
            post_phase = - t
            pre_phase = 0


    sim_time = max(sim_time, (num_pairs * time_between_pairs) + abs(t))

    # Neuron populations
    pre_pop = sim.Population(1, model, cell_params)
    post_pop = sim.Population(1, model, cell_params)

    # Stimulating populations
    pre_times = [i for i in range(pre_phase, sim_time, time_between_pairs)]
    post_times = [i for i in range(post_phase, sim_time, time_between_pairs)]
    pre_stim = sim.Population(1, sim.SpikeSourceArray,
                              {'spike_times': [pre_times, ]})
    post_stim = sim.Population(1, sim.SpikeSourceArray,
                               {'spike_times': [post_times, ]})

    # Connections between spike sources and neuron populations
    ee_connector = sim.OneToOneConnector(weights=0.035)
    sim.Projection(pre_stim, pre_pop, ee_connector, target='excitatory')
    sim.Projection(post_stim, post_pop, ee_connector, target='excitatory')

    # Plastic Connection between pre_pop and post_pop
    stdp_model = sim.STDPMechanism(
        timing_dependence=sim.SpikePairRule(
            tau_plus=tau_plus, tau_minus=tau_minus),
        weight_dependence=sim.AdditiveWeightDependence(
            w_min=w_min, w_max=w_max, A_plus=A_plus, A_minus=A_minus),
    )

    projections.append(sim.Projection(
        pre_pop, post_pop, sim.OneToOneConnector(weights=start_w),
        synapse_dynamics=sim.SynapseDynamics(slow=stdp_model)
    ))

print("Simulating for %us" % (sim_time / 1000))

# Run simulation
sim.run(sim_time)

# Get weight from each projection
end_w = [p.getWeights()[0] for p in projections]

# End simulation on SpiNNaker
sim.end()

# -------------------------------------------------------------------
# Plot curve
# -------------------------------------------------------------------
# Calculate deltas from end weights
delta_w = [(w - start_w) / start_w for w in end_w]

# Plotting options
# ensure we use viridis as the default cmap
import matplotlib as mlib

plt.viridis()

# ensure we use the same rc parameters for all matplotlib outputs
mlib.rcParams.update({'font.size': 24})
mlib.rcParams.update({'errorbar.capsize': 5})
mlib.rcParams.update({'figure.autolayout': True})

# Plot STDP curve
figure, axis = plt.subplots(figsize=(10, 6), dpi=600)
axis.set_xlabel(r"$t_{j} - t_{i} (ms) $")
axis.set_ylabel(r"$\frac{\Delta w_{ij}}{w_{ij}}$",
                rotation="horizontal")
axis.axhline(color="grey", linestyle=":", alpha=0.7)
axis.axvline(color="grey", linestyle=":", alpha=0.7)
axis.scatter(delta_t, delta_w, color='#414C82', alpha=0.8)
axis.plot(delta_t, delta_w, color='#414C82')

plt.savefig(
    "stdp_curve_cond.pdf",
    bbox_inches='tight', dpi=800)
plt.savefig(
    "stdp_curve_cond.svg",
    bbox_inches='tight', dpi=800)
end_time = plt.datetime.datetime.now()
total_time = end_time - start_datetime

print("Total time elapsed -- " + str(total_time))

suffix = end_time.strftime("_%H%M%S_%d%m%Y")
filename = "stdp_curve" + str(suffix)

np.savez(filename,
         delta_t=delta_t,
         delta_w=delta_w,
         sim_time=sim_time,
         total_time=total_time)
# plt.show()
