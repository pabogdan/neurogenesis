import spynnaker8 as sim
import numpy
import math
import unittest
import numpy as np
from pyNN.utility.plotting import Figure, Panel
# import matplotlib.pyplot as plt
import pylab as plt

from spynnaker8.utilities.neo_convertor import convert_spikes

start_time = plt.datetime.datetime.now()
v_reset = -65
v_thresh = -50
rngseed = 98766987
parallel_safe = True
rng = sim.NumpyRNG(seed=rngseed, parallel_safe=parallel_safe)

sim_time = 5000
connection_probability = .1
connection_weight = 0.03
coupling_multiplier = 2.
# connection_weight = sim.RandomDistribution('normal', [0.05, 10**-3],
#                                            rng=rng)
delay = sim.RandomDistribution('uniform', [1, 14], rng=rng)

cell_params_exc = {
    'tau_m': 20.0, 'cm': 1.0, 'v_rest': -65.0, 'v_reset': -65.0,
    'v_thresh': -50.0, 'tau_syn_E': 5.0, 'tau_syn_I': 15.0,
    'tau_refrac': 0.3, 'i_offset': 0}

cell_params_inh = {
    'tau_m': 20.0, 'cm': 1.0, 'v_rest': -65.0, 'v_reset': -65.0,
    'v_thresh': -50.0, 'tau_syn_E': 5.0, 'tau_syn_I': 5.0,
    'tau_refrac': 0.3, 'i_offset': 0}

# Initialise simulator
sim.setup(timestep=1, min_delay=1, max_delay=15)

# Spike sources
poisson_spike_source = sim.Population(500, sim.SpikeSourcePoisson(
    rate=50, duration=sim_time), label='poisson_source')

# spike_times = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
spike_times = [sim_time//2]
spike_source_array = sim.Population(125, sim.SpikeSourceArray,
                                    {'spike_times': spike_times},
                                    label='spike_source')

# Neuronal populations
pop_exc = sim.Population(500, sim.IF_curr_exp(**cell_params_exc),
                         label='excitatory_pop')
pop_inh = sim.Population(125, sim.IF_curr_exp(**cell_params_inh),
                         label='inhibitory_pop')
uniformDistr = sim.RandomDistribution('uniform', [v_reset, v_thresh], rng=rng)
pop_exc.set(v=uniformDistr)

# Spike input projections
spike_source_projection = sim.Projection(
    spike_source_array, pop_exc,
    sim.FixedProbabilityConnector(p_connect=connection_probability),
    sim.StaticSynapse(weight=2*connection_weight, delay=delay),
    receptor_type='excitatory')
# Poisson source projections
poisson_projection_exc = sim.Projection(
    poisson_spike_source, pop_exc,
    sim.FixedProbabilityConnector(p_connect=connection_probability),
    synapse_type=sim.StaticSynapse(weight=2 * connection_weight, delay=delay),
    receptor_type='excitatory')
poisson_projection_inh = sim.Projection(
    poisson_spike_source, pop_inh,
    sim.FixedProbabilityConnector(p_connect=connection_probability),
    sim.StaticSynapse(weight=connection_weight, delay=delay),
    receptor_type='excitatory')

# Recurrent projections
exc_exc_rec = sim.Projection(
    pop_exc, pop_exc,
    # sim.OneToOneConnector(),
    sim.FixedProbabilityConnector(
        p_connect=connection_probability),
    synapse_type=sim.StaticSynapse(
        weight=connection_weight, delay=delay),
    receptor_type='excitatory')
exc_exc_one_to_one_rec = sim.Projection(
    pop_exc, pop_exc,
    sim.OneToOneConnector(),
    synapse_type=sim.StaticSynapse(
        weight=connection_weight,
        delay=delay),
    receptor_type='excitatory')
inh_inh_rec = sim.Projection(
    pop_inh, pop_inh,
    sim.FixedProbabilityConnector(
        p_connect=connection_probability),
    synapse_type=sim.StaticSynapse(
        weight=connection_weight, delay=delay),
    receptor_type='inhibitory')

# Neuronal population projections -- Coupling
exc_to_inh = sim.Projection(
    pop_exc, pop_inh,
    sim.FixedProbabilityConnector(p_connect=2 * connection_probability),
    synapse_type=sim.StaticSynapse(
        weight=coupling_multiplier * connection_weight, delay=delay),
    receptor_type='excitatory')
inh_to_exc = sim.Projection(
    pop_inh, pop_exc,
    sim.FixedProbabilityConnector(p_connect=2 * connection_probability),
    synapse_type=sim.StaticSynapse(
        weight=coupling_multiplier * connection_weight, delay=delay),
    receptor_type='inhibitory')

# Specify output recording
pop_exc.record('spikes')
# pop_inh.record('v', 'spikes')
pop_inh.record('spikes')

# Run simulation
sim.run(simtime=sim_time)

# Get results data
exc_data = pop_exc.get_data('spikes')
inh_data = pop_inh.get_data('spikes')

# Plot spikes
# Figure(
#     Panel(exc_data.segments[0].spiketrains,
#           yticks=True, markersize=0.2, xlim=(0, sim_time)),
#     Panel(inh_data.segments[0].spiketrains,
#           yticks=True, markersize=0.2, xlim=(0, sim_time)),
# )
# plt.show()

# Exit simulation
sim.end()

end_time = plt.datetime.datetime.now()
total_time = end_time - start_time

exc_spikes = convert_spikes(exc_data)
inh_spikes = convert_spikes(inh_data)

print("Total time elapsed -- " + str(total_time))

suffix = end_time.strftime("_%H%M%S_%d%m%Y")
filename = "random_balanced_network_results" + str(suffix)

np.savez(filename,
         exc_spikes=exc_spikes,
         inh_spikes=inh_spikes,
         sim_time=sim_time,
         total_time=total_time)
print("Results in", filename)


# Plotting


import matplotlib as mlib

mlib.rcParams.update({'font.size': 30})


def plot_spikes(exc_spikes, inh_spikes, title, simtime=sim_time):
    f, ax1 = plt.subplots(1, 1, figsize=(20, 10))
    ax1.set_xlim((0, simtime))
    ax1.scatter([i[1] for i in exc_spikes], [i[0] for i in exc_spikes], s=1,
                marker="*")

    ax1.scatter([i[1] for i in inh_spikes], [i[0] + 500 for i in inh_spikes],
                s=1, marker="*", c='r')
    ax1.set_xlabel('Time(ms)')
    ax1.set_ylabel('Neuron ID')
    ax1.set_title(title)
    plt.savefig(title + ".png", dpi=800)
    plt.tight_layout()
    plt.show()

plot_spikes(exc_spikes, inh_spikes, "Random Balanced Network")
