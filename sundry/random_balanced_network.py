import spynnaker8 as sim
import numpy
import math
import unittest
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

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
    rate=50, duration=10000), label='poisson_source')

# spike_times = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
spike_times = [11000]
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
    sim.StaticSynapse(weight=connection_weight, delay=delay),
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
    # sim.FixedProbabilityConnector(p_connect=connection_probability),
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

# Neuronal population projections
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
Figure(
    Panel(exc_data.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, sim_time)),
    Panel(inh_data.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, sim_time)),
)
plt.show()

# Exit simulation
sim.end()
