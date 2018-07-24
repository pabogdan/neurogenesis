from __future__ import print_function
import spynnaker8 as sim
import numpy as np
from pyNN.utility.plotting import Figure, Panel
import pylab as plt

start_time = plt.datetime.datetime.now()
v_reset = -65
v_thresh = -50
rngseed = 98766987
parallel_safe = True
rng = sim.NumpyRNG(seed=rngseed, parallel_safe=parallel_safe)

sim_time = 5000
coupling_multiplier = 2.
delay_distribution = sim.RandomDistribution('uniform', [1, 14], rng=rng)

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

# Inject a spike into the network half-way through the run
spike_source_array = sim.Population(125, sim.SpikeSourceArray,
                                    {'spike_times': [sim_time // 2]},
                                    label='spike_source')

# Neuronal populations
pop_exc = sim.Population(500, sim.IF_curr_exp(**cell_params_exc),
                         label='excitatory_pop')
pop_inh = sim.Population(125, sim.IF_curr_exp(**cell_params_inh),
                         label='inhibitory_pop')

# Initialise cell membrane potential to some sub-threshold level
pop_exc.set(v=sim.RandomDistribution('uniform',
                                     [v_reset, v_thresh],
                                     rng=rng))

# Spike input projections
spike_source_projection = sim.Projection(
    spike_source_array, pop_exc,
    sim.FixedProbabilityConnector(p_connect=.1),
    synapse_type=sim.StaticSynapse(weight=0.06, delay=delay_distribution),
    receptor_type='excitatory')

# Poisson source projections
poisson_projection_exc = sim.Projection(
    poisson_spike_source, pop_exc,
    sim.FixedProbabilityConnector(p_connect=.1),
    synapse_type=sim.StaticSynapse(weight=0.06, delay=delay_distribution),
    receptor_type='excitatory')
poisson_projection_inh = sim.Projection(
    poisson_spike_source, pop_inh,
    sim.FixedProbabilityConnector(p_connect=.1),
    synapse_type=sim.StaticSynapse(weight=0.03, delay=delay_distribution),
    receptor_type='excitatory')

# Recurrent projections
exc_exc_rec = sim.Projection(
    pop_exc, pop_exc,
    sim.FixedProbabilityConnector(p_connect=.1),
    synapse_type=sim.StaticSynapse(weight=0.03, delay=delay_distribution),
    receptor_type='excitatory')
exc_exc_one_to_one_rec = sim.Projection(
    pop_exc, pop_exc,
    sim.OneToOneConnector(),
    synapse_type=sim.StaticSynapse(weight=0.03, delay=delay_distribution),
    receptor_type='excitatory')
inh_inh_rec = sim.Projection(
    pop_inh, pop_inh,
    sim.FixedProbabilityConnector(p_connect=.1),
    synapse_type=sim.StaticSynapse(weight=0.03, delay=delay_distribution),
    receptor_type='inhibitory')

# Projections between neuronal populations
exc_to_inh = sim.Projection(
    pop_exc, pop_inh,
    sim.FixedProbabilityConnector(p_connect=.2),
    synapse_type=sim.StaticSynapse(
        weight=coupling_multiplier * 0.03, delay=delay_distribution),
    receptor_type='excitatory')
inh_to_exc = sim.Projection(
    pop_inh, pop_exc,
    sim.FixedProbabilityConnector(p_connect=.2),
    synapse_type=sim.StaticSynapse(
        weight=coupling_multiplier * 0.03, delay=delay_distribution),
    receptor_type='inhibitory')

# Specify output recording
pop_exc.record('spikes')
pop_inh.record('spikes')

# Run simulation
sim.run(simtime=sim_time)

# Get results data
exc_data = pop_exc.get_data('spikes')
inh_data = pop_inh.get_data('spikes')

# Record duration of experiment
end_time = plt.datetime.datetime.now()
total_time = end_time - start_time

# Plot spikes
Figure(
    Panel(inh_data.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, sim_time),
          marker="*", c='C3'),
    Panel(exc_data.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, sim_time),
          marker="*", c='C0', xlabel="Time (ms)",
          xticks=np.linspace(0, sim_time, 5)),
    settings={'font.size': 14,
              'savefig.dpi': 800,
              },
    title="Random Balanced Network"
).save("rbn.png")
plt.show()

# Exit simulation -- release and clear the machine
sim.end()

# Report duration of experiment
print("Total time elapsed -- " + str(total_time))
