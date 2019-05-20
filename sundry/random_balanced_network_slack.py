import spynnaker8 as sim
import numpy as np
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

# Neuronal populations
pop_exc = sim.Population(500, sim.IF_curr_exp(**cell_params_exc),
                         label='excitatory_pop')
pop_inh = sim.Population(125, sim.IF_curr_exp(**cell_params_inh),
                         label='inhibitory_pop')
uniformDistr = sim.RandomDistribution('uniform', [v_reset, v_thresh], rng=rng)
pop_exc.set(v=uniformDistr)

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

mlib.rcParams.update({'font.size': 25})


def instant_rates(spikes, simtime=sim_time, chunk_size=1, N_layer=500):
    per_neuron_instaneous_rates = np.empty((N_layer, int(np.ceil(simtime / chunk_size))))
    for neuron_index in np.arange(N_layer):

        firings_for_neuron = spikes[
            spikes[:, 0] == neuron_index]
        for chunk_index in np.arange(per_neuron_instaneous_rates.shape[
                                         1]):
            per_neuron_instaneous_rates[neuron_index, chunk_index] = \
                np.count_nonzero(
                    np.logical_and(
                        firings_for_neuron[:, 1] >= (
                                chunk_index * chunk_size),
                        firings_for_neuron[:, 1] < (
                                (chunk_index + 1) * chunk_size)
                    )
                ) / (1.0 * chunk_size)
    instaneous_rates = np.sum(per_neuron_instaneous_rates,
                              axis=0)  # / float(N_layer)  # uncomment this if you want mean firing rate

    return instaneous_rates


def plot_spikes(exc_spikes, inh_spikes, title, simtime=sim_time):
    f, (ax3, ax1, ax2) = plt.subplots(3, 1, figsize=(20, 10), sharex=True, gridspec_kw={'height_ratios': [1, 3, 1],
                                                                                        'hspace': .05})

    chunk_size = 1  # ms
    inh_N_layer = 125
    inh_instant_rates_1_ms = instant_rates(inh_spikes, simtime, chunk_size, inh_N_layer)
    ax3.bar(np.arange(0, simtime, chunk_size), inh_instant_rates_1_ms, width=chunk_size, color='#b2dd2c')

    ax2.set_xlabel('Time (ms)')

    ax1.set_xlim((0, simtime))
    ax1.scatter([i[1] for i in exc_spikes], [i[0] for i in exc_spikes], s=1,
                marker="*", color='#440357')

    ax1.scatter([i[1] for i in inh_spikes], [i[0] + 500 for i in inh_spikes],
                s=1, marker="*", color='#b2dd2c')
    ax1.set_ylabel('Neuron ID')
    ax3.set_title(title)
    # Include a histogram of mean firing activity

    N_layer = 500
    instant_rates_1_ms = instant_rates(exc_spikes, simtime, chunk_size, N_layer)
    ax2.bar(np.arange(0, simtime, chunk_size), instant_rates_1_ms, width=chunk_size, color='#440357')

    # Saving the figure
    plt.savefig(title + ".pdf", bbox_inches="tight")
    plt.tight_layout()


plot_spikes(exc_spikes, inh_spikes, "Random Balanced Network")
