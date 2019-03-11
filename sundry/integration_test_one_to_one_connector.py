"""
Integration test for one to one connector.

Q:Does having both a 1:1 connector + an ALL:ALL connector between the same
pair of populations affect the behaviour of one of them? Particularly,
do edges relevant for the ALL to ALL connector get pruned?
"""
import spynnaker7.pyNN as p
import matplotlib.pyplot as plt
import numpy as np

runtime = 100
p.setup(timestep=1.0, min_delay=1.0, max_delay=14)
nNeurons = 2  # number of neurons in each population
p.set_number_of_neurons_per_core("IF_curr_exp", 1)
p.set_number_of_neurons_per_core("SpikeSourceArray", 1)

cell_params_lif = {'cm': 0.25,
                   'i_offset': 0.0,
                   'tau_m': 20.0,
                   'tau_refrac': 2.0,
                   'tau_syn_E': 5.0,
                   'tau_syn_I': 5.0,
                   'v_reset': -70.0,
                   'v_rest': -65.0,
                   'v_thresh': -50.0
                   }

model = p.IF_curr_exp
weight_to_spike = 0.1
all_weight_to_spike = 3.0
delay = 1

spikeArray = {'spike_times': [[0], []]}

input_pop = p.Population(nNeurons, p.SpikeSourceArray, spikeArray,
                     label='inputSpikes_1')

target_pop = p.Population(nNeurons, model, cell_params_lif,
                      label='pop_1')


# one_to_one_conn = p.Projection(
#     input_pop, target_pop, p.OneToOneConnector(weights=weight_to_spike,
#                                                  delays=delay))
all_to_all_conn = p.Projection(
    input_pop, target_pop, p.AllToAllConnector(weights=all_weight_to_spike,
                                                 delays=delay))

target_pop.record(['v', 'spikes'])

p.run(runtime)
spikes = target_pop.getSpikes(
        compatible_output=True)

print(spikes)
def plot_spikes(spikes, title, run_time, filename):
    if spikes is not None:
        recast_spikes = []
        for index in np.unique(spikes[:, 0]):
            recast_spikes.append(spikes[spikes[:, 0] == index][:, 1])
        f, ax1 = plt.subplots(1, 1, figsize=(10, 5), dpi=300)
        ax1.set_xlim((0, run_time))
        ax1.eventplot(recast_spikes, linelengths=.8)
        ax1.set_xlabel('Time(ms)')
        ax1.set_ylabel('Neuron ID')
        ax1.set_title(title)
        plt.savefig(filename, bbox_inches='tight')
        plt.show()

plot_spikes(spikes,
            "Synfire all spikes",
            runtime,
            "spikes_one_to_one_connector.png")
p.end()
