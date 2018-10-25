# Scalable synfire chain
# Includes:
#   assertions of spike times at various points in the synfire
#   set input spike times and locations
#   controllable number of populations, neurons in a population, injection
#       sites
#   one to one connectors between subsequent populations
#   results stored in compressed numpy archives

# Imports
import traceback
# Argparser also includes defaults
from synfire_argparser import *
import numpy as np
import pylab as plt
import os
import ntpath
import spynnaker7.pyNN as sim

start_time = plt.datetime.datetime.now()
sim.setup(timestep=1.0, min_delay=1.0, max_delay=15)

# Population parameters taken from synfire if curr exp example
model = sim.IF_curr_exp
cell_params_lif = {
    'cm': 0.25,
    'i_offset': 0.0,
    'tau_m': 5.0,
    'tau_refrac': 1.0,
    'tau_syn_E': 5.0,
    'tau_syn_I': 5.0,
    'v_reset': -90.0,
    'v_rest': -65.0,
    'v_thresh': -50.0
}
weight = 4.0
delay = 1

# values from argparser
n = args.n
no_populations = args.no_pops
no_injection_sites = args.no_injection_sites

# compute population indices where to inject a spike via SSA
injection_pop_skip = no_populations / no_injection_sites
injection_pop_indices = np.arange(no_injection_sites) * injection_pop_skip

# keep track of populations and projections
all_populations = []
all_projections = []
all_ssas = []
all_injection_projections = []
all_spikes = np.asarray([[]])

for pop_id in xrange(no_populations):
    # create a population
    all_populations.append(sim.Population(n, model, cell_params_lif,
                                          label='pop_{}'.format(str(pop_id))))
    # enable recording
    all_populations[pop_id].record()

    # connect the current population to the previous one except if this is not
    # the first population to be created
    if pop_id > 0:
        all_projections.append(
            sim.Projection(all_populations[pop_id - 1],
                           all_populations[pop_id],
                           sim.OneToOneConnector(weights=weight,
                                                 delays=delay),
                           label='proj_{}'.format(str(pop_id - 1)))
        )

    # if pop is supposed to receive spike injections create SSA
    if pop_id in injection_pop_indices:
        all_ssas.append(
            sim.Population(n, sim.SpikeSourceArray,
                           {'spike_times': [[0]] * n},
                           label='SSA_for_pop_{}'.format(pop_id)))
        all_injection_projections.append(
            sim.Projection(
                all_ssas[
                    np.argwhere(injection_pop_indices == pop_id).ravel()[0]],
                all_populations[pop_id],
                sim.OneToOneConnector(weights=weight,
                                      delays=delay))
        )
# add the last wrap-around projection
all_projections.append(
    sim.Projection(all_populations[no_populations - 1],
                   all_populations[0],
                   sim.OneToOneConnector(weights=weight,
                                         delays=delay),
                   label='proj_{}'.format(str(no_populations - 1)))
)
# run so that each population has spiked args.no_loops times
run_time = 2 * (injection_pop_skip + n) * args.no_loops
sim.run(run_time=run_time)

for pop_id in xrange(no_populations):
    spikes_current_pop = np.asarray(all_populations[pop_id].getSpikes(
        compatible_output=True))
    spikes_current_pop[:, 0] += (pop_id * n + (n - 1))
    if all_spikes.size == 0:
        all_spikes = spikes_current_pop
    else:
        all_spikes = np.concatenate((all_spikes, spikes_current_pop))
all_spikes = np.asarray(all_spikes)

# end the simulation
sim.end()

end_time = plt.datetime.datetime.now()
total_time = end_time - start_time
suffix = end_time.strftime("_%H%M%S_%d%m%Y")
print("Total time elapsed -- " + str(total_time))

# save simulation results in compressed numpy archive
filename = ''
if args.filename:
    filename = args.filename
else:
    filename = "scalable_synfire" + suffix

np.savez_compressed(
    filename,
    all_spikes=all_spikes,
    run_time=run_time,
    n=n,
    no_populations=no_populations,
    no_injection_sites=no_injection_sites,
)


# plot spikes
def plot_spikes(spikes, title, run_time, filename):
    if spikes is not None:
        recast_spikes = []
        for index in np.unique(spikes[:, 0]):
            recast_spikes.append(spikes[spikes[:, 0] == index][:, 1])
        f, ax1 = plt.subplots(1, 1, figsize=(15, 6), dpi=600)
        ax1.set_xlim((0, run_time))
        ax1.eventplot(recast_spikes, linelengths=.8)
        ax1.set_xlabel('Time/ms')
        ax1.set_ylabel('Neuron ID')
        ax1.set_title(title)
        plt.savefig(filename, bbox_inches='tight')
        plt.show()


plot_spikes(all_spikes,
            "Synfire all spikes",
            run_time,
            "synfire_spike_raster_plot.pdf")

# assert that each neuron in each population has spiked exactly
# args.no_loops times and at the correct time!


print("Total time elapsed -- " + str(total_time))
