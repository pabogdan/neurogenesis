# 1000 VRPSS neurons firing from 1 Hz - 1 kHz
import spynnaker8 as sim
from spynnaker8.extra_models import SpikeSourcePoissonVariable
import numpy as np

runtime = 1000
sim.setup(timestep=1.0, min_delay=1.0, max_delay=10)
sim.set_number_of_neurons_per_core(SpikeSourcePoissonVariable, 16)
# sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 16)

N_layer = 1000  # number of neurons in each population

cell_params = {
    "tau_m": 1000,
    "cm": 1,
    "v_rest": 0,
    "v_reset": 0,
    "v_thresh": 0.01,
    "tau_syn_E": 0.5,
    "tau_syn_I": 0.1,
    "tau_refrac": 0,
    "i_offset": 0
}

t_stim = runtime
number_of_slots = int(runtime / t_stim)
range_of_slots = np.arange(number_of_slots)
slots_starts = np.ones((N_layer, number_of_slots)) * (range_of_slots * t_stim)
durations = np.ones((N_layer, number_of_slots)) * t_stim
rates = np.asanyarray([np.arange(N_layer)]).T

poisson_spike_source = sim.Population(N_layer, SpikeSourcePoissonVariable(
    starts=slots_starts, rates=rates, durations=durations), label='poisson_source')

lif_pop = sim.Population(N_layer, sim.IF_curr_exp, cell_params, label='pop_1')

sim.Projection(
    poisson_spike_source, lif_pop, sim.OneToOneConnector(),
    sim.StaticSynapse(weight=1, delay=1))

poisson_spike_source.record(['spikes'])
lif_pop.record(['spikes'])

sim.run(runtime)
pss_spikes = poisson_spike_source.spinnaker_get_data('spikes')
lif_spikes = lif_pop.spinnaker_get_data('spikes')
sim.end()

np.savez_compressed("vrpss_rate_check_results",
                    pss_spikes=pss_spikes,
                    lif_spikes=lif_spikes,
                    rates=rates,
                    runtime=runtime,
                    simtime=runtime,
                    N_layer=N_layer)

pss_bincount = np.bincount(pss_spikes[:, 0].astype(int), minlength=1000)
lif_bincount = np.bincount(lif_spikes[:, 0].astype(int), minlength=1000)

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8, 7), dpi=300)
plt.plot(rates, label="Desired rate")
plt.plot(pss_bincount, alpha=.8, label="Recorded VRPSS rate")
plt.plot(lif_bincount, alpha=.8, label="LIF rate")
plt.legend(loc="best")

plt.savefig("vrpss_rate_analysis.png", bbox_inches='tight')
plt.show()
