# Copyright (c) 2017-2019 The University of Manchester
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import matplotlib.pyplot as pylab
try:
    import pyNN.spiNNaker as sim
except Exception:
    import spynnaker8 as sim

# ------------------------------------------------------------------
# This example uses the sPyNNaker implementation of pair-based STDP
# To reproduce the eponymous STDP curve first
# Plotted by Bi and Poo (1998)
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Common parameters
# ------------------------------------------------------------------
time_between_pairs = 1000
num_pairs = 60
start_w = 0.005
delta_t = [-100, -60, -40, -30, -20, -10, -1, 1, 10, 20, 30, 40, 60, 100]
start_time = 200
mad = True

# Population parameters
model = sim.IF_cond_exp
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

# SpiNNaker setup
sim.setup(timestep=0.1, min_delay=0.1, max_delay=1.0)

# -------------------------------------------------------------------
# Experiment loop
# -------------------------------------------------------------------
projections = []
sim_time = 0
for t in delta_t:
    # Calculate phase of input spike trains
    # If M.A.D., take into account dendritic delay
    if mad:
        # Pre after post
        if t > 0:
            post_phase = 0
            pre_phase = t + 1
        # Post after pre
        else:
            post_phase = -t
            pre_phase = 1
    # Otherwise, take into account axonal delay
    else:
        # Pre after post
        if t > 0:
            post_phase = 1
            pre_phase = t
        # Post after pre
        else:
            post_phase = 1 - t
            pre_phase = 0

    sim_time = max(sim_time, (num_pairs * time_between_pairs) + abs(t))

    # Neuron populations
    pre_pop = sim.Population(1, model(**cell_params))
    post_pop = sim.Population(1, model, cell_params)

    # Stimulating populations
    pre_times = [i for i in range(pre_phase, sim_time, time_between_pairs)]
    post_times = [i for i in range(post_phase, sim_time, time_between_pairs)]
    pre_stim = sim.Population(
        1, sim.SpikeSourceArray(spike_times=[pre_times]))
    post_stim = sim.Population(
        1, sim.SpikeSourceArray(spike_times=[post_times]))

    weight = 0.035

    # Connections between spike sources and neuron populations
    ee_connector = sim.OneToOneConnector()
    sim.Projection(
        pre_stim, pre_pop, ee_connector, receptor_type='excitatory',
        synapse_type=sim.StaticSynapse(weight=weight))
    sim.Projection(
        post_stim, post_pop, ee_connector, receptor_type='excitatory',
        synapse_type=sim.StaticSynapse(weight=weight))

    # Plastic Connection between pre_pop and post_pop
    stdp_model = sim.STDPMechanism(
        timing_dependence=sim.SpikePairRule(
            tau_plus=16.7, tau_minus=33.7, A_plus=0.005, A_minus=0.005),
        weight_dependence=sim.AdditiveWeightDependence(
            w_min=0.0, w_max=0.0175), weight=start_w)

    projections.append(sim.Projection(
        pre_pop, post_pop, sim.OneToOneConnector(),
        synapse_type=stdp_model))

print("Simulating for %us" % (sim_time / 1000))

# Run simulation
sim.run(sim_time)

# Get weight from each projection
end_w = [p.get('weight', 'list', with_address=False)[0] for p in projections]

# End simulation on SpiNNaker
sim.end()

# -------------------------------------------------------------------
# Plot curve
# -------------------------------------------------------------------
# Calculate deltas from end weights
delta_w = [(w - start_w) / start_w for w in end_w]

# Plot STDP curve
figure, axis = pylab.subplots()
axis.set_xlabel(r"$(t_{j} - t_{i}/ms)$")
axis.set_ylabel(r"$(\frac{\Delta w_{ij}}{w_{ij}})$",
                rotation="horizontal", size="xx-large")
axis.plot(delta_t, delta_w)
axis.axhline(color="grey", linestyle="--")
axis.axvline(color="grey", linestyle="--")
pylab.savefig("stdp_curve_at_01ms.png")
