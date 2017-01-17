import spynnaker.pyNN as p
from spynnaker_external_devices_plugin.pyNN.connections. \
    spynnaker_live_spikes_connection import SpynnakerLiveSpikesConnection
import spynnaker_external_devices_plugin.pyNN as ex
import spinn_breakout
import numpy as np
from breakout_utils.dealing_with_neuron_ids import *
import matplotlib.pyplot as plt

# Layout of pixels
X_BITS = 8
Y_BITS = 8

# Game resolution
X_RESOLUTION = 160
Y_RESOLUTION = 128

# UDP port to read spikes from
UDP_PORT = 17893

# Setup pyNN simulation
p.setup(timestep=1.0)

cell_params_lif = {'cm': 0.25,
                   'i_offset': 0.0,
                   'tau_m': 20.0,
                   'tau_refrac': 0.0,
                   'tau_syn_E': 5.0,
                   'tau_syn_I': 5.0,
                   'v_reset': -70.0,
                   'v_rest': -65.0,
                   'v_thresh': -50.0
                   }

weight_to_spike = 2.
delay = 1

# Create breakout population and activate live output for it
breakout_pop = p.Population(1, spinn_breakout.Breakout, {}, label="breakout")
ex.activate_live_output_for(breakout_pop, host="0.0.0.0", port=UDP_PORT)

# Create spike injector to inject keyboard input into simulation
key_input = p.Population(2, ex.SpikeInjector, {"port": 12367}, label="key_input")
key_input_connection = SpynnakerLiveSpikesConnection(send_labels=["key_input"])

# Connect key spike injector to breakout population
p.Projection(key_input, breakout_pop, p.OneToOneConnector(weights=2))

# Create visualiser
visualiser = spinn_breakout.Visualiser(
    UDP_PORT, key_input_connection,
    x_res=X_RESOLUTION, y_res=Y_RESOLUTION,
    x_bits=X_BITS, y_bits=Y_BITS)

# input_p2 = sim.Population(1, sim.SpikeSourceArray, {'spike_times': np.linspace(1000, 2500, 300)},label='inputSpikes_p2')
# sim.Projection(input_p2, breakout_pop, sim.FromListConnector([(0, 0, 3, 1)]))
#
# input_p = sim.Population(1, sim.SpikeSourceArray, {'spike_times': np.linspace(2000, 3000, 400)},label='inputSpikes_p')
# sim.Projection(input_p, breakout_pop, sim.FromListConnector([(0, 1, 3, 1)]))

ids = get_on_neuron_ids()
off_ids = get_off_neuron_ids()

no_paddle_on_ids = ids[0:-1, :]
no_paddle_off_ids = off_ids[0:-1, :]

list_of_on_connections = []
list_of_off_connections = []

for i in range(GAME_WIDTH):
    for j in no_paddle_on_ids[:, i]:
        if i < GAME_WIDTH // 2:
            list_of_on_connections.append((j, 0, weight_to_spike, delay))
        else:

            list_of_on_connections.append((j, 1, weight_to_spike, delay))

for i in range(GAME_WIDTH):
    for j in no_paddle_off_ids[:, i]:
        list_of_off_connections.append((j, i, weight_to_spike, delay))


no_paddle_on_population = p.Population(GAME_WIDTH, p.IF_curr_exp, cell_params_lif)
no_paddle_off_population = p.Population(GAME_WIDTH, p.IF_curr_exp, cell_params_lif)

p.Projection(breakout_pop, no_paddle_on_population, p.FromListConnector(list_of_on_connections),
             label='Ball on x position')
p.Projection(breakout_pop, no_paddle_off_population, p.FromListConnector(list_of_off_connections),
             label='Ball off x position')

# Synfire rate generators

p1 = p.Population(GAME_WIDTH, p.IF_curr_exp, cell_params_lif,
                  label='pop_1')
p2 = p.Population(GAME_WIDTH, p.IF_curr_exp, cell_params_lif,
                  label='pop_2')

# p1.record_v()
# p1.record_gsyn()
# p1.record()

p.Projection(p1, p2, p.OneToOneConnector(weight_to_spike), label='excite 1->2')
p.Projection(p2, p1, p.OneToOneConnector(weight_to_spike), label='excite 2->1')

p.Projection(no_paddle_on_population, p1, p.OneToOneConnector(weight_to_spike),
             label='excite no_paddle_on_population->1')

# Inhibitory connections
p.Projection(no_paddle_off_population, p1, p.OneToOneConnector(weight_to_spike), target='inhibitory',
             label='inhib no_paddle_off_population->1')
p.Projection(no_paddle_off_population, p2, p.OneToOneConnector(weight_to_spike), target='inhibitory',
             label='inhib no_paddle_off_population->2')

# Connecting left and right controllers

left_connections = []
right_connections = []

for i in range(GAME_WIDTH):
    if i < GAME_WIDTH // 2:
        left_connections.append((i, 0 ,weight_to_spike, delay))
    else:
        right_connections.append((i, 1, weight_to_spike, delay))

p.Projection(no_paddle_on_population, breakout_pop, p.FromListConnector(list_of_on_connections), label='left')
# p.Projection(p1, breakout_pop, p.FromListConnector(right_connections), label='right')


# p.Projection(p2, breakout_pop, p.FromListConnector(left_connections), label='left')
# p.Projection(p2, breakout_pop, p.FromListConnector(right_connections), label='right')

# Run simulation (non-blocking)
p.run(None)

# Show visualiser (blocking)
visualiser.show()

# v_p1 = p1.get_v(compatible_output=True)
# gsyn_p1 = p1.get_gsyn(compatible_output=True)
# spikes_p1 = p1.getSpikes(compatible_output=True)
# End simulation

# plt.plot(v_p1[:, 2])
# plt.show()

p.end()
