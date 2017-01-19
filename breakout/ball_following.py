import spynnaker.pyNN as p
from spynnaker_external_devices_plugin.pyNN.connections. \
    spynnaker_live_spikes_connection import SpynnakerLiveSpikesConnection
import spynnaker_external_devices_plugin.pyNN as ex
import spinn_breakout
import numpy as np
from breakout_utils.dealing_with_neuron_ids import *
import matplotlib.pyplot as plt
from breakout_utils.plotting import *

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
                   'tau_refrac': 2.0,
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

ids = get_on_neuron_ids()

no_paddle_on_ids = ids[0:-1, :]

list_of_on_connections = []

for i in range(GAME_WIDTH):
    for j in no_paddle_on_ids[:, i]:
        if i < GAME_WIDTH // 3:
            list_of_on_connections.append((j, 1, weight_to_spike, delay))
        elif i > (2 * GAME_WIDTH )// 3:
            list_of_on_connections.append((j, 2, weight_to_spike, delay))


no_paddle_on_population = p.Population(GAME_WIDTH, p.IF_curr_exp, cell_params_lif)

p.Projection(breakout_pop, no_paddle_on_population, p.FromListConnector(list_of_on_connections),
             label='Ball on x position')


p.Projection(no_paddle_on_population, breakout_pop, p.OneToOneConnector(weight_to_spike), label='left')

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
