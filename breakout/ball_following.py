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
p.set_number_of_neurons_per_core("IF_curr_exp", 100)

# Parameters
cell_params_lif = {'cm': 0.25,
                   'i_offset': 0.0,
                   'tau_m': 20.0,
                   'tau_refrac': 5.0,
                   'tau_syn_E': 5.0,
                   'tau_syn_I': 5.0,
                   'v_reset': -70.0,
                   'v_rest': -65.0,
                   'v_thresh': -50.0
                   }

weight_to_spike = 2.
paddle_weight = 1.5
delay = 1
rate_delay = 16
pool_size = 8

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

direction_population = p.Population(3, p.IF_curr_exp, cell_params_lif)
p.Projection(direction_population, breakout_pop, p.FromListConnector(
    [(1, 1, weight_to_spike, delay),
     (2, 2, weight_to_spike, delay)]))

## Code to implement https://trello.com/c/rLwRIOxq/111-hard-coded-breakout follows
on_ids = get_on_neuron_ids()
off_ids = get_off_neuron_ids()
no_paddle_on_ids = on_ids[:-1, :]
only_paddle_on_ids = on_ids[-1, :]
only_paddle_off_ids = off_ids[-1, :]
paddle_presence_weight = .7
ball_presence_weight = .7

# Create  all needed populations

ball_position = p.Population(GAME_WIDTH // pool_size, p.IF_curr_exp, cell_params_lif,
                             label="Ball position pop")
paddle_position = p.Population(GAME_WIDTH // pool_size, p.IF_curr_exp, cell_params_lif,
                               label="Paddle position pop")
left_receptive_field = p.Population(GAME_WIDTH // pool_size, p.IF_curr_exp, cell_params_lif,
                                    label="Left receptive field pop")
right_receptive_field = p.Population(GAME_WIDTH // pool_size, p.IF_curr_exp, cell_params_lif,
                                     label="Right receptive field pop")
rate_generator = p.Population(GAME_WIDTH // pool_size, p.IF_curr_exp, cell_params_lif,
                              label="Rate generation population")
p.Projection(rate_generator, rate_generator, p.OneToOneConnector(paddle_weight, rate_delay), target='excitatory',
             label='rate_pop->rate_pop')
# TODO Wiring up breakout pop to ball position
list_of_on_connections = []

for i in range(GAME_WIDTH):
    for j in no_paddle_on_ids[:, i]:
        list_of_on_connections.append((j, i//pool_size, weight_to_spike, delay))

p.Projection(breakout_pop, ball_position, p.FromListConnector(list_of_on_connections))

# TODO Wiring up breakout pop to paddle position rate generators (on & off neurons)
paddle_on_connection = []
paddle_off_connection = []
for i in range(GAME_WIDTH):
    paddle_on_connection.append((only_paddle_on_ids[i], i//pool_size, paddle_weight, delay))
    paddle_off_connection.append((only_paddle_off_ids[i], i // pool_size, paddle_weight, delay))

p.Projection(breakout_pop, rate_generator, p.FromListConnector(paddle_on_connection))
p.Projection(breakout_pop, rate_generator, p.FromListConnector(paddle_off_connection))
# TODO Wiring up rate generators to paddle position pop
p.Projection(rate_generator, paddle_position, p.OneToOneConnector(paddle_presence_weight/2., delay), target='excitatory',
             label='rate_pop->paddle_pop')

# Wiring up paddle_position pop to L & R receptive fields
p.Projection(paddle_position, left_receptive_field, p.OneToOneConnector(paddle_presence_weight),
             label="Paddle -> Left connection")
p.Projection(paddle_position, right_receptive_field, p.OneToOneConnector(paddle_presence_weight),
             label="Paddle -> Right connection")

# Wiring up ball position pop to L & R receptive fields
left_receptive_field_connections = []
right_receptive_field_connections = []
for current_index in xrange(GAME_WIDTH//pool_size):
    for left_index in xrange(current_index):
        left_receptive_field_connections.append((left_index, current_index,
                                                 ball_presence_weight, delay))
    for right_index in xrange(current_index+1, GAME_WIDTH//pool_size):
        right_receptive_field_connections.append((right_index, current_index,
                                                 ball_presence_weight, delay))

p.Projection(ball_position, left_receptive_field,
             p.FromListConnector(left_receptive_field_connections), label="L receptive field conn")
p.Projection(ball_position, right_receptive_field,
             p.FromListConnector(right_receptive_field_connections), label="R receptive field conn")

# Wiring up L & R receptive fields to direction controlling population
left_direction = []
right_direction = []

for i in xrange(GAME_WIDTH//pool_size):
    left_direction.append((i, 1, weight_to_spike, delay))
    right_direction.append((i, 2, weight_to_spike, delay))

p.Projection(left_receptive_field, direction_population, p.FromListConnector(left_direction),
             label="Left direction conn")
p.Projection(right_receptive_field, direction_population, p.FromListConnector(right_direction),
             label="Right direction conn")

# Run simulation (non-blocking)
p.run(None)

# Show visualiser (blocking)
visualiser.show()

# End simulation
p.end()
