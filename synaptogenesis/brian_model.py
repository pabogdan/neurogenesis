import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
import time
import os

plt.viridis()
plt.close()

# Membrane
v_rest = -70 * mvolt  # mV
e_ext = 0 * mvolt  # V
v_thr = -54 * mvolt  # mV
g_max = 0.2
tau_m = 20 * ms  # ms
tau_ex = 5 * ms  # ms
e = np.e
g = g_max
pre_t = 0 * ms

# STDP
Apre = a_plus = 0.1
b = 1.2
taupre = tau_plus = 20 * ms  # ms
taupost = tau_minus = 64 * ms  # ms
Apost = a_minus = (a_plus * tau_plus * b) / tau_minus


class BrianModel(object):
    def __init__(self, seed=None, **kwargs):
        if 'no_iterations' in kwargs:
            self.no_iterations = kwargs['no_iterations']
        else:
            self.no_iterations = None
        self.seed = None
        if seed:
            self.seed = seed
            np.random.seed(self.seed)
        self.recording_filename = str(np.random.randint(100000, 1000000))

        # Wiring
        self.n = 16
        self.N = self.n ** 2
        self.S = (self.n, self.n)

        self.s_max = 32
        self.sigma_form_forward = 2.5
        self.sigma_form_lateral = 1
        self.p_form_lateral = 1
        self.p_form_forward = 0.16
        self.p_elim_dep = 0.0245
        self.p_elim_pot = 1.36 * np.e ** -4
        self.f_rew = 10 ** 4 * Hz

        # Membrane
        self.v_rest = -70 * mvolt
        self.e_ext = 0 * mvolt
        self.v_thr = -54 * mvolt
        self.g_max = 0.2
        self.tau_m = 20 * ms
        self.tau_ex = 5 * ms
        self.e = np.e
        self.g = 0
        self.pre_t = 0 * ms

        # Inputs
        self.f_mean = 20 * Hz
        self.f_base = 5 * Hz
        self.f_peak = 152.8 * Hz
        self.sigma_stim = 2
        self.t_stim = 0.02 * second
        self.rate = 200 * Hz

        # STDP
        self.Apre = self.a_plus = 0.1
        self.b = 1.2
        self.taupre = self.tau_plus = 20 * ms
        self.taupost = self.tau_minus = 64 * ms
        self.Apost = self.a_minus = (self.a_plus * self.tau_plus * self.b) / self.tau_minus

        # Recordings
        self.recordings = {'spikes': True,
                           'states': True,
                           'rewiring': True,
                           'use_files': True,
                           }
        self.statemon = None
        self.spikemon = None

        self.feedforward = None
        self.lateral = None

        self.rewire_trigger = 0
        self.form_trigger = {
            0: 0,
            1: 0,
        }
        self.elim_trigger = {
            0: 0,
            1: 0,
        }
        self.formations = {
            0: 0,
            1: 0,
        }
        self.eliminations = {
            0: 0,
            1: 0,
        }
        self.stimulus_trigger = 0

    def distance(self, s, t, dist_type='euclidian'):
        '''
        Function that computes distance in a grid of neurons taking into account periodic boundry conditions.

        First, translate source into the center of the grid.
        Second, translate target by the same amount.
        Finally, perform desired distance computation.
        '''
        s = np.asarray(s)
        t = np.asarray(t)
        _grid_size = np.asarray(self.S)
        trans = s - (_grid_size // 2)
        s = np.mod(s - trans, _grid_size)
        t = np.mod(t - trans, _grid_size)
        if dist_type == 'manhattan':
            return s[0] - t[0] + s[1] - t[1]
        return np.sqrt((s[0] - t[0]) ** 2 + (s[1] - t[1]) ** 2)

    def generate_rates(self, s, dist_type='euclidian'):
        '''
        Function that generates an array the same shape as the input layer so that
        each cell has a value corresponding to the firing rate for the neuron
        at that position.
        '''
        _rates = np.zeros(self.S)
        for x, y in np.ndindex(self.S):
            _d = self.distance(s, (x, y), dist_type=dist_type)
            _rates[x, y] = self.f_base + self.f_peak * np.e ** (-_d / (2 * self.sigma_stim ** 2))
        return _rates * Hz

    def potential_presynaptic_neuron(self, projection, postsynaptic_index):
        potential_neurons = \
            np.nonzero(np.invert(projection.synapse_connected.reshape(self.N, self.N)[:, postsynaptic_index]))[0]
        if len(potential_neurons) == 0:
            return None
        random_index = potential_neurons[np.random.randint(0, len(potential_neurons))]
        return random_index

    def simulate(self, duration=None, dt_=0.1 * ms):
        start = time.time() * second
        if not duration:
            if not self.no_iterations:
                raise EnvironmentError("You have not defined a run time for the simulation")
                duration = self.no_iterations * dt_
        start_scope()
        projections = []
        layers = []
        neuron_dynamics = '''
        dv/dt = (v_rest-v + gex * (e_ext - v))/tau_m : volt
        dgex/dt = -gex/tau_ex: 1
        s : 1
        '''
        G = NeuronGroup(self.N, neuron_dynamics, threshold='v > -54 * mV',
                        reset='v = -70 * mV', method='euler', dt=0.1 * ms, name="target_layer")
        G.v = [-70 * mV, ] * self.N
        G.gex = [0.0, ] * self.N
        G.s = [0, ] * self.N
        layers.append(G)
        if self.recordings['states']:
            statemon = StateMonitor(G, ['v', ], record=True)

        if self.recordings['spikes']:
            spikemon = SpikeMonitor(G)

        # Pre pop
        location = np.random.randint(0, self.n, 2)
        _rates = self.generate_rates(location)
        inp = NeuronGroup(self.N, 'rates : Hz', threshold='rand()<rates*dt', dt=0.1 * ms, name="source_layer")
        layers.append(inp)
        inp.rates = _rates.ravel()

        synapse_model = '''
                         w : 1
                         dapre/dt = -apre/taupre : 1 (event-driven)
                         dapost/dt = -apost/taupost : 1 (event-driven)
                         '''
        on_pre_model = '''
                         gex_post += w
                         apre += Apre
                         w = clip(w+apost, 0, g_max)
                         '''
        on_post_model = '''
                         apost += Apost
                         w = clip(w+apre, 0, g_max)
                         '''

        # Feedforward connections (from source to target)
        feedforward = Synapses(inp, G, synapse_model,
                               on_pre=on_pre_model,
                               on_post=on_post_model,
                               dt=dt_, name="feedforward_projections")
        feedforward.connect()
        feedforward.add_attribute('synapse_connected')
        feedforward.synapse_connected = np.zeros(self.N ** 2, dtype=np.bool_)
        feedforward.w = [0, ] * (self.N ** 2)
        projections.append(feedforward)

        # Lateral connections (from target to target)

        lateral = Synapses(G, G, synapse_model,
                           on_pre=on_pre_model,
                           on_post=on_post_model,
                           dt=dt_, name="lateral_projections")
        lateral.connect()
        lateral.add_attribute('synapse_connected')
        lateral.synapse_connected = np.zeros(self.N ** 2, dtype=np.bool_)
        lateral.w = [0, ] * (self.N ** 2)
        projections.append(lateral)

        def elimination_rule(projection, synapse_index):
            '''
            Delete a synapse based on the elimination probability
            '''
            # Pre is the row index
            pre = synapse_index // self.N
            # Post is the column index
            post = synapse_index % self.N
            # Generate a value to check against form / elimination probabilities
            r = np.random.rand()
            if (projection.w[synapse_index] <= .5 * g_max and r < self.p_elim_dep) or r < self.p_elim_pot:
                if 'feedforward' in projection.name:
                    self.eliminations[0] += 1
                else:
                    self.eliminations[1] += 1
                projection.w[synapse_index] = 0
                projection.synapse_connected[synapse_index] = False
                projection.target.s[post] -= 1

        def formation_rule(projection, synapse_index):
            '''
            Create a new synapse based on the formation probability
            Also, take into account the number of existing presynaptic projections into the current neuron
            '''
            # Pre is the row index
            pre = synapse_index // self.N
            # Post is the column index
            post = synapse_index % self.N
            # Generate a value to check against form / elimination probabilities
            r = np.random.rand()
            if 'feedforward' in projection.name and r < self.p_form_forward * \
                            np.e ** (
                                -(self.distance((pre // self.n, pre % self.n),
                                                (post // self.n, post % self.n)) ** 2) / (
                                        2 * self.sigma_form_forward ** 2)):
                self.formations[0] += 1
                # Form synapse
                projection.w[synapse_index] = g_max
                projection.synapse_connected[synapse_index] = True
                projection.target.s[post] += 1
            elif 'lateral' in projection.name and r < self.p_form_lateral * \
                            np.e ** (
                                -(self.distance((pre // self.n, pre % self.n),
                                                (post // self.n, post % self.n)) ** 2) / (
                                        2 * self.sigma_form_lateral ** 2)):
                self.formations[1] += 1
                # Form synapse
                projection.w[synapse_index] = g_max
                projection.synapse_connected[synapse_index] = True
                projection.target.s[post] += 1

        @network_operation(dt=1. / self.f_rew)
        def rewire():
            '''
            At a fixed rate, a potential synapse is chosen. If it already exists, follow elimination rule,
            else follow formation rule.

            '''
            self.rewire_trigger += 1
            # First, choose a type of projection (FF or LAT)
            projection_index = np.random.randint(0, len(projections))
            # Second, choose a postsynaptic neuron from that projection
            postsynaptic_neuron_index = np.random.randint(0, self.N)

            # potential_pre = self.potential_presynaptic_neuron(projections[projection_index], postsynaptic_neuron_index)
            potential_pre = np.random.randint(0, self.N)
            _2d_to_1d_arithmetic = potential_pre * self.N + postsynaptic_neuron_index
            #         _2d_to_1d_arithmetic = pre_x * N + pre_y
            # Third, check if the synapse exists or not and follow the appropriate rule
            if projections[projection_index].synapse_connected[_2d_to_1d_arithmetic]:
                self.elim_trigger[projection_index] += 1
                elimination_rule(projections[projection_index], _2d_to_1d_arithmetic)
            elif projections[projection_index].target.s[postsynaptic_neuron_index] < self.s_max:
                self.form_trigger[projection_index] += 1
                formation_rule(projections[projection_index], _2d_to_1d_arithmetic)

        @network_operation(dt=self.t_stim)
        def change_stimulus():
            self.stimulus_trigger += 1
            location = np.random.randint(0, self.n, 2)
            _rates = self.generate_rates(location)
            inp.rates = _rates.ravel()

        _initialise = True
        if os.path.isfile(self.recording_filename + ".npz"):
            print "Using previously generated initialization..."
            _initialise = False
            with np.load(self.recording_filename + ".npz")as init_data:
                print init_data.keys()
                G.s = init_data['s']
                feedforward.w = init_data['ff_w']
                feedforward.synapse_connected = init_data['ff_conn']
                lateral.w = init_data['lat_w']
                lateral.synapse_connected = init_data['lat_conn']

        # Initial synapses
        if _initialise:
            for postsynaptic_neuron_index in range(self.N):
                # Place Feedforward first (S_max // 2)
                while G.s[postsynaptic_neuron_index] < self.s_max // 2:
                    potential_pre = self.potential_presynaptic_neuron(projections[0], postsynaptic_neuron_index)
                    _2d_to_1d_arithmetic = potential_pre * self.N + postsynaptic_neuron_index
                    formation_rule(projections[0], _2d_to_1d_arithmetic)

                # Place Lateral now (S_max // 2)
                while G.s[postsynaptic_neuron_index] < self.s_max:
                    potential_pre = self.potential_presynaptic_neuron(projections[1], postsynaptic_neuron_index)
                    _2d_to_1d_arithmetic = potential_pre * self.N + postsynaptic_neuron_index
                    formation_rule(projections[1], _2d_to_1d_arithmetic)
            # Save everything to a file to not have to wait as much next time
            if self.recordings['use_files']:
                print "Saving initialization..."
                np.savez(self.recording_filename,
                         s=G.s.variable.get_value(),
                         ff_w=self.feedforward.w.variable.get_value(),
                         ff_conn=self.feedforward.synapse_connected,
                         lat_w=self.lateral.w.variable.get_value(),
                         lat_conn=self.lateral.synapse_connected)
        init_done = time.time() * second
        print "Initialization done in ", init_done - start
        print "Starting sim"
        run(duration)
        print "Sim done"

        end = time.time() * second
        print "Simulation finished in", end - init_done
        output = (statemon, spikemon, feedforward, lateral)
        self.statemon = statemon
        self.spikemon = spikemon
        self.feedforward = feedforward
        self.lateral = lateral
        return output

    @property
    def target_spike_monitor(self):
        return self.spikemon

    @property
    def feedforward_projection(self):
        return self.feedforward

    @property
    def lateral_projection(self):
        return self.lateral


if __name__ == "__main__":
    duration = 2000*ms
    brian_model = BrianModel(seed=7)
    state = brian_model.simulate(duration=duration)
    print brian_model.target_spike_monitor.num_spikes / (duration)
    print brian_model.spikemon.num_spikes / (duration)
    print brian_model.target_spike_monitor.num_spikes
    print brian_model.spikemon.num_spikes
    print
    print brian_model.rewire_trigger
    print brian_model.elim_trigger
    print brian_model.form_trigger
    print brian_model.formations
    print brian_model.eliminations
    print brian_model.stimulus_trigger

    plot(brian_model.statemon.t / ms, brian_model.statemon.v[0])
    xlabel('Time (ms)')
    ylabel('v')
    show()
