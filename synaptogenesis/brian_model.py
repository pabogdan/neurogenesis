import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
import time
import os
from synaptogenesis_model import SynaptogenesisModel

plt.viridis()
plt.close()


class BrianModel(SynaptogenesisModel):
    '''
    Simeon Bamford's synaptogenesis VLSI model re-created in Brian
    '''

    def __init__(self, seed=None, **kwargs):
        super(BrianModel, self).__init__(seed=seed, **kwargs)
        for variable_name, variable_value in kwargs.items():
            if not hasattr(self, variable_name):
                raise AttributeError(
                    "You are trying to modify a simulation parameter that doesn't exist -- \"" + variable_name + "\"")
            setattr(self, variable_name, variable_value)


    def simulate(self, duration=None, dt_=0.1 * ms):
        start = time.time() * second
        if not duration:
            if not self.no_iterations:
                raise EnvironmentError("You have not defined a run time for the simulation")
            duration = self.no_iterations * dt_
        start_scope()
        # Simulation parameters
        # Membrane
        v_rest = self.v_rest  # mV
        e_ext = self.e_ext  # V
        v_thr = self.v_thr  # mV
        g_max = self.g_max
        tau_m = self.tau_m  # ms
        tau_ex = self.tau_ex  # ms
        e = self.e
        pre_t = self.pre_t

        # STDP
        Apre = self.a_plus
        b = self.b
        taupre = self.tau_plus
        taupost = self.tau_minus
        Apost = (self.a_plus * self.tau_plus * self.b) / self.tau_minus

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
        statemon = None
        if self.recordings['states']:
            statemon = StateMonitor(G, ['v', ], record=True)
        spikemon = None
        if self.recordings['spikes']:
            spikemon = SpikeMonitor(G)
        ratemon = None
        if self.recordings['rates']:
            ratemon = PopulationRateMonitor(G)

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

        self.statemon = statemon
        self.spikemon = spikemon
        self.feedforward = feedforward
        self.lateral = lateral
        self.ratemon = ratemon

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
                                                (post // self.n, post % self.n), grid_shape=self.S,
                                                dimensions=self.dimensions) ** 2) / (
                                        2 * self.sigma_form_forward ** 2)):
                self.formations[0] += 1
                # Form synapse
                projection.w[synapse_index] = g_max
                projection.synapse_connected[synapse_index] = True
                projection.target.s[post] += 1
            elif 'lateral' in projection.name and r < self.p_form_lateral * \
                            np.e ** (
                                -(self.distance((pre // self.n, pre % self.n),
                                                (post // self.n, post % self.n), grid_shape=self.S,
                                                dimensions=self.dimensions) ** 2) / (
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
            if self.case == SynaptogenesisModel.CASE_CORR_AND_REW or self.case == SynaptogenesisModel.CASE_REW_NO_CORR:
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
            if self.case == SynaptogenesisModel.CASE_CORR_AND_REW or self.case == SynaptogenesisModel.CASE_CORR_NO_REW:
                self.stimulus_trigger += 1
                location = np.random.randint(0, self.n, 2)
                _rates = self.generate_rates(location)
                inp.rates = _rates.ravel()

        _initialise = True
        # TODO -- Add support for directories
        if os.path.isfile(self.recording_filename + ".npz"):
            print "Using previously generated initialization..."
            _initialise = False
            with self.load() as init_data:
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
                    potential_pre = self.formation_presynaptic_neuron(projections[0], postsynaptic_neuron_index)
                    _2d_to_1d_arithmetic = potential_pre * self.N + postsynaptic_neuron_index
                    formation_rule(projections[0], _2d_to_1d_arithmetic)

                # Place Lateral now (S_max // 2)
                while G.s[postsynaptic_neuron_index] < self.s_max:
                    potential_pre = self.formation_presynaptic_neuron(projections[1], postsynaptic_neuron_index)
                    _2d_to_1d_arithmetic = potential_pre * self.N + postsynaptic_neuron_index
                    formation_rule(projections[1], _2d_to_1d_arithmetic)
            # Save everything to a file to not have to wait as much next time
            if self.recordings['use_files']:
                print "Saving initialization..."
                self.save(s=G.s.variable.get_value(),
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
        # Save simulation results to a file

        output = (statemon, spikemon, feedforward, lateral)
        if self.recordings['use_files']:
            print "Saving simulation data..."
            self.save(prefix="save-",
                      simulator="brian",
                      state=self.statemon.v,
                      spikes=self.spikemon.spike_trains(),
                      rates=self.ratemon.smooth_rate(window='flat', width=0.5 * ms),
                      ff_w=self.feedforward.w.variable.get_value(),
                      ff_conn=self.feedforward.synapse_connected,
                      lat_w=self.lateral.w.variable.get_value(),
                      lat_conn=self.lateral.synapse_connected,
                      final_s=G.s.variable.get_value(),
                      duration=duration)

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

    def statistics(self):
        pass


if __name__ == "__main__":
    case = 1
    # while case < 4:
    duration = 200 * ms
    brian_model = BrianModel(seed=7, dimensions=1, case=case)
    state = brian_model.simulate(duration=duration)
    print brian_model.target_spike_monitor.num_spikes / (duration) / (16 ** 2)
    print brian_model.spikemon.num_spikes / (duration) / (16 ** 2)
    print brian_model.target_spike_monitor.num_spikes / (16 ** 2)
    print brian_model.spikemon.num_spikes
    print
    print brian_model.rewire_trigger
    print brian_model.elim_trigger
    print brian_model.form_trigger
    print brian_model.formations
    print brian_model.eliminations
    print brian_model.stimulus_trigger

    print "Final mean target spike rate", \
        np.mean(brian_model.ratemon.smooth_rate(window='flat', width=duration) / (16. ** 2))

    subplot(211)
    plot(brian_model.statemon.t / ms, brian_model.statemon.v[0])
    xlim(0, duration / ms)

    xlabel('Time (ms)')
    ylabel('v')

    subplot(212)
    plot(brian_model.ratemon.t / ms, brian_model.ratemon.smooth_rate(window='flat', width=0.1 * ms) / Hz / (16. ** 2))
    xlim(0, duration / ms)
    show()
