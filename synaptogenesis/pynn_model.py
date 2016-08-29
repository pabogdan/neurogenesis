try:
    import pyNN.spiNNaker as p
except Exception as e:
    import spynnaker.pyNN as p
import matplotlib.pyplot as plt
from synaptogenesis_model import SynaptogenesisModel
import numpy as np
import time
import os
from brian2.units import *


class PyNNModel(SynaptogenesisModel):
    '''
    Simeon Bamford's synaptogenesis VLSI model re-created in PyNN / sPyNNaker
    '''

    def __init__(self, timestep=0.1 * ms, min_delay=0.1 * ms, max_delay=0.3 * ms, seed=None, **kwargs):
        super(PyNNModel, self).__init__(seed=seed, **kwargs)
        p.setup(timestep=timestep / ms, min_delay=min_delay / ms, max_delay=max_delay / ms)
        p.set_number_of_neurons_per_core("IF_cond_exp", self.N / 2)

        # Class variables for PyNN simulation variables
        self.cm = 1.0 * mfarad
        self.i_offset = 0. * amp
        self.tau_refrac = 0. * ms
        self.tau_syn_E = 5. * ms
        self.tau_syn_I = 5. * ms
        self.v_reset = -70. * mV
        self.e_rev_E = 0. * mV
        self.e_rev_I = -80. * mV

        self.w = 0.014 * amp
        self.delay = 0.1 * ms
        # Equate PyNN variables with Sim's variables

        self.equivalences = {
            'cm': None,
            'i_offset': None,
            'tau_refrac': None,
            'tau_syn_E': 'tau_ex',
            'tau_syn_I': None,
            'v_reset': None,
            'v_rest': 'v_rest',
            'v_thresh': 'v_thr',
            'e_rev_E': 'e_ext',
            'e_rev_I': None
        }

        # Network
        # self.layers = []
        # self.projections = []

        self.spike_times = []
        self.spike_array = {'spike_times': self.spike_times}

        self.target = None
        self.source = None

        for variable_name, variable_value in kwargs.items():
            if not hasattr(self, variable_name):
                raise AttributeError(
                    "You are trying to modify a simulation parameter that doesn't exist -- \"" + variable_name + "\"")
            if variable_name in self.equivalences and self.equivalences[variable_name]:
                setattr(self, self.equivalences[variable_name], variable_value)
            setattr(self, variable_name, variable_value)

    def network_setup(self):
        cell_params_lif = {
            'cm': self.cm / mfarad,
            'tau_m': self.tau_m / ms,
            'tau_refrac': self.tau_refrac / ms,
            'tau_syn_E': self.tau_ex / ms,
            'v_reset': self.v_rest / mV,
            'v_rest': self.v_rest / mV,
            'v_thresh': self.v_thr / mV,
            'e_rev_E': self.e_ext / mV,
        }
        # TODO -- Add variables to Pop and Proj that I need (s vs. s_max, synapse_connected)
        self.target = p.Population(self.N, p.IF_cond_exp, cell_params_lif, label='target_layer')
        self.source = p.Population(self.N, p.SpikeSourceArray, self.spike_array, label='source_layer')

        self.feedforward = p.Projection(self.source, self.target, p.AllToAllConnector(weights=[.2, ] * self.N))
        self.lateral = p.Projection(self.target, self.target, p.AllToAllConnector(weights=[0., ] * self.N))

        if self.recordings['states']:
            self.target.record_v()
            self.target.record_gsyn()
        if self.recordings['spikes']:
            self.target.record()

    def simulate(self, duration=100 * ms):
        self.network_setup()
        p.run(duration/ms)

    def statistics(self):
        pass

    def end(self):
        p.end()

    def set_spike_times(self, spike_times):
        self.spike_times = spike_times
        self.spike_array['spike_times'] = spike_times

    def __enter__(self):
        raise NotImplementedError()

    def __exit__(self, type, value, traceback):
        raise NotImplementedError()


if __name__ == "__main__":
    case = 1
    duration = 100 * ms
    rate = 157.8 * Hz
    spike_times = np.linspace(0, duration, num=rate * duration)/ms
    pynn_model = PyNNModel(seed=6, dimensions=1, case=case, N=1)
    pynn_model.set_spike_times(spike_times)
    pynn_model.simulate(duration)

    v = pynn_model.target.get_v(compatible_output=True)
    gsyn = pynn_model.target.get_gsyn(compatible_output=True)
    spikes = pynn_model.target.getSpikes(compatible_output=True)

    if spikes is not None:
        print spikes
        plt.figure()
        plt.plot([i[1] for i in spikes], [i[0] for i in spikes], ".")
        plt.xlabel('Time/ms')
        plt.ylabel('spikes')
        plt.title('spikes')
        plt.show()
    else:
        print "No spikes received"

    if v is not None:
        ticks = len(v) / pynn_model.N
        plt.figure()
        plt.xlabel('Time/ms')
        plt.ylabel('v')
        plt.title('v')
        for pos in range(0, pynn_model.N, 20):
            v_for_neuron = v[pos * ticks: (pos + 1) * ticks]
            plt.plot([i[2] for i in v_for_neuron])
        plt.show()

    if gsyn is not None:
        ticks = len(gsyn) / pynn_model.N
        plt.figure()
        plt.xlabel('Time/ms')
        plt.ylabel('gsyn')
        plt.title('gsyn')
        for pos in range(0, pynn_model.N, 20):
            gsyn_for_neuron = gsyn[pos * ticks: (pos + 1) * ticks]
            plt.plot([i[2] for i in gsyn_for_neuron])
        plt.show()
    pynn_model.end()
