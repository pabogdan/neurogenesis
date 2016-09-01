from unittest import TestCase
from synaptogenesis.pynn_model import PyNNModel
import numpy as np
from brian2.units import *


class TestPyNNModel(TestCase):
    def test_simulate(self):
        self.fail()

    def test_set_spike_times(self):
        self.fail()

    def test_generate_spike_times_single_chunk_1(self):
        '''
        Should generate a sequence of times (floats) for spikes. This sequence *HAS* to be strictly monotonic and
        ascending. Should generate this kind of sequence for each neuron and place all of them in a list.

        '''
        pynn_model = PyNNModel(N=1)
        spike_times = pynn_model.generate_spike_times((0,0))
        times = np.asarray(spike_times).ravel()
        self.assertTrue(all(times[i] <= times[i + 1] for i in xrange(len(times) - 1)),
                        "Times are not ordered in a single chunk")

    def test_generate_spike_times_single_chunk_256(self):
        pynn_model = PyNNModel(N=256)
        spike_times = pynn_model.generate_spike_times((0,0))
        times_array = np.asarray(spike_times)
        for times in times_array.ravel():
            self.assertTrue(all(times[i] <= times[i + 1] for i in xrange(len(times) - 1)),
                            "Times are not ordered in a single chunk")

    def test_generate_spike_times_2_chunks_1(self):
        pynn_model = PyNNModel(N=1)
        spike_times = pynn_model.generate_spike_times((0, 0))
        spike_times.append(pynn_model.generate_spike_times((0, 0), chunk=10*ms))
        times = np.asarray(spike_times).ravel()
        self.assertTrue(all(times[i] <= times[i + 1] for i in xrange(len(times) - 1)),
                        "Times are not ordered in a single chunk")

    def test_generate_spike_times_2_chunks_256(self):
        pynn_model = PyNNModel(N=256)
        spike_times = pynn_model.generate_spike_times((0, 0))
        _temp_spikes = pynn_model.generate_spike_times((5, 5), chunk=10 * ms)
        for index, value in np.ndenumerate(_temp_spikes):
            spike_times[index[0]].append(value)
        times_array = np.asarray(spike_times)
        for times in times_array.ravel():
            self.assertTrue(all(times[i] <= times[i + 1] for i in xrange(len(times) - 1)),
                            "Times are not ordered in a single chunk")
