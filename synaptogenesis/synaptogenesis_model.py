from brian2.units import *
import numpy as np
from scipy.spatial.distance import minkowski


class SynaptogenesisModel(object):
    '''
    Simeon Bamford's synaptogenesis VLSI model re-created in Python package. This
    will be the base 'class' for the BRAIN2 and PyNN simulations.
    '''
    CASE_CORR_AND_REW = 1
    CASE_CORR_NO_REW = 2
    CASE_REW_NO_CORR = 3

    def __init__(self, seed=None, **kwargs):
        if 'no_iterations' in kwargs:
            self.no_iterations = kwargs['no_iterations']
        else:
            self.no_iterations = None
        self.seed = None
        if seed:
            self.seed = seed
            np.random.seed(self.seed)

        # TODO -- Add support for recordings directories
        self.recording_filename = str(np.random.randint(100000, 1000000))
        self.recording_directory = ""

        self.dimensions = 2

        self.case = SynaptogenesisModel.CASE_CORR_AND_REW

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
        self.a_plus = 0.1
        self.b = 1.2
        self.tau_plus = 20 * ms
        self.tau_minus = 64 * ms
        self.a_minus = (self.a_plus * self.tau_plus * self.b) / self.tau_minus

        # Recordings
        self.recordings = {'spikes': True,
                           'states': True,
                           'rewiring': True,
                           'use_files': True,
                           'rates': True,
                           }
        self.statemon = None
        self.spikemon = None
        self.ratemon = None

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

    def simulate(self):
        pass

    def load(self, filename=None, directory=None, prefix=None, extension=".npz"):
        # TODO -- Add support for directories
        if not filename:
            filename = self.recording_filename
        if not directory:
            directory = self.recording_directory
        if not prefix:
            prefix = ""
        return np.load(prefix + filename + extension)

    def save(self, filename=None, directory=None, prefix=None, **kwargs):
        # TODO -- Add support for directories
        if not filename:
            filename = self.recording_filename
        if not directory:
            directory = self.recording_directory
        if not prefix:
            prefix = ""
        np.savez(prefix + filename, **kwargs)

    @staticmethod
    def distance(s, t, grid_shape, dimensions):
        '''
        Function that computes distance in a grid of neurons taking into account periodic boundry conditions.

        First, translate source into the center of the grid.
        Second, translate target by the same amount.
        Finally, perform desired distance computation.
        '''
        s = np.asarray(s)
        t = np.asarray(t)
        _grid_size = np.asarray(grid_shape)
        trans = s - (_grid_size // 2)
        s = np.mod(s - trans, _grid_size)
        t = np.mod(t - trans, _grid_size)
        return minkowski(s, t, dimensions)

    def generate_rates(self, s, dimensions=2):
        '''
        Function that generates an array the same shape as the input layer so that
        each cell has a value corresponding to the firing rate for the neuron
        at that position.
        '''
        _rates = np.zeros(self.S)
        for x, y in np.ndindex(self.S):
            _d = self.distance(s, (x, y), grid_shape=self.S, dimensions=dimensions)
            _rates[x, y] = self.f_base + self.f_peak * np.e ** (-_d / (2 * self.sigma_stim ** 2))
        return _rates * Hz

    def generate_spike_times(self, s, dt=.1 * ms, chunk=20 * ms, time_offset=0 * second, dimensions=2):
        rates = self.generate_rates(s, dimensions)
        spike_times = []
        for rate in rates.ravel():
            spikes = np.random.poisson(rate / Hz / 1000., int(chunk / dt))
            spike_times.append(((np.nonzero(spikes)[0] * dt + time_offset)/ms).tolist())
        return spike_times

    def formation_presynaptic_neuron(self, projection, postsynaptic_index):
        potential_neurons = \
            np.nonzero(np.invert(projection.synapse_connected.reshape(self.N, self.N)[:, postsynaptic_index]))[0]
        if len(potential_neurons) == 0:
            return None
        random_index = potential_neurons[np.random.randint(0, len(potential_neurons))]
        return random_index

    def statistics(self):
        # TODO
        raise NotImplementedError()

    def record(self, name, value=True):
        '''
        Tell the simulation which values are of interest and should be monitored.
        :param name: Name of recording
        :type name: str
        :param value: Recording On/Off for specified recording
        :type value: bool, Default=True
        '''
        self.recordings[name] = value
