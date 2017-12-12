import numpy as np
from spinn_utilities.progress_bar import ProgressBar


# +-------------------------------------------------------------------+
# | Function definitions                                              |
# +-------------------------------------------------------------------+


# Periodic boundaries
# https://github.com/pabogdan/neurogenesis/blob/master/notebooks/neurogenesis-in-numbers/Periodic%20boundaries.ipynb
# def distance(x0, x1, grid=np.asarray([16, 16]), type='euclidian'):
#     x0 = np.asarray(x0)
#     x1 = np.asarray(x1)
#     delta = np.abs(x0 - x1)
#     delta = np.where(delta > grid * .5, delta - grid, delta)
#
#     if type == 'manhattan':
#         return np.abs(delta).sum(axis=-1)
#     return np.sqrt((delta ** 2).sum(axis=-1))
#

def distance(x0, x1, grid=np.asarray([16, 16]), type='euclidian'):
    x0 = np.asarray(x0)
    x1 = np.asarray(x1)
    delta = np.abs(x0 - x1)
    #     delta = np.where(delta > grid * .5, delta - grid, delta)
    #     print delta, grid
    if delta[0] > grid[0] * .5 and grid[0] > 0:
        delta[0] -= grid[0]

    if delta[1] > grid[1] * .5 and grid[1] > 0:
        delta[1] -= grid[1]

    if type == 'manhattan':
        return np.abs(delta).sum(axis=-1)
    return np.sqrt((delta ** 2).sum(axis=-1))


# Poisson spike source as from list spike source
# https://github.com/project-rig/pynn_spinnaker_bcpnn/blob/master/examples/modular_attractor/network.py#L115-L148
def poisson_generator(rate, t_start, t_stop):
    n = int((t_stop - t_start) / 1000.0 * rate)
    number = np.ceil(n + 3 * np.sqrt(n))
    if number < 100:
        number = min(5 + np.ceil(2 * n), 100)

    if number > 0:
        isi = np.random.exponential(1.0 / rate, int(number)) * 1000.0
        if number > 1:
            spikes = np.add.accumulate(isi)
        else:
            spikes = isi
    else:
        spikes = np.array([])

    spikes += t_start
    i = np.searchsorted(spikes, t_stop)

    extra_spikes = []
    if len(spikes) == i:
        # ISI buf overrun

        t_last = spikes[-1] + np.random.exponential(1.0 / rate, 1)[0] * 1000.0

        while (t_last < t_stop):
            extra_spikes.append(t_last)
            t_last += np.random.exponential(1.0 / rate, 1)[0] * 1000.0

            spikes = np.concatenate((spikes, extra_spikes))
    else:
        spikes = np.resize(spikes, (i,))

    # Return spike times, rounded to millisecond boundaries
    unique_rounded_times = np.unique(np.array([round(x) for x in spikes]))
    return unique_rounded_times[unique_rounded_times < t_stop]


def generate_rates(s, grid, f_base=5., f_peak=152.8, sigma_stim=2.):
    '''
    Function that generates an array the same shape as the input layer so that
    each cell has a value corresponding to the firing rate for the neuron
    at that position.
    '''
    _rates = np.empty(grid)
    for x in range(grid[0]):
        for y in range(grid[1]):
            _d = distance(s, (x, y), grid)
            _rates[x, y] = f_base + (f_peak * (np.exp(
                (-_d * 2) / (sigma_stim ** 2))))
    return _rates


def generate_multimodal_rates(s, grid, f_base=5, f_peak=152.8, sigma_stim=2):
    '''
    Function that generates an array the same shape as the input layer so that
    each cell has a value corresponding to the firing rate for the neuron
    at that position.
    '''
    _rates = np.zeros(grid)
    for pos in s:
        for x in range(grid[0]):
            for y in range(grid[1]):
                _d = distance(pos, (x, y), grid)
                _rates[x, y] = f_base + (f_peak * (np.exp(
                    (-_d * 2) / (sigma_stim ** 2))))
    return _rates


def formation_rule(potential_pre, post, sigma, p_form):
    d = distance(potential_pre, post)
    r = np.random.rand()
    p = p_form * (np.e ** (-(d ** 2 / (2 * (sigma ** 2)))))
    if r < p:
        return True
    return False


# Initial connectivity

def generate_initial_connectivity(s, connections, sigma, p, msg,
                                  N_layer=256, n=16, s_max=16, g_max=.2,
                                  delay=1.):
    # print "|", 256 // 4 * "-", "|"
    # print "|",
    pbar = ProgressBar(total_number_of_things_to_do=N_layer,
                       string_describing_what_being_progressed=msg)
    for postsynaptic_neuron_index in range(N_layer):
        # if postsynaptic_neuron_index % 8 == 0:
        #     print "=",
        pbar.update()
        post = (postsynaptic_neuron_index // n, postsynaptic_neuron_index % n)
        while s[postsynaptic_neuron_index] < s_max:
            potential_pre_index = np.random.randint(0, N_layer)
            pre = (potential_pre_index // n, potential_pre_index % n)
            # Commenting this 2 lines to allow for multapses

            # if potential_pre_index not in existing_pre[
            #     postsynaptic_neuron_index]:
            if formation_rule(pre, post, sigma, p):
                s[postsynaptic_neuron_index] += 1
                connections.append((potential_pre_index,
                                    postsynaptic_neuron_index, g_max, delay))
                # print " |"


def generate_equivalent_connectivity(s, connections, sigma, p, msg,
                                  N_layer=256, n=16, g_max=.2,
                                  delay=1.):
    for postsynaptic_neuron_index in range(N_layer):
        post = (postsynaptic_neuron_index // n, postsynaptic_neuron_index % n)
        while s[postsynaptic_neuron_index] > 0:
            potential_pre_index = np.random.randint(0, N_layer)
            pre = (potential_pre_index // n, potential_pre_index % n)
            if formation_rule(pre, post, sigma, p):
                s[postsynaptic_neuron_index] -= 1
                connections.append((potential_pre_index,
                                    postsynaptic_neuron_index, g_max, delay))
