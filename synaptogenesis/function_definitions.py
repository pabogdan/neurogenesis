import collections

import numpy as np
from spinn_utilities.progress_bar import ProgressBar


# +-------------------------------------------------------------------+
# | Function definitions                                              |
# +-------------------------------------------------------------------+


# Periodic boundaries

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


def generate_rates(s, grid, f_base=5., f_peak=152.8, sigma_stim=2.,
                   f_mean=20.):
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


def generate_gaussian_input_rates(s, grid, f_base=5., f_peak=152.8,
                                  sigma_stim=2., f_mean=20.):
    '''
    Function that generates an array the same shape as the input layer so that
    each cell has a value corresponding to the firing rate for the neuron
    at that position.
    '''
    _rates = np.empty(grid)
    for x in range(grid[0]):
        for y in range(grid[1]):
            _d = distance(s, (x, y), grid)
            _rates[x, y] = f_peak * (np.exp(
                (-_d ** 2) / (sigma_stim ** 2 * 2)))

    means_ratio = float(f_mean - f_base) / np.mean(_rates)
    _rates = _rates * means_ratio + f_base

    assert np.isclose(np.mean(_rates), f_mean, 0.01, 0.01), "{} vs. {}".format(
        np.mean(_rates), f_mean)

    return _rates


def generate_multimodal_gaussian_rates(s, grid, f_base=5, f_peak=152.8,
                                       sigma_stim=2, f_mean=20.):
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
                _rates[x, y] += (f_peak * (np.exp(
                    (-_d ** 2) / (sigma_stim ** 2 * 2))))

    for x in range(grid[0]):
        for y in range(grid[1]):
            _rates[x, y] += f_base
    return _rates


def generate_multimodal_rates(s, grid, f_base=5, f_peak=152.8, sigma_stim=2,
                              f_mean=20.):
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


def generate_square_rates(s, grid=np.asarray([16, 16]), f_base=5.,
                          f_peak=152.8, sigma_stim=2., f_mean=20.):
    # f_peak kept for signature compatibility
    rates = np.empty(grid)
    for x in range(grid[0]):
        for y in range(grid[1]):
            rates[x, y] = 1 if distance((x, y), s, grid=grid) <= float(
                sigma_stim) else 0

    scalar_ratio = ((f_mean - f_base) * rates.size) / np.count_nonzero(rates)
    rates = rates * scalar_ratio
    rates += f_base

    assert np.isclose(np.mean(rates), f_mean, 0.01, 0.01), "{} vs. {}".format(
        np.mean(rates), f_mean)

    return rates


def generate_scaled_pointy_rates(s, grid=np.asarray([16, 16]), f_base=5.,
                                 f_peak=152.8, sigma_stim=2., f_mean=20.):
    rates = generate_rates(s,
                           grid=grid,
                           f_base=f_base,
                           f_peak=f_peak,
                           sigma_stim=sigma_stim)
    denoised_rates = rates - f_base
    mean_no_base = np.mean(denoised_rates)
    means_ratio = float(f_mean - f_base) / (mean_no_base)
    final_rates = denoised_rates * means_ratio + f_base

    assert np.isclose(np.mean(final_rates), f_mean), "{} vs {}".format(
        np.mean(final_rates), f_mean)

    return final_rates


def formation_rule(potential_pre, post, sigma, p_form):
    d = distance(potential_pre, post)
    r = np.random.rand()
    p = p_form * (np.e ** (-(d ** 2 / (2 * (sigma ** 2)))))
    if r < p:
        return True
    return False


# Initial connectivity

def generate_initial_connectivity(sigma, p, msg,
                                  N_layer=256, n=16, s_max=16, g_max=.2,
                                  delay=1.):
    # print "|", 256 // 4 * "-", "|"
    # print "|",
    connections = []
    s = np.zeros(N_layer)
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
    return connections


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


import os
import glob
import bz2
import pickle


def load_mnist_rates(in_path, class_idx, min_noise=0, max_noise=0,
                     mean_rate=None, suffix=None):
    from tempfile import mkdtemp
    import os
    import glob

    ON, OFF = 0, 1
    IDX, RATE = 0, 1

    def load_compressed(fname):
        with bz2.BZ2File(fname, 'rb') as f:
            obj = pickle.load(f)
            return obj

    if suffix is not None:
        fnames = glob.glob(os.path.join(in_path, "*%s.pickle.bz2" % suffix))
    else:
        fnames = glob.glob(os.path.join(in_path, "*.pickle.bz2"))

    #     print(fnames)
    on_rates = None
    off_rates = None
    for fname in fnames:
        spl = os.path.basename(fname).split('__')
        cls = int(spl[0].split('_')[1])
        if cls == class_idx:
            n_smpls = int(spl[1].split('_')[0])
            width = int(spl[2].split('_')[1])
            height = int(spl[3].split('_')[1])
            if mean_rate is not None:
                new_rate = float(width * height * mean_rate)
            on_rates, off_rates = None, None
            data = load_compressed(fname)

            tmp_dir = mkdtemp()
            np.random.seed()
            filename = os.path.join(tmp_dir, 'on_rates.dat')
            on_rates = np.memmap(filename, dtype='uint16', mode='w+',
                                 shape=(n_smpls, height, width))
            on_rates[:] = np.round(np.random.uniform(min_noise, max_noise,
                                                     size=(n_smpls, height,
                                                           width))).astype(
                'uint16')
            for i in xrange(len(data[ON][IDX])):
                for idx in data[ON][IDX][i]:
                    r = idx // width
                    c = idx % width
                    if mean_rate is None:
                        on_rates[i, r, c] += data[ON][RATE][i]
                    else:
                        on_rates[i, r, c] += round(
                            new_rate / len(data[ON][IDX][i]))
            on_rates[on_rates < 0] = 0.

            np.random.seed()
            filename = os.path.join(tmp_dir, 'off_rates.dat')
            off_rates = np.memmap(filename, dtype='uint16', mode='w+',
                                  shape=(n_smpls, height, width))
            off_rates[:] = np.round(np.random.uniform(min_noise, max_noise,
                                                      size=(n_smpls, height,
                                                            width))).astype(
                'uint16')
            for i in xrange(len(data[OFF][IDX])):
                for idx in data[OFF][IDX][i]:
                    r = idx // width
                    c = idx % width
                    if mean_rate is None:
                        off_rates[i, r, c] += data[OFF][RATE][i]
                    else:
                        off_rates[i, r, c] += round(
                            new_rate / len(data[OFF][IDX][i]))

            off_rates[off_rates < 0] = 0.
            del data

    return on_rates, off_rates


def generate_rates_very_pointy(s, grid, f_base=5., f_peak=152.8, sigma_stim=2.,
                               f_mean=20.):
    '''
    Function that generates an array the same shape as the input layer so that
    each cell has a value corresponding to the firing rate for the neuron
    at that position.

    THIS FUNCTION DOES NOT PRODUCE THE CORRECT FIRING PATTERN!
    USE ANY OF THE OTHERS!
    '''
    _rates = np.empty(grid)
    for x in range(grid[0]):
        for y in range(grid[1]):
            _d = distance(s, (x, y), grid)
            _rates[x, y] = f_base + (f_peak * (np.exp(
                (-_d) / (2 * sigma_stim ** 2))))
    return _rates


def tile_grating_times(one_cycle, simtime):
    all_gratings = []
    cycle_end = 10000
    for i in range(len(one_cycle)):
        temp_times = np.asarray(one_cycle[i])
        tiled_times = np.tile(temp_times, simtime // cycle_end)
        if tiled_times.size > 0:
            for another_i in range(tiled_times.size // temp_times.size):
                tiled_times[temp_times.size * another_i:temp_times.size * (
                        another_i + 1)] += cycle_end * another_i

        noisy_times = tiled_times + np.random.randint(-1, 2,
                                                      size=tiled_times.size)
        all_gratings.append(noisy_times)
    return all_gratings


def generate_bar_input(simtime, chunk, N_layer, angles=np.arange(0, 360, 5),
                       in_folder="spiking_moving_bar_input", offset=0):
    n = int(np.sqrt(N_layer))
    actual_angles = np.random.choice(angles, int(simtime / chunk))
    spike_times = []
    for _ in range(N_layer):
        spike_times.append([])
    on_files = []
    off_files = []
    for angle in angles:
        if chunk == 200:
            fname = ("/moving_bar_res_%02dx%02d__w_3__angle_%03d__fps_200__"
                     "cycles_0.20s.npz" % (n, n, angle))
        else:
            fname = ("/moving_bar_res_%02dx%02d__w_3__angle_%03d__fps_200__"
                     "cycles_0.40s.npz" % (n, n, angle))
        data = np.load(in_folder + fname)
        on_files.append(data['spikes_on'])
        off_files.append(data['spikes_off'])
        data.close()

    on_spikes = []
    off_spikes = []
    for _ in range(N_layer):
        on_spikes.append([])
        off_spikes.append([])

    for chunk_no in range(int(simtime / chunk)):
        current_angle = actual_angles[chunk_no]
        angle_index = int(np.argwhere(angles == current_angle))

        on_entries = on_files[angle_index] + (chunk_no * (chunk)) + offset
        off_entries = off_files[angle_index] + (chunk_no * (chunk)) + offset

        for index, value in np.ndenumerate(on_entries):
            if not isinstance(value, collections.Iterable) or len(value) == 1:
                on_spikes[index[0]].append(value)

        for index, value in np.ndenumerate(off_entries):
            if not isinstance(value, collections.Iterable) or len(value) == 1:
                off_spikes[index[0]].append(value)

    return actual_angles, on_spikes, off_spikes


def jitter_the_input(ons, offs):
    on_gratings = np.copy(ons)
    final_on_gratings = []
    for row in on_gratings:
        row = np.asarray(row)
        final_on_gratings.append(row + np.random.randint(-1, 2,
                                                         size=row.shape))
    off_gratings = np.copy(offs)
    final_off_gratings = []
    for row in off_gratings:
        row = np.asarray(row)
        final_off_gratings.append(row + np.random.randint(-1, 2,
                                                          size=row.shape))

    return final_on_gratings, final_off_gratings
