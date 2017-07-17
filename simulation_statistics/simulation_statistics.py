import argparse
from collections import Iterable

import numpy as np
import matplotlib.pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import scipy
import scipy.stats as stats
from glob import glob
from pprint import pprint as pp

parser = argparse.ArgumentParser(
    description='Module to analyse the quality of the topographic maps generated by SpiNNaker')
parser.add_argument('path', help='path of .npz archive', nargs='*')
parser.add_argument('-o', '--output', type=str, help="name of the numpy archive storing simulation results",
                    dest='filename')

args = parser.parse_args()

# Wiring
n = 16
N_layer = n ** 2
S = (n, n)
# S = (256, 1)
grid = np.asarray(S)


# Function definitions

def distance(x0, x1, grid=np.asarray([16, 16]), type='euclidian'):
    x0 = np.asarray(x0)
    x1 = np.asarray(x1)
    delta = np.abs(x0 - x1)
    delta = np.where(delta > grid * .5, delta - grid, delta)

    if type == 'manhattan':
        return np.abs(delta).sum(axis=-1)
    return np.sqrt((delta ** 2).sum(axis=-1))


def sigma_and_ad(connectivity_matrix, unitary_weights=False, N_layer=256, n=16, resolution=1.):
    # Datastructure setup
    connectivity_matrix = np.copy(connectivity_matrix)
    variances = np.ones((N_layer, int(N_layer * resolution))) * np.nan
    preferred_locations = np.ones((n, n)) * np.nan
    active_synapses_indices = np.where(np.isfinite(connectivity_matrix))
    if unitary_weights:
        connectivity_matrix[active_synapses_indices] = 1.
    # Ignore the argmin, first just compute weighted variances for all possible source locations
    for target_location in xrange(N_layer):  # np.ndindex(16,16):
        for source_location in xrange(int(N_layer * resolution)):  # np.ndindex(16,16):
            # Distance must be computed between source_Location and the location of all presynaptic_neurons |pix|
            possible_sources = np.argwhere(active_synapses_indices[1] == target_location).ravel()
            top_sum = 0
            sum_of_weights = 0
            for p_s in possible_sources:
                other_source = active_synapses_indices[0][p_s]
                distances = distance(((source_location / resolution) // n, (source_location / resolution) % n),
                                     (other_source // n, other_source % n),
                                     grid) ** 2
                top_sum += connectivity_matrix[other_source, target_location] * distances
                sum_of_weights += connectivity_matrix[other_source, target_location]

            variances[target_location, source_location] = top_sum / sum_of_weights
    min_variances = np.nanmin(variances, axis=1).reshape(16, 16)
    #     print min_variances.shape
    stds = np.sqrt(min_variances)
    preferred_indices = np.argmin(variances, axis=1)
    AD = np.ones(int(N_layer)) * np.nan
    for index in range(AD.size):
        AD[index] = distance(((index) // n, (index) % n), (
            (preferred_indices[index] / resolution) // n, (preferred_indices[index] / resolution) % n),
                             grid)

    # return mean std, stds, mean AD, ADs
    return np.mean(stds), stds, np.mean(AD), AD, min_variances


def formation_rule(potential_pre, post, sigma, p_form):
    d = distance(potential_pre, post)
    r = np.random.rand()
    p = p_form * (np.e ** (-(d ** 2 / (2 * (sigma ** 2)))))
    if r < p:
        return True
    return False


def generate_initial_connectivity(s, layer_size, sigma, p, weight):
    preexisting = []
    for _ in range(layer_size):
        preexisting.append([])

    if not isinstance(s, Iterable):
        s = np.ones(layer_size) * s

    current_s = np.zeros(layer_size)

    connectivity_matrix = np.ones((layer_size, layer_size)) * np.nan
    for postsynaptic_neuron_index in range(layer_size):
        post = (postsynaptic_neuron_index // n, postsynaptic_neuron_index % n)
        while current_s[postsynaptic_neuron_index] < s[postsynaptic_neuron_index]:
            potential_pre_index = np.random.randint(0, N_layer)
            pre = (potential_pre_index // n, potential_pre_index % n)
            if potential_pre_index not in preexisting[postsynaptic_neuron_index]:
                if formation_rule(pre, post, sigma, p):
                    current_s[postsynaptic_neuron_index] += 1
                    preexisting[postsynaptic_neuron_index].append(
                        potential_pre_index)
                    connectivity_matrix[potential_pre_index, postsynaptic_neuron_index] = weight
    return connectivity_matrix


def weight_shuffle(connectivity_matrix):
    positions_to_be_shuffled = np.argwhere(np.isfinite(connectivity_matrix))
    permutation = np.random.permutation(positions_to_be_shuffled)
    permuted_connectivity_matrix = np.ones(connectivity_matrix.shape) * np.nan
    for index in range(positions_to_be_shuffled.shape[0]):
        permuted_connectivity_matrix[positions_to_be_shuffled[index][0], positions_to_be_shuffled[index][1]] = \
        connectivity_matrix[permutation[index][0], permutation[index][1]]
    return permuted_connectivity_matrix


paths = []
for file in args.path:
    if "*" in file:
        globbed_files = glob(file)
        for globbed_file in globbed_files:
            if "npz" in globbed_file:
                paths.append(globbed_file)
    else:
        paths.append(file)
for file in paths:
    try:
        start_time = pylab.datetime.datetime.now()
        print "\n\nAnalysing file", str(file)
        data = np.load(file)
        simdata = np.array(data['sim_params']).ravel()[0]

        if 'case' in simdata:
            print "Case", simdata['case'], "analysis"
        else:
            print "Case unknown"
        simtime = int(data['simtime'])
        post_spikes = data['post_spikes']

        count_spikes = np.zeros(256)
        for id, time in post_spikes:
            count_spikes[int(id)] += 1

        target_neuron_mean_spike_rate = count_spikes / float(simtime) * 1000.

        total_target_neuron_mean_spike_rate = np.mean(target_neuron_mean_spike_rate)

        ff_last = data['final_pre_weights'].reshape(256, 256)
        lat_last = data['final_post_weights'].reshape(256, 256)
        init_ff_weights = data['init_ff_connections']
        init_lat_weights = data['init_lat_connections']
        g_max = simdata['g_max']

        try:
            # retrieve some important sim params
            grid = simdata['grid']
            s_max = simdata['s_max']
            sigma_form_forward = simdata['sigma_form_forward']
            sigma_form_lateral = simdata['sigma_form_lateral']
            p_form_lateral = simdata['p_form_lateral']
            p_form_forward = simdata['p_form_forward']
            p_elim_dep = simdata['p_elim_dep']
            p_elim_pot = simdata['p_elim_pot']
            f_rew = simdata['f_rew']
        except:
            # use defaults
            grid = np.asarray([16, 16])
            s_max = 16
            sigma_form_forward = 2.5
            sigma_form_lateral = 1
            p_form_lateral = 1
            p_form_forward = 0.16
            p_elim_dep = 0.0245
            p_elim_pot = 1.36 * np.e ** -4
            f_rew = 10 ** 4  # Hz

        number_ff_incoming_connections = np.count_nonzero(np.isfinite(ff_last), axis=0)
        final_mean_number_ff_synapses = np.mean(number_ff_incoming_connections)

        initial_weight_mean = np.nanmean(init_ff_weights)

        final_weight_mean = np.nanmean(ff_last)

        final_weight_proportion = final_weight_mean / initial_weight_mean

        # a
        init_mean_std, init_stds, init_mean_AD, init_AD, init_min_variances = sigma_and_ad(
            init_ff_weights,
            unitary_weights=True,
            resolution=1.)
        # b
        fin_mean_std_conn, fin_stds_conn, fin_mean_AD_conn, fin_AD_conn, fin_min_variances_conn = sigma_and_ad(
            ff_last,
            unitary_weights=True,
            resolution=1.)

        # c
        generated_ff_conn = generate_initial_connectivity(
            number_ff_incoming_connections, grid[0] * grid[1],
            sigma_form_forward, p_form_forward, g_max)

        fin_mean_std_conn_shuf, fin_stds_conn_shuf, fin_mean_AD_conn_shuf, fin_AD_conn_shuf, fin_min_variances_conn_shuf = sigma_and_ad(
            generated_ff_conn,
            unitary_weights=True,
            resolution=1.)

        wsr_sigma_fin_conn_fin_conn_shuffle = stats.wilcoxon(fin_stds_conn.ravel(), fin_stds_conn_shuf.ravel())
        wsr_AD_fin_conn_fin_conn_shuffle = stats.wilcoxon(fin_AD_conn.ravel(), fin_AD_conn_shuf.ravel())
        # d
        fin_mean_std_weight, fin_stds_weight, fin_mean_AD_weight, fin_AD_weight, fin_min_variances_weight = sigma_and_ad(
            ff_last,
            unitary_weights=False,
            resolution=1.)
        
        # e
        shuf_weights = weight_shuffle(ff_last)
        fin_mean_std_weight_shuf, fin_stds_weight_shuf, fin_mean_AD_weight_shuf, fin_AD_weight_shuf, fin_min_variances_weight_shuf = sigma_and_ad(
            shuf_weights,
            unitary_weights=False,
            resolution=1.)
        wsr_sigma_fin_conn_fin_conn_shuffle = stats.wilcoxon(fin_stds_weight.ravel(), fin_stds_weight_shuf.ravel())
        wsr_AD_fin_conn_fin_conn_shuffle = stats.wilcoxon(fin_AD_weight.ravel(), fin_AD_weight_shuf.ravel())

        pp(simdata)
        print "'%-60s'" % "Target neuron spike rate", total_target_neuron_mean_spike_rate, "Hz"
        print "'%-60s'" % "Final mean number of feedforward synapses", final_mean_number_ff_synapses
        print "'%-60s'" % "Initial ff weight mean", initial_weight_mean, "(should be .2, obviously)"
        print "'%-60s'" % "Final ff weight mean", final_weight_mean
        print "'%-60s'" % "Weight as proportion of max", final_weight_proportion
        print "'%-60s'" % "Mean sigma aff init", init_mean_std
        print "'%-60s'" % "Mean sigma aff fin conn shuffle", fin_mean_std_conn_shuf
        print "'%-60s'" % "Mean sigma aff fin conn", fin_mean_std_conn
        print "'%-60s'" % "p(WSR sigma aff fin conn vs sigma aff fin conn shuffle)", wsr_sigma_fin_conn_fin_conn_shuffle.pvalue
        print "'%-60s'" % "Mean sigma aff fin weight shuffle", fin_mean_std_weight_shuf
        print "'%-60s'" % "Mean sigma aff fin weight", fin_mean_std_weight
        print "'%-60s'" % "p(WSR sigma aff fin weight vs sigma aff fin weight shuffle)", wsr_sigma_fin_conn_fin_conn_shuffle.pvalue
        print "'%-60s'" % "Mean AD init", init_mean_AD
        print "'%-60s'" % "Mean AD fin conn shuffle", fin_mean_AD_conn_shuf
        print "'%-60s'" % "Mean AD fin conn", fin_mean_AD_conn
        print "'%-60s'" % "p(WSR AD fin conn vs AD fin conn shuffle)", wsr_AD_fin_conn_fin_conn_shuffle.pvalue
        print "'%-60s'" % "Mean AD fin weight shuffle", fin_mean_AD_weight_shuf
        print "'%-60s'" % "Mean AD fin weight", fin_mean_AD_weight
        print "'%-60s'" % "p(WSR AD fin weight vs AD fin weight shuffle)", wsr_AD_fin_conn_fin_conn_shuffle.pvalue

        end_time = pylab.datetime.datetime.now()
        suffix = end_time.strftime("_%H%M%S_%d%m%Y")

        elapsed_time = end_time - start_time

        print "Total time elapsed -- " + str(elapsed_time)

        if args.filename:
            filename = args.filename
        else:
            filename = "analysis_" + str(suffix)

        np.savez(filename, recording_archive_name=file,
                 target_neurom_mean_spike_rate=target_neuron_mean_spike_rate,
                 final_mean_number_ff_synapses=final_mean_number_ff_synapses,
                 final_weight_proportion=final_weight_proportion,
                 init_ff_weights=init_ff_weights,
                 init_lat_connections=init_lat_weights,
                 final_pre_weights=ff_last,
                 final_post_weights=lat_last,
                 init_mean_std=init_mean_std, init_stds=init_stds, init_mean_AD=init_mean_AD,
                 init_AD=init_AD, init_min_variances=init_min_variances,
                 total_time=elapsed_time)
    except Exception as e:
        print "Error:", e
