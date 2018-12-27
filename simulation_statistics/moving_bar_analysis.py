from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm as cm_mlib
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import scipy
from matplotlib import animation, rc, colors
import brian2.units as bunits
import matplotlib as mlib
from scipy import stats
from pprint import pprint as pp
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
import traceback
import os
from argparser import *
from gari_analysis_functions import *
from analysis_functions_definitions import *
from synaptogenesis.function_definitions import generate_equivalent_connectivity
from gari_analysis_functions import get_filtered_dsi_per_neuron
import copy
# imports related to Elephant analysis
from elephant import statistics, spade, spike_train_correlation, spike_train_dissimilarity, conversion
import elephant.cell_assembly_detection as cad
import neo
from datetime import datetime
from quantities import s, ms, Hz
import pandas as pd

# ensure we use viridis as the default cmap
plt.viridis()

# ensure we use the same rc parameters for all matplotlib outputs
mlib.rcParams.update({'font.size': 24})
mlib.rcParams.update({'errorbar.capsize': 5})
mlib.rcParams.update({'figure.autolayout': True})

# define better cyclical cmap
# https://gist.github.com/MatthewJA/5a0a6d75748bf5cb5962cb9d5572a6ce
cyclic_viridis = colors.LinearSegmentedColormap.from_list(
    'cyclic_viridis',
    [(0, cm_mlib.viridis.colors[0]),
     (0.25, cm_mlib.viridis.colors[256 // 3]),
     (0.5, cm_mlib.viridis.colors[2 * 256 // 3]),
     (0.75, cm_mlib.viridis.colors[-1]),
     (1.0, cm_mlib.viridis.colors[0])])

# some defaults
# root_stats = "C:\Work\phd\simulation_statistics\\"
# root_syn = "C:\Work\phd\synaptogenesis\\"
root_stats = args.root_stats
root_syn = args.root_syn
fig_folder = args.fig_folder
testing_data = np.load(
    os.path.join(root_syn, "spiking_moving_bar_input", "spiking_moving_bar_motif_bank_simtime_1200s.npz"))
# check if the figures folder exist
if not os.path.isdir(fig_folder) and not os.path.exists(fig_folder):
    os.mkdir(fig_folder)


# TODO make this useful and use it!
def save_figure(fig, filename, suffix, fig_folder=fig_folder, extensions=["pdf", "svg"]):
    # TODO check that extensions is iterable
    for ext in extensions:
        full_filename = filename + suffix + "." + ext
        plt.savefig(
            fig_folder + full_filename,
            bbox_inches='tight')


def print_and_save_text(message, location):
    '''
    Print to terminal and save text to a file to be able to view statistics after running the analysis
    :param message:
    :type message:
    :param location:
    :type location:
    :return:
    :rtype:
    '''
    pass


def generate_suffix(training_angles):
    unique_tas = np.unique(training_angles)
    no_training_angles = unique_tas.size
    if unique_tas.size != 72:
        suffix_test = "_" + str(no_training_angles)
    else:
        suffix_test = "_all"
    if no_training_angles == 1:
        suffix_test += "_angle"
    else:
        suffix_test += "_angles"
    if unique_tas.size <= 4:
        for ta in unique_tas:
            suffix_test += "_" + str(ta)
    print("The suffix for this set of figures is ", suffix_test)
    return suffix_test


def analyse_one(archive, out_filename=None, extra_suffix=None, show_plots=False):
    # in the default case, we are only looking at understanding a number of
    # behaviours of a single simulation
    cached_data = np.load(archive + ".npz")

    # load all the data
    rate_means = cached_data['rate_means']
    rate_stds = cached_data['rate_stds']
    rate_sem = cached_data['rate_sem']
    all_rates = cached_data['all_rates']
    radians = cached_data['radians']
    instaneous_rates = cached_data['instaneous_rates']
    per_neuron_instaneous_rates = cached_data['per_neuron_instaneous_rates']
    angles = cached_data['angles']
    actual_angles = cached_data['actual_angles']
    target_neuron_mean_spike_rate = cached_data[
        'target_neuron_mean_spike_rate']

    # Connection information
    ff_connections = cached_data['ff_connections']
    lat_connections = cached_data['lat_connections']
    noise_connections = cached_data['noise_connections']
    ff_off_connections = cached_data['ff_off_connections']

    final_ff_conn_field = cached_data['final_ff_conn_field']
    final_ff_num_field = cached_data['final_ff_num_field']
    final_lat_conn_field = cached_data['final_lat_conn_field']
    final_lat_num_field = cached_data['final_lat_num_field']

    ff_last = cached_data['ff_last']
    off_last = cached_data['off_last']
    noise_last = cached_data['noise_last']
    lat_last = cached_data['lat_last']

    per_neuron_all_rates = cached_data['per_neuron_all_rates']

    try:
        # this is not backwards compatible
        # Inhibitory info
        inh_rate_means = cached_data['inh_rate_means']
        inh_rate_stds = cached_data['inh_rate_stds']
        inh_rate_sem = cached_data['inh_rate_sem']
        inh_all_rates = cached_data['inh_all_rates']
        inh_instaneous_rates = cached_data['inh_instaneous_rates']
        inh_per_neuron_instaneous_rates = cached_data[
            'inh_per_neuron_instaneous_rates']
        inh_per_neuron_all_rates = cached_data['inh_per_neuron_all_rates']

        lat_num_network = cached_data['lat_num_network']
        ff_num_network = cached_data['ff_num_network']

        inh_lat_last = cached_data['inh_lat_last']
        exh_to_inh_last = cached_data['exh_to_inh_last']
        inh_to_exh_last = cached_data['inh_to_exh_last']
        inh_ff_last = cached_data['inh_ff_last']
        inh_off_last = cached_data['inh_off_last']
        inh_noise_last = cached_data['inh_noise_last']
    except Exception as e:
        print("Something failed!!!!!")
        traceback.print_exc()
        inh_rate_means = []
        inh_rate_stds = []
        inh_rate_sem = []
        inh_all_rates = []
        inh_per_neuron_all_rates = []
        inh_per_neuron_instaneous_rates = np.asarray([])
        inh_instaneous_rates = np.asarray([])
        lat_num_network = []
        ff_num_network = []
    try:
        sim_params = np.array(cached_data['testing_sim_params']).ravel()[0]
        training_sim_params = \
            np.array(cached_data['training_sim_params']).ravel()[0]
        grid = sim_params['grid']
        N_layer = grid[0] * grid[1]
        n = grid[0]
        g_max = sim_params['g_max']
    except:
        print("Something failed!!!!!")
        sim_params = {}
        N_layer = 32 ** 2
        n = int(np.sqrt(N_layer))
        grid = [n, n]
        g_max = .1
    s_max = training_sim_params['s_max']

    training_angles = training_sim_params['training_angles']
    suffix_test = generate_suffix(training_sim_params['training_angles'])
    cached_data.close()
    if extra_suffix:
        suffix_test += "_" + extra_suffix

    print("{:45}".format("Analysing a single experiment"), ":", archive)
    print("{:45}".format("Results appended the following suffix"), ":", suffix_test)
    # Begin the plotting
    # response histograms
    fig = plt.figure(figsize=(16, 8), dpi=600)
    y, binEdges = np.histogram(angles, bins=angles.size)
    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    width = 5
    plt.bar(angles, rate_means, width=width, yerr=rate_sem,
            color='#414C82', edgecolor='k')
    plt.xlabel("Degree")
    plt.ylabel("Hz")
    plt.savefig(fig_folder + "firing_rate_with_angle_hist{}.pdf".format(
        suffix_test))
    plt.close(fig)
    fig = plt.figure(figsize=(16, 8), dpi=600)
    y, binEdges = np.histogram(angles, bins=angles.size)
    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    width = 5
    plt.bar(angles - 180, np.roll(rate_means, 180 // 5), width=width,
            yerr=np.roll(rate_sem, 180 // 5),
            color='#414C82', edgecolor='k')
    plt.xlabel("Degree")
    plt.ylabel("Hz")
    plt.savefig(
        fig_folder + "firing_rate_with_angle_hist_centred{}.pdf".format(
            suffix_test))
    if show_plots:
        plt.show()
    plt.close(fig)
    # mean excitatory population response in Hz
    fig = plt.figure(figsize=(10, 10), dpi=600)
    ax = plt.subplot(111, projection='polar')
    N = 150
    r = 2 * np.random.rand(N)
    theta = 2 * np.pi * np.random.rand(N)
    c = plt.scatter(radians, rate_means, c=rate_sem,
                    s=100)
    c.set_alpha(0.8)
    plt.ylim([0, 1.2 * np.max(rate_means)])
    plt.xlabel("Angle")

    ax.set_title("Mean excitatory firing rate for specific input angle\n",
                 va='bottom')
    plt.savefig(
        fig_folder + "firing_rate_with_angle{}.pdf".format(suffix_test),
        bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close(fig)
    # min mean max population responses in Hz
    fig, ax = plt.subplots(1, 1, figsize=(15, 8),
                           subplot_kw=dict(projection='polar'), dpi=600)

    # '#440357'  '#228b8d', '#b2dd2c'
    minimus = 0

    c = ax.fill(radians, rate_means, fill=False, edgecolor='#228b8d',
                lw=2, alpha=.8, label="Mean response")
    mins = [np.min(r) for r in all_rates]
    ax.fill(radians, mins, fill=False, edgecolor='#440357', lw=2,
            alpha=.8, label="Min response")
    maxs = [np.max(r) for r in all_rates]
    ax.fill(radians, maxs, fill=False, edgecolor='#b2dd2c', lw=2,
            alpha=1, label="Max response")
    maximus = np.max(maxs)
    ax.set_ylim([.8 * minimus, 1.1 * maximus])
    ax.set_xlabel("Random delays")
    plt.savefig(
        fig_folder + "rate_means_min_max_mean{}.pdf".format(suffix_test),
        bbox_inches='tight')
    plt.savefig(
        fig_folder + "rate_means_min_max_mean{}.svg".format(suffix_test),
        bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close(fig)

    # distributions of rates (excitatory and inhibitory)
    fig = plt.figure(figsize=(14, 8))
    n, bins, patches = plt.hist(instaneous_rates, bins=40, normed=True,
                                alpha=.75)
    if inh_instaneous_rates.size > 0:
        n, bins, patches = plt.hist(inh_instaneous_rates, bins=40, normed=True,
                                    alpha=.75)
    print("mean", np.mean(instaneous_rates))
    print("skew", stats.skew(n))
    print("kurtosis", stats.kurtosis(n))
    print("normality", stats.normaltest(n))
    plt.title("Distribution of rates")
    plt.xlabel("Instantaneous rate (Hz)")
    plt.savefig(fig_folder + "rate_distribution{}.pdf".format(suffix_test),
                bbox_inches='tight')
    plt.savefig(fig_folder + "rate_distribution{}.svg".format(suffix_test),
                bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close(fig)

    # distribution of delays present in the network after training
    fig = plt.figure(figsize=(14, 8))

    all_delays = np.concatenate((ff_last[:, -1], off_last[:, -1],
                                 noise_last[:, -1], lat_last[:, -1],
                                 inh_to_exh_last[:, -1]))
    y, binEdges = np.histogram(all_delays, bins=np.unique(all_delays).size)
    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    width = 1
    print("Delays:", np.unique(all_delays))
    plt.bar(np.unique(all_delays), y, width=width,
            color='#414C82', edgecolor='k')

    plt.xticks(np.unique(all_delays))
    plt.title("Distribution of delays")
    plt.xlabel("Delay (ms)")
    plt.savefig(fig_folder + "delay_distribution{}.pdf".format(suffix_test),
                bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close(fig)
    # evolution of rates during testing (mostly to check everything is fine
    # all of the time)
    fig = plt.figure(figsize=(14, 8))
    plt.plot(instaneous_rates, alpha=.75)
    if inh_instaneous_rates.size > 0:
        plt.plot(inh_instaneous_rates, alpha=.75)
    plt.title("Rates during testing")
    plt.xlabel("Testing iteration")
    plt.savefig(fig_folder + "rates_over_testing{}.pdf".format(suffix_test),
                bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close(fig)

    # KDE of firing rates
    fig = plt.figure(figsize=(7.5, 8), dpi=600)
    k = stats.kde.gaussian_kde(all_rates[0], 'silverman')
    # rnge= np.arange(0, maximus)
    rnge = np.linspace(0, maximus, 1000)
    plt.plot(rnge, k(rnge), label="$0^{\circ}$ ")
    k = stats.kde.gaussian_kde(all_rates[45 // 5], 'silverman')
    plt.plot(rnge, k(rnge), label="$45^{\circ}$ ")
    k = stats.kde.gaussian_kde(all_rates[90 // 5], 'silverman')
    plt.plot(rnge, k(rnge), label="$90^{\circ}$ ")
    k = stats.kde.gaussian_kde(all_rates[180 // 5], 'silverman')
    plt.plot(rnge, k(rnge), label="$180^{\circ}$ ")
    k = stats.kde.gaussian_kde(all_rates[270 // 5], 'silverman')
    plt.plot(rnge, k(rnge), label="$270^{\circ}$ ")
    k = stats.kde.gaussian_kde(all_rates[315 // 5], 'silverman')
    plt.plot(rnge, k(rnge), label="$315^{\circ}$ ")

    plt.legend(loc='best')
    plt.xlabel("Firing rate (Hz)")
    plt.ylabel("PDF")
    plt.savefig(
        fig_folder + "firing_rate_pdf_random{}.pdf".format(suffix_test),
        bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close(fig)

    # some computation for determining preferred direction and generating a
    # quiver plot
    all_average_responses_with_angle = np.empty((N_layer, angles.size, 2))
    for angle in angles:
        current_angle_responses = per_neuron_all_rates[angle // 5].reshape(
            N_layer, per_neuron_all_rates[angle // 5].shape[0] // N_layer)
        for i in range(N_layer):
            current_response = current_angle_responses[i, :]
            all_average_responses_with_angle[i, angle // 5, 0] = np.mean(
                current_response)
            all_average_responses_with_angle[i, angle // 5, 1] = stats.sem(
                current_response)
    max_average_responses_with_angle = np.empty((N_layer))
    sem_responses_with_angle = np.empty((N_layer))
    for i in range(N_layer):
        max_average_responses_with_angle[i] = np.argmax(
            all_average_responses_with_angle[i, :, 0]) * 5
        sem_responses_with_angle[i] = all_average_responses_with_angle[
            i, int(max_average_responses_with_angle[i] // 5), 1]
    # quiver plot
    fig, (ax) = plt.subplots(1, 1, figsize=(12, 10), dpi=600)
    i = ax.imshow(max_average_responses_with_angle.reshape(grid[0], grid[1]),
                  vmin=0, vmax=355, cmap=cyclic_viridis)

    dx = np.cos(max_average_responses_with_angle.reshape(grid[0], grid[1]))
    dy = np.sin(max_average_responses_with_angle.reshape(grid[0], grid[1]))
    plt.quiver(dx, dy, color='w',
               angles=max_average_responses_with_angle.reshape(grid[0],
                                                               grid[1]),
               pivot='mid')

    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", "5%", pad="3%")
    cbar = plt.colorbar(i, cax=cax)
    cbar.set_label("Preferred angle")
    plt.savefig(fig_folder + "per_angle_response{}.pdf".format(suffix_test),
                bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close(fig)

    # histogram with number of neurons preferring each direction
    fig = plt.figure(figsize=(16, 8))
    y, binEdges = np.histogram(max_average_responses_with_angle,
                               bins=angles.size)
    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    width = 5
    plt.bar(angles, y, width=width,  # yerr=sem_responses_with_angle,
            color='#414C82', edgecolor='k')
    plt.xlabel("Degree")
    plt.ylabel("# of sensitised neurons")
    plt.savefig(
        fig_folder + "no_sensitised_neurons_with_angle_hist{}.pdf".format(
            suffix_test))
    plt.close(fig)
    fig = plt.figure(figsize=(16, 8))
    y, binEdges = np.histogram(max_average_responses_with_angle,
                               bins=angles.size)
    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    width = 5
    plt.bar(angles - 180, np.roll(y, 180 // 5), width=width,
            color='#414C82', edgecolor='k')
    plt.xlabel("Degree")
    plt.ylabel("# of sensitised neurons")
    plt.savefig(
        fig_folder + "no_sensitised_neurons_with_angle_hist_centred{}.pdf".format(suffix_test))
    if show_plots:
        plt.show()
    plt.close(fig)
    fig = plt.figure(figsize=(10, 10), dpi=600)
    ax = plt.subplot(111, projection='polar')
    c = plt.fill(radians, y,
                 fill=False, lw=4, color='#414C82')
    plt.ylim([0, 1.2 * np.max(y)])
    plt.xlabel("Angle")

    ax.set_title("Number of sensitised neurons for specific angle\n",
                 va='bottom')
    plt.savefig(
        fig_folder + "number_of_sensitised_neurons_with_angle{}.pdf".format(
            suffix_test),
        bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close(fig)

    # selective neuron responses
    selective_ids = []
    for angle in angles:
        tmp_responses = per_neuron_all_rates[angle // 5].reshape(grid[0],
                                                                 grid[1],
                                                                 per_neuron_all_rates[
                                                                     angle // 5].shape[
                                                                     0] // N_layer)
        mean_responses = np.mean(tmp_responses, axis=2)
        max_neuron_position = \
            np.argwhere(mean_responses == np.max(mean_responses))[0]
        selective_ids.append(
            max_neuron_position[0] * grid[0] + max_neuron_position[1])
    selective_0s = selective_ids[0 // 5]
    selective_90s = selective_ids[90 // 5]
    selective_180s = selective_ids[180 // 5]
    selective_270s = selective_ids[270 // 5]
    the_4 = [selective_0s, selective_90s, selective_180s, selective_270s]

    super_selective_tuning_curves = np.empty((4, angles.size))

    for angle in angles:
        current_angle_responses = per_neuron_all_rates[angle // 5].reshape(
            N_layer, per_neuron_all_rates[angle // 5].shape[0] // N_layer)
        for i in range(4):
            current_response = current_angle_responses[the_4[i], :]
            super_selective_tuning_curves[i, angle // 5] = np.mean(
                current_response)
    # max responsive
    viridis_cmap = mlib.cm.get_cmap('viridis')
    fig, axes = plt.subplots(2, 2, figsize=(12, 12),
                             subplot_kw=dict(projection='polar'))

    minimus = np.min(super_selective_tuning_curves)
    maximus = np.max(super_selective_tuning_curves)
    no_files = super_selective_tuning_curves.shape[0]
    for index, ax in np.ndenumerate(axes):
        i = index[0] * 2 + index[1]
        ax.axvline((i * 90 / 180.) * np.pi, color="#bbbbbb", lw=4, zorder=1)
        c = ax.fill(radians, super_selective_tuning_curves[i, :],
                    c=viridis_cmap(float(i) / (no_files - 1)),
                    alpha=0.9, fill=False, lw=4, zorder=2)
        #     ax.set_xlabel("${}^\circ$".format(i*90))
        ax.set_ylim([0, 1.1 * maximus])

    # f.suptitle("Mean firing rate for specific input angle", va='bottom')
    plt.tight_layout(pad=10)
    plt.savefig(
        fig_folder + "individual_max_sensitive_tuning_curves{}.pdf".format(
            suffix_test),
        bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close(fig)

    # best searched response
    selective_ids = []
    for angle in angles:
        tmp_responses = per_neuron_all_rates[angle // 5].reshape(grid[0],
                                                                 grid[1],
                                                                 per_neuron_all_rates[
                                                                     angle // 5].shape[
                                                                     0] // N_layer)

        # Filtering mechanism
        # Modify the following line to change selection mechanism
        negative_reinforcement = 0

        # All angles in the opposite direction if projected in 1D
        #     lt_filter = angles[(90 + angle)%360<= angles]
        #     lt_gt_filter = lt_filter[(-90 + angle)%360<= lt_filter]
        #     undesirable_angles= lt_gt_filter // 5
        # Perpendicular angles + opposite
        undesirable_angles = [(180 + angle) % 360 // 5,
                              (90 + angle) % 360 // 5,
                              (-90 + angle) % 360 // 5]
        for undesirable_angle in undesirable_angles:
            tmp_opposite_responses = per_neuron_all_rates[
                undesirable_angle].reshape(grid[0], grid[1],
                                           per_neuron_all_rates[
                                               undesirable_angle].shape[
                                               0] // N_layer)
            negative_reinforcement += np.mean(tmp_opposite_responses, axis=2)
        #     negative_reinforcement = negative_reinforcement/len(undesirable_angles)
        # Application of the filter
        mean_responses = np.mean(tmp_responses,
                                 axis=2) - negative_reinforcement
        max_neuron_position = \
            np.argwhere(mean_responses == np.max(mean_responses))[0]
        selective_ids.append(
            max_neuron_position[0] * grid[0] + max_neuron_position[1])
    selective_0s = selective_ids[0 // 5]
    selective_90s = selective_ids[90 // 5]
    selective_180s = selective_ids[180 // 5]
    selective_270s = selective_ids[270 // 5]
    the_4 = [selective_0s, selective_90s, selective_180s, selective_270s]
    print("the 4", the_4)

    super_selective_tuning_curves = np.empty((4, angles.size))

    for angle in angles:
        current_angle_responses = per_neuron_all_rates[angle // 5].reshape(
            N_layer, per_neuron_all_rates[angle // 5].shape[0] // N_layer)
        for i in range(4):
            current_response = current_angle_responses[the_4[i], :]
            super_selective_tuning_curves[i, angle // 5] = np.mean(
                current_response)

    viridis_cmap = mlib.cm.get_cmap('viridis')
    fig, axes = plt.subplots(2, 2, figsize=(12, 12),
                             subplot_kw=dict(projection='polar'), dpi=600)

    minimus = np.min(super_selective_tuning_curves)
    maximus = np.max(super_selective_tuning_curves)
    no_files = super_selective_tuning_curves.shape[0]
    for index, ax in np.ndenumerate(axes):
        i = index[0] * 2 + index[1]
        ax.axvline((i * 90 / 180.) * np.pi, color="#aaaaaa", lw=4, zorder=1)
        c = ax.fill(radians, super_selective_tuning_curves[i, :],
                    c=viridis_cmap(float(i) / (no_files - 1)),
                    alpha=0.9, fill=False, lw=4, zorder=2)
        #     ax.set_xlabel("${}^\circ$".format(i*90))
        ax.set_ylim([0, 1.1 * maximus])

    # f.suptitle("Mean firing rate for specific input angle", va='bottom')
    plt.tight_layout(pad=10)
    plt.savefig(
        fig_folder + "individual_custom_filter_sensitive_tuning_curves{"
                     "}.pdf".format(
            suffix_test), bbox_inches='tight')
    plt.savefig(fig_folder + "individual_custom_filter_sensitive_tuning_curves{}.svg".format(suffix_test),
                bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close(fig)

    # connectivity plots and info

    number_of_afferents = get_number_of_afferents(N_layer, ff_num_network, lat_num_network)

    fig, (ax) = plt.subplots(1, 1, figsize=(12, 10))
    i = ax.imshow(number_of_afferents.reshape(grid[0], grid[1]), vmin=0)

    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", "5%", pad="3%")
    cbar = plt.colorbar(i, cax=cax)
    cbar.set_label("Number of afferents")
    plt.savefig(
        fig_folder + "synaptic_capacity_usage{}.pdf".format(suffix_test),
        bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close(fig)
    print("{:45}".format("Mean Synaptic capacity usage"), ":", np.mean(number_of_afferents))
    print("{:45}".format("STD Synaptic capacity usage"), ":", np.std(number_of_afferents))

    # weight histograms
    fig, axes = plt.subplots(1, 5, figsize=(15, 7), sharey=True)

    conns = (ff_last, off_last, noise_last, lat_last, inh_to_exh_last)
    conns_names = ["$ff_{on}$", "$ff_{off}$", "$ff_{noise}$", "$lat_{exc}$",
                   "$lat_{inh}$"]

    minimus = 0
    maximus = 1
    for index, ax in np.ndenumerate(axes):
        i = index[0]
        ax.hist(conns[i][:, 2] / g_max, bins=20, color='#414C82',
                edgecolor='k', normed=True)
        ax.set_title(conns_names[i])
        ax.set_xlim([minimus, maximus])
        ax.set_xticklabels(["0", "0.5", "1"])
    plt.tight_layout()
    plt.savefig(
        fig_folder + "weight_histograms{}.pdf".format(suffix_test),
        bbox_inches='tight')
    plt.savefig(
        fig_folder + "weight_histograms{}.svg".format(suffix_test),
        bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close(fig)

    pre_neurons = grid[0] * grid[1]
    percentage_conn = []
    all_conns = np.concatenate(conns)
    for index, ax in np.ndenumerate(conns):
        i = index[0]
        percentage_conn.append((conns[i].shape[0] * 100.) / pre_neurons ** 2)
        print("{:>10} has a total of ".format(conns_names[i]), conns[i].shape[
            0], " connections. This is ", percentage_conn[
                  i], "% of the possible connectivity (this number includes "
                      "multapses and autapses, if applicable")
    percentage_conn = np.asarray(percentage_conn)
    print("Total connectivity =", all_conns.shape[0], "or ", np.sum(
        percentage_conn), "% of the possible connectivity")
    # TODO return a list of neuron indices in descending order of
    # selectivity to the training angles and / or in 45 deg increments

    # TODO print stats along the way

    # Gari's plots here --------------------------------------------------------
    # Get covariances for input connectivity
    keys = ['ff_last', 'off_last', 'lat_last']
    dict_of_arrays = {
        'ff_last': ff_last,
        'off_last': off_last,
        'lat_last': lat_last
    }
    fig = plt.figure(figsize=(len(keys) * 10, 10))

    for nk, k in enumerate(keys):
        widths, heights, centres, angs = [], [], [], []
        # put appropriate neuron IDs here!
        for post_id in range(32 * 32):
            centre, shape, angle = get_variance_ellipse(
                conns4post(dict_of_arrays[k], post_id), 32, 32)
            widths.append(shape[0])
            heights.append(shape[1])
            centres.append(centre)
            angs.append(angle)

        horiz = [i for i in range(len(widths)) if widths[i] > heights[i]]
        vert = [i for i in range(len(widths)) if widths[i] < heights[i]]

        ax = plt.subplot(1, len(keys), nk + 1, aspect='equal')
        ax.set_title(k)
        for i in range(len(angs)):
            width = widths[i]
            height = heights[i]
            angle = angs[i]
            ell = Ellipse(xy=(0, 0), width=width, height=height, angle=angle,
                          facecolor='none', edgecolor='#414C82', alpha=0.2)
            ax.add_artist(ell)

        wavg = np.mean(widths)
        havg = np.mean(heights)
        aavg = np.mean(angs)
        # ell = Ellipse(xy=(0, 0), width=wavg, height=havg, angle=aavg,
        #               facecolor='none', edgecolor='#b2dd2c', linewidth=3.0)
        # ax.add_artist(ell)

        xlim = np.max(widths) / 2.0
        ylim = np.max(heights) / 2.0
        maxlim = max(xlim, ylim)
        ax.set_xlim(-maxlim, maxlim)
        ax.set_ylim(-maxlim, maxlim)

        ax.set_xlabel('Pixels')
        ax.set_ylabel('Pixels')

    plt.savefig(
        fig_folder + "covariance_ff_exc{}.pdf".format(suffix_test),
        bbox_inches='tight')
    plt.savefig(
        fig_folder + "covariance_ff_exc{}.svg".format(suffix_test),
        bbox_inches='tight')

    if show_plots:
        plt.show()
    plt.close(fig)

    # Direction Selectivty Index (DSI)
    # Setup some parameters
    angle_diff = 5
    sigma = 2.0
    kernel_width = 7

    in_angle = training_angles

    models = [conv_model(all_average_responses_with_angle[neuron_id, :, 0],
                         kernel_width, sigma) for neuron_id in range(N_layer)]
    max_ang = 360
    delta_ang = 50
    delta_resp = 5.
    min_diff = 10.
    min_dsi = 0.5
    min_osi = 0.5
    selective = []

    # Begin computation
    for nid in range(N_layer):
        curr_ang = i2a(np.argmax(models[nid]), angle_diff)
        opp_ang = get_opp_ang(curr_ang)
        #     opp_idx = a2i(opp_ang, angle_diff)
        opp_idx = get_local_max_idx(opp_ang, delta_ang, models[nid], angle_diff)
        opp_ang = i2a(opp_idx, angle_diff)

        ang_steps = np.arange(curr_ang - delta_ang, max_ang + delta_ang, delta_ang * 2)
        ang_steps = np.append(
            np.arange(max(0, curr_ang - 3 * delta_ang), 0, -delta_ang * 2)[::-1], ang_steps)
        # ang_steps[:] = np.clip(ang_steps, 0, max_ang)

        mean_resp = np.mean(models[nid])
        max_resp = np.max(models[nid])
        opp_resp = models[nid][opp_idx]
        min_resp = np.min(models[nid])
        dsi = get_wdsi(models[nid], curr_ang, angle_diff)
        osi = get_wosi(models[nid], curr_ang, angle_diff)
        other_max = has_other_max(curr_ang, delta_ang, max_resp,
                                  delta_resp, models[nid], angle_diff)
        if (max_resp - mean_resp) <= min_diff:
            continue

        if dsi <= min_dsi:
            continue

        if osi <= min_osi:
            continue

        if other_max:
            continue

        selective.append(nid)
    colors = cyclic_viridis(np.linspace(0, 1, len(angles)))
    polar = bool(1)
    min_zero = 30
    min_var = 30
    min_dsi = 0.5
    ang_delta = 1

    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(1, 1, 1, projection='polar')
    angs = np.deg2rad(angles)
    angs = np.append(angs, angs[0])

    sums = {ang: np.zeros_like(models[0]) for ang in in_angle}
    counts = {ang: 0.0 for ang in in_angle}
    # for i in horiz:
    # for i in range(N_layer):
    max_dsi = {ang: (-1, 0) for ang in in_angle}
    max_act = {ang: (-1, 0) for ang in in_angle}

    for i in selective:
        vals = models[i]
        #     vals = all_average_responses_with_angle[i, :, 0]

        idx = np.argmax(vals)
        ang = i2a(idx, angle_diff)
        dsi = get_wdsi(vals, ang, angle_diff)
        dsiv = np.max(dsi)
        maxv = vals[idx]
        rang = -1
        for aang in in_angle:
            if aang == 0 and \
                    (ang < ang_delta or ang > 360 - ang_delta):
                rang = aang
                break
            elif ang > (aang - ang_delta) and ang < (aang + ang_delta):
                rang = aang
                break

        if rang == -1:
            continue

        sums[rang] += vals
        counts[rang] += 1
        max_dsi[rang] = (dsiv, i) if max_dsi[rang][0] < dsiv else max_dsi[rang]
        max_act[rang] = (maxv, i) if max_act[rang][0] < maxv else max_act[rang]

        if polar:
            vals = np.append(vals, vals[0])

        plt.plot(angs, vals, alpha=0.1, color=colors[idx])

    for aang in in_angle:
        # ------------------------------------------
        idx = a2i(aang, angle_diff)
        vals = sums[aang] / counts[aang]
        if polar:
            vals = np.append(vals, vals[0])
            ang_val = np.deg2rad(aang)
        else:
            ang_val = aang

        plt.plot(angs, vals, label='mean act %d' % aang,
                 color=colors[idx])

        # ------------------------------------------
        iii = max_dsi[aang][1]
        vals = models[iii]
        if polar:
            vals = np.append(vals, vals[0])
            ang_val = np.deg2rad(aang)
        else:
            ang_val = aang

        plt.plot(angs, vals, label='max dsi %d' % aang,
                 color=colors[idx], linestyle=':')

        # ------------------------------------------
        iii = max_act[aang][1]
        vals = models[iii]
        if polar:
            vals = np.append(vals, vals[0])
            ang_val = np.deg2rad(aang)
        else:
            ang_val = aang

        plt.plot(angs, vals, label='max act %d' % aang,
                 color=colors[idx], linestyle='-.')

        plt.axvline(ang_val, color=colors[idx], linestyle='--')

    # plt.legend()
    plt.savefig(
        fig_folder + "dsi_responses{}.pdf".format(suffix_test),
        bbox_inches='tight')
    plt.savefig(
        fig_folder + "dsi_responses{}.svg".format(suffix_test),
        bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close(fig)

    # Back to analysis by PAB
    dsi_thresh = .5
    dsi_selective, dsi_not_selective = get_filtered_dsi_per_neuron(all_average_responses_with_angle, N_layer,
                                                                   dsi_thresh=dsi_thresh)
    dsi_selective = np.asarray(dsi_selective)
    dsi_not_selective = np.asarray(dsi_not_selective)
    if dsi_selective.size > 0 and dsi_not_selective.size > 0:
        all_dsi = np.concatenate((dsi_selective[:, -1], dsi_not_selective[:, -1]))
        print(suffix_test, "sees ", dsi_selective.shape[0], "selective neurons and", dsi_not_selective.shape[0],
              "ones that are not selective to any angle\n\n")
    elif dsi_selective.size == 0:
        all_dsi = dsi_not_selective[:, -1]
        print(suffix_test, "sees NO, zero, 0 selective neurons and", dsi_not_selective.shape[0],
              "ones that are not selective to any angle\n\n")
    else:
        all_dsi = dsi_selective[:, -1]
        print(suffix_test, "sees ", dsi_selective.shape[0],
              "selective neurons and none that are not selective to any angle. PERFECT!\n\n")

    # Histogram plot showing number of neurons at each level of DSI with the threshold displayed as axvline
    hist_weights = np.ones_like(all_dsi) / float(N_layer)
    fig = plt.figure(figsize=(16, 8))
    plt.hist(all_dsi, bins=np.linspace(0, 1, 21), color='#414C82',
             edgecolor='k', weights=hist_weights)
    plt.axvline(dsi_thresh, color='#b2dd2c', ls=":")
    plt.xticks(np.linspace(0, 1, 11))
    plt.xlabel("DSI")
    plt.ylabel("% of neurons")
    plt.savefig(
        fig_folder + "dsi_histogram{}.pdf".format(
            suffix_test))
    plt.savefig(
        fig_folder + "dsi_histogram{}.svg".format(
            suffix_test))
    if show_plots:
        plt.show()
    plt.close(fig)

    # plot a few of the best neurons based on DSI
    if dsi_selective.size > 0:
        unique_angles_of_interest = np.unique(dsi_selective[:, 1]).astype(int)  # unique_aoi s
        dsi_selective_firing_curves = np.empty((unique_angles_of_interest.size, angles.size))
        dsi_selective_neuron_ids = np.empty((unique_angles_of_interest.size)).astype(int)
        dsi_selective_dsi_values = np.empty((unique_angles_of_interest.size))

        for index, unique_aoi in np.ndenumerate(unique_angles_of_interest):
            i = int(index[0])
            current_angle_responses = get_per_angle_responses(per_neuron_all_rates, unique_aoi, N_layer)
            max_dsi_index = np.argmax(dsi_selective[dsi_selective[:, 1] == unique_aoi][:, -1])
            dsi_selective_neuron_ids[i] = int(dsi_selective[max_dsi_index, 0])
            dsi_selective_dsi_values[i] = dsi_selective[max_dsi_index, -1]
            dsi_selective_firing_curves[i, :] = get_omnidirectional_neural_response_for_neuron(
                dsi_selective_neuron_ids[i], per_neuron_all_rates, angles, N_layer)

        maximus = np.max(dsi_selective_firing_curves)
        no_files = dsi_selective_firing_curves.shape[0]
        size_scale = 8
        fig, axes = plt.subplots(1, unique_angles_of_interest.size,
                                 figsize=(unique_angles_of_interest.size * size_scale, 8),
                                 subplot_kw=dict(projection='polar'))

        viridis_cmap = mlib.cm.get_cmap('viridis')
        for curr_ax_id, curr_ax in np.ndenumerate(axes):
            i = int(curr_ax_id[0])
            curr_ax.axvline(np.deg2rad(unique_angles_of_interest[i]), color="#bbbbbb", lw=4, zorder=1)
            curr_ax.fill(radians, dsi_selective_firing_curves[i, :],
                         c=viridis_cmap(float(i) / (no_files - 1)),
                         alpha=0.9, fill=False, lw=4, zorder=2)
            curr_ax.set_xlabel("${:3}^\circ$ - DSI {:.2} - id {:4}".format(unique_angles_of_interest[i],
                                                                           dsi_selective_dsi_values[i],
                                                                           dsi_selective_neuron_ids[i]))
            curr_ax.set_ylim([0, 1.1 * maximus])

        plt.savefig(
            fig_folder + "dsi_individual_neurons{}.pdf".format(suffix_test),
            bbox_inches='tight')
        plt.savefig(
            fig_folder + "dsi_individual_neurons{}.svg".format(suffix_test),
            bbox_inches='tight')
        if show_plots:
            plt.show()
        plt.close(fig)

    # Entropy play
    entropy = compute_per_neuron_entropy(per_neuron_all_rates, angles, N_layer)
    max_entropy = (-np.log2(1. / angles.size))
    assert np.all(entropy <= max_entropy), entropy

    print("{:45}".format("Mean Entropy"), ":", np.mean(entropy))
    print("{:45}".format("Max possible Entropy"), ":", max_entropy)

    fig, (ax) = plt.subplots(1, 1, figsize=(10, 10), dpi=600)
    i = ax.imshow(entropy.reshape(grid[0], grid[1]), vmin=0, vmax=max_entropy)

    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", "5%", pad="3%")
    cbar = plt.colorbar(i, cax=cax)
    cbar.set_label("Entropy")

    plt.savefig(
        fig_folder + "per_neuron_entropy{}.pdf".format(suffix_test),
        bbox_inches='tight')
    plt.savefig(
        fig_folder + "per_neuron_entropy{}.svg".format(suffix_test),
        bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close(fig)

    entropy_selective_n_to_plot = 5
    entropy_selective_firing_curves = np.empty((entropy_selective_n_to_plot, angles.size))
    argsorted_entropy = np.argsort(entropy)
    entropy_selective_neuron_ids = argsorted_entropy[:entropy_selective_n_to_plot]

    for i in np.arange(entropy_selective_n_to_plot):
        entropy_selective_firing_curves[i, :] = get_omnidirectional_neural_response_for_neuron(
            entropy_selective_neuron_ids[i], per_neuron_all_rates, angles, N_layer)

    maximus = np.max(entropy_selective_firing_curves)
    no_files = entropy_selective_n_to_plot
    size_scale = 8
    available_dsi = np.ones(entropy_selective_n_to_plot) * np.nan
    # if dsi_selective_dsi_values.size > 0:
    #     available_dsi = dsi_selective_dsi_values[dsi_selective_neuron_ids[entropy_selective_neuron_ids]]
    fig, axes = plt.subplots(1, entropy_selective_n_to_plot,
                             figsize=(entropy_selective_n_to_plot * size_scale, 8),
                             subplot_kw=dict(projection='polar'))

    viridis_cmap = mlib.cm.get_cmap('viridis')
    for curr_ax_id, curr_ax in np.ndenumerate(axes):
        i = int(curr_ax_id[0])
        curr_ax.fill(radians, entropy_selective_firing_curves[i, :],
                     c=viridis_cmap(float(i) / (no_files - 1)),
                     alpha=0.9, fill=False, lw=4, zorder=2)
        curr_ax.set_xlabel("Entropy {:1.3f} - id {:4}".format(
            entropy[entropy_selective_neuron_ids[i]],
            # available_dsi[i],
            entropy_selective_neuron_ids[i]))
        curr_ax.set_ylim([0, 1.1 * maximus])

    plt.savefig(
        fig_folder + "entropy_individual_neurons{}.pdf".format(suffix_test),
        bbox_inches='tight')
    plt.savefig(
        fig_folder + "entropy_individual_neurons{}.svg".format(suffix_test),
        bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close(fig)

    # Entropy histogram
    normalised_entropy = entropy / max_entropy
    hist_weights = np.ones_like(normalised_entropy) / float(N_layer)

    fig = plt.figure(figsize=(16, 8))
    plt.hist(normalised_entropy, bins=np.linspace(0, 1, 21), color='#414C82',
             edgecolor='k', weights=hist_weights)
    plt.xticks(np.linspace(0, 1, 11))
    plt.xlabel("Normalised Entropy")
    plt.ylabel("% of neurons")
    plt.savefig(
        fig_folder + "entropy_histogram{}.pdf".format(
            suffix_test))
    plt.savefig(
        fig_folder + "entropy_histogram{}.svg".format(
            suffix_test))
    if show_plots:
        plt.show()
    plt.close(fig)

    return suffix_test, dsi_selective, dsi_not_selective


def comparison(archive_random, archive_constant, out_filename=None, extra_suffix=None,
               custom_labels=None, show_plots=False):
    # in the default use case, the 2 archives would relate to two networks
    # differing only in the way that structural plasticity chooses delays
    if ".npz" in archive_random:
        # random_delay_data = np.load(root_stats + archive_random)
        random_delay_data = np.load(os.path.join(root_stats, archive_random))
    else:
        # random_delay_data = np.load(root_stats + archive_random + ".npz")
        random_delay_data = np.load(os.path.join(root_stats, archive_random + ".npz"))

    if ".npz" in archive_random:
        # constant_delay_data = np.load(root_stats + archive_constant)
        constant_delay_data = np.load(os.path.join(root_stats, archive_constant))
    else:
        # constant_delay_data = np.load(root_stats + archive_constant + ".npz")
        constant_delay_data = np.load(os.path.join(root_stats, archive_constant + ".npz"))

    # LOAD DATA
    # RANDOM DELAYS
    random_rate_means = random_delay_data['rate_means']
    random_rate_stds = random_delay_data['rate_stds']
    random_rate_sem = random_delay_data['rate_sem']
    random_all_rates = random_delay_data['all_rates']
    radians = random_delay_data['radians']
    random_instaneous_rates = random_delay_data['instaneous_rates']
    angles = random_delay_data['angles']
    actual_angles = random_delay_data['actual_angles']
    random_target_neuron_mean_spike_rate = random_delay_data[
        'target_neuron_mean_spike_rate']

    # Connection information
    random_ff_connections = random_delay_data['ff_connections']
    random_lat_connections = random_delay_data['lat_connections']
    random_noise_connections = random_delay_data['noise_connections']
    random_ff_off_connections = random_delay_data['ff_off_connections']

    random_final_ff_conn_field = random_delay_data['final_ff_conn_field']
    random_final_ff_num_field = random_delay_data['final_ff_num_field']
    random_final_lat_conn_field = random_delay_data['final_lat_conn_field']
    random_final_lat_num_field = random_delay_data['final_lat_num_field']

    random_ff_last = random_delay_data['ff_last']
    random_off_last = random_delay_data['off_last']
    random_noise_last = random_delay_data['noise_last']
    random_lat_last = random_delay_data['lat_last']

    random_per_neuron_all_rates = random_delay_data['per_neuron_all_rates']

    # CONSTANT DELAYS
    constant_rate_means = constant_delay_data['rate_means']
    constant_rate_stds = constant_delay_data['rate_stds']
    constant_rate_sem = constant_delay_data['rate_sem']
    constant_all_rates = constant_delay_data['all_rates']
    constant_instaneous_rates = constant_delay_data['instaneous_rates']
    constant_target_neuron_mean_spike_rate = constant_delay_data[
        'target_neuron_mean_spike_rate']

    # Connection information
    constant_ff_connections = constant_delay_data['ff_connections']
    constant_lat_connections = constant_delay_data['lat_connections']
    constant_noise_connections = constant_delay_data['noise_connections']
    constant_ff_off_connections = constant_delay_data['ff_off_connections']

    constant_final_ff_conn_field = constant_delay_data['final_ff_conn_field']
    constant_final_ff_num_field = constant_delay_data['final_ff_num_field']
    constant_final_lat_conn_field = constant_delay_data['final_lat_conn_field']
    constant_final_lat_num_field = constant_delay_data['final_lat_num_field']

    constant_ff_last = constant_delay_data['ff_last']
    constant_off_last = constant_delay_data['off_last']
    constant_noise_last = constant_delay_data['noise_last']
    constant_lat_last = constant_delay_data['lat_last']
    constant_per_neuron_all_rates = constant_delay_data['per_neuron_all_rates']

    sim_params = np.array(random_delay_data['training_sim_params']).ravel()[0]

    grid = sim_params['grid']
    N_layer = grid[0] * grid[1]
    n = grid[0]
    g_max = sim_params['g_max']

    viridis_cmap = mlib.cm.get_cmap('viridis')
    # TODO Begin asserts

    # generate suffix
    suffix_test = generate_suffix(sim_params['training_angles'])
    if extra_suffix:
        suffix_test += "_" + extra_suffix

    # Mean firing rate comparison
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 8),
                                  subplot_kw=dict(projection='polar'), dpi=800)

    maximus = np.max((random_rate_means, constant_rate_means))
    # minimus = np.min((random_rate_means, constant_rate_means))
    minimus = 0

    c = ax.scatter(radians, random_rate_means, c=random_rate_stds, s=100)
    c.set_alpha(0.8)
    ax.set_ylim([minimus, 1.1 * maximus])
    # plt.ylabel("Hz")
    # ax2 = plt.subplot(222, projection='polar')
    c2 = ax2.scatter(radians, constant_rate_means, c=constant_rate_stds, s=100)
    c2.set_alpha(0.8)

    # ax.set_xlabel("Angle")
    # ax2.set_xlabel("Angle")
    if custom_labels:
        ax.set_xlabel(custom_labels[0])
        ax2.set_xlabel(custom_labels[1])
    else:
        ax.set_xlabel("Random delays")
        ax2.set_xlabel("Constant delays")
    ax2.set_ylim([minimus, 1.1 * maximus])

    fig.suptitle("Mean firing rate for specific input angle", va='bottom')
    plt.tight_layout(pad=10)
    plt.savefig(
        fig_folder + "comparison_firing_rate_with_angle{}.pdf".format(
            suffix_test),
        bbox_inches='tight')
    plt.savefig(
        fig_folder + "comparison_firing_rate_with_angle{}.svg".format(
            suffix_test),
        bbox_inches='tight')

    if show_plots:
        plt.show()
    plt.close(fig)

    # Min Max Mean comparison
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 8),
                                  subplot_kw=dict(projection='polar'), dpi=800)

    maximus = np.max((random_rate_means, constant_rate_means))
    minimus = 0

    c = ax.fill(radians, random_rate_means, fill=False, edgecolor='#228b8d',
                lw=2, alpha=.8, label="Mean response")
    mins = [np.min(r) for r in random_all_rates]
    ax.fill(radians, mins, fill=False, edgecolor='#440357', lw=2, alpha=.8,
            label="Min response")
    maxs = [np.max(r) for r in random_all_rates]
    ax.fill(radians, maxs, fill=False, edgecolor='#b2dd2c', lw=2, alpha=1,
            label="Max response")
    maximus = np.max(maxs)
    rand_max = np.max(maximus)
    c2 = ax2.fill(radians, constant_rate_means, fill=False,
                  edgecolor='#228b8d', lw=2, alpha=.8, label="Mean response")
    mins = [np.min(r) for r in constant_all_rates]
    ax2.fill(radians, mins, fill=False, edgecolor='#440357', lw=2, alpha=.8,
             label="Min response")
    maxs = [np.max(r) for r in constant_all_rates]
    ax2.fill(radians, maxs, fill=False, edgecolor='#b2dd2c', lw=2, alpha=1,
             label="Max response")
    const_max = np.max(maxs)
    rand_max, const_max, np.max([rand_max, const_max])
    ax.set_ylim([minimus, 1.1 * np.max([rand_max, const_max])])
    ax2.set_ylim([minimus, 1.1 * np.max([rand_max, const_max])])

    if custom_labels:
        ax.set_xlabel(custom_labels[0])
        ax2.set_xlabel(custom_labels[1])
    else:
        ax.set_xlabel("Random delays")
        ax2.set_xlabel("Constant delays")

    # f.suptitle("Mean firing rate for specific input angle", va='bottom')
    # plt.tight_layout(pad=10)
    plt.savefig(
        fig_folder + "comparison_firing_rate_with_angle_min_max_mean{}.pdf".format(
            suffix_test), bbox_inches='tight')
    plt.savefig(
        fig_folder + "comparison_firing_rate_with_angle_min_max_mean{"
                     "}.svg".format(
            suffix_test), bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close(fig)
    # Histogram comparison

    fig = plt.figure(figsize=(15, 8), dpi=800)
    y, binEdges = np.histogram(angles, bins=angles.size)
    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    width = 5

    plt.bar(angles - 180, np.roll(random_rate_means, 180 // 5), width=width,
            yerr=np.roll(random_rate_sem, 180 // 5),
            color='#b2dd2c', edgecolor='k')
    plt.bar(angles - 180, np.roll(constant_rate_means, 180 // 5), width=width,
            yerr=np.roll(constant_rate_sem, 180 // 5),
            color='#414C82', edgecolor='k')
    plt.xlabel("Degree")
    plt.ylabel("Hz")
    plt.xlim([-177.5, 177.5])
    plt.xticks(np.concatenate((np.arange(-180, 180, 45), [175])))
    plt.savefig(
        fig_folder + "comparison_firing_rate_with_angle_hist_centred{" \
                     "}.pdf".format(
            suffix_test))
    plt.savefig(
        fig_folder + "comparison_firing_rate_with_angle_hist_centred{" \
                     "}.svg".format(
            suffix_test))
    if show_plots:
        plt.show()
    plt.close(fig)

    # Compute paired independent t-test
    ttests = []

    for i in range(angles.size):
        ttests.append(
            stats.ttest_ind(random_all_rates[i], constant_all_rates[i],
                            equal_var=False))

    # T-test
    fig = plt.figure(figsize=(7.5, 8), dpi=600)
    ax = plt.subplot(111, projection='polar')
    c = plt.fill(radians, [t[0] for t in ttests], fill=False,
                 edgecolor='#aaaaaa', lw=2, alpha=.8, zorder=5)
    plt.ylim([1.1 * np.min([t[0] for t in ttests]),
              .9 * np.max([t[0] for t in ttests])])

    minimus_t = np.min([t[0] for t in ttests])
    if minimus_t < 0:
        scaling_min_t = 1.1
    else:
        scaling_min_t = .9
    plt.yticks(np.linspace(scaling_min_t * minimus_t,
                           1.1 * np.max([t[0] for t in ttests]), 5, dtype=int))

    tstat = np.asarray([t[0] for t in ttests])
    pstat = np.asarray([t[1] for t in ttests])

    below_threshold = pstat <= 0.01
    above_threshold = pstat > 0.01
    plt.scatter(radians[above_threshold], tstat[above_threshold],
                c='orangered', zorder=6)
    plt.xlabel("Independent t-test")
    plt.savefig(
        fig_folder + "ttest_with_angle{}.pdf".format(suffix_test),
        bbox_inches='tight')
    plt.savefig(
        fig_folder + "ttest_with_angle{}.svg".format(suffix_test),
        bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close(fig)

    # First fname kde
    fig = plt.figure(figsize=(7.5, 8), dpi=800)
    maxs = [np.max(r) for r in random_all_rates]
    maximus = np.max(maxs)
    rnge = np.linspace(0, maximus, 1000)

    k = stats.kde.gaussian_kde(random_all_rates[0], 'silverman')
    plt.plot(rnge, k(rnge), label="$0^{\circ}$ ")
    k = stats.kde.gaussian_kde(random_all_rates[45 // 5], 'silverman')
    plt.plot(rnge, k(rnge), label="$45^{\circ}$ ")
    k = stats.kde.gaussian_kde(random_all_rates[90 // 5], 'silverman')
    plt.plot(rnge, k(rnge), label="$90^{\circ}$ ")
    k = stats.kde.gaussian_kde(random_all_rates[180 // 5], 'silverman')
    plt.plot(rnge, k(rnge), label="$180^{\circ}$ ")
    k = stats.kde.gaussian_kde(random_all_rates[270 // 5], 'silverman')
    plt.plot(rnge, k(rnge), label="$270^{\circ}$ ")
    k = stats.kde.gaussian_kde(random_all_rates[315 // 5], 'silverman')
    plt.plot(rnge, k(rnge), label="$315^{\circ}$ ")

    plt.legend(loc='best')
    plt.xlabel("Firing rate (Hz)")
    plt.ylabel("PDF")
    plt.savefig(
        fig_folder + "firing_rate_pdf_random{}.pdf".format(suffix_test),
        bbox_inches='tight')
    plt.savefig(
        fig_folder + "firing_rate_pdf_random{}.svg".format(suffix_test),
        bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close(fig)

    # Fname2 KDE
    fig = plt.figure(figsize=(7.5, 8), dpi=800)
    k = stats.kde.gaussian_kde(constant_all_rates[0], 'silverman')
    maxs = [np.max(r) for r in constant_all_rates]
    maximus = np.max(maxs)
    rnge = np.linspace(0, maximus, 1000)
    plt.plot(rnge, k(rnge), label="$0^{\circ}$ ")
    k = stats.kde.gaussian_kde(constant_all_rates[45 // 5], 'silverman')
    plt.plot(rnge, k(rnge), label="$45^{\circ}$ ")
    k = stats.kde.gaussian_kde(constant_all_rates[90 // 5], 'silverman')
    plt.plot(rnge, k(rnge), label="$90^{\circ}$ ")
    k = stats.kde.gaussian_kde(constant_all_rates[180 // 5], 'silverman')
    plt.plot(rnge, k(rnge), label="$180^{\circ}$ ")
    k = stats.kde.gaussian_kde(constant_all_rates[270 // 5], 'silverman')
    plt.plot(rnge, k(rnge), label="$270^{\circ}$ ")
    k = stats.kde.gaussian_kde(constant_all_rates[315 // 5], 'silverman')
    plt.plot(rnge, k(rnge), label="$315^{\circ}$ ")

    # plt.legend(loc='best')
    plt.xlabel("Firing rate (Hz)")
    plt.ylabel("PDF")
    plt.savefig(
        fig_folder + "firing_rate_pdf_constant{}.pdf".format(suffix_test),
        bbox_inches='tight')
    plt.savefig(
        fig_folder + "firing_rate_pdf_constant{}.svg".format(suffix_test),
        bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close(fig)

    # Max preference per neuron comparison
    random_all_average_responses_with_angle = np.empty(
        (N_layer, angles.size, 2))
    for angle in angles:
        current_angle_responses = random_per_neuron_all_rates[
            angle // 5].reshape(N_layer,
                                random_per_neuron_all_rates[angle // 5].shape[
                                    0] // N_layer)
        for i in range(N_layer):
            current_response = current_angle_responses[i, :]
            random_all_average_responses_with_angle[
                i, angle // 5, 0] = np.mean(current_response)
            random_all_average_responses_with_angle[
                i, angle // 5, 1] = stats.sem(current_response)
    random_max_average_responses_with_angle = np.empty((N_layer))
    sem_responses_with_angle = np.empty((N_layer))
    for i in range(N_layer):
        random_max_average_responses_with_angle[i] = np.argmax(
            random_all_average_responses_with_angle[i, :, 0]) * 5
        sem_responses_with_angle[i] = random_all_average_responses_with_angle[
            i, int(random_max_average_responses_with_angle[i] // 5), 1]

    constant_all_average_responses_with_angle = np.empty(
        (N_layer, angles.size, 2))
    for angle in angles:
        current_angle_responses = constant_per_neuron_all_rates[
            angle // 5].reshape(N_layer,
                                constant_per_neuron_all_rates[angle //
                                                              5].shape[
                                    0] // N_layer)
        for i in range(N_layer):
            current_response = current_angle_responses[i, :]
            constant_all_average_responses_with_angle[
                i, angle // 5, 0] = np.mean(current_response)
            constant_all_average_responses_with_angle[
                i, angle // 5, 1] = stats.sem(current_response)
    constant_max_average_responses_with_angle = np.empty((N_layer))
    sem_responses_with_angle = np.empty((N_layer))
    for i in range(N_layer):
        constant_max_average_responses_with_angle[i] = np.argmax(
            constant_all_average_responses_with_angle[i, :, 0]) * 5
        sem_responses_with_angle[i] = \
            constant_all_average_responses_with_angle[
                i, int(constant_max_average_responses_with_angle[i] // 5), 1]

    fig = plt.figure(figsize=(15, 8), dpi=800)
    img_grid = ImageGrid(fig, 111,
                         nrows_ncols=(1, 2),
                         axes_pad=0.15,
                         share_all=True,
                         cbar_location="right",
                         cbar_mode="single",
                         cbar_size="7%",
                         cbar_pad=0.15,
                         )
    imgs = [random_max_average_responses_with_angle.reshape(grid[0], grid[1]),
            constant_max_average_responses_with_angle.reshape(grid[0],
                                                              grid[1])]

    dxs = [np.cos(
        random_max_average_responses_with_angle.reshape(grid[0], grid[1])),
        np.cos(constant_max_average_responses_with_angle.reshape(grid[0],
                                                                 grid[1]))]

    dys = [np.sin(
        random_max_average_responses_with_angle.reshape(grid[0], grid[1])),
        np.sin(constant_max_average_responses_with_angle.reshape(grid[0],
                                                                 grid[1]))]
    # Add data to image grid

    index = 0
    for ax in img_grid:
        im = ax.imshow(imgs[index], vmin=0, vmax=355, cmap=cyclic_viridis)
        ax.quiver(dxs[index], dys[index], color='w', angles=imgs[index],
                  pivot='mid')
        index += 1

    # Colorbar
    ax.cax.colorbar(im)
    ax.cax.toggle_label(True)

    img_grid[0].set_xlabel("Neuron ID")
    img_grid[0].set_ylabel("Neuron ID")
    img_grid[1].set_xlabel("Neuron ID")

    plt.savefig(
        fig_folder + "comparison_per_angle_response{}.pdf".format(suffix_test),
        bbox_inches='tight', dpi=800)
    plt.savefig(
        fig_folder + "comparison_per_angle_response{}.svg".format(suffix_test),
        bbox_inches='tight', dpi=800)
    if show_plots:
        plt.show()
    plt.close(fig)

    rand_y, binEdges = np.histogram(
        random_max_average_responses_with_angle, bins=angles.size)
    const_y, binEdges = np.histogram(
        constant_max_average_responses_with_angle, bins=angles.size)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 8),
                                  subplot_kw=dict(projection='polar'), dpi=800)

    maximus = np.max((rand_y, const_y))
    # minimus = np.min((random_rate_means, constant_rate_means))
    minimus = 0

    c = ax.fill(radians, rand_y, alpha=0.7, fill=False, lw=4)

    ax.set_ylim([minimus, 1.1 * maximus])
    c2 = ax2.fill(radians, const_y, alpha=0.7, fill=False, lw=4)

    if custom_labels:
        ax.set_xlabel(custom_labels[0])
        ax2.set_xlabel(custom_labels[1])
    else:
        ax.set_xlabel("Random delays")
        ax2.set_xlabel("Constant delays")
    ax2.set_ylim([minimus, 1.1 * maximus])

    # f.suptitle("Mean firing rate for specific input angle", va='bottom')
    plt.tight_layout(pad=10)
    plt.savefig(
        fig_folder + "comparison_number_of_sensitised_neurons_with_angle{}.pdf".format(suffix_test),
        bbox_inches='tight')
    plt.savefig(
        fig_folder + "comparison_number_of_sensitised_neurons_with_angle{}.svg".format(suffix_test),
        bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close(fig)

    # TODO dsi comparison
    random_all_average_responses_with_angle, _, _ = compute_all_average_responses_with_angle(
        random_per_neuron_all_rates, angles, N_layer)
    random_dsi_selective, random_dsi_not_selective = get_filtered_dsi_per_neuron(
        random_all_average_responses_with_angle, N_layer)
    random_dsi_selective = np.asarray(random_dsi_selective)
    random_dsi_not_selective = np.asarray(random_dsi_not_selective)

    constant_all_average_responses_with_angle, _, _ = compute_all_average_responses_with_angle(
        constant_per_neuron_all_rates, angles, N_layer)
    constant_dsi_selective, constant_dsi_not_selective = get_filtered_dsi_per_neuron(
        constant_all_average_responses_with_angle, N_layer)
    constant_dsi_selective = np.asarray(constant_dsi_selective)
    constant_dsi_not_selective = np.asarray(constant_dsi_not_selective)

    random_max_dsi = np.empty((N_layer))
    random_dsi_pref_angle = np.ones((N_layer)) * np.nan
    constant_max_dsi = np.empty((N_layer))
    constant_dsi_pref_angle = np.ones((N_layer)) * np.nan
    for nid in range(N_layer):
        temp_dsi = 0
        if random_dsi_selective.size > 0 and nid in random_dsi_selective[:, 0]:
            temp_dsi = random_dsi_selective[random_dsi_selective[:, 0] == nid].ravel()[-1]
            random_dsi_pref_angle[nid] = random_dsi_selective[random_dsi_selective[:, 0] == nid].ravel()[1]
        elif random_dsi_not_selective.size > 0 and nid in random_dsi_not_selective[:, 0]:
            temp_dsi = random_dsi_not_selective[random_dsi_not_selective[:, 0] == nid].ravel()[-1]
        random_max_dsi[nid] = temp_dsi

        temp_dsi = 0
        if constant_dsi_selective.size > 0 and nid in constant_dsi_selective[:, 0]:
            temp_dsi = constant_dsi_selective[constant_dsi_selective[:, 0] == nid].ravel()[-1]
            constant_dsi_pref_angle[nid] = constant_dsi_selective[constant_dsi_selective[:, 0] == nid].ravel()[1]
        elif constant_dsi_not_selective.size > 0 and nid in constant_dsi_not_selective[:, 0]:
            temp_dsi = constant_dsi_not_selective[constant_dsi_not_selective[:, 0] == nid].ravel()[-1]
        constant_max_dsi[nid] = temp_dsi

    fig = plt.figure(figsize=(15, 8), dpi=800)
    img_grid = ImageGrid(fig, 111,
                         nrows_ncols=(1, 2),
                         axes_pad=0.15,
                         share_all=True,
                         cbar_location="right",
                         cbar_mode="single",
                         cbar_size="7%",
                         cbar_pad=0.15,
                         )

    imgs = [random_max_dsi.reshape(grid[0], grid[1]),
            constant_max_dsi.reshape(grid[0], grid[1])]

    dxs = [np.cos(random_dsi_pref_angle.reshape(grid[0], grid[1])),
           np.cos(constant_dsi_pref_angle.reshape(grid[0], grid[1]))]

    dys = [np.sin(
        random_dsi_pref_angle.reshape(grid[0], grid[1])),
        np.sin(constant_dsi_pref_angle.reshape(grid[0], grid[1]))]
    # Add data to image grid

    index = 0
    for ax in img_grid:
        im = ax.imshow(imgs[index], vmin=0, vmax=1)
        ax.quiver(dxs[index], dys[index], color='w', angles=imgs[index],
                  pivot='mid')
        index += 1

    # Colorbar
    ax.cax.colorbar(im)
    ax.cax.toggle_label(True)
    # ax.cax.set_label("DSI")

    img_grid[0].set_xlabel("Neuron ID")
    img_grid[0].set_ylabel("Neuron ID")
    img_grid[1].set_xlabel("Neuron ID")

    plt.savefig(
        fig_folder + "comparison_per_angle_dsi_response{}.pdf".format(suffix_test), dpi=800,
        bbox_inches='tight')  # pad_inches=1)
    plt.savefig(
        fig_folder + "comparison_per_angle_dsi_response{}.svg".format(suffix_test), dpi=800,
        bbox_inches='tight')  # , pad_inches=1)
    if show_plots:
        plt.show()
    plt.close(fig)

    random_entropy = compute_per_neuron_entropy(random_per_neuron_all_rates, angles, N_layer)
    constant_entropy = compute_per_neuron_entropy(constant_per_neuron_all_rates, angles, N_layer)
    max_entropy = get_max_entropy(angles)
    assert np.all(random_entropy <= max_entropy), random_entropy
    assert np.all(constant_entropy <= max_entropy), constant_entropy

    print("{:45}".format("Mean Random Entropy"), ":", np.mean(random_entropy))
    print("{:45}".format("Mean Constant Entropy"), ":", np.mean(constant_entropy))
    print("{:45}".format("Max possible Entropy"), ":", max_entropy)

    fig = plt.figure(figsize=(15, 8), dpi=800)
    img_grid = ImageGrid(fig, 111,
                         nrows_ncols=(1, 2),
                         axes_pad=0.15,
                         share_all=True,
                         cbar_location="right",
                         cbar_mode="single",
                         cbar_size="7%",
                         cbar_pad=0.15,
                         )
    imgs = [random_entropy.reshape(grid[0], grid[1]),
            constant_entropy.reshape(grid[0], grid[1])]
    # Add data to image grid

    index = 0
    for ax in img_grid:
        im = ax.imshow(imgs[index], vmin=0, vmax=max_entropy, cmap=viridis_cmap)
        index += 1

    # Colorbar
    ax.cax.colorbar(im)
    ax.cax.toggle_label(True)

    img_grid[0].set_xlabel("Neuron ID")
    img_grid[0].set_ylabel("Neuron ID")
    img_grid[1].set_xlabel("Neuron ID")

    plt.savefig(
        fig_folder + "comparison_per_neuron_entropy{}.pdf".format(suffix_test),
        bbox_inches='tight')
    plt.savefig(
        fig_folder + "comparison_per_neuron_entropy{}.svg".format(suffix_test),
        bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close(fig)


def evolution(filenames, times, suffix, path=None, show_plots=False):
    # https://gist.github.com/MatthewJA/5a0a6d75748bf5cb5962cb9d5572a6ce
    viridis_cmap = mlib.cm.get_cmap('viridis')
    no_files = len(filenames)
    assert len(filenames) == len(times)
    all_rate_means = []
    all_rate_sems = []
    all_radians = []
    all_dsis = []
    angles = []
    N_layer = None
    s_max = None
    radians = None
    all_entropies = []
    times = np.asarray(times)
    times_in_minutes = times / (60 * bunits.second)
    times_in_minutes = times_in_minutes.astype(dtype=int)
    num_afferents = []

    print("{:45}".format("Experiment evolution. # of snapshots"), ":", len(filenames))
    print("{:45}".format("Results appended the following suffix"), ":", suffix)

    if not path:
        path = root_syn

    for fn in filenames:
        # cached_data = np.load(root_stats + fn + ".npz")
        cached_data = np.load(os.path.join(path, fn + ".npz"))
        testing_data = np.load(
            os.path.join(root_syn, "spiking_moving_bar_input", "spiking_moving_bar_motif_bank_simtime_1200s.npz"))
        sim_params = np.array(cached_data['testing_sim_params']).ravel()[0]
        grid = sim_params['grid']
        s_max = sim_params['s_max']
        N_layer = grid[0] * grid[1]
        rate_means = cached_data['rate_means']
        rate_stds = cached_data['rate_stds']
        rate_sem = cached_data['rate_sem']
        all_rates = cached_data['all_rates']
        radians = cached_data['radians']
        angles = cached_data['angles']
        per_neuron_all_rates = cached_data['per_neuron_all_rates']
        ff_num_network = cached_data['ff_num_network']
        lat_num_network = cached_data['lat_num_network']

        all_rate_means.append(rate_means)
        all_rate_sems.append(rate_sem)
        all_radians = radians
        num_afferents.append(get_number_of_afferents(N_layer, ff_num_network, lat_num_network))

        if "dsi_selective" in cached_data.files and "dsi_not_selective" in cached_data.files:
            dsi_selective = cached_data['dsi_selective']
            dsi_not_selective = cached_data['dsi_not_selective']
        else:
            dsi_selective, dsi_not_selective = backward_compatibility_get_dsi(per_neuron_all_rates, angles, N_layer)
        concatenated_dsis = get_concatenated_dsis(dsi_selective, dsi_not_selective)
        all_dsis.append(concatenated_dsis)

        assert concatenated_dsis.size == N_layer
        # Compute entropy
        all_entropies.append(compute_per_neuron_entropy(per_neuron_all_rates, angles, N_layer))
        # Close files
        cached_data.close()
        testing_data.close()

    all_dsis = np.asarray(all_dsis)
    all_entropies = np.asarray(all_entropies)
    num_afferents = np.asarray(num_afferents)

    # stlye the median of boxplots
    medianprops = dict(color='#414C82', linewidth=1.5)

    fig = plt.figure(figsize=(10, 10), dpi=800)
    ax = plt.subplot(111, projection='polar')

    for i in range(no_files):
        plt.fill(radians, all_rate_means[i],
                 c=viridis_cmap(float(i) / (no_files - 1)),
                 label="{} minutes".format(times_in_minutes[i]),
                 alpha=0.7, fill=False, lw=4)

    art = []
    plt.ylim([0, 1.1 * np.max(all_rate_means)])
    lgd = plt.legend(bbox_to_anchor=(1.1, .7), loc=2, borderaxespad=0.)
    art.append(lgd)
    plt.savefig(
        fig_folder + "firing_rate_evolution_{}.pdf".format(suffix),
        dpi=800,
        additional_artists=art,
        bbox_inches="tight")
    plt.savefig(
        fig_folder + "firing_rate_evolution_{}.svg".format(suffix),
        dpi=800,
        additional_artists=art,
        bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)

    # plot evolution of mean DSI when increasing training
    fig = plt.figure(figsize=(16, 8), dpi=600)

    plt.axhline(.5, color='#b2dd2c', ls=":")
    bp = plt.boxplot(all_dsis.T, notch=True, medianprops=medianprops)  # , patch_artist=True)
    # plt.setp(bp['medians'], color='#414C82')
    # plt.setp(bp['boxes'], alpha=0)

    plt.xticks(np.arange(times_in_minutes.shape[0]) + 1, times_in_minutes)
    plt.xlabel("Time (minutes)")
    plt.ylabel("DSI")
    plt.ylim([-.05, 1.05])
    plt.grid(True, which='major', axis='y')
    plt.savefig(fig_folder + "dsi_evolution_boxplot_{}.pdf".format(suffix))
    plt.savefig(fig_folder + "dsi_evolution_boxplot_{}.svg".format(suffix))
    if show_plots:
        plt.show()
    plt.close(fig)

    max_entropy = get_max_entropy(angles)
    # plot evolution of entropy when increasing training
    fig = plt.figure(figsize=(16, 8), dpi=600)

    bp = plt.boxplot(all_entropies.T, notch=True, medianprops=medianprops)  # , patch_artist=True)
    # plt.setp(bp['medians'], color='#414C82')
    # plt.setp(bp['boxes'], alpha=0)

    plt.xticks(np.arange(times_in_minutes.shape[0]) + 1, times_in_minutes)
    plt.xlabel("Time (minutes)")
    plt.ylabel("Entropy")
    plt.ylim([-.05, max_entropy + 0.05])
    plt.grid(True, which='major', axis='y')
    plt.savefig(fig_folder + "entropy_evolution_boxplot_{}.pdf".format(suffix))
    plt.savefig(fig_folder + "entropy_evolution_boxplot_{}.svg".format(suffix))
    if show_plots:
        plt.show()
    plt.close(fig)

    # synaptic capacity evolution
    fig = plt.figure(figsize=(16, 8), dpi=600)

    plt.axhline(s_max, color='#b2dd2c', ls=":")
    bp = plt.boxplot(num_afferents.T, notch=True, medianprops=medianprops)

    plt.xticks(np.arange(times_in_minutes.shape[0]) + 1, times_in_minutes)
    plt.xlabel("Time (minutes)")
    plt.ylabel("Mean synaptic capacity usage")
    plt.grid(True, which='major', axis='y')
    plt.savefig(fig_folder + "number_of_afferents_evolution_boxplot_{}.pdf".format(suffix))
    plt.savefig(fig_folder + "number_of_afferents_evolution_boxplot_{}.svg".format(suffix))
    if show_plots:
        plt.show()
    plt.close(fig)



def batch_analyser(batch_data_file, batch_info_file, extra_suffix=None, show_plots=False):
    # Read the archives
    # batch_data = np.load(root_stats + batch_data_file + ".npz")
    # batch_info = np.load(root_stats + batch_info_file + ".npz")

    batch_data = np.load(os.path.join(root_stats, batch_data_file + ".npz"))
    batch_info = np.load(os.path.join(root_stats, batch_info_file + ".npz"))
    # Read files from archives
    batch_parameters = batch_info['parameters_of_interest'].ravel()[0]
    result_keys = batch_data['files']
    batch_argparser_data = batch_data['params']
    N_layer = batch_argparser_data[0][0]['argparser']['n'] ** 2
    angles = batch_data[result_keys[0]].ravel()[0]['angles']
    radians = batch_data[result_keys[0]].ravel()[0]['radians']
    files_to_ignore = []
    for k in batch_data.files:
        if k not in result_keys:
            files_to_ignore.append(k)
    file_shape = []
    value_list = []
    suffix_test = "_batch"
    for poi in batch_parameters.keys():
        file_shape.append(batch_parameters[poi].size)
        # set up parameters to generate all combinations of these
        value_list.append(batch_parameters[poi])
        suffix_test += "_" + poi
    if extra_suffix:
        suffix_test += extra_suffix
    value_list = np.asarray(value_list)
    file_matrix = np.empty(file_shape, dtype="S200")
    # for index, value in np.ndenumerate(file_matrix):
    #     print(index, value)
    for row in batch_argparser_data:
        argparser_info = row[0]['argparser']
        file_info = row[1]
        position_to_fill = []
        keys = batch_parameters.keys()
        for index, poi in np.ndenumerate(keys):
            i, = np.where(np.isclose(value_list[index[0]], argparser_info[poi]))
            position_to_fill.append(i)
        file_matrix[position_to_fill] = file_info

    # Print some information
    print("{:45}".format("Batch Data Archive"), ":", batch_data_file)
    print("{:45}".format("Batch Info Archive"), ":", batch_info_file)
    print("{:45}".format("Batch Data meta information in"), ":", files_to_ignore)
    print("{:45}".format("recording_archive_name in batch_data.files"), ":", "recording_archive_name" in
          batch_data.files)
    print("{:45}".format("params in batch_data.files"), ":", "params" in batch_data.files)
    print("{:45}".format("files in batch_data.files"), ":", "files" in batch_data.files)
    print("{:45}".format("Batch completed in"), ":", batch_info['total_time'])
    print("{:45}".format("Batch focused on the following params"), ":", batch_parameters.keys())
    print("{:45}".format("Shape of result matrices"), ":", file_shape)
    print("{:45}".format("Suffix for generated figures"), ":", suffix_test)
    # print("<File matrix>", "-" * 50)
    # print(file_matrix)
    # print("</File matrix>", "-" * 50)

    dsi_comparison = np.ones(file_shape) * np.nan

    exp_shape_layer = copy.deepcopy(file_shape)
    exp_shape_layer.append(N_layer)

    all_dsis = np.ones(exp_shape_layer) * np.nan
    all_exc_entropies = np.ones(exp_shape_layer) * np.nan
    all_inh_entropies = np.ones(exp_shape_layer) * np.nan
    mean_exc_entropy = np.empty(dsi_comparison.shape)
    mean_inh_entropy = np.empty(dsi_comparison.shape)

    exp_shape_angle = copy.deepcopy(file_shape)
    exp_shape_angle.append(angles.size)
    all_mean_rates = np.ones(exp_shape_angle) * np.nan
    for file_index, file_key in np.ndenumerate(file_matrix):
        if file_key == '' or ".npz" not in file_key or file_key not in batch_data.files:
            continue  # I don't particularly want this, but otherwise I indent everything too much
        current_results = batch_data[file_key].ravel()[0]
        dsi_selective = current_results['dsi_selective']
        dsi_not_selective = current_results['dsi_not_selective']
        rate_means = current_results['rate_means']
        dsi_selective = np.asarray(dsi_selective)
        dsi_not_selective = np.asarray(dsi_not_selective)
        all_dsi = get_concatenated_dsis(dsi_selective, dsi_not_selective)
        average_dsi = np.mean(all_dsi)
        dsi_comparison[file_index] = average_dsi
        all_dsis[file_index] = all_dsi
        all_mean_rates[file_index] = rate_means
        all_exc_entropies[file_index] = current_results['exc_entropy']
        all_inh_entropies[file_index] = current_results['inh_entropy']
        mean_exc_entropy[file_index] = np.mean(all_exc_entropies[file_index])
        mean_inh_entropy[file_index] = np.mean(all_inh_entropies[file_index])

    # exc_entropy_description = stats.describe(all_exc_entropies.reshape(file_matrix.size, N_layer), axis=1)
    # dsi_description = stats.describe(all_dsis.reshape(file_matrix.size, N_layer), axis=1)
    # print("{:45}".format("Describe exh entropy"), ":", exc_entropy_description)
    # print("{:45}".format("Describe DSI"), ":", dsi_description)

    # covariance matrix between exc and inh Entropy
    # TODO I need to check how normal the input distributions are
    # TODO switch to scipy pearsonr (need to manually to an all to all comparison)
    exc_inh_entropy_covariance = np.corrcoef(all_exc_entropies.reshape(file_matrix.size, N_layer),
                                             all_inh_entropies.reshape(file_matrix.size, N_layer))

    print("{:45}".format("Mean exh inh covariance coeff"), ":", np.mean(exc_inh_entropy_covariance))

    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 8), dpi=800)
    i = ax1.matshow(exc_inh_entropy_covariance, vmin=-1, vmax=1)
    ax1.grid(visible=False)
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", "5%", pad="3%")
    cbar = plt.colorbar(i, cax=cax)
    cbar.set_label("Covariance")
    plt.savefig(
        fig_folder + "exc_inh_entropy_covariance{}.pdf".format(
            suffix_test))
    plt.savefig(
        fig_folder + "exc_inh_entropy_covariance{}.svg".format(
            suffix_test))
    if show_plots:
        plt.show()
    plt.close(fig)

    # covariance matrix between DSI and Entropy
    # TODO switch to scipy pearsonr (need to manually to an all to all comparison)
    dsi_entropy_covariance = np.corrcoef(all_exc_entropies.reshape(file_matrix.size, N_layer),
                                         all_dsis.reshape(file_matrix.size, N_layer))
    # dsi_entropy_covariance = np.empty(file_shape)
    # dsi_entropy_pearson_p = np.empty(file_shape)
    # for index, _ in np.ndenumerate(dsi_entropy_covariance):
    #     # for each run in the sensitivity analysis compute a covariance and an associated p-value
    #     dsi_entropy_covariance[index], dsi_entropy_pearson_p[index] = stats.pearsonr(
    #         all_exc_entropies[index],
    #         all_dsis[index])
    print("{:45}".format("Mean entropy dsi covariance coeff"), ":", np.mean(dsi_entropy_covariance))
    # create a significance mask where insignificant results get multiplied by nan
    # 2 sigma?
    # p_threshold = 0.01
    # dsi_entropy_significance_mask = np.ones(file_shape) * np.nan
    # dsi_entropy_significance_mask[np.where(dsi_entropy_pearson_p < p_threshold)] = 1

    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 8), dpi=800)
    i = ax1.matshow(dsi_entropy_covariance, vmin=-1, vmax=1)
    ax1.grid(visible=False)
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", "5%", pad="3%")
    cbar = plt.colorbar(i, cax=cax)
    cbar.set_label("Covariance")
    plt.savefig(
        fig_folder + "dsi_entropy_covariance{}.pdf".format(
            suffix_test))
    plt.savefig(
        fig_folder + "dsi_entropy_covariance{}.svg".format(
            suffix_test))
    if show_plots:
        plt.show()
    plt.close(fig)

    # Mean DSI comparison
    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 8), dpi=800)
    i = ax1.matshow(dsi_comparison)
    ax1.set_ylabel(batch_parameters.keys()[0])
    ax1.set_yticks(np.arange(value_list[0].size))
    ax1.set_yticklabels(value_list[0])
    ax1.set_xlabel(batch_parameters.keys()[1])
    ax1.set_xticks(np.arange(value_list[1].size))
    ax1.set_xticklabels(value_list[1], rotation='vertical')
    ax1.grid(visible=False)
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", "5%", pad="3%")
    cbar = plt.colorbar(i, cax=cax)
    cbar.set_label("DSI")
    plt.savefig(
        fig_folder + "dsi_comparison{}.pdf".format(
            suffix_test))
    plt.savefig(
        fig_folder + "dsi_comparison{}.svg".format(
            suffix_test))
    if show_plots:
        plt.show()
    plt.close(fig)

    # Plot DSI histograms
    dsi_thresh = 0.5
    size_scale = 8
    fig, axes = plt.subplots(value_list[0].size, value_list[1].size, figsize=(value_list[0].size * size_scale,
                                                                              value_list[1].size * size_scale))

    for y_axis in np.arange(all_dsis.shape[0]):
        for x_axis in np.arange(all_dsis.shape[0]):
            # for all_dsi_index, curent_all_dsi in np.ndenumerate(all_dsis):
            curent_all_dsi = all_dsis[y_axis, x_axis]
            hist_weights = np.ones_like(curent_all_dsi) / float(N_layer)
            curr_ax = axes[y_axis, x_axis]
            curr_ax.hist(curent_all_dsi, bins=np.linspace(0, 1, 21), color='#414C82',
                         edgecolor='k', weights=hist_weights)
            curr_ax.axvline(dsi_thresh, color='#b2dd2c', ls=":")
            curr_ax.set_xticks(np.linspace(0, 1, 11))
    # plt.xlabel("DSI")
    # plt.ylabel("% of neurons")
    plt.savefig(
        fig_folder + "dsi_histograms_comparison{}.pdf".format(
            suffix_test))
    plt.savefig(
        fig_folder + "dsi_histograms_comparison{}.svg".format(
            suffix_test))
    if show_plots:
        plt.show()
    plt.close(fig)

    # Plot Entropy histograms
    size_scale = 8
    fig, axes = plt.subplots(value_list[0].size, value_list[1].size, figsize=(value_list[0].size * size_scale,
                                                                              value_list[1].size * size_scale))

    for y_axis in np.arange(all_exc_entropies.shape[0]):
        for x_axis in np.arange(all_exc_entropies.shape[0]):
            curent_all_entropy = all_exc_entropies[y_axis, x_axis]
            max_entropy = get_max_entropy(angles)
            normalised_entropy = curent_all_entropy / max_entropy
            hist_weights = np.ones_like(normalised_entropy) / float(N_layer)
            curr_ax = axes[y_axis, x_axis]
            curr_ax.hist(normalised_entropy, bins=np.linspace(0, 1, 21), color='#414C82',
                         edgecolor='k', weights=hist_weights)
            curr_ax.set_xticks(np.linspace(0, 1, 11))
    plt.savefig(
        fig_folder + "entropy_histograms_comparison{}.pdf".format(
            suffix_test))
    plt.savefig(
        fig_folder + "entropy_histograms_comparison{}.svg".format(
            suffix_test))
    if show_plots:
        plt.show()
    plt.close(fig)

    # Plot firing rate profiles
    fig, axes = plt.subplots(value_list[0].size, value_list[1].size,
                             figsize=(value_list[0].size * size_scale, value_list[1].size * size_scale),
                             subplot_kw=dict(projection='polar'))

    for y_axis in np.arange(all_mean_rates.shape[0]):
        for x_axis in np.arange(all_mean_rates.shape[0]):
            curent_mean_rate = all_mean_rates[y_axis, x_axis]
            curr_ax = axes[y_axis, x_axis]
            curr_ax.fill(radians, curent_mean_rate, fill=False, edgecolor='#228b8d',
                         lw=2, alpha=.8, label="Mean response")
    plt.savefig(
        fig_folder + "rate_means_comparison{}.pdf".format(suffix_test),
        bbox_inches='tight')
    plt.savefig(
        fig_folder + "rate_means_comparison{}.svg".format(suffix_test),
        bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close(fig)

    batch_data.close()
    batch_info.close()

    # Mean Entropy comparison
    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 8), dpi=800)
    i = ax1.matshow(mean_exc_entropy)
    ax1.set_ylabel(batch_parameters.keys()[0])
    ax1.set_yticks(np.arange(value_list[0].size))
    ax1.set_yticklabels(value_list[0])
    ax1.set_xlabel(batch_parameters.keys()[1])
    ax1.set_xticks(np.arange(value_list[1].size))
    ax1.set_xticklabels(value_list[1], rotation='vertical')
    ax1.grid(visible=False)
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", "5%", pad="3%")
    cbar = plt.colorbar(i, cax=cax)
    cbar.set_label("Entropy")
    plt.savefig(
        fig_folder + "entropy_comparison{}.pdf".format(
            suffix_test))
    plt.savefig(
        fig_folder + "entropy_comparison{}.svg".format(
            suffix_test))
    if show_plots:
        plt.show()
    plt.close(fig)

    # inhibitory

    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 8), dpi=800)
    i = ax1.matshow(mean_inh_entropy)
    ax1.set_ylabel(batch_parameters.keys()[0])
    ax1.set_yticks(np.arange(value_list[0].size))
    ax1.set_yticklabels(value_list[0])
    ax1.set_xlabel(batch_parameters.keys()[1])
    ax1.set_xticks(np.arange(value_list[1].size))
    ax1.set_xticklabels(value_list[1], rotation='vertical')
    ax1.grid(visible=False)
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", "5%", pad="3%")
    cbar = plt.colorbar(i, cax=cax)
    cbar.set_label("Entropy")
    plt.savefig(
        fig_folder + "inh_entropy_comparison{}.pdf".format(
            suffix_test))
    plt.savefig(
        fig_folder + "inh_entropy_comparison{}.svg".format(
            suffix_test))
    if show_plots:
        plt.show()
    plt.close(fig)


def sigma_and_ad_analyser(archive, out_filename=None, extra_suffix=None, show_plots=False):
    # This is useless and probably broken for s_max > 32
    import warnings
    warnings.warn("These results are probably completely wrong for values of s_max > 32! Don't trust them!")

    if "npz" in str(archive):
        data = np.load(archive)
    else:
        data = np.load(archive + ".npz")
    simdata = np.array(data['sim_params']).ravel()[0]

    grid = simdata['grid']
    N_layer = grid[0] * grid[1]
    n = int(np.sqrt(N_layer))
    g_max = simdata['g_max']
    s_max = simdata['s_max']
    sigma_form_forward = simdata['sigma_form_forward']
    sigma_form_lateral = simdata['sigma_form_lateral']
    p_form_lateral = simdata['p_form_lateral']
    p_form_forward = simdata['p_form_forward']
    p_elim_dep = simdata['p_elim_dep']
    p_elim_pot = simdata['p_elim_pot']
    f_rew = simdata['f_rew']

    # Connection information
    ff_last = data['ff_last']
    off_last = data['off_last']
    noise_last = data['noise_last']
    lat_last = data['lat_last']
    inh_to_exh_last = data['inh_to_exh_last']

    conns = (ff_last, off_last, noise_last, lat_last, inh_to_exh_last)

    ff_conns = np.concatenate((ff_last, off_last, noise_last))
    lat_conns = np.concatenate((lat_last, inh_to_exh_last))

    last_conn, last_weight = correct_smax_list_to_post_pre(ff_conns, lat_conns, s_max, N_layer)

    # b
    final_fan_in = fan_in(last_conn, last_weight, 'conn', 'ff')
    fin_mean_projection, fin_means_and_std_devs, fin_means_for_plot, \
    fin_mean_centred_projection = centre_weights(
        final_fan_in, s_max)
    fin_mean_std_conn = np.mean(fin_means_and_std_devs[:, 5])
    fin_mean_AD_conn = np.mean(fin_means_and_std_devs[:, 4])
    fin_stds_conn = fin_means_and_std_devs[:, 5]
    fin_AD_conn = fin_means_and_std_devs[:, 4]

    fin_conn_ff_odc = odc(final_fan_in)

    # c

    init_ff_connections = []
    ff_s = np.zeros(N_layer, dtype=np.uint)
    lat_s = np.zeros(N_layer, dtype=np.uint)

    # populate ff_s and lat_s
    # ff_last, lat_last
    for post_id in range(N_layer):
        ff_s[post_id] = ff_last[ff_last[:, 1] == post_id].shape[0]
        lat_s[post_id] = lat_last[lat_last[:, 1] == post_id].shape[0]

    existing_pre_ff = []
    generated_ff_conn = []
    generated_lat_conn = []

    generate_equivalent_connectivity(
        ff_s, generated_ff_conn,
        sigma_form_forward, p_form_forward,
        "\nGenerating initial feedforward connectivity...",
        N_layer=N_layer, n=n, g_max=g_max)

    generate_equivalent_connectivity(
        lat_s, generated_lat_conn,
        sigma_form_lateral, p_form_lateral,
        "\nGenerating initial lateral connectivity...",
        N_layer=N_layer, n=n, g_max=g_max)

    gen_init_conn, gen_init_weight = \
        list_to_post_pre(np.asarray(generated_ff_conn),
                         np.asarray(generated_lat_conn), s_max,
                         N_layer)

    gen_fan_in = fan_in(gen_init_conn, gen_init_weight, 'conn', 'ff')

    fin_mean_projection_shuf, fin_means_and_std_devs_shuf, \
    fin_means_for_plot_shuf, fin_mean_centred_projection_shuf = \
        centre_weights(gen_fan_in, s_max)

    fin_mean_std_conn_shuf = np.mean(fin_means_and_std_devs_shuf[:, 5])
    fin_mean_AD_conn_shuf = np.mean(fin_means_and_std_devs_shuf[:, 4])
    fin_stds_conn_shuf = fin_means_and_std_devs_shuf[:, 5]
    fin_AD_conn_shuf = fin_means_and_std_devs_shuf[:, 4]

    wsr_sigma_fin_conn_fin_conn_shuffle = stats.wilcoxon(
        fin_stds_conn.ravel(), fin_stds_conn_shuf.ravel())
    wsr_AD_fin_conn_fin_conn_shuffle = stats.wilcoxon(
        fin_AD_conn.ravel(),
        fin_AD_conn_shuf.ravel())
    # d
    final_fan_in_weight = fan_in(last_conn, last_weight, 'weight',
                                 'ff')
    # final_fan_in_weight = conn_matrix_to_fan_in(ff_last, mode='weight')
    fin_mean_projection_weight, fin_means_and_std_devs_weight, fin_means_for_plot_weight, fin_mean_centred_projection_weight = centre_weights(
        final_fan_in_weight, s_max)
    fin_mean_std_weight = np.mean(fin_means_and_std_devs_weight[:, 5])
    fin_mean_AD_weight = np.mean(fin_means_and_std_devs_weight[:, 4])
    fin_stds_weight = fin_means_and_std_devs_weight[:, 5]
    fin_AD_weight = fin_means_and_std_devs_weight[:, 4]

    fin_weight_ff_odc = odc(final_fan_in_weight)

    # e
    weight_copy = weight_shuffle(last_conn, last_weight, 'ff')
    shuf_weights = fan_in(last_conn, weight_copy, 'weight', 'ff')

    fin_mean_projection_weight_shuf, fin_means_and_std_devs_weight_shuf, \
    fin_means_for_plot_weight_shuf, fin_mean_centred_projection_weight_shuf = centre_weights(
        shuf_weights, s_max)
    fin_mean_std_weight_shuf = np.mean(
        fin_means_and_std_devs_weight_shuf[:, 5])
    fin_mean_AD_weight_shuf = np.mean(
        fin_means_and_std_devs_weight_shuf[:, 4])
    fin_stds_weight_shuf = fin_means_and_std_devs_weight_shuf[:, 5]
    fin_AD_weight_shuf = fin_means_and_std_devs_weight_shuf[:, 4]

    wsr_sigma_fin_weight_fin_weight_shuffle = stats.wilcoxon(
        fin_stds_weight.ravel(), fin_stds_weight_shuf.ravel())
    wsr_AD_fin_weight_fin_weight_shuffle = stats.wilcoxon(
        fin_AD_weight.ravel(), fin_AD_weight_shuf.ravel())
    print("\n\n\n")
    print("%-60s" % "Mean sigma aff fin conn shuffle", fin_mean_std_conn_shuf)
    print("%-60s" % "Mean sigma aff fin conn", fin_mean_std_conn)
    print("%-60s" % "p(WSR sigma aff fin conn vs sigma aff fin conn shuffle)",
          wsr_sigma_fin_conn_fin_conn_shuffle.pvalue)
    print("%-60s" % "Mean sigma aff fin weight shuffle", fin_mean_std_weight_shuf)
    print("%-60s" % "Mean sigma aff fin weight", fin_mean_std_weight)
    print("%-60s" % "p(WSR sigma aff fin weight vs sigma aff fin weight shuffle)",
          wsr_sigma_fin_weight_fin_weight_shuffle.pvalue)
    print("%-60s" % "Mean AD fin conn shuffle", fin_mean_AD_conn_shuf)
    print("%-60s" % "Mean AD fin conn", fin_mean_AD_conn)
    print("%-60s" % "p(WSR AD fin conn vs AD fin conn shuffle)", wsr_AD_fin_conn_fin_conn_shuffle.pvalue)
    print("%-60s" % "Mean AD fin weight shuffle", fin_mean_AD_weight_shuf)
    print("%-60s" % "Mean AD fin weight", fin_mean_AD_weight)
    print("%-60s" % "p(WSR AD fin weight vs AD fin weight shuffle)", wsr_AD_fin_weight_fin_weight_shuffle.pvalue)


def package_neo_block(spikes, label, N_layer, simtime, archive_name):
    # Could also use NeoIO and save this stuff to a file so I don't have to re-assemble the Block
    # over and over again
    block = neo.Block()

    # build segment for the current data to be gathered in
    segment = neo.Segment(
        name="segment0",
        description="manufactured segment",
        rec_datetime=datetime.now())

    for neuron_id in range(N_layer):
        spiketrain = neo.SpikeTrain(
            times=spikes[spikes[:, 0] == neuron_id][:, 1],
            t_start=0 * ms,
            t_stop=simtime,
            units=ms,
            sampling_rate=1,
            source_population=label)
        # get times per atom
        segment.spiketrains.append(spiketrain)

    block.segments.append(segment)
    block.name = label  # population.label
    block.description = archive_name  # self._population.describe() -- all info inside archive
    block.rec_datetime = block.segments[0].rec_datetime
    # block.annotate(**self._metadata())

    return block


def elephant_analysis(archive, extra_suffix=None, show_plots=False, time_to_waste=False):
    # Pass in a testing file name. This file needs to have spikes in the raw format
    # data = np.load(root_syn + archive + ".npz")
    data = np.load(root_syn + archive + ".npz")
    sim_params = np.array(data['sim_params']).ravel()[0]
    simtime = int(data['simtime']) * ms
    exc_spikes = data['post_spikes']
    inh_spikes = data['inh_post_spikes']
    grid = sim_params['grid']
    N_layer = grid[0] * grid[1]
    n = grid[0]
    g_max = sim_params['g_max']
    s_max = sim_params['s_max']
    training_angles = sim_params['training_angles']
    suffix_test = generate_suffix(training_angles)
    data.close()
    if extra_suffix:
        suffix_test += "_" + extra_suffix
    print("{:45}".format("Beginning Elephant analysis"))
    print("{:45}".format("Suffix for generated figures"), ":", suffix_test)
    print("{:45}".format("Simtime"), ":", simtime)

    print("{:45}".format("Assembling neo blocks..."))
    start_time = datetime.now()
    exc_block = package_neo_block(exc_spikes, "Excitatory population", N_layer, simtime, archive)
    inh_block = package_neo_block(inh_spikes, "Inhibitory population", N_layer, simtime, archive)
    end_time = datetime.now()
    total_time = end_time - start_time
    print("{:45}".format("Neo blocks assembled. Process took {}".format(total_time)))
    gathered_results = {}

    exc_segment = exc_block.segments[0]
    inh_segment = inh_block.segments[0]

    exc_spike_trains = exc_segment.spiketrains
    inh_spike_trains = inh_segment.spiketrains

    gathered_results['exc_block'] = exc_block
    gathered_results['inh_block'] = inh_block

    # Begin the analysis
    # analysis using https://elephant.readthedocs.io/en/latest/reference/statistics.html

    # The following is EXTREMELY EXPENSIVE! DON'T UNCOMMENT
    # exc_ifrs = np.empty((N_layer, int(simtime/ms)))
    # inh_ifrs = np.empty((N_layer, int(simtime/ms)))
    # print("{:45}".format("Computing instantaneous firing rates... (ETA ~1 hour)"))
    # start_time = datetime.now()
    # exc_ifrs = statistics.instantaneous_rate(exc_segment.spiketrains[:], sampling_period=200 * ms)
    # inh_ifrs = statistics.instantaneous_rate(inh_segment.spiketrains[:], sampling_period=200 * ms)
    # end_time = datetime.now()
    # total_time = end_time - start_time
    # print("{:45}".format("Computing IFRs only took {}".format(total_time)))
    #
    #
    # fig = plt.figure(figsize=(10, 7))
    # plt.plot(exc_ifrs/Hz, color='C0', alpha=0.8)
    # plt.plot(inh_ifrs/Hz, color='C1', alpha=0.8)
    #
    # plt.xlabel("Time (ms)")
    # plt.ylabel("IFR (Hz)")
    # plt.savefig(
    #     fig_folder + "ifr_elephant{}.pdf".format(
    #         suffix_test),
    #     bbox_inches='tight')
    # plt.savefig(
    #     fig_folder + "ifr_elephant{}.svg".format(
    #         suffix_test),
    #     bbox_inches='tight')
    # if show_plots:
    #     plt.show()
    # plt.close(fig)

    # analysis using SPADE https://elephant.readthedocs.io/en/latest/reference/spade.html

    if not time_to_waste:
        print("{:45}".format("Seems we have no time to waste. Exiting this test ..."))
        return gathered_results

    # analysis using CAD https://elephant.readthedocs.io/en/latest/reference/cell_assembly_detection.html
    binsize = 20 * ms
    print("{:45}".format("Binning excitatory spikes ..."))
    start_time = datetime.now()
    binned_spikes = conversion.BinnedSpikeTrain(exc_spike_trains, binsize=binsize, t_start=0 * ms, t_stop=simtime / 10)
    end_time = datetime.now()
    total_time = end_time - start_time
    print("{:45}".format("Excitatory spikes binned. Process took {}".format(total_time)))
    print("{:45}".format("Running cell assembly detection ..."))
    start_time = datetime.now()
    patterns = cad.cell_assembly_detection(binned_spikes, maxlag=10)[0]
    end_time = datetime.now()
    total_time = end_time - start_time
    print("{:45}".format("Completed cell assembly detection. Process took {}".format(total_time)))
    fig = plt.figure(figsize=(15, 8), dpi=800)
    for neu in patterns['neurons']:
        if neu == 0:
            plt.plot(patterns['times'] * binsize, [neu] * len(patterns['times']), 'ro', label='pattern')
        else:
            plt.plot(patterns['times'] * binsize, [neu] * len(patterns['times']), 'ro')
        # Raster plot of the data
    for st_idx, st in enumerate(exc_spike_trains):
        if st_idx == 0:
            plt.plot(st.rescale(ms), [st_idx] * len(st), 'k.', label='spikes')
        else:
            plt.plot(st.rescale(ms), [st_idx] * len(st), 'k.')
    plt.ylim([-1, len(exc_spike_trains)])
    plt.xlabel('time (ms)')
    plt.ylabel('neurons ids')
    plt.legend()
    plt.savefig(
        fig_folder + "cad{}.pdf".format(
            suffix_test), bbox_inches='tight')
    plt.savefig(
        fig_folder + "cad{}.svg".format(
            suffix_test), bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close(fig)

    # analysis using spike_train_dissimilarity
    # https://elephant.readthedocs.io/en/latest/reference/spike_train_dissimilarity.html
    # TODO Modify the following to see if closer neurons have similar activations
    # also, incorporate knowledge about individual neuron preferences to see if they have similar activity and
    # if they are different from other angles

    numbers = np.random.choice(np.arange(N_layer), 100, replace=False)
    numbers = np.sort(numbers)
    print("{:45}".format("Computing van Rossum spike train dissimilarity between some EXC and INH neurons ("
                         "independent)"))
    print("{:45}".format("Neuron ids selected"), ":", numbers)

    list_of_spiketrains = []
    for no in numbers:
        list_of_spiketrains.append(exc_spike_trains[no])
    van_rossum_distance_exc = spike_train_dissimilarity.van_rossum_dist(
        list_of_spiketrains)

    list_of_spiketrains = []
    for no in numbers:
        list_of_spiketrains.append(inh_spike_trains[no])
    van_rossum_distance_inh = spike_train_dissimilarity.van_rossum_dist(
        list_of_spiketrains)

    maximus = np.max([np.max(van_rossum_distance_exc.ravel()), np.max(van_rossum_distance_inh.ravel())])

    fig = plt.figure(figsize=(15, 8), dpi=800)
    img_grid = ImageGrid(fig, 111,
                         nrows_ncols=(1, 2),
                         axes_pad=0.15,
                         share_all=True,
                         cbar_location="right",
                         cbar_mode="single",
                         cbar_size="7%",
                         cbar_pad=0.15,
                         )
    imgs = [van_rossum_distance_exc, van_rossum_distance_inh]

    # Add data to image grid

    index = 0
    for ax in img_grid:
        im = ax.matshow(imgs[index], vmin=0, vmax=maximus)
        index += 1

    # Colorbar
    ax.cax.colorbar(im)
    ax.cax.toggle_label(True)
    # ax.cax.set_label("van Rossum distance")
    plt.savefig(
        fig_folder + "van_rossum_distance{}.pdf".format(
            suffix_test), bbox_inches='tight')
    plt.savefig(
        fig_folder + "van_rossum_distance{}.svg".format(
            suffix_test), bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close(fig)

    numbers = np.random.choice(np.arange(N_layer), 10, replace=False)
    numbers = np.sort(numbers)
    print("{:45}".format("[EXPENSIVE] Computing Victor-Purpura spike train dissimilarity between some EXC and INH "
                         "neurons ("
                         "independent)"))
    print("{:45}".format("Neuron ids selected"), ":", numbers)
    list_of_spiketrains = []
    for no in numbers:
        list_of_spiketrains.append(exc_spike_trains[no])
    vp_distance_exc = spike_train_dissimilarity.victor_purpura_dist(
        list_of_spiketrains)

    list_of_spiketrains = []
    for no in numbers:
        list_of_spiketrains.append(inh_spike_trains[no])
    vp_distance_inh = spike_train_dissimilarity.victor_purpura_dist(
        list_of_spiketrains)

    maximus = np.max([np.max(vp_distance_exc.ravel()), np.max(vp_distance_inh.ravel())])

    fig = plt.figure(figsize=(15, 8), dpi=800)
    img_grid = ImageGrid(fig, 111,
                         nrows_ncols=(1, 2),
                         axes_pad=0.15,
                         share_all=True,
                         cbar_location="right",
                         cbar_mode="single",
                         cbar_size="7%",
                         cbar_pad=0.15,
                         )
    imgs = [vp_distance_exc, vp_distance_inh]

    # Add data to image grid

    index = 0
    for ax in img_grid:
        im = ax.matshow(imgs[index], vmin=0, vmax=maximus)
        index += 1

    # Colorbar
    ax.cax.colorbar(im)
    ax.cax.toggle_label(True)
    # ax.cax.set_label("van Rossum distance")
    plt.savefig(
        fig_folder + "vp_distance{}.pdf".format(
            suffix_test), bbox_inches='tight')
    plt.savefig(
        fig_folder + "vp_distance{}.svg".format(
            suffix_test), bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close(fig)
    gathered_results['van_rossum_distance_exc'] = van_rossum_distance_exc
    gathered_results['van_rossum_distance_inh'] = van_rossum_distance_inh
    gathered_results['vp_distance_exc'] = vp_distance_exc
    gathered_results['vp_distance_inh'] = vp_distance_inh
    return gathered_results


def comparative_elephant_analysis(archive1, archive2, extra_suffix=None, show_plots=False):
    # Pass in a testing file name. This file needs to have spikes in the raw format
    # data1 = np.load(root_syn + archive1 + ".npz")
    # data2 = np.load(root_syn + archive2 + ".npz")
    data1 = np.load(os.path.join(root_syn, archive1 + ".npz"))
    data2 = np.load(os.path.join(root_syn, archive2 + ".npz"))

    # Load data1 stuff
    sim_params1 = np.array(data1['sim_params']).ravel()[0]
    simtime1 = int(data1['simtime']) * ms
    exc_spikes1 = data1['post_spikes']
    inh_spikes1 = data1['inh_post_spikes']
    grid1 = sim_params1['grid']
    N_layer1 = grid1[0] * grid1[1]
    training_angles1 = sim_params1['training_angles']

    # Load data2 stuff
    sim_params2 = np.array(data2['sim_params']).ravel()[0]
    simtime2 = int(data2['simtime']) * ms
    exc_spikes2 = data2['post_spikes']
    inh_spikes2 = data2['inh_post_spikes']
    grid2 = sim_params2['grid']
    N_layer2 = grid2[0] * grid2[1]
    training_angles2 = sim_params2['training_angles']

    # Assertions that need to be true
    assert N_layer1 == N_layer2
    assert np.all(grid1 == grid2)

    # Continue
    suffix_test = generate_suffix(training_angles1)
    data1.close()
    data2.close()
    if extra_suffix:
        suffix_test += "_" + extra_suffix
    print("{:45}".format("Beginning Comparative analysis using Elephant"))
    print("{:45}".format("Suffix for generated figures"), ":", suffix_test)
    print("{:45}".format("Simtimes"), ":", simtime1, " and ", simtime2)
    print("{:45}".format("Assembling neo blocks..."))
    start_time = datetime.now()
    exc_block1 = package_neo_block(exc_spikes1, "Excitatory population 1", N_layer1, simtime1, archive1)
    inh_block1 = package_neo_block(inh_spikes1, "Inhibitory population 1", N_layer1, simtime1, archive1)
    exc_block2 = package_neo_block(exc_spikes2, "Excitatory population 2", N_layer2, simtime2, archive2)
    inh_block2 = package_neo_block(inh_spikes2, "Inhibitory population 2", N_layer2, simtime2, archive2)
    end_time = datetime.now()
    total_time = end_time - start_time
    print("{:45}".format("Neo blocks assembled. Process took {}".format(total_time)))

    exc_segment1 = exc_block1.segments[0]
    inh_segment1 = inh_block1.segments[0]
    exc_segment2 = exc_block2.segments[0]
    inh_segment2 = inh_block2.segments[0]

    # analysis using spike_train_dissimilarity
    # https://elephant.readthedocs.io/en/latest/reference/spike_train_dissimilarity.html
    # Could use that to see exactly how Random and Constant delay networks are different
    return exc_block1, inh_block1, exc_block2, inh_block2


if __name__ == "__main__":
    import sys

    # fname = args.preproc_folder + "motion_batch_analysis_120019_22122018"
    # info_fname = args.preproc_folder + "batch_5499ba5019881fd475ec21bd36e4c8b0"
    # batch_analyser(fname, info_fname)

    # filenames = [
    #     "results_for_testing_random_delay_smax_128_gmax_1_24k_sigma_7.5_3_angle_0_evo",
    #     "results_for_testing_random_delay_smax_128_gmax_1_48k_sigma_7.5_3_angle_0_evo",
    #     "results_for_testing_random_delay_smax_128_gmax_1_96k_sigma_7.5_3_angle_0_evo",
    #     "results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0",
    #     "results_for_testing_random_delay_smax_128_gmax_1_384k_sigma_7.5_3_angle_0_evo",
    #     "results_for_testing_random_delay_smax_128_gmax_1_768k_sigma_7.5_3_angle_0_evo"]
    #
    # times = [2400 * bunits.second, 4800 * bunits.second, 9600 * bunits.second, 19200 * bunits.second,
    #          38400 * bunits.second, 76800 * bunits.second]
    #
    # evolution(filenames, times, path=args.preproc_folder, suffix="1_angles_0")
    # sys.exit()

    # Single experiment analysis
    # Runs for 192k ms or ~5 hours ---------------------------
    # 1 angle
    print("{:45}".format("Generating single experiment plots ..."))
    fname = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0"
    analyse_one(fname)

    fname = args.preproc_folder + "results_for_testing_constant_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_evo"
    analyse_one(fname, extra_suffix="constant")

    fname = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_45_evo"
    analyse_one(fname)

    # 2 angles
    fname = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_evo"
    analyse_one(fname)

    fname = args.preproc_folder + "results_for_testing_constant_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90"
    analyse_one(fname, extra_suffix="constant")

    fname = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_45_135_evo"
    analyse_one(fname)

    # 4 angles
    fname = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_NESW_evo"
    analyse_one(fname)

    fname = args.preproc_folder + "results_for_testing_constant_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_NESW"
    analyse_one(fname, extra_suffix="constant")

    # all angles
    fname = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_all_angles"
    analyse_one(fname)

    fname = args.preproc_folder + "results_for_testing_constant_delay_smax_128_gmax_1_192k_sigma_7.5_3_all_angles"
    analyse_one(fname, extra_suffix="constant")

    # Runs for 384k ms or ~10 hours ---------------------------
    # 1 angle
    fname = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_384k_sigma_7.5_3_angle_0_evo"
    analyse_one(fname, extra_suffix="384k")

    fname = args.preproc_folder + "results_for_testing_constant_delay_smax_128_gmax_1_384k_sigma_7.5_3_angle_0_evo"
    analyse_one(fname, extra_suffix="constant_384k")

    fname = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_384k_sigma_7.5_3_angle_45_evo"
    analyse_one(fname, extra_suffix="384k")

    # 2 angles
    fname = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_384k_sigma_7.5_3_angle_0_90_evo"
    analyse_one(fname, extra_suffix="384k")

    # TODO
    # fname = args.preproc_folder + "results_for_testing_constant_delay_smax_128_gmax_1_384k_sigma_7.5_3_angle_0_90"
    # analyse_one(fname, extra_suffix="constant")

    # TODO
    fname = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_384k_sigma_7.5_3_angle_45_135_evo"
    analyse_one(fname, extra_suffix="384k")

    # 4 angles
    fname = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_384k_sigma_7.5_3_angle_NESW_evo"
    analyse_one(fname, extra_suffix="384k")

    # TODO
    # fname = args.preproc_folder + "results_for_testing_constant_delay_smax_128_gmax_1_384k_sigma_7.5_3_angle_NESW"
    # analyse_one(fname, extra_suffix="constant")

    # all angles
    fname = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_384k_sigma_7.5_3_angle_all_evo"
    analyse_one(fname, extra_suffix="384k")
    # TODO
    # fname = args.preproc_folder + "results_for_testing_constant_delay_smax_128_gmax_1_384k_sigma_7.5_3_all_angles"
    # analyse_one(fname, extra_suffix="constant_384k")

    # Comparison between 2 experiments
    # Runs for 192k ms or ~5 hours ---------------------------
    # 1 angle
    print("{:45}".format("Generating 2 experiment comparison plots ..."))

    fname1 = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0"
    fname2 = args.preproc_folder + "results_for_testing_constant_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_evo"
    comparison(fname1, fname2)

    # 2 angles

    fname1 = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_evo"
    fname2 = args.preproc_folder + "results_for_testing_constant_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90"
    comparison(fname1, fname2)

    # 4 angles

    fname1 = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_NESW_evo"
    fname2 = args.preproc_folder + "results_for_testing_constant_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_NESW"
    comparison(fname1, fname2)

    # all angles

    fname1 = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_all_angles"
    fname2 = args.preproc_folder + "results_for_testing_constant_delay_smax_128_gmax_1_192k_sigma_7.5_3_all_angles"
    comparison(fname1, fname2)

    # Comparison between 2 experiments of different durations
    diff_duration_custom_labels = ["~5 hours", "~10 hours"]
    # 1 angle
    # 0 degrees
    fname1 = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0"
    fname2 = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_384k_sigma_7.5_3_angle_0_evo"

    comparison(fname1, fname2, extra_suffix="192k_vs_384k", custom_labels=diff_duration_custom_labels)

    fname1 = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_384k_sigma_7.5_3_angle_0_evo"
    fname2 = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_768k_sigma_7.5_3_angle_0_evo"

    comparison(fname1, fname2, extra_suffix="192k_vs_384k", custom_labels=["~10 hours", "~20 hours"])

    # 45 degrees
    fname1 = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_45_evo"
    fname2 = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_384k_sigma_7.5_3_angle_45_evo"

    comparison(fname1, fname2, extra_suffix="192k_vs_384k", custom_labels=diff_duration_custom_labels)

    # 0 vs 45
    diff_angles_custom_labels = ["0", "45"]
    fname1 = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0"
    fname2 = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_45_evo"
    comparison(fname1, fname2, extra_suffix="0_vs_45", custom_labels=diff_angles_custom_labels)

    # and longer

    fname1 = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_384k_sigma_7.5_3_angle_0_evo"
    fname2 = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_384k_sigma_7.5_3_angle_45_evo"
    comparison(fname1, fname2, extra_suffix="0_vs_45_384k", custom_labels=diff_angles_custom_labels)

    # 2 angles
    # 0 and 90 degrees
    fname1 = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_evo"
    fname2 = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_384k_sigma_7.5_3_angle_0_90_evo"

    comparison(fname1, fname2, extra_suffix="192k_vs_384k", custom_labels=diff_duration_custom_labels)
    # 45 and 135 degrees
    fname1 = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_45_135_evo"
    fname2 = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_384k_sigma_7.5_3_angle_45_135_evo"

    comparison(fname1, fname2, extra_suffix="192k_vs_384k", custom_labels=diff_duration_custom_labels)

    # 0+90 vs 45+135
    diff_angles_custom_labels = ["0 and 90", "45 and 135"]
    fname1 = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_evo"
    fname2 = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_45_135_evo"
    comparison(fname1, fname2, extra_suffix="0_90_vs_45_135", custom_labels=diff_angles_custom_labels)

    # and longer
    fname1 = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_384k_sigma_7.5_3_angle_0_90_evo"
    fname2 = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_384k_sigma_7.5_3_angle_45_135_evo"
    comparison(fname1, fname2, extra_suffix="0_90_vs_45_135_384k", custom_labels=diff_angles_custom_labels)

    # 4 angles
    fname1 = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_NESW_evo"
    fname2 = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_384k_sigma_7.5_3_angle_NESW_evo"

    comparison(fname1, fname2, extra_suffix="192k_vs_384k", custom_labels=diff_duration_custom_labels)

    # all angles
    fname1 = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_all_evo"
    fname2 = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_384k_sigma_7.5_3_angle_all_evo"

    comparison(fname1, fname2, extra_suffix="192k_vs_384k", custom_labels=diff_duration_custom_labels)

    # Generating evolution plots
    # 1 angle, random delays
    print("{:45}".format("Generating evolution plots ..."))
    filenames = [
        "results_for_testing_random_delay_smax_128_gmax_1_24k_sigma_7.5_3_angle_0_evo",
        "results_for_testing_random_delay_smax_128_gmax_1_48k_sigma_7.5_3_angle_0_evo",
        "results_for_testing_random_delay_smax_128_gmax_1_96k_sigma_7.5_3_angle_0_evo",
        "results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0",
        "results_for_testing_random_delay_smax_128_gmax_1_384k_sigma_7.5_3_angle_0_evo",
        "results_for_testing_random_delay_smax_128_gmax_1_768k_sigma_7.5_3_angle_0_evo"]

    times = [2400 * bunits.second, 4800 * bunits.second, 9600 * bunits.second, 19200 * bunits.second,
             38400 * bunits.second, 76800 * bunits.second]

    evolution(filenames, times, path=args.preproc_folder, suffix="1_angles_0")

    # 1 angle, constant delays
    filenames = [
        "results_for_testing_constant_delay_smax_128_gmax_1_24k_sigma_7.5_3_angle_0_evo",
        "results_for_testing_constant_delay_smax_128_gmax_1_48k_sigma_7.5_3_angle_0_evo",
        "results_for_testing_constant_delay_smax_128_gmax_1_96k_sigma_7.5_3_angle_0_evo",
        "results_for_testing_constant_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_evo",
        "results_for_testing_constant_delay_smax_128_gmax_1_384k_sigma_7.5_3_angle_0_evo",
        "results_for_testing_constant_delay_smax_128_gmax_1_768k_sigma_7.5_3_angle_0_evo"]

    evolution(filenames, times, path=args.preproc_folder, suffix="constant_1_angles_0")

    # 2 angles, random delays, 0 + 90
    filenames = [
        "results_for_testing_random_delay_smax_128_gmax_1_24k_sigma_7.5_3_angle_0_90_evo",
        "results_for_testing_random_delay_smax_128_gmax_1_48k_sigma_7.5_3_angle_0_90_evo",
        "results_for_testing_random_delay_smax_128_gmax_1_96k_sigma_7.5_3_angle_0_90_evo",
        "results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_evo",
        "results_for_testing_random_delay_smax_128_gmax_1_384k_sigma_7.5_3_angle_0_90_evo",
        "results_for_testing_random_delay_smax_128_gmax_1_768k_sigma_7.5_3_angle_0_90_evo"
    ]
    evolution(filenames, times, path=args.preproc_folder, suffix="2_angles_0_90")

    # 45 degrees
    filenames = [
        "results_for_testing_random_delay_smax_128_gmax_1_24k_sigma_7.5_3_angle_45_evo",
        "results_for_testing_random_delay_smax_128_gmax_1_48k_sigma_7.5_3_angle_45_evo",
        "results_for_testing_random_delay_smax_128_gmax_1_96k_sigma_7.5_3_angle_45_evo",
        "results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_45_evo",
        "results_for_testing_random_delay_smax_128_gmax_1_384k_sigma_7.5_3_angle_45_evo"
        # TODO extra time
    ]
    times = [2400 * bunits.second, 4800 * bunits.second, 9600 * bunits.second, 19200 * bunits.second,
             38400 * bunits.second
             ]
    evolution(filenames, times, path=args.preproc_folder, suffix="_1_angles_45")

    # 2 angles, random delays, 45 and 135
    filenames = [
        "results_for_testing_random_delay_smax_128_gmax_1_24k_sigma_7.5_3_angle_45_135_evo",
        "results_for_testing_random_delay_smax_128_gmax_1_48k_sigma_7.5_3_angle_45_135_evo",
        "results_for_testing_random_delay_smax_128_gmax_1_96k_sigma_7.5_3_angle_45_135_evo",
        "results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_45_135_evo",
        "results_for_testing_random_delay_smax_128_gmax_1_384k_sigma_7.5_3_angle_45_135_evo",
        # TODO extra times
    ]
    times = [2400 * bunits.second, 4800 * bunits.second, 9600 * bunits.second, 19200 * bunits.second,
             38400 * bunits.second,
             ]
    evolution(filenames, times, path=args.preproc_folder, suffix="_2_angles_45_135")

    # 4 angles, random delays, 0 + 90 + 180 + 270
    filenames = [
        "results_for_testing_random_delay_smax_128_gmax_1_24k_sigma_7.5_3_angle_NESW_evo",
        "results_for_testing_random_delay_smax_128_gmax_1_48k_sigma_7.5_3_angle_NESW_evo",
        "results_for_testing_random_delay_smax_128_gmax_1_96k_sigma_7.5_3_angle_NESW_evo",
        "results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_NESW_evo",
        "results_for_testing_random_delay_smax_128_gmax_1_384k_sigma_7.5_3_angle_NESW_evo",
        "results_for_testing_random_delay_smax_128_gmax_1_768k_sigma_7.5_3_angle_NESW_evo"
    ]

    times = [2400 * bunits.second, 4800 * bunits.second, 9600 * bunits.second, 19200 * bunits.second,
             38400 * bunits.second, 76800 * bunits.second]
    evolution(filenames, times, path=args.preproc_folder, suffix="_4_angles_0_90_180_270")

    filenames = [
        "results_for_testing_without_noise_random_delay_smax_128_gmax_1_24k_sigma_7.5_3_angle_NESW_evo",
        "results_for_testing_without_noise_random_delay_smax_128_gmax_1_48k_sigma_7.5_3_angle_NESW_evo",
        "results_for_testing_without_noise_random_delay_smax_128_gmax_1_96k_sigma_7.5_3_angle_NESW_evo",
        "results_for_testing_without_noise_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_NESW_evo",
        "results_for_testing_without_noise_random_delay_smax_128_gmax_1_384k_sigma_7.5_3_angle_NESW_evo",
        "results_for_testing_without_noise_random_delay_smax_128_gmax_1_768k_sigma_7.5_3_angle_NESW_evo"
    ]

    times = [2400 * bunits.second, 4800 * bunits.second, 9600 * bunits.second, 19200 * bunits.second,
             38400 * bunits.second, 76800 * bunits.second]
    evolution(filenames, times, path=args.preproc_folder, suffix="_4_angles_0_90_180_270_testing_without_noise")

    filenames = [
        "results_for_testing_training_without_noise_random_delay_smax_128_gmax_1_24k_sigma_7.5_3_angle_NESW_evo",
        "results_for_testing_training_without_noise_random_delay_smax_128_gmax_1_48k_sigma_7.5_3_angle_NESW_evo",
        "results_for_testing_training_without_noise_random_delay_smax_128_gmax_1_96k_sigma_7.5_3_angle_NESW_evo",
        "results_for_testing_training_without_noise_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_NESW_evo",
        "results_for_testing_training_without_noise_random_delay_smax_128_gmax_1_384k_sigma_7.5_3_angle_NESW_evo",
        "results_for_testing_training_without_noise_random_delay_smax_128_gmax_1_768k_sigma_7.5_3_angle_NESW_evo"
    ]

    times = [2400 * bunits.second, 4800 * bunits.second, 9600 * bunits.second, 19200 * bunits.second,
             38400 * bunits.second,
             76800 * bunits.second
             ]
    evolution(filenames, times, path=args.preproc_folder, suffix="_4_angles_0_90_180_270_training_without_noise")

    # all angles
    filenames = [
        "results_for_testing_random_delay_smax_128_gmax_1_24k_sigma_7.5_3_angle_all_evo",
        "results_for_testing_random_delay_smax_128_gmax_1_48k_sigma_7.5_3_angle_all_evo",
        "results_for_testing_random_delay_smax_128_gmax_1_96k_sigma_7.5_3_angle_all_evo",
        "results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_all_evo",
        "results_for_testing_random_delay_smax_128_gmax_1_384k_sigma_7.5_3_angle_all_evo",
        # "results_for_testing_random_delay_smax_128_gmax_1_768k_sigma_7.5_3_angle_all_evo"
    ]

    times = [2400 * bunits.second, 4800 * bunits.second, 9600 * bunits.second, 19200 * bunits.second,
             38400 * bunits.second,
             # 76800 * bunits.second
             ]
    evolution(filenames, times, path=args.preproc_folder, suffix="_all_angles")

    # Experiment batch analysis -- usually, these are sensitivity analysis
    print("{:45}".format("Generating batch analysis plots ..."))
    fname = args.preproc_folder + "motion_batch_analysis_120019_22122018"
    info_fname = args.preproc_folder + "batch_5499ba5019881fd475ec21bd36e4c8b0"
    batch_analyser(fname, info_fname)

    # Elephant analysis of single experiments
    print("{:45}".format("Generating Elephant plots ..."))
    fname = "testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0"
    elephant_analysis(fname, time_to_waste=args.time_to_waste)
