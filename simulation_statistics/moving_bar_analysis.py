from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm as cm_mlib
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import scipy
from matplotlib import animation, rc, colors
from brian2.units import *
import matplotlib as mlib
from scipy import stats
from pprint import pprint as pp
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
import traceback, os
from argparser import *
from gari_analysis_functions import *
from analysis_functions_definitions import *
from synaptogenesis.function_definitions import generate_equivalent_connectivity

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
root_stats = "C:\Work\phd\simulation_statistics\\"
root_syn = "C:\Work\phd\synaptogenesis\\"
fig_folder = args.fig_folder
testing_data = np.load(
    root_syn + "spiking_moving_bar_input\spiking_moving_bar_motif_bank_simtime_1200s.npz")
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


def analyse_one(archive, out_filename=None, extra_suffix=None, show_plots=True):
    # in the default case, we are only looking at understanding a number of
    # behaviours of a single simulation
    cached_data = np.load(root_stats + archive + ".npz")

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

    suffix_test = generate_suffix(training_sim_params['training_angles'])
    if extra_suffix:
        suffix_test += "_" + extra_suffix

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
        ax.axvline((i * 90 / 180.) * np.pi, color="#aaaaaa", lw=4, zorder=1)
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

    number_of_afferents = np.empty(N_layer)
    for index, value in np.ndenumerate(number_of_afferents):
        number_of_afferents[index] = np.nansum(
            ff_num_network[:, index[0]]) + np.nansum(
            lat_num_network[:, index[0]])

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
                edgecolor='k')
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
        ell = Ellipse(xy=(0, 0), width=wavg, height=havg, angle=aavg,
                      facecolor='none', edgecolor='#b2dd2c', linewidth=3.0)
        ax.add_artist(ell)

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

    cached_data.close()
    return suffix_test


def comparison(archive_random, archive_constant, out_filename=None,
               show_plots=True):
    # in the default use case, the 2 archives would relate to two networks
    # differing only in the way that structural plasticity chooses delays
    if ".npz" in archive_random:
        random_delay_data = np.load(root_stats + archive_random)
    else:
        random_delay_data = np.load(root_stats + archive_random + ".npz")

    if ".npz" in archive_random:
        constant_delay_data = np.load(root_stats + archive_constant)
    else:
        constant_delay_data = np.load(root_stats + archive_constant + ".npz")

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
    # TODO Begin asserts

    # generate suffix
    suffix_test = generate_suffix(sim_params['training_angles'])

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

    # ax.set_xlabel("Angle")
    # ax2.set_xlabel("Angle")
    ax.set_xlabel("Random delays")
    ax2.set_xlabel("Constant delays")
    ax2.set_ylim([minimus, 1.1 * maximus])

    # f.suptitle("Mean firing rate for specific input angle", va='bottom')
    plt.tight_layout(pad=10)
    plt.savefig(
        fig_folder + "comparison_number_of_sensitised_neurons_with_angle{" \
                     "}.pdf".format(
            suffix_test), bbox_inches='tight')
    plt.savefig(
        fig_folder + "comparison_number_of_sensitised_neurons_with_angle{" \
                     "}.svg".format(
            suffix_test), bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close(fig)


def evolution(filenames, times, suffix, show_plots=False):
    # https://gist.github.com/MatthewJA/5a0a6d75748bf5cb5962cb9d5572a6ce
    viridis_cmap = mlib.cm.get_cmap('viridis')
    # root_stats = "D:\Work\Neurogenesis-PhD\simulation_statistics\\"
    root_stats = "C:\Work\phd\simulation_statistics\\preproc\\"
    # root_syn = "D:\Work\Neurogenesis-PhD\synaptogenesis\\"
    root_syn = "C:\Work\phd\synaptogenesis\\"
    no_files = len(filenames)
    assert len(filenames) == len(times)
    all_rate_means = []
    all_rate_sems = []
    all_radians = []
    for fn in filenames:
        cached_data = np.load(root_stats + fn + ".npz")
        testing_data = np.load(
            root_syn + "spiking_moving_bar_input\spiking_moving_bar_motif_bank_simtime_1200s.npz")
        rate_means = cached_data['rate_means']
        rate_stds = cached_data['rate_stds']
        rate_sem = cached_data['rate_sem']
        all_rates = cached_data['all_rates']
        radians = cached_data['radians']

        all_rate_means.append(rate_means)
        all_rate_sems.append(rate_sem)
        all_radians = radians
    fig = plt.figure(figsize=(10, 10), dpi=800)
    ax = plt.subplot(111, projection='polar')

    for i in range(no_files):
        c = plt.fill(radians, all_rate_means[i],
                     c=viridis_cmap(float(i) / (no_files - 1)),
                     label="{} minutes".format(times[i] / (60 * second)),
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


def batch_analyser(archive_batch, out_folder):
    pass


def sigma_and_ad_analyser(archive, out_filename=None, extra_suffix=None, show_plots=True):
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


if __name__ == "__main__":
    # Single experiment analysis
    # 1 angle
    fname = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0"
    analyse_one(fname, show_plots=False)

    fname = args.preproc_folder + "results_for_testing_constant_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_evo"
    analyse_one(fname, extra_suffix="constant", show_plots=False)

    fname = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_45_evo"
    analyse_one(fname, show_plots=False)

    # 2 angles
    fname = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_evo"
    analyse_one(fname, show_plots=False)

    fname = args.preproc_folder + "results_for_testing_constant_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90"
    analyse_one(fname, extra_suffix="constant", show_plots=False)

    fname = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_45_135_evo"
    analyse_one(fname, show_plots=False)

    # 4 angles
    fname = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_NESW_evo"
    analyse_one(fname, show_plots=False)

    fname = args.preproc_folder + "results_for_testing_constant_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_NESW"
    analyse_one(fname, extra_suffix="constant", show_plots=False)

    # all angles
    fname = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_all_angles"
    analyse_one(fname, show_plots=False)

    fname = args.preproc_folder + "results_for_testing_constant_delay_smax_128_gmax_1_192k_sigma_7.5_3_all_angles"
    analyse_one(fname, extra_suffix="constant", show_plots=False)

    # Comparison between 2 experiments
    # 1 angle

    fname1 = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0"

    fname2 = args.preproc_folder + "results_for_testing_constant_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_evo"

    comparison(fname1, fname2, show_plots=False)

    # 2 angles

    fname1 = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_evo"

    fname2 = args.preproc_folder + "results_for_testing_constant_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90"

    comparison(fname1, fname2, show_plots=False)

    # 4 angles

    fname1 = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_NESW_evo"
    fname2 = args.preproc_folder + "results_for_testing_constant_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_NESW"
    comparison(fname1, fname2, show_plots=False)

    # all angles

    fname1 = args.preproc_folder + "results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_all_angles"
    fname2 = args.preproc_folder + "results_for_testing_constant_delay_smax_128_gmax_1_192k_sigma_7.5_3_all_angles"
    comparison(fname1, fname2, show_plots=False)

    # Generating evolution plots
    # 1 angle, random delays
    filenames = [
        "results_for_testing_random_delay_smax_128_gmax_1_24k_sigma_7.5_3_angle_0_evo",
        "results_for_testing_random_delay_smax_128_gmax_1_48k_sigma_7.5_3_angle_0_evo",
        "results_for_testing_random_delay_smax_128_gmax_1_96k_sigma_7.5_3_angle_0_evo",
        "results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0",
        "results_for_testing_random_delay_smax_128_gmax_1_384k_sigma_7.5_3_angle_0_evo",
        "results_for_testing_random_delay_smax_128_gmax_1_768k_sigma_7.5_3_angle_0_evo"]

    times = [2400 * second, 4800 * second, 9600 * second, 19200 * second,
             38400 * second, 76800 * second]

    evolution(filenames, times, suffix="1_angles_0")

    # 1 angle, constant delays
    filenames = [
        "results_for_testing_constant_delay_smax_128_gmax_1_24k_sigma_7.5_3_angle_0_evo",
        "results_for_testing_constant_delay_smax_128_gmax_1_48k_sigma_7.5_3_angle_0_evo",
        "results_for_testing_constant_delay_smax_128_gmax_1_96k_sigma_7.5_3_angle_0_evo",
        "results_for_testing_constant_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_evo",
        "results_for_testing_constant_delay_smax_128_gmax_1_384k_sigma_7.5_3_angle_0_evo",
        "results_for_testing_constant_delay_smax_128_gmax_1_768k_sigma_7.5_3_angle_0_evo"]

    evolution(filenames, times, suffix="constant_1_angles_0")

    # 2 angles, random delays, 0 + 90
    filenames = [
        "results_for_testing_random_delay_smax_128_gmax_1_24k_sigma_7.5_3_angle_0_90_evo",
        "results_for_testing_random_delay_smax_128_gmax_1_48k_sigma_7.5_3_angle_0_90_evo",
        "results_for_testing_random_delay_smax_128_gmax_1_96k_sigma_7.5_3_angle_0_90_evo",
        "results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_evo",
        "results_for_testing_random_delay_smax_128_gmax_1_384k_sigma_7.5_3_angle_0_90_evo",
        "results_for_testing_random_delay_smax_128_gmax_1_768k_sigma_7.5_3_angle_0_90_evo"
    ]
    evolution(filenames, times, suffix="2_angles_0_90")

    # 45 degrees
    filenames = [
        "results_for_testing_random_delay_smax_128_gmax_1_24k_sigma_7.5_3_angle_45_evo",
        "results_for_testing_random_delay_smax_128_gmax_1_48k_sigma_7.5_3_angle_45_evo",
        "results_for_testing_random_delay_smax_128_gmax_1_96k_sigma_7.5_3_angle_45_evo",
        "results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_45_evo",
        "results_for_testing_random_delay_smax_128_gmax_1_384k_sigma_7.5_3_angle_45_evo"
        # TODO extra time
    ]
    times = [2400 * second, 4800 * second, 9600 * second, 19200 * second,
             38400 * second
             ]
    evolution(filenames, times, suffix="_1_angles_45")

    # 2 angles, random delays, 45 and 135
    filenames = [
        "results_for_testing_random_delay_smax_128_gmax_1_24k_sigma_7.5_3_angle_45_135_evo",
        "results_for_testing_random_delay_smax_128_gmax_1_48k_sigma_7.5_3_angle_45_135_evo",
        "results_for_testing_random_delay_smax_128_gmax_1_96k_sigma_7.5_3_angle_45_135_evo",
        "results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_45_135_evo",
        # TODO extra times
    ]
    times = [2400 * second, 4800 * second, 9600 * second, 19200 * second,
             ]
    evolution(filenames, times, suffix="_2_angles_45_135")

    # 4 angles, random delays, 0 + 90 + 180 + 270
    filenames = [
        "results_for_testing_random_delay_smax_128_gmax_1_24k_sigma_7.5_3_angle_NESW_evo",
        "results_for_testing_random_delay_smax_128_gmax_1_48k_sigma_7.5_3_angle_NESW_evo",
        "results_for_testing_random_delay_smax_128_gmax_1_96k_sigma_7.5_3_angle_NESW_evo",
        "results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_NESW_evo",
        "results_for_testing_random_delay_smax_128_gmax_1_384k_sigma_7.5_3_angle_NESW_evo",
        # TODO
        #     "results_for_testing_random_delay_smax_128_gmax_1_768k_sigma_7.5_3_angle_45_diff_shared_seeds"
    ]

    times = [2400 * second, 4800 * second, 9600 * second, 19200 * second,
             38400 * second,
             #          76800*second
             ]
    evolution(filenames, times, suffix="_4_angles_0_90_180_270")

    # all angles
    # TODO
