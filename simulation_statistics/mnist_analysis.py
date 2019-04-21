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
from pprint import pprint as pp
from sklearn.metrics import classification_report, confusion_matrix
import neo
# imports related to Elephant analysis
# from elephant import statistics, spade, spike_train_correlation, spike_train_dissimilarity, conversion
# import elephant.cell_assembly_detection as cad
# import neo
from datetime import datetime
# from quantities import s, ms, Hz
from brian2.units import ms, Hz, second
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ensure we use viridis as the default cmap
plt.viridis()

# ensure we use the same rc parameters for all matplotlib outputs
mlib.rcParams.update({'font.size': 22})
mlib.rcParams.update({'errorbar.capsize': 5})
# mlib.rcParams.update({'figure.autolayout': True})

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

# check if the figures folder exist
if not os.path.isdir(fig_folder) and not os.path.exists(fig_folder):
    os.mkdir(fig_folder)


def mnist_analysis(archive, out_filename=None, extra_suffix=None, show_plots=False):
    if ".npz" in archive:
        data = np.load(os.path.join(root_syn, archive))
        testing_data = np.load(os.path.join(root_syn, "testing_" + archive))
    else:
        data = np.load(os.path.join(root_syn, archive + ".npz"))
        testing_data = np.load(os.path.join(root_syn, "testing_" + archive + ".npz"))
    sim_params = data['sim_params'].ravel()[0]
    is_input_cs = False

    suffix = "_case_" + str(sim_params['case'])

    if 'final_pre_on_weights' in data.files:
        suffix += "_cs"
        is_input_cs = True
    else:
        suffix += "_rate"

    if extra_suffix:
        suffix += "_" + extra_suffix

    print("{:45}".format("Beginning MNIST analysis"))
    print("{:45}".format("Archive name"), ":", archive)
    print("{:45}".format("Suffix"), ":", suffix)
    print("{:45}".format("Reporting some parameters used in the current simulation"))
    print("{:45}".format("Lateral Inhibition"), ":", bool(sim_params['lateral_inhibition']))
    print("{:45}".format("Simulation time"), ":", sim_params['simtime'])
    print("{:45}".format("STDP t_minus"), ":", sim_params['t_minus'])
    print("{:45}".format("STDP t_plus"), ":", sim_params['t_plus'])
    print("{:45}".format("SR Synaptic capacity"), ":", sim_params['s_max'])
    print("{:45}".format("SR sigma_form_forward"), ":", sim_params['sigma_form_forward'])
    print("{:45}".format("SR sigma_form_lateral"), ":", sim_params['sigma_form_lateral'])
    print("{:45}".format("Grid shape"), ":", sim_params['grid'])
    print("{:45}".format("Input type"), ":", sim_params['input_type'])

    N_layer = sim_params['grid'][0] * sim_params['grid'][1]
    s_max = sim_params['s_max']
    g_max = sim_params['g_max']

    simtime = data['simtime'].ravel()[0]
    post_spikes = data['post_spikes']

    new_post_spikes = []
    if isinstance(post_spikes[0], neo.Block):
        for i in range(10):
            new_post_spikes.append(convert_spikes(post_spikes[i]))
    if len(new_post_spikes) > 0:
        post_spikes = new_post_spikes

    if is_input_cs:
        final_ff_on_conn = data['ff_on_connections'][-10:]
        final_ff_off_conn = data['ff_off_connections'][-10:]
        final_ff_conn = []
        for on_conn, off_conn in zip(final_ff_on_conn, final_ff_off_conn):
            final_ff_conn.append(np.concatenate((on_conn, off_conn)))
    else:
        final_ff_conn = data['ff_connections'][-10:]
    final_lat_conn = data['lat_connections'][-10:]

    testing_simtime = testing_data['simtime'].ravel()[0]
    testing_numbers = testing_data['testing_numbers']

    # stlye the median of boxplots
    medianprops = dict(color='#414C82', linewidth=1.5)

    # Final post-synaptic firing
    source_hits = np.empty(28 ** 2)
    source_weighted_hits = np.empty(28 ** 2)
    rates_for_number = np.zeros((10, 28 ** 2))
    print("-" * 60)
    print("{:45}".format("Average training firing rates (Hz)"))
    for number in range(10):
        for neuron_id in range(28 ** 2):
            rates_for_number[number, neuron_id] = np.count_nonzero(
                post_spikes[number][:, 0] == neuron_id)

        print("{:45}".format("Average firing rate (Hz) for number " + str(number)), ":",
              np.mean(rates_for_number[number, :])/(simtime * ms))
    print("-"*60)

    const_fig_width = 23
    const_fig_height = 7

    fig_conn, axes = plt.subplots(2, 5, figsize=(const_fig_width, const_fig_height), dpi=600, sharey=True)

    silly_ax = []
    maximus = [-1 * Hz]
    minimus = [2 ** 31 * Hz]

    for index, val in np.ndenumerate(axes):
        x, y = index
        source_weighted_hits = rates_for_number[x * 5 + y, :].reshape(28, 28) / (simtime * ms)
        maximus = np.maximum(maximus, source_weighted_hits.max())
        minimus = np.minimum(minimus, source_weighted_hits.min())

        silly_ax.append(axes[x, y].matshow(source_weighted_hits / Hz))

    # ff_conn_ax = axes[0, 0].matshow(source_hits.reshape(28, 28))
    # weighted_conn_ax = axes[1, 1].matshow(source_weighted_hits.reshape(28, 28))

    # ax1.set_title("Hits\n")
    # ax1.set_xlabel("Neuron ID")
    axes[0, 0].set_ylabel("Neuron ID")
    # ax2.set_title("Weighted hits\n")
    # ax2.set_xlabel("Neuron ID")
    axes[1, 0].set_ylabel("Neuron ID")

    for arg in range(5):
        axes[1, arg].set_xlabel("Neuron ID")

    norm = colors.Normalize(vmin=minimus / Hz, vmax=maximus / Hz)
    for index, val in np.ndenumerate(axes):
        x, y = index
        silly_ax[x * 5 + y].set_norm(norm)
    # fig_conn.subplots_adjust(right=0.8)
    # cbar_ax = fig_conn.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig_conn.colorbar(silly_ax[4], cax=cbar_ax)

    # from mpl_toolkits.axes_grid1 import make_axes_locatable
    # divider = make_axes_locatable(plt.gca())
    # cax = divider.append_axes("right", "5%", pad="3%")
    # plt.colorbar(silly_ax[4], cax=cax)
    fig_conn.colorbar(silly_ax[-1], ax=axes.ravel().tolist(), label="Firing rate (Hz)")

    # cbaxes = fig_conn.add_axes([0.9, 0.01, 0., 1.])
    # fig_conn.colorbar(silly_ax[-2], ax=cbaxes, label="Firing rate (Hz)")

    # plt.tight_layout()
    plt.savefig(fig_folder + "mnist_total_target_hits_rate_based{}.pdf".format(suffix))
    plt.savefig(fig_folder + "mnist_total_target_hits_rate_based{}.svg".format(suffix))
    if show_plots:
        plt.show()
    plt.close(fig_conn)

    # Connectivity reconstruction

    fig_conn, axes = plt.subplots(2, 5, figsize=(const_fig_width, const_fig_height), dpi=500, sharey=True)

    silly_ax = []
    maximus = [-1]
    minimus = [2 ** 31]

    print("-" * 60)
    print("{:45}".format("Average weight proportion [0, 1]"))
    weights_per_number = []
    for index, val in np.ndenumerate(axes):
        x, y = index
        source_weighted_hits = np.empty(28 ** 2)
        number = x * 5 + y
        conn_list = final_ff_conn[number]
        weights_per_number.append([conn_list[:, 2]/g_max])
        for i in range(28 ** 2):
            source_weighted_hits[i] = np.sum(conn_list[conn_list[:, 0] == i, 2])
        maximus = np.maximum(maximus, source_weighted_hits.max())
        minimus = np.minimum(minimus, source_weighted_hits.min())

        # Report some stats
        print("{:45}".format("Average normalised weight for number " + str(number)), ":",
              np.mean(conn_list[:, 2]) / g_max)

        silly_ax.append(axes[x, y].matshow(source_weighted_hits.reshape(28, 28)))

    print("-" * 60)
    axes[0, 0].set_ylabel("Neuron ID")
    axes[1, 0].set_ylabel("Neuron ID")

    for arg in range(5):
        axes[1, arg].set_xlabel("Neuron ID")

    norm = colors.Normalize(vmin=minimus, vmax=maximus)
    for index, val in np.ndenumerate(axes):
        x, y = index
        silly_ax[x * 5 + y].set_norm(norm)
    # fig_conn.subplots_adjust(right=0.8)
    # cbar_ax = fig_conn.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig_conn.colorbar(silly_ax[4], cax=cbar_ax)

    # from mpl_toolkits.axes_grid1 import make_axes_locatable
    # divider = make_axes_locatable(plt.gca())
    # cax = divider.append_axes("right", "5%", pad="3%")
    # plt.colorbar(silly_ax[4], cax=cax)

    fig_conn.colorbar(silly_ax[-1], ax=axes.ravel().tolist(), label="Conn. strength")

    # plt.tight_layout()
    plt.savefig(fig_folder + "mnist_all_digits_weighted_rate_based{}.pdf".format(suffix))
    plt.savefig(fig_folder + "mnist_all_digits_weighted_rate_based{}.svg".format(suffix))
    if show_plots:
        plt.show()
    plt.close(fig_conn)

    # Weight boxplot
    fig = plt.figure(figsize=(16, 8), dpi=600)

    # plt.axhline(s_max, color='#b2dd2c', ls=":")
    bp = plt.boxplot(weights_per_number, notch=True, medianprops=medianprops)

    plt.xticks(np.arange(rates_for_number.shape[0]) + 1, np.arange(rates_for_number.shape[0]))
    plt.xlabel("Target layer")
    plt.ylabel("Normalised weight")
    plt.grid(True, which='major', axis='y')
    plt.savefig(fig_folder + "mnist_normalised_weight_boxplot{}.pdf".format(suffix))
    plt.savefig(fig_folder + "mnist_normalised_weight_boxplot{}.svg".format(suffix))
    if show_plots:
        plt.show()
    plt.close(fig)

    fig_conn, axes = plt.subplots(2, 5, figsize=(const_fig_width, const_fig_height), dpi=500, sharey=True)

    silly_ax = []
    maximus = [-1]
    minimus = [2 ** 31]

    for index, val in np.ndenumerate(axes):
        x, y = index
        source_weighted_hits = np.empty(28 ** 2)
        conn_list = final_ff_conn[x * 5 + y]
        for i in range(28 ** 2):
            source_hits[i] = np.count_nonzero(conn_list[:, 0] == i)
        maximus = np.maximum(maximus, source_hits.max())
        minimus = np.minimum(minimus, source_hits.min())

        silly_ax.append(axes[x, y].matshow(source_hits.reshape(28, 28)))

    # ff_conn_ax = axes[0, 0].matshow(source_hits.reshape(28, 28))
    # weighted_conn_ax = axes[1, 1].matshow(source_weighted_hits.reshape(28, 28))

    # ax1.set_title("Hits\n")
    # ax1.set_xlabel("Neuron ID")
    axes[0, 0].set_ylabel("Neuron ID")
    # ax2.set_title("Weighted hits\n")
    # ax2.set_xlabel("Neuron ID")
    axes[1, 0].set_ylabel("Neuron ID")

    for arg in range(5):
        axes[1, arg].set_xlabel("Neuron ID")

    norm = colors.Normalize(vmin=minimus, vmax=maximus)
    for index, val in np.ndenumerate(axes):
        x, y = index
        silly_ax[x * 5 + y].set_norm(norm)
    fig_conn.colorbar(silly_ax[-1], ax=axes.ravel().tolist(), label="# of conn.")

    plt.savefig(fig_folder + "mnist_all_digits_rate_based{}.pdf".format(suffix))
    plt.savefig(fig_folder + "mnist_all_digits_rate_based{}.svg".format(suffix))
    if show_plots:
        plt.show()
    plt.close(fig_conn)

    chunk = 200
    instaneous_rates = np.empty((10, 300000 // chunk))
    for index, value in np.ndenumerate(instaneous_rates):
        number_index, chunk_index = index
        instaneous_rates[number_index, chunk_index] = np.count_nonzero(
            np.logical_and(
                post_spikes[number_index][:, 1] >= (chunk_index * chunk),
                post_spikes[number_index][:, 1] <= ((chunk_index + 1) * chunk)
            )
        ) / (chunk * ms)

    instaneous_rates /= float(N_layer)
    # firing rate per digit
    fig = plt.figure(figsize=(16, 8), dpi=600)

    # plt.axhline(s_max, color='#b2dd2c', ls=":")
    bp = plt.boxplot(instaneous_rates.T / Hz, notch=True, medianprops=medianprops)

    plt.xticks(np.arange(instaneous_rates.shape[0]) + 1, np.arange(instaneous_rates.shape[0]))
    plt.xlabel("Target layer")
    plt.ylabel("Firing rate (Hz)")
    plt.grid(True, which='major', axis='y')
    plt.savefig(fig_folder + "mnist_training_firing_rate_boxplot{}.pdf".format(suffix))
    plt.savefig(fig_folder + "mnist_training_firing_rate_boxplot{}.svg".format(suffix))
    if show_plots:
        plt.show()
    plt.close(fig)

    ff_all_conns = np.asarray([])
    for ff_conn in final_ff_conn:
        if ff_all_conns.size == 0:
            ff_all_conns = ff_conn
        else:
            ff_all_conns = np.concatenate((ff_all_conns, ff_conn), axis=0)

    lat_all_conns = np.asarray([])
    if final_lat_conn.size > 0:
        for lat_conn in final_lat_conn:
            if lat_all_conns.size == 0:
                lat_all_conns = lat_conn
            else:
                lat_all_conns = np.concatenate((lat_all_conns, lat_conn), axis=0)
    # weight histograms
    if final_lat_conn.size > 0:
        conns = (ff_all_conns, lat_all_conns)
        conns_names = ["$ff$", "$lat$"]
    else:
        conns = (ff_all_conns)
        conns_names = ["$ff$"]

    minimus = 0
    maximus = np.max([1., np.max(ff_all_conns[:, 2] / g_max)])

    fig, axes = plt.subplots(1, len(conns_names), figsize=(7.5 * len(conns_names), 7), sharey=True)
    for index, ax in np.ndenumerate(axes):
        if len(index) == 0:
            i = 0
            curr_conn = ff_all_conns
        else:
            i = index[0]
            curr_conn = conns[i]
        ax.hist(curr_conn[:, 2] / g_max, bins=20, color='#414C82', edgecolor='k', normed=True)
        ax.set_title(conns_names[i])
        ax.set_xlim([minimus, maximus])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_xticks([0, .5, 1])
    plt.tight_layout()
    plt.savefig(fig_folder + "mnist_weight_histograms{}.pdf".format(suffix), bbox_inches='tight')
    plt.savefig(fig_folder + "mnist_weight_histograms{}.svg".format(suffix), bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close(fig)

    # ----------------- Analysing the test ------------------------

    post_spikes = testing_data['post_spikes']
    simtime = testing_simtime
    print("{:45}".format("Testing Simulation time"), ":", testing_simtime)

    new_post_spikes = []
    if isinstance(post_spikes[0], neo.Block):
        for i in range(10):
            new_post_spikes.append(convert_spikes(post_spikes[i]))
    if len(new_post_spikes) > 0:
        post_spikes = new_post_spikes
    rates_for_number = np.zeros((10, 28 ** 2))


    print("-" * 60)
    print("{:45}".format("Average testing firing rates (Hz)"))
    for number in range(10):
        for neuron_id in range(28 ** 2):
            rates_for_number[number, neuron_id] = np.count_nonzero(
                post_spikes[number][:, 0] == neuron_id)
        print("{:45}".format("Average firing rate (Hz) for number " + str(number)), ":",
              np.mean(rates_for_number[number, :]) / (simtime * ms))
    print("-" * 60)

    fig_conn, axes = plt.subplots(2, 5, figsize=(const_fig_width, const_fig_height), dpi=500, sharey=True)

    silly_ax = []
    maximus = [-1 * Hz]
    minimus = [2 ** 31 * Hz]

    for index, val in np.ndenumerate(axes):
        x, y = index
        source_weighted_hits = rates_for_number[x * 5 + y, :].reshape(28, 28) / (simtime * ms)
        maximus = np.maximum(maximus, source_weighted_hits.max())
        minimus = np.minimum(minimus, source_weighted_hits.min())

        silly_ax.append(axes[x, y].matshow(source_weighted_hits / Hz))

    axes[0, 0].set_ylabel("Neuron ID")
    axes[1, 0].set_ylabel("Neuron ID")

    for arg in range(5):
        axes[1, arg].set_xlabel("Neuron ID")

    norm = colors.Normalize(vmin=minimus / Hz, vmax=maximus / Hz)
    for index, val in np.ndenumerate(axes):
        x, y = index
        silly_ax[x * 5 + y].set_norm(norm)

    fig_conn.colorbar(silly_ax[-1], ax=axes.ravel().tolist(), label="Firing rate (Hz)")

    plt.savefig(fig_folder + "testing_total_target_hits_rate_based{}.pdf".format(suffix))
    plt.savefig(fig_folder + "testing_total_target_hits_rate_based{}.svg".format(suffix))
    if show_plots:
        plt.show()
    plt.close(fig_conn)

    chunk = 200
    instaneous_rates = np.empty((10, 300000 // chunk))
    for index, value in np.ndenumerate(instaneous_rates):
        number_index, chunk_index = index
        instaneous_rates[number_index, chunk_index] = np.count_nonzero(
            np.logical_and(
                post_spikes[number_index][:, 1] >= (chunk_index * chunk),
                post_spikes[number_index][:, 1] <= ((chunk_index + 1) * chunk)
            )
        ) / (chunk * ms)

    instaneous_rates /= float(N_layer)
    what_network_thinks = np.empty(300000 // chunk)
    for i in range(what_network_thinks.shape[0]):
        what_network_thinks[i] = np.argmax(instaneous_rates[:, i])
    conf_mat = confusion_matrix(testing_numbers, what_network_thinks, labels=range(10))
    conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)
    fig_conn, ax1 = plt.subplots(1, 1, figsize=(9, 9), dpi=800)

    ff_conn_ax = ax1.matshow(conf_mat, vmin=0, vmax=1)

    ax1.set_title("Confusion matrix\n")
    ax1.set_xlabel("Predicted label")
    ax1.set_ylabel("True label")

    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", "5%", pad="3%")
    cbar = plt.colorbar(ff_conn_ax, cax=cax)
    cbar.set_label("Percentage")

    plt.tight_layout()
    plt.savefig(fig_folder + "mnist_confusion_matrix_rate_based{}.pdf".format(suffix), bbox_inches='tight')
    plt.savefig(fig_folder + "mnist_confusion_matrix_rate_based{}.svg".format(suffix), bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close(fig_conn)
    print(classification_report(testing_numbers, what_network_thinks))

    rmse = np.sqrt(np.mean((testing_numbers - what_network_thinks) ** 2))
    print("{:45}".format("RMSE"), ":", rmse)

    number_of_afferents = []
    if final_lat_conn.size > 0:
        for ff_conn, lat_conn in zip(final_ff_conn, final_lat_conn):
            number_of_afferents.append(get_number_of_afferents_from_list(N_layer, ff_conn, lat_conn))
    else:
        for ff_conn in final_ff_conn:
            number_of_afferents.append(get_number_of_afferents_from_list(N_layer, ff_conn, np.array([])))
    number_of_afferents = np.asarray(number_of_afferents)

    # synaptic capacity per digit
    fig = plt.figure(figsize=(16, 8), dpi=600)

    plt.axhline(s_max, color='#b2dd2c', ls=":")
    bp = plt.boxplot(number_of_afferents.T, notch=True, medianprops=medianprops)

    plt.xticks(np.arange(number_of_afferents.shape[0]) + 1, np.arange(number_of_afferents.shape[0]))
    plt.xlabel("Target layer")
    plt.ylabel("Mean synaptic capacity usage")
    plt.grid(True, which='major', axis='y')
    plt.savefig(fig_folder + "mnist_number_of_afferents_boxplot{}.pdf".format(suffix))
    plt.savefig(fig_folder + "mnist_number_of_afferents_boxplot{}.svg".format(suffix))
    if show_plots:
        plt.show()
    plt.close(fig)

    # firing rate per digit
    fig = plt.figure(figsize=(16, 8), dpi=600)

    # plt.axhline(s_max, color='#b2dd2c', ls=":")
    bp = plt.boxplot(instaneous_rates.T / Hz, notch=True, medianprops=medianprops)

    plt.xticks(np.arange(number_of_afferents.shape[0]) + 1, np.arange(number_of_afferents.shape[0]))
    plt.xlabel("Target layer")
    plt.ylabel("Firing rate (Hz)")
    plt.grid(True, which='major', axis='y')
    plt.savefig(fig_folder + "mnist_testing_firing_rate_boxplot{}.pdf".format(suffix))
    plt.savefig(fig_folder + "mnist_testing_firing_rate_boxplot{}.svg".format(suffix))
    if show_plots:
        plt.show()
    plt.close(fig)


def mnist_comparison(archive_1, archive_2, out_filename=None, extra_suffix=None,
                     custom_labels=None, show_plots=False):
    # TODO
    pass


if __name__ == "__main__":
    import sys

    # Case disambiguisation:
    #   1 - rewiring and     STDP
    #   2 - rewiring and     STDP, but no lateral connections
    #   3 - rewiring, but no STDP


    filename = "mnist_case_1_fixed_signal_20_sigma"
    mnist_analysis(filename, extra_suffix="fixed_signal_sigma")
    filename = "mnist_case_2_fixed_signal_20_sigma"
    mnist_analysis(filename, extra_suffix="fixed_signal_sigma")
    filename = "mnist_case_3_fixed_signal_20_sigma"
    mnist_analysis(filename, extra_suffix="fixed_signal_sigma")

    sys.exit()


    # Rate-based input experiments

    filename = "mnist_case_1_5hz_rate_smax_96_sigma_lat_2"
    mnist_analysis(filename)

    filename = "mnist_case_2_5hz_rate_smax_96"
    mnist_analysis(filename)

    filename = "mnist_case_3_5hz_rate_smax_96_sigma_lat_2"
    mnist_analysis(filename)

    # Centre Surround (Filtered) input experiments

    filename = "mnist_case_1_5hz_cs_on_off_smax_96_sigma_lat_2"
    mnist_analysis(filename)

    filename = "mnist_case_3_5hz_cs_on_off_smax_96_sigma_lat_2"
    mnist_analysis(filename)
    sys.exit()

    filename = "mnist_case_3_fixed_signal"
    mnist_analysis(filename, extra_suffix="fixed_signal")
    filename = "mnist_case_1_fixed_signal"
    mnist_analysis(filename, extra_suffix="fixed_signal")
    filename = "mnist_case_2_fixed_signal"
    mnist_analysis(filename, extra_suffix="fixed_signal")

    sys.exit()

    # filename = "mnist_case_1__600s"
    # mnist_analysis(filename, "600s")
    # sys.exit()
    #
    # filename = "mnist_case_1_fixed_signal"
    # mnist_analysis(filename, extra_suffix="fixed_signal_signal_10_fbase_2")
    # mnist_analysis(filename, extra_suffix="fixed_signal_signal_10")
    #
    # sys.exit()
    #

    filename = "mnist_case_3_300s_cont_master"
    mnist_analysis(filename, extra_suffix="pynn8_cspc")
    # sys.exit()
    filename = "mnist_case_1_300s_cont_master"
    mnist_analysis(filename, extra_suffix="pynn8_cspc")
    filename = "mnist_case_2_300s_cont_master"
    mnist_analysis(filename, extra_suffix="pynn8_cspc")

    sys.exit()

    filename = "mnist_case_3_cont_pynn8"
    mnist_analysis(filename, extra_suffix="pynn8")
    sys.exit()

    filename = "mnist_case_3_cont_pynn8_cspc"
    mnist_analysis(filename, extra_suffix="pynn8_cspc")
    sys.exit()

    filename = "mnist_case_1_300s_cont_fmean_5.npz"
    mnist_analysis(filename, extra_suffix="300s_continuous_test")

    filename = "mnist_case_1_300s_cont"
    mnist_analysis(filename, extra_suffix="300s_continuous_test")

    filename = "mnist_case_3_300s_cont"
    mnist_analysis(filename, extra_suffix="300s_continuous_test")

    filename = "mnist_case_3_600s_cont"
    mnist_analysis(filename, extra_suffix="600s_continuous_test")

    filename = "mnist_case_1_400s_cont"
    mnist_analysis(filename, extra_suffix="400s_continuous_test")
    # sys.exit()

    filename = "mnist_case_1_smax_128_cont"
    mnist_analysis(filename, extra_suffix="smax_128_continuous_test")

    filename = "mnist_case_1_cont"
    mnist_analysis(filename, extra_suffix="continuous_test")

    filename = "mnist_case_3_cont"
    mnist_analysis(filename, extra_suffix="continuous_test")

    # Rate-based input experiments

    filename = "mnist_case_1_5hz_rate_smax_96_sigma_lat_2"
    mnist_analysis(filename)

    filename = "mnist_case_2_5hz_rate_smax_96"
    mnist_analysis(filename)

    filename = "mnist_case_3_5hz_rate_smax_96_sigma_lat_2"
    mnist_analysis(filename)

    # Centre Surround (Filtered) input experiments

    filename = "mnist_case_1_5hz_cs_on_off_smax_96_sigma_lat_2"
    mnist_analysis(filename)

    filename = "mnist_case_3_5hz_cs_on_off_smax_96_sigma_lat_2"
    mnist_analysis(filename)
