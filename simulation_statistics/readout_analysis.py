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
from analysis_functions_definitions import *
import traceback
from sklearn.metrics import confusion_matrix, classification_report
import sklearn.metrics as metrics
import itertools
from argparser import *
import os

# ensure we use viridis as the default cmap
plt.viridis()

# ensure we use the same rc parameters for all matplotlib outputs
mlib.rcParams.update({'font.size': 24})
mlib.rcParams.update({'errorbar.capsize': 5})
mlib.rcParams.update({'figure.autolayout': True})

# some defaults
root_stats = "C:\Work\phd\simulation_statistics\\"
root_syn = "C:\Work\phd\synaptogenesis\\"
fig_folder = args.fig_folder

# check if the figures folder exist
if not os.path.isdir(fig_folder) and not os.path.exists(fig_folder):
    os.mkdir(fig_folder)


def class_assignment(spikes, classes, actual_classes, training_type,
                     simtime, chunk):
    # Compute rates and ranks
    # Winner-takes-all
    instaneous_rates = np.empty((classes.size, int((simtime / ms) // chunk)))
    for index, value in np.ndenumerate(instaneous_rates):
        number_index, chunk_index = index
        instaneous_rates[number_index, chunk_index] = np.count_nonzero(
            np.logical_and(
                spikes[spikes[:, 0] == number_index][:, 1] >= (chunk_index * chunk),
                spikes[spikes[:, 0] == number_index][:, 1] < ((chunk_index + 1) * chunk)
            )
        )
    what_network_thinks = np.empty(int((simtime / ms) // chunk))
    for i in range(what_network_thinks.shape[0]):
        #         what_network_thinks[i] = np.argmax(instaneous_rates[:, i])
        # random tie-breaking
        ir_max = np.max(instaneous_rates[:, i])
        what_network_thinks[i] = np.random.choice(np.flatnonzero(instaneous_rates[:, i] == ir_max))
    # Rank-order
    #     first_to_spike = np.ones(int((simtime/ms)//chunk))*0  # Change this to follow all entries that do not spike
    first_to_spike = np.random.randint(0, 2, size=int((simtime / ms) // chunk))
    for index, value in np.ndenumerate(first_to_spike):
        chunk_index = index[0]
        try:
            first_to_spike[chunk_index] = np.sort(spikes[
                                                      np.where(np.logical_and(
                                                          spikes[:, 1] >= (chunk_index * chunk),
                                                          spikes[:, 1] < ((chunk_index + 1) * chunk)
                                                      ))])[0, 0]
        except:
            print("No spikes", chunk_index)
            first_to_spike[chunk_index] = -1
            pass

    if training_type == "uns":
        all_class_permutations = list(itertools.permutations(classes))
        wta_likely_classes = []
        wta_max_acc = 0
        rank_order_likely_classes = []
        rank_order_max_acc = 0
        rmse_classes = []
        #     min_rmse = -1.
        min_rmse = 10000
        #     print "all class permutations", all_class_permutations
        for perm in all_class_permutations:
            perm = np.asarray(perm)
            acc_score = metrics.accuracy_score(actual_classes.ravel(), perm[what_network_thinks.astype(int)].ravel())

            if acc_score > wta_max_acc:
                print("wta_", acc_score)
                wta_max_acc = acc_score
                wta_likely_classes = perm

            acc_score = metrics.accuracy_score(actual_classes.ravel(), perm[first_to_spike.astype(int)].ravel())
            if acc_score > rank_order_max_acc:
                print("ro_", acc_score)
                rank_order_max_acc = acc_score
                rank_order_likely_classes = perm

            rmse = np.sqrt(
                np.mean(((actual_classes.ravel() - perm[first_to_spike.astype(int)].astype(float).ravel()) ** 2)))
            if rmse < min_rmse:
                print("rmse_", acc_score)
                min_rmse = acc_score
                rmse_classes = perm
        wta_predictions = what_network_thinks.astype(int).ravel()
        rank_order_predictions = first_to_spike.astype(int).ravel()
        return wta_predictions, rank_order_predictions, wta_likely_classes, rank_order_likely_classes, rmse_classes
    else:
        wta_predictions = what_network_thinks.astype(int).ravel()
        rank_order_predictions = first_to_spike.astype(int).ravel()
        return wta_predictions, rank_order_predictions, classes, classes, classes


def generate_readout_suffix(training_angles):
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
    return suffix_test


def plot_spikes(spikes, title, classes, filename, chunk=200, end_time=1800):
    if spikes is not None:
        recast_spikes = []
        for index, value in np.ndenumerate(classes):
            recast_spikes.append(spikes[spikes[:, 0] == index[0]][:, 1])
        fig, ax1 = plt.subplots(1, 1, figsize=(15, 6), dpi=600)
        ax1.set_xlim((0, end_time))
        ax1.eventplot(recast_spikes, linelengths=.8, colors=['#414C82'] * classes.size)
        ax1.set_xlabel('Time/ms')
        ax1.set_ylabel('Class neuron')
        ax1.set_title(title)
        ax1.set_yticks(np.arange(classes.size))
        ax1.set_yticklabels(np.sort(classes))
        # chunk separators
        bin_edges = (np.arange(int(end_time / chunk) - 1) + 1) * chunk
        for i in bin_edges:
            ax1.axvline(i, color='#B2DD2C')
        plt.savefig(fig_folder + filename + ".pdf", bbox_inches='tight')
        plt.savefig(fig_folder + filename + ".svg", bbox_inches='tight')
        return fig
    return None


def readout_neuron_analysis(fname, training_type="uns", extra_suffix="", show_plots=False):
    training_fname = "training_readout_for_" + training_type + "_" + fname + extra_suffix
    testing_fname = "testing_readout_for_" + training_type + "_" + fname + extra_suffix

    training_data = np.load(root_syn + training_fname + ".npz")
    testing_data = np.load(root_syn + testing_fname + ".npz")

    # Retreive data from testing data
    testing_target_spikes = testing_data['target_spikes']
    testing_inhibitory_spikes = testing_data['inhibitory_spikes']
    testing_readout_spikes = testing_data['readout_spikes']
    testing_actual_classes = testing_data['actual_classes'].ravel()
    testing_target_readout_projection = testing_data['target_readout_projection']


    readout_sim_params = testing_data['readout_sim_params'].ravel()[0]
    w_max = readout_sim_params['argparser']['w_max']
    simtime = testing_data['simtime'] * ms
    chunk = testing_data['chunk']

    is_rewiring_enable = False
    if 'rewiring' in training_data.files:
        is_rewiring_enable = training_data['rewiring']

    # Retreive data from training data
    training_actual_classes = training_data['actual_classes']
    training_readout_spikes = training_data['readout_spikes']
    target_readout_projection = training_data['target_readout_projection']
    wta_projection = training_data['wta_projection']
    training_sim_params = training_data['input_sim_params'].ravel()[0]

    original_delays_are_constant = training_sim_params['constant_delay']

    suffix_test = generate_readout_suffix(training_actual_classes)
    suffix_test += "_" + training_type
    if extra_suffix:
        suffix_test += extra_suffix
    if is_rewiring_enable:
        suffix_test += "_rewiring"
    if original_delays_are_constant:
        suffix_test += "_constant"
    print("="*45)
    print("{:45}".format("The suffix for this set of figures is "), ":",  suffix_test)
    print("{:45}".format("The training archive name is "), ":", training_fname)
    print("{:45}".format("The testing archive name is "), ":", testing_fname)

    target_readout_projection = target_readout_projection.reshape(target_readout_projection.size / 4, 4)
    wta_projection = wta_projection.reshape(wta_projection.size / 4, 4)
    classes = np.sort(np.unique(testing_actual_classes))

    training_data.close()
    testing_data.close()

    # assert np.all(testing_target_readout_projection == target_readout_projection), target_readout_projection

    fig = plot_spikes(training_readout_spikes, "Readout neuron spikes (training)",
                      np.unique(testing_actual_classes),
                      "readout_training_spikes{}".format(suffix_test))
    if show_plots:
        plt.show()
    plt.close(fig)

    try:
        fig = plot_spikes(testing_readout_spikes, "Readout neuron spikes (testing)", np.unique(testing_actual_classes),
                          "readout_testing_spikes{}".format(suffix_test))
        if show_plots:
            plt.show()
        plt.close(fig)
    except Exception as e:
        traceback.print_exc()

    spikes = training_readout_spikes
    print("Training ----------------")
    for index, value in np.ndenumerate(classes):
        print("Number of spikes for class", value, ":",
              spikes[spikes[:, 0] == index[0]].size, "equivalent of ", spikes[spikes[:, 0] == index[0]].size / simtime)

    spikes = testing_readout_spikes
    print("Testing ----------------")
    for index, value in np.ndenumerate(classes):
        print("Number of spikes for class", value, ":",
              spikes[spikes[:, 0] == index[0]].size, "equivalent of ", spikes[spikes[:, 0] == index[0]].size / simtime)

    wta_predictions, rank_order_predictions, wta_likely_classes, \
    rank_order_likely_classes, rmse_classes = class_assignment(
        testing_readout_spikes,
        classes=classes,
        actual_classes=testing_actual_classes,
        training_type=training_type,
        simtime=simtime,
        chunk=chunk)

    print(" WTA ---------------")
    print(classification_report(testing_actual_classes, wta_likely_classes[wta_predictions]))
    print(" WTA LIKELY CLASSES :", wta_likely_classes)
    print()
    print(" Rank order --------")
    print()
    print(classification_report(testing_actual_classes,
                                rank_order_likely_classes[rank_order_predictions]))
    print(" RANK ORDER LIKELY CLASSES :", rank_order_likely_classes)

    print(" RMSE Rank order --------")
    print()
    print(classification_report(testing_actual_classes,
                                rmse_classes[rank_order_predictions]))
    print(" RMSE LIKELY CLASSES :", rmse_classes)

    conns = []
    conns_names = []
    for index, value in np.ndenumerate(classes):
        conns.append(target_readout_projection[target_readout_projection[:, 1] == index[0]])
        conns_names.append("$readout_{%s}$" % str(value))
    fig, axes = plt.subplots(1, classes.size, figsize=(classes.size * 5, 7), dpi=800, sharey=True)

    minimus = 0
    maximus = 1
    for index, ax in np.ndenumerate(axes):
        i = index[0]
        ax.hist(conns[i][:, 2] / w_max, bins=20, color='#414C82', edgecolor='k')
        ax.set_title(conns_names[i])
        ax.set_xlim([minimus, maximus])
        print(np.max(conns[i][:, 2]))
        # assert np.max(conns[i][:, 2]) <= w_max
    plt.tight_layout()
    plt.savefig(fig_folder + "readout_weight_histograms{}.pdf".format(suffix_test), bbox_inches='tight')
    plt.savefig(fig_folder + "readout_weight_histograms{}.svg".format(suffix_test), bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close(fig)
    print("="*45,"\n\n")


if __name__ == "__main__":
    import sys
    # Attempting readout of constant delay network

    fname="random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_cont"
    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_NESW")
    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_NESW_80s")
    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_rewiring_NESW_rew_p_0")
    sys.exit()

    fname = "constant_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_cont"

    readout_neuron_analysis(fname, training_type="uns")
    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_rewiring_p_.0_smax_32_rew_wta")
    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_rewiring_p_.0_smax_32_20s_rew_wta")

    sys.exit()

    fname = "random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_cont"
    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_p_.2_b_1.1")  # perfect

    # sys.exit()

    fname = "random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_cont"
    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_rewiring_p_.1_b_1.2")
    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_rewiring_p_.05_b_1.2_smax_64")
    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_rewiring_p_.0_b_1.2_smax_32")
    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_rewiring_p_.0_smax_32")
    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_rewiring_p_.0_smax_32_80s")
    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_rewiring_p_.0_smax_32_frew_1000")
    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_rewiring_p_.0_smax_32_20s_rew_wta")
    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_rewiring_p_.0_smax_32_rew_wta")
    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_rewiring_p_.0_smax_32_80s_rew_wta")

    sys.exit()
    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_p_.2")
    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_30s")
    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_p_.2_b_1")
    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_p_.2_b_1.1")  # perfect
    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_p_.2_b_1.2")  # rewiring run

    # These simulations seem to have an issue with depressing connections too much

    sys.exit()
    fname = "random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_cont"
    readout_neuron_analysis(fname, training_type="uns")

    fname = "random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_cont"
    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_1")

    fname = "random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_cont"
    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_2")

    fname = "random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_cont"
    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_3")

    fname = "random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_cont"
    readout_neuron_analysis(fname, training_type="max", extra_suffix="")


    fname = "random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_cont"
    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_b_1")

    fname = "random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_cont"
    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_b_1.1")

    fname = "random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_cont"
    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_b_1.2")

    sys.exit()  # These simulations seem to have an issue with depressing connections too much


    # The following is the reference simulation
    fname = "random_delay_smax_128_gmax_1_384k_sigma_7.5_3_angle_0_90_evo"
    extra_suffix = "_rerun"
    readout_neuron_analysis(fname, training_type="uns", extra_suffix=extra_suffix)  # perfect

