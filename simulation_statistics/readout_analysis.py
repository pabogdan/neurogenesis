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
from collections import Iterable
import copy
from colorama import Fore, Back, Style
from scipy.spatial import distance

# ensure we use viridis as the default cmap
plt.viridis()
viridis_cmap = mlib.cm.get_cmap('viridis')

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


def _power_on_self_test(fname, training_type="uns", extra_suffix="", show_plots=False):
    # check shield_for_class_assignment vs. class_assignment
    # i.e this is the POST area
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
    print(Fore.RED + "=" * 45)
    print("{:45}".format("POST POST POST POST POST"))
    print("{:45}".format("The suffix for this set of figures is "), ":", suffix_test)
    print("{:45}".format("The training archive name is "), ":", training_fname)
    print("{:45}".format("The testing archive name is "), ":", testing_fname)

    target_readout_projection = target_readout_projection.reshape(target_readout_projection.size / 4, 4)
    wta_projection = wta_projection.reshape(wta_projection.size / 4, 4)
    classes = np.sort(np.unique(testing_actual_classes))

    training_data.close()
    testing_data.close()
    wta_predictions, rank_order_predictions, wta_likely_classes, \
    rank_order_likely_classes, rmse_classes = class_assignment(
        testing_readout_spikes,
        classes=classes,
        actual_classes=testing_actual_classes,
        training_type=training_type,
        simtime=simtime,
        chunk=chunk)

    print(" (VANILLA) WTA ---------------")
    print(classification_report(testing_actual_classes, wta_likely_classes[wta_predictions]))
    print(" (VANILLA) WTA LIKELY CLASSES :", wta_likely_classes)
    print(classification_report(testing_actual_classes,
                                rank_order_likely_classes[rank_order_predictions]))
    print(" (VANILLA) RANK ORDER LIKELY CLASSES :", rank_order_likely_classes)
    result_dict = shield_for_class_assignment(
        testing_readout_spikes,
        classes=classes,
        actual_classes=testing_actual_classes,
        training_type=training_type,
        simtime=simtime,
        chunk=chunk)
    print(" (SHIELD) WTA ---------------")
    print(classification_report(testing_actual_classes, result_dict['wta_predictions']))
    print(" (SHIELD) WTA LIKELY CLASSES :", result_dict['wta_likely_classes'])
    print(classification_report(testing_actual_classes,
                                result_dict['ro_predictions']))
    print(" (SHIELD) RANK ORDER LIKELY CLASSES :", result_dict['ro_likely_classes'])

    # Stop! Assert time!
    try:
        assert (np.all(result_dict['ro_likely_classes'] == rank_order_likely_classes))
        assert (np.all(result_dict['wta_likely_classes'] == wta_likely_classes))
        assert (np.all(result_dict['ro_predictions'] == rank_order_likely_classes[rank_order_predictions]))
        assert (np.all(result_dict['wta_predictions'] == wta_likely_classes[wta_predictions]))
    except:
        traceback.print_exc()
    finally:
        print("{:45}".format("DONE POST DONE POST DONE POST DONE POST"), Style.RESET_ALL)


def shield_for_class_assignment(spikes, classes, actual_classes, training_type,
                                simtime, chunk):
    (wta_predictions,
     rank_order_predictions,
     wta_likely_classes,
     rank_order_likely_classes,
     rmse_classes) = class_assignment(spikes, classes, actual_classes, training_type,
                                      simtime, chunk)

    corrected_wta_predictions = []
    for pred in wta_predictions:
        if pred != -1:
            corrected_wta_predictions.append(wta_likely_classes[pred])
        else:
            # the value is -1
            corrected_wta_predictions.append(pred)

    corrected_ro_predictions = []
    for pred in rank_order_predictions:
        if pred != -1:
            corrected_ro_predictions.append(rank_order_likely_classes[pred])
        else:
            # the value is -1
            corrected_ro_predictions.append(pred)

    corrected_wta_predictions = np.asarray(corrected_wta_predictions)
    corrected_ro_predictions = np.asarray(corrected_ro_predictions)

    result_dict = {}
    result_dict['wta_predictions'] = corrected_wta_predictions
    result_dict['wta_likely_classes'] = wta_likely_classes
    result_dict['ro_predictions'] = corrected_ro_predictions
    result_dict['ro_likely_classes'] = rank_order_likely_classes
    # classification reports
    result_dict['wta_classification_report'] = \
        classification_report(actual_classes, corrected_wta_predictions,
                              output_dict=True)
    result_dict['ro_classification_report'] = \
        classification_report(actual_classes, corrected_ro_predictions,
                              output_dict=True)
    # accuracy values
    result_dict['wta_classification_acc'] = \
        metrics.accuracy_score(actual_classes, corrected_wta_predictions)
    result_dict['ro_classification_acc'] = \
        metrics.accuracy_score(actual_classes, corrected_ro_predictions)

    # precision values
    result_dict['wta_classification_precision'] = \
        metrics.precision_score(actual_classes, corrected_wta_predictions,
                                average='weighted')
    result_dict['ro_classification_precision'] = \
        metrics.precision_score(actual_classes, corrected_ro_predictions,
                                average='weighted')

    # recall values
    result_dict['wta_classification_recall'] = \
        metrics.recall_score(actual_classes, corrected_wta_predictions,
                             average='weighted')
    result_dict['ro_classification_recall'] = \
        metrics.recall_score(actual_classes, corrected_ro_predictions,
                             average='weighted')

    # F1 / F-score / F-measure values
    result_dict['wta_classification_f1'] = \
        metrics.f1_score(actual_classes, corrected_wta_predictions,
                         average='weighted')
    result_dict['ro_classification_f1'] = \
        metrics.f1_score(actual_classes, corrected_ro_predictions,
                         average='weighted')
    # precision_recall_fscore_support
    result_dict['wta_classification_precision_recall_fscore_support'] = \
        metrics.precision_recall_fscore_support(
            actual_classes, corrected_wta_predictions,
            average='weighted')
    result_dict['ro_classification_precision_recall_fscore_support'] = \
        metrics.precision_recall_fscore_support(
            actual_classes, corrected_ro_predictions,
            average='weighted')

    # Count instances with no spikes
    result_dict['wta_count_no_spikes'] = np.count_nonzero(
        corrected_wta_predictions == -1)
    result_dict['ro_count_no_spikes'] = np.count_nonzero(
        corrected_ro_predictions == -1)

    return result_dict


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
    instaneous_rates = instaneous_rates.astype(int)
    what_network_thinks = np.empty(int((simtime / ms) // chunk))
    for i in range(what_network_thinks.shape[0]):
        #         what_network_thinks[i] = np.argmax(instaneous_rates[:, i])
        # random tie-breaking
        if np.all(instaneous_rates[:, i] == 0):
            what_network_thinks[i] = -1
        else:
            ir_max = np.max(instaneous_rates[:, i])
            what_network_thinks[i] = np.random.choice(np.flatnonzero(instaneous_rates[:, i] == ir_max))
    # Rank-order
    #     first_to_spike = np.ones(int((simtime/ms)//chunk))*0  # Change this to follow all entries that do not spike
    inst_means_rates = np.mean(instaneous_rates, axis=1)
    inst_means_std = np.std(instaneous_rates, axis=1)
    first_to_spike = np.random.randint(0, 2, size=int((simtime / ms) // chunk))
    for index, value in np.ndenumerate(first_to_spike):
        chunk_index = index[0]
        # order the spike times in the current bin
        sorted_spikes = np.sort(
            spikes[np.where(np.logical_and(
                spikes[:, 1] >= (chunk_index * chunk),
                spikes[:, 1] < ((chunk_index + 1) * chunk)
            ))])
        if sorted_spikes.size > 0:
            # select the source of the spike occurring at the lowest time
            # i.e. the first spike in the bin
            first_to_spike[chunk_index] = sorted_spikes[0, 0]
        else:
            # print("No spikes", chunk_index)
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
            wta_perm = what_network_thinks.astype(int).ravel()
            construct_wta_responses = []
            for f in wta_perm:
                if f != -1:
                    construct_wta_responses.append(perm[f])
                else:
                    construct_wta_responses.append(f)
            construct_wta_responses = np.asarray(construct_wta_responses)
            acc_score = metrics.f1_score(actual_classes.ravel(), construct_wta_responses,
                                         average='weighted')

            if acc_score > wta_max_acc:
                # print("wta_", acc_score)
                wta_max_acc = copy.deepcopy(acc_score)
                wta_likely_classes = np.copy(perm)

            first_perm = first_to_spike.astype(int).ravel()
            construct_first_responses = []
            for f in first_perm:
                if f != -1:
                    construct_first_responses.append(perm[f])
                else:
                    construct_first_responses.append(f)
            construct_first_responses = np.asarray(construct_first_responses)
            acc_score = metrics.f1_score(actual_classes.ravel(), construct_first_responses,
                                         average='weighted')
            if acc_score > rank_order_max_acc:
                # print("ro_", acc_score)
                rank_order_max_acc = copy.deepcopy(acc_score)
                rank_order_likely_classes = np.copy(perm)

            rmse = np.sqrt(
                np.mean(((actual_classes.ravel() - construct_first_responses.astype(float).ravel()) ** 2)))
            if rmse < min_rmse:
                # print("rmse_", acc_score)
                min_rmse = rmse
                rmse_classes = np.copy(perm)
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
        ax1.set_yticklabels(classes)
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
    argparser_info = readout_sim_params['argparser']

    if len(target_readout_projection.shape) == 3 or len(target_readout_projection.shape) == 1:
        target_readout_projection = target_readout_projection[-1]
    if len(wta_projection.shape) == 3 or len(wta_projection.shape) == 1:
        wta_projection = wta_projection[-1]
    original_delays_are_constant = training_sim_params['constant_delay']

    suffix_test = generate_readout_suffix(training_actual_classes)
    suffix_test += "_" + training_type
    if extra_suffix:
        suffix_test += extra_suffix
    if is_rewiring_enable:
        suffix_test += "_rewiring"
    if original_delays_are_constant:
        suffix_test += "_constant"
    print("=" * 45)
    print("{:45}".format("The suffix for this set of figures is "), ":", suffix_test)
    print("{:45}".format("The training archive name is "), ":", training_fname)
    print("{:45}".format("The testing archive name is "), ":", testing_fname)
    print("{:45}".format("Argparser Parameters of Interest"))
    print("{:45}".format("s_max"), ":", argparser_info['s_max'])
    print("{:45}".format("rewiring"), ":", argparser_info['rewiring'])
    print("{:45}".format("p_connect"), ":", argparser_info['p_connect'])
    print("{:45}".format("w_max"), ":", argparser_info['w_max'])

    # target_readout_projection = target_readout_projection.reshape(target_readout_projection.size / 4, 4)
    # wta_projection = wta_projection.reshape(wta_projection.size / 4, 4)
    classes = np.sort(np.unique(testing_actual_classes))

    training_data.close()
    testing_data.close()

    # assert np.all(testing_target_readout_projection == target_readout_projection), target_readout_projection

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
    print("{:45}".format("WTA LIKELY CLASSES"), ":", wta_likely_classes)
    print()
    print(" Rank order --------")
    print()
    print(classification_report(testing_actual_classes,
                                rank_order_likely_classes[rank_order_predictions]))
    print("{:45}".format("RANK ORDER LIKELY CLASSES"), ":", rank_order_likely_classes)

    print(" RMSE Rank order --------")
    print()
    print(classification_report(testing_actual_classes,
                                rmse_classes[rank_order_predictions]))
    print("{:45}".format("RMSE LIKELY CLASSES"), ":", rmse_classes)

    print("{:45}".format("Connectivity stats"))

    fig = plot_spikes(training_readout_spikes, "Readout neuron spikes (training)",
                      rank_order_likely_classes,
                      "readout_training_spikes{}".format(suffix_test))
    if show_plots:
        plt.show()
    plt.close(fig)

    try:
        fig = plot_spikes(testing_readout_spikes, "Readout neuron spikes (testing)", rank_order_likely_classes,
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
    print("{:45}".format("First 10 testing classes"), ":", testing_actual_classes[:10])

    conns = []
    conns_names = []
    for index, value in np.ndenumerate(classes):
        conns.append(target_readout_projection[target_readout_projection[:, 1] == index[0]][:, 2])
        label = "$readout_{%s}$" % str(value)
        conns_names.append(label)
        print("{:25}".format(label), "{:20}".format("# total"), ":", conns[index[0]].size)
        print("{:25}".format(label), "{:20}".format("# potentiated"), ":", np.count_nonzero(conns[index[0]] >= (w_max / 2)))
        print("{:25}".format(label), "{:20}".format("# depressed"), ":", np.count_nonzero(conns[index[0]] < (w_max / 2)))
        print("-" * 52)
    fig, axes = plt.subplots(1, classes.size, figsize=(classes.size * 5, 7), dpi=800, sharey=True)

    minimus = 0
    maximus = 1
    for index, ax in np.ndenumerate(axes):
        i = index[0]
        ax.hist(conns[i] / w_max, bins=20, color='#414C82', edgecolor='k')
        ax.set_title(conns_names[i])
        ax.set_xlim([minimus, maximus])
        print(np.max(conns[i]))
    plt.tight_layout()
    plt.savefig(fig_folder + "readout_weight_histograms{}.pdf".format(suffix_test), bbox_inches='tight')
    plt.savefig(fig_folder + "readout_weight_histograms{}.svg".format(suffix_test), bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close(fig)

    if wta_projection.size > 0:
        conns = []
        conns_names = []
        for index, value in np.ndenumerate(classes):
            conns.append(wta_projection[wta_projection[:, 1] == index[0]][:, 2])
            label = "$lat_{%s}$" % str(value)
            conns_names.append(label)
            print("{:25}".format(label), "{:20}".format("# total"), ":", conns[index[0]].size)
            print("{:25}".format(label), "{:20}".format("# potentiated"), ":", np.count_nonzero(conns[index[0]] >= (w_max / 2)))
            print("{:25}".format(label), "{:20}".format("# depressed"), ":", np.count_nonzero(conns[index[0]] < (w_max / 2)))
            print("-" * 52)
        fig, axes = plt.subplots(1, classes.size, figsize=(classes.size * 5, 7), dpi=800, sharey=True)

        minimus = 0
        maximus = 1
        for index, ax in np.ndenumerate(axes):
            i = index[0]
            ax.hist(conns[i] / w_max, bins=20, color='#414C82', edgecolor='k')
            ax.set_title(conns_names[i])
            # ax.set_xlim([minimus, maximus])
            # print(np.max(conns[i]))
        plt.tight_layout()
        plt.savefig(fig_folder + "readout_lat_weight_histograms{}.pdf".format(suffix_test), bbox_inches='tight')
        plt.savefig(fig_folder + "readout_lat_weight_histograms{}.svg".format(suffix_test), bbox_inches='tight')
        if show_plots:
            plt.show()
        plt.close(fig)
    print("=" * 45, "\n\n")


def analyse_multiple_runs(fname, runs, training_type="uns", extra_suffix="",
                          preproc_archive=None, show_plots=False,
                          testing_suffix=""):
    if isinstance(runs, Iterable):
        run_nos = np.sort(np.asarray(runs))
    else:
        run_nos = np.arange(runs)
    number_of_runs = run_nos.size
    # Create structure to hold the results for each simulation and their snapshots

    # a dictionary whose keys are the run being analysed
    # the values of these dicts should also be {} based on snapshots
    wta_predicted_classes = {}
    wta_predictions = {}
    ro_predicted_classes = {}
    ro_predictions = {}
    classes_snapshots = {}
    readout_spikes = {}
    readout_connectivity = {}
    inter_readout_connectivity = {}
    results_dict = {}
    plot_width = 8
    training_snapshots = {}

    weights_per_run = {}
    # Iterate over simulations and the snapshots in testing archives
    for run in run_nos:
        training_fname = "training_readout_for_" + training_type + "_" + fname + "_run_" + str(run) + extra_suffix
        testing_fname = "testing_readout_for_" + training_type + "_" + fname + "_run_" + str(run) + testing_suffix + extra_suffix
        training_data = np.load(root_syn + training_fname + ".npz")
        testing_data = np.load(root_syn + testing_fname + ".npz")
        print("{:45}".format("The training archive name is "), ":", training_fname)
        print("{:45}".format("The testing archive name is "), ":", testing_fname)

        # Retreive data from testing data
        testing_readout_spikes = testing_data['readout_spikes']
        testing_actual_classes = testing_data['actual_classes'].ravel()
        testing_target_readout_projection = testing_data['target_readout_projection']
        snapshots_present = training_data['snapshots_present'].ravel()[0]
        target_snapshots = training_data['target_snapshots'].ravel()[0]
        actual_classes_snapshots = testing_data['actual_classes_snapshots'].ravel()[0]
        wta_snapshots = training_data['wta_snapshots'].ravel()[0]

        readout_spikes_snapshots = testing_data['readout_spikes_snapshots'].ravel()[0]

        # add a list to which to add normalised weights
        # weights_per_run.append([])

        readout_sim_params = testing_data['readout_sim_params'].ravel()[0]
        training_sim_params = training_data['readout_sim_params'].ravel()[0]
        w_max = readout_sim_params['argparser']['w_max']
        t_record = readout_sim_params['argparser']['t_record']
        simtime = testing_data['simtime'] * ms
        chunk = testing_data['chunk']

        is_rewiring_enable = False
        if 'rewiring' in training_data.files:
            is_rewiring_enable = training_data['rewiring']

        # Retreive data from training data
        training_actual_classes = training_data['actual_classes']
        # training_readout_spikes = training_data['readout_spikes']
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
        suffix_test += testing_suffix
        # target_readout_projection = target_readout_projection.reshape(target_readout_projection.size / 4, 4)
        # wta_projection = wta_projection.reshape(wta_projection.size / 4, 4)
        classes = np.sort(np.unique(testing_actual_classes))

        training_data.close()
        testing_data.close()
        # run-local (rl) parameters (To be copied into the global dicts)

        rl_wta_predicted_classes = {}
        rl_wta_predictions = {}
        rl_ro_predicted_classes = {}
        rl_ro_predictions = {}
        rl_actual_classes_snapshots = {}
        rl_readout_spikes = {}
        rl_readout_connecitvity = {}
        rl_inter_readout_connectivity = {}
        rl_weights_per_run = {}
        rl_corrected_results_dict = {}
        rl_snapshots = {}
        # difference of firing rate (fr, Hz) for the 2 classes
        rl_fr_mean_std = {}
        # difference of firing time for first spikes (ft, ms) for the 2 classes
        rl_delta_ft = {}
        # loop over snapshots
        ordered_snapshots = np.sort(np.asarray(readout_spikes_snapshots.keys()))
        for snap_keys in ordered_snapshots:
            rl_readout_spikes[snap_keys] = np.copy(readout_spikes_snapshots[snap_keys])
            rl_readout_connecitvity[snap_keys] = np.copy(target_snapshots[snap_keys])
            rl_inter_readout_connectivity[snap_keys] = np.copy(wta_snapshots[snap_keys])
            # read actual classes
            rl_actual_classes_snapshots[snap_keys] = np.copy(actual_classes_snapshots[snap_keys])
            rl_weights_per_run[snap_keys] = target_snapshots[snap_keys][:, 2] / float(w_max)

            rl_corrected_results_dict[snap_keys] = shield_for_class_assignment(
                rl_readout_spikes[snap_keys],
                classes=classes,
                actual_classes=rl_actual_classes_snapshots[snap_keys],
                training_type=training_type,
                simtime=simtime,
                chunk=chunk)
            # extract classes and class predictions
            _wta_predictions = rl_corrected_results_dict[snap_keys]['wta_predictions']
            _rank_order_predictions = rl_corrected_results_dict[snap_keys]['ro_predictions']
            _wta_likely_classes = rl_corrected_results_dict[snap_keys]['wta_likely_classes']
            _rank_order_likely_classes = rl_corrected_results_dict[snap_keys]['ro_likely_classes']
            # extract classification reports
            _wta_acc = rl_corrected_results_dict[snap_keys]['wta_classification_acc']
            _ro_acc = rl_corrected_results_dict[snap_keys]['ro_classification_acc']

            rl_wta_predicted_classes[snap_keys] = np.copy(_wta_likely_classes)
            rl_wta_predictions[snap_keys] = np.copy(_wta_predictions)
            rl_ro_predicted_classes[snap_keys] = np.copy(_rank_order_likely_classes)
            rl_ro_predictions[snap_keys] = np.copy(_rank_order_predictions)

            print(Fore.GREEN + "{:45}".format("Run {} snap {}".format(run, snap_keys)), Style.RESET_ALL)
            print("{:45}".format("WTA Predicted classes"), ":", rl_wta_predicted_classes[snap_keys])
            print("{:45}".format("WTA Accuracy"), ":", _wta_acc)
            print("{:45}".format("WTA classification report"), ":",
                  rl_corrected_results_dict[snap_keys]['wta_classification_report'])
            print("{:45}".format("WTA # of 0 spikes"), ":", rl_corrected_results_dict[snap_keys]['wta_count_no_spikes'])
            print("{:45}".format("RO Predicted classes"), ":", rl_ro_predicted_classes[snap_keys])
            print("{:45}".format("RO Accuracy"), ":", _ro_acc)
            print("{:45}".format("RO classification report"), ":",
                  rl_corrected_results_dict[snap_keys]['ro_classification_report'])
            print("{:45}".format("RO # of 0 spikes"), ":", rl_corrected_results_dict[snap_keys]['ro_count_no_spikes'])
            print()

        # store all of this information in the global dicts
        wta_predicted_classes[run] = copy.deepcopy(rl_wta_predicted_classes)
        wta_predictions[run] = copy.deepcopy(rl_wta_predictions)
        ro_predicted_classes[run] = copy.deepcopy(rl_ro_predicted_classes)
        ro_predictions[run] = copy.deepcopy(rl_ro_predictions)
        classes_snapshots[run] = copy.deepcopy(rl_actual_classes_snapshots)
        readout_spikes[run] = copy.deepcopy(rl_readout_spikes)
        readout_connectivity[run] = copy.deepcopy(rl_readout_connecitvity)
        inter_readout_connectivity[run] = copy.deepcopy(rl_inter_readout_connectivity)
        results_dict[run] = copy.deepcopy(rl_corrected_results_dict)
        weights_per_run[run] = copy.deepcopy(rl_weights_per_run)
        training_snapshots[run] = np.copy(ordered_snapshots)

    # plot the average weight histogram (average over runs for the same snap)
    ordered_snapshots = training_snapshots[run_nos[0]]
    wta_accuracies = np.empty((number_of_runs, ordered_snapshots.size))
    ro_accuracies = np.empty((number_of_runs, ordered_snapshots.size))
    wta_precisions = np.empty((number_of_runs, ordered_snapshots.size))
    ro_precisions = np.empty((number_of_runs, ordered_snapshots.size))
    wta_fscores = np.empty((number_of_runs, ordered_snapshots.size))
    ro_fscores = np.empty((number_of_runs, ordered_snapshots.size))
    ro_no_spikes = np.empty((number_of_runs, ordered_snapshots.size))
    wta_no_spikes = np.empty((number_of_runs, ordered_snapshots.size))
    afferents = np.zeros((number_of_runs, ordered_snapshots.size))
    ff_weights = []
    for _ in ordered_snapshots:
        ff_weights.append([])
    for run, run_no in np.ndenumerate(run_nos):
        run = run[0]
        for index, snap_keys in np.ndenumerate(ordered_snapshots):
            i = int(index[0])
            # accuracy
            wta_accuracies[run, i] = results_dict[run_no][snap_keys]['wta_classification_acc']
            ro_accuracies[run, i] = results_dict[run_no][snap_keys]['ro_classification_acc']
            # precision
            wta_precisions[run, i] = results_dict[run_no][snap_keys]['wta_classification_precision']
            ro_precisions[run, i] = results_dict[run_no][snap_keys]['ro_classification_precision']
            # F1 score
            wta_fscores[run, i] = results_dict[run_no][snap_keys]['wta_classification_f1']
            ro_fscores[run, i] = results_dict[run_no][snap_keys]['ro_classification_f1']
            # Count of bins containing no spikes
            ro_no_spikes[run, i] = results_dict[run_no][snap_keys]['ro_count_no_spikes']
            wta_no_spikes[run, i] = results_dict[run_no][snap_keys]['wta_count_no_spikes']
            ff_weights[i] += weights_per_run[run_no][snap_keys].ravel().tolist()
            # count number of connections
            afferents[run, i] += len(weights_per_run[run_no][snap_keys].ravel().tolist())
            assert (ro_no_spikes[run, i] == wta_no_spikes[run, i])
    # normalise afferents
    afferents = afferents / float(classes.size)
    metrics_to_plot = [wta_accuracies, wta_precisions, wta_fscores,
                       ro_accuracies, ro_precisions, ro_fscores]
    metrics_names = ["wta_acc", "wta_prec", "wta_fscore",
                     "ro_acc", "ro_prec", "ro_fscore"]

    for metric, metric_name in zip(metrics_to_plot, metrics_names):
        print("{:45}".format("Ploting metric"), ":", metric_name)
        fig, ax = plt.subplots(figsize=(plot_width, 8), dpi=600)
        for run in np.arange(number_of_runs):
            cmap_i = (run + 1) / float(number_of_runs)
            current_color = viridis_cmap(cmap_i)
            ax.plot((ordered_snapshots) / 1000, metric[run, :],
                    c=current_color, alpha=.7)
        ax.errorbar((ordered_snapshots) / 1000, np.mean(metric, axis=0),
                    yerr=stats.sem(metric, axis=0),
                    c='k', alpha=.7)
        plt.xlabel("Training time (seconds)")
        plt.ylabel("%", rotation=0)
        plt.ylim([-.05, 1.05])
        plt.grid(True, which='major', axis='y')
        plt.savefig(fig_folder + "readout_{}_evo{}.pdf".format(metric_name, suffix_test))
        plt.savefig(fig_folder + "readout_{}_evo{}.svg".format(metric_name, suffix_test))
        if show_plots:
            plt.show()
        plt.close(fig)

    # plot the evolution of weights as boxplot
    # stlye the median of boxplots
    medianprops = dict(color='#414C82', linewidth=1.5)
    fig = plt.figure(figsize=(plot_width, 8), dpi=600)
    # plt.axhline(1. / len(classes), color='#b2dd2c', ls=":")

    bp = plt.boxplot(wta_accuracies, notch=True, medianprops=medianprops)

    _, labels = plt.xticks(np.arange(ordered_snapshots.shape[0]) + 1, (ordered_snapshots + t_record) / 1000)
    every_nth = 4
    for n, label in enumerate(labels):
        if n % every_nth != 0:
            label.set_visible(False)
    plt.xlabel("Training time (seconds)")
    plt.ylabel("Accuracy")
    # plt.ylim([-.05, 1.05])
    plt.grid(True, which='major', axis='y')
    plt.savefig(fig_folder + "readout_wta_accuracy_boxplot{}.pdf".format(suffix_test))
    plt.savefig(fig_folder + "readout_wta_accuracy_boxplot{}.svg".format(suffix_test))
    if show_plots:
        plt.show()
    plt.close(fig)

    fig = plt.figure(figsize=(plot_width, 8), dpi=600)
    # plt.axhline(1. / len(classes), color='#b2dd2c', ls=":")

    bp = plt.boxplot(ro_accuracies, notch=True, medianprops=medianprops)

    _, labels = plt.xticks(np.arange(ordered_snapshots.shape[0]) + 1, (ordered_snapshots + t_record) / 1000)
    every_nth = 4
    for n, label in enumerate(labels):
        if n % every_nth != 0:
            label.set_visible(False)
    plt.xlabel("Training time (seconds)")
    plt.ylabel("Accuracy")
    # plt.ylim([-.05, 1.05])
    plt.grid(True, which='major', axis='y')
    plt.savefig(fig_folder + "readout_ro_accuracy_boxplot{}.pdf".format(suffix_test))
    plt.savefig(fig_folder + "readout_ro_accuracy_boxplot{}.svg".format(suffix_test))
    if show_plots:
        plt.show()
    plt.close(fig)

    # if not is_rewiring_enable:
    # np_ff_weights = np.asarray(ff_weights)
    fig = plt.figure(figsize=(plot_width, 8), dpi=600)
    bp = plt.boxplot(ff_weights, notch=True, medianprops=medianprops)

    _, labels = plt.xticks(np.arange(ordered_snapshots.shape[0]) + 1, (ordered_snapshots + t_record) / 1000)
    every_nth = 4
    for n, label in enumerate(labels):
        if n % every_nth != 0:
            label.set_visible(False)
    plt.xlabel("Training time (seconds)")
    plt.ylabel(r"$\frac{g}{g_{max}}$", rotation="horizontal")
    plt.ylim([-.05, 1.05])
    plt.grid(True, which='major', axis='y')
    plt.savefig(fig_folder + "readout_weight_boxplot_evo{}.pdf".format(suffix_test))
    plt.savefig(fig_folder + "readout_weight_boxplot_evo{}.svg".format(suffix_test))
    if show_plots:
        plt.show()
    plt.close(fig)

    fig = plt.figure(figsize=(plot_width, 8), dpi=600)
    bp = plt.boxplot(afferents, notch=True, medianprops=medianprops)

    _, labels = plt.xticks(np.arange(ordered_snapshots.shape[0]) + 1, (ordered_snapshots + t_record) / 1000)
    every_nth = 4
    for n, label in enumerate(labels):
        if n % every_nth != 0:
            label.set_visible(False)
    plt.xlabel("Training time (seconds)")
    plt.ylabel("Number of afferents")
    plt.grid(True, which='major', axis='y')
    plt.savefig(fig_folder + "readout_no_afferents_boxplot_evo{}.pdf".format(suffix_test))
    plt.savefig(fig_folder + "readout_no_afferents_boxplot_evo{}.svg".format(suffix_test))
    if show_plots:
        plt.show()
    plt.close(fig)

    fig = plt.figure(figsize=(plot_width, 8), dpi=600)
    # bp = plt.boxplot(ro_no_spikes, notch=True, medianprops=medianprops)
    plt.errorbar((ordered_snapshots + t_record) / 1000,
                 np.mean(ro_no_spikes, axis=0),
                 yerr=np.std(ro_no_spikes, axis=0),
                 color=viridis_cmap(0.25))
    # plt.xticks(np.arange(ordered_snapshots.shape[0]) + 1, (ordered_snapshots + t_record) / 1000)
    plt.xlabel("Training time (seconds)")
    plt.ylabel("# of bins with no spikes")
    # plt.ylim([-.05, 1.05])
    plt.grid(True, which='major', axis='y')
    plt.savefig(fig_folder + "readout_no_spikes_evo{}.pdf".format(suffix_test))
    plt.savefig(fig_folder + "readout_no_spikes_evo{}.svg".format(suffix_test))
    if show_plots:
        plt.show()
    plt.close(fig)

    # TODO for each snapshot point plot the accuracy, recall and STD (2 plots)

    # TODO preproc archive -- what is the DSI and entropy of the target neurons
    # which still have potentiated synapses with the readout neurons

    if preproc_archive:
        preproc_path = os.path.join(root_stats, "preproc", preproc_archive)
        preproc_data = np.load(preproc_path + ".npz")
        selective_dsi = np.asarray(preproc_data['dsi_selective'])
        not_selective_dsi = np.asarray(preproc_data['dsi_not_selective'])
        concatenated_dsis = get_concatenated_dsis(selective_dsi, not_selective_dsi, order_by_id=True)
        angles = preproc_data['angles']
        target_grid = preproc_data['sim_params'].ravel()[0]['grid']
        target_n_layer = target_grid[0] * target_grid[1]

        exc_entropy = np.asarray(preproc_data['exc_entropy'])
        max_entropy = get_max_entropy(angles)
        dsi_thresh = 0.5

        # Look at the sources for the top and bottom half of weights
        # what is the average DSI for them?
        top_dsi = np.empty((number_of_runs, ordered_snapshots.size))
        bottom_dsi = np.empty((number_of_runs, ordered_snapshots.size))
        top_entropy = np.empty((number_of_runs, ordered_snapshots.size))
        bottom_entropy = np.empty((number_of_runs, ordered_snapshots.size))
        dsi_class_distance = np.empty((number_of_runs, ordered_snapshots.size))
        per_class_top_dsi = np.empty((classes.size, number_of_runs, ordered_snapshots.size))
        for run, run_no in np.ndenumerate(run_nos):
            run = run[0]
            for index, snap_keys in np.ndenumerate(ordered_snapshots):
                i = int(index[0])
                curr_connectivity = readout_connectivity[run_no][snap_keys]
                _top = curr_connectivity[curr_connectivity[:, 2] >= (w_max / 2.)]
                _bot = curr_connectivity[curr_connectivity[:, 2] < (w_max / 2.)]
                _avg_top_dsi = np.mean(concatenated_dsis[_top[:, 0].astype(int)])
                _avg_bot_dsi = np.mean(concatenated_dsis[_bot[:, 0].astype(int)])
                _avg_top_ent = np.mean(exc_entropy[_top[:, 0].astype(int)])
                _avg_bot_ent = np.mean(exc_entropy[_bot[:, 0].astype(int)])
                top_dsi[run, i] = _avg_top_dsi
                bottom_dsi[run, i] = _avg_bot_dsi
                top_entropy[run, i] = _avg_top_ent
                bottom_entropy[run, i] = _avg_bot_ent

                for c in np.arange(classes.size):
                    _class_top = _top[_top[:, 1] == c]
                    per_class_top_dsi[c, run, i] = np.mean(concatenated_dsis[_class_top[:, 0].astype(int)])

                # for each readout neuron create a weight vector for all
                # possible pre-synaptic neurons
                # -- each entry in the vector represents the proportion of the maximum weight
                # this value can exceed 1.0 if there exists strong multapses between the pair
                # of neurons
                # TODO
                weight_vectors = np.zeros((classes.size, target_n_layer))
                angle_vector = np.ones(target_n_layer) * np.nan

                selective_ids = selective_dsi[:, 0].astype(int)

                # compute the histogram of DSI-selected angle
                for pre_neuron_id in range(target_n_layer):
                    loc = np.argwhere(selective_ids == pre_neuron_id)
                    if loc.size > 0:
                        angle_vector[pre_neuron_id] = selective_dsi[loc, 1]

                for readout_neuron_id in range(classes.size):
                    pre_ids = curr_connectivity[curr_connectivity[:, 1] == readout_neuron_id][:, 0].astype(int)
                    post_weights = curr_connectivity[curr_connectivity[:, 1] == readout_neuron_id][:, 2]
                    for each_index in np.arange(pre_ids.size):
                        weight_vectors[readout_neuron_id, pre_ids[each_index]] += \
                            post_weights[each_index]

                # normalising weight matrix
                weight_vectors /= w_max
                weight_vectors[np.where(np.isclose(weight_vectors, 0))] = np.nan

                # multiply weight vectors with histogram
                unique_angles = np.unique(angle_vector[np.isfinite(angle_vector)]).astype(int)
                bins = {}
                classes_based_on_dsi_and_weights = np.empty(classes.size)
                # TODO argmax selects class for that readout neuron
                for class_id in np.arange(classes.size):
                    bins[class_id] = {}
                    _max_val = -1
                    _max_ang = -1
                    for ua in unique_angles:
                        ua_position = np.argwhere(angle_vector == ua)
                        bins[class_id][ua] = np.sum(np.nan_to_num(weight_vectors[class_id, ua_position]))
                        if bins[class_id][ua] > _max_val:
                            _max_val = bins[class_id][ua]
                            _max_ang = ua
                    classes_based_on_dsi_and_weights[class_id] = _max_ang
                # print("{:45}".format("DSI + weight class"), ":", classes_based_on_dsi_and_weights.astype(int))
                # print("{:45}".format("RO class"), ":", ro_predicted_classes[run_no][snap_keys])

                dsi_class_distance[run, i] = scipy.spatial.distance.hamming(
                    ro_predicted_classes[run_no][snap_keys],
                    classes_based_on_dsi_and_weights.astype(int)
                )
        print("{:45}".format("Ploting metric"), ":", "Hamming Distance")
        fig, ax = plt.subplots(figsize=(plot_width, 8), dpi=600)
        for run in np.arange(number_of_runs):
            cmap_i = (run + 1) / float(number_of_runs)
            current_color = viridis_cmap(cmap_i)
            ax.plot((ordered_snapshots) / 1000, dsi_class_distance[run, :],
                    c=current_color, alpha=.7)
        # ax.errorbar((ordered_snapshots) / 1000, np.mean(dsi_class_distance, axis=0),
        #             yerr=np.std(dsi_class_distance, axis=0),
        #             c='k', alpha=.8)
        plt.xlabel("Training time (seconds)")
        plt.ylabel("Average DSI")
        # plt.legend(loc='best')
        # plt.ylim([-.05, 1.05])
        plt.grid(True, which='major', axis='y')
        plt.savefig(fig_folder + "readout_hamming_dist_evo{}.pdf".format(suffix_test))
        plt.savefig(fig_folder + "readout_hamming_dist_evo{}.svg".format(suffix_test))
        if show_plots:
            plt.show()
        plt.close(fig)

        print("{:45}".format("Ploting metric"), ":", "Bottom and top weight avg DSI")
        fig, ax = plt.subplots(figsize=(plot_width, 8), dpi=600)

        ax.errorbar((ordered_snapshots) / 1000, np.mean(bottom_dsi, axis=0),
                    yerr=np.std(bottom_dsi, axis=0),
                    c=viridis_cmap(.25), alpha=.8, label=r"$w<\theta_{g}$")
        ax.errorbar((ordered_snapshots) / 1000, np.mean(top_dsi, axis=0),
                    yerr=np.std(top_dsi, axis=0),
                    c=viridis_cmap(.75), alpha=.8, label=r"$w\geq\theta_{g}$")
        ax.axhline(dsi_thresh, color='#b2dd2c', ls=":")
        plt.xlabel("Training time (seconds)")
        plt.ylabel("Average DSI")
        plt.legend(loc='best')
        plt.ylim([-.05, 1.05])
        plt.grid(True, which='major', axis='y')
        plt.savefig(fig_folder + "readout_average_dsi_evo{}.pdf".format(suffix_test))
        plt.savefig(fig_folder + "readout_average_dsi_evo{}.svg".format(suffix_test))
        if show_plots:
            plt.show()
        plt.close(fig)

        print("{:45}".format("Ploting metric"), ":", "Bottom and top weight avg Entropy")
        fig, ax = plt.subplots(figsize=(plot_width, 8), dpi=600)

        ax.errorbar((ordered_snapshots) / 1000,
                    np.mean(bottom_entropy, axis=0),
                    yerr=np.std(bottom_entropy, axis=0),
                    c=viridis_cmap(.25), alpha=.8, label=r"$w<\theta_{g}$")
        ax.errorbar((ordered_snapshots) / 1000, np.mean(top_entropy, axis=0),
                    yerr=np.std(top_entropy, axis=0),
                    c=viridis_cmap(.75), alpha=.8, label=r"$w\geq\theta_{g}$")
        plt.xlabel("Training time (seconds)")
        plt.ylabel("Average Entropy")
        plt.legend(loc='best')
        # plt.ylim([-.05, 1.05])
        plt.grid(True, which='major', axis='y')
        plt.savefig(fig_folder + "readout_average_entropy_evo{}.pdf".format(suffix_test))
        plt.savefig(fig_folder + "readout_average_entropy_evo{}.svg".format(suffix_test))
        if show_plots:
            plt.show()
        plt.close(fig)

        # TODO do readout neuron classes match DSI
    print(Fore.GREEN, "{:45}".format("The suffix for this set of figures has been"), ":", suffix_test, Style.RESET_ALL)
    print("=" * 45)


if __name__ == "__main__":
    import sys

    #  post area
    # fname = "random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_cont"
    # _power_on_self_test(fname, training_type="uns", extra_suffix="_p_.2_b_1.1")  # perfect
    # _power_on_self_test(fname, training_type="uns", extra_suffix="_run_0_100s")
    # /post area

    fname = "random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_cont"

    analyse_multiple_runs(fname, runs=[0,1,2,7,8], training_type="uns", extra_suffix="_400s_new_defaults_b_1.3_p_0.05",
                          preproc_archive="results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_cont",
                          testing_suffix="_class_dev")
    analyse_multiple_runs(fname, runs=[0,1,2,7,8], training_type="uns", extra_suffix="_400s_new_defaults_b_1.3_p_0.05",
                          preproc_archive="results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_cont",
                          testing_suffix="_class_dev_jitter")
    analyse_multiple_runs(fname, runs=[0,1,2,7,8], training_type="uns", extra_suffix="_400s_new_defaults_b_1.3_p_0.05",
                          preproc_archive="results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_cont",
                          testing_suffix="_jitter")
    # ok rewiring results
    analyse_multiple_runs(fname, runs=3, training_type="uns", extra_suffix="_rewiring_400s_new_defaults_b_1.3_frew_10",
                          preproc_archive="results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_cont")
    # legit results
    analyse_multiple_runs(fname, runs=[0,1,2,7,8], training_type="uns", extra_suffix="_400s_new_defaults_b_1.3_p_0.05",
                          preproc_archive="results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_cont")

    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_run_0_400s_new_defaults_b_1.3_p_0.05")
    sys.exit()
    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_run_1_400s_new_defaults_b_1.3_p_0.05")
    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_run_8_400s_new_defaults_b_1.3_p_0.05")
    sys.exit()
    analyse_multiple_runs(fname, runs=1, training_type="uns", extra_suffix="_rewiring_400s_new_defaults_b_1.3_frew_100",
                          preproc_archive="results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_cont")
    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_run_0_rewiring_400s_new_defaults_b_1.3_frew_100")

    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_run_0_rewiring_400s_new_defaults_b_1.3_frew_10")
    sys.exit()
    analyse_multiple_runs(fname, runs=1, training_type="uns", extra_suffix="_rewiring_400s_new_defaults_b_1.3",
                          preproc_archive="results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_cont")
    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_run_0_rewiring_400s_new_defaults_b_1.3")
    sys.exit()
    analyse_multiple_runs(fname, runs=1, training_type="uns", extra_suffix="_rewiring_400s_new_defaults_b_1.3_smax_16",
                          preproc_archive="results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_cont")
    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_run_0_rewiring_400s_new_defaults_b_1.3_smax_16")
    sys.exit()
    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_run_0_200s_new_defaults_b_1.3_p_0.05")
    # sys.exit()
    analyse_multiple_runs(fname, runs=10, training_type="uns", extra_suffix="_200s_new_defaults_b_1.3_p_0.05",
                          preproc_archive="results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_cont")
    sys.exit()
    analyse_multiple_runs(fname, runs=10, training_type="uns", extra_suffix="_rewiring_200s_new_defaults_b_1.3_smax_32",
                          preproc_archive="results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_cont")
    analyse_multiple_runs(fname, runs=5, training_type="uns", extra_suffix="_200s_new_defaults_b_1.3_p_0.05",
                          preproc_archive="results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_cont",
                          testing_suffix="_class_dev")
    analyse_multiple_runs(fname, runs=5, training_type="uns", extra_suffix="_200s_new_defaults_b_1.3_p_0.05",
                          preproc_archive="results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_cont",
                          testing_suffix="_class_dev_jitter")
    analyse_multiple_runs(fname, runs=5, training_type="uns", extra_suffix="_200s_new_defaults_b_1.3_p_0.05",
                          preproc_archive="results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_cont",
                          testing_suffix="_jitter")

    analyse_multiple_runs(fname, runs=5, training_type="uns", extra_suffix="_200s_new_defaults_b_1.3_p_0.05_test_nesw",
                          preproc_archive="results_for_testing_random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_cont")
    sys.exit()
    # readout_neuron_analysis(fname, training_type="uns", extra_suffix="_run_0_rewiring_200s_new_defaults_b_1.3_smax_32")
    # readout_neuron_analysis(fname, training_type="uns", extra_suffix="_run_4_rewiring_200s_new_defaults_b_1.3_smax_32")
    # readout_neuron_analysis(fname, training_type="uns", extra_suffix="_run_0_rewiring_200s_new_defaults_b_1.3_smax_64_f_rew_100_p_elim")
    # sys.exit()
    analyse_multiple_runs(fname, runs=10, training_type="uns", extra_suffix="_200s_new_defaults_b_1.3_p_0.1")
    # analyse_multiple_runs(fname, runs=10, training_type="uns", extra_suffix="_200s_new_defaults_b_1.3_p_0.05")
    analyse_multiple_runs(fname, runs=10, training_type="uns", extra_suffix="_200s_new_defaults_b_1.3")
    analyse_multiple_runs(fname, runs=10, training_type="uns", extra_suffix="_rewiring_200s_new_defaults_b_1.3_smax_64_f_rew_100_p_elim")
    sys.exit()

    # fname = "random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_NESW_cont"
    # analyse_multiple_runs(fname, runs=5, training_type="uns", extra_suffix="_rewiring_100s_new_defaults_p_.2")
    fname = "random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_cont"
    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_run_0_100s_new_defaults")
    fname = "random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_cont"
    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_run_0_rewiring_100s_new_defaults")
    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_run_9_rewiring_100s_new_defaults")
    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_run_6_rewiring_100s_new_defaults_b_1.3_smax_32")
    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_run_0_100s_new_defaults_b_1.3")
    # readout_neuron_analysis(fname, training_type="uns", extra_suffix="_run_6_100s_new_defaults_b_1.3_smax_32")

    sys.exit()
    # fname = "random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_cont"
    # analyse_multiple_runs(fname, runs=[0], training_type="uns", extra_suffix="_100s_new_defaults")
    # analyse_multiple_runs(fname, runs=[0], training_type="uns", extra_suffix="_rewiring_100s_new_defaults")

    # NESW
    # fname = "random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_NESW_cont"
    # analyse_multiple_runs(fname, runs=[0,1,2,3,4,7,8], training_type="uns", extra_suffix="_rewiring_100s_new_defaults")
    # analyse_multiple_runs(fname, runs=[0,1,2,3,4,5,6,7,8,9], training_type="uns", extra_suffix="_100s_new_defaults")

    sys.exit()
    fname = "random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_cont"
    analyse_multiple_runs(fname, runs=7, training_type="uns", extra_suffix="_rewiring_100s_new_defaults_b_1.3_smax_32")
    analyse_multiple_runs(fname, runs=10, training_type="uns", extra_suffix="_rewiring_100s_new_defaults_smax_128")
    analyse_multiple_runs(fname, runs=9, training_type="uns", extra_suffix="_100s_new_defaults")
    # analyse_multiple_runs(fname, runs=10, training_type="uns", extra_suffix="_100s_new_defaults_p_.2")
    analyse_multiple_runs(fname, runs=10, training_type="uns", extra_suffix="_rewiring_100s_new_defaults")
    analyse_multiple_runs(fname, runs=6, training_type="uns", extra_suffix="_100s_new_defaults_b_1.3")
    analyse_multiple_runs(fname, runs=6, training_type="uns", extra_suffix="_rewiring_100s_new_defaults_b_1.3")

    fname = "constant_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_cont"
    analyse_multiple_runs(fname, runs=7, training_type="uns", extra_suffix="_rewiring_100s_new_defaults_b_1.3_smax_32")
    analyse_multiple_runs(fname, runs=9, training_type="uns", extra_suffix="_100s_new_defaults")  # TODO
    # analyse_multiple_runs(fname, runs=10, training_type="uns", extra_suffix="_100s_new_defaults_p_.2")
    # analyse_multiple_runs(fname, runs=[
    #     0, 2, 3, 4, 5, 6, 7, 8, 9
    # ], training_type="uns", extra_suffix="_rewiring_100s_new_defaults")
    analyse_multiple_runs(fname, runs=10, training_type="uns", extra_suffix="_rewiring_100s_new_defaults_smax_128")
    analyse_multiple_runs(fname, runs=6, training_type="uns", extra_suffix="_100s_new_defaults_b_1.3")
    analyse_multiple_runs(fname, runs=6, training_type="uns", extra_suffix="_rewiring_100s_new_defaults_b_1.3")

    sys.exit()
    fname = "random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_cont"
    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_run_9_100s_p_.2_b_1.1")
    fname = "random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_cont"
    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_run_0_100s_p_.2_b_1.1")

    # sys.exit()
    fname = "random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_cont"
    analyse_multiple_runs(fname, runs=10, training_type="uns", extra_suffix="_100s_p_.2_b_1.1")
    analyse_multiple_runs(fname, runs=5, training_type="uns", extra_suffix="_100s")
    analyse_multiple_runs(fname, runs=4, training_type="uns", extra_suffix="_100s_nesw")
    analyse_multiple_runs(fname, runs=4, training_type="uns", extra_suffix="_100s_nesw")

    fname = "random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_cont"
    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_p_.2_b_1.1")  # perfect

    fname = "constant_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_cont"
    analyse_multiple_runs(fname, runs=5, training_type="uns", extra_suffix="_100s")
    analyse_multiple_runs(fname, runs=5, training_type="uns", extra_suffix="_rewiring_100s")

    sys.exit()

    # Attempting readout of constant delay network

    fname = "random_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_cont"
    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_NESW")
    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_NESW_80s")
    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_rewiring_NESW_rew_p_0")
    sys.exit()

    fname = "constant_delay_smax_128_gmax_1_192k_sigma_7.5_3_angle_0_90_cont"

    readout_neuron_analysis(fname, training_type="uns")
    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_rewiring_p_.0_smax_32_rew_wta")
    readout_neuron_analysis(fname, training_type="uns", extra_suffix="_rewiring_p_.0_smax_32_20s_rew_wta")

    sys.exit()

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
