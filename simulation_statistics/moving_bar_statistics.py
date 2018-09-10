from __future__ import division, print_function
from collections import Iterable
import traceback
import numpy as np
import matplotlib.pyplot as plt
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import animation, rc, colors
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import scipy
import scipy.stats as stats
from glob import glob
from pprint import pprint as pp
from analysis_functions_definitions import *
from argparser import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
from brian2.units import *
import os
import ntpath

import matplotlib as mlib

from spinn_utilities.progress_bar import ProgressBar

mlib.rcParams.update({'font.size': 24})
mlib.rcParams.update({'errorbar.capsize': 5})
mlib.rcParams.update({'figure.autolayout': True})

start_time = plt.datetime.datetime.now()
paths = []
for file in args.path:
    if "*" in file:
        globbed_files = glob(file)
        for globbed_file in globbed_files:
            if "npz" in globbed_file:
                paths.append(globbed_file)
    else:
        paths.append(file)

sensitivity_analysis = False
if len(paths) > 1:
    sensitivity_analysis = True
    # don't display plots

cached = False
if sensitivity_analysis:
    # set up final matrix
    batch_matrix_results = []
    # also set up final snapshots
    batch_snapshots = []
    # don't forget about sim_params
    batch_params = []  # append into this sim params in order
    batch_files = []
    print()
    print("BATCH ANALYSIS!")
    print()

for file in paths:
    try:
        start_time = plt.datetime.datetime.now()
        print("\n\nAnalysing file", str(file))
        if "npz" in str(file):
            data = np.load(file)
        else:
            data = np.load(str(file) + ".npz")

        sim_params = np.array(data['sim_params']).ravel()[0]
        if sensitivity_analysis:
            batch_params.append((sim_params, file))

        if 'case' in sim_params:
            print("Case", sim_params['case'], "analysis")
        else:
            print("Case unknown")

        cached = False
        # Do we already have a cached version of the results?
        filename = "results_for_" + str(ntpath.basename(file))

        if os.path.isfile(filename + ".npz"):
            print("Analysis has been run before & Cached version of results "
                  "exists!")
            cached = True

        # Don't do extra work if we've already done all of this
        simtime = int(data['simtime']) * ms
        post_spikes = data['post_spikes']
        inh_post_spikes = data['inh_post_spikes']

        try:
            # retrieve some important sim params
            grid = sim_params['grid']
            N_layer = grid[0] * grid[1]
            n = int(np.sqrt(N_layer))
            g_max = sim_params['g_max']
            s_max = sim_params['s_max']
            sigma_form_forward = sim_params['sigma_form_forward']
            sigma_form_lateral = sim_params['sigma_form_lateral']
            p_form_lateral = sim_params['p_form_lateral']
            p_form_forward = sim_params['p_form_forward']
            p_elim_dep = sim_params['p_elim_dep']
            p_elim_pot = sim_params['p_elim_pot']
            f_rew = sim_params['f_rew']
        except:
            # use defaults
            print("USING DEFAULTS! SOMETHING WENT WRONG!")
            grid = np.asarray([16, 16])
            N_layer = 256
            n = 32
            s_max = 16
            sigma_form_forward = 2.5
            sigma_form_lateral = 1
            p_form_lateral = 1
            p_form_forward = 0.16
            p_elim_dep = 0.0245
            p_elim_pot = 1.36 * np.e ** -4
            f_rew = 10 ** 4  # Hz
            g_max = .2

        input_grating_fname = "spiking_moving_bar_motif_bank_simtime_" \
                              "{}x{}_{}s.npz".format(n, n, int(simtime /
                                                               second))

        testing_data = np.load(
            "../synaptogenesis/spiking_moving_bar_input/" +
            input_grating_fname)

        if ".npz" in data['testing'].ravel()[0]:
            conn_data_filename = data['testing'].ravel()[0]
        else:
            conn_data_filename = data['testing'].ravel()[0] + ".npz"
        connection_data = np.load(os.path.join(ntpath.dirname(file),
                                               conn_data_filename))

        chunk = testing_data['chunk'] * ms
        actual_angles = testing_data['actual_angles']
        training_sim_params = np.array(connection_data['sim_params']).ravel()[
            0]
        # ff_last = data['final_pre_weights']
        # lat_last = data['final_post_weights']
        # init_ff_weights = data['init_ff_connections']
        # init_lat_weights = data['init_lat_connections']
        # ff_init = data['init_ff_connections']
        # lat_init = data['init_lat_connections']

        if not cached:
            target_neuron_mean_spike_rate = \
                post_spikes.shape[0] / (simtime * N_layer)
            # instaneous_rates = np.empty(int(simtime / chunk))
            per_neuron_instaneous_rates = np.empty((N_layer,
                                                    int(simtime / chunk)))
            chunk_size = chunk / ms
            # Cache coherent implementation
            # Excitatory population
            pbar = ProgressBar(total_number_of_things_to_do=N_layer,
                               string_describing_what_being_progressed=
                               "\nBinning firing activity per excitatory "
                               "neuron...")
            for neuron_index in np.arange(N_layer):

                firings_for_neuron = post_spikes[
                    post_spikes[:, 0] == neuron_index]
                for chunk_index in np.arange(per_neuron_instaneous_rates.shape[
                                                 1]):
                    per_neuron_instaneous_rates[neuron_index, chunk_index] = \
                        np.count_nonzero(
                            np.logical_and(
                                firings_for_neuron[:, 1] >= (
                                        chunk_index * chunk_size),
                                firings_for_neuron[:, 1] < (
                                        (chunk_index + 1) * chunk_size)
                            )
                        ) / (1 * chunk)
                pbar.update()
            instaneous_rates = np.sum(per_neuron_instaneous_rates,
                                      axis=0) / N_layer

            # Inhibitory population
            if inh_post_spikes.size > 0:
                inh_per_neuron_instaneous_rates = np.empty(
                    (N_layer, int(simtime / chunk)))
                pbar = ProgressBar(total_number_of_things_to_do=N_layer,
                                   string_describing_what_being_progressed=
                                   "\nBinning firing activity per inhibitory "
                                   "neuron...")
                for neuron_index in np.arange(N_layer):

                    firings_for_neuron = inh_post_spikes[
                        inh_post_spikes[:, 0] == neuron_index]
                    for chunk_index in \
                            np.arange(
                                inh_per_neuron_instaneous_rates.shape[1]):
                        inh_per_neuron_instaneous_rates[neuron_index,
                                                        chunk_index] = \
                            np.count_nonzero(
                                np.logical_and(
                                    firings_for_neuron[:, 1] >= (
                                            chunk_index * chunk_size),
                                    firings_for_neuron[:, 1] < (
                                            (chunk_index + 1) * chunk_size)
                                )
                            ) / (1 * chunk)
                    pbar.update()
                inh_instaneous_rates = np.sum(inh_per_neuron_instaneous_rates,
                                              axis=0) / N_layer
                inh_rate_means = []
                inh_rate_stds = []
                inh_rate_sem = []
                inh_all_rates = []
                inh_per_neuron_all_rates = []
                angles = np.arange(0, 360, 5)
                for angle in angles:
                    inh_rates_for_current_angle = inh_instaneous_rates[
                        np.where(actual_angles == angle)]
                    inh_rate_means.append(np.mean(inh_rates_for_current_angle))
                    inh_rate_stds.append(np.std(inh_rates_for_current_angle))
                    inh_rate_sem.append(stats.sem(inh_rates_for_current_angle))
                    inh_all_rates.append(inh_rates_for_current_angle)
                    inh_per_neuron_all_rates.append(
                        inh_per_neuron_instaneous_rates[:,
                        np.where(
                            actual_angles == angle)].ravel())
                inh_rate_means = np.asarray(inh_rate_means)
                inh_rate_stds = np.asarray(inh_rate_stds)
                inh_rate_sem = np.asarray(inh_rate_sem)
                inh_all_rates = np.asarray(inh_all_rates)
            else:
                inh_rate_means = []
                inh_rate_stds = []
                inh_rate_sem = []
                inh_all_rates = []
                inh_per_neuron_all_rates = []
                inh_per_neuron_instaneous_rates = np.asarray([])
                inh_instaneous_rates = np.asarray([])

            rate_means = []
            rate_stds = []
            rate_sem = []
            all_rates = []
            per_neuron_all_rates = []
            angles = np.arange(0, 360, 5)
            for angle in angles:
                rates_for_current_angle = instaneous_rates[
                    np.where(actual_angles == angle)]
                rate_means.append(np.mean(rates_for_current_angle))
                rate_stds.append(np.std(rates_for_current_angle))
                rate_sem.append(stats.sem(rates_for_current_angle))
                all_rates.append(rates_for_current_angle)
                per_neuron_all_rates.append(per_neuron_instaneous_rates[:,
                                            np.where(
                                                actual_angles == angle)].ravel())
            rate_means = np.asarray(rate_means)
            rate_stds = np.asarray(rate_stds)
            rate_sem = np.asarray(rate_sem)
            all_rates = np.asarray(all_rates)
            radians = angles * np.pi / 180.

            # Connection information

            ff_connections = connection_data['ff_connections'][0]
            lat_connections = connection_data['lat_connections'][0]
            init_ff_connections = connection_data['init_ff_connections']
            noise_connections = connection_data['noise_connections'][0]
            ff_off_connections = connection_data['ff_off_connections'][0]
            inh_connections = connection_data['inh_connections']

            final_ff_conn_network = np.ones((N_layer, N_layer)) * np.nan
            final_lat_conn_network = np.ones((N_layer, N_layer)) * np.nan
            init_ff_conn_network = np.ones((N_layer, N_layer)) * np.nan

            ff_num_network = np.zeros((N_layer, N_layer))
            lat_num_network = np.zeros((N_layer, N_layer))

            final_ff_conn_field = np.ones(N_layer) * 0
            final_lat_conn_field = np.ones(N_layer) * 0

            final_ff_num_field = np.ones(N_layer) * 0
            final_lat_num_field = np.ones(N_layer) * 0

            init_ff_num_network = np.zeros((N_layer, N_layer))
            for source, target, weight, delay in ff_connections:
                if np.isnan(final_ff_conn_network[int(source), int(target)]):
                    final_ff_conn_network[int(source), int(target)] = weight
                else:
                    final_ff_conn_network[int(source), int(target)] += weight
                ff_num_network[int(source), int(target)] += 1

            for source, target, weight, delay in noise_connections:
                if np.isnan(final_ff_conn_network[int(source), int(target)]):
                    final_ff_conn_network[int(source), int(target)] = weight
                else:
                    final_ff_conn_network[int(source), int(target)] += weight
                ff_num_network[int(source), int(target)] += 1

            for source, target, weight, delay in ff_off_connections:
                if np.isnan(final_ff_conn_network[int(source), int(target)]):
                    final_ff_conn_network[int(source), int(target)] = weight
                else:
                    final_ff_conn_network[int(source), int(target)] += weight
                ff_num_network[int(source), int(target)] += 1

            for source, target, weight, delay in lat_connections:
                if np.isnan(final_lat_conn_network[int(source), int(target)]):
                    final_lat_conn_network[int(source), int(target)] = weight
                else:
                    final_lat_conn_network[int(source), int(target)] += weight
                lat_num_network[int(source), int(target)] += 1

            # NB: for these purposes I count afferent inhibition as a lateral
            # signal
            for source, target, weight, delay in inh_connections:
                if np.isnan(final_lat_conn_network[int(source), int(target)]):
                    final_lat_conn_network[int(source), int(target)] = weight
                else:
                    final_lat_conn_network[int(source), int(target)] += weight
                lat_num_network[int(source), int(target)] += 1

            for row in range(final_ff_conn_network.shape[0]):
                final_ff_conn_field += np.roll(
                    np.nan_to_num(final_ff_conn_network[row, :]),
                    (N_layer // 2 + n // 2) - row)
                final_lat_conn_field += np.roll(
                    np.nan_to_num(final_lat_conn_network[row, :]),
                    (N_layer // 2 + n // 2) - row)

            for row in range(ff_num_network.shape[0]):
                final_ff_num_field += np.roll(
                    np.nan_to_num(ff_num_network[row, :]),
                    (N_layer // 2 + n // 2) - row)
                final_lat_num_field += np.roll(
                    np.nan_to_num(lat_num_network[row, :]),
                    (N_layer // 2 + n // 2) - row)

            # Incoming connections to target EXCITATORY population
            ff_last = connection_data['ff_connections'][0]
            off_last = connection_data['ff_off_connections'][0]
            noise_last = connection_data['noise_connections'][0]
            lat_last = connection_data['lat_connections'][0]
            inh_to_exh_last = connection_data['inh_connections']

            # Incoming connections to target INHIBITORY population
            inh_ff_last = connection_data['inh_connections'][0]
            inh_off_last = connection_data['off_inh_connections'][0]
            inh_noise_last = connection_data['noise_inh_connections']
            inh_lat_last = connection_data['inh_inh_connections']
            exh_to_inh_last = connection_data['exh_connections']

        else:
            print("Using cached data.")
            cached_data = np.load(filename + ".npz")
            rate_means = cached_data['rate_means']
            rate_stds = cached_data['rate_stds']
            rate_sem = cached_data['rate_sem']
            all_rates = cached_data['all_rates']
            radians = cached_data['radians']
            instaneous_rates = cached_data['instaneous_rates']
            angles = cached_data['angles']
            actual_angles = cached_data['actual_angles']
            target_neuron_mean_spike_rate = cached_data[
                'target_neuron_mean_spike_rate']
            per_neuron_instaneous_rates = cached_data[
                'per_neuron_instaneous_rates']
            per_neuron_all_rates = cached_data['per_neuron_all_rates']
            # Inhibitory info
            inh_rate_means = cached_data['inh_rate_means']
            inh_rate_stds = cached_data['inh_rate_stds']
            inh_rate_sem = cached_data['inh_rate_sem']
            inh_all_rates = cached_data['inh_all_rates']
            inh_instaneous_rates = cached_data['inh_instaneous_rates']
            inh_per_neuron_instaneous_rates = cached_data[
                'inh_per_neuron_instaneous_rates']
            inh_per_neuron_all_rates = cached_data['inh_per_neuron_all_rates']

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

            lat_num_network = cached_data['lat_num_network']
            ff_num_network = cached_data['ff_num_network']

            inh_lat_last = cached_data['inh_lat_last']
            exh_to_inh_last = cached_data['exh_to_inh_last']
            inh_to_exh_last = cached_data['inh_to_exh_last']
            inh_ff_last = cached_data['inh_ff_last']
            inh_off_last = cached_data['inh_off_last']
            inh_noise_last = cached_data['inh_noise_last']

        print()
        pp(sim_params)
        print()
        print("%-60s" % "Target neuron spike rate",
              target_neuron_mean_spike_rate, "Hz")

        if not cached:
            np.savez_compressed(
                filename, recording_archive_name=file,
                target_neuron_mean_spike_rate=target_neuron_mean_spike_rate,
                sim_params=sim_params,

                # Response information
                # Excitatory
                instaneous_rates=instaneous_rates,
                rate_means=rate_means,
                rate_stds=rate_stds,
                rate_sem=rate_sem,
                all_rates=all_rates,
                actual_angles=actual_angles,
                angles=angles,
                radians=radians,
                # Inhibitory
                inh_instaneous_rates=inh_instaneous_rates,
                inh_rate_means=inh_rate_means,
                inh_rate_stds=inh_rate_stds,
                inh_rate_sem=inh_rate_sem,
                inh_all_rates=inh_all_rates,

                # Per neuron response information
                # Excitatory
                per_neuron_instaneous_rates=per_neuron_instaneous_rates,
                per_neuron_all_rates=per_neuron_all_rates,
                # Inhibitory
                inh_per_neuron_instaneous_rates=inh_per_neuron_instaneous_rates,
                inh_per_neuron_all_rates=inh_per_neuron_all_rates,

                # Connection information
                ff_connections=ff_connections,
                ff_off_connections=ff_off_connections,
                lat_connections=lat_connections,
                noise_connections=noise_connections,
                ff_last=ff_last,
                off_last=off_last,
                noise_last=noise_last,
                lat_last=lat_last,
                inh_lat_last=inh_lat_last,
                exh_to_inh_last=exh_to_inh_last,
                inh_to_exh_last=inh_to_exh_last,
                inh_ff_last=inh_ff_last,
                inh_off_last=inh_off_last,
                inh_noise_last=inh_noise_last,

                final_ff_conn_field=final_ff_conn_field,
                final_ff_num_field=final_ff_num_field,
                final_lat_conn_field=final_lat_conn_field,
                final_lat_num_field=final_lat_num_field,

                lat_num_network=lat_num_network,
                ff_num_network=ff_num_network,

                # Simulation parameters
                testing_sim_params=sim_params,
                training_sim_params=training_sim_params,
            )
        else:
            print("Not re-saving the npz archive...")
        if sensitivity_analysis:
            batch_matrix_results.append((
                target_neuron_mean_spike_rate,
                np.copy(instaneous_rates),
                np.copy(per_neuron_instaneous_rates),
                np.copy(rate_means),
                np.copy(rate_stds),
                np.copy(rate_sem),
                np.copy(all_rates),
                np.copy(actual_angles),
                np.copy(angles),
                np.copy(radians),
                np.copy(ff_connections),
                np.copy(ff_off_connections),
                np.copy(lat_connections),
                np.copy(noise_connections),
                np.copy(ff_last),
                np.copy(off_last),
                np.copy(noise_last),
                np.copy(lat_last),
            ))
            batch_files.append(file)
        if args.plot and not sensitivity_analysis:
            fig = plt.figure(figsize=(16, 8), dpi=600)
            y, binEdges = np.histogram(angles, bins=angles.size)
            bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
            width = 5
            plt.bar(angles, rate_means, width=width, yerr=rate_sem,
                    color='#414C82', edgecolor='k')
            plt.xlabel("Degree")
            plt.ylabel("Hz")
            plt.savefig("firing_rate_with_angle_hist.png")
            plt.show()

            fig = plt.figure(figsize=(16, 8), dpi=600)
            y, binEdges = np.histogram(angles, bins=angles.size)
            bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
            width = 5
            plt.bar(angles, rate_means, width=width, yerr=rate_stds,
                    color='#414C82', edgecolor='k')
            plt.xlabel("Degree")
            plt.ylabel("Hz")
            plt.savefig("firing_rate_with_angle_hist_std.png")
            plt.show()

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

            ax.set_title("Mean firing rate for specific input angle\n",
                         va='bottom')
            plt.savefig("firing_rate_with_angle.png")
            plt.show()

            f, ax = plt.subplots(1, 1, figsize=(15, 8),
                                 subplot_kw=dict(projection='polar'), dpi=800)

            # '#440357'  '#228b8d', '#b2dd2c'

            maximus = np.max((rate_means))
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
            ax.set_ylim([minimus, 1.1 * maximus])
            ax.set_xlabel("Random delays")
            plt.savefig("rate_means_min_max_mean.png", bbox_inches='tight')
            plt.show()

            plt.figure(figsize=(14, 8), dpi=600)
            n, bins, patches = plt.hist(instaneous_rates, bins=40, normed=True)
            print("mean", np.mean(instaneous_rates))
            print("skew", stats.skew(n))
            print("kurtosis", stats.kurtosis(n))
            print("normality", stats.normaltest(n))
            plt.title("Distribution of rates")
            plt.xlabel("Instantaneous rate (Hz)")
            plt.show()

            plt.figure(figsize=(14, 8), dpi=600)
            plt.plot(instaneous_rates)
            plt.title("Evolution of rates")
            plt.xlabel("Testing iteration")
            plt.show()


    except IOError as e:
        print("IOError:", e)
        traceback.print_exc()
    except MemoryError:
        print("Out of memory. Did you use HDF5 slices to read in data?", e)
    finally:
        data.close()

if sensitivity_analysis:
    curr_time = plt.datetime.datetime.now()
    suffix_total = curr_time.strftime("_%H%M%S_%d%m%Y")
    np.savez_compressed("batch_analysis" + suffix_total,
                        recording_archive_name=file,
                        snapshots=batch_snapshots,
                        params=batch_params,
                        results=batch_matrix_results,
                        files=batch_files
                        )
end_time = plt.datetime.datetime.now()
total_time = end_time - start_time
print("Results in", filename)
print("Total time elapsed -- " + str(total_time))
if cached:
    print("Used cached data!")
