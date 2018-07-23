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
# from brian2.units import *
import os

import matplotlib as mlib

mlib.rcParams.update({'font.size': 24})
mlib.rcParams.update({'errorbar.capsize': 5})
mlib.rcParams.update({'figure.autolayout': True})

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

if sensitivity_analysis:
    # set up final matrix
    batch_matrix_results = []
    # also set up final snapshots
    batch_snapshots = []
    # don't forget about sim_params
    batch_params = []  # append into this sim params in order
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
        filename = "results_for" + str(file)

        if os.path.isfile(filename + ".npz"):
            print("Analysis has been run before & Cached version of results "
                  "exists!")
            cached = True

        # Don't do extra work if we've already done all of this
        simtime = int(data['simtime'])
        post_spikes = data['post_spikes']

        input_grating_fname = "spiking_moving_bar_motif_bank_simtime_{" \
                              "}s.npz".format(simtime // 1000)

        testing_data = np.load(
            "../synaptogenesis/spiking_moving_bar_input/" +
            input_grating_fname)
        chunk = testing_data['chunk']
        actual_angles = testing_data['actual_angles']

        # ff_last = data['final_pre_weights']
        # lat_last = data['final_post_weights']
        # init_ff_weights = data['init_ff_connections']
        # init_lat_weights = data['init_lat_connections']
        # ff_init = data['init_ff_connections']
        # lat_init = data['init_lat_connections']

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

        if not cached:
            total_target_neuron_mean_spike_rate = \
                post_spikes.shape[0] / float(simtime) * 1000. / N_layer
            instaneous_rates = np.empty(int(simtime / chunk))
            for index, value in np.ndenumerate(instaneous_rates):
                chunk_index = index[0]
                chunk_size = chunk

                instaneous_rates[chunk_index] = np.count_nonzero(
                    np.logical_and(
                        post_spikes[:, 1] >= (chunk_index * chunk_size),
                        post_spikes[:, 1] <= ((chunk_index + 1) * chunk_size)
                    )
                ) / (N_layer * chunk)

            rate_means = []
            rate_stds = []
            rate_sem = []
            all_rates = []
            angles = np.arange(0, 360, 5)
            for angle in angles:
                rates_for_current_angle = instaneous_rates[
                    np.where(actual_angles == angle)]
                rate_means.append(np.mean(rates_for_current_angle))
                rate_stds.append(np.std(rates_for_current_angle))
                rate_sem.append(stats.sem(rates_for_current_angle))
                all_rates.append(rates_for_current_angle)
            rate_means = np.asarray(rate_means)
            rate_stds = np.asarray(rate_stds)
            rate_sem = np.asarray(rate_sem)
            all_rates = np.asarray(all_rates)
            radians = angles * np.pi / 180.
        else:
            cached_data = np.load(filename + ".npz")
            rate_means = cached_data['rate_means']
            rate_stds = cached_data['rate_stds']
            rate_sem = cached_data['rate_sem']
            all_rates = cached_data['all_rates']
            radians = cached_data['radians']
            instaneous_rates = cached_data['instaneous_rates']
            angles = cached_data['angles']
            actual_angles = cached_data['actual_angles']

        print()
        pp(sim_params)
        print()
        print("%-60s" % "Target neuron spike rate",
              total_target_neuron_mean_spike_rate, "Hz")
        # print("%-60s" % "Final mean number of feedforward synapses", final_mean_number_ff_synapses)
        # print("%-60s" % "Weight as proportion of max", final_weight_proportion)
        # print("%-60s" % "Mean sigma aff init", init_mean_std)
        # print("%-60s" % "Mean sigma aff fin conn shuffle", fin_mean_std_conn_shuf)
        # print("%-60s" % "Mean sigma aff fin conn", fin_mean_std_conn)
        # print("%-60s" % "p(WSR sigma aff fin conn vs sigma aff fin conn shuffle)", wsr_sigma_fin_conn_fin_conn_shuffle.pvalue)
        # print("%-60s" % "Mean sigma aff fin weight shuffle", fin_mean_std_weight_shuf)
        # print("%-60s" % "Mean sigma aff fin weight", fin_mean_std_weight)
        # print("%-60s" % "p(WSR sigma aff fin weight vs sigma aff fin weight shuffle)", wsr_sigma_fin_weight_fin_weight_shuffle.pvalue)
        # print("%-60s" % "Mean AD init", init_mean_AD)
        # print("%-60s" % "Mean AD fin conn shuffle", fin_mean_AD_conn_shuf)
        # print("%-60s" % "Mean AD fin conn", fin_mean_AD_conn)
        # print("%-60s" % "p(WSR AD fin conn vs AD fin conn shuffle)", wsr_AD_fin_conn_fin_conn_shuffle.pvalue)
        # print("%-60s" % "Mean AD fin weight shuffle", fin_mean_AD_weight_shuf)
        # print("%-60s" % "Mean AD fin weight", fin_mean_AD_weight)
        # print("%-60s" % "p(WSR AD fin weight vs AD fin weight shuffle)", wsr_AD_fin_weight_fin_weight_shuffle.pvalue)
        #
        # if sensitivity_analysis:
        #     batch_matrix_results.append((
        #         total_target_neuron_mean_spike_rate,
        #         final_mean_number_ff_synapses,
        #         final_weight_proportion,
        #         init_mean_std,
        #         fin_mean_std_conn_shuf,
        #         fin_mean_std_conn,
        #         wsr_sigma_fin_conn_fin_conn_shuffle.pvalue,
        #         fin_mean_std_weight_shuf,
        #         fin_mean_std_weight,
        #         wsr_sigma_fin_weight_fin_weight_shuffle.pvalue,
        #         init_mean_AD,
        #         fin_mean_AD_conn_shuf,
        #         fin_mean_AD_conn,
        #         wsr_AD_fin_conn_fin_conn_shuffle.pvalue,
        #         fin_mean_AD_weight_shuf,
        #         fin_mean_AD_weight,
        #         wsr_AD_fin_weight_fin_weight_shuffle.pvalue,
        #         file
        #     ))
        # # final weight histogram
        # # ff weight histogram
        # plt.figure(figsize=(10, 5), dpi=600)
        # plt.hist(ff_last[:, 2]/g_max, bins=100, normed=True)
        # plt.title("Histogram of feedforward weights")
        # plt.tight_layout()
        # plt.show()
        # # lat weight histogram
        # plt.figure(figsize=(10, 5), dpi=600)
        # plt.hist(lat_last[:, 2]/g_max, bins=100, normed=True)
        # plt.title("Histogram of lateral weights")
        # plt.tight_layout()
        # plt.show()
        # # LAT connection bar chart
        #
        # init_fan_in_rec = fan_in(init_conn, init_weight, 'conn', 'rec')
        #
        # mean_projection_rec, means_and_std_devs_rec, \
        # means_for_plot_rec, mean_centred_projection_rec = centre_weights(
        #     init_fan_in_rec, 16)
        #
        # init_fan_in_rec_rad = radial_sample(mean_projection_rec, 100)
        #
        # final_fan_in_rec = fan_in(last_conn, last_weight, 'weight',
        #                           'rec')
        #
        # final_mean_projection_rec, final_means_and_std_devs_rec, \
        # final_means_for_plot_rec, final_mean_centred_projection_rec = centre_weights(
        #     final_fan_in_rec, 16)
        #
        # final_fan_in_rec_rad = \
        #     radial_sample(final_mean_projection_rec, 100)
        #
        # final_fan_in_rec_conn = fan_in(last_conn, last_weight, 'conn',
        #                                'rec')
        #
        # final_mean_projection_rec_conn, final_means_and_std_devs_rec_conn, \
        # final_means_for_plot_rec_conn, final_mean_centred_projection_rec_conn = centre_weights(
        #     final_fan_in_rec_conn, 16)
        #
        # final_fan_in_rec_rad_conn = \
        #     radial_sample(final_mean_projection_rec_conn, 100)
        #
        # ## FF connection bar chart
        #
        # init_fan_in_ff = fan_in(init_conn, init_weight, 'conn', 'ff')
        #
        # mean_projection_ff, means_and_std_devs_ff, \
        # means_for_plot_ff, mean_centred_projection_ff = centre_weights(
        #     init_fan_in_ff, 16)
        #
        # init_fan_in_ff_rad = radial_sample(mean_projection_ff, 100)
        #
        # final_fan_in_ff = fan_in(last_conn, last_weight, 'weight',
        #                          'ff')
        #
        # final_mean_projection_ff, final_means_and_std_devs_ff, \
        # final_means_for_plot_ff, final_mean_centred_projection_ff = centre_weights(
        #     final_fan_in_ff, 16)
        #
        # final_fan_in_ff_rad = \
        #     radial_sample(final_mean_projection_ff, 100)
        #
        # final_fan_in_ff_conn = fan_in(last_conn, last_weight, 'conn',
        #                               'ff')
        #
        # final_mean_projection_ff_conn, final_means_and_std_devs_ff_conn, \
        # final_means_for_plot_ff_conn, final_mean_centred_projection_ff_conn = centre_weights(
        #     final_fan_in_ff_conn, 16)
        #
        # final_fan_in_ff_rad_conn = \
        #     radial_sample(final_mean_projection_ff_conn, 100)
        #
        #
        # # Time stuff
        #
        # end_time = plt.datetime.datetime.now()
        # suffix = end_time.strftime("_%H%M%S_%d%m%Y")
        #
        # elapsed_time = end_time - start_time
        #
        # print("Total time elapsed -- " + str(elapsed_time))
        #

        np.savez(filename, recording_archive_name=file,
                 target_neurom_mean_spike_rate=total_target_neuron_mean_spike_rate,
                 instaneous_rates=instaneous_rates,
                 rate_means=rate_means,
                 rate_stds=rate_stds,
                 rate_sem=rate_sem,
                 all_rates=all_rates,
                 actual_angles=actual_angles,
                 angles=angles,
                 radians=radians
                 )

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
            # ax.plot(pol_conn[:,0], pol_conn[:,-1], ls="*")
            # ax.set_rmax(2)
            # ax.set_rticks([0.5, 1, 1.5, 2])  # less radial ticks
            # ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
            # ax.grid(True)
            N = 150
            r = 2 * np.random.rand(N)
            theta = 2 * np.pi * np.random.rand(N)
            # area = 2 * pol_conn[:,3]**2
            # ax = plt.subplot(111, polar=True)
            # c = plt.scatter(radians, rate_stds, c=rate_means)#, c=pol_conn[:,2], s=area)
            c = plt.scatter(radians, rate_means, c=rate_sem,
                            s=100)  # , c=pol_conn[:,2], s=area)
            c.set_alpha(0.8)
            plt.ylim([0, 1.2 * np.max(rate_means)])
            # plt.ylabel("Hz")
            plt.xlabel("Angle")

            ax.set_title("Mean firing rate for specific input angle\n",
                         va='bottom')
            plt.savefig("firing_rate_with_angle.png")
            plt.show()

            f, ax = plt.subplots(1, 1, figsize=(15, 8),
                                 subplot_kw=dict(projection='polar'), dpi=800)

            # '#440357'  '#228b8d', '#b2dd2c'

            maximus = np.max((rate_means))
            # minimus = np.min((random_rate_means, constant_rate_means))
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
            # c.set_alpha(0.8)
            ax.set_ylim([minimus, 1.1 * maximus])
            # plt.ylabel("Hz")
            # ax2 = plt.subplot(222, projection='polar')

            # c2.set_alpha(0.8)

            # ax.set_xlabel("Angle")
            # ax2.set_xlabel("Angle")
            ax.set_xlabel("Random delays")

            # f.suptitle("Mean firing rate for specific input angle", va='bottom')
            # plt.tight_layout(pad=10)
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

        # if args.snapshots:
        #
        #     all_ff_connections = data['ff_connections']
        #     all_lat_connections = data['lat_connections']
        #     if data:
        #         data.close()
        #     number_of_recordings = all_ff_connections.shape[0]
        #     all_mean_sigmas = np.ones(number_of_recordings) * np.nan
        #     all_mean_ADs = np.ones(number_of_recordings) * np.nan
        #
        #     all_mean_sigmas_conn = np.ones(number_of_recordings) * np.nan
        #     all_mean_ADs_conn = np.ones(number_of_recordings) * np.nan
        #
        #     all_mean_s = np.zeros(number_of_recordings)
        #     for index in range(number_of_recordings):
        #         conn, weight = \
        #             list_to_post_pre(all_ff_connections[index],
        #                              all_lat_connections[index], 16,
        #                              N_layer)
        #
        #         current_fan_in = fan_in(conn, weight, 'weight', 'ff')
        #         mean_projection, means_and_std_devs, means_for_plot, mean_centred_projection = centre_weights(
        #             current_fan_in, 16)
        #
        #         all_mean_sigmas[index] = np.mean(means_and_std_devs[:, 5])
        #         all_mean_ADs[index] = np.mean(means_and_std_devs[:, 4])
        #
        #         all_mean_s[index] = conn[conn != -1].size / float(N_layer)
        #
        #         current_fan_in_conn = fan_in(conn, weight, 'conn', 'ff')
        #         mean_projection_conn, means_and_std_devs_conn, \
        #         means_for_plot_conn, mean_centred_projection_conn = centre_weights(
        #             current_fan_in_conn, 16)
        #
        #         all_mean_sigmas_conn[index] = np.mean(
        #             means_and_std_devs_conn[:, 5])
        #         all_mean_ADs_conn[index] = np.mean(
        #             means_and_std_devs_conn[:, 4])
        #
        #         # mean_std, stds, mean_AD, AD, variances = sigma_and_ad(
        #         #     all_ff_connections[index, :, :],
        #         #     unitary_weights=False,
        #         #     resolution=args.resolution)
        #         # all_mean_sigmas[index] = mean_std
        #         # all_mean_ADs[index] = mean_AD
        #     np.savez("last_std_ad_evo", recording_archive_name=file,
        #              all_mean_sigmas=all_mean_sigmas,
        #              all_mean_ads=all_mean_ADs,
        #              all_mean_sigmas_conn=all_mean_sigmas_conn,
        #              all_mean_ads_conn=all_mean_ADs_conn)
        #     if sensitivity_analysis:
        #         batch_snapshots.append((
        #             np.copy(all_mean_sigmas),
        #             np.copy(all_mean_ADs),
        #             np.copy(all_mean_sigmas_conn),
        #             np.copy(all_mean_ADs_conn),
        #             file
        #         ))
        #     if args.plot and not sensitivity_analysis:
        #         plt.plot(all_mean_sigmas)
        #         plt.ylim([0, 1.1 * np.max(all_mean_sigmas)])
        #         plt.show()
        #         plt.plot(all_mean_ADs)
        #         plt.ylim([0, 1.1 * np.max(all_mean_ADs)])
        #         plt.show()
        #
        #         # Plot evolution of mean synaptic capacity usage per
        #         # postsynaptic neuron
        #         f, (ax1) = plt.subplots(1, 1, figsize=(16, 8))
        #         i = ax1.plot(np.arange(all_mean_s.shape[0]) * 30, all_mean_s,
        #                      label='Mean synaptic capacity usage')
        #         ax1.grid(visible=False)
        #         ax1.set_title(
        #             "Evolution of synaptic capacity usage",
        #             fontsize=16)
        #
        #         # ax1.plot(final_ff_capacities + final_lat_capacities, c='y',
        #         #          alpha=.9,
        #         #          label='Total synaptic capacity usage')
        #         ax1.axhline(y=s_max * 2, xmin=0, xmax=ff_last.shape[1], c='r',
        #                     label='$S_{max}$')
        #         ax1.legend(loc='best')
        #         ax1.set_ylim([0, 33])
        #         ax1.set_xlabel("Time(s)")
        #         ax1.set_ylabel("Mean number of afferent connections")
        #         plt.show()
        #
        #

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
    np.savez("batch_analysis" + suffix_total, recording_archive_name=file,
             snapshots=batch_snapshots,
             params=batch_params,
             results=batch_matrix_results
             )
