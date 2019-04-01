"""
Train a layer of readout neurons for a layer formed using topographic map
formation using STDP and synaptic rewiring.
"""
# Imports
import traceback

from function_definitions import *
# Argparser also includes defaults
from readout_argparser import *

import numpy as np
import pylab as plt
import os
import ntpath
import copy

import spynnaker8 as sim
# from spynnaker8.extra_models import SpikeSourcePoissonVariable

# SpiNNaker setup
TRAINING_PHASE = 0
TESTING_PHASE = 1

PHASES = [TRAINING_PHASE, TESTING_PHASE]
PHASES_NAMES = ["training", "testing"]

if (not args.min_supervised and
        not args.max_supervised and
        not args.unsupervised):
    raise AttributeError("Testing setup insufficiently defined! "
                         "What kind of training regime should be used "
                         "for the readout neurons (i.e. supervised or "
                         "unsupervised)?")
if len(args.path) == 0:
    raise AttributeError("Testing setup insufficiently defined! "
                         "Please specify connectivity npz file.")

def generate_readout_filename(path, phase, args, run, e):
    # need to retrieve name of the file (not the entire path)
    prefix = "training_readout_for_"
    if phase == TESTING_PHASE:
        prefix = "testing_readout_for_"
    if args.min_supervised:
        prefix += "min_"
    elif args.max_supervised:
        prefix += "max_"
    elif args.unsupervised:
        prefix += "uns_"
    filename = prefix + str(ntpath.basename(path))
    if ".npz" in filename:
        filename = filename[:-4]

    filename += "_run_" + str(run)

    if e:
        filename = "error_" + filename

    if args.rewiring:
        filename += "_rewiring"

    if args.suffix:
        filename += "_" + args.suffix
    return filename

# check if the figures folder exist
sim_dir = args.sim_dir
if not os.path.isdir(sim_dir) and not os.path.exists(sim_dir):
    print("Making sims dir ...")
    os.mkdir(sim_dir)

initial_weight = 0
if args.min_supervised:
    initial_weight = DEFAULT_W_MIN
if args.max_supervised:
    initial_weight = DEFAULT_W_MAX
for path in args.path:
    # Outer setup
    if ".npz" not in path:
        filename = path + ".npz"
    else:
        filename = path
    data = np.load(os.path.join(sim_dir, filename))

    # Read all required parameters
    sim_params = np.array(data['sim_params']).ravel()[0]
    cell_params = sim_params['cell_params']
    grid = sim_params['grid']
    topology = data['topology']
    N_layer = int(grid[0] * grid[1])  # Total number of neurons
    n = int(np.sqrt(N_layer))
    f_base = sim_params['f_base']  # Hz
    if (n == 32 and not args.mnist) or (n == 28 and args.mnist):
        chunk = 200  # ms
    elif n == 64:
        chunk = 400  # ms
    else:
        raise AttributeError("What chunk to use for the specified grid size?")
    simtime = args.no_iterations
    testing_no_iterations_per_class = args.testing_no_iterations_per_class
    testing_simtime = testing_no_iterations_per_class * len(args.classes) * chunk
    current_training_file = None
    snapshot_no = 1
    snap_keys = [0]
    for run in np.arange(args.runs):
        print("Run ", run)
        current_error = None
        for phase in PHASES:
            print("Phase ", PHASES_NAMES[phase])
            target_snapshots = {}
            wta_snapshots = {}
            readout_spikes_snapshots = {}
            target_spikes_snapshots = {}
            inhibitory_spikes_snapshots = {}
            actual_classes_snapshots = {}
            mock_filename = generate_readout_filename(path, phase, args, run, None)

            if mock_filename and os.path.isfile(mock_filename + ".npz") and not args.no_cache:
                print("Simulation has been run before & Cached version of results "
                      "exists!")
                print(mock_filename)
                current_training_file = mock_filename
                continue
            if current_error:
                print("Something broke... aborting this run!")
                break
            if phase == TESTING_PHASE:
                if current_training_file is None:
                    raise AttributeError(
                        "Training failed or something else went wrong")
                readout_training_data = np.load(
                    os.path.join(sim_dir, current_training_file + ".npz"))
                snapshots_present = readout_training_data['snapshots_present']
                target_snapshots = readout_training_data['target_snapshots'].ravel()[0]
                wta_snapshots = readout_training_data['wta_snapshots'].ravel()[0]
                snap_keys = target_snapshots.keys()
                snapshot_no = 1 if not snapshots_present else len(snap_keys)
                simtime = testing_simtime
            start_time = plt.datetime.datetime.now()

            # if the network has snapshots and run this for every simulation
            for snap in range(snapshot_no):
                # Generate the input (Moving bar or MNIST)
                actual_classes = []
                if not args.mnist:
                    # Generate equal number of instances of classes
                    if phase == TESTING_PHASE:
                        actual_classes = np.repeat(args.classes, testing_no_iterations_per_class)
                        # shuffle actual_classes in place
                        np.random.shuffle(actual_classes)
                    else:
                        actual_classes = None
                    aa, final_on_gratings, final_off_gratings = \
                        generate_bar_input(simtime, chunk, N_layer,
                                           angles=args.classes,
                                           actual_angles=actual_classes)
                    aa = np.asarray(aa)
                    actual_classes = aa
                # actual_classes = np.asarray(actual_classes)
                # Begin all the simulation stuff
                sim.setup(timestep=1.0, min_delay=1.0, max_delay=15)
                sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 50)
                sim.set_number_of_neurons_per_core(sim.IF_cond_exp, 256 // 10)
                sim.set_number_of_neurons_per_core(sim.SpikeSourcePoisson, 256 // 13)
                # sim.set_number_of_neurons_per_core(SpikeSourcePoissonVariable, 256 // 16)

                # +-------------------------------------------------------------------+
                # | General Parameters                                                |
                # +-------------------------------------------------------------------+

                # Population parameters
                model = sim.IF_cond_exp
                readout_cell_params = {
                    'cm': 20.0,  # nF
                    'i_offset': 0.0,
                    'tau_m': 20.0,
                    'tau_refrac': DEFAULT_TAU_REFRAC,
                    'tau_syn_E': 5.0,
                    'tau_syn_I': 5.0,
                    'v_reset': -70.0,
                    'v_rest': -70.0,
                    'v_thresh': -50.0,
                    'e_rev_E': 0.,
                    'e_rev_I': -80.
                }
                # Readout set parameters
                tau_minus = args.tau_minus
                tau_plus = args.tau_plus
                a_plus = args.a_plus
                b = args.b
                a_minus = (a_plus * tau_plus * b) / tau_minus
                w_max = args.w_max
                w_min = args.w_min
                p_connect = args.p_connect
                classes = np.asarray(args.classes)
                label_time_offset = np.asarray(args.label_time_offset)
                inhibition_weight_multiplier = 8
                if args.unsupervised:
                    inhibition_weight_multiplier = 8

                # Wiring
                s_max = args.s_max
                sigma_form_forward = args.sigma_form_ff
                sigma_form_lateral = args.sigma_form_lat
                p_form_lateral = args.p_form_lateral
                p_form_forward = args.p_form_forward
                p_elim_dep = args.p_elim_dep * 10.
                p_elim_pot = args.p_elim_pot / 10.
                f_rew = args.f_rew  # 10 ** 4  # Hz

                # store ALL parameters
                readout_sim_params = {  # 'g_max': g_max,
                    'simtime': simtime,
                    'sim_params': sim_params,
                    'f_base': f_base,
                    'readout_cell_params': readout_cell_params,
                    'cell_params': cell_params,
                    'grid': grid,
                    't_record': args.t_record,
                    'path': path,
                    # 's_max': s_max,
                    # 'sigma_form_forward': sigma_form_forward,
                    # 'sigma_form_lateral': sigma_form_lateral,
                    # 'p_form_lateral': p_form_lateral,
                    # 'p_form_forward': p_form_forward,
                    # 'p_elim_dep': p_elim_dep,
                    # 'p_elim_pot': p_elim_pot,
                    # 'f_rew': f_rew,
                    # 'delay': args.delay_distribution,
                    'b': b,
                    't_minus': tau_minus,
                    't_plus': tau_plus,
                    'tau_refrac': DEFAULT_TAU_REFRAC,
                    'a_minus': a_minus,
                    'a_plus': a_plus,
                    # 'input_type': args.input_type,
                    # 'random_partner': args.random_partner,
                    # 'lesion': args.lesion,
                    # 'delay_interval': delay_interval,
                    # 'constant_delay': args.constant_delay,
                    # 'training_angles': training_angles,
                    'argparser': vars(args),
                    'phase': phase,
                    'actual_classes': np.copy(actual_classes)
                }

                stdp_model = sim.STDPMechanism(
                    timing_dependence=sim.SpikePairRule(
                        tau_plus=tau_plus, tau_minus=tau_minus,
                        A_plus=a_plus, A_minus=a_minus),
                    weight_dependence=sim.AdditiveWeightDependence(
                        w_min=w_min, w_max=w_max),
                    weight=w_max
                )

                structure_model_w_stdp = sim.StructuralMechanismSTDP(
                    stdp_model=stdp_model,
                    weight=w_max,
                    delay=1,
                    s_max=s_max,
                    grid=grid,
                    f_rew=f_rew,
                    p_elim_dep=p_elim_dep,
                    p_elim_pot=p_elim_pot,
                    sigma_form_forward=sigma_form_forward,
                    sigma_form_lateral=sigma_form_lateral,
                    p_form_forward=p_form_forward,
                    p_form_lateral=p_form_lateral,
                    is_distance_dependent=False
                )
                # Setup input populations
                source_pop = sim.Population(N_layer,
                                            sim.SpikeSourceArray,
                                            {'spike_times': final_on_gratings},
                                            label="Moving grating on population")

                source_pop_off = sim.Population(N_layer,
                                                sim.SpikeSourceArray,
                                                {'spike_times': final_off_gratings},
                                                label="Moving grating off population")

                noise_pop = sim.Population(N_layer,
                                           sim.SpikeSourcePoisson,
                                           {'rate': f_base,
                                            'start': 0,
                                            'duration': simtime},
                                           label="Noise population")
                # Setup target populations
                target_pop = sim.Population(N_layer, model,
                                            cell_params,
                                            label="TARGET_POP")
                if topology != 1:
                    inh_pop = sim.Population(N_layer, model,
                                             cell_params,
                                             label="INH_POP")
                # Setup readout population
                readout_pop = sim.Population(classes.size, model,
                                             readout_cell_params,
                                             label="READOUT_POP")
                # Setup readout connectivity
                if phase == TRAINING_PHASE:
                    # Generate plastic connectivity from excitatory target to
                    # readout population.
                    # A couple of options are available for learning rules:
                    #   STDP
                    #   STDP + Synaptic Rewiring
                    # -----------------------------------------------------------------
                    # Several options are available for supervision
                    #   Supervised (label provided) w/ weights starting at w_min
                    #   Supervised (label provided) w/ weights starting at w_max
                    #   Unsupervised (label inferred) -- requires lateral inhibition
                    # -----------------------------------------------------------------
                    if args.min_supervised or args.max_supervised:
                        # Supervision provided by an extra Spike Source Array
                        # Generate spikes for each class
                        label_spikes = []
                        for index, cls in np.ndenumerate(classes):
                            # Add the spikes for this class to the list
                            class_slots = np.argwhere(actual_classes.ravel() == cls)
                            # Compute base offsets
                            base_offsets = class_slots * 200  # ms
                            # Repeat bases for as many offsets as you have then
                            # repeat the offsets and add them together to generate all
                            # the spikes times
                            repeated_bases = np.repeat(base_offsets,
                                                       label_time_offset.size)
                            repeated_time_offsets = np.tile(label_time_offset,
                                                              base_offsets.size)
                            spike_times_for_current_class = repeated_bases + \
                                                            repeated_time_offsets
                            label_spikes.append(spike_times_for_current_class)

                        label_pop = sim.Population(classes.size,
                                                   sim.SpikeSourceArray,
                                                   {'spike_times': label_spikes},
                                                   label="Label population")
                        if args.min_supervised:
                            # Sample from target_pop with initial weight of w_min
                            target_readout_projection = sim.Projection(
                                target_pop, readout_pop,
                                sim.FixedProbabilityConnector(p_connect=p_connect,
                                                              weights=w_min),
                                synapse_type=structure_model_w_stdp if args.rewiring else stdp_model,
                                label="min_readout_sampling", receptor_type="excitatory")
                            # Supervision provided by an extra Spike Source Array
                            # with high connection weight
                            label_projection = sim.Projection(
                                label_pop, readout_pop,
                                sim.OneToOneConnector(weights=4 * w_max),
                                label="min_label_projection", receptor_type="excitatory"
                            )
                        elif args.max_supervised:
                            # Sample from target_pop with initial weight of w_max
                            target_readout_projection = sim.Projection(
                                target_pop, readout_pop,
                                sim.FixedProbabilityConnector(p_connect=p_connect,
                                                              weights=w_max),
                                synapse_type=structure_model_w_stdp if args.rewiring else stdp_model,
                                label="max_readout_sampling",
                                receptor_type="excitatory")
                            # Supervision provided by an extra Spike Source Array
                            # with lower connection weight
                            label_projection = sim.Projection(
                                label_pop, readout_pop,
                                sim.OneToOneConnector(weights=8 * w_max),
                                label="max_label_projection",
                                receptor_type="excitatory"
                            )

                    if args.unsupervised:
                        # Sample from target_pop with initial weight of w_max
                        # because there is no extra signal that can cause readout
                        # neurons to fire
                        target_readout_projection = sim.Projection(
                            target_pop, readout_pop,
                            sim.FixedProbabilityConnector(p_connect=p_connect),
                            synapse_type=structure_model_w_stdp if args.rewiring else stdp_model,
                            label="unsupervised_readout_sampling",
                            receptor_type="excitatory")

                    # Setup lateral connections between readout neurons
                    if args.wta_readout or args.unsupervised:
                        # Create a strong inhibitory projection between the readout
                        # neurons
                        # AllToAll connector is behaving weirdly
                        all_to_all_connections = []
                        for i in range(classes.size):
                            for j in range(classes.size):
                                all_to_all_connections.append(
                                    (i, j, inhibition_weight_multiplier*w_max, 1))

                        if args.rewiring:
                            wta_projection = sim.Projection(
                                readout_pop, readout_pop,
                                sim.FixedProbabilityConnector(0.),
                                synapse_type=structure_model_w_stdp,
                                label="wta_strong_inhibition_readout_rewired",
                                receptor_type="inhibitory")
                        else:
                            wta_projection = sim.Projection(
                                readout_pop, readout_pop,
                                sim.FromListConnector(all_to_all_connections),
                                label="wta_strong_inhibition_readout",
                                receptor_type="inhibitory")

                elif phase == TESTING_PHASE:
                    # Extract static connectivity from the training phase
                    # Retrieve readout connectivity (ff and lat)
                    # Always retrieve ff connectivity
                    if snapshots_present:
                        trained_target_readout_connectivity = \
                            target_snapshots[snap_keys[snap]]
                        trained_wta_connectivity = \
                            wta_snapshots[snap_keys[snap]]
                    else:
                        trained_target_readout_connectivity = \
                            readout_training_data['target_readout_projection'][-1]
                        trained_wta_connectivity = \
                            readout_training_data['wta_projection'][-1]

                    target_readout_projection = sim.Projection(
                        target_pop, readout_pop,
                        sim.FromListConnector(trained_target_readout_connectivity),
                        label="target_readout_sampling",
                        receptor_type="excitatory")

                    # Sometimes retrieve lateral connectivity
                    if args.wta_readout or args.unsupervised:
                        wta_projection = sim.Projection(
                            readout_pop, readout_pop,
                            sim.FromListConnector(trained_wta_connectivity),
                            label="wta_strong_inhibition_readout",
                            receptor_type="inhibitory")

                    # We can ignore the label_pop in the testing phase

                else:
                    raise AttributeError(
                        "Phase {} unrecognised. What is  the connectivity for "
                        "readout neurons?".format(phase))

                # record spikes for everything. simulations are so short that this
                # can't hurt, right?
                readout_pop.record(['spikes'])
                target_pop.record(['spikes'])
                if topology != 1:
                    inh_pop.record(['spikes'])

                # The following are to be performed regardless of phase
                # ---------------------------------------------------------------------
                # Setup static connectivity
                trained_ff_on_connectivity = data['ff_connections'][-1]
                trained_ff_off_connectivity = data['ff_off_connections'][-1]
                trained_lat_connectivity = data['lat_connections'][-1]
                trained_noise_connectivity = data['noise_connections'][-1]
                if topology != 1:
                    trained_inh_lat_connectivity = data['inh_connections']
                    trained_exh_lat_connectivity = data['exh_connections']
                    trained_inh_inh_connectivity = data['inh_inh_connections']
                if topology == 3:
                    trained_on_inh_connectivity = data['on_inh_connections']
                    trained_off_inh_connectivity = data['off_inh_connections']
                    trained_noise_inh_connectivity = data['noise_inh_connections']

                ff_projection = sim.Projection(
                    source_pop, target_pop,
                    sim.FromListConnector(trained_ff_on_connectivity),
                    label="plastic_ff_projection"
                )

                ff_off_projection = sim.Projection(
                    source_pop_off, target_pop,
                    sim.FromListConnector(trained_ff_off_connectivity),
                    label="ff_off_projection"
                )

                noise_projection = sim.Projection(
                    noise_pop, target_pop,
                    sim.FromListConnector(trained_noise_connectivity),
                    label="noise_projection"
                )

                lat_projection = sim.Projection(
                    target_pop, target_pop,
                    sim.FromListConnector(trained_lat_connectivity),
                    label="plastic_lat_projection",
                    receptor_type="excitatory"
                )

                if topology != 1:
                    inh_projection = sim.Projection(
                        inh_pop, target_pop,
                        sim.FromListConnector(trained_inh_lat_connectivity),
                        label="plastic_inh_lat_projection",
                        receptor_type="inhibitory"
                    )
                    inh_inh_projection = sim.Projection(
                        inh_pop, inh_pop,
                        sim.FromListConnector(trained_inh_inh_connectivity),
                        label="plastic_inh_inh_projection",
                        receptor_type="inhibitory"
                    )
                    exh_projection = sim.Projection(
                        target_pop, inh_pop,
                        sim.FromListConnector(trained_exh_lat_connectivity),
                        label="plastic_exh_lat_projection",
                        receptor_type="excitatory"
                    )
                if topology == 3:
                    ff_inh_projection = sim.Projection(
                        source_pop, inh_pop,
                        sim.FromListConnector(trained_on_inh_connectivity),
                        label="plastic_ff_inh_projection"
                    )

                    ff_off_inh_projection = sim.Projection(
                        source_pop_off, inh_pop,
                        sim.FromListConnector(trained_off_inh_connectivity),
                        label="ff_off_inh_projection"
                    )

                    noise_inh_projection = sim.Projection(
                        noise_pop, inh_pop,
                        sim.FromListConnector(trained_noise_inh_connectivity),
                        label="noise_inh_projection"
                    )
                target_weights = []
                target_spikes = []
                wta_weights = []
                readout_spikes = []
                inhibitory_spikes = []
                e = None
                print("Starting the sim")
                if phase == TRAINING_PHASE:
                    t_record = args.t_record
                else:
                    t_record = args.testing_t_record

                no_runs = simtime // t_record
                run_duration = t_record
                if no_runs == 0:
                    no_runs = 1
                    run_duration = simtime
                # Try catch around run
                try:
                    for current_run in range(no_runs):
                        print("run", current_run + 1, "of", no_runs)
                        sim.run(run_duration)
                        if phase == TRAINING_PHASE and (args.snapshots or current_run == no_runs - 1):
                            target_weights.append(
                                np.array([
                                    target_readout_projection._get_synaptic_data(True, 'source'),
                                    target_readout_projection._get_synaptic_data(True, 'target'),
                                    target_readout_projection._get_synaptic_data(True, 'weight'),
                                    target_readout_projection._get_synaptic_data(True, 'delay')]).T)
                            target_snapshots[current_run * run_duration] = \
                                np.copy(np.asarray(target_weights[-1]))
                            if args.wta_readout or args.unsupervised:
                                wta_weights.append(
                                    np.array([
                                        wta_projection._get_synaptic_data(True, 'source'),
                                        wta_projection._get_synaptic_data(True, 'target'),
                                        wta_projection._get_synaptic_data(True, 'weight'),
                                        wta_projection._get_synaptic_data(True, 'delay')]).T)
                                wta_snapshots[current_run * run_duration] = \
                                    np.copy(np.asarray(wta_weights[-1]))

                    # Retrieve recordings
                    target_spikes = target_pop.spinnaker_get_data('spikes')
                    if topology != 1:
                        inhibitory_spikes = inh_pop.spinnaker_get_data('spikes')
                    readout_spikes = readout_pop.spinnaker_get_data('spikes')
                    sim.end()

                    target_spikes = np.asarray(target_spikes)
                    inhibitory_spikes = np.asarray(inhibitory_spikes)
                    readout_spikes = np.asarray(readout_spikes)

                    target_spikes_snapshots[snap_keys[snap]] = np.copy(target_spikes)
                    inhibitory_spikes_snapshots[snap_keys[snap]] = np.copy(inhibitory_spikes)
                    readout_spikes_snapshots[snap_keys[snap]] = np.copy(readout_spikes)
                    actual_classes_snapshots[snap_keys[snap]] = np.copy(actual_classes)

                except Exception as e:
                    # Print exception traceback
                    traceback.print_exc()
            end_time = plt.datetime.datetime.now()
            total_time = end_time - start_time
            print("Total time elapsed -- " + str(total_time))


            target_weights = np.asarray(target_weights)
            wta_weights = np.asarray(wta_weights)

            if e:
                current_error = e

            filename = generate_readout_filename(path, phase, args, run, e)

            # This has to be set after all the filename adjustments
            if phase == TRAINING_PHASE:
                current_training_file = filename
            # save testing and training results
            # save training and testing connectivity
            # save actual training and testing classes
            # save whether the file is training or testing
            np.savez_compressed(
                sim_dir + filename,
                # Spiking information
                target_spikes=target_spikes,
                inhibitory_spikes=inhibitory_spikes,
                readout_spikes=readout_spikes,
                # Input file information
                input_path=path,
                input_topology=topology,
                input_sim_params=sim_params,
                # Connection information
                target_readout_projection=target_weights,
                wta_projection=wta_weights,
                # Simulation information
                readout_sim_params=readout_sim_params,
                simtime=simtime,
                total_time=total_time,
                exception=str(e),
                phase=phase,
                phase_name=PHASES_NAMES[phase],
                actual_classes=actual_classes,
                chunk=chunk,  # ms
                label_time_offset=label_time_offset,
                # Rewiring present
                rewiring=args.rewiring,
                # Snapshots present
                snapshots_present=args.snapshots,
                target_snapshots=target_snapshots,
                wta_snapshots=wta_snapshots,
                target_spikes_snapshots=target_spikes_snapshots,
                readout_spikes_snapshots=readout_spikes_snapshots,
                inhibitory_spikes_snapshots=inhibitory_spikes_snapshots,
                actual_classes_snapshots=actual_classes_snapshots
            )
            print("Results in", filename)
