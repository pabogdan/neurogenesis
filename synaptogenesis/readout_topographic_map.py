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
from enum import Enum

import spynnaker7.pyNN as sim

# SpiNNaker setup
start_time = plt.datetime.datetime.now()

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
    data = np.load(filename)

    # Read all required parameters
    sim_params = np.array(data['sim_params']).ravel()[0]
    cell_params = sim_params['cell_params']
    grid = sim_params['grid']
    topology = data['topology']
    N_layer = grid[0] * grid[1]  # Total number of neurons
    n = np.sqrt(N_layer)
    f_base = sim_params['f_base']  # Hz
    if (n == 32 and not args.mnist) or (n == 28 and args.mnist):
        chunk = 200  # ms
    elif n == 64:
        chunk = 400  # ms
    else:
        raise AttributeError("What chunk to use for the specified grid size?")
    simtime = args.no_iterations
    for phase in PHASES:
        print("Phase ", phase)

        # Generate the input (Moving bar or MNIST)
        actual_classes = []
        if not args.mnist:
            aa, final_on_gratings, final_off_gratings = \
                generate_bar_input(args.no_iterations, chunk, N_layer,
                                   angles=args.classes)
            actual_classes.append(aa)

        # Begin all the simulation stuff
        sim.setup(timestep=1.0, min_delay=1.0, max_delay=10)
        sim.set_number_of_neurons_per_core(
            "IF_curr_exp", 50)
        sim.set_number_of_neurons_per_core(
            "IF_cond_exp", 256 // 10)
        sim.set_number_of_neurons_per_core(
            "SpikeSourcePoisson", 256 // 13)
        sim.set_number_of_neurons_per_core(
            "SpikeSourcePoissonVariable", 256 // 13)

        # +-------------------------------------------------------------------+
        # | General Parameters                                                |
        # +-------------------------------------------------------------------+

        # Population parameters
        model = sim.IF_cond_exp
        readout_cell_params = {
            'cm': 10.0,  # nF
            'i_offset': 0.0,
            'tau_m': 20.0,
            'tau_refrac': args.tau_refrac,
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
        a_minus = args.a_minus
        a_plus = args.a_plus
        w_max = args.w_max
        w_min = args.w_min

        stdp_model = sim.STDPMechanism(
            timing_dependence=sim.SpikePairRule(tau_plus=tau_plus,
                                                tau_minus=tau_minus),
            weight_dependence=sim.AdditiveWeightDependence(w_min=w_min,
                                                           w_max=w_max,
                                                           A_plus=a_plus,
                                                           A_minus=a_minus)
        )
        # Setup input populations
        source_pop = sim.Population(N_layer,
                                    sim.SpikeSourceArray,
                                    {'spike_times': final_on_gratings
                                     }, label="Moving grating on population")

        source_pop_off = sim.Population(N_layer,
                                        sim.SpikeSourceArray,
                                        {'spike_times': final_off_gratings
                                         },
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
        if args.topology != 1:
            inh_pop = sim.Population(N_layer, model,
                                     cell_params,
                                     label="INH_POP")
        # Setup readout population
        readout_pop = sim.Population(len(args.classes), model,
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
            pass
        elif phase == TESTING_PHASE:
            # Extract static connectivity from the training phase
            pass
        else:
            raise AttributeError(
                "Phase {} unrecognised. What is  the connectivity for "
                "readout neurons?".format(phase))
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
            target="inhibitory" if args.lateral_inhibition else "excitatory"
        )

        if topology != 1:
            inh_projection = sim.Projection(
                inh_pop, target_pop,
                sim.FromListConnector(trained_inh_lat_connectivity),
                label="plastic_inh_lat_projection",
                target="inhibitory"
            )
            inh_inh_projection = sim.Projection(
                inh_pop, inh_pop,
                sim.FromListConnector(trained_inh_inh_connectivity),
                label="plastic_inh_inh_projection",
                target="inhibitory"
            )
            exh_projection = sim.Projection(
                target_pop, inh_pop,
                sim.FromListConnector(trained_exh_lat_connectivity),
                label="plastic_exh_lat_projection",
                target="excitatory"
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
