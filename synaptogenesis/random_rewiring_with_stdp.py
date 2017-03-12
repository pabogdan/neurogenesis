import pylab

from pacman.model.constraints.placer_constraints.placer_chip_and_core_constraint import PlacerChipAndCoreConstraint

try:
    import pyNN.spiNNaker as sim
except Exception as e:
    import spynnaker.pyNN as sim

# SpiNNaker setup
sim.setup(timestep=1.0, min_delay=1.0, max_delay=10.0)
sim.set_number_of_neurons_per_core("IF_curr_exp", 50)

# Population parameters
model = sim.IF_curr_exp

n_atoms = 25

cell_params = {'cm': 0.25,
               'i_offset': 0.0,
               'tau_m': 20.0,
               'tau_refrac': 2.0,
               'tau_syn_E': 5.0,
               'tau_syn_I': 5.0,
               'v_reset': -70.0,
               'v_rest': -65.0,
               'v_thresh': -50.0
               }

# Define layers
source = sim.Population(n_atoms, model, cell_params, label='source_layer')
target = sim.Population(n_atoms, model, cell_params, label='target_layer')

target.set_constraint(PlacerChipAndCoreConstraint(0,1))
# Define learning
# Plastic Connections between pre_pop and post_pop
stdp_model = sim.STDPMechanism(
    timing_dependence=sim.SpikePairRule(tau_plus=20., tau_minus=20.0,
                                        nearest=True),
    weight_dependence=sim.AdditiveWeightDependence(w_min=0, w_max=0.9,
                                                   A_plus=0.02, A_minus=0.02)
)
structure_model_w_stdp = sim.StructuralMechanism(stdp_model=stdp_model)
# Define connections
plastic_projection = sim.Projection(
    source, source, sim.FixedNumberPreConnector(32),
    synapse_dynamics=sim.SynapseDynamics(slow=structure_model_w_stdp), label="plastic_projection"
)
# Add a sprinkle of Poisson noise

# Add some spatial pattern to be repeated

# Start Simulation

# Recover provenance

# End simulation