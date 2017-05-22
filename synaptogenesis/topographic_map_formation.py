"""
Test for topographic map formation using STDP and synaptic rewiring.

http://hdl.handle.net/1842/3997
"""
# Imports
import numpy as np
import pylab
try:
    import pyNN.spiNNaker as sim
except Exception as e:
    import spynnaker.pyNN as sim


# SpiNNaker setup
sim.setup(timestep=1.0, min_delay=1.0, max_delay=10.0)

# +-------------------------------------------------------------------+
# | Function definitions                                              |
# +-------------------------------------------------------------------+

# Periodic boundaries
# https://github.com/pabogdan/neurogenesis/blob/master/notebooks/neurogenesis-in-numbers/Periodic%20boundaries.ipynb
def distance(x0, x1, grid, type='euclidian'):
    x0 = np.asarray(x0)
    x1 = np.asarray(x1)
    delta = np.abs(x0 - x1)
    delta = np.where(delta > grid * .5, delta - grid, delta)

    if type == 'manhattan':
        return np.abs(delta).sum(axis=-1)
    return np.sqrt((delta ** 2).sum(axis=-1))

# Poisson spike source as from list spike source
# https://github.com/project-rig/pynn_spinnaker_bcpnn/blob/master/examples/modular_attractor/network.py#L115-L148
def poisson_generator(rate, t_start, t_stop):
    n = (t_stop - t_start) / 1000.0 * rate
    number = np.ceil(n + 3 * np.sqrt(n))
    if number < 100:
        number = min(5 + np.ceil(2 * n),100)

    if number > 0:
        isi = np.random.exponential(1.0/rate, number)*1000.0
        if number > 1:
            spikes = np.add.accumulate(isi)
        else:
            spikes = isi
    else:
        spikes = np.array([])

    spikes += t_start
    i = np.searchsorted(spikes, t_stop)

    extra_spikes = []
    if len(spikes) == i:
        # ISI buf overrun

        t_last = spikes[-1] + np.random.exponential(1.0 / rate, 1)[0] * 1000.0

        while (t_last<t_stop):
            extra_spikes.append(t_last)
            t_last += np.random.exponential(1.0 / rate, 1)[0] * 1000.0

            spikes = np.concatenate((spikes, extra_spikes))
    else:
        spikes = np.resize(spikes,(i,))

    # Return spike times, rounded to millisecond boundaries
    return [round(x) for x in spikes]

# +-------------------------------------------------------------------+
# | General Parameters                                                |
# +-------------------------------------------------------------------+

# Population parameters
model = sim.IF_curr_exp

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

# +-------------------------------------------------------------------+
# | Rewiring Parameters                                               |
# +-------------------------------------------------------------------+
no_iterations = 3000000

# Wiring
n = 16
N_layer = n ** 2
S = (n, n)

s = (n//2, n//2)
s_max = 32
sigma_form_forward = 2.5
sigma_form_lateral = 1
p_form_lateral = 1
p_form_forward = 0.16
p_elim_dep = 0.0245
p_elim_pot = 1.36 * np.e ** -4
f_rew = 10 ** 4 #Hz

# Inputs
f_mean = 20 #Hz
f_base = 5 #Hz
f_peak = 152.8 #Hz
sigma_stim = 2
t_stim = 0.02 #s

# Membrane
v_rest = -70 #mV
e_ext = 0 #V
v_thr = -54 #mV
g_max = 0.2
tau_m = 20 #ms
tau_ex = 5 #ms

# STDP
a_plus = 0.1
b = 1.2
tau_plus = 20 #ms
tau_minus = 64 #ms

# +-------------------------------------------------------------------+
# | Initial network setup                                             |
# +-------------------------------------------------------------------+

# Neuron populations
source_pop = sim.Population(N_layer, model, cell_params, label="SOURCE_POP")
target_pop = sim.Population(N_layer, model, cell_params, label="TARGET_POP")

# Connections


# Need to setup the moving input

