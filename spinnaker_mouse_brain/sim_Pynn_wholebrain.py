import sys
import numpy as np
from pylab import *
import h5py

# from pyNN.nest import *
import spynnaker.pyNN as sim
import tables

h5filename = "ptneu_brain.h5"
neuroIDs = None
synTypes = ["AAV", "KERN"]
simTime = 1000.0
syn_scale_factor = 0.01


class BaseFile(object):
    """
    Base class for PyNN File classes.
    """

    def __init__(self, filename, mode='rb'):
        """
        Open a file with the given filename and mode.
        """
        self.name = filename
        self.mode = mode
        dir = os.path.dirname(filename)
        if dir and not os.path.exists(dir):
            try:  # wrapping in try...except block for MPI
                os.makedirs(dir)
            except IOError:
                pass  # we assume that the directory was already created by another MPI node
        try:  # Need this because in parallel, file names are changed
            self.fileobj = open(self.name, mode, DEFAULT_BUFFER_SIZE)
        except Exception as err:
            self.open_error = err

    def __del__(self):
        self.close()

    def _check_open(self):
        if not hasattr(self, 'fileobj'):
            raise self.open_error

    def rename(self, filename):
        self.close()
        try:  # Need this because in parallel, only one node will delete the file with NFS
            os.remove(self.name)
        except Exception:
            pass
        self.name = filename
        self.fileobj = open(self.name, self.mode, DEFAULT_BUFFER_SIZE)

    def write(self, data, metadata):
        """
        Write data and metadata to file. `data` should be a NumPy array,
        `metadata` should be a dictionary.
        """
        raise NotImplementedError

    def read(self):
        """
        Read data from the file and return a NumPy array.
        """
        raise NotImplementedError

    def get_metadata(self):
        """
        Read metadata from the file and return a dict.
        """
        raise NotImplementedError

    def close(self):
        """Close the file."""
        if hasattr(self, 'fileobj'):
            self.fileobj.close()


class HDF5ArrayFile(BaseFile):
    """
    Data are saved as an array within a node named "data". Metadata are
    saved as attributes of this node.
    """

    def __init__(self, filename, mode='r', title="PyNN data file"):
        """
        Open an HDF5 file with the given filename, mode and title.
        """
        self.name = filename
        self.mode = mode
        self.fileobj = tables.openFile(filename, mode=mode, title=title)

    # may not work with old versions of PyTables < 1.3, since they only support numarray, not numpy
    def write(self, data, metadata):
        __doc__ = BaseFile.write.__doc__
        if len(data) > 0:
            try:
                node = self.fileobj.createArray(self.fileobj.root, "data", data)
            except tables.HDF5ExtError as e:
                raise tables.HDF5ExtError("%s. data.shape=%s, metadata=%s" % (e, data.shape, metadata))
            for name, value in metadata.items():
                setattr(node.attrs, name, value)
            self.fileobj.close()

    def read(self):
        __doc__ = BaseFile.read.__doc__
        return self.fileobj.root.data.read()

    def get_metadata(self):
        __doc__ = BaseFile.get_metadata.__doc__
        D = {}
        node = self.fileobj.root.data
        for name in node._v_attrs._f_list():
            D[name] = node.attrs.__getattr__(name)
        return D


# loading the h5 format
# @profile
def load_pointneuron_circuit_PyNN(h5_filename, synTypes=[], returnParas=[], randomize_neurons=False, neuroIDs=None):
    sim.setup(timestep=0.1, min_delay=0.1, max_delay=14.0)

    circuit = {}
    print("Opening h5 datafile " + '\033[96m' + "\"" + h5_filename + "\"" + '\033[0m' + " ... ")
    h5file = h5py.File(h5_filename, 'r')
    h5fileKeys = h5file.keys()
    # get sub ids
    Nh5 = h5file["x"][:].shape[0]
    if neuroIDs == None:
        N = Nh5
        neuroIDs_ = np.array(range(N))
        translatorDict = np.array(range(N))
    else:
        N = neuroIDs.shape[0]
        print("Sorting new ID list")
        neuroIDs_ = np.sort(neuroIDs)
        print("OK")
        translatorDict = -np.ones(Nh5, np.int32)
        translatorDict[neuroIDs_] = np.array(range(N))
    # return custom keys
    for rP in returnParas:
        circuit[rP] = h5file[rP][:][neuroIDs_]
    # subgroup location for synapses
    synapse_dataset_location = h5file["synapse_dataset_location"][:][neuroIDs_]
    # retrieving neuronal and synaptic model if it exists 
    if "neuronModel" in h5fileKeys:
        neuronModel = h5file["neuronModel"].value.decode('UTF-8')
    else:
        neuronModel = "aeif_cond_exp"
    if "synapseModel" in h5fileKeys:
        synapseModel = h5file["synapseModel"].value.decode('UTF-8')
    else:
        synapseModel = "tsodyks2_synapse"

    # creating neurons

    # use default parameters from nest 2.10 in pynn units
    default_parameters = {
        'cm': 0.281,  # Capacitance of the membrane in nF
        'tau_refrac': 0.0,  # Duration of refractory period in ms.
        'v_spike': 0.0,  # Spike detection threshold in mV.
        'v_reset': -60.0,  # Reset value for V_m after a spike. In mV.
        'v_rest': -70.6,  # Resting membrane potential (Leak reversal potential) in mV.
        'tau_m': 9.366666666666667,  # Membrane time constant in ms
        'i_offset': 0.0,  # Offset current in nA
        'a': 4.0,  # Subthreshold adaptation conductance in nS.
        'b': 0.0805,  # Spike-triggered adaptation in nA
        'delta_T': 2.0,  # Slope factor in mV
        'tau_w': 144.0,  # Adaptation time constant in ms
        'v_thresh': -50.4,  # Spike initiation threshold in mV
        'e_rev_E': 0.0,  # Excitatory reversal potential in mV.
        'tau_syn_E': 0.2,  # Rise time of excitatory synaptic conductance in ms (alpha function).
        'e_rev_I': -85.0,  # Inhibitory reversal potential in mV.
        'tau_syn_I': 2.0,  # Rise time of the inhibitory synaptic conductance in ms (alpha function).
    }

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

    cells = sim.Population(N, sim.IF_cond_exp, cell_params)
    circuit["cells"] = cells

    neuroParams = h5file["neuroParams"].keys()
    # create instance for translation

    params = {}

    # convert parameters from nest to pynn units
    if 'C_m' in neuroParams:
        params['cm'] = np.float64(h5file["neuroParams"]['C_m'][:][neuroIDs_]) / 1000.0  # nF -> pF
    if 't_ref' in neuroParams:
        params['tau_refrac'] = np.float64(h5file["neuroParams"]['t_ref'][:][neuroIDs_])
    if 'V_peak' in neuroParams:
        params['v_spike'] = np.float64(h5file["neuroParams"]['V_peak'][:][neuroIDs_])
    if 'V_reset' in neuroParams:
        params['v_reset'] = np.float64(h5file["neuroParams"]['V_reset'][:][neuroIDs_])
    if 'C_m' in neuroParams:
        params['v_rest'] = np.float64(h5file["neuroParams"]['E_L'][:][neuroIDs_])
    if 'C_L' in neuroParams and 'g_L' in neuroParams:
        params['tau_m'] = np.float64(h5file["neuroParams"]['C_L'][:][neuroIDs_]) / np.float64(
            h5file["neuroParams"]['g_L'][:][neuroIDs_])
    if 'I_e' in neuroParams:
        params['i_offset'] = np.float64(h5file["neuroParams"]['I_e'][:][neuroIDs_]) / 1000.0  # nA -> pA
    if 'a' in neuroParams:
        params['a'] = np.float64(h5file["neuroParams"]['a'][:][neuroIDs_])
    if 'b' in neuroParams:
        params['b'] = np.float64(h5file["neuroParams"]['b'][:][neuroIDs_]) / 1000.0  # nA -> pA.
    if 'Delta_T' in neuroParams:
        params['delta_T'] = np.float64(h5file["neuroParams"]['Delta_T'][:][neuroIDs_])
    if 'tau_w' in neuroParams:
        params['tau_w'] = np.float64(h5file["neuroParams"]['tau_w'][:][neuroIDs_])
    if 'V_th' in neuroParams:
        params['v_thresh'] = np.float64(h5file["neuroParams"]['V_th'][:][neuroIDs_])
    if 'E_ex' in neuroParams:
        params['e_rev_E'] = np.float64(h5file["neuroParams"]['E_ex'][:][neuroIDs_])
    if 'tau_syn_ex' in neuroParams:
        params['tau_syn_E'] = np.float64(h5file["neuroParams"]['tau_syn_ex'][:][neuroIDs_])
    if 'E_in' in neuroParams:
        params['e_rev_I'] = np.float64(h5file["neuroParams"]['E_in'][:][neuroIDs_])
    if 'tau_syn_in' in neuroParams:
        params['tau_syn_I'] = np.float64(h5file["neuroParams"]['tau_syn_in'][:][neuroIDs_])

    # set converterd neuron parameters
    for p in params:
        cells.tset(p, params[p])

    totNumSyns = 0
    for sT in synTypes:
        print(sT, "synapse counting:")
        for i, gid in enumerate(neuroIDs_):
            if i % 100 == 0:
                print 100.0 * float(i) / float(N)

            for dir_ in ["IN", "OUT"]:
                hasSyns = True
                try:
                    synT = translatorDict[h5file["syngroup_" + str(synapse_dataset_location[gid])][
                                              "syn_" + sT + "_" + dir_ + "_T_" + str(gid)][:]]
                    validSynapses = np.where(synT != -1)[0]

                    # only take synapses that target the current circuit
                    synT = synT[validSynapses]
                    totNumSyns += synT.shape[0]
                except:
                    hasSyns = False

    totNumSyns_offset = 0
    connections = np.zeros([totNumSyns, 4])
    for sT in synTypes:
        print(sT, "synapse creation:")
        for i, gid in enumerate(neuroIDs_):
            if i % 100 == 0:
                print 100.0 * float(i) / float(N)

            for dir_ in ["IN", "OUT"]:
                hasSyns = True
                currentNumSyns = 0
                try:
                    synT = translatorDict[h5file["syngroup_" + str(synapse_dataset_location[gid])][
                                              "syn_" + sT + "_" + dir_ + "_T_" + str(gid)][:]]
                    validSynapses = np.where(synT != -1)[0]

                    syn = h5file["syngroup_" + str(synapse_dataset_location[gid])][
                              "syn_" + sT + "_" + dir_ + "_" + str(gid)][:, :]
                    # only take synapses that target the current circuit
                    synT = synT[validSynapses]
                    syn = syn[:, validSynapses]
                    currentNumSyns = synT.shape[0]
                except:
                    hasSyns = False
                if hasSyns:
                    synT = np.float64(np.int64(synT[:]))
                    syn = np.float64(syn[:])
                    if type(synT) == np.float64:
                        synT = np.array([synT]);
                        # syn  = np.array([syn])
                    if dir_ == "IN":

                        for isyn in range(syn.shape[1]):
                            # translate hardcoded: weight has a different unit
                            connections[totNumSyns_offset + isyn] = (
                                np.int64(synT)[isyn], i, syn[1, isyn] / 1000. * syn_scale_factor, syn[0, isyn])
                        totNumSyns_offset += currentNumSyns
                    elif dir_ == "OUT":
                        # connections = np.zeros([syn.shape[1],7])
                        for isyn in range(syn.shape[1]):
                            # translate hardcoded: weight has a different unit
                            connections[totNumSyns_offset + isyn] = (
                                i, np.int64(synT[isyn]), syn[1, isyn] / 1000. * syn_scale_factor, syn[0, isyn])
                        totNumSyns_offset += currentNumSyns

    if len(connections) > 0:
        print("Create connector")
        connector = sim.FromListConnector(connections[:totNumSyns_offset])
        print("Create projection")
        projection = sim.Projection(cells, cells, connector)

    print("Total number of neurons:", N)
    print("Total number of synapses:", totNumSyns)
    h5file.close()
    return circuit


circuit = load_pointneuron_circuit_PyNN(h5filename, synTypes=synTypes, neuroIDs=neuroIDs,
                                        returnParas=["IO_ids", "IO_x", "IO_y"])
circuit["cells"].record()
circuit["cells"].set('delta_T', 0.1)
circuit["cells"][list(np.where(circuit["IO_ids"] == 10)[0])].set('i_offset', 600.0 * 0.001)

print("Numer of simulated neurons", len(list(np.where(circuit["IO_ids"] == 10)[0])))
print("Start simulation")
sim.run(simTime)
print("Finished simulation")

## show statistic
print("MeanSpikeCount", circuit["cells"].meanSpikeCount())

## store results to disk
from pyNN.recording.files import HDF5ArrayFile

h5file = HDF5ArrayFile("pynn_results.h5", "w")
circuit["cells"].printSpikes(h5file)
h5file.close()

sim.end()