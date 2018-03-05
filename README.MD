# Towards simulating neurogenesis on real-time neuromorphic hardware

I am attempting to simulate hippocampal neurogenesis on 
[SpiNNaker](http://apt.cs.manchester.ac.uk/projects/SpiNNaker/) using the  
[PyNN](http://neuralensemble.org/PyNN/) simulator. This process entails several phases of potentially 
increasing difficulty (this list is not exhaustive):


1.  Create a SpiNNaker based framework for defining rewiring synapses

2.  Software (on host) synaptogenesis simulation -- replication of [Simeon Bamford's](http://www.sim.me.uk/) doctoral [thesis](http://www.sim.me.uk/neural/thesis.pdf)
    - sPyNNaker (SpiNNaker's implementation of PyNN)

3.  Extend framework to allow for Neurogenesis


