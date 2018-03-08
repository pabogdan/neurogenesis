# Towards simulating neurogenesis on real-time neuromorphic hardware

I am attempting to simulate hippocampal or striatal adult neurogenesis on 
[SpiNNaker](http://apt.cs.manchester.ac.uk/projects/SpiNNaker/). Network is described using the  
[PyNN](http://neuralensemble.org/PyNN/) simulator-independent language. This process entails several phases of potentially 
increasing difficulty (this list is not exhaustive):


1.  Create a SpiNNaker based framework for defining rewiring synapses

2.  [sPyNNaker](http://spinnakermanchester.github.io/) (SpiNNaker's implementation of PyNN) synaptogenesis simulation -- replication of [Simeon Bamford's](http://www.sim.me.uk/) doctoral [thesis](http://www.sim.me.uk/neural/thesis.pdf)

3.  Extend framework to allow for Neurogenesis


