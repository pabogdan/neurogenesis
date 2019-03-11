# Towards simulating neurogenesis on real-time neuromorphic hardware

I have been attempting to simulate hippocampal or striatal adult neurogenesis on 
[SpiNNaker](http://apt.cs.manchester.ac.uk/projects/SpiNNaker/). The network would have been described using the  [PyNN](http://neuralensemble.org/PyNN/) simulator-independent language. This process entails several phases of potentially 
increasing difficulty (this list is not exhaustive):


1.  Create a SpiNNaker based framework for defining rewiring synapses

2.  [sPyNNaker](http://spinnakermanchester.github.io/) (SpiNNaker's implementation of PyNN) synaptogenesis simulation -- replication of [Simeon Bamford's](http://www.sim.me.uk/) doctoral [thesis](http://www.sim.me.uk/neural/thesis.pdf)

3.  Extend framework to allow for Neurogenesis

At the time of writing, neurogenesis is not realised. However, extensive experiments have been performed using the synaptic rewiring model implemented on SpiNNaker. For more details, see below.

# Publications

1.  Student Conference Proceedings

	**Bogdan PA**, Furber SB (2017) "Neurogenesis on SpiNNaker". Proceedings of the 1st HBP Student Conference - Transdisciplinary Research Linking Neuroscience, Brain Medicine and Computer Science, 75-77, ISBN: 978-2-88945-421-1

1.  Conference Proceedings

	**Bogdan PA**, Pineda-García G, Davidson S, Hopkins M, James R and Furber SB *in press* (2019), "Event-based computation: Unsupervised elementary motion decomposition". Proceedings of Emerging Technology Conference
  
1.  Journal Articles

	**Bogdan PA**, Rowley AGD, Rhodes O and Furber SB (2018) "Structural Plasticity on the SpiNNaker Many-Core Neuromorphic System". Front. Neurosci. 12:434. doi: 10.3389/fnins.2018.00434


	Hopkins M, Pineda-García G, **Bogdan PA**, Furber SB (2018) "Spiking neural networks for computer vision". Interface Focus 8 20180007. doi: 10.1098/rsfs.2018.0007
    
1.  Data repositories

	**Bogdan PA**, Rhodes O, Rowley AGD, Furber SB (2018) "Structural plasticity on the SpiNNaker many-core neuromorphic system", Mendeley Data, v1. http://dx.doi.org/10.17632/xfp84r5hb7.1

	**Bogdan PA**, Furber SB, Pineda-Garcia G, Hopkins M (2018) “Spiking Neural Networks for Computer Vision”, Mendeley Data, v1. http://dx.doi.org/10.17632/84pvnm3rj3.1

	**Bogdan PA**, Pineda-García G, Davidson S, Hopkins M, James R and Furber SB (2019), "Supporting Data and Software for Event-based computation: Unsupervised elementary motion decomposition", Mendeley Data, v1 http://dx.doi.org/10.17632/wpzxh93vhx.1
  
1.  Posters

	**Bogdan PA**, Furber SB (2018) "Simulating synaptic rewiring on SpiNNaker". figshare. Poster. https://doi.org/10.6084/m9.figshare.6025460.v1
    
    **Bogdan PA**, Furber SB (2018) "Structural plasticity for motion detection on SpiNNaker using PyNN". figshare. Poster. https://doi.org/10.6084/m9.figshare.6949091.v1
