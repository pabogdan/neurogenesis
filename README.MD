# Towards simulating neurogenesis on real-time neuromorphic hardware

The goal at the outset of my PhD was to simulate hippocampal or striatal 
adult neurogenesis on 
[SpiNNaker](http://apt.cs.manchester.ac.uk/projects/SpiNNaker/). 
The network would have been described using the  
[PyNN](http://neuralensemble.org/PyNN/) simulator-independent language. 
This process entails several phases of potentially 
increasing difficulty (this list is not exhaustive):


1.  Create a SpiNNaker based framework for defining rewiring synapses

2.  [sPyNNaker](http://spinnakermanchester.github.io/) 
(SpiNNaker's implementation of PyNN) synaptogenesis simulation -- 
replication of [Simeon Bamford's](http://www.sim.me.uk/) doctoral 
[thesis](http://www.sim.me.uk/neural/thesis.pdf)

3.  Extend framework to allow for Neurogenesis

At the time of writing, neurogenesis is not realised. 
However, extensive experiments have been performed using the synaptic 
rewiring model implemented on SpiNNaker. For more details, see below.

This repository is a collection of different experiments, mostly revolving 
around synaptic rewiring ([synaptogenesis](synaptogenesis)) with associated 
analysis scripts([simulation_statistics](simulation_statistics)), sundry 
experiments ([sundry](sundry)) and ipynb notebooks ([notebooks](notebooks)).

# Using this research

I have set up a [Docker](https://hub.docker.com/r/pabogdan/spinnaker) 
container to easily experiment with structural plasticity
on SpiNNaker. It can be found under the tag `structural_plasticity` for the 
most up-to-date development version of the tools, while 
`structural_plasticity_5.1.0` contains the most recent (to the time of writing) 
release of the SpiNNaker tools. 
These containers include 3 iPython Notebooks which immediately run the most 
common experiments presented in my thesis which use structural plasticity: 

1.  Topographic map formation
1.  MNIST classification
1.  Motion detection 


# Publications

1.  Student Conference Proceedings

	**Bogdan PA**, Furber SB (2017) "Neurogenesis on SpiNNaker". 
	Proceedings of the 1st HBP Student Conference - 
	Transdisciplinary Research Linking Neuroscience, Brain Medicine and 
	Computer Science, 75-77, ISBN: 978-2-88945-421-1 
	https://www.frontiersin.org/books/1st_HBP_Student_Conference_-_Transdisciplinary_Research_Linking_Neuroscience_Brain_Medicine_and_Com/1532

1.  Conference Proceedings

	**Bogdan PA**, Pineda-García G, Davidson S, Hopkins M, James R and 
	Furber SB (2019), "Event-based computation: Unsupervised elementary motion 
	decomposition". Proceedings of Emerging Technology Conference
  
1.  Journal Articles

	**Bogdan PA**, Rowley AGD, Rhodes O and Furber SB (2018) 
	"Structural Plasticity on the SpiNNaker Many-Core Neuromorphic System". 
	Front. Neurosci. 12:434.  http://dx.doi.org/10.3389/fnins.2018.00434

	Hopkins M, Pineda-García G, **Bogdan PA**, Furber SB (2018) 
	"Spiking neural networks for computer vision". Interface Focus 8 20180007.  
	http://dx.doi.org/10.1098/rsfs.2018.0007
   
1. Thesis

    **Bogdan, PA** (2020). Structural Plasticity on SpiNNaker 
    (The University of Manchester). https://doi.org/10.13140/RG.2.2.33591.06568 
       
1.  Data repositories

	**Bogdan PA**, Rhodes O, Rowley AGD, Furber SB (2018) 
	"Structural plasticity on the SpiNNaker many-core neuromorphic system", 
	Mendeley Data, v1. http://dx.doi.org/10.17632/xfp84r5hb7.1

	**Bogdan PA**, Furber SB, Pineda-Garcia G, Hopkins M (2018) 
	“Spiking Neural Networks for Computer Vision”, Mendeley Data, v1. 
	http://dx.doi.org/10.17632/84pvnm3rj3.1

	**Bogdan PA**, Pineda-García G, Davidson S, Hopkins M, James R and 
	Furber SB (2019), 
	"Supporting Data and Software for Event-based computation: 
	Unsupervised elementary motion decomposition", Mendeley Data, v1 
	http://dx.doi.org/10.17632/wpzxh93vhx.1
  
1.  Posters

	**Bogdan PA**, Furber SB (2018) "Simulating synaptic rewiring on SpiNNaker". 
	figshare. Poster. https://doi.org/10.6084/m9.figshare.6025460.v1
    
    **Bogdan PA**, Furber SB (2018) "Structural plasticity for motion 
    detection on SpiNNaker using PyNN". figshare. Poster. 
    https://doi.org/10.6084/m9.figshare.6949091.v1
