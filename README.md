# Deep NN as a Monte Carlo aid for 2HiggsDoublet
Project uses neural networks to fit the parameter space of the two Higgs doublet model https://arxiv.org/abs/1612.01309 to aid in Monte Carlo exploration. This is then minimised via a particle swarm algorithm using a chi squared measure. The scripts then suggest a series of point around the found minima that should be fed into the original scanning algorithm.


Uses algortihms:

	- Deep Neural Networks
	
	- Differential Evolution Minimisation
	
	- Particle Swarm Minimisation

Test case consists of using the same procedure to find the global minimum of the 2D Ackley function (see https://www.sfu.ca/~ssurjano/ackley.html).


Generalisation:
	- Can be used for any Monte Carlo exploration. Will write further modularity at a later date.

![aass](Ackley_Plots/Ackley_3D.pdf)

<!-- ![](Ackley_Plots/DE_Selection.gif) -->
