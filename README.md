# LipschitzBoundedEquilibriumNetworks

This code is was used to generate the experiments in the paper "Lipschitz bounded equilibrium networks" https://arxiv.org/abs/2010.01732. 

This work extends the work of [Winston and Kolter](https://arxiv.org/abs/2010.01732) in a number of interesting ways:  
- we propose less restrictive conditions on the weight matrices that still guarantee the existence and uniqueness of fixed points
- We show how a Lipschitz bound on the input/output map can be easily encorporated
- we show connections to neural ODEs and convex optimization problems

The code is based on the code originally written by Winston and Kolter https://github.com/locuslab/monotone_op_net
and provides a more flexible model class and the ability to enforce lipschitz bounds on the networks.
