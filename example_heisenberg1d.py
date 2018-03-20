# Python implementation to optimize Neural Quantum States (NQS) from the paper
# _____________________________________________________________________________
# | G. Carleo, and M. Troyer                                                  |
# | Solving the quantum many-body problem with artificial neural-networks     |
# |___________________________________________________________________________|

import pickle

from nqslearn import Heisenberg1D
from nqslearn import NQS
from nqslearn import SRoptimizer


def main():
    # create a Hamiltonian
    H = Heisenberg1D(40, 10, True)
    # create a Neural Quantum State (NQS)
    nqs = NQS()
    # Metropolis-Hastings sampler parameters
    sampler_params = {'n_sweeps': 1000, 'therm_factor': 0.,
                      'sweep_factor': 1, 'n_flips': 1}
    # create stochastic-reconfiguration optimizer
    optimizer = SRoptimizer(nqs, H, sampler_params=sampler_params,
                            learning_rate=0.005)
    optimizer.run(100)

    with open('Optimizer.pkl', 'wb') as f:
        pickle.dump(optimizer, f)

main()
