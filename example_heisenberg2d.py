# Python implementation to optimize Neural Quantum States (NQS) from the paper
# _____________________________________________________________________________
# | G. Carleo, and M. Troyer                                                  |
# | Solving the quantum many-body problem with artificial neural-networks     |
# |___________________________________________________________________________|

import pickle

from nqslearn import Heisenberg2D
from nqslearn import NQS
from nqslearn import SRoptimizer


def main():
    # create a Hamiltonian
    H = Heisenberg2D(36, 6, 10, True)
    # create a Neural Quantum State (NQS)
    nqs = NQS(n_visible=36, n_hidden=36)
    # Metropolis-Hastings sampler parameters
    sampler_params = {'n_sweeps': 10000, 'therm_factor': 0.1,
                      'sweep_factor': 1, 'n_flips': 2}
    # create stochastic-reconfiguration optimizer
    optimizer = SRoptimizer(nqs, H, sampler_params=sampler_params,
                            learning_rate=0.001)
    optimizer.run(1000)

    with open('Optimizer.pkl', 'wb') as f:
        pickle.dump(optimizer, f)

main()
