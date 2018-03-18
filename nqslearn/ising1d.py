# Python implantation of metropolis-hastings sampler for quantum states
# The original programs we have modified require the following notice

############################ COPYRIGHT NOTICE #################################
#
# Code provided by G. Carleo and M. Troyer, written by G. Carleo, December 2016
#
# Permission is granted for anyone to copy, use, modify, or distribute the
# accompanying programs and documents for any purpose, provided this copyright
# notice is retained and prominently displayed, along with a complete citation
# of the published version of the paper:
# _____________________________________________________________________________
# | G. Carleo, and M. Troyer                                                  |
# | Solving the quantum many-body problem with artificial neural-networks     |
# |___________________________________________________________________________|
#
# The programs and documents are distributed without any warranty, express or
# implied.
#
# These programs were written for research purposes only, and are meant to
# demonstrate and reproduce the main results obtained in the paper.
#
# All use of these programs is entirely at the user's own risk.
#
###############################################################################

import numpy as np
from .hamiltonian import Hamiltonian


class Ising1D(Hamiltonian):
    """
    Class represents the Hamiltonian of the 1D ising model with
    transverse field h_x and exchange J_z=1
    """

    def __init__(self, n_spins, h_x, periodic):
        super().__init__()
        self.min_flip = 1
        self.n_spins = n_spins
        self.h_x = h_x
        self.periodic = periodic
        self.matrix_elements = np.zeros((n_spins + 1)) - self.h_x
        self.spin_flip_transitions = [[]] + [[i] for i in range(self.n_spins)]

    # ===================================================================
    # Access basic parameters of the Hamiltonian
    # ===================================================================
    def min_flips(self):
        return self.min_flip

    def num_spins(self):
        return self.n_spins

    def field(self):
        return self.h_x

    def is_periodic(self):
        return self.periodic

    def find_matrix_elements(self, state):
        """
        inputs
            state: list of integers, with each corresponding to quantum number
        returns:
            transitions: list of states s such that <s|H|state> is nonzero.
                s are represented as a list of integers corresponding to which
                quantum variables got swapped
            matrix_elements: complex list <s|H|state> for each s in transitions
        """
        matrix_elements = self.matrix_elements

        # computing interaction part Sz*Sz
        matrix_elements[0] = 0.0
        for i in range(self.n_spins - 1):
            matrix_elements[0] -= state[i] * state[i + 1]
        if self.periodic:
            matrix_elements[0] -= state[self.n_spins - 1] * state[0]

        return matrix_elements, self.spin_flip_transitions
