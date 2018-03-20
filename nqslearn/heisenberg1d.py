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

from .hamiltonian import Hamiltonian


class Heisenberg1D(Hamiltonian):
    """
    Class represents the Hamiltonian of the 1D Heisenberg model with
    transverse exchange J_z
    """

    def __init__(self, n_spins, j_z, periodic):
        super().__init__()
        self.min_flip = 2
        self.n_spins = n_spins
        self.j_z = j_z
        self.periodic = periodic

    def min_flips(self):
        return self.min_flip

    def num_spins(self):
        return self.n_spins

    def field(self):
        return self.j_z

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
        matrix_elements = [0]
        spin_flip_transitions = [[]]

        # computing interaction part Sz*Sz
        for i in range(self.n_spins - 1):
            matrix_elements[0] += state[i] * state[i + 1]
        if self.periodic:
            matrix_elements[0] += state[self.n_spins - 1] * state[0]
        matrix_elements[0] *= self.j_z

        for i in range(self.n_spins - 1):
            if state[i] != state[i+1]:
                matrix_elements.append(-2)
                spin_flip_transitions.append([i, i+1])
        if self.periodic:
            if state[0] != state[self.n_spins - 1]:
                matrix_elements.append(-2)
                spin_flip_transitions.append([self.n_spins - 1, 0])

        return matrix_elements, spin_flip_transitions
