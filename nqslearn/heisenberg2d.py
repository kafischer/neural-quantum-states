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


class Heisenberg2D(Hamiltonian):
    """
    Class represents the Hamiltonian of the 1D Heisenberg model with
    transverse field h_x and exchange J_z=1
    """

    def __init__(self, n_spins, lattice, j_z, periodic):
        super().__init__()
        if n_spins != lattice ** 2:
            raise ValueError('N_spins not compatible with lattice size.')
        self.l = lattice
        self.min_flip = 2
        self.n_spins = n_spins
        self.j_z = j_z
        self.periodic = periodic
        self.nearest_neighbors, self.bonds =  self.find_nearest_neighbors()
        
    def min_flips(self):
        return self.min_flip

    def num_spins(self):
        return self.n_spins

    def field(self):
        return self.j_z

    def is_periodic(self):
        return self.periodic
    
    def pbc_h(self, nn, s):
        if s % self.l == 0 and nn == s-1:
            # s is at left side of lattice; return rightmost element
            return s+self.l-1
        elif (s+1) % self.l == 0 and nn == (s+1):
            # s is at right side of lattice; return leftmost element
            return s-self.l+1
        else:
            return nn  # s is in middle of lattice; return element to left

    def pbc_v_lower(self, nn):
        if nn < self.l:
            return self.l*(self.l-1) + nn
        else:
            return nn - self.l

    def pbc_v_higher(self, nn):
        if self.l*(self.l-1) <= nn <= self.n_spins:
            return nn - self.l*(self.l-1)
        else:
            return nn + self.l
    
    def find_nearest_neighbors(self):
        nearest_neighbors = np.zeros((self.n_spins, 4))
        bonds = []
        for i in range(self.n_spins):
            nearest_neighbors[i][0] = self.pbc_h(i-1, i)
            nearest_neighbors[i][1] = self.pbc_h(i+1, i)
            nearest_neighbors[i][2] = self.pbc_v_lower(i)
            nearest_neighbors[i][3] = self.pbc_v_higher(i)
 
        for i in range(self.n_spins):
            for k in range(4):
                j = int(nearest_neighbors[i][k])
                if i < j:
                    bonds.append((i, j))
        return nearest_neighbors, bonds
                                        
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
        for i in range(len(self.bonds)): 
            matrix_elements[0] += state[self.bonds[i][0]] * \
                                  state[self.bonds[i][1]]

        matrix_elements[0] *= self.j_z

        # look for spin flips
        for i in range(len(self.bonds)):
            si = self.bonds[i][0]
            sj = self.bonds[i][1]
            
            if state[si] != state[sj]:
                matrix_elements.append(-2)
                spin_flip_transitions.append([si, sj])

        return matrix_elements, spin_flip_transitions
