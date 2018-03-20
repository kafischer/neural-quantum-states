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


class NQS(object):
    """Represents a quantum state using a restricted Boltzmann machine."""

    def __init__(self, a=None, b=None, W=None,
                 n_hidden=40, n_visible=40, init_weight=0.001):
        """
        Initialize a state by entering a,b,W (np arrays) and dimensions (ints)
        """

        if a:
            self.a = a.astype(dtype=complex)
        else:
            self.a = init_weight * (np.random.rand(n_visible, 1)-0.5)
        if b:
            self.b = b.astype(dtype=complex)
        else:
            self.b = init_weight * (np.random.rand(n_visible, 1) - 0.5)
        if W:
            self.W = W
        else:
            self.W = init_weight * (np.random.rand(n_visible, n_hidden)-0.5)

        self.a = self.a.astype(dtype=complex)
        self.b = self.b.astype(dtype=complex)
        self.W = self.W.astype(dtype=complex)
        self.n_hidden = n_hidden
        self.n_visible = n_visible
        self.thetas = b  # will become b + W*state

    def from_file(self, cls, filename):
        """Initializes an NQS wavefunction from a file."""
        with open(filename, 'r') as f:
            nv = int(f.readline())
            nh = int(f.readline())
            a = [self._string_to_complex(f.readline()) for _ in range(nv)]
            b = [self._string_to_complex(f.readline()) for _ in range(nh)]
            W = [[self._string_to_complex(f.readline()) for _ in range(nh)]
                 for _ in range(nv)]
        a = np.array(a).reshape(nv, 1)
        b = np.array(b).reshape(nh, 1)
        W = np.array(W)
        print('Loaded visible biases of shape: ', a.shape)
        print('Loaded hidden biases of shape: ', b.shape)
        print('Loaded weights of shape: ', W.shape)
        return cls(a, b, W, nh, nv)

    def update_params(self, update_vals):
        """
        updates the biases and weights of the network
        """
        da = update_vals[:self.n_visible]
        db = update_vals[self.n_visible:self.n_visible+self.n_hidden]
        dW = update_vals[self.n_visible+self.n_hidden:]
        dW = dW.reshape((self.n_visible, self.n_hidden))
        self.a -= da
        self.b -= db
        self.W -= dW
        self.thetas = self.b

    def operator_derivatives_of_current_lt(self, state):
        """
        compute parameter derivatives of the wave function.
        see Carleo and Troyer appendix C
        """
        s = np.array(state, dtype=float).reshape((self.n_visible, 1))
        th = np.tanh(self.thetas)
        return s, th, np.dot(s, th)

    def amplitude_ratio(self, state, flips_to_new_state):
        """
        computes the ratio of wave function amplitudes Psi(state')/Psi(state)
        where state' is obtained by flipping the spins of state

        flips_to_new_state contains a list of which spins to flip to get from
        state to state
        """
        return np.exp(self.log_amplitude_ratio(state, flips_to_new_state))

    def log_amplitude_ratio(self, state, flips_to_new_state):
        """
        computationally more reliable to compute log of ratio of complex
        amplitudes
        """
        if len(flips_to_new_state) == 0:
            return 0.

        v = np.zeros((self.n_visible, 1))
        for flip_loc in flips_to_new_state:
            v[flip_loc] = 2 * state[flip_loc]

        logpop = - np.dot(self.a.T, v) + \
                 np.sum(
                     np.log(np.cosh(self.thetas - np.dot(self.W.T, v))) -
                     np.log(np.cosh(self.thetas))
                 )

        return logpop

    def init_lookup_tables(self, state):
        """
        Given state as an array, update the list of theta values which compute
        complex prefactor of state. See Carleo and Troyer Supplemental Info
        """
        s = np.array(state, dtype=float).reshape((self.n_visible, 1))
        self.thetas = self.b + np.dot(self.W.T, s)

    def update_lookup_tables(self, state, flips_to_new_state):
        """
        given a transition in metropolis algorithm as implemented in
        sampler.py this method updates the lookup tables to now store state
        'where state' is obtained by taking the original state, and fliping the
        quantum variable at locations listed in flips_to_new_state

        flips_to_new_state is list of integers specifying where the flips occur
        """
        if len(flips_to_new_state) == 0:
            return
        v = np.zeros((self.n_visible, 1))
        for flip_loc in flips_to_new_state:
            v[flip_loc] = 2 * state[flip_loc]

        self.thetas -= np.dot(self.W.T, v)

    @staticmethod
    def _string_to_complex(string):
        translator = str.maketrans('', '', '()\n')
        string_list = string.translate(translator).split(',')
        return complex(float(string_list[0]), float(string_list[1]))
