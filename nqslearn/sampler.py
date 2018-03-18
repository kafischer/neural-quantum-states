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
import random
import math


class MetropolisSampler(object):
    """
    Generates samples of configurations from an neural quantum state (NQS)
    using a Metropolis-Hastings algorithm.
    """

    def __init__(self, hamiltonian, nqs, zero_magnetization=True,
                 start_seed=0, filename=None, initial_state=None):
        self.hamiltonian = hamiltonian
        self.nqs = nqs
        self.start_seed = start_seed
        self.n_visible = nqs.n_visible
        self.samples_file = filename
        self.current_Hloc = 0
        self.local_energies = []
        self.state_history = []
        self.acceptances = 0.0
        self.num_moves = 0.0
        self.zero_magnetization = zero_magnetization
        self.nqs_energy = None
        self.nqs_energy_err = None
        self.correlation_time = None
        if initial_state is None:
            self.init_random_state()
        else:
            self.current_state = initial_state

    def init_random_state(self):
        self.current_state = [random.randrange(-1, 2, 2) # should make into numpy array!
                              for _ in range(self.n_visible)]
        if self.zero_magnetization:
            if self.n_visible % 2 == 1:
                raise ValueError('Cannot initialize a random state with ' +
                                 'zero magnetization for odd number of spins')
            total_mag = sum(self.current_state)
            if total_mag > 0:
                while total_mag != 0:
                    index = self._pick_site()
                    while self.current_state[index] < 0:
                        index = self._pick_site()
                    self.current_state[index] *= -1
                    total_mag -= 1
            elif total_mag < 0:
                while total_mag != 0:
                    index = self._pick_site()
                    while self.current_state[index] > 0:
                        index = self._pick_site()
                    self.current_state[index] *= -1
                    total_mag += 1

    def generate_spin_flips(self, n_flips):
        """
        generates a random spin flip. Note that if the flips returns the same
        state then this returns an empty list. it does not resample to ensure
        a new state has been generated
        """
        first_flip_site = self._pick_site()
        first_flip_value = self.current_state[first_flip_site]

        if n_flips == 2:
            next_flip_site = self._pick_site()
            next_flip_value = self.current_state[next_flip_site]
            if self.zero_magnetization:
                if first_flip_value == next_flip_value:
                    return []
                else:
                    return [first_flip_site, next_flip_site]
            else:
                if first_flip_site == next_flip_site:
                    return []
                else:
                    return [first_flip_site, next_flip_site]
        else:
            return [first_flip_site]

    def reset_sampler_values(self):
        self.acceptances = 0.0
        self.num_moves = 0.0
        self.state_history = []

    def acceptance_rate(self):
        return self.acceptances / self.num_moves

    def move(self, n_flips):
        flips = self.generate_spin_flips(n_flips)
        if len(flips) > 0:
            wf_ratio = self.nqs.amplitude_ratio(
                self.current_state, flips
            )
            acceptance_probability = np.square(np.abs(wf_ratio))

            # Metropolis-Hastings test
            if acceptance_probability > random.random():
                self.nqs.update_lookup_tables(
                    self.current_state, flips
                )
                for flip in flips:
                    self.current_state[flip] *= -1
                self.acceptances += 1

    def run(self, n_sweeps, therm_factor=0.1, sweep_factor=1, n_flips=None):
        """
        a sweep consists of (n_spins * sweep_factor) steps
        that is a user specifies a sweep to consist of flipping each spin an
        expected number of n_flips times.
        """
        if self.samples_file:
            f = open(self.samples_file, 'w')

        if n_flips:
            n_flips = self.hamiltonian.min_flips()
        if n_flips != 1 and n_flips != 2:
            raise ValueError('Invalid number of spin flips')
        if not (0 <= therm_factor <= 1):
            raise ValueError('The thermalization factor should be a real '
                             'number between 0 and 1')
        if n_sweeps < 50:
            raise ValueError('Too few steps in MC. Please use at least 50')

        print('Starting MC Sampling')
        print('Will perform {} steps'.format(n_sweeps))

        self.nqs.init_lookup_tables(self.current_state)
        self.reset_sampler_values()

        if therm_factor != 0:
            print('Starting Thermalization')

            n_moves = int(therm_factor * n_sweeps) * \
                int(sweep_factor * self.n_visible)
            for _ in range(n_moves):
                self.move(n_flips)

            print('Completed Thermalization')

        self.reset_sampler_values()

        print('Starting Monte Carlo Sampling')

        for i in range(int(n_sweeps)):
            for _ in range(int(sweep_factor * self.n_visible)):
                self.move(n_flips)
            self.current_Hloc = self.local_energy()
            self.state_history.append(np.array(self.current_state))
            self.local_energies.append(self.current_Hloc)
            if self.samples_file:
                self.write_current_state(f)

        print('Completed Monte Carlo Sampling')

        if self.samples_file:
            f.close()

        return self.estimate_wf_energy()

    def local_energy(self):
        """computes the local energy of the current state"""
        state = self.current_state
        (matrix_elements, transitions) = \
            self.hamiltonian.find_matrix_elements(state)
        energy_list = [self.nqs.amplitude_ratio(state, transitions[i]) * mel
                       for (i, mel) in enumerate(matrix_elements)]
        return sum(energy_list)

    def estimate_wf_energy(self):
        """
        computes a stochastic estimate of the energy of the nqs
        represented by the neural network
        """
        nblocks = 50
        blocksize = int(len(self.local_energies) / nblocks)
        enmean = 0
        enmeansq = 0
        enmean_unblocked = 0
        enmeansq_unblocked = 0

        for b in range(nblocks):
            eblock = 0.0
            for j in range(b * blocksize, (b + 1) * blocksize):
                eblock += self.local_energies[j].real
                delta = self.local_energies[j].real - enmean_unblocked
                enmean_unblocked += delta / (j + 1)
                delta2 = self.local_energies[j].real - enmean_unblocked
                # delta != delta2 because of update to enmean_unblocked
                enmeansq_unblocked += delta * delta2
            eblock /= blocksize
            delta = eblock - enmean
            enmean += delta / (b + 1)
            delta2 = eblock - enmean
            # delta != delta2 because of update to enmean
            enmeansq += delta * delta2

        enmeansq /= (nblocks - 1)
        enmeansq_unblocked /= (nblocks * blocksize - 1)
        est_avg = enmean / self.n_visible
        est_error = math.sqrt(enmeansq / nblocks) / self.n_visible
        self.nqs_energy = np.squeeze(est_avg)
        self.nqs_energy_err = np.squeeze(est_error)

        energy_report = 'Estimated average energy per spin: {} +/- {}'
        print(energy_report.format(est_avg, est_error))
        bin_report = 'Estimate from binning analysis. ' + \
                     '{} bins of {} samples each'
        print(bin_report.format(nblocks, blocksize))
        autocorrelation = 'Estimated autocorrelation time is {}'
        self.correlation_time = 0.5 * blocksize * enmeansq / enmeansq_unblocked
        print(autocorrelation.format(self.correlation_time))

    def write_current_state(self, f):
        line = ''
        for s in self.current_state:
            line += '{:>2}'.format(str(s)) + ' '
        eloc = self.current_Hloc
        line += '{}'.format((eloc.real, eloc.imag))
        f.write(line + '\n')

    def _pick_site(self):
        return random.randint(0, self.n_visible - 1)
