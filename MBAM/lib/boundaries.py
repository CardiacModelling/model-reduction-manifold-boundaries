#!/usr/bin/env python3
#
# Pints Boundaries that limit the transition rates
#
from __future__ import division, print_function
import numpy as np
import pints


class Boundaries(pints.Boundaries):
    """
    Boundary constraints on the parameters
    """
    def __init__(self, b_rates, n_params, conductance=True):
        super(Boundaries, self).__init__()

        self.lower = [-16.2 for i in range(n_params)]
        self.upper = [12 for i in range(n_params)]

        for i in range(n_params):
            if i in b_rates:
                self.lower[i] = 1e-7
                self.upper[i] = 0.4

        # Conductance
        if conductance:
            self.lower[-1] = -4.6
            self.upper[-1] = 2.3

        self.n_params = n_params

    def n_parameters(self):
        ''' Return number of parameters
        '''
        return self.n_params

    def check(self, parameters):
        ''' Check if parameters fall within the boundaries
        '''
        debug = False

        # Check parameter boundaries
        if np.any(parameters < self.lower):
            if debug: print('Lower')
            return False
        if np.any(parameters > self.upper):
            if debug: print('Upper')
            return False

        return True
        
