#!/usr/bin/env python3
#
# Pints Boundaries that limit the transition rates in the Beattie et al model.
#
from __future__ import division, print_function
import numpy as np
import pints


class LogPrior(pints.LogPrior):
    """
    Boundary constraints on the parameters
    """
    def __init__(self, which_model='wang', transformation=None):
        super(LogPrior, self).__init__()

        self.which_model = which_model

        # Conductance limits
        self.lower_conductance = 1e-2
        self.upper_conductance = 1e0

        # Limits on p1-p8
        self.lower_alpha = 1e-7              # Kylie: 1e-7
        self.upper_alpha = 1e3               # Kylie: 1e3
        self.upper_big_alpha = 1e6
        self.lower_beta  = 1e-7              # Kylie: 1e-7
        self.upper_beta  = 0.4               # Kylie: 0.4
        self.lower_rate  = 1.67e-5
        self.upper_rate  = 1000

        # Lower and upper bounds for all parameters
        if self.which_model == 'mazhari':
            self.lower = np.array([
                self.lower_alpha,
                self.lower_beta,
                self.lower_alpha,
                self.lower_beta,
                self.lower_alpha,
                self.lower_beta,
                self.lower_alpha,
                self.lower_beta,
                self.lower_alpha,
                self.lower_beta,
                self.lower_alpha,
                self.lower_beta,
                self.lower_alpha,
                self.lower_beta,
                self.lower_rate,
                self.lower_rate,                                                
                self.lower_conductance
            ])
            self.upper = np.array([
                self.upper_alpha,
                self.upper_beta,
                self.upper_alpha,
                self.upper_beta,
                self.upper_alpha,
                self.upper_beta,
                self.upper_alpha,
                self.upper_beta,
                self.upper_alpha,
                self.upper_beta,
                self.upper_alpha,
                self.upper_beta,
                self.upper_alpha,
                self.upper_beta,
                self.upper_rate,
                self.upper_rate,                                                
                self.upper_conductance
            ])
        elif self.which_model == 'mazhari-reduced':
            self.lower = np.array([
                self.lower_alpha, 
                self.lower_beta,  
                self.lower_alpha, 
                self.lower_beta,  
                self.lower_rate, 
                self.lower_alpha, 
                self.lower_beta, 
                self.lower_rate, 
                self.lower_alpha, 
                self.lower_beta,   
                self.lower_conductance
            ])
            self.upper = np.array([
                self.upper_alpha, 
                self.upper_beta,  
                self.upper_alpha, 
                self.upper_beta,  
                self.upper_rate, 
                self.upper_alpha, 
                self.upper_beta, 
                self.upper_rate, 
                self.upper_alpha, 
                self.upper_beta,   
                self.upper_conductance
            ])
        elif self.which_model == 'wang':
            self.lower = np.array([
                self.lower_alpha,
                self.lower_beta,
                self.lower_alpha,
                self.lower_beta,
                self.lower_rate,
                self.lower_rate,
                self.lower_alpha,
                self.lower_beta,
                self.lower_alpha,
                self.lower_beta,
                self.lower_alpha,
                self.lower_beta,
                self.lower_alpha,
                self.lower_beta,                                              
                self.lower_conductance
            ])
            self.upper = np.array([
                self.upper_alpha,
                self.upper_beta,
                self.upper_alpha,
                self.upper_beta,
                self.upper_rate,
                self.upper_rate,
                self.upper_alpha,
                self.upper_beta,
                self.upper_alpha,
                self.upper_beta,
                self.upper_alpha,
                self.upper_beta,
                self.upper_alpha,
                self.upper_beta,                                            
                self.upper_conductance
            ])
        elif self.which_model == 'wang-r1':
            self.lower = np.array([
                self.lower_alpha,
                self.lower_beta,
                self.lower_alpha,
                self.lower_beta,
                self.lower_rate,
                self.lower_rate,
                self.lower_alpha,
                self.lower_beta,
                self.lower_alpha,
                self.lower_beta,
                self.lower_alpha,
                self.lower_beta,
                self.lower_rate,                                        
                self.lower_conductance
            ])
            self.upper = np.array([
                self.upper_alpha,
                self.upper_beta,
                self.upper_alpha,
                self.upper_beta,
                self.upper_rate,
                self.upper_rate,
                self.upper_alpha,
                self.upper_beta,
                self.upper_alpha,
                self.upper_beta,
                self.upper_alpha,
                self.upper_beta,
                self.upper_rate,                                         
                self.upper_conductance
            ])
        elif self.which_model == 'wang-r2':
            self.lower = np.array([
                self.lower_alpha,
                self.lower_beta,
                self.lower_alpha,
                self.lower_beta,
                self.lower_rate,
                self.lower_rate,
                self.lower_alpha,
                self.lower_beta,
                self.lower_alpha,
                self.lower_beta,
                self.lower_alpha,
                self.lower_beta,                                   
                self.lower_conductance
            ])
            self.upper = np.array([
                self.upper_alpha,
                self.upper_beta,
                self.upper_alpha,
                self.upper_beta,
                self.upper_rate,
                self.upper_rate,
                self.upper_alpha,
                self.upper_beta,
                self.upper_alpha,
                self.upper_beta,
                self.upper_big_alpha,
                self.upper_beta,                                      
                self.upper_conductance
            ])
        elif self.which_model == 'wang-r3':
            self.lower = np.array([
                self.lower_alpha,
                self.lower_beta,
                self.lower_alpha,
                self.lower_beta,
                self.lower_alpha,
                self.lower_rate,
                self.lower_alpha,
                self.lower_beta, 
                self.lower_alpha,
                self.lower_beta, 
                self.lower_beta,                                         
                self.lower_conductance
            ])
            self.upper = np.array([
                self.upper_alpha,
                self.upper_beta,
                self.upper_alpha,
                self.upper_beta,
                self.upper_alpha,
                self.upper_rate,
                self.upper_alpha,
                self.upper_beta, 
                self.upper_alpha,
                self.upper_beta, 
                self.upper_beta,                                          
                self.upper_conductance
            ])  
        elif self.which_model == 'wang-r4':
            self.lower = np.array([
                self.lower_rate,
                self.lower_alpha,
                self.lower_beta,
                self.lower_alpha,
                self.lower_rate,
                self.lower_alpha,
                self.lower_beta, 
                self.lower_alpha,
                self.lower_beta, 
                self.lower_beta,                                         
                self.lower_conductance
            ])
            self.upper = np.array([
                self.upper_rate,
                self.upper_alpha,
                self.upper_beta,
                self.upper_alpha,
                self.upper_rate,
                self.upper_alpha,
                self.upper_beta, 
                self.upper_alpha,
                self.upper_beta, 
                self.upper_beta,                                          
                self.upper_conductance
            ])   
        elif self.which_model == 'wang-r5':
            self.lower = np.array([
                self.lower_alpha,
                self.lower_beta,
                self.lower_alpha,
                self.lower_rate,
                self.lower_alpha,
                self.lower_beta, 
                self.lower_alpha,
                self.lower_beta, 
                self.lower_beta,                                         
                self.lower_conductance
            ])
            self.upper = np.array([
                self.upper_big_alpha,
                self.upper_beta,
                self.upper_alpha,
                self.upper_rate,
                self.upper_alpha,
                self.upper_beta, 
                self.upper_alpha,
                self.upper_beta, 
                self.upper_beta,                                          
                self.upper_conductance
            ])   
        elif self.which_model == 'wang-r6':
            self.lower = np.array([
                self.lower_beta,
                self.lower_alpha,
                self.lower_alpha,
                self.lower_alpha,
                self.lower_beta,
                self.lower_alpha,
                self.lower_beta,
                self.lower_beta,                                       
                self.lower_conductance
            ])
            self.upper = np.array([
                self.upper_beta,
                self.upper_alpha,
                self.upper_alpha,
                self.upper_alpha,
                self.upper_beta,
                self.upper_alpha,
                self.upper_beta,
                self.upper_beta,                                       
                self.upper_conductance
            ])  
        elif self.which_model == 'wang-r7':
            self.lower = np.array([
                self.lower_beta,
                self.lower_alpha,
                self.lower_alpha,
                self.lower_alpha,
                self.lower_beta,
                self.lower_rate,
                self.lower_beta,                                      
                self.lower_conductance
            ])
            self.upper = np.array([
                self.upper_beta,
                self.upper_alpha,
                self.upper_alpha,
                self.upper_alpha,
                self.upper_beta,
                self.upper_rate,
                self.upper_beta,                                      
                self.upper_conductance
            ])   
        elif self.which_model == 'wang-r8':
            self.lower = np.array([
                self.lower_beta,
                self.lower_alpha,
                self.lower_alpha,
                self.lower_alpha,
                self.lower_beta,
                self.lower_beta,                                     
                self.lower_conductance
            ])
            self.upper = np.array([
                self.upper_beta,
                self.upper_alpha,
                self.upper_alpha,
                self.upper_big_alpha,
                self.upper_beta,
                self.upper_beta,                                     
                self.upper_conductance
            ])   
        else:
            pass         

        self.minf = -float('inf')

        # Limits on maximum reaction rates
        self.rmin = 1e-5 # Reduced as original 1.67e-5 is not obeyed by Mazhari model
        self.rmax = 1000
        self.super_rmax = 1e6

        # Voltages used to calculate maximum rates
        self.vmin = -120
        self.vmax =  60

        # Optional transformation
        self.transformation = transformation

        # Number of parameters
        if self.which_model == 'mazhari':
            n_params = 17
        elif self.which_model == 'mazhari-reduced':
            n_params = 11
        elif self.which_model == 'wang':
            n_params = 15
        elif self.which_model == 'wang-r1':
            n_params = 14
        elif self.which_model == 'wang-r2':
            n_params = 13
        elif self.which_model == 'wang-r3':
            n_params = 12
        elif self.which_model == 'wang-r4':
            n_params = 11
        elif self.which_model == 'wang-r5':
            n_params = 10
        elif self.which_model == 'wang-r6':
            n_params = 9
        elif self.which_model == 'wang-r7':
            n_params = 8
        elif self.which_model == 'wang-r8':
            n_params = 7
        else:
            pass

        self.n_params = n_params

    def n_parameters(self):
        return self.n_params

    def __call__(self, parameters):

        debug = True

        # Transform parameters back to model space
        if self.transformation is not None:
            parameters = self.transformation.detransform(parameters,self.which_model)

        # Check parameter boundaries
        if np.any(parameters < self.lower):
            if debug: print('Lower')
            return self.minf
        if np.any(parameters > self.upper):
            if debug: print('Upper')
            return self.minf

        if self.which_model == 'mazhari':
            # Check maximum rate constants
            p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17 = parameters

            # Check positive signed rates
            r = p1 * np.exp(p2 * self.vmax)
            if r < self.rmin or r > self.rmax:
                if debug: print('r1')
                return self.minf
            r = p5 * np.exp(p6 * self.vmax)
            if r < self.rmin or r > self.rmax:
                if debug: print('r2')
                return self.minf
            r = p9 * np.exp(p10 * self.vmax)
            if r < self.rmin or r > self.rmax:
                if debug: print('r3')
                return self.minf
            r = p13 * np.exp(p14 * self.vmax)
            if r < self.rmin or r > self.rmax:
                if debug: print('r4')
                return self.minf

            # Check negative signed rates
            r = p3 * np.exp(-p4 * self.vmin)
            if r < self.rmin or r > self.rmax:
                if debug: print('r5')
                return self.minf
            r = p7 * np.exp(-p8 * self.vmin)
            if r < self.rmin or r > self.rmax:
                if debug: print('r6')
                return self.minf
            r = p11 * np.exp(-p12 * self.vmin)
            if r < self.rmin or r > self.rmax:
                if debug: print('r7')
                return self.minf
        
        elif self.which_model == 'mazhari-reduced':
            # Check maximum rate constants
            p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11 = parameters

            # Check positive signed rates
            r = p1 * np.exp(p2 * self.vmax)
            if r < self.rmin or r > self.rmax:
                if debug: print('k23')
                return self.minf

            # Check negative signed rates
            r = p3 * np.exp(-p4 * self.vmin)
            if r < self.rmin or r > self.rmax:
                if debug: print('k32')
                return self.minf
            r = p6 * np.exp(-p7 * self.vmin)
            if r < self.rmin or r > self.rmax:
                if debug: print('k43')
                return self.minf
            r = p9 * np.exp(-p10 * self.vmin)
            if r < self.rmin or r > self.rmax:
                if debug: print('k15')
                return self.minf
        
        elif self.which_model == 'wang':
            # Check maximum rate constants
            p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15 = parameters

            # Check positive signed rates
            r = p1 * np.exp(p2 * self.vmax)
            if r < self.rmin or r > self.rmax:
                if debug: print('k23')
                return self.minf
            r = p7 * np.exp(p8 * self.vmax)
            if r < self.rmin or r > self.rmax:
                if debug: print('k34')
                return self.minf
            r = p11 * np.exp(p12 * self.vmax)
            if r < self.rmin or r > self.rmax:
                if debug: print('k51')
                return self.minf

            # Check negative signed rates
            r = p3 * np.exp(-p4 * self.vmin)
            if r < self.rmin or r > self.rmax:
                if debug: print('k32')
                return self.minf
            r = p9 * np.exp(-p10 * self.vmin)
            if r < self.rmin or r > self.rmax:
                if debug: print('k43')
                return self.minf
            r = p13 * np.exp(-p14 * self.vmin)
            if r < self.rmin or r > self.rmax:
                if debug: print('k15')
                return self.minf

        elif self.which_model == 'wang-r1':
            # Check maximum rate constants
            p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14 = parameters

            # Check positive signed rates
            r = p1 * np.exp(p2 * self.vmax)
            if r < self.rmin or r > self.rmax:
                if debug: print('k23')
                return self.minf
            r = p7 * np.exp(p8 * self.vmax)
            if r < self.rmin or r > self.rmax:
                if debug: print('k34')
                return self.minf
            r = p11 * np.exp(p12 * self.vmax)
            if r < self.rmin or r > self.rmax:
                if debug: print('k51')
                return self.minf

            # Check negative signed rates
            r = p3 * np.exp(-p4 * self.vmin)
            if r < self.rmin or r > self.rmax:
                if debug: print('k32')
                return self.minf
            r = p9 * np.exp(-p10 * self.vmin)
            if r < self.rmin or r > self.rmax:
                if debug: print('k43')
                return self.minf

        elif self.which_model == 'wang-r2':
            # Check maximum rate constants
            p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13 = parameters

            # Check positive signed rates
            r = p1 * np.exp(p2 * self.vmax)
            if r < self.rmin or r > self.rmax:
                if debug: print('C3toO')
                return self.minf
            r = p7 * np.exp(p8 * self.vmax)
            if r < self.rmin or r > self.rmax:
                if debug: print('OtoI')
                return self.minf

            # Check negative signed rates
            r = p3 * np.exp(-p4 * self.vmin)
            if r < self.rmin or r > self.rmax:
                if debug: print('OtoC3')
                return self.minf
            r = p9 * np.exp(-p10 * self.vmin)
            if r < self.rmin or r > self.rmax:
                if debug: print('ItoO')
                return self.minf

        elif self.which_model == 'wang-r3':
            # Check maximum rate constants
            p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12 = parameters

            # Check positive signed rates
            r = p1 * np.exp(p2 * self.vmax)
            if r < self.rmin or r > self.rmax:
                if debug: print('C3toO')
                return self.minf
            r = p7 * np.exp(p8 * self.vmax)
            if r < self.rmin or r > self.rmax:
                if debug: print('OtoI')
                return self.minf
            r = p5 * np.exp(p11 * self.vmax)
            if r < self.rmin or r > self.rmax:
                if debug: print('C12toC3')
                return self.minf

            # Check negative signed rates
            r = p3 * np.exp(-p4 * self.vmin)
            if r < self.rmin or r > self.rmax:
                if debug: print('OtoC3')
                return self.minf
            r = p9 * np.exp(-p10 * self.vmin)
            if r < self.rmin or r > self.rmax:
                if debug: print('ItoO')
                return self.minf

        elif self.which_model == 'wang-r4':
            # Check maximum rate constants
            p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11 = parameters

            # Check positive signed rates
            r = p6 * np.exp(p7 * self.vmax)
            if r < self.rmin or r > self.rmax:
                if debug: print('OtoI')
                return self.minf
            r = p4 * np.exp(p10 * self.vmax)
            if r < self.rmin or r > self.rmax:
                if debug: print('C12toC3')
                return self.minf

            # Check negative signed rates
            r = p2 * np.exp(-p3 * self.vmin)
            if r < self.rmin or r > self.rmax:
                if debug: print('OtoC3')
                return self.minf
            r = p8 * np.exp(-p9 * self.vmin)
            if r < self.rmin or r > self.rmax:
                if debug: print('ItoO')
                return self.minf

        elif self.which_model == 'wang-r5':
            # Check maximum rate constants
            p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = parameters

            # Check positive signed rates
            r = p3 * np.exp(p9 * self.vmax)
            if r < self.rmin or r > self.rmax:
                if debug: print('C12toC3O')
                return self.minf
            r = p5 * np.exp(p6 * self.vmax)
            if r < self.rmin or r > self.rmax:
                if debug: print('C3OtoI')
                return self.minf

            # Check negative signed rates
            r = p7 * np.exp(-p8 * self.vmin)
            if r < self.rmin or r > self.rmax:
                if debug: print('ItoC3O')
                return self.minf

        elif self.which_model == 'wang-r6':
            # Check maximum rate constants
            p1, p2, p3, p4, p5, p6, p7, p8, p9 = parameters

            # Check positive signed rates
            r = p4 * np.exp(p5 * self.vmax)
            if r < self.rmin or r > self.rmax:
                if debug: print('C3OtoI')
                return self.minf
            r = p2 * np.exp(p8 * self.vmax)
            if r < self.rmin or r > self.rmax:
                if debug: print('C12toC3O')
                return self.minf

            # Check negative signed rates
            r = p3 * np.exp(-p1 * self.vmin)
            if r < self.rmin or r > self.rmax:
                if debug: print('C3OtoC12')
                return self.minf
            r = p6 * np.exp(-p7 * self.vmin)
            if r < self.rmin or r > self.rmax:
                if debug: print('ItoC3O')
                return self.minf

        elif self.which_model == 'wang-r7':
            # Check maximum rate constants
            p1, p2, p3, p4, p5, p6, p7, p8 = parameters

            # Check positive signed rates
            r = p2 * np.exp(p7 * self.vmax)
            if r < self.rmin or r > self.rmax:
                if debug: print('C12toC3O')
                return self.minf
            r = p4 * np.exp(p5 * self.vmax)
            if r < self.rmin or r > self.rmax:
                if debug: print('C3OtoI')
                return self.minf

            # Check negative signed rates
            r = p3 * np.exp(-p1 * self.vmin)
            if r < self.rmin or r > self.rmax:
                if debug: print('C3OtoC12')
                return self.minf

        elif self.which_model == 'wang-r8':
            # Check maximum rate constants
            p1, p2, p3, p4, p5, p6, p7 = parameters

            # Check positive signed rates
            r = p2 * np.exp(p6 * self.vmax)
            if r < self.rmin or r > self.rmax:
                if debug: print('C12toC3O')
                return self.minf

            # Check negative signed rates
            r = p3 * np.exp(-p1 * self.vmin)
            if r < self.rmin or r > self.rmax:
                if debug: print('C3OtoC12')
                return self.minf

        else:
            pass

        return True

    def _sample_partial(self, v):
        """
        Sample a pair of parameters - uniformly in the transformed space - that
        satisfy the maximum transition rate constraints.
        """
        for i in range(100):
            a = np.exp(np.random.uniform(
                np.log(self.lower_alpha), np.log(self.upper_alpha)))
            b = np.random.uniform(self.lower_beta, self.upper_beta)
            r = a * np.exp(b * v)
            if r >= self.rmin and r <= self.rmax:
                return a, b
        raise ValueError('Too many iterations')

    def sample(self, n=1):

        if n > 1:
            raise NotImplementedError

        p = np.zeros(self.n_params)

        if self.which_model == 'mazhari':
            # Sample forward rates
            p[0:2] = self._sample_partial(self.vmax)
            p[4:6] = self._sample_partial(self.vmax)
            p[8:10] = self._sample_partial(self.vmax)
            p[12:14] = self._sample_partial(self.vmax)

            # Sample backward rates
            p[2:4] = self._sample_partial(-self.vmin)
            p[6:8] = self._sample_partial(-self.vmin)
            p[10:12] = self._sample_partial(-self.vmin)

            # Sample voltage-independent rates
            p[14:16] = np.random.uniform(
                self.lower_rate, self.upper_rate)

            # Sample conductance
            p[16] = np.random.uniform(
                self.lower_conductance, self.upper_conductance)
        
        elif self.which_model == 'mazhari-reduced':
            # Sample forward rates
            p[0:2] = self._sample_partial(self.vmax)

            # Sample backward rates
            p[2:4] = self._sample_partial(-self.vmin)
            p[5:7] = self._sample_partial(-self.vmin)
            p[8:10] = self._sample_partial(-self.vmin)

            # Sample voltage-independent rates
            p[4] = np.exp(np.random.uniform(
                np.log(self.lower_rate), np.log(self.upper_rate)))
            p[7] = np.exp(np.random.uniform(
                np.log(self.lower_rate), np.log(self.upper_rate)))

            # Sample conductance
            p[10] = np.random.uniform(
                self.lower_conductance, self.upper_conductance)


        elif self.which_model == 'wang':
            # Sample forward rates
            p[0:2] = self._sample_partial(self.vmax)
            p[6:8] = self._sample_partial(self.vmax)
            p[10:12] = self._sample_partial(self.vmax)

            # Sample backward rates
            p[2:4] = self._sample_partial(-self.vmin)
            p[8:10] = self._sample_partial(-self.vmin)
            p[12:14] = self._sample_partial(-self.vmin)

            # Sample voltage-independent rates
            p[4:6] = np.random.uniform(
                self.lower_rate, self.upper_rate)

            # Sample conductance
            p[14] = np.random.uniform(
                self.lower_conductance, self.upper_conductance)


        elif self.which_model == 'wang-r1':
            # Sample forward rates
            p[0:2] = self._sample_partial(self.vmax)
            p[6:8] = self._sample_partial(self.vmax)
            p[10:12] = self._sample_partial(self.vmax)

            # Sample backward rates
            p[2:4] = self._sample_partial(-self.vmin)
            p[8:10] = self._sample_partial(-self.vmin)

            # Sample voltage-independent rates
            p[4:6] = np.random.uniform(
                self.lower_rate, self.upper_rate)
            p[12] = np.random.uniform(
                self.lower_rate, self.upper_rate)

            # Sample conductance
            p[13] = np.random.uniform(
                self.lower_conductance, self.upper_conductance)


        elif self.which_model == 'wang-r2':
            # Sample forward rates
            p[0:2] = self._sample_partial(self.vmax)
            p[6:8] = self._sample_partial(self.vmax)

            # Sample backward rates
            p[2:4] = self._sample_partial(-self.vmin)
            p[8:10] = self._sample_partial(-self.vmin)

            # Sample voltage-independent rates
            p[4:6] = np.random.uniform(
                self.lower_rate, self.upper_rate)
            p[10] = np.random.uniform(
                self.lower_alpha, self.upper_big_alpha)
            p[11] = np.random.uniform(
                self.lower_beta, self.upper_beta)

            # Sample conductance
            p[12] = np.random.uniform(
                self.lower_conductance, self.upper_conductance)


        elif self.which_model == 'wang-r3':
            # Sample forward rates
            p[0], p[1] = self._sample_partial(self.vmax)
            p[4], p[10] = self._sample_partial(self.vmax)
            p[6], p[7] = self._sample_partial(self.vmax)

            # Sample backward rates
            p[2], p[3] = self._sample_partial(-self.vmin)
            p[8], p[9] = self._sample_partial(-self.vmin)

            # Sample voltage-independent rates
            p[5] = np.random.uniform(
                self.lower_rate, self.upper_rate)

            # Sample conductance
            p[11] = np.random.uniform(
                self.lower_conductance, self.upper_conductance)

        elif self.which_model == 'wang-r4':
            # Sample forward rates
            p[3], p[9] = self._sample_partial(self.vmax)
            p[5], p[6] = self._sample_partial(self.vmax)

            # Sample backward rates
            p[1], p[2] = self._sample_partial(-self.vmin)
            p[7], p[8] = self._sample_partial(-self.vmin)

            # Sample voltage-independent rates
            p[0] = np.random.uniform(
                self.lower_rate, self.upper_rate)
            p[4] = np.random.uniform(
                self.lower_rate, self.upper_rate)

            # Sample conductance
            p[10] = np.random.uniform(
                self.lower_conductance, self.upper_conductance)

        elif self.which_model == 'wang-r5':
            # Sample forward rates
            p[2], p[8] = self._sample_partial(self.vmax)
            p[4], p[5] = self._sample_partial(self.vmax)

            # Sample backward rates
            p[6], p[7] = self._sample_partial(-self.vmin)

            # Sample voltage-independent rates
            p[0] = np.random.uniform(
                self.lower_alpha, self.upper_big_alpha)
            p[1] = np.random.uniform(
                self.lower_beta, self.upper_beta)
            p[3] = np.random.uniform(
                self.lower_rate, self.upper_rate)

            # Sample conductance
            p[9] = np.random.uniform(
                self.lower_conductance, self.upper_conductance)

        elif self.which_model == 'wang-r6':
            # Sample forward rates
            p[3], p[4] = self._sample_partial(self.vmax)
            p[1], p[7] = self._sample_partial(self.vmax)

            # Sample backward rates
            p[2], p[0] = self._sample_partial(-self.vmin)
            p[5], p[6] = self._sample_partial(-self.vmin)

            # Sample conductance
            p[8] = np.random.uniform(
                self.lower_conductance, self.upper_conductance)

        elif self.which_model == 'wang-r7':
            # Sample forward rates
            p[1], p[6] = self._sample_partial(self.vmax)
            p[3], p[4] = self._sample_partial(self.vmax)

            # Sample backward rates
            p[2], p[0] = self._sample_partial(-self.vmin)

            # Sample voltage-independent rates
            p[5] = np.random.uniform(
                self.lower_rate, self.upper_rate)

            # Sample conductance
            p[7] = np.random.uniform(
                self.lower_conductance, self.upper_conductance)

        elif self.which_model == 'wang-r8':
            # Sample forward rates
            p[1], p[5] = self._sample_partial(self.vmax)

            # Sample backward rates
            p[2], p[0] = self._sample_partial(-self.vmin)

            # Sample voltage-independent rates
            p[3] = np.random.uniform(
                self.lower_alpha, self.upper_big_alpha)
            p[4] = np.random.uniform(
                self.lower_beta, self.upper_beta)

            # Sample conductance
            p[6] = np.random.uniform(
                self.lower_conductance, self.upper_conductance)

        else:
            pass

        # Transform from model to search space, if required
        if self.transformation is not None:
            p = self.transformation.transform(p,self.which_model)

        # The Boundaries interface requires a matrix ``(n, n_parameters)``
        p.reshape(1, self.n_params)
        
        return p
