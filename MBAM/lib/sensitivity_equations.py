import numpy as np
import symengine as se
from scipy.integrate import odeint
import lib.FiniteDifference as FiniteDifference
import time
from itertools import chain

class GetSensitivityEquations(object):

    '''
    A class to generate and solve sensitivity ODEs for ion channel models

    Arguments are provided using the Params class
    '''

    def __init__(self, global_stuff, x, y, v, rates, A, B, ICs, para, second_order, conductance=1):

        # Get global settings from the Params object
        self.par = global_stuff

        # The time steps we want to output at
        self.obs_times = np.linspace(0, self.par.tmax, self.par.tmax + 1)
        self.par.obs_times = self.obs_times

        # Define settings for functions in this class
        self.second_order = second_order
        self.minf = -float('inf')

        # Pass provided arguments into function to compute sensitivity equations
        self. A = A
        self.B = B
        rhs = A * se.Matrix(y) + B
        self.compute_sensitivity_equations_rhs(x, y, v, rates, rhs, ICs, para, conductance)

    def compute_sensitivity_equations_rhs(self, x, y, v, rates, rhs, ICs, para, conductance):

        print('Creating rates function...')

        # Inputs for rate function
        rate_inputs = x
        rate_inputs.append(v)

        # Create rates function
        self.func_rates = se.lambdify(rate_inputs, rates)
        
        # Create conductance function
        self.func_conductance = se.lambdify(rate_inputs, [conductance])

        # Create dg/dp function
        dgdp = [se.diff(conductance, x[i]) for i in range(self.par.n_params)]
        self.func_dgdp = se.lambdify(rate_inputs, dgdp)

        print('Creating RHS function...')

        # Inputs for RHS ODEs
        inputs = [(y[i]) for i in range(self.par.n_state_vars)]
        [inputs.append(x[j]) for j in range(self.par.n_params)]
        inputs.append(v)

        self.rhs0 = ICs

        # Create RHS function
        frhs = [rhs[i] for i in range(self.par.n_state_vars)]
        self.func_rhs = se.lambdify(inputs, frhs)

        # Create Jacobian of the RHS function
        jrhs = [se.Matrix(rhs).jacobian(se.Matrix(y))]
        self.jfunc_rhs = se.lambdify(inputs, jrhs)

        print('Creating 1st order sensitivities function...')

        # Create symbols for 1st order sensitivities
        dydx = [[se.symbols('dy%d' % i + 'dx%d' % j) for j in range(self.par.n_params)] \
            for i in range(self.par.n_state_vars)]

        # Append 1st order sensitivities to inputs
        [[inputs.append(dydx[i][j]) for i in range(self.par.n_state_vars)] for j in range(self.par.n_params)]

        # Initialise 1st order sensitivities
        dS = [[0 for j in range(self.par.n_params)] for i in range(self.par.n_state_vars)]
        S = [[dydx[i][j] for j in range(self.par.n_params)] for i in range(self.par.n_state_vars)]

        # Create 1st order sensitivities function
        fS1, Ss = [], []
        for i in range(self.par.n_state_vars):
            for j in range(self.par.n_params):
                dS[i][j] = se.diff(rhs[i], x[j])
                for l in range(self.par.n_state_vars):
                    dS[i][j] = dS[i][j] + se.diff(rhs[i], y[l]) * S[l][j]

        # Flatten 1st order sensitivities for function
        [[fS1.append(dS[i][j]) for i in range(self.par.n_state_vars)] for j in range(self.par.n_params)]
        [[Ss.append(S[i][j]) for i in range(self.par.n_state_vars)] for j in range(self.par.n_params)]

        self.func_S1 = se.lambdify(inputs, fS1)

        # Define number of 1st order sensitivities
        self.par.n_state_var_sensitivities = self.par.n_params * self.par.n_state_vars

        # Append 1st order sensitivities to initial conditions
        dydxs = np.zeros((self.par.n_state_var_sensitivities))
        self.drhsdx0 = np.concatenate((ICs, dydxs))

        # Concatenate RHS and 1st order sensitivities
        Ss = np.concatenate((y, Ss))
        fS1 = np.concatenate((frhs, fS1))

        # Create Jacobian of the 1st order sensitivities function
        jS1 = [se.Matrix(fS1).jacobian(se.Matrix(Ss))]
        self.jfunc_S1 = se.lambdify(inputs, jS1)

        if self.second_order:
            print('Creating 2nd order sensitivities function...')
            assert conductance == 1, "2nd order sensitivities method currently only works with everything \
                contained in the RHS equations"

            # Create symbols for 2nd order sensitivities
            d2ydxdx = []
            [[d2ydxdx.append([se.symbols(str(S[i][j]) + 'dx%d' % l) for l in range(self.par.n_params)]) \
                for i in range(self.par.n_state_vars)] for j in range(self.par.n_params)]

            # Append 2nd order sensitivities to inputs
            [[inputs.append(d2ydxdx[i][j]) for i in range(self.par.n_state_var_sensitivities)] \
                for j in range(self.par.n_params)]

            # Initialise 2nd order sensitivities
            dSS = [[0 for j in range(self.par.n_params)] for i in range(self.par.n_state_var_sensitivities + \
                self.par.n_state_vars)]
            SS = [[d2ydxdx[i][j] for j in range(self.par.n_params)] \
                for i in range(self.par.n_state_var_sensitivities)]

            # Concatenate RHS and 1st order sensitivities
            SS = np.concatenate((S, SS))

            # Create 2nd order sensitivities function
            fS2, SSs = [], []
            for i in range(self.par.n_state_var_sensitivities + self.par.n_state_vars):
                for j in range(self.par.n_params):
                    dSS[i][j] = se.diff(fS1[i], x[j])
                    for l in range(self.par.n_state_var_sensitivities + self.par.n_state_vars):
                        dSS[i][j] = dSS[i][j] + se.diff(fS1[i], Ss[l]) * SS[l][j]

            # Append 1st order sensitivities to outputs
            [[fS2.append(dSS[i][j]) for i in range(self.par.n_state_vars)] for j in range(self.par.n_params)]

            # Append 2nd order sensitivities to outputs
            [[fS2.append(dSS[i+self.par.n_state_vars][j]) for i in range(self.par.n_state_var_sensitivities)] \
                for j in range(self.par.n_params)]
            [[SSs.append(SS[i][j]) for i in range(self.par.n_state_var_sensitivities)] \
                for j in range(self.par.n_params)]

            self.func_S2 = se.lambdify(inputs, fS2)

            # Define number of 2nd order sensitivities
            self.par.n_state_var_2nd_sensitivities = self.par.n_state_var_sensitivities * self.par.n_params

            # Append 2nd order sensitivities to initial conditions
            d2ydxdxs = np.zeros((self.par.n_state_var_2nd_sensitivities))
            self.d2rhsdxdx0 = np.concatenate((self.drhsdx0, d2ydxdxs))

            # Concatenate RHS, 1st and 2nd order sensitivities
            SSs = np.concatenate((Ss, SSs))
            fS2 = np.concatenate((frhs, fS2))

            # Create Jacobian of the 2nd order sensitivities function
            jS2 = [se.Matrix(fS2).jacobian(se.Matrix(SSs))]
            self.jfunc_S2 = se.lambdify(inputs, jS2)

        print('Getting ' + str(self.par.holding_potential) + ' mV steady state initial conditions...')

        # Get steady state initial conditions for RHS
        rhs_inf = (-(self.A.inv()) * self.B).subs(v, self.par.holding_potential)
        self.rhs0 = [float(expr.evalf()) for expr in rhs_inf.subs(x, para)]
        print('RHS ICs: ' + str(self.rhs0))

        # Get steady state initial conditions for 1st order sensitivities
        S1_inf = [float(se.diff(rhs_inf[i], x[j]).subs(x, para).evalf()) for j in range(0, self.par.n_params) \
            for i in range(0, self.par.n_state_vars)]

        self.drhs0 = np.concatenate((self.rhs0, S1_inf))

        print('Finished computing sensitivity equations.')

    def rhs(self, y, t, x):
        ''' Evaluate the RHS of the model '''
        return self.func_rhs((*y, *x, self.voltage(t)))

    def jrhs(self, y, t, x):
        ''' Evaluate the Jacobian of the RHS
            Having this function can speed up solving the system
        '''
        return self.jfunc_rhs((*y, *x, self.voltage(t)))

    # Returns our observation vector
    def rhs_obs_vector(self, x, full_output=False):
        ''' Solve the RHS of the system and return the current 'observation vector' at requested time points
        '''
        o = odeint(self.rhs, self.rhs0, self.obs_times, atol=1e-8, rtol=1e-8, Dfun=self.jrhs, args=(x, ))[:, self.par.observed_variable]
        current = np.array([self.func_conductance(np.append(x, self.voltage(t))) * o[t] * (self.voltage(t) - self.par.Erev) for t, _ in enumerate(self.obs_times)])
        return current if full_output else current[221::130]

    # Return all states
    def rhs_full(self, x):
        ''' Solve the RHS of the system and return the output in full
        '''
        return odeint(self.rhs, self.rhs0, self.obs_times, atol=1e-8, rtol=1e-8, Dfun=self.jrhs, args=(x, ))

    def drhsdx(self, y, t, x):
        """ Evaluate the RHS and 1st order sensitivities
        """
        outputs = self.func_rhs((*y[:self.par.n_state_vars], *x, self.voltage(t)))
        if not isinstance(outputs, list):
            outputs = [outputs]
        outputs.extend(self.func_S1((*y[:self.par.n_state_vars], *x, self.voltage(t), *y[self.par.n_state_vars:])))

        return outputs

    def jdrhsdx(self, y, t, x):
        """ Evaluate the Jacobian of the RHS and 1st order sensitivities
            Having this function can speed up solving the system
        """
        return self.jfunc_S1((*y[:self.par.n_state_vars], *x, self.voltage(t), *y[self.par.n_state_vars:]))

    # Return 1st order sensitivities
    def S1_obs_vector(self, x):
        ''' Solve the 1st order sensitivities and return at requested time points
        '''
        drhsdx = odeint(self.drhsdx, self.drhsdx0, self.obs_times, Dfun=self.jdrhsdx, args=(x, ))[:,self.par.n_state_vars+self.par.observed_variable::self.par.n_state_vars]
        o = odeint(self.rhs, self.rhs0, self.obs_times, atol=1e-8, rtol=1e-8, Dfun=self.jrhs, args=(x, ))[:, self.par.observed_variable]
        # Return only data corresponding to the observed variable
        drhsdx = np.array([(np.array(o[t]) * self.func_dgdp(np.append(x, self.voltage(t))) + self.func_conductance(np.append(x, self.voltage(t))) * drhsdx[t, :]) \
            * (self.voltage(t) - self.par.Erev) for t, _ in enumerate(self.obs_times)])
        return drhsdx[221::130, :]

    def drhsdx_full(self, x):
        ''' Solve the 1st order sensitivities of the system and return the output in full
        '''
        return odeint(self.drhsdx, self.drhsdx0, self.obs_times, atol=1e-8, rtol=1e-8, Dfun=self.jdrhsdx, args=(x, ))[:, self.par.n_state_vars:]

    def d2rhsdxdx(self, y, t, x):
        """ Evaluate the RHS, 1st and 2nd order sensitivities
        """
        outputs = self.func_rhs((*y[:self.par.n_state_vars], *x, self.voltage(t)))
        if not isinstance(outputs, list):
            outputs = [outputs]
        outputs.extend(self.func_S2((*y[:self.par.n_state_vars], *x, self.voltage(t), *y[self.par.n_state_vars:])))

        return outputs

    def jd2rhsdxdx(self, y, t, x):
        """ Evaluate the Jacobian of the 2nd order sensitivities function
            Having this function can speed up solving the system
        """
        return self.jfunc_S2((*y[:self.par.n_state_vars], *x, self.voltage(t), *y[self.par.n_state_vars:]))

    def Avv(self, x, v):
        ''' Solve the 2nd order sensitivities and return at requested time points
        '''
        if self.second_order:
            second_order = odeint(self.d2rhsdxdx, self.d2rhsdxdx0, self.obs_times, Dfun=self.jd2rhsdxdx, args=(x, ))[:, (self.par.n_state_vars+\
                self.par.n_state_var_sensitivities):]
            # Retain only data corresponding to the observed variable
            second_order = np.array([second_order[t, self.par.observed_variable::self.par.n_state_vars] * (self.voltage(t) - self.par.Erev) for t, _ in enumerate(self.obs_times)])
            second_order = second_order[221::130, :]
            # Multiply second order sensitivites by velocities
            v = v.reshape(1, self.par.n_params)
            return np.dot(list(chain.from_iterable(np.dot(v.T, v))), second_order.T)
        else:
            # Approximate Avv using finite differences
            return FiniteDifference.AvvCD(self.rhs_obs_vector, x, v, 1e-2)


    def voltage(self, t):
        ''' Return custom 'mini staircase' voltage protocol
        '''
        if (t >= 200.0 and t < 300.0):
            voltage = -110
        elif (t >= 500.0 and t < 1000.0):
            voltage = 40
        elif (t >= 1000.0 and t < 1250.0):
            voltage = -140
        elif (t >= 1500.0 and t < 2000.0):
            voltage = 20
        elif (t >= 2000.0 and t < 2200.0):
            voltage = -20
        elif (t >= 2400.0 and t < 2600.0):
            voltage = 0
        elif (t >= 2600.0 and t < 2800.0):
            voltage = -60
        elif (t >= 2800.0 and t < 3000.0):
            voltage = -20
        elif (t >= 3000.0 and t < 3500.0):
            voltage = 20
        elif (t >= 3500.0 and t < 4000.0):
            voltage = -40
        elif (t >= 4000.0 and t < 4250.0):
            voltage = 40
        elif (t >= 4250.0 and t < 4500.0):
            voltage = -120
        else:
            voltage = -80

        return voltage


    def SimulateCurrent(self, x):
        ''' Simulate and return current for given parameter set under the defined voltage protocol
        '''
        o = odeint(self.rhs, self.rhs0, self.obs_times, atol=1e-8, rtol=1e-8, Dfun=self.jrhs, args=(x, ))[:, self.par.observed_variable]
        return np.array([self.func_conductance(np.append(x, self.voltage(t))) * o[t] * (self.voltage(t) - self.par.Erev) for t, _ in enumerate(self.obs_times)])


    def GetStateVariables(self, x, normalise=True):
        '''
        Get all state variables
        The final state is constructed from the others
        '''
        states = self.rhs_full(x) 
        if normalise:
            states = states / x[-1] # Normalise to conductance

        state1 = np.zeros(self.par.tmax + 1)
        for t in range(self.par.tmax + 1):
            state1[t] = 1.0 - np.sum(states[t, :])
        state1 = state1.reshape(len(state1), 1)
        states = np.concatenate((state1, states), axis=1)
        return states


    def I_min(self, x, I_data, check_rates):
        ''' Minimise difference between model and provided data
        '''
        if check_rates:
            if (self.CheckRates(x)):
                return -self.minf

        I = self.SimulateCurrent(x)

        return np.sum((I - I_data)**2)    


    def CheckRates(self, x):
        ''' Check that transition rates fall within defined sensible ranges in order to avoid numerical issues
        '''
        if any(i < 1e-7 for i in self.func_rates(np.append(x, -120))):
            return True
        if any(j > 1e3 for j in self.func_rates(np.append(x, 60))):
            return True

    def Rates(self, x, v):
        ''' Return transition rates for a given parameter set and voltage
        '''
        return self.func_rates(np.append(x, v))

def GetSymbols(p):
    ''' Create symbols for parameters (x), state variables (y), and voltage (v)
    '''

    # Create parameter symbols
    x = [se.symbols('x%d' % j) for j in range(p.n_params)]

    # Create state variable symbols
    y = [se.symbols('y%d' % i) for i in range(p.n_state_vars)]

    # Create voltage symbol
    v = se.symbols('v')

    return x, y, v

