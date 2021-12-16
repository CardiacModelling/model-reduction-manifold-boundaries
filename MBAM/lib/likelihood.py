import pints
import numpy as np

class LogLikelihood(pints.LogPDF):
    '''
    Error function for optimisation
    '''
    def __init__(self, I_data, I_min_func, n_params, check_rates=True):
        self.I_data = I_data
        self.I_min_func = I_min_func
        self.n_params = n_params
        self.check_rates = check_rates

    def __call__(self, x):
        return -self.I_min_func(x, self.I_data, self.check_rates)

    def n_parameters(self):
        return self.n_params

def compute_eMRMS(current_simulator, para, I_data):
    ''' Compute mixed root mean square error
    '''
    error = np.sum(((I_data - current_simulator(para)) / (1 + abs(I_data)))**2)
    return np.sqrt(error / len(I_data))

def evaluate_cost_function(current_simulator, para, I_data):
    ''' Evaluate cost function
    '''
    return np.sum((I_data - current_simulator(para))**2)
