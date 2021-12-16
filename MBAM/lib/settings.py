
class Params(object):
    def __init__(self):

        '''
        Fixed parameter settings used for simulations
        '''
        self.n_state_vars = 0
        self.n_params = 0
        self.observed_variable = 0

        self.tmax = 5000
        self.Erev = -93.04
        self.GKr = 2.22871056087719094e-01

        self.holding_potential = -80