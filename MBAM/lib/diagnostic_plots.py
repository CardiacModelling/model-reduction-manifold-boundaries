import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.axisbelow'] = True
import os

from lib.settings import Params

class DiagnosticPlots(object):

    '''
    A class to define functions for diagnostic plots
    '''

    def __init__(self, global_stuff, current_simulator):

        # Get global settings from the Params object
        self.par = global_stuff

        self.current_simulator = current_simulator
        self.legend = ['p' + str(i+1) for i in range(self.par.n_params)]
        self.parameter_index_legend = [str(i+1) for i in range(self.par.n_params)]


    def obs_points(self, para, voltage_func, I_data, compare_model_outputs=False, title_str=None):
        
        ''' Compare model output with full model output at requested observation points
        '''
        
        # Simulate model
        ts = self.par.obs_times[221::130]
        I = self.current_simulator(para)
        v = [voltage_func(t) for t, _ in enumerate(self.par.obs_times)]

        # Extract full model output 'data' at requested time points
        I_obs = [I_data[i] for i, _ in enumerate(I_data) if i in ts]
        t_obs = [self.par.obs_times[i] for i, _ in enumerate(self.par.obs_times) if i in ts]

        # Create figure
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax1.set_xlim([0, self.par.tmax])
        [label.set_visible(False) for label in ax1.get_xticklabels()]
        ax1.set_ylabel('Voltage (mV)')
        ax1.plot(self.par.obs_times, v)
        ax1.grid(True)
        if title_str is not None:
            ax1.set_title(title_str)
        ax2 = fig.add_subplot(212)
        ax2.set_xlim([0, self.par.tmax])
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Current (nA)')
        if compare_model_outputs:
            ax2.plot(self.par.obs_times, I_data, label='Original model current')
            ax2.plot(self.par.obs_times, I, label='Reduced model current', linestyle='dashed')
        else:
            ax2.plot(self.par.obs_times, I, label='Current')
            ax2.scatter(t_obs, I_obs, facecolor='none', edgecolor='red', label='Observation times')
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()


    def plot_state_vars(self, rhs, para, state_labels=None, normalise=False):

        ''' Plot state variables of models for given RHS and parameter set
        '''

        times = self.par.obs_times
        states = rhs(para) 
        if normalise:
            states = states / np.exp(para[-1]) # Normalise to conductance

        # Get state labels
        if state_labels is None:
            state_labels = ['State ' + str(i+1) for i in range(self.par.n_state_vars + 1)]
        elif len(state_labels) != (self.par.n_state_vars + 1):
            raise ValueError(
                'Incorrect length of state_labels')

        # Construct missing state from the others
        state1 = np.zeros(self.par.tmax + 1)
        for t in range(self.par.tmax + 1):
            state1[t] = 1.0 - np.sum(states[t, :])
        state1 = state1.reshape(len(state1), 1)
        states = np.concatenate((state1, states), axis=1)

        # Create figure
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        for i in range(self.par.n_state_vars + 1):
            ax1.plot(times, states[:, i], label=state_labels[i])
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('State occupancy')
        ax1.legend()
        ax1.grid(True)
        plt.tight_layout()


    def plot_state_vars_pairwise(self, rhs, para, s1, s2, state_labels=None, normalise=False):

        ''' Create pairwise plot of two adjacent states
        '''

        assert s1 != s2, "Provided states must be different"
        assert abs(s1 - s2) == 1, "States must be adjacent"

        times = self.par.obs_times
        states = rhs(para)
        if normalise:
            states = states / np.exp(para[-1]) # Normalise to conductance

        # Get state labels
        if state_labels is None:
            state_labels = ['State ' + str(i+1) for i in range(self.par.n_state_vars + 1)]
        elif len(state_labels) != (self.par.n_state_vars + 1):
            raise ValueError(
                'Incorrect length of state_labels')

        # Construct missing state from the others
        state1 = np.zeros(self.par.tmax + 1)
        for t in range(self.par.tmax + 1):
            state1[t] = 1.0 - np.sum(states[t, :])
        state1 = state1.reshape(len(state1), 1)
        states = np.concatenate((state1, states), axis=1)

        # Create figure
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(states[:, s1], states[:, s2])
        ax1.set_xlabel(state_labels[s1] + ' occupancy')
        ax1.set_ylabel(state_labels[s2] + ' occupancy')
        ax1.grid(True)
        plt.tight_layout()


    def eigenvals(self, gvs, grid=True, figx=7, figy=5):

        ''' Plot eigenvalues at the start and end of the geodesic path
        '''

        evals_start = gvs[0] / np.sqrt(np.sum(gvs[0]**2))
        evals_end = gvs[-1] / np.sqrt(np.sum(gvs[-1]**2))

        # Create figure
        fig = plt.figure(figsize=(figx, figy))
        ax1 = fig.add_subplot(211)
        ax1.set_ylabel('Initial\ncomponents')
        ax1.bar(self.parameter_index_legend, evals_start)
        if grid:
            ax1.grid(True)
        else:
            ax1.axhline(0, color='silver', lw=1, zorder=-1)
        ax2 = fig.add_subplot(212, sharey=ax1)
        ax2.set_ylabel('Final\ncomponents')
        ax2.bar(self.parameter_index_legend, evals_end)
        ax2.set_xlabel('Parameter index')
        if grid:
            ax2.grid(True)
        else:
            ax2.axhline(0, color='silver', lw=1, zorder=-1)
        plt.tight_layout()


    def eigenval_spectrum_single(self, svals, ymin=1e-18, ymax=1e3, grid=True, figx=7, figy=5):

        ''' Plot eigenvalue spectrum of the Hessian matrix at the start of the geodesic path
        '''

        # Create figure
        fig = plt.figure(figsize=(figx, figy))
        ax1 = fig.add_subplot(111)
        ax1.set_yscale('log')
        for s in svals:
            ax1.axhline(s**2, xmin=0.2, xmax=0.8, lw=1)
        ax1.set_ylim([ymin, ymax])
        ax1.set_xticks([])
        if grid:
            ax1.grid(True)
        plt.tight_layout()


    def eigenval_spectrum_double(self, svals, grid=True, figx=7, figy=5):

        ''' Plot eigenvalue spectrum of the Hessian matrix at the start of the geodesic path
        '''

        svals_start = svals[1]
        svals_end = svals[-1]

        # Create figure
        fig = plt.figure(figsize=(figx, figy))
        ax1 = fig.add_subplot(121)
        ax1.set_title('Start')
        ax1.set_yscale('log')
        ax1.set_ylabel('Eigenvalues')
        for s in svals_start:
            ax1.axhline(s**2, xmin=0.2, xmax=0.8, linewidth=1)
        ax1.set_xticks([])
        if grid:
            ax1.grid(True)
        ax2 = fig.add_subplot(122, sharey=ax1)
        ax2.set_title('End')
        ax2.set_yscale('log')
        for e in svals_end:
            ax2.axhline(e**2, xmin=0.2, xmax=0.8, linewidth=1)
        ax2.set_xticks([])
        [label.set_visible(False) for label in ax2.get_yticklabels()]
        if grid:
            ax2.grid(True)
        plt.tight_layout()


    def geodesics(self, ts, xs, vs):

        ''' Plot parameter values and velocities along the geodesic
        '''

        log_legend = ['log(' + i + ')' for i in self.legend]

        # Create figure
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax1.plot(ts, np.exp(xs))
        ax1.legend(log_legend)
        ax1.set_yscale('log')
        ax1.set_xlabel("tau")
        ax1.set_ylabel("Parameter Values")
        ax1.grid(True)
        ax2 = fig.add_subplot(122)
        ax2.plot(ts, vs)
        ax2.set_xlabel("tau")
        ax2.set_ylabel("Parameter Velocities")
        plt.tight_layout()
        ax2.grid(True)


    def geodesics_2D_plot(self, xs, param1_number, param2_number, grid=True, figx=5, figy=5):

        ''' Plot parameter values and velocities along the geodesic
        '''

        log_legend = ['log(' + i + ')' for i in self.legend]

        # Create figure
        fig = plt.figure(figsize=(figx, figy))
        ax1 = fig.add_subplot(111)
        ax1.plot(xs[0, param1_number-1], xs[0, param2_number-1], 'ko')
        ax1.plot(xs[:, param1_number-1], xs[:, param2_number-1], c='k')
        ax1.arrow(xs[-2, param1_number-1], xs[-2, param2_number-1], \
            xs[-1, param1_number-1] - xs[-2, param1_number-1], xs[-1, \
            param2_number-1] - xs[-2, param2_number-1], head_width=0.02, \
            head_length=0.25, overhang=0.5, fc='k', ec='k', width=0, lw=1.5)
        ax1.set_xlabel(log_legend[param1_number-1])
        ax1.set_ylabel(log_legend[param2_number-1])
        if grid:
            ax1.grid(True)
        plt.tight_layout()


    def geodesics_2D_contour_plot(self, r, paras, param1_number, param2_number, \
        xlims=[-6, 0], ylims=[-6, 0], figx=5, figy=5):

        ''' Plot parameter values and velocities along the geodesic
        '''

        log_legend = ['log(' + i + ')' for i in self.legend]

        xlim1, xlim2 = xlims
        ylim1, ylim2 = ylims

        params = np.copy(paras)
        para = params[0]

        r0 = r(para)
        xs = np.linspace(xlim1, xlim2, 20)
        ys = np.linspace(ylim1, ylim2, 20)
        C = np.empty((20, 20))
        print('Making 2D contour plot...')
        for i, x in enumerate(xs):
            percent_done = (i+1) * 5
            print(str(percent_done) + '% done')
            for j, y in enumerate(ys):
                para[param1_number-1] = x
                para[param2_number-1] = y
                temp = r(para)
                C[j, i] = np.linalg.norm(temp - r0)**2

        # Create figure
        fig = plt.figure(figsize=(figx, figy))
        ax1 = fig.add_subplot(111)
        ax1.set_xlim(xlims)
        ax1.set_ylim(ylims)
        ax1.contourf(xs, ys, C, 15, cmap='Blues', extend="both") # levels=np.linspace(0,10,101)
        ax1.plot(paras[0, param1_number-1], paras[0, param2_number-1], 'ko')
        ax1.plot(paras[:, param1_number-1], paras[:, param2_number-1], c='k')
        ax1.set_xlabel(log_legend[param1_number-1])
        ax1.set_ylabel(log_legend[param2_number-1])
        plt.tight_layout()


    def rates(self, geodesic_paths, rate_func, rates_str):

        ''' Plot transition rates as a function of voltage at the start and end of the geodesic path
        '''

        # Compute transition rates and store in lists
        voltages = np.linspace(-140, 40, 37)
        rates_start, rates_end = [], []
        for v in voltages:
            rates_start.append(rate_func(geodesic_paths[0], v))
            rates_end.append(rate_func(geodesic_paths[-1], v))

        rates_start = np.transpose(rates_start)
        rates_end = np.transpose(rates_end)

        # Create figure
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(121)
        for i in range(len(rates_start)):
            ax1.plot(voltages, rates_start[i], label=rates_str[i])
        ax1.legend()
        ax1.set_xlabel("Voltage (mV)")
        ax1.set_yscale('log')
        ax1.grid(True)
        ax1.set_title('Start')
        ax2 = fig.add_subplot(122)
        for i in range(len(rates_end)):
            ax2.plot(voltages, rates_end[i], label=rates_str[i])
        ax2.legend()
        ax2.set_xlabel("Voltage (mV)")
        ax2.set_yscale('log')
        ax2.grid(True)
        ax2.set_title('End')
        plt.tight_layout()


    def rate_trajectory(self, geodesic_paths, rate_func, rates_str, rate_no):

        ''' Plot trajectory of given transition rate along the geodesic path
        '''

        # Compute rate trajectory
        voltages = np.linspace(-140, 40, 37)
        rate_trajectory = np.zeros((len(geodesic_paths), len(voltages)))
        for i, g in enumerate(geodesic_paths):
            for j, v in enumerate(voltages):
                rate_trajectory[i][j] = rate_func(g, v)[rate_no]

        # Define colour map
        cmap = matplotlib.cm.get_cmap('viridis')
        norm = matplotlib.colors.Normalize(0, len(geodesic_paths))

        # Create figure
        fig = plt.figure(figsize=(8, 8))
        ax1 = fig.add_subplot(211)
        for i, g in enumerate(geodesic_paths):
            ax1.plot(voltages, rate_trajectory[i, :], color=cmap(norm(i)))
        ax1.grid(True)
        [label.set_visible(False) for label in ax1.get_xticklabels()]
        ax1.set_title(rates_str[rate_no])
        ax2 = fig.add_subplot(212)
        ax2.set_yscale('log')
        for i, g in enumerate(geodesic_paths):
            ax2.plot(voltages, rate_trajectory[i, :], color=cmap(norm(i)))
        ax2.set_xlabel("Voltage (mV)")
        ax2.grid(True)
        plt.tight_layout()


    def state_fluxes(self, rhs, para, voltage_func, rate_func, rates_str, state_labels=None, normalise=False):

        ''' Plot state fluxes (function still under construction)
        '''

        times = self.par.obs_times
        states = rhs(para)
        if normalise:
            states = states / np.exp(para[-1]) # Normalise to conductance

        # Construct missing state from the others
        state1 = np.zeros(self.par.tmax + 1)
        for t in range(self.par.tmax + 1):
            state1[t] = 1.0 - np.sum(states[t, :])
        state1 = state1.reshape(len(state1), 1)
        states = np.concatenate((state1, states), axis=1)

        # Compute transition rates
        volts = [voltage_func(t) for t, _ in enumerate(self.par.obs_times)]
        rates = np.zeros((len(rates_str), self.par.tmax + 1))
        for u, v in enumerate(volts):
            rates[:, u] = rate_func(para, v)

        v = [voltage_func(t) for t, _ in enumerate(self.par.obs_times)]

        # Create figure
        fig = plt.figure(10)
        ax1 = fig.add_subplot(211)
        ax1.plot(times, v)
        ax2 = fig.add_subplot(212)
        ax2.plot(times[1:], rates[0, 1:]*states[1:, 1], label='C3*C3toC1')
        ax1.plot(times[1:], rates[1, 1:]*states[1:, 0], label='C1*C1toC3')
        ax2.legend()
        ax2.set_yscale('log')
        plt.tight_layout()


    def show(self):

        ''' Plot all created figures '''
        plt.show()

