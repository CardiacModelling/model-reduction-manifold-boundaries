
def main():

    # Check input arguments
    parser = argparse.ArgumentParser(
        description='Reduce an ion channel model using the MBAM')
    parser.add_argument("-d", "--done", action='store_true', help="whether geodesic has been done or not",
        default=False)
    parser.add_argument("-p", "--plot", action='store_true', help="whether to show plots or not",
        default=False)
    args = parser.parse_args()

    def voltage(t):
        ''' Return custom 'mini staircase' voltage protocol
        '''
        # if (t >= 200.0 and t < 300.0):
        #     voltage = -110
        # elif (t >= 500.0 and t < 1000.0):
        #     voltage = 40
        # elif (t >= 1000.0 and t < 1250.0):
        #     voltage = -140
        # elif (t >= 1500.0 and t < 2000.0):
        #     voltage = 20
        # elif (t >= 2000.0 and t < 2200.0):
        #     voltage = -20
        # elif (t >= 2400.0 and t < 2600.0):
        #     voltage = 0
        # elif (t >= 2600.0 and t < 2800.0):
        #     voltage = -60
        # elif (t >= 2800.0 and t < 3000.0):
        #     voltage = -20
        # elif (t >= 3000.0 and t < 3500.0):
        #     voltage = 20
        # elif (t >= 3500.0 and t < 4000.0):
        #     voltage = -40
        # elif (t >= 4000.0 and t < 4250.0):
        #     voltage = 40
        # elif (t >= 4250.0 and t < 4500.0):
        #     voltage = -120
        # else:
        #     voltage = -80

        if (t >= 500.0 and t < 2500.0):
            voltage = -60
        elif (t >= 2500.0 and t < 3000.0):
            voltage = -80
        elif (t >= 3000.0 and t < 4500.0):
            voltage = -60
        elif (t >= 4500.0):
            voltage = -50
        else:
            voltage = -80

        return voltage

    def voltage2(t):
        if (t >= 500.0 and t < 2500.0):
            voltage = -60.5
        elif (t >= 2500.0 and t < 3000.0):
            voltage = -80
        elif (t >= 3000.0 and t < 4500.0):
            voltage = -60.5
        elif (t >= 4500.0):
            voltage = -50
        else:
            voltage = -80

        return voltage

    def voltage3(t):
        if (t >= 500.0 and t < 2500.0):
            voltage = -59.5
        elif (t >= 2500.0 and t < 3000.0):
            voltage = -80
        elif (t >= 3000.0 and t < 4500.0):
            voltage = -59.5
        elif (t >= 4500.0):
            voltage = -50
        else:
            voltage = -80

        return voltage

    par = Params()

    # Define iteration string
    iter_str = 'i9'

    txt_folder = 'txt_files/'
    os.makedirs(txt_folder, exist_ok=True)  # Create CMA-ES output destination folder

    geodesic_paths = np.loadtxt(txt_folder + 'geodesic_paths_' + iter_str + '.txt')
    n_vars, n_params, obs_var = np.loadtxt(txt_folder + 'settings_' + iter_str + '.txt', dtype=int)

    # Get end of geodesic path
    x0 = geodesic_paths[-1]

    # Remove redundant parameter
    x0 = np.delete(x0, 1)
    x0 = np.delete(x0, 4)

    b_rates = [0]
    for x, _ in enumerate(x0):
        if x in b_rates:
            x0[x] = np.exp(x0[x])

    # Update global parameters
    par.n_state_vars = n_vars
    par.n_params = n_params - 2
    par.observed_variable = obs_var

    # Create symbols for symbolic functions
    x, y, v = GetSymbols(par)

    # # Define system equations and initial conditions
    # # C12, O
    OtoC12 = se.exp(x[2]) * se.exp(-x[0] * v) / (1 + se.exp(x[3]) * se.exp(x[4] * v))
    conductance = se.exp(x[4]) / (1 + se.exp(x[2]) * se.exp(x[3] * v))

    t = se.symbols('t')
    exponent = se.exp(-(se.exp(x[1]) * se.exp(-x[0] * v) / (1 + se.exp(x[2]) * se.exp(x[3] * v))) * t)
    exponent_v = se.exp(x[1]) * se.exp(-x[0] * v) / (1 + se.exp(x[2]) * se.exp(x[3] * v))

    # Inputs
    conductance_inputs = x
    conductance_inputs.append(v)

    func_conductance = se.lambdify(conductance_inputs, [conductance])
    func_voltage_dependent_part_exponential = se.lambdify(conductance_inputs, [exponent_v])

    # Create RHS function
    conductance_inputs.append(t)
    func_rhs = se.lambdify(conductance_inputs, [exponent])

    def SimulateCurrent(x0, func_rhs, func_conductance, voltage):
        obs_times = np.linspace(0, par.tmax, par.tmax + 1)
        conductance_term, open_state, current, driving_term = [], [], [], []
        for t, _ in enumerate(obs_times):
            conductance_term.append(func_conductance((*x0, voltage(t))))
            open_state.append(func_rhs((*x0, voltage(t), t)))
            driving_term.append(voltage(t) - par.Erev)
            current.append(conductance_term[t] * open_state[t] * driving_term[t])
        return current

    def SimulateEverything(x0, func_rhs, func_conductance, voltage):
        obs_times = np.linspace(0, par.tmax, par.tmax + 1)
        conductance_term, open_state, current, driving_term = [], [], [], []
        for t, _ in enumerate(obs_times):
            conductance_term.append(func_conductance((*x0, voltage(t))))
            # conductance_term.append(func_voltage_dependent_part_exponential((*x0, voltage(t))))
            open_state.append(func_rhs((*x0, voltage(t), t)))
            driving_term.append(voltage(t) - par.Erev)
            current.append(conductance_term[t] * open_state[t] * driving_term[t])
        return conductance_term, open_state, driving_term

    def VoltageDependence(x0, func_conductance, func_voltage_dependent_part_exponential, voltages):
        con, vol = [], []
        for v in voltages:
            con.append(func_conductance((*x0, v)))
            vol.append(func_voltage_dependent_part_exponential((*x0, v)))

        return con, vol

    def I_min(x, I_data, check_rates=False):
        ''' Minimise difference between model and provided data
        '''
        I = SimulateCurrent(x, func_rhs, func_conductance, voltage)

        return np.sum((I - I_data)**2)   

    # Extract full model output
    I_data = np.loadtxt(txt_folder + 'current_i1.txt')

    obs_times = np.linspace(0, par.tmax, par.tmax + 1)
    ts = obs_times[221::130]
    I_obs = [I_data[i] for i, _ in enumerate(I_data) if i in ts]
    t_obs = [obs_times[i] for i, _ in enumerate(obs_times) if i in ts]

    if args.plot:
        import matplotlib.pyplot as plt

    if not args.done:
        # Fix random seed for reproducibility
        np.random.seed(100)
        
        # Define log-likelihood function
        LL = LogLikelihood(I_data, I_min, len(x0), check_rates=False)
        
        params, scores = [], []
        for i in range(1):
            print('Repeat ' + str(i+1))
            x0_params = x0
            opt = pints.OptimisationController(LL, x0_params, method=pints.CMAES) 
            # opt.optimiser().set_population_size(100)
            opt.set_parallel(True)
            opt.set_log_to_file(txt_folder + 'CMAES_' + iter_str + '.txt')

            # Run optimisation
            with np.errstate(all='ignore'):  # Tell numpy not to issue warnings
                xbest, fbest = opt.run()
                params.append(xbest)
                scores.append(-fbest)

        # Order from best to worst
        order = np.argsort(scores)
        scores = np.asarray(scores)[order]
        params = np.asarray(params)[order]

        np.savetxt(txt_folder + 'params_' + iter_str + '.txt', params[0])

    para_new = np.loadtxt(txt_folder + 'params_' + iter_str + '.txt')

    error = np.sum(((I_data - SimulateCurrent(para_new, func_rhs, func_conductance, voltage)) / (1 + abs(I_data)))**2)
    eMRMS = np.sqrt(error / len(I_data))
    print('eMRMS = ' + str(eMRMS))
    np.savetxt(txt_folder + 'eMRMS_' + iter_str + '.txt', [eMRMS])

    if args.plot:

        conductance_t, open_s, driving_t = SimulateEverything(para_new, func_rhs, func_conductance, voltage)
        conductance_t2, open_s2, driving_t2 = SimulateEverything(para_new, func_rhs, func_conductance, voltage2)
        conductance_t3, open_s3, driving_t3 = SimulateEverything(para_new, func_rhs, func_conductance, voltage3)

        # fig = plt.figure()
        # ax1 = fig.add_subplot(211)
        # ax1.set_xlim([0, par.tmax])
        # [label.set_visible(False) for label in ax1.get_xticklabels()]
        # ax1.set_ylabel('Voltage (mV)')
        # ax1.plot([voltage(t) for t, _ in enumerate(obs_times)])
        # ax1.grid(True)
        # ax2 = fig.add_subplot(212)
        # ax2.set_xlim([0, par.tmax])
        # ax2.set_xlabel('Time (ms)')
        # ax2.set_ylabel('Current (nA)')
        # ax2.plot(I_data, label='Original model current')
        # ax2.plot(SimulateCurrent(para_new, func_rhs, func_conductance, voltage), linestyle='dashed', label='Reduced model current')
        # ax2.scatter(t_obs, I_obs, facecolor='none', edgecolor='red', label='Observation times')
        # ax2.legend()
        # ax2.grid(True)
        # plt.tight_layout()
        # plt.show()

        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(321)
        ax1.set_title('voltage (mV)')
        ax1.plot([voltage(t) for t, _ in enumerate(obs_times)])
        ax1.plot([voltage2(t) for t, _ in enumerate(obs_times)])
        ax1.plot([voltage3(t) for t, _ in enumerate(obs_times)])
        ax2 = fig.add_subplot(322)
        ax2.set_title('voltage (mV)')
        ax2.plot([voltage(t) for t, _ in enumerate(obs_times)])
        ax2.plot([voltage2(t) for t, _ in enumerate(obs_times)])
        ax2.plot([voltage3(t) for t, _ in enumerate(obs_times)])
        ax3 = fig.add_subplot(323)
        ax3.set_title('conductance')
        ax3.plot(conductance_t[1:])
        ax3.plot(conductance_t2[1:])
        ax3.plot(conductance_t3[1:])
        ax4 = fig.add_subplot(324)
        ax4.set_title("O''")
        ax4.plot(open_s[1:])
        ax4.plot(open_s2[1:])
        ax4.plot(open_s3[1:])
        ax5 = fig.add_subplot(325)
        ax5.set_title("conductance * O''")
        product, product2, product3 = [], [], []
        for c, _ in enumerate(conductance_t[1:]):
            product.append(conductance_t[c+1] * open_s[c+1])
            product2.append(conductance_t2[c+1] * open_s2[c+1])
            product3.append(conductance_t3[c+1] * open_s3[c+1])
        ax5.plot(product)
        ax5.plot(product2)
        ax5.plot(product3)
        ax6 = fig.add_subplot(326)
        ax6.set_title('current (nA)')
        # ax6.plot(I_data, label='Original model current', color='silver')
        ax6.plot(SimulateCurrent(para_new, func_rhs, func_conductance, voltage)[1:], linestyle='dashed', label='Reduced model current')
        ax6.plot(SimulateCurrent(para_new, func_rhs, func_conductance, voltage2)[1:], linestyle='dashed')
        ax6.plot(SimulateCurrent(para_new, func_rhs, func_conductance, voltage3)[1:], linestyle='dashed')
        ax6.legend(fontsize=8)

        [label.set_visible(False) for label in ax1.get_xticklabels()]
        [label.set_visible(False) for label in ax2.get_xticklabels()]
        [label.set_visible(False) for label in ax3.get_xticklabels()]
        [label.set_visible(False) for label in ax4.get_xticklabels()]
        ax5.set_xlabel('Time (ms)')
        ax6.set_xlabel('Time (ms)')

        plt.tight_layout()
        plt.show() 

        vs = np.linspace(-120, 40, 17)
        con, vol = VoltageDependence(para_new, func_conductance, func_voltage_dependent_part_exponential, vs)

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # # ax.plot(vs, con)
        # ax.plot(vs, vol)
        # ax.set_yscale('log')
        # ax.grid(True)
        # plt.show()


if __name__=="__main__":

    import os
    import numpy as np
    import sympy as sp
    import symengine as se
    import pints
    import argparse

    from lib.settings import Params
    from lib.sensitivity_equations import GetSensitivityEquations, GetSymbols
    from lib.diagnostic_plots import DiagnosticPlots
    from lib.likelihood import LogLikelihood, compute_eMRMS, evaluate_cost_function
    import lib.boundaries as boundaries
    from scipy.integrate import odeint
    
    main()

