
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

    par = Params()

    # Define iteration string
    iter_str = 'i9'

    txt_folder = 'txt_files/'
    os.makedirs(txt_folder, exist_ok=True)  # Create CMA-ES output destination folder

    geodesic_paths = np.loadtxt(txt_folder + 'geodesic_paths_' + iter_str + '.txt')
    n_vars, n_params, obs_var = np.loadtxt(txt_folder + 'settings_' + iter_str + '.txt', dtype=int)

    # Get end of geodesic path
    x0 = geodesic_paths[-1]
    x0[1] = x0[1] - np.exp(x0[5])

    # Remove redundant parameter
    x0 = np.delete(x0, 5)

    b_rates = [0]
    for x, _ in enumerate(x0):
        if x in b_rates:
            x0[x] = np.exp(x0[x])
    print(x0)
    x0 = [3.961538472624422269e+00,
    0,
    -2.457611947774498731e+02,
    2.315507672364830682e+00,
    3.016447382731618498e-02,
    -1.980420265999121909e+00]

    # Update global parameters
    par.n_state_vars = n_vars
    par.n_params = n_params - 1
    par.observed_variable = obs_var

    # Create symbols for symbolic functions
    x, y, v = GetSymbols(par)

    # # Define system equations and initial conditions
    # # C12, O
    OtoC12 = se.exp(x[2]) * se.exp(-x[0] * v) / (1 + se.exp(x[3]) * se.exp(x[4] * v))
    conductance = se.exp(x[5]) / (1 + se.exp(x[3]) * se.exp(x[4] * v))

    t = se.symbols('t')
    # exponent = (se.Piecewise((1, v > -80), (0, v <= -80))) * se.exp(-(se.exp(x[2]) * se.exp(-x[0] * v) / (1 + se.exp(x[3]) * se.exp(x[4] * v))) * t)
    exponent = se.Piecewise((1, v > -x[1]), (se.exp(-(se.exp(x[2]) * se.exp(-x[0] * v) / (1 + se.exp(x[3]) * se.exp(x[4] * v))) * t), v <= -x[1]))

    conductance_inputs = x
    conductance_inputs.append(v)

    func_conductance = se.lambdify(conductance_inputs, [conductance])

    # Inputs for RHS
    # inputs = [(y[i]) for i in range(par.n_state_vars)]
    # [inputs.append(x[j]) for j in range(par.n_params)]
    # inputs.append(v)

    # Create RHS function
    # frhs = [rhs[i] for i in range(par.n_state_vars)]
    conductance_inputs.append(t)
    func_rhs = se.lambdify(conductance_inputs, [exponent])
    # inputs = np.append(x0, voltage(0))
    # inputs = np.append(inputs, 0)
    # print(func_rhs((*x0, voltage(0), 0)))

    def SimulateCurrent(x0, func_rhs, func_conductance, voltage):
        obs_times = np.linspace(0, par.tmax, par.tmax + 1)
        conductance_term, open_state, current, driving_term = [], [], [], []
        for t, _ in enumerate(obs_times):
            conductance_term.append(func_conductance((*x0, voltage(t))))
            open_state.append(func_rhs((*x0, voltage(t), t)))
            driving_term.append(voltage(t) - par.Erev)
            current.append(conductance_term[t] * open_state[t] * driving_term[t])
        return current

    def I_min(x, I_data, check_rates=False):
        ''' Minimise difference between model and provided data
        '''
        I = SimulateCurrent(x, func_rhs, func_conductance, voltage)

        return np.sum((I - I_data)**2)   

    # Extract full model output
    I_data = np.loadtxt(txt_folder + 'current_i1.txt')

    if args.plot:
        import matplotlib.pyplot as plt

        # plt.figure()
        # plt.plot(SimulateCurrent(x0, func_rhs, func_conductance, voltage))
        # plt.show()


    # eMRMS = compute_eMRMS(funcs.SimulateCurrent, x0, I_data)
    # if eMRMS > 0.1:
    #     print('Starting error is very big. Check the reduction is correct!')

    if not args.done:
        # Fix random seed for reproducibility
        np.random.seed(100)
        
        # Define log-likelihood function
        LL = LogLikelihood(I_data, I_min, len(x0), check_rates=False)

        # Tell CMA-ES about the bounds of this optimisation problem (helps it work out sensible sigma)
        # bounds = boundaries.Boundaries(b_rates, par.n_params)
        
        params, scores = [], []
        for i in range(2):
            print('Repeat ' + str(i+1))
            if i < 1:
                x0_params = x0
            else:
                x0_perturbed = np.random.uniform(-12, 1, size=len(x0))
                x0_params = x0_perturbed
            opt = pints.OptimisationController(LL, x0_params, method=pints.CMAES) 
            opt.optimiser().set_population_size(100)
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
    # eMRMS = compute_eMRMS(funcs.SimulateCurrent, para_new, I_data)
    # print('eMRMS = ' + str(eMRMS))
    # np.savetxt(txt_folder + 'eMRMS_' + iter_str + '.txt', [eMRMS])
    # C = evaluate_cost_function(funcs.SimulateCurrent, para_new, I_data)
    # print('C = ' + str(C))

    if args.plot:
        # plot.obs_points(para_new, funcs.voltage, I_data=I_data, compare_model_outputs=True, title_str='After calibration')
        # plot.plot_state_vars(funcs.rhs_full, para_new, state_labels=states)

        plt.figure()
        plt.plot(SimulateCurrent(para_new, func_rhs, func_conductance, voltage))
        plt.show()

        # plot.show()

    # rates = [OtoC12]
    # rates_str = ['OtoC12']
    # states = ['C12', 'O'] # C12 is calculated from the other states


    # rhs = [ - OtoC12 * y[0] ]  
           
    # # Write in matrix form taking y = ([O])^T
    # # RHS = A * y + B

    # A = se.Matrix([-OtoC12])
    # B = se.Matrix([0.0])

    # ICs = [0]

    # assert len(rhs) == par.n_state_vars, "RHS dimensions do not match the number of state variables"

    # # Create conductance function
    # rate_inputs = x
    # rate_inputs.append(v)

    # func_conductance = se.lambdify(rate_inputs, [conductance])

    # # Inputs for RHS ODEs
    # inputs = [(y[i]) for i in range(par.n_state_vars)]
    # [inputs.append(x[j]) for j in range(par.n_params)]
    # inputs.append(v)

    # rhs0 = ICs

    # # Create RHS function
    # frhs = [rhs[i] for i in range(par.n_state_vars)]
    # func_rhs = se.lambdify(inputs, frhs)

    # # Create Jacobian of the RHS function
    # jrhs = [se.Matrix(rhs).jacobian(se.Matrix(y))]
    # jfunc_rhs = se.lambdify(inputs, jrhs)

    # # Get steady state initial conditions for RHS
    # rhs_inf = (-(A.inv()) * B).subs(v, par.holding_potential)
    # rhs0 = [float(expr.evalf()) for expr in rhs_inf.subs(x, para)]
    # print('RHS ICs: ' + str(rhs0))

    # funcs = GetSensitivityEquations(par, x, y, v, rates, A, B, ICs, x0, second_order=False, conductance=conductance)

    # def SimulateCurrent(par, x, rhs, jrhs, ICs, func_conductance, voltage):
    #     ''' Simulate and return current for given parameter set under the defined voltage protocol
    #     '''
    #     o = odeint(rhs, ICs, par.obs_times, atol=1e-8, rtol=1e-8, Dfun=jrhs, args=(x, ))[:, par.observed_variable]
    #     # IKr = np.array([func_conductance(np.append(x, voltage(t))) * o[t] * (voltage(t) - par.Erev) for t, _ in enumerate(par.obs_times)])
    #     IKr = []
    #     for t, _ in enumerate(par.obs_times):
    #         if voltage(t) > 20:
    #             IKr.append(func_conductance(np.append(x, voltage(t))) * (voltage(t) - par.Erev))
    #         else:
    #             IKr.append(func_conductance(np.append(x, voltage(t))) * o[t] * (voltage(t) - par.Erev))
    #     return IKr

    # I = SimulateCurrent(par, x0, funcs.rhs, funcs.jrhs, ICs, func_conductance, funcs.voltage)

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(I)
    # plt.show()

#     # Extract full model output
#     I_data = np.loadtxt(txt_folder + 'current_i1.txt')

#     if args.plot:
#         plot = DiagnosticPlots(par, funcs.SimulateCurrent)
#         plot.obs_points(x0, funcs.voltage, I_data=I_data, compare_model_outputs=True, title_str='Before calibration')
#         plot.plot_state_vars(funcs.rhs_full, x0, state_labels=states)

#     eMRMS = compute_eMRMS(funcs.SimulateCurrent, x0, I_data)
#     if eMRMS > 0.1:
#         print('Starting error is very big. Check the reduction is correct!')

#     if not args.done:
#         # Fix random seed for reproducibility
#         np.random.seed(100)
        
#         # Define log-likelihood function
#         LL = LogLikelihood(I_data, funcs.I_min, len(x0), check_rates=False)

#         # Tell CMA-ES about the bounds of this optimisation problem (helps it work out sensible sigma)
#         # bounds = boundaries.Boundaries(b_rates, par.n_params)
        
#         params, scores = [], []
#         for i in range(1):
#             print('Repeat ' + str(i+1))
#             if i < 1:
#                 x0_params = x0
#             # x0_perturbed = np.random.normal(1, 1, size=len(x0)) * x0
#             else:
#                 x0_perturbed = np.random.uniform(-12, 1, size=len(x0))
#                 x0_params = x0_perturbed
#             opt = pints.OptimisationController(LL, x0_params, method=pints.CMAES) 
#             # opt.optimiser().set_population_size(100)
#             opt.set_parallel(True)
#             opt.set_log_to_file(txt_folder + 'CMAES_' + iter_str + '.txt')

#             # Run optimisation
#             with np.errstate(all='ignore'):  # Tell numpy not to issue warnings
#                 xbest, fbest = opt.run()
#                 params.append(xbest)
#                 scores.append(-fbest)

#         # Order from best to worst
#         order = np.argsort(scores)
#         scores = np.asarray(scores)[order]
#         params = np.asarray(params)[order]

#         np.savetxt(txt_folder + 'params_' + iter_str + '.txt', params[0])

#     para_new = np.loadtxt(txt_folder + 'params_' + iter_str + '.txt')
#     eMRMS = compute_eMRMS(funcs.SimulateCurrent, para_new, I_data)
#     print('eMRMS = ' + str(eMRMS))
#     np.savetxt(txt_folder + 'eMRMS_' + iter_str + '.txt', [eMRMS])
#     C = evaluate_cost_function(funcs.SimulateCurrent, para_new, I_data)
#     print('C = ' + str(C))

#     if args.plot:
#         # plot.obs_points(para_new, funcs.voltage, I_data=I_data, compare_model_outputs=True, title_str='After calibration')
#         # plot.plot_state_vars(funcs.rhs_full, para_new, state_labels=states)
#         plot.show()

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

