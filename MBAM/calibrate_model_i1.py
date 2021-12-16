
def main():

    # Check input arguments
    parser = argparse.ArgumentParser(
        description='Reduce an ion channel model using the MBAM')
    parser.add_argument("-d", "--done", action='store_true', help="whether geodesic has been done or not",
        default=False)
    parser.add_argument("-p", "--plot", action='store_true', help="whether to show plots or not",
        default=False)
    args = parser.parse_args()


    par = Params()

    # Define iteration string
    iter_str = 'i1'

    txt_folder = 'txt_files/'
    os.makedirs(txt_folder, exist_ok=True)  # Create CMA-ES output destination folder

    geodesic_paths = np.loadtxt(txt_folder + 'geodesic_paths_' + iter_str + '.txt')
    n_vars, n_params, obs_var = np.loadtxt(txt_folder + 'settings_' + iter_str + '.txt', dtype=int)

    # Get end of geodesic path
    x0 = geodesic_paths[-1]
    print(x0)

    # Remove redundant parameter
    x0 = np.delete(x0, 13)

    b_rates = [1, 3, 7, 9, 11]
    for x, _ in enumerate(x0):
        if x in b_rates:
            x0[x] = np.exp(x0[x])
    print(x0)

    # Update global parameters
    par.n_state_vars = n_vars
    par.n_params = n_params - 1
    par.observed_variable = obs_var

    # Create symbols for symbolic functions
    x, y, v = GetSymbols(par)

    # Define system equations and initial conditions
    # C2, C3, O, I
    C2toC1 = se.exp(x[12])
    C1toC2 = se.exp(x[10]) * se.exp(x[11] * v)
    C3toC2 = se.exp(x[5])
    C2toC3 = se.exp(x[4])
    OtoC3 = se.exp(x[2]) * se.exp(-x[3] * v)
    C3toO = se.exp(x[0]) * se.exp(x[1] * v)
    ItoO = se.exp(x[8]) * se.exp(-x[9] * v)
    OtoI = se.exp(x[6]) * se.exp(x[7] * v)
    conductance = se.exp(x[13])

    rates = [C2toC1, C1toC2, C3toC2, C2toC3, OtoC3, C3toO, ItoO, OtoI]
    states = ['C1', 'C2', 'C3', 'O', 'I'] # C1 is calculated from the other states

    rhs = [C1toC2 * (1 - y[0] - y[1] - y[2] - y[3]) + C3toC2 * y[1] - (C2toC1 + C2toC3) * y[0],
           C2toC3 * y[0] + OtoC3 * y[2] - (C3toC2 + C3toO) * y[1],
           C3toO * y[1] + ItoO * y[3] - (OtoC3 + OtoI) * y[2],
           OtoI * y[2] - ItoO * y[3]]
           
    # Write in matrix form taking y = ([C2], [C3], [O], [I])^T
    # RHS = A * y + B

    A = se.Matrix([[-C1toC2 - C2toC1 - C2toC3,  -C1toC2 + C3toC2,   -C1toC2,        -C1toC2], \
                   [C2toC3,                     -C3toC2 - C3toO,    OtoC3,          0], \
                   [0,                          C3toO,              -OtoC3 - OtoI,  ItoO], \
                   [0,                          0,                  OtoI,           -ItoO]])
    B = se.Matrix([C1toC2, 0, 0, 0])

    ICs = [0.0, 0.0, 0.0, 0.0]

    assert len(rhs) == par.n_state_vars, "RHS dimensions do not match the number of state variables"

    funcs = GetSensitivityEquations(par, x, y, v, rates, A, B, ICs, x0, second_order=False, conductance=conductance)

    # Extract full model output
    I_data = np.loadtxt(txt_folder + 'current_i1.txt')

    if args.plot:
        plot = DiagnosticPlots(par, funcs.SimulateCurrent)
        plot.obs_points(x0, funcs.voltage, I_data=I_data, compare_model_outputs=True, title_str='Before calibration')

    eMRMS = compute_eMRMS(funcs.SimulateCurrent, x0, I_data)
    if eMRMS > 0.1:
        print('Starting error is very big. Check the reduction is correct!')

    if not args.done:
        # Fix random seed for reproducibility
        np.random.seed(100)
        
        # Define log-likelihood function
        LL = LogLikelihood(I_data, funcs.I_min, len(x0))

        # Tell CMA-ES about the bounds of this optimisation problem (helps it work out sensible sigma)
        bounds = boundaries.Boundaries(b_rates, par.n_params)

        opt = pints.OptimisationController(LL, x0, boundaries=bounds, method=pints.CMAES) 
        opt.set_parallel(False)
        opt.set_log_to_file(txt_folder + 'CMAES_' + iter_str + '.txt')

        # Run optimisation
        with np.errstate(all='ignore'):  # Tell numpy not to issue warnings
            xbest, fbest = opt.run()
            para_new = xbest
            scores = -fbest

        np.savetxt(txt_folder + 'params_' + iter_str + '.txt', para_new)

    para_new = np.loadtxt(txt_folder + 'params_' + iter_str + '.txt')
    eMRMS = compute_eMRMS(funcs.SimulateCurrent, para_new, I_data)
    print('eMRMS = ' + str(eMRMS))
    np.savetxt(txt_folder + 'eMRMS_' + iter_str + '.txt', [eMRMS])
    C = evaluate_cost_function(funcs.SimulateCurrent, para_new, I_data)
    print('C = ' + str(C))

    if args.plot:
        plot.obs_points(para_new, funcs.voltage, I_data=I_data, compare_model_outputs=True, title_str='After calibration')
        plot.plot_state_vars(funcs.rhs_full, para_new, state_labels=states)
        plot.show()

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
    
    main()

