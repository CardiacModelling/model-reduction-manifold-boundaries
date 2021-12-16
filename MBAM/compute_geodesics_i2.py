import numpy as np
import sympy as sp
import symengine as se
import matplotlib.pyplot as plt
plt.rcParams['axes.axisbelow'] = True
import os

from lib.geodesic import geodesic, InitialVelocity
from lib.settings import Params
from lib.sensitivity_equations import GetSensitivityEquations, GetSymbols
from lib.diagnostic_plots import DiagnosticPlots
from lib.command_line_args import GetParser

# Check input arguments
parser = GetParser()
args = parser.parse_args()

# Define iteration string
iter_str = 'i2'

par = Params()

# Define number of parameters and variables
par.n_state_vars = 4
par.n_params = 14
par.observed_variable = 2

txt_folder = 'txt_files/'
os.makedirs(txt_folder, exist_ok=True)  # Create CMA-ES output destination folder

# Choose starting parameters
para = np.loadtxt(txt_folder + 'params_i1.txt')

b_rates = [1, 3, 7, 9, 11]
for x, _ in enumerate(para):
    if x in b_rates:
        para[x] = np.log(para[x])

np.savetxt(txt_folder + 'settings_' + iter_str + '.txt', (par.n_state_vars, par.n_params, par.observed_variable), \
    fmt='%d')

# Create symbols for symbolic functions
x, y, v = GetSymbols(par)

# Define system equations and initial conditions
# C2, C3, O, I
C2toC1 = se.exp(x[12])
C1toC2 = se.exp(x[10]) * se.exp(se.exp(x[11]) * v)
C3toC2 = se.exp(x[5])
C2toC3 = se.exp(x[4])
OtoC3 = se.exp(x[2]) * se.exp(-se.exp(x[3]) * v)
C3toO = se.exp(x[0]) * se.exp(se.exp(x[1]) * v)
ItoO = se.exp(x[8]) * se.exp(-se.exp(x[9]) * v)
OtoI = se.exp(x[6]) * se.exp(se.exp(x[7]) * v)
conductance = se.exp(x[13])

rates = [C2toC1, C1toC2, C3toC2, C2toC3, OtoC3, C3toO, ItoO, OtoI]
rates_str = ['C2toC1', 'C1toC2', 'C3toC2', 'C2toC3', 'OtoC3', 'C3toO', 'ItoO', 'OtoI']
states = ['C1', 'C2', 'C3', 'O', 'I'] # C1 is calculated from the other states

rhs = [C1toC2 * (1 - y[0] - y[1] - y[2] - y[3]) + C3toC2 * y[1] - (C2toC1 + C2toC3) * y[0],
       C2toC3 * y[0] + OtoC3 * y[2] - (C3toC2 + C3toO) * y[1],
       C3toO * y[1] + ItoO * y[3] - (OtoC3 + OtoI) * y[2],
       OtoI * y[2] - ItoO * y[3]]
       
# Write in matrix form taking y = ([C2], [C3], [O], [I])^T
# RHS = A * y + B

A = se.Matrix([[-C1toC2 - C2toC1 - C2toC3,  -C1toC2 + C3toC2,   -C1toC2,        -C1toC2], \
               [C2toC3,                     -C3toC2 - C3toO,     OtoC3,          0], \
               [0,                           C3toO,              -OtoC3 - OtoI,  ItoO], \
               [0,                          0,                   OtoI,           -ItoO]])
B = se.Matrix([C1toC2, 0, 0, 0])

ICs = [0.0, 0.0, 0.0, 0.0]

second_order = True if args.gamma else False
funcs = GetSensitivityEquations(par, x, y, v, rates, A, B, ICs, para, second_order=second_order, conductance=conductance)

np.savetxt(txt_folder + 'current_' + iter_str + '.txt', funcs.SimulateCurrent(para))

iv, evals = InitialVelocity(para, funcs.S1_obs_vector, funcs.Avv, par, args.invert, args.eig)

if args.plot:
    plot = DiagnosticPlots(par, funcs.SimulateCurrent)
    plot.eigenval_spectrum_single(evals, figx=2.5, figy=4, grid=False)

if not args.done:
    # Callback function used to monitor the geodesic after each step
    def callback(geo):
        # Integrate until the norm of the velocity has grown by a factor of 10
        # and print out some diagnostic along the way
        mag = np.sqrt(np.sum(geo.vs[-1]**2))
        print('******************************** Iteration ' + str(len(geo.vs)) + \
            ' ********************************')
        print('Current direction: ' + str(geo.vs[-1]/mag))
        print('Current tau: ' + str('%3f' % geo.ts[-1]))
        print('Current smallest singular value: ' + str(np.min(geo.ss[-1])))
        v_threshold, ssv_threshold = 1000.0, args.ssv_threshold
        if np.linalg.norm(geo.vs[-1]) > v_threshold:
            print('|v| exceeded threshold')
        if np.min(geo.ss[-1]) < ssv_threshold:
            print('Smallest singular value threshold met')
        return np.linalg.norm(geo.vs[-1]) < v_threshold and np.min(geo.ss[-1]) > ssv_threshold

    # Construct the geodesic
    # It is usually not necessary to be very accurate here, so we set small tolerances
    geo_forward = geodesic(r=funcs.rhs_obs_vector, j=funcs.S1_obs_vector, Avv=funcs.Avv, N=par.n_params, x=para, v=iv, \
        atol=1e-3, rtol=1e-3, callback=callback)  

    # Integrate
    import time
    start = time.time()
    geo_forward.integrate(25.0)
    end = time.time()
    print('Time elapsed: ' + str("%.2f" % (end-start)) + ' s')

    print('Final parameters: ' + str(np.exp(geo_forward.xs[-1])))

    # # Save geodesic paths
    np.savetxt(txt_folder + 'geodesic_taus_' + iter_str + '.txt', geo_forward.ts)
    np.savetxt(txt_folder + 'geodesic_paths_' + iter_str + '.txt', geo_forward.xs)
    np.savetxt(txt_folder + 'geodesic_velocities_' + iter_str + '.txt', geo_forward.vs)
    np.savetxt(txt_folder + 'geodesic_errors_' + iter_str + '.txt', geo_forward.es)
    np.savetxt(txt_folder + 'geodesic_svals_' + iter_str + '.txt', geo_forward.ss)
    names = ['Invert', 'SSV_threshold', 'Gamma', 'Eig']
    inputs = [args.invert, args.ssv_threshold, args.gamma, args.eig]
    np.savetxt(txt_folder + 'input_settings_' + iter_str + '.csv', \
        [p for p in zip(names, inputs)], delimiter=',', fmt='%s')

if args.plot:
    # Extract geodesic paths
    geodesic_taus = np.loadtxt(txt_folder + 'geodesic_taus_' + iter_str + '.txt')
    geodesic_paths = np.loadtxt(txt_folder + 'geodesic_paths_' + iter_str + '.txt')
    geodesic_velocities = np.loadtxt(txt_folder + 'geodesic_velocities_' + iter_str + '.txt')

    plot.plot_state_vars(funcs.rhs_full, geodesic_paths[-1], state_labels=states)
    plot.eigenvals(geodesic_velocities)
    plot.geodesics(geodesic_taus, geodesic_paths, geodesic_velocities)
    plot.obs_points(geodesic_paths[-1], funcs.voltage, I_data=np.loadtxt(txt_folder + 'current_i1.txt'))
    if args.show_rates:
        plot.rates(geodesic_paths, funcs.Rates, rates_str)
        plot.rate_trajectory(geodesic_paths, funcs.Rates, rates_str, 0)
        
    plot.show()
