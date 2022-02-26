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
iter_str = 'i8'

par = Params()

# Define number of parameters and variables
par.n_state_vars = 2
par.n_params = 8
par.observed_variable = 0

txt_folder = 'txt_files/'
os.makedirs(txt_folder, exist_ok=True)  # Create CMA-ES output destination folder

# Choose starting parameters
para = np.loadtxt(txt_folder + 'params_i7.txt')

b_rates = [0, 4, 6]
for x, _ in enumerate(para):
    if x in b_rates:
        para[x] = np.log(para[x])

np.savetxt(txt_folder + 'settings_' + iter_str + '.txt', (par.n_state_vars, par.n_params, par.observed_variable), \
    fmt='%d')

# Create symbols for symbolic functions
x, y, v = GetSymbols(par)

# Define system equations and initial conditions
# C12, O, I 
C12toO = se.exp(x[1]) * se.exp(se.exp(x[6]) * v) 
OtoC12 = se.exp(x[2]) * (se.exp(-se.exp(x[0]) * v))
OtoI = se.exp(x[3]) * se.exp(se.exp(x[4]) * v) 
ItoO = se.exp(x[5]) 
conductance = se.exp(x[7])

rates = [C12toO, OtoC12, OtoI, ItoO]
rates_str = ['C12toO', 'OtoC12', 'OtoI', 'ItoO']
states = ['C12', 'O', 'I'] # C12 is calculated from the other states


rhs = [-(OtoC12 + OtoI) * y[0] + C12toO * (1 - y[0] - y[1]) + ItoO * y[1],
       -ItoO * y[1] + OtoI * y[0] ]
           
# Write in matrix form taking y = ([O], [I])^T
# RHS = A * y + B

A = se.Matrix([[-OtoC12 - OtoI - C12toO, -C12toO + ItoO], \
               [OtoI,                        -ItoO]])
B = se.Matrix([ C12toO, 0])

ICs = [ 0.0, 0.0]


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
    plot.eigenvals(geodesic_velocities, grid=False, figx=6, figy=3)
    plot.geodesics(geodesic_taus, geodesic_paths, geodesic_velocities)
    plot.obs_points(geodesic_paths[-1], funcs.voltage, I_data=np.loadtxt(txt_folder + 'current_i1.txt'))
    if args.show_rates:
        plot.rates(geodesic_paths, funcs.Rates, rates_str)
        plot.rate_trajectory(geodesic_paths, funcs.Rates, rates_str, 0)
        
    plot.show()
