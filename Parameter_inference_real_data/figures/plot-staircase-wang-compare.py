import myokit
import myokit.pacing as pacing
import numpy as np
import matplotlib
import matplotlib.pyplot as pl
import myokit.lib.markov as markov
import pints
import argparse
import os
import sys

# Load project modules
sys.path.append(os.path.abspath(os.path.join('../', 'python')))
import cells
import data

# Check input arguments
parser = argparse.ArgumentParser(
    description='Plot model and experimental data')
parser.add_argument('--cell', type=int, default=2, metavar='N',
                    help='repeat number : 1, 2, 3, 4, 5, 6')
parser.add_argument('--model', type=str, default='wang', metavar='N',
                    help='which model to use')
parser.add_argument('--protocol', type=int, default=1, metavar='N',
                    help='which protocol is used to fit the data: 1 for staircase #1, 2 for sine wave')
parser.add_argument("--show", action='store_true',
                    help="whether to show figures instead of saving them",
                    default=False)
parser.add_argument("--grid", action='store_true',
                    help="whether to add grid to figures or not",
                    default=False)
args = parser.parse_args()

cell = args.cell

#
# Simple IKr test script
#

# Get model
p = myokit.load_protocol('../model-and-protocols/staircase1.mmt')

current = 'ikr.IKr'

ek = cells.ek(cell)

print('Reversal potential ' + str(ek) + ' mV')

model_str_dict = {'mazhari': 'Mazhari', 'mazhari-reduced': 'Maz-red', 'wang': 'Wang', 'wang-r3': 'Wang-r3', \
    'wang-r4': 'Wang-r4', 'wang-r6': 'Wang-r6'}
model_str = model_str_dict[args.model]

if args.protocol == 1:
    protocol_str = 'staircase1'
else:
    protocol_str = 'sine-wave'

# Run simulation
dt = 0.1

fig, (a0, a1) = pl.subplots(2, 1, gridspec_kw={'height_ratios': [1, 2]}, figsize=(4, 2.5), dpi=200)
a0.set_ylim([-140, 50])
a0.set_ylabel( '(mV)' )
if args.grid:
    a0.grid(True)
a0.set_xlim([0, 15400])
[label.set_visible(False) for label in a0.get_xticklabels()]
a1.set_xlabel( 'Time (ms)' )
a1.set_ylabel( 'Current (nA)' )
if args.grid:
    a1.grid(True)
a1.set_xlim([0, 15400])

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
param_file = '../cmaesfits/' + model_str + '-model-fit-' + protocol_str + '-iid-noise'

params = [30, 1]
for n, param in enumerate(params):

    m = myokit.load_model('../model-and-protocols/' + args.model + '-ikr-markov.mmt')
    if args.model == 'mazhari':
        states = ['ikr.c2', 'ikr.c3', 'ikr.o', 'ikr.i']
    elif args.model == 'mazhari-reduced':
        states = ['ikr.c1', 'ikr.c3', 'ikr.o', 'ikr.i']
    elif args.model == 'wang':
        states = ['ikr.c2', 'ikr.c3', 'ikr.o', 'ikr.i']
    elif args.model == 'wang-r3':
        states = ['ikr.c3', 'ikr.o', 'ikr.i']
    elif args.model == 'wang-r4':
        states = ['ikr.c3', 'ikr.o', 'ikr.i']
    elif args.model == 'wang-r6':
        states = ['ikr.c3o', 'ikr.i']
    else:
        pass
    n_params = int(m.get('misc.n_params').value())
    m = markov.convert_markov_models_to_full_ode_form(m)

    # Set steady state potential
    LJP = m.get('misc.LJP').value()
    ss_V = -80 - LJP

    x_found = np.loadtxt(param_file + '-parameters-' + str(param) + '.txt', unpack=True)

    parameters = []
    for i in range(n_params):
        parameters.append('ikr.p'+str(i+1))

    d = ['engine.time', 'membrane.V', 'ikr.IKr']

    # Run simulation
    m.get('nernst.EK').set_rhs(ek)

    print('Updating model to steady-state for ' + str(ss_V) + ' mV')
    m.get('membrane.V').set_label('membrane_potential')

    mm = markov.LinearModel.from_component(m.get('ikr'))

    x = mm.steady_state(ss_V, x_found)
    for i in range(len(states)):
        m.get(states[i]).set_state_value(x[i])
    print(x)
        
    m.get('membrane.V').set_rhs(
       'piecewise(engine.time <= 1236.2,'
       + ' -80 - misc.LJP,'
       + 'engine.time >= 14410.1 and engine.time < 14510.0,'
       + ' -70 - misc.LJP'
       + ' - 0.4 * (engine.time - 14410.1)'
       + ', engine.pace - misc.LJP)')

    s = myokit.Simulation(m, p)
    s.set_tolerance(1e-8, 1e-8)
    s.set_max_step_size(0.1)

    # Update model parameters
    for i in range(n_params):
        s.set_constant('ikr.p'+str(i+1), x_found[i])

    d = s.run(p.characteristic_time(), log_interval=dt, log=d)

    signals2 = [d.time(), d['ikr.IKr'], d['membrane.V']]
    d = myokit.DataLog()
    d.set_time_key('time')
    d['time'] = signals2[0]
    d['current'] = signals2[1]
    d['voltage'] = signals2[2]

    # Filtered simulated data
    d = d.npview()
    if n == 0:
        a0.plot(d.time(), d['voltage'], color='grey', linewidth=1)
    a1.plot(d.time(), d['current'], color='red' if n == 0 else 'midnightblue', label='Parameter set ' + str(n+1), linewidth=1, linestyle='dashed' if n > 0 else 'solid')

a1.legend(fontsize=8)
pl.tight_layout()

if args.show == True:
    pl.show()
else:
    filename = 'Staircase-Wang-model-compare-fit-' + protocol_str + '-iid-noise'
    pl.savefig('All_figures/' + filename + '.svg')
