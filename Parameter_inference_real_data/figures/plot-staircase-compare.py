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
parser.add_argument('--protocol', type=int, default=1, metavar='N',
                    help='which protocol is used to fit the data: 1 for staircase #1, 2 for sine wave')
parser.add_argument("--show", action='store_true',
                    help="whether to show figures instead of saving them",
                    default=False)
parser.add_argument('--params', type=int, default=1, metavar='N',
                    help='which params to use')
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

if args.protocol == 1:
    protocol_str = 'staircase1'
else:
    protocol_str = 'sine-wave'

# Run simulation
dt = 0.1

fig, (a0, a1) = pl.subplots(2, 1, gridspec_kw={'height_ratios': [1, 3]}, figsize=(7, 5), dpi=200)
a0.set_ylabel( 'Voltage (mV)' )
a0.grid(True)
[label.set_visible(False) for label in a0.get_xticklabels()]
a1.set_xlabel( 'Time (ms)' )
a1.set_ylabel( 'Current (nA)' )
a1.grid(True)

e = myokit.DataLog.load_csv('../data/SFU-data/Staircase/staircase1-WT-cell-' + str(cell) + '.csv').npview()

# Apply capacitance filtering for experiment and simulated data
signals = [e.time(), e['current']]
voltage = 'voltage' in e
if voltage:
    signals.append(e['voltage'])
signals = data.capacitance(p, dt, *signals)

e = myokit.DataLog()
e.set_time_key('time')
e['time'] = signals[0]
e['current'] = signals[1] / 1000 # Correct units
if voltage:
    e['voltage'] = signals[2]

# Filtered experimental data
e = e.npview()

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

models = ['wang', 'wang-r6']
for n, model in enumerate(models):

    m = myokit.load_model('../model-and-protocols/' + model + '-ikr-markov.mmt')
    if model == 'mazhari':
        model_str = 'Mazhari'
        states = ['ikr.c2', 'ikr.c3', 'ikr.o', 'ikr.i']
    elif model == 'mazhari-reduced':
        model_str = 'Maz-red'
        states = ['ikr.c1', 'ikr.c3', 'ikr.o', 'ikr.i']
    elif model == 'wang':
        model_str = 'Wang'
        states = ['ikr.c2', 'ikr.c3', 'ikr.o', 'ikr.i']
    elif model == 'wang-r3':
        model_str = 'Wang-r3'
        states = ['ikr.c3', 'ikr.o', 'ikr.i']
    elif model == 'wang-r4':
        model_str = 'Wang-r4'
        states = ['ikr.c3', 'ikr.o', 'ikr.i']
    elif model == 'wang-r6':
        model_str = 'Wang-r6'
        states = ['ikr.c3o', 'ikr.i']
    else:
        pass
    n_params = int(m.get('misc.n_params').value())
    m = markov.convert_markov_models_to_full_ode_form(m)

    # Set steady state potential
    LJP = m.get('misc.LJP').value()
    ss_V = -80 - LJP

    x_found = np.loadtxt('../cmaesfits/' + model_str + '-model-fit-' + protocol_str + '-iid-noise-parameters-' + str(args.params) + '.txt', unpack=True)

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
        a0.plot(d.time(), d['voltage'], color='grey')
    if n == 0:
        a1.plot(e.time(), e['current'], color='silver', label='Experiment')
    a1.plot(d.time(), d['current'], color=colors[n], label=model_str + ' model', linestyle='dashed' if n > 0 else 'solid')

a1.legend()
pl.tight_layout()

if args.show == True:
    pl.show()
else:
    filename = 'Staircase-model-compare-fit-' + protocol_str + '-iid-noise'
    pl.savefig('All_figures/' + filename + '.png')
