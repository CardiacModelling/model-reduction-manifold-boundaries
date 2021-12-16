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

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

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
parser.add_argument('--params', type=int, default=1, metavar='N',
                    help='which params to use')
parser.add_argument('--figsize', type=float, nargs='+', default=[4, 3.5], \
                    help='Figure size in x and y, e.g. --figsize2 2.5 3.5')
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

if args.protocol == 1:
    protocol_str = 'staircase1'
else:
    protocol_str = 'sine-wave'

# Run simulation
dt = 0.1

m = myokit.load_model('../model-and-protocols/' + args.model + '-ikr-markov.mmt')
if args.model == 'mazhari':
    model_str = 'Mazhari'
    states = ['ikr.c2', 'ikr.c3', 'ikr.o', 'ikr.i']
elif args.model == 'mazhari-reduced':
    model_str = 'Maz-red'
    states = ['ikr.c1', 'ikr.c3', 'ikr.o', 'ikr.i']
elif args.model == 'wang':
    model_str = 'Wang'
    states = ['ikr.c2', 'ikr.c3', 'ikr.o', 'ikr.i']
elif args.model == 'wang-r3':
    model_str = 'Wang-r3'
    states = ['ikr.c3', 'ikr.o', 'ikr.i']
elif args.model == 'wang-r4':
    model_str = 'Wang-r4'
    states = ['ikr.c3', 'ikr.o', 'ikr.i']
elif args.model == 'wang-r6':
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
# s.set_max_step_size(0.1)

# Update model parameters
for i in range(n_params):
    s.set_constant('ikr.p'+str(i+1), x_found[i])

d = s.run(p.characteristic_time(), log_interval=dt, log=d)

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

signals2 = [d.time(), d['ikr.IKr'], d['membrane.V']]
d = myokit.DataLog()
d.set_time_key('time')
d['time'] = signals2[0]
d['current'] = signals2[1]
d['voltage'] = signals2[2]

# Filtered simulated data
d = d.npview()

# Filtered experimental data
e = e.npview()

fig, (a0, a1) = pl.subplots(2, 1, gridspec_kw={'height_ratios': [1, 3]}, figsize=args.figsize, constrained_layout=True)
a0.set_xlim([0, 15400])
a0.set_ylim([-140, 50])
a0.set_ylabel( '(mV)' )
a0.plot(d.time(), d['voltage'], color='grey', lw=1)
if args.grid:
    a0.grid(True)
[label.set_visible(False) for label in a0.get_xticklabels()]
a1.set_xlim([0, 15400])
a1.set_ylim([-10, 2.5])
a1.set_xlabel( 'Time (ms)' )
a1.set_ylabel( 'Current (nA)' )
a1.plot(e.time(), e['current'], color='#1f77b4', lw=1)
a1.plot(d.time(), d['current'], color='red', lw=1)
if args.grid:
    a1.grid(True)

axins = zoomed_inset_axes(a1, 5, loc='lower center') # zoom-factor: 5
axins.plot(e.time(), e['current'], color='#1f77b4', lw=1)
axins.plot(d.time(), d['current'], color='red', lw=1)
x1, x2, y1, y2 = 4500, 7300, 0.5, 1.9 # specify the limits
# x1, x2, y1, y2 = 6800, 9600, 0.4, 1.8 # specify the limits
# x1, x2, y1, y2 = 9600, 12400, 0.8, 2.2 # specify the limits
axins.set_xlim(x1, x2) # apply the x-limits
axins.set_ylim(y1, y2) # apply the y-limits
pl.yticks(visible=False)
pl.xticks(visible=False)
mark_inset(a1, axins, loc1=2, loc2=1, fc="none", ec="0.5")
axins.legend(['Expt.', 'Model'], loc='best', fontsize=8)

if args.show == True:
    pl.show()
else:
    filename = 'Staircase-model-' + model_str + '-fit-' + protocol_str + '-paper'
    pl.savefig('All_figures/' + filename + '.svg')
