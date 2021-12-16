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
import biomarkers

# Check input arguments
parser = argparse.ArgumentParser(
    description='Plot model and experimental data')
parser.add_argument('--cell', type=int, default=2, metavar='N',
                    help='repeat number : 1, 2, 3, 4, 5, 6')
parser.add_argument('--model', type=str, default='wang', metavar='N',
                    help='which model to use')
parser.add_argument('--repeats', type=int, default=25, metavar='N',
                    help='number of CMA-ES runs from different initial guesses')
parser.add_argument('--protocol', type=int, default=1, metavar='N',
                    help='which protocol is used to fit the data: 1 for staircase #1, 2 for sine wave')
parser.add_argument("--show", action='store_true',
                    help="whether to show figures instead of saving them",
                    default=False)
parser.add_argument('--params', type=int, default=1, metavar='N',
                    help='which params to use')
args = parser.parse_args()

cell = args.cell

pr_steps = [
    (16000, 1000), #-50
    (46000, 1000), #-35
    (76000, 1000), #-20
    (106000, 1000), #-5
    (136000, 1000), #10
    (166000, 1000), #25
    (196000, 1000), #40
]
pr_voltages = np.array([-50, -35, -20, -5, 10, 25, 40])

#
# Simple IKr test script
#

# Get model
p = myokit.load_protocol('../model-and-protocols/activation-ramp.mmt')

current = 'ikr.IKr'

ek = cells.ek(cell)

print('Reversal potential ' + str(ek) + ' mV')

if args.protocol == 1:
    protocol_str = 'staircase1'
elif args.protocol == 2:
    protocol_str = 'sine-wave'
elif args.protocol == 2:
    protocol_str = 'complex-AP'
else:
    protocol_str = 'staircase1-cAP'

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
elif args.model == 'wang-r1':
    model_str = 'Wang-r1'
    states = ['ikr.c2', 'ikr.c3', 'ikr.o', 'ikr.i']
elif args.model == 'wang-r2':
    model_str = 'Wang-r2'
    states = ['ikr.c3', 'ikr.o', 'ikr.i']
elif args.model == 'wang-r3':
    model_str = 'Wang-r3'
    states = ['ikr.c3', 'ikr.o', 'ikr.i']
elif args.model == 'wang-r4':
    model_str = 'Wang-r4'
    states = ['ikr.c3', 'ikr.o', 'ikr.i']
elif args.model == 'wang-r5':
    model_str = 'Wang-r5'
    states = ['ikr.c3o', 'ikr.i']
elif args.model == 'wang-r6':
    model_str = 'Wang-r6'
    states = ['ikr.c3o', 'ikr.i']
elif args.model == 'wang-r7':
    model_str = 'Wang-r7'
    states = ['ikr.c3o', 'ikr.i']
elif args.model == 'wang-r8':
    model_str = 'Wang-r8'
    states = ['ikr.o']
else:
    pass
n_params = int(m.get('misc.n_params').value())
m = markov.convert_markov_models_to_full_ode_form(m)
    
# Number of steps in pacing protocol
n_steps = 7

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

m.get('membrane.V').set_rhs('engine.pace - misc.LJP')

s = myokit.Simulation(m, p)
s.set_tolerance(1e-8, 1e-8)
s.set_max_step_size(0.1)

# Update model parameters
for i in range(n_params):
    s.set_constant('ikr.p'+str(i+1), x_found[i])

d = s.run(p.characteristic_time(), log_interval=dt, log=d)

e = myokit.DataLog.load_csv('../data/SFU-data/Activation/activation-WT-cell-' + str(cell) + '.csv').npview()

# Apply capacitance filtering for experiment and simulated data
signals = [e.time(), e['current']]
voltage = 'voltage' in e
if voltage:
    signals.append(e['voltage'])
signals = data.capacitance(p, dt, *signals)

e = myokit.DataLog()
e.set_time_key('time')
e['time'] = signals[0]
e['current'] = signals[1] / 1000
if voltage:
    e['voltage'] = signals[2]

signals2 = [d.time(), d['ikr.IKr'], d['membrane.V']]
d = myokit.DataLog()
d.set_time_key('time')
d['time'] = signals2[0]
d['current'] = signals2[1]
d['voltage'] = signals2[2]

# Create colormap for plotting
cmap = matplotlib.cm.get_cmap('viridis')
norm = matplotlib.colors.Normalize(0, n_steps)

# Filtered simulated data
d = d.npview()
df = d.fold(3000)

# Filtered experimental data
e = e.npview()
e = e.fold(3000)

# Compute activation curve
v1, g1 = biomarkers.steady_state_activation(pr_steps, pr_voltages, log=d)
v2, g2 = biomarkers.iv_curve_activation(pr_steps, pr_voltages, log=d)

g1n = g1 / np.max(g1)
g2n = g2 / np.max(g2)

import scipy.interpolate as sp

interp = sp.interp1d(v1, g1, kind='cubic')
interp2 = sp.interp1d(v2, g2, kind='cubic')
xnew = np.linspace(-50, 40, 50)

a = np.loadtxt('Biomarkers/act_ss_exp_WT_cell' + str(cell) + '.txt', unpack=True)
b = np.loadtxt('Biomarkers/act_iv_exp_WT_cell' + str(cell) + '.txt', unpack=True)

exp_ss_v = a[0]
exp_ss_i = a[1]
exp_ss_in = a[1] / np.max(a[1])

exp_iv_v = b[0]
exp_iv_i = b[1]
exp_iv_in = b[1] / np.max(b[1])

fig = pl.figure(1, figsize=(6, 3), dpi=200)
ax1 = fig.add_subplot(1,2,1)
pl.title('Activation')
ax1.set_xlabel('Voltage (mV)')
ax1.set_ylabel('Normalised current')
ax1.plot(v1, g1n, c='dodgerblue', linestyle='--')
pl.scatter(exp_ss_v,exp_ss_in, s=100, facecolors='none', edgecolors='limegreen',marker='o')
ax1.legend([model_str + ' model', 'Experiment'], loc = 'best')
ax1.grid(True)
ax2 = fig.add_subplot(1,2,2)
pl.title('I-V relation')
ax2.set_xlabel('Voltage (mV)')
ax2.plot(v2, g2n, c='dodgerblue', linestyle='--')
pl.scatter(exp_iv_v,exp_iv_in, s=100, facecolors='none', edgecolors='limegreen',marker='o')
ax2.grid(True)
pl.tight_layout()
if args.show == True:
    pl.show()
else:
    filename = 'All_figures/Biomarkers-activation-model-' + model_str + '-fit-' + protocol_str + '-iid-noise'
    pl.savefig(filename + '.png')

fig = pl.figure(2, figsize=(7, 5), dpi=200)
ax1 = fig.add_subplot(311)
ax1.grid(True)
# pl.title('Cell ' + str(cell))
ax1.set_ylabel( 'Voltage (mV)' )
ax2 = fig.add_subplot(312)
ax2.grid(True)
ax2.set_ylabel( 'Current (nA)' )
ax3 = fig.add_subplot(313, sharey=ax2)
ax3.grid(True)
ax3.set_xlabel( 'Time (ms)' )
ax3.set_ylabel( 'Current (nA)' )
for k in range(n_steps):
    [label.set_visible(False) for label in ax1.get_xticklabels()]
    ax1.plot(df.time(), df['voltage',k], color=cmap(norm(k)))
    [label.set_visible(False) for label in ax2.get_xticklabels()]
    ax2.plot(df.time(), df['current',k], color=cmap(norm(k)))
    ax2.legend([model_str + ' model'], loc = 'upper left')
    ax3.plot(e.time(), e['current',k], color=cmap(norm(k)))
    ax3.legend(['Experiment'], loc = 'upper left')
pl.tight_layout()

if args.show == True:
    pl.show()
else:
    filename = 'Activation-model-' + model_str + '-fit-' + protocol_str + '-iid-noise'
    pl.savefig('All_figures/' + filename + '.png')
