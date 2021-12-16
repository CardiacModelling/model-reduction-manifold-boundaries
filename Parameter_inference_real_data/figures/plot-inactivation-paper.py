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
parser.add_argument('--protocol', type=int, default=1, metavar='N',
                    help='which protocol is used to fit the data: 1 for staircase #1, 2 for sine wave')
parser.add_argument("--show", action='store_true',
                    help="whether to show figures instead of saving them",
                    default=False)
parser.add_argument('--params', type=int, default=1, metavar='N',
                    help='which params to use')
parser.add_argument('--figsize1', type=float, nargs='+', default=[3.5, 2], \
                    help='Figure 1 size in x and y, e.g. --figsize1 6 4')
parser.add_argument('--figsize2', type=float, nargs='+', default=[2.5, 3.5], \
                    help='Figure 2 size in x and y, e.g. --figsize2 2.5 3.5')
parser.add_argument("--grid", action='store_true',
                    help="whether to add grid to figures or not",
                    default=False)
args = parser.parse_args()

cell = args.cell

pr_steps = [
    (6000, 1000), #-140
    (26000, 1000), #-110
    (46000, 1000), #-80
    (66000, 1000), #-50
    (86000, 1000), #-20
    (106000, 1000), #10
    (126000, 1000), #40
]

pr_voltages = np.array([-140, -110, -80, -50, -20, 10, 40])

pr_steps_deact = [
    (6025, 7000), #-140
    (26025, 7000), #-110
    (46025, 7000), #-80
    (66025, 7000), #-50
]

pr_voltages_deact = np.array([-140, -110, -80, -50])

#
# Simple IKr test script
#

# Get model
p = myokit.load_protocol('../model-and-protocols/inactivation-ramp.mmt')

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

print('Updating model to steady-state for ' + str(ss_V) + ' mV ')
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

e = myokit.DataLog.load_csv('../data/SFU-data/Inactivation/inactivation-WT-cell-' + str(cell) + '.csv').npview()

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
cmap = matplotlib.cm.get_cmap('winter_r')
norm = matplotlib.colors.Normalize(0, n_steps)

# Filtered simulated data
d = d.npview()
df = d.fold(2000)

# Filtered experimental data
e = e.npview()
e = e.fold(2000)

# Compute inactivation curve
v1, g1, voltages1, peaks1 = biomarkers.steady_state_inactivation(pr_steps=pr_steps, pr_voltages=pr_voltages, log=d, erev=ek)
v2, g2 = biomarkers.time_constant_of_deactivation(pr_steps_deact, pr_voltages_deact, log=d, erev=ek)

g1n = g1 / np.max(g1)

a = np.loadtxt('Biomarkers/inact_ss_exp_WT_cell' + str(cell) + '.txt', unpack=True)
b = np.loadtxt('Biomarkers/inact_deact_exp_WT_cell' + str(cell) + '.txt', unpack=True)

exp_ss_v = a[0]
exp_ss_i = a[1]
exp_ss_in = a[1] / np.max(a[1])

exp_deact_v = b[0]
exp_deact_i = b[1]

fig = pl.figure(1, figsize=args.figsize1, dpi=200)
ax1 = fig.add_subplot(1,2,1)
pl.title('Inactivation', fontsize=10)
ax1.set_xlabel('Voltage (mV)')
ax1.set_ylabel('Norm. current')
ax1.plot(v1, g1n, c='red', linestyle='--', lw=1)
pl.scatter(exp_ss_v, exp_ss_in, s=36, facecolors='none', edgecolors='#1f77b4', marker='o')
if args.grid:
    ax1.grid(True)
ax2 = fig.add_subplot(1,2,2)
pl.title('Deactivation', fontsize=10)
ax2.set_xlabel('Voltage (mV)')
ax2.set_ylabel(r'$\tau$ (ms)')
ax2.plot(v2, g2, c='red', linestyle='--', lw=1)
pl.scatter(exp_deact_v, exp_deact_i, s=36, facecolors='none', edgecolors='#1f77b4', marker='o')
ax2.legend(['Model', 'Expt.'], loc='best', fontsize=8)
if args.grid:
    ax2.grid(True)
ax2.set_yscale('log')
pl.tight_layout()
if args.show == True:
    pl.show(block=False)
else:
    filename = 'All_figures/Biomarkers-inactivation-model-' + model_str + '-fit-' + protocol_str + '-paper'
    pl.savefig(filename + '.svg')

fig = pl.figure(2, figsize=args.figsize2, dpi=200)
ax1 = fig.add_subplot(311)
ax1.set_xlim([500, 1600])
if args.grid:
    ax1.grid(True)
ax1.set_ylabel( 'Voltage (mV)' )
ax2 = fig.add_subplot(312)
ax2.set_xlim([500, 1600])
if args.grid:
    ax2.grid(True)
ax2.set_ylabel( 'Predicted\ncurrent (nA)' )
ax3 = fig.add_subplot(313, sharey=ax2)
ax3.set_xlim([500, 1600])
if args.grid:
    ax3.grid(True)
ax3.set_xlabel( 'Time (ms)' )
ax3.set_ylabel( 'Experimental\ncurrent (nA)' )
for k in range(n_steps):
    [label.set_visible(False) for label in ax1.get_xticklabels()]
    ax1.plot(df.time(), df['voltage',k], color=cmap(norm(k)), lw=1)
    [label.set_visible(False) for label in ax2.get_xticklabels()]
    ax2.plot(df.time(), df['current',k], color=cmap(norm(k)), lw=1)
    ax3.plot(e.time(), e['current',k], color=cmap(norm(k)), lw=1)
pl.tight_layout()

if args.show == True:
    pl.show()
else:
    filename = 'Inactivation-model-' + model_str + '-fit-' + protocol_str + '-paper'
    pl.savefig('All_figures/' + filename + '.svg')
