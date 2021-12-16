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

pl.rcParams['axes.axisbelow'] = True

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
                    help='Figure size in x and y, e.g. --figsize1 2.5 3.5')
parser.add_argument('--figsize2', type=float, nargs='+', default=[2.5, 3.5], \
                    help='Figure size in x and y, e.g. --figsize2 2.5 3.5')
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

fig1 = pl.figure(1, figsize=args.figsize1, dpi=200)
ax11 = fig1.add_subplot(1,2,1)
pl.title('Inactivation', fontsize=10)
ax11.set_xlabel('Voltage (mV)')
ax11.set_ylabel('Norm. current')
if args.grid:
    ax11.grid(True)
ax12 = fig1.add_subplot(1,2,2)
pl.title('Deactivation', fontsize=10)
ax12.set_xlabel('Voltage (mV)')
ax12.set_ylabel(r'$\tau$ (ms)')
if args.grid:
    ax12.grid(True)
ax12.set_yscale('log')

fig2 = pl.figure(2, figsize=args.figsize2, dpi=200)
ax21 = fig2.add_subplot(311)
[label.set_visible(False) for label in ax21.get_xticklabels()]
ax21.set_xlim([500, 1600])
if args.grid:
    ax21.grid(True)
ax21.set_ylabel( 'Voltage (mV)' )
ax22 = fig2.add_subplot(312)
[label.set_visible(False) for label in ax22.get_xticklabels()]
ax22.set_xlim([500, 1600])
if args.grid:
    ax22.grid(True)
ax22.set_ylabel( 'Predicted\ncurrent (nA)' )
ax23 = fig2.add_subplot(313, sharey=ax22)
ax23.set_xlim([500, 1600])
if args.grid:
    ax23.grid(True)
ax23.set_xlabel( 'Time (ms)' )
ax23.set_ylabel( 'Experimental\ncurrent (nA)' )

colors = ['orange', 'red', 'limegreen']
linestyles = ['solid', 'dashed', 'dashdot']

models = ['wang', 'wang-r6', 'wang-r8']
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
    elif model == 'wang-r1':
        model_str = 'Wang-r1'
        states = ['ikr.c2', 'ikr.c3', 'ikr.o', 'ikr.i']
    elif model == 'wang-r2':
        model_str = 'Wang-r2'
        states = ['ikr.c3', 'ikr.o', 'ikr.i']
    elif model == 'wang-r3':
        model_str = 'Wang-r3'
        states = ['ikr.c3', 'ikr.o', 'ikr.i']
    elif model == 'wang-r4':
        model_str = 'Wang-r4'
        states = ['ikr.c3', 'ikr.o', 'ikr.i']
    elif model == 'wang-r5':
        model_str = 'Wang-r5'
        states = ['ikr.o', 'ikr.i']
    elif model == 'wang-r6':
        model_str = 'Wang-r6'
        states = ['ikr.o', 'ikr.i']
    elif model == 'wang-r7':
        model_str = 'Wang-r7'
        states = ['ikr.o', 'ikr.i']
    elif model == 'wang-r8':
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

    # print('Updating model to steady-state for ' + str(ss_V) + ' mV ')
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
    cmap = matplotlib.cm.get_cmap('viridis')
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

    ax11.plot(v1, g1n, linestyle=linestyles[n], color=colors[n], lw=1, label=model_str)
    if n == 0:
        ax11.scatter(exp_ss_v, exp_ss_in, s=36, facecolors='none', edgecolors='#1f77b4', marker='o', label='Expt.')
    ax12.plot(v2, g2, linestyle=linestyles[n], color=colors[n], lw=1)
    if n == 0:
        ax12.scatter(exp_deact_v, exp_deact_i, s=36, facecolors='none', edgecolors='#1f77b4', marker='o')

    for k in range(n_steps):
        if n == 0:
            ax21.plot(df.time(), df['voltage',k], color='#1f77b4', lw=1)
            ax23.plot(e.time(), e['current',k], color='#1f77b4', lw=1)
        ax22.plot(df.time(), df['current',k], color=colors[n], lw=1, linestyle=linestyles[n], label=model_str if k == 0 else '')

# ax11.legend(fontsize=8)
# ax22.legend(fontsize=8)
fig1.tight_layout()
fig2.tight_layout()

if args.show == True:
    pl.show()
else:
    filename1 = 'All_figures/Biomarkers-inactivation-models-compare-fit-' + protocol_str + '-iid-noise'
    filename2 = 'All_figures/Inactivation-models-compare-fit-' + protocol_str + '-iid-noise'
    fig1.savefig(filename1 + '.svg')
    fig2.savefig(filename2 + '.svg')
