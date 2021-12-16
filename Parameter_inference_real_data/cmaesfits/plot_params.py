import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse

# Check input arguments
parser = argparse.ArgumentParser(
    description='Plot model and experimental data')
parser.add_argument('--cell', type=int, default=2, metavar='N',
                    help='repeat number : 1, 2, 3, 4, 5, 6')
parser.add_argument('--model', type=str, default='wang', metavar='N',
                    help='which model to use')
parser.add_argument('--protocol', type=int, default=1, metavar='N',
                    help='which protocol is used to fit the data: 1 for staircase #1, 2 for sine wave')
parser.add_argument('--repeats', type=int, default=50, metavar='N',
                    help='number of CMA-ES runs from different initial guesses')
parser.add_argument("--show", action='store_true',
                    help="whether to show figures instead of saving them",
                    default=False)
args = parser.parse_args()

# Get model string and params
model_str_dict = {'mazhari': 'Mazhari', 'mazhari-reduced': 'Maz-red', 'wang': 'Wang', 'wang-r1': 'Wang-r1', \
    'wang-r2': 'Wang-r2', 'wang-r3': 'Wang-r3', 'wang-r4': 'Wang-r4', 'wang-r5': 'Wang-r5', 'wang-r6': 'Wang-r6', \
    'wang-r7': 'Wang-r7', 'wang-r8': 'Wang-r8'}
model_str = model_str_dict[args.model]

if args.protocol == 1:
    protocol_str = 'staircase1'
else:
    protocol_str = 'sine-wave'

param_file = model_str + '-model-fit-' + protocol_str + '-iid-noise'

n_params = len(np.loadtxt(param_file + '.txt'))
params = np.linspace(1, n_params, n_params, dtype=int)
param_labels = ['p' + str(i) for i in params]

best_score = np.loadtxt(param_file + '-errors.txt')[0]

n_fits = 30
# for i in range(args.repeats):
#     if (np.loadtxt(param_file + '-errors.txt')[i] / best_score) < 1.05:
#         n_fits += 1
# print('# satisfactory fits = ' + str(n_fits))

# Create colormap for plotting
cmap = matplotlib.cm.get_cmap('winter_r')
norm = matplotlib.colors.Normalize(0, n_fits-1)

fig = plt.figure(figsize=(4, 2.5), dpi=200)
ax1 = fig.add_subplot(111)
ax1.set_yscale('log')
for i in range(n_fits):
    # Go from worst (green) to best (blue)
    ax1.scatter(params, np.loadtxt(param_file + '-parameters-' + str(n_fits-i) + '.txt'), color=cmap(norm(i)))
ax1.set_xticks(params)
ax1.set_xticklabels(param_labels, rotation=45)
plt.tight_layout()

if args.show == True:
    plt.show()
else:
    filename = 'Params-model-' + model_str + '-protocol-' + protocol_str + '-iid-noise'
    plt.savefig('../figures/All_figures/' + filename + '.svg')
