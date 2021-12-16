#!/usr/bin/env python3
#
from __future__ import division, print_function
import os
import sys
import pints
import numpy as np
import myokit
import argparse

if __name__=="__main__":

    import platform
    parallel = True
    if platform.system() == 'Darwin':
        import multiprocessing
        multiprocessing.set_start_method('fork')
    elif platform.system() == 'Windows':
        parallel = False

    # Check input arguments
    parser = argparse.ArgumentParser(
        description='Fit all the hERG models to sine wave data')
    parser.add_argument('--cell', type=int, default=2, metavar='N',
                        help='repeat number : 1, 2, 3, 4, 5, 6')
    parser.add_argument('--model', type=str, default='wang', metavar='N',
                        help='which model to use')
    parser.add_argument('--repeats', type=int, default=25, metavar='N',
                        help='number of CMA-ES runs from different initial guesses')
    parser.add_argument('--protocol', type=int, default=1, metavar='N',
                        help='which protocol is used to fit the data: 1 for staircase #1, \
                        2 for sine wave, 3 for complex AP')
    parser.add_argument("--big_pop_size", action='store_true', default=False,
                        help="whether to use big population size of 100 rather than default")
    args = parser.parse_args()

    cell = args.cell

    # Load project modules
    sys.path.append(os.path.abspath(os.path.join('python')))
    import priors
    import cells
    import transformation
    import data
    import model

    # Get model string and params
    if args.model == 'mazhari':
        model_str = 'Mazhari'
        x_found = np.loadtxt('cmaesfits/parameter-sets/mazhari-params.txt', unpack=True)
    elif args.model == 'mazhari-reduced':
        model_str = 'Maz-red'
        x_found = np.loadtxt('cmaesfits/parameter-sets/mazhari-reduced-params.txt', unpack=True)
    elif args.model == 'wang':
        model_str = 'Wang'
        x_found = np.loadtxt('cmaesfits/parameter-sets/wang-params.txt', unpack=True)   
    elif args.model == 'wang-r1':
        model_str = 'Wang-r1'
        x_found = np.loadtxt('cmaesfits/parameter-sets/wang-r1-params.txt', unpack=True)   
        for i in {0, 2, 4, 5, 6, 8, 13}:
            x_found[i] = np.exp(x_found[i])
    elif args.model == 'wang-r2':
        model_str = 'Wang-r2'
        x_found = np.loadtxt('cmaesfits/parameter-sets/wang-r2-params.txt', unpack=True)   
        for i in {0, 2, 4, 5, 6, 8, 12}:
            x_found[i] = np.exp(x_found[i])
    elif args.model == 'wang-r3':
        model_str = 'Wang-r3'
        x_found = np.loadtxt('cmaesfits/parameter-sets/wang-r3-params.txt', unpack=True)   
        for i in {0, 2, 4, 5, 6, 8, 11}:
            x_found[i] = np.exp(x_found[i])
    elif args.model == 'wang-r4':
        model_str = 'Wang-r4'
        x_found = np.loadtxt('cmaesfits/parameter-sets/wang-r4-params.txt', unpack=True)   
        for i in {0, 1, 3, 4, 5, 7, 10}:
            x_found[i] = np.exp(x_found[i])
    elif args.model == 'wang-r5':
        model_str = 'Wang-r5'
        x_found = np.loadtxt('cmaesfits/parameter-sets/wang-r5-params.txt', unpack=True)   
        for i in {0, 2, 3, 4, 6, 9}:
            x_found[i] = np.exp(x_found[i])
    elif args.model == 'wang-r6':
        model_str = 'Wang-r6'
        x_found = np.loadtxt('cmaesfits/parameter-sets/wang-r6-params.txt', unpack=True)   
        for i in {1, 2, 3, 5, 8}:
            x_found[i] = np.exp(x_found[i])
    elif args.model == 'wang-r7':
        model_str = 'Wang-r7'
        x_found = np.loadtxt('cmaesfits/parameter-sets/wang-r7-params.txt', unpack=True)   
        for i in {1, 2, 3, 5, 7}:
            x_found[i] = np.exp(x_found[i])
    elif args.model == 'wang-r8':
        model_str = 'Wang-r8'
        x_found = np.loadtxt('cmaesfits/parameter-sets/wang-r8-params.txt', unpack=True)   
        for i in range(len(x_found)):
            x_found[i] = np.exp(x_found[i])
    else:
        pass

    trans = transformation.Transformation()

    # Fix seed
    np.random.seed(100)

    # Get string and protocols for each mutant
    # Set Ek based on cell-specific estimate
    ek = -93.04

    if args.protocol < 3:
        protocols = [args.protocol]
    else:
        protocols = [1, 3]

    if args.protocol == 1:
        protocol_str = 'staircase1'
    elif args.protocol == 2:
        protocol_str = 'sine-wave'
    elif args.protocol == 3:
        protocol_str = 'complex-AP'
    else:
        protocol_str = 'staircase1-cAP'

    x_initial = trans.transform(x_found, args.model)

    filename = model_str + '-model-fit-' + protocol_str + '-iid-noise'
    print('Selected model ' + model_str)
    print('Selected fitting protocol ' + protocol_str)
    print('Storing results to ' + filename + '.txt')

    # Define problems
    models = []
    for protocol in protocols:

        # Create protocol
        p = data.load_protocol_values(protocol)

        # Create forward models
        models.append(model.Model(
            p,
            EK=ek,
            which_model=args.model,
            sine_wave=(protocol==2),
            staircase1=(protocol==1),
            transformation=trans
        ))

    # Load data, create single output problems
    problems = []
    sigma_noise = []
    for k, protocol in enumerate(protocols):
        log = data.load(cell, protocol)
        time = log.time()
        current = log['current']/1000 #change units from pA to nA

        debug = False
        if debug:
            test = models[k].simulate(x_initial, time)
            print(np.max(test))
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(time, current)
            plt.plot(time, test)
            plt.grid(True)
            plt.show()

        # Estimate noise from the first 100 ms of data
        sigma_noise.append(np.std(current[:1000], ddof=1))
        problems.append(pints.SingleOutputProblem(models[k], time, current))
        del(log)

    # Define log-likelihood

    # Create log-posterior
    log_likelihoods = []
    for k, problem in enumerate(problems):
        print('Sigma noise', sigma_noise[k])
        log_likelihoods.append(pints.GaussianKnownSigmaLogLikelihood(problem, sigma_noise[k]))
    if(len(log_likelihoods) > 1):
        log_likelihood = pints.SumOfIndependentLogPDFs(log_likelihoods)
    else:
        log_likelihood = log_likelihoods[0]
    log_prior = priors.LogPrior(args.model, trans)
    log_posterior = pints.LogPosterior(log_likelihood, log_prior)
    f = log_posterior

    print('Score at initial parameters: ',
        f(x_initial))

    #
    # Run
    #
    b = myokit.Benchmarker()
    repeats = args.repeats
    params, scores = [], []
    times = []
    for i in range(repeats):
        print('Repeat ' + str(1 + i))

        # Choose random starting point
        if i == 0:
            q0 = x_initial
        else:
            q0 = log_prior.sample()    # Search space

        # Create optimiser
        opt = pints.OptimisationController(
            f, q0, method=pints.CMAES)
        if args.big_pop_size:
            opt.optimiser().set_population_size(100)
        opt.set_log_to_file(filename + '-log-' + str(i) + '.txt')
        opt.set_max_iterations(None)
        opt.set_parallel(parallel)

        # Run optimisation
        try:
            with np.errstate(all='ignore'): # Tell numpy not to issue warnings
                b.reset()
                q, s = opt.run()            # Search space
                times.append(b.time())
                p = trans.detransform(q, args.model)    # Model space
                params.append(p)
                scores.append(-s)
        except ValueError:
            import traceback
            traceback.print_exc()

    # Order from best to worst
    order = np.argsort(scores)
    scores = np.asarray(scores)[order]
    params = np.asarray(params)[order]
    times = np.asarray(times)[order]

    # Show results
    print('Best scores:')
    for score in scores[:10]:
        print(-score)
    print('Mean & std of score:')
    print(-np.mean(scores))
    print(np.std(scores))
    print('Worst score:')
    print(scores[-1])

    # Extract best
    obtained_score = scores[0]
    obtained_parameters = params[0]

    # Store results
    print('Storing best result...')
    with open(filename + '.txt', 'w') as f:
        for x in obtained_parameters:
            f.write(pints.strfloat(x) + '\n')

    print('Storing all errors')
    with open(filename + '-errors.txt', 'w') as f:
        for score in scores:
            f.write(pints.strfloat(-score) + '\n')

    print('Storing all parameters')
    for i, param in enumerate(params):
        with open(filename + '-parameters-' + str(1 + i) + '.txt', 'w') as f:
            for x in param:
                f.write(pints.strfloat(x) + '\n')

    print('Storing all simulation times')
    with open(filename + '-times.txt', 'w') as f:
        for time in times:
            f.write(pints.strfloat(time) + '\n')

