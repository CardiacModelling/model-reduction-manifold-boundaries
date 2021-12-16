#!/usr/bin/env python3
#
# Pints ForwardModel that runs simulations with Kylie's model.
# Sine waves optional
#
from __future__ import division, print_function
import myokit
import myokit.lib.markov as markov
import numpy as np
import pints

from data import model_path


class Model(pints.ForwardModel):
    """
    Pints ForwardModel that runs simulations with Kylie's model.
    Sine waves or data protocol optional.

    Arguments:

        ``protocol``
            A myokit.Protocol or a tuple (times, voltage)
        ``EK``
            The reversal potential
        ``sine_wave``
            Set to True if sine-wave protocol is being used.
        ``start_steady``
            Start at steady state for -80mV. Note that this should be disabled
            to get Kylie's original results.
        ``analytical``
            Use an analytical simulation.

    """

    def __init__(
        self, protocol, EK=-88.0, which_model='wang', start_steady=False, sine_wave=False, \
        staircase1=False, analytical=False, transformation=None):

        # Load model
        self.which_model = which_model
        model = myokit.load_model(model_path(which_model + '-ikr-markov.mmt'))
        if self.which_model == 'mazhari':
            self.states = ['ikr.c2', 'ikr.c3', 'ikr.o', 'ikr.i']
        elif self.which_model == 'mazhari-reduced':
            self.states = ['ikr.c1', 'ikr.c3', 'ikr.o', 'ikr.i']
        elif self.which_model in {'wang', 'wang-r1'}:
            self.states = ['ikr.c2', 'ikr.c3', 'ikr.o', 'ikr.i']
        elif self.which_model in {'wang-r2', 'wang-r3', 'wang-r4'}:
            self.states = ['ikr.c3', 'ikr.o', 'ikr.i']
        elif self.which_model in {'wang-r5', 'wang-r6', 'wang-r7'}:
            self.states = ['ikr.o', 'ikr.i']
        elif self.which_model in {'wang-r8'}:
            self.states = ['ikr.o']
        else:
            pass
        n_params = int(model.get('misc.n_params').value())
        model = markov.convert_markov_models_to_full_ode_form(model)

        parameters = np.zeros(n_params)
        for i in range(n_params):
            parameters[i] = model.get('ikr.p'+str(i+1)).value()

        current = 'ikr.IKr'
        self.current = current

        self.parameters = parameters
        self.n_params = n_params
        LJP = model.get('misc.LJP').value()

        self.ss_V = -80 - LJP

        # Set reversal potential
        print('EK:', EK)
        model.get('nernst.EK').set_rhs(EK)

        # Start at steady-state for -80mV for Claydon 37C data
        if start_steady:
            print('Updating model to steady-state for ' + str(self.ss_V) + 'mV.')
            model.get('membrane.V').set_label('membrane_potential')
            mm = markov.LinearModel.from_component(model.get('ikr'))
            # Update states
            x = mm.steady_state(self.ss_V, self.parameters)
            for i in range(len(self.states)):
                model.get(self.states[i]).set_state_value(x[i])

        if sine_wave:
            model.get('membrane.V').set_rhs(
                'piecewise(engine.time >= 300.0 and engine.time < 900.0,'
                + ' -140'
                + ' + 0.1 * (engine.time - 300.0),'
                + 'engine.time >= 3599.9 and engine.time < 7100.1,'
                + ' - 30'
                + ' + 54 * sin(0.007 * (engine.time - 3100.1))'
                + ' + 26 * sin(0.037 * (engine.time - 3100.1))'
                + ' + 10 * sin(0.190 * (engine.time - 3100.1)),'
                + 'engine.time >= 7100.1 and engine.time < 7200.1,'
                + ' -70'
                + ' - 0.3 * (engine.time - 7100.0)'
                + ', engine.pace)')

        if staircase1:
            model.get('membrane.V').set_rhs(
               'piecewise(engine.time <= 1236.2,'
               + ' -80 - misc.LJP,'
               + 'engine.time >= 14410.1 and engine.time < 14510.0,'
               + ' -70 - misc.LJP'
               + ' - 0.4 * (engine.time - 14410.1)'
               + ', engine.pace - misc.LJP)')

        # Create simulation
        self._analytical = analytical
        if not self._analytical:
            self.simulation = myokit.Simulation(model)
            # Add protocol
            if isinstance(protocol, myokit.Protocol):
                self.simulation.set_protocol(protocol)
            else:
                # Apply data-clamp
                times, voltage = protocol
                self.simulation.set_fixed_form_protocol(times, voltage)

                # Set max step size
                self.simulation.set_max_step_size(0.1)

            # Set solver tolerances
            self.simulation.set_tolerance(1e-8, 1e-8)
        else:
            if sine_wave or staircase1 or staircase2 or RPR:
                raise ValueError(
                    'Analytical simulation cannot be used with sine wave or staircase protocols.')
            elif not isinstance(protocol, myokit.Protocol):
                raise ValueError(
                    'Analytical simulation cannot be used with data clamp.')
            if not start_steady:
                model.get('membrane.V').set_label('membrane_potential')
                mm = markov.LinearModel.from_component(model.get('ikr'))
            self.simulation = markov.AnalyticalSimulation(mm, protocol)

        self.transformation = transformation
        # Set a maximum duration for each simulation.
        self._timeout = myokit.Timeout(60)

    def n_parameters(self):
        return self.n_params

    def set_tolerances(self, tol):
        self.simulation.set_tolerance(tol, tol)

    def simulate(self, parameters, times):

        if self.transformation is not None:
            parameters = self.transformation.detransform(parameters, self.which_model)

        # Update model parameters
        for i in range(int(self.n_params)):
            self.simulation.set_constant('ikr.p'+str(i+1), parameters[i])

        # Run
        self.simulation.reset()
        try:
            if self._analytical:
                d = self.simulation.run(
                    times[-1] + times[1],
                    log_times=times,
                    ).npview()
            else:
                d = self.simulation.run(
                    times[-1] + times[1],
                    log_times=times,
                    log=['ikr.IKr', 'membrane.V'],
                    progress=self._timeout,
                    ).npview()
        except myokit.SimulationError:
            return times * float('inf')
        except myokit.SimulationCancelledError:
            return times * float('inf')

        # Return
        return d['ikr.IKr']

