#!/usr/bin/env python3
#
# Python module that knows where all the data, models, and protocols are, and
# can load them.
#
from __future__ import division, print_function
import inspect
import myokit
import numpy as np
import os

# Get root of this project
try:
    frame = inspect.currentframe()
    ROOT = os.path.dirname(inspect.getfile(frame))
finally:
    del(frame) # Must be manually deleted
ROOT = os.path.join(ROOT, '..')


# Data directory
DATA = os.path.join(ROOT, 'data')

# Model directory
MODEL = os.path.join(ROOT, 'model-and-protocols')

# Protocol directory
PROTO = os.path.join(ROOT, 'model-and-protocols')


def load(cell, protocol, cached=None):
    """
    Returns data for the given cell and protocol, with capacitance filtering
    applied.

    Arguments:

    ``cell``
        The cell to use (integer).
    ``protocol``
        The protocol to use (integer)
    ``cached``
        Optional cached data. If given, this will be returned directly.

    Returns a myokit DataLog.
    """
    if cached is not None:
        return cached

    # Get path to data file
    SFU = os.path.join(DATA, 'SFU-data')
    data_files = {
        1: os.path.join(SFU, 'Staircase/staircase1-WT-cell-' + str(cell)),
        2: os.path.join(SFU, 'SW/sine-wave-WT-cell-' + str(cell)),
        3: os.path.join(SFU, 'AP/complex-AP-WT-cell-' + str(cell))
    }
    data_file = data_files[protocol]

    # Load protocol for capacitance filtering.
    protocol = load_myokit_protocol(protocol)

    # Load data from zip or csv
    if os.path.exists(data_file + '.zip'):
        print('Loading ' + data_file + '.zip')
        log = myokit.DataLog.load(data_file + '.zip').npview()
    else:
        print('Loading ' + data_file + '.csv')
        log = myokit.DataLog.load_csv(data_file + '.csv').npview()
        log.save(data_file + '.zip')

    # Apply capacitance filtering
    dt = 0.1
    signals = [log.time(), log['current']]
    voltage = 'voltage' in log
    if voltage:
        signals.append(log['voltage'])
    signals = capacitance(protocol, dt, *signals)

    log = myokit.DataLog()
    log.set_time_key('time')
    log['time'] = signals[0]
    log['current'] = signals[1]
    if voltage:
        log['voltage'] = signals[2]

    # Return
    return log


def load_model(which_model):
    """
    Loads the HH version of the Beattie model.
    """
    if which_model == 'mazhari':
        return myokit.load_model(os.path.join(MODEL, 'mazhari-ikr-markov.mmt'))
    elif which_model == 'mazhari-reduced':
        return myokit.load_model(os.path.join(MODEL, 'mazhari-reduced-MBAM-ikr-markov.mmt'))
    elif which_model == 'wang':
        return myokit.load_model(os.path.join(MODEL, 'wang-ikr-markov.mmt'))
    elif which_model == 'wang-r1':
        return myokit.load_model(os.path.join(MODEL, 'wang-r1-ikr-markov.mmt'))
    elif which_model == 'wang-r2':
        return myokit.load_model(os.path.join(MODEL, 'wang-r2-ikr-markov.mmt'))
    elif which_model == 'wang-r3':
        return myokit.load_model(os.path.join(MODEL, 'wang-r3-ikr-markov.mmt'))
    elif which_model == 'wang-r4':
        return myokit.load_model(os.path.join(MODEL, 'wang-r4-ikr-markov.mmt'))
    elif which_model == 'wang-r5':
        return myokit.load_model(os.path.join(MODEL, 'wang-r5-ikr-markov.mmt'))
    elif which_model == 'wang-r6':
        return myokit.load_model(os.path.join(MODEL, 'wang-r6-ikr-markov.mmt'))
    elif which_model == 'wang-r7':
        return myokit.load_model(os.path.join(MODEL, 'wang-r7-ikr-markov.mmt'))
    else:
        pass


def load_myokit_protocol(protocol):
    """
    Loads the Myokit protocol with the given index (1-7). For Pr6 and Pr7, the
    protocol only has the steps for capacitance filtering.
    """
    protocol_files = {
        1: os.path.join(PROTO, 'staircase1.mmt'),
        2: os.path.join(PROTO, 'sine-wave-ramp-steps.mmt'),
        3: os.path.join(PROTO, 'pr6-ap-steps.mmt')
    }

    protocol = protocol_files[protocol]

    # Load Myokit protocol
    return myokit.load_protocol(protocol)

def load_ap_protocol():
    """
    Returns a tuple ``(times, values)`` representing Pr6.
    """
    data_file = os.path.join(DATA, 'SFU-data/Protocols', 'AP-protocol')

    # Load data from zip or csv
    if os.path.exists(data_file + '.zip'):
        print('Loading ' + data_file + '.zip')
        log = myokit.DataLog.load(data_file + '.zip').npview()
    else:
        print('Loading ' + data_file + '.csv')
        log = myokit.DataLog.load_csv(data_file + '.csv').npview()
        log.save(data_file + '.zip')

    return log

def load_protocol_values(protocol, no_filter=True, which_model='wang'):
    """
    Returns a (capacitance filtered) tuple ``(times, voltages)`` for the
    selected ``protocol``.
    """
    p = load_myokit_protocol(protocol)

    if protocol == 1:
        m = load_model(which_model)
        m.get('membrane.V').set_rhs(
           'piecewise(engine.time <= 1236.2,'
           + ' -80 - misc.LJP,'
           + 'engine.time >= 14410.1 and engine.time < 14510.0,'
           + ' -70 - misc.LJP'
           + ' - 0.4 * (engine.time - 14410.1)'
           + ', engine.pace - misc.LJP)')
        p = load_myokit_protocol(protocol)
    elif protocol == 2:
        m = load_model(which_model)
        m.get('membrane.V').set_rhs(
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
        p = load_myokit_protocol(protocol)
    elif protocol == 3:
        m = load_model(which_model)
        log = load_ap_protocol().npview()
        t, v = log['time'], log['voltage']
        p = t, v
    else:
        t = np.arange(0, p.characteristic_time(), 0.1)
        v = np.array(p.value_at_times(t))
        p = t, v

    if no_filter:
        return p
    else:
        return capacitance(p, 0.2, t, v)


def capacitance(protocol, dt, *signals):
    """
    Creates and applies a capacitance filter, based on a Myokit protocol.

    Arguments:

    ``protocol``
        A Myokit protocol.
    ``dt``
        The sampling interval of the given signals.
    ``signals``
        One or more signal files to filter.

    Returns a filtered version of the given signals.
    """
    cap_duration = 1    # Kylie and Michael used 5 ms, this is far too long for 37C data, so 1 ms is used instead
    fcap = np.ones(len(signals[0]), dtype=int) # fcap = length of signals[0], which is time
    steps = [step for step in protocol]
    for step in steps[1:]:
        i1 = int(step.start() / dt)
        i2 = i1 + int(cap_duration / dt)
        fcap[i1:i2] = 0
    fcap = fcap > 0

    debug = False
    if debug:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(signals[0], signals[1])
        plt.plot(signals[0], signals[1])
        for step in steps[1:]:
            plt.axvline(step.start(), color='green', alpha=0.25)
        plt.show()

    # Apply filter
    return [x[fcap] for x in signals]


def model_path(model_file):
    """
    Returns the path to the given Myokit model file.
    """
    return os.path.join(MODEL, model_file)


def protocol_path(protocol_file):
    """
    Returns the path to the given Myokit protocol file.
    """
    return os.path.join(PROTO, protocol_file)

