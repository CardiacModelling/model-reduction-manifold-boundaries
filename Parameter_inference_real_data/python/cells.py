#!/usr/bin/env python3
#
# Cell based data and functions
#
from __future__ import division
import numpy as np


def lower_conductance(cell):
    """
    Returns a lower limit for the conductance of the cell with the
    given integer index ``cell``.
    """
    #
    # Guesses for lower conductance
    #
    lower_conductances = {
        1: 0.1524,
        2: 0.1524,
        3: 0.1524,
        4: 0.1524,
        5: 0.1524
    }
    return lower_conductances[cell]


def reversal_potential(temperature):
    """
    Calculates the reversal potential for Potassium ions, using Nernst's
    equation for a given ``temperature`` in degrees celsius and the internal
    and external [K]+ concentrations used in the experiments.
    """
    T = 273.15 + temperature
    F = 96485
    R = 8314
    K_i = 130
    k_o = 4
    return ((R*T)/F) * np.log(k_o/K_i)


def temperature(cell):
    """
    Returns the temperature (in degrees Celsius) for the given integer index
    ``cell``.
    """
    temperatures = {
        1: 37.0,
        2: 37.0,
        3: 37.0,
        4: 37.0,
        5: 37.0
    }
    return temperatures[cell]


def ek(cell):
    """
    Returns the WT reversal potential (in mV) for the given integer index
    ``cell``.
    """
    reversal_potentials = {
        1: -91.6,
        2: -92.8,
        3: -95.1,
        4: -92.3,
        5: -106.1
    }
    return reversal_potentials[cell]

