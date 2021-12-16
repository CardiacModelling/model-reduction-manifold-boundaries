#!/usr/bin/env python3
#
# Parameter transformations
#
from __future__ import division, print_function
import numpy as np


class Transformation(object):
    """
    Transforms from model to search space (and back).
    """

    def transform(self, parameters, which_model, noise=False):
        """
        Transform from model into search space.
        """
        if which_model == 'mazhari':
            x = np.array([
                np.log(parameters[0]),
                parameters[1],
                np.log(parameters[2]),
                parameters[3],
                np.log(parameters[4]),
                parameters[5],
                np.log(parameters[6]),
                parameters[7],
                np.log(parameters[8]),
                parameters[9],
                np.log(parameters[10]),
                parameters[11],
                np.log(parameters[12]),
                parameters[13],
                np.log(parameters[14]),
                np.log(parameters[15]),
                # Conductance
                parameters[16]
            ])
        elif which_model == 'mazhari-reduced':
            x = np.array([
                np.log(parameters[0]),
                parameters[1],
                np.log(parameters[2]),
                parameters[3],
                np.log(parameters[4]),
                np.log(parameters[5]),
                parameters[6],
                np.log(parameters[7]),
                np.log(parameters[8]),
                parameters[9],
                # Conductance
                parameters[10]
            ])
        elif which_model == 'wang':
            x = np.array([
                np.log(parameters[0]),
                parameters[1],
                np.log(parameters[2]),
                parameters[3],
                np.log(parameters[4]),
                np.log(parameters[5]),
                np.log(parameters[6]),
                parameters[7],
                np.log(parameters[8]),
                parameters[9],
                np.log(parameters[10]),
                parameters[11],
                np.log(parameters[12]),
                parameters[13],
                # Conductance
                parameters[14]
            ])
        elif which_model == 'wang-r1':
            x = np.array([
                np.log(parameters[0]),
                parameters[1],
                np.log(parameters[2]),
                parameters[3],
                np.log(parameters[4]),
                np.log(parameters[5]),
                np.log(parameters[6]),
                parameters[7],
                np.log(parameters[8]),
                parameters[9],
                np.log(parameters[10]),
                parameters[11],
                np.log(parameters[12]),
                # Conductance
                parameters[13]
            ])
        elif which_model == 'wang-r2':
            x = np.array([
                np.log(parameters[0]),
                parameters[1],
                np.log(parameters[2]),
                parameters[3],
                np.log(parameters[4]),
                np.log(parameters[5]),
                np.log(parameters[6]),
                parameters[7],
                np.log(parameters[8]),
                parameters[9],
                np.log(parameters[10]),
                parameters[11],
                # Conductance
                parameters[12]
            ])
        elif which_model == 'wang-r3':
            x = np.array([
                np.log(parameters[0]),
                parameters[1],
                np.log(parameters[2]),
                parameters[3],
                np.log(parameters[4]),
                np.log(parameters[5]),
                np.log(parameters[6]),
                parameters[7],
                np.log(parameters[8]),
                parameters[9],
                parameters[10],
                # Conductance
                parameters[11]
            ])
        elif which_model == 'wang-r4':
            x = np.array([
                np.log(parameters[0]),
                np.log(parameters[1]),
                parameters[2],
                np.log(parameters[3]),
                np.log(parameters[4]),
                np.log(parameters[5]),
                parameters[6],
                np.log(parameters[7]),
                parameters[8],
                parameters[9],
                # Conductance
                parameters[10]
            ])
        elif which_model == 'wang-r5':
            x = np.array([
                np.log(parameters[0]),
                parameters[1],
                np.log(parameters[2]),
                np.log(parameters[3]),
                np.log(parameters[4]),
                parameters[5],
                np.log(parameters[6]),
                parameters[7],
                parameters[8],
                # Conductance
                parameters[9]
            ])
        elif which_model == 'wang-r6':
            x = np.array([
                parameters[0],
                np.log(parameters[1]),
                np.log(parameters[2]),
                np.log(parameters[3]),
                parameters[4],
                np.log(parameters[5]),
                parameters[6],
                parameters[7],
                # Conductance
                parameters[8]
            ])
        elif which_model == 'wang-r7':
            x = np.array([
                parameters[0],
                np.log(parameters[1]),
                np.log(parameters[2]),
                np.log(parameters[3]),
                parameters[4],
                np.log(parameters[5]),
                parameters[6],
                # Conductance
                parameters[7]
            ])
        elif which_model == 'wang-r8':
            x = np.array([
                parameters[0],
                np.log(parameters[1]),
                np.log(parameters[2]),
                np.log(parameters[3]),
                parameters[4],
                parameters[5],
                # Conductance
                parameters[6]
            ])
        else:
            pass

        return x

    def detransform(self, transformed_parameters, which_model, noise=False):
        """
        Transform back from search space to model space.
        """
        if which_model == 'mazhari':
            x = np.array([
                np.exp(transformed_parameters[0]),
                transformed_parameters[1],
                np.exp(transformed_parameters[2]),
                transformed_parameters[3],
                np.exp(transformed_parameters[4]),
                transformed_parameters[5],
                np.exp(transformed_parameters[6]),
                transformed_parameters[7],
                np.exp(transformed_parameters[8]),
                transformed_parameters[9],
                np.exp(transformed_parameters[10]),
                transformed_parameters[11],
                np.exp(transformed_parameters[12]),
                transformed_parameters[13],
                np.exp(transformed_parameters[14]),
                np.exp(transformed_parameters[15]),
                # Conductance
                transformed_parameters[16]
            ])
        elif which_model == 'mazhari-reduced':
            x = np.array([
                np.exp(transformed_parameters[0]),
                transformed_parameters[1],
                np.exp(transformed_parameters[2]),
                transformed_parameters[3],
                np.exp(transformed_parameters[4]),
                np.exp(transformed_parameters[5]),
                transformed_parameters[6],
                np.exp(transformed_parameters[7]),
                np.exp(transformed_parameters[8]),
                transformed_parameters[9],
                # Conductance
                transformed_parameters[10]
            ])
        elif which_model == 'wang':
            x = np.array([
                np.exp(transformed_parameters[0]),
                transformed_parameters[1],
                np.exp(transformed_parameters[2]),
                transformed_parameters[3],
                np.exp(transformed_parameters[4]),
                np.exp(transformed_parameters[5]),
                np.exp(transformed_parameters[6]),
                transformed_parameters[7],
                np.exp(transformed_parameters[8]),
                transformed_parameters[9],
                np.exp(transformed_parameters[10]),
                transformed_parameters[11],
                np.exp(transformed_parameters[12]),
                transformed_parameters[13],
                # Conductance
                transformed_parameters[14]
            ])
        elif which_model == 'wang-r1':
            x = np.array([
                np.exp(transformed_parameters[0]),
                transformed_parameters[1],
                np.exp(transformed_parameters[2]),
                transformed_parameters[3],
                np.exp(transformed_parameters[4]),
                np.exp(transformed_parameters[5]),
                np.exp(transformed_parameters[6]),
                transformed_parameters[7],
                np.exp(transformed_parameters[8]),
                transformed_parameters[9],
                np.exp(transformed_parameters[10]),
                transformed_parameters[11],
                np.exp(transformed_parameters[12]),
                # Conductance
                transformed_parameters[13]
            ])
        elif which_model == 'wang-r2':
            x = np.array([
                np.exp(transformed_parameters[0]),
                transformed_parameters[1],
                np.exp(transformed_parameters[2]),
                transformed_parameters[3],
                np.exp(transformed_parameters[4]),
                np.exp(transformed_parameters[5]),
                np.exp(transformed_parameters[6]),
                transformed_parameters[7],
                np.exp(transformed_parameters[8]),
                transformed_parameters[9],
                np.exp(transformed_parameters[10]),
                transformed_parameters[11],
                # Conductance
                transformed_parameters[12]
            ])
        elif which_model == 'wang-r3':
            x = np.array([
                np.exp(transformed_parameters[0]),
                transformed_parameters[1],
                np.exp(transformed_parameters[2]),
                transformed_parameters[3],
                np.exp(transformed_parameters[4]),
                np.exp(transformed_parameters[5]),
                np.exp(transformed_parameters[6]),
                transformed_parameters[7],
                np.exp(transformed_parameters[8]),
                transformed_parameters[9],
                transformed_parameters[10],
                # Conductance
                transformed_parameters[11]
            ])
        elif which_model == 'wang-r4':
            x = np.array([
                np.exp(transformed_parameters[0]),
                np.exp(transformed_parameters[1]),
                transformed_parameters[2],
                np.exp(transformed_parameters[3]),
                np.exp(transformed_parameters[4]),
                np.exp(transformed_parameters[5]),
                transformed_parameters[6],
                np.exp(transformed_parameters[7]),
                transformed_parameters[8],
                transformed_parameters[9],
                # Conductance
                transformed_parameters[10]
            ])
        elif which_model == 'wang-r5':
            x = np.array([
                np.exp(transformed_parameters[0]),
                transformed_parameters[1],
                np.exp(transformed_parameters[2]),
                np.exp(transformed_parameters[3]),
                np.exp(transformed_parameters[4]),
                transformed_parameters[5],
                np.exp(transformed_parameters[6]),
                transformed_parameters[7],
                transformed_parameters[8],
                # Conductance
                transformed_parameters[9]
            ])
        elif which_model == 'wang-r6':
            x = np.array([
                transformed_parameters[0],
                np.exp(transformed_parameters[1]),
                np.exp(transformed_parameters[2]),
                np.exp(transformed_parameters[3]),
                transformed_parameters[4],
                np.exp(transformed_parameters[5]),
                transformed_parameters[6],
                transformed_parameters[7],
                # Conductance
                transformed_parameters[8]
            ])
        elif which_model == 'wang-r7':
            x = np.array([
                transformed_parameters[0],
                np.exp(transformed_parameters[1]),
                np.exp(transformed_parameters[2]),
                np.exp(transformed_parameters[3]),
                transformed_parameters[4],
                np.exp(transformed_parameters[5]),
                transformed_parameters[6],
                # Conductance
                transformed_parameters[7]
            ])
        elif which_model == 'wang-r8':
            x = np.array([
                transformed_parameters[0],
                np.exp(transformed_parameters[1]),
                np.exp(transformed_parameters[2]),
                np.exp(transformed_parameters[3]),
                transformed_parameters[4],
                transformed_parameters[5],
                # Conductance
                transformed_parameters[6]
            ])
        else:
            pass

        return x

