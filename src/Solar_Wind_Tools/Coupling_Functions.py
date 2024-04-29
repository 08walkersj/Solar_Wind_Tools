#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 10:41:07 2021

@author: simon
"""
import numpy as np
def newell_coupling_function(Vx, By, Bz):
    """
    Calculate Newell coupling function for solar wind-magnetosphere coupling.

    The Newell coupling function quantifies the coupling between the solar wind and the
    Earth's magnetosphere. It is based on the solar wind velocity (Vx), and the magnetic
    field components in the y (By) and z (Bz) directions.

    Parameters:
    - Vx (float or numpy.ndarray): Solar wind velocity component in the x-direction (km/s).
    - By (float or numpy.ndarray): Magnetic field component in the y-direction (nT).
    - Bz (float or numpy.ndarray): Magnetic field component in the z-direction (nT).

    Returns:
    - float or numpy.ndarray: Newell coupling function values.

    Formula:
    Newell coupling function = abs((10**-3)*(abs(Vx)**(4/3))*(np.sqrt(By**2 + Bz**2)**(2/3))*np.sin(theta/2)**(8/3))

    where theta is the angle between By and Bz.

    Example:
    >>> Vx = 400  # km/s
    >>> By = 5    # nT
    >>> Bz = -10  # nT
    >>> result = newell_coupling_function(Vx, By, Bz)
    """
    theta = np.arctan2(By, Bz)
    return abs((10**-3)*(abs(Vx)**(4/3))*(np.sqrt(By**2 + Bz**2)**(2/3))*abs(np.sin(theta/2))**(8/3))