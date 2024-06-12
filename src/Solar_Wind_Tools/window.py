#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 09:56:54 2021

@author: simon
"""
import pandas as pd
import numpy as np
from scipy.stats import sem
try:
    from progressbar import progressbar
except ImportError:
    def progressbar(*args, **kwargs):
        return args[0]

#engine_kwargs= {'nopython':True, 'nogil':False, 'parallel': True}
def circular_variance(file_path, save_path, window=30, load_key='omni', key='omni_window'):
    """
    Calculate circular variance for magnetic field components in a rolling window and save results to an HDF file.

    Parameters:
    - file_path (str): Path to the input HDF file containing the data.
    - save_path (str): Path to the output HDF file to save the results.
    - window (int, optional): Size of the rolling window for calculating variance and mean. Default is 30 minutes.
    - load_key (str, optional): Key to load the data from the input HDF file. Default is 'omni'.
    - key (str, optional): Key to save the data in the output HDF file. Default is 'omni_window'.

    Returns:
    None

    Example:
    >>> circular_variance('input.h5', 'output.h5')
    """

    def sin(theta):
        return np.nanmean(np.sin(theta))**2

    def cos(theta):
        return np.nanmean(np.cos(theta))**2

    def Variance(X):
        X = X[np.isfinite(X)]
        if len(X) <= 1:
            return np.nan
        else:
            return np.nansum((X - np.nanmean(X))**2) / (len(X) - 1)
    def Count(X):
        return sum(np.isfinite(X))
    def SEM(X):
        n= np.sqrt(np.isfinite(X).sum())
        if not n:
            n=np.nan
        return np.nanstd(X)/n
    omni = pd.read_hdf(file_path, key=load_key)
    columns = ['IMF', 'BX_GSE', 'BY_GSM', 'BZ_GSM']

    for column in progressbar(columns, max_value=len(columns)):
        omni[column+'_Var'] = omni[column].rolling(window=f'{window}min', min_periods=0).apply(Variance, engine='numba', raw=True)
        omni[column+'_Mean'] = omni[column].rolling(window=f'{window}min', min_periods=0).apply(np.nanmean, engine='numba', raw=True)
        omni[column+'_SEM'] = omni[column].rolling(window=f'{window}min', min_periods=0).apply(SEM, engine='numba', raw=True)
    omni['points'] = omni[column].rolling(window=f'{window}min', min_periods=0).apply(Count, engine='numba', raw=True)

    omni['Clock_GSM'] = np.arctan2(omni.BY_GSM, omni.BZ_GSM)
    tmp = pd.DataFrame({'Theta': np.arctan2(omni['BY_GSM'], omni['BZ_GSM'])}).rolling(window=f'{window}min', min_periods=0)
    omni['Circular_Variance_GSM'] = 1 - np.sqrt(tmp.apply(sin, engine='numba', raw=True) + tmp.apply(cos, engine='numba', raw=True))
    
    omni.to_hdf(save_path, key=key)

def clock_angle_statistics(file_path, save_path, window=30, load_key='omni', key='omni_window'):
    from scipy.stats import directional_stats, bootstrap, circstd
    from functools import partial
    omni = pd.read_hdf(file_path, key=load_key)
    def Count(X):
        return sum(np.isfinite(X))

    def mean_clock_angle(By, Bz, axis):
        return np.arctan2(np.nanmean(By, axis=axis), np.nanmean(Bz, axis= axis))
    def sin(theta):
        return np.nanmean(np.sin(theta))**2

    def cos(theta):
        return np.nanmean(np.cos(theta))**2
    def clock_SEM(By, data, col):
        Bz= data.loc[By.index, [col]].values.reshape(-1)
        By= By.values
        ind= (np.isfinite(By))&(np.isfinite(Bz))
        if not np.sum(ind):
            return np.nan
        if np.sum(ind)==1:
            return 0
        By, Bz=  By[ind], Bz[ind]
        theta= bootstrap((By, Bz), mean_clock_angle, paired=True).bootstrap_distribution
        tmp = pd.DataFrame({'Theta': theta})
        return (np.sqrt(1 - np.sqrt(tmp.apply(sin, engine='numba', raw=True) +\
                             tmp.apply(cos, engine='numba', raw=True)))/np.sqrt(np.isfinite(tmp.Theta).sum()))\
                             .values[0]

    if not 'points' in omni:
        omni['points'] = omni['BX_GSE'].rolling(window=f'{window}min', min_periods=0).apply(Count, engine='numba', raw=True)

    omni['Clock_GSM_Mean'] = np.arctan2(omni.BY_GSM_Mean, omni.BZ_GSM_Mean)
    omni['Clock_GSM_SEM']= omni['BY_GSM'].rolling(window=f'{window}min', min_periods=0).\
        apply(clock_SEM, args=(omni, 'BZ_GSM'))

    omni['Clock_GSM_STD']= omni['Clock_GSM'].rolling(window=f'{window}min', min_periods=0).apply(circstd)
    omni.to_hdf(save_path, key=key)

def coupling(file_path, save_path, window=30, load_key='omni', key='omni_window'):
    """
    Calculate Newell coupling function and its mean in a rolling window and save results to an HDF file.

    Parameters:
    - file_path (str): Path to the input HDF file containing the data.
    - save_path (str): Path to the output HDF file to save the results.
    - window (int, optional): Size of the rolling window. Default is 30 minutes.
    - load_key (str, optional): Key to load the data from the input HDF file. Default is 'omni'.
    - key (str, optional): Key to save the data in the output HDF file. Default is 'omni_window'.

    Returns:
    None

    Example:
    >>> coupling('input.h5', 'output.h5')
    """
    data = pd.read_hdf(file_path, key=load_key)
    
    from .Coupling_Functions import newell_coupling_function

    data['Newell_Epsilon'] = newell_coupling_function(data.Vx, data.BY_GSM, data.BZ_GSM)
    data['Newell_Epsilon_Mean'] = data.Newell_Epsilon.rolling(window=f'{window}min', min_periods=0).apply(np.nanmean, engine='numba', raw=True)
    
    data.to_hdf(save_path, key=key)


def dipole(file_path, save_path, load_key='omni', key='omni_window'):
    """
    Calculate dipole tilt angle and save results to an HDF file.

    Parameters:
    - file_path (str): Path to the input HDF file containing the data.
    - save_path (str): Path to the output HDF file to save the results.
    - window (int, optional): Size of the rolling window. Default is 30 minutes.
    - load_key (str, optional): Key to load the data from the input HDF file. Default is 'omni'.
    - key (str, optional): Key to save the data in the output HDF file. Default is 'omni_window'.

    Returns:
    None

    Example:
    >>> dipole('input.h5', 'output.h5')
    """
    from dipole import Dipole

    data = pd.read_hdf(file_path, key=load_key)
    years = data.index.year
    data['Dipole_Tilt'] = np.concatenate([Dipole(year).tilt(data.index.values[years==year]) for year in np.unique(years)])
    
    data.to_hdf(save_path, key=key)


def time_shift(file_path, save_path, shift, load_key='omni', key='omni_window'):
    """
    Shift mean and variance columns by a specified number of steps and save results to an HDF file.

    Parameters:
    - file_path (str): Path to the input HDF file containing the data.
    - save_path (str): Path to the output HDF file to save the results.
    - start (int, optional): Number of steps to start the shift. Default is 20 minutes prior.
    - end (int, optional): Number of steps to end the shift. Default is 10 minutes prior.
    - load_key (str, optional): Key to load the data from the input HDF file. Default is 'omni'.
    - key (str, optional): Key to save the data in the output HDF file. Default is 'omni_window'.

    Returns:
    None

    Example:
    >>> time_shift('input.h5', 'output.h5')
    """
    data = pd.read_hdf(file_path, key=load_key)
    
    for col in data.columns:
        if col.endswith('_Mean') or col.endswith('_Var') or col.endswith('_STD') or col.endswith('_SEM') or col=='points' or col=='Circular_Variance_GSM':
            data[col] = data[col].shift(shift)
    
    data.to_hdf(save_path, key=key)
