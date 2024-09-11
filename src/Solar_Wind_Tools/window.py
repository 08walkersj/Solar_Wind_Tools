import pandas as pd
import numpy as np
from scipy.stats import sem
try:
    from progressbar import progressbar
except ImportError:
    def progressbar(*args, **kwargs):
        return args[0]
def statistics(file_path, save_path, window=30, load_key='omni', key='omni_window'):
    """
    Calculate the mean, median, variance and standard error of the mean for magnetic field components and clock angle in a rolling window and save results to an HDF file.

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
    # Import necessary libraries
    import pandas as pd
    import numpy as np

    def sin(theta):
        """
        Computes the square of the mean sine of the angles.
        """
        return np.nanmean(np.sin(theta))**2

    def cos(theta):
        """
        Computes the square of the mean cosine of the angles.
        """
        return np.nanmean(np.cos(theta))**2

    def Variance(X):
        """
        Calculates the variance of finite values in X.
        """
        X = X[np.isfinite(X)]
        if len(X) <= 1:
            return np.nan
        else:
            return np.nansum((X - np.nanmean(X))**2) / (len(X) - 1)
    
    def Count(X):
        """
        Counts the number of finite values in X.
        """
        return sum(np.isfinite(X))
    
    def SEM(X):
        """
        Calculates the standard error of the mean (SEM) for the values in X.
        """
        n = np.sqrt(np.isfinite(X).sum())
        if not n:
            n = np.nan
        return np.nanstd(X) / n

    # Load the data from the HDF file
    omni = pd.read_hdf(file_path, key=load_key)
    columns = ['IMF', 'BX_GSE', 'BY_GSM', 'BZ_GSM']

    # Calculate variance, mean, and SEM for each column in a rolling window
    for column in progressbar(columns, max_value=len(columns)):
        omni[column + '_Var'] = omni[column].rolling(window=f'{window}min', min_periods=0).apply(Variance, engine='numba', raw=True)
        omni[column + '_Mean'] = omni[column].rolling(window=f'{window}min', min_periods=0).apply(np.nanmean, engine='numba', raw=True)
        omni[column + '_Median'] = omni[column].rolling(window=f'{window}min', min_periods=0).apply(np.nanmedian, engine='numba', raw=True)
        omni[column + '_SEM'] = omni[column].rolling(window=f'{window}min', min_periods=0).apply(SEM, engine='numba', raw=True)
    
    # Add a column for the number of valid points in each rolling window
    omni['points'] = omni[column].rolling(window=f'{window}min', min_periods=0).apply(Count, engine='numba', raw=True)

    # Calculate the clock angle for GSM coordinates
    omni['Clock_GSM'] = np.arctan2(omni.BY_GSM, omni.BZ_GSM)

    # Calculate the circular variance for the GSM clock angle
    tmp = pd.DataFrame({'Theta': np.arctan2(omni['BY_GSM'], omni['BZ_GSM'])}).rolling(window=f'{window}min', min_periods=0)
    omni['Circular_Variance_GSM'] = 1 - np.sqrt(tmp.apply(sin, engine='numba', raw=True) + tmp.apply(cos, engine='numba', raw=True))

    # Calculate the standard error of the mean for the GSM clock angle
    omni['Clock_GSM_SEM']= np.rad2deg(np.sqrt(omni['Circular_Variance_GSM'])/np.sqrt(omni['points']))

    # Calculate the mean of the GSM clock angle
    omni['Clock_GSM_Mean'] = np.arctan2(omni.BY_GSM_Mean, omni.BZ_GSM_Mean)

    # Calculate the median of the GSM clock angle
    omni['Clock_GSM_Median'] = np.arctan2(omni.BY_GSM_Median, omni.BZ_GSM_Median)

    # Save the results to the output HDF file
    omni.to_hdf(save_path, key=key, format='t', data_columns=True)


def coupling(file_path, save_path, window=False, load_key='omni', key='omni_window'):
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
    if window:
        data['Newell_Epsilon_Mean'] = data.Newell_Epsilon.rolling(window=f'{window}min', min_periods=0).apply(np.nanmean, engine='numba', raw=True)
    
    data.to_hdf(save_path, key=key, format='t', data_columns=True)


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
    
    data.to_hdf(save_path, key=key, format='t', data_columns=True)


def time_shift(file_path, save_path, shift, load_key='omni', key='omni_window'):
    """
    Shift mean, variance, standard deviation and standard error of the mean columns by a specified number of steps and save results to an HDF file.

    Parameters:
    - file_path (str): Path to the input HDF file containing the data.
    - save_path (str): Path to the output HDF file to save the results.
    - shift (int): time shift in minutes. Positive value will make a time step be associated with a value from shift minutes ago
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
    
    data.to_hdf(save_path, key=key, format='t', data_columns=True)
