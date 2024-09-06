from .window import statistics, coupling, dipole, time_shift
from .download import download_omni_1min
import os
from datetime import datetime as dt
import time
import warnings
warnings.simplefilter('always', ImportWarning)

class ArgumentError(Exception):
     pass
def format_elapsed_time(seconds):
    days = seconds // 86400
    seconds %= 86400
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return f"{days}Days:{hours}Hrs:{minutes}Mins:{seconds}Secs"

def run_all(save_path, omni_path=False, window=30, shift=10, start_year=1981, end_year=False, key='omni_window', run=[]):
    """
    Run a series of functions on the OMNI dataset and save the results.

    Parameters:
    - save_path (str): Path to save the output HDF5 file.
    - omni_path (str, optional): Path to the OMNI dataset HDF5 file. If not provided, it uses save_path.
    - window (int, optional): Size of the rolling window for calculating statistics. Default is 30 minutes.
    - shift (int, optional): time shift in minutes. Positive value will make a time step be associated with a value from shift minutes ago. Default is 10 minutes.
    - start_year (int, optional): Start year for downloading the OMNI dataset.
    - end_year (int, optional): End year for downloading the OMNI dataset. If not provided, it uses the current year.
    - key (str, optional): Key for the windowed data from the HDF5 files.
    - run (list, optional): List of functions to run. Default is ['Download', 'Statistics', 'Coupling', 'Dipole', 'Time Shift'].

    Returns:
    None

    Example:
    >>> run_all('output.h5', shift=20, start_year=2000, end_year=2005, run=['Download', 'Statistics', 'Coupling'])
    """
    # Check if run list is empty and set default functions to run
    if not len(run):
        run = ['Download', 'Statistics', 'Coupling', 'Dipole', 'Time Shift']
    # Check if all options chosen are known
    elif not all(r in ['Download', 'Statistics', 'Coupling', 'Dipole', 'Time Shift'] for r in run):
        raise ArgumentError(f"a choice in {run} is not understood please choose from: f{['Download', 'Statistics', 'Coupling', 'Dipole', 'Time Shift']}")
    # Ensure save_path ends with .h5 or .hdf5
    if not (save_path.endswith('.h5') or save_path.endswith('.hdf5')):
        save_path += '.h5'

    # If omni_path is not provided, use save_path
    if not omni_path:
        omni_path = save_path

    # Ensure omni_path ends with .h5 or .hdf5
    if not (omni_path.endswith('.h5') or omni_path.endswith('.hdf5')):
        omni_path += '.h5'

    # Download OMNI dataset if specified in run list and omni_path file does not exist
    if 'Download' in run:
        if not os.path.isfile(omni_path):
            if not end_year:
                end_year = dt.now().year
            download_omni_1min(start_year, end_year, monthFirstYear=1, monthLastYear=12, path=omni_path)

    i = 0
    # Run Circular Variance if specified in run list
    if 'Statistics' in run:
        if i == 0:
            load_key = 'omni'
        else:
            load_key = key
        print('Statistics')
        start_time = time.time()
        statistics(omni_path, save_path, window=window, load_key=load_key, key=key)
        i += 1
        end_time = time.time()
        elapsed_time = end_time - start_time
        formatted_time = format_elapsed_time(int(elapsed_time))
        print('\n', f"Time elapsed: {formatted_time}", '\n')
    # Run Coupling if specified in run list
    if 'Coupling' in run:
        if i == 0:
            load_key = 'omni'
        else:
            load_key = key
        print('Coupling')
        start_time = time.time()
        coupling(omni_path, save_path, window=window, load_key=key, key=key)
        i += 1
        end_time = time.time()
        elapsed_time = end_time - start_time
        formatted_time = format_elapsed_time(int(elapsed_time))
        print('\n', f"Time elapsed: {formatted_time}", '\n')
    # Run Dipole if specified in run list
    if 'Dipole' in run:
        if i == 0:
            load_key = 'omni'
        else:
            load_key = key
        print('Dipole')
        start_time = time.time()
        try:
            dipole(omni_path, save_path, load_key=key, key=key)
        except ImportError:
            warnings.warn('Dipole package likely missing. Available at: https://github.com/klaundal/dipole', ImportWarning)
            raise
        i += 1
        end_time = time.time()
        elapsed_time = end_time - start_time
        formatted_time = format_elapsed_time(int(elapsed_time))
        print('\n', f"Time elapsed: {formatted_time}", '\n')
    # Run Time Shift if specified in run list
    if 'Time Shift' in run:
        if i == 0:
            load_key = 'omni'
        else:
            load_key = key
        print('Time Shift')
        start_time = time.time()
        time_shift(omni_path, save_path, shift=shift, load_key=key, key=key)
        i += 1
        end_time = time.time()
        elapsed_time = end_time - start_time
        formatted_time = format_elapsed_time(int(elapsed_time))
        print('\n', f"Time elapsed: {formatted_time}", '\n')