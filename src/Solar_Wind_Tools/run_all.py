from .window import circular_variance, coupling, dipole, time_shift
from .download import download_omni_1min
import os
from datetime import datetime as dt
def run_all(save_path, omni_path=False, start=20, end=10, start_year=1981, end_year=False, key='omni_window', run=[]):
    """
    Run a series of functions on the OMNI dataset and save the results.

    Parameters:
    - save_path (str): Path to save the output HDF5 file.
    - omni_path (str, optional): Path to the OMNI dataset HDF5 file. If not provided, it uses save_path.
    - start (int, optional): Start window size for the analysis.
    - end (int, optional): End window size for the analysis.
    - start_year (int, optional): Start year for downloading the OMNI dataset.
    - end_year (int, optional): End year for downloading the OMNI dataset. If not provided, it uses the current year.
    - key (str, optional): Key for loading data from the HDF5 files.
    - run (list, optional): List of functions to run. Default is ['Download', 'Circular Variance', 'Coupling', 'Dipole', 'Time Shift'].

    Returns:
    None

    Example:
    >>> run_all('output.h5', start=10, end=5, start_year=2000, end_year=2005, run=['Download', 'Circular Variance', 'Coupling'])
    """
    # Check if run list is empty and set default functions to run
    if not len(run):
        run = ['Download', 'Circular Variance', 'Coupling', 'Dipole', 'Time Shift']

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
    if 'Circular Variance' in run:
        if i == 0:
            load_key = 'omni'
        else:
            load_key = key
        print('Circular Variance')
        circular_variance(omni_path, save_path, window=start + end, load_key=load_key, key=key)
        i += 1

    # Run Coupling if specified in run list
    if 'Coupling' in run:
        if i == 0:
            load_key = 'omni'
        else:
            load_key = key
        print('Coupling')
        coupling(omni_path, save_path, window=start + end, load_key=key, key=key)
        i += 1

    # Run Dipole if specified in run list
    if 'Dipole' in run:
        if i == 0:
            load_key = 'omni'
        else:
            load_key = key
        print('Dipole')
        dipole(omni_path, save_path, window=start + end, load_key=key, key=key)
        i += 1

    # Run Time Shift if specified in run list
    if 'Time Shift' in run:
        if i == 0:
            load_key = 'omni'
        else:
            load_key = key
        print('Time Shift')
        time_shift(omni_path, save_path, start, end, load_key=key, key=key)
        i += 1