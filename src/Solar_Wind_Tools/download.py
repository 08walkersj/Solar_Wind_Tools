
"""
(Download function is based on code by Anders Ohma: GitHub aohma)
"""

import cdflib # pip install cdflib
import pandas as pd
import numpy as np
import os
import glob
import shutil
def validinput(inputstr, positive_answer, negative_answer):
    """
    Prompt the user for a valid input and return a boolean based on the response.

    Parameters:
    - inputstr (str): The prompt string displayed to the user.
    - positive_answer (str): The positive answer expected from the user.
    - negative_answer (str): The negative answer expected from the user.

    Returns:
    - bool: True if the user's input matches the positive_answer,
            False if the user's input matches the negative_answer.

    Example:
    >>> validinput('Continue? (y/n)', 'y', 'n')
    """
    answer = input(inputstr + '\n').lower()
    if answer == positive_answer:
        return True
    elif answer == negative_answer:
        return False
    else:
        print('Invalid response should be either ' + str(positive_answer) + ' or ' + str(negative_answer))
        return validinput(inputstr, positive_answer, negative_answer)

from platform import system
# If Linux download uses wget
if 'Linux' in system():
    def download(url, filename):
        return os.system(f'wget {url} -O {filename}')
# If not Linux use urllib.request.urlretrieve
else:
    from urllib.request import urlretrieve
    import sys
    def download_progress_hook(count, block_size, total_size):
        """
        Report hook to display a progress bar for downloading.
        
        :param count: Current block number being downloaded.
        :param block_size: Size of each block (in bytes).
        :param total_size: Total size of the file (in bytes).
        """
        # Calculate percentage of the download
        downloaded_size = count * block_size
        percentage = min(100, downloaded_size * 100 / total_size)
        
        # Create a simple progress bar
        progress_bar = f"\rDownloading: {percentage:.2f}% [{downloaded_size}/{total_size} bytes]"
        
        # Update the progress on the same line
        sys.stdout.write(progress_bar)
        sys.stdout.flush()

        # When download is complete
        if downloaded_size >= total_size:
            print("\nDownload complete!")
    def download(url, file_name):
        print(file_name)
        return urlretrieve(url, file_name, reporthook=download_progress_hook)

def download_omni_1min(fromYear, toYear, monthFirstYear=1, monthLastYear=12, path='./omni_1min.h5', parallel=True):
    """
    Download OMNI 1min data and store it in an HDF file.

    Parameters:
    - fromYear (int): Download data from this year onwards.
    - toYear (int): Download data up to this year.
    - monthFirstYear (int, optional): First month to include for the first year. Default is 1.
    - monthLastYear (int, optional): Last month to include for the last year. Default is 12.
    - path (str, optional): Path to save the HDF file. Default is './omni_1min.h5'.
    - parallel (bool, optional): enables parallel download to improve download speed. Default is True
 
    Raises:
    - ValueError: If fromYear is less than 1981.
                  If the file already exists and the user chooses not to continue.

    Returns:
    None

    Example:
    >>> download_omni_1min(2000, 2005)
    """
    if fromYear < 1981:
        raise ValueError('fromYear must be >=1981')
    if os.path.isfile(path):
        if not validinput('file already exists and more omni will be added which can lead to duplication of data continue? (y/n)', 'y', 'n'):
            raise ValueError('User Cancelled Download, Alter file name or path or remove or move the existing file and retry')
    years = np.arange(fromYear, toYear + 1, 1)
    months = []
    for i in np.arange(1, 13, 1):
        months.append('%02i' % i)
    urls= []
    for y in years:
        for m in months:
            if not ((y == years[0]) & (int(m) < monthFirstYear)) | ((y == years[-1]) & (int(m) > monthLastYear)):
                urls.append('https://cdaweb.gsfc.nasa.gov/sp_phys/data/omni/hro_1min/' + str(y) + \
                          '/omni_hro_1min_' + str(y) + str(m) + '01_v01.cdf')
                # file= 'omni_hro_1min_' + str(y) + str(m) + '01_v01.cdf'


    os.makedirs('./omni_tempfiles/', exist_ok=True)
    if parallel:
        from joblib import Parallel, delayed
        download_args = [(url, './omni_tempfiles/'+url.split('/')[-1]) for url in urls]
        Parallel(n_jobs=12, backend='threading')(delayed(download)(*args) for args in download_args)
    else:
        for url in urls:
            download(url, './omni_tempfiles/'+url.split('/')[-1])
    files = glob.glob('./omni_tempfiles/*.cdf')
    files.sort()
    for file in files:
        omni = pd.DataFrame()
        cdf_file = cdflib.CDF(file)
        varlist = cdf_file.cdf_info().zVariables
        for v in varlist:
            omni[v] = cdf_file.varget(v)
            fillval = cdf_file.varattsget(v)['FILLVAL']
            omni[v] = omni[v].replace(fillval, np.nan)
        omni.index = pd.to_datetime(cdflib.cdfepoch.unixtime(cdf_file.varget('Epoch')), unit='s')
        omni[['AE_INDEX', 'AL_INDEX', 'AU_INDEX', 'PC_N_INDEX']] = omni[
            ['AE_INDEX', 'AL_INDEX', 'AU_INDEX', 'PC_N_INDEX']].astype('float64')
        omni.to_hdf(path, key='omni', mode='a', append=True, format='t', data_columns=True)
    shutil.rmtree('./omni_tempfiles')
