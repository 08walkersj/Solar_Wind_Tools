#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 12:29:06 2020

@author: aohma (Download function is based on code by Anders Ohma)
"""

import cdflib # pip install cdflib
import pandas as pd
import numpy as np
import os
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


def download_omni_1min(fromYear, toYear, monthFirstYear=1, monthLastYear=12, path='./omni_1min.h5'):
    """
    Download OMNI 1min data and store it in an HDF file.

    Parameters:
    - fromYear (int): Download data from this year onwards.
    - toYear (int): Download data up to this year.
    - monthFirstYear (int, optional): First month to include for the first year. Default is 1.
    - monthLastYear (int, optional): Last month to include for the last year. Default is 12.
    - path (str, optional): Path to save the HDF file. Default is './omni_1min.h5'.

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

    for y in years:
        for m in months:
            if not ((y == years[0]) & (int(m) < monthFirstYear)) | ((y == years[-1]) & (int(m) > monthLastYear)):
                command = 'wget https://cdaweb.gsfc.nasa.gov/sp_phys/data/omni/hro_1min/' + str(y) + \
                          '/omni_hro_1min_' + str(y) + str(m) + '01_v01.cdf'
                os.system(command)

                omni = pd.DataFrame()
                cdf_file = cdflib.CDF('omni_hro_1min_' + str(y) + str(m) + '01_v01.cdf')
                varlist = cdf_file.cdf_info().zVariables
                for v in varlist:
                    omni[v] = cdf_file.varget(v)
                    fillval = cdf_file.varattsget(v)['FILLVAL']
                    omni[v] = omni[v].replace(fillval, np.nan)
                omni.index = pd.to_datetime(cdflib.cdfepoch.unixtime(cdf_file.varget('Epoch')), unit='s')
                omni[['AE_INDEX', 'AL_INDEX', 'AU_INDEX', 'PC_N_INDEX']] = omni[
                    ['AE_INDEX', 'AL_INDEX', 'AU_INDEX', 'PC_N_INDEX']].astype('float64')
                omni.to_hdf(path, key='omni', mode='a', append=True, format='t', data_columns=True)
                # cdf_file.close()
                os.remove('omni_hro_1min_' + str(y) + str(m) + '01_v01.cdf')
