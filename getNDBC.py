#!/usr/bin/env python3
"""
Script Name: getNDBC.py
Description: This script downloads and processes historical oceanographic data 
             from the National Data Buoy Center (NDBC). It focuses on extracting 
             wave height, wave period, and water temperature data for specified 
             buoys and time periods, converting timestamps to GPS time.

Author: Rana X Adhikari / Caltech
Created on: 25 - Dec - 2023
Last Modified: chech github

Dependencies:
    Requires Python 3 and the following libraries: 
    - requests
    - pandas
    - h5py
    - numpy
"""

import requests
import h5py
import pandas as pd
import io
import os
import numpy as np

data_dir = 'Data/'

def download_ndbc_historical_data(buoy_id='46026', year = 2022):
    url = f"https://www.ndbc.noaa.gov/view_text_file.php?filename={buoy_id}h{year}.txt.gz&dir=data/historical/stdmet/"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve data for buoy {buoy_id} for {year}")
        return None
    return response.content.decode('utf-8')

def parse_data_to_dict(data, months = list(range(1,12+1))):
    df = pd.read_fwf(io.StringIO(data), skiprows=[1], header=0)
    df_filtered = df[df['MM'].isin(months)].copy()

    required_columns = ['#YY', 'MM', 'DD', 'hh', 'mm']
    if not all(column in df.columns for column in required_columns):
        raise ValueError("One or more required columns are missing in the data.")

    for column in required_columns:
        df_filtered.loc[:, column] = df_filtered[column].apply(lambda x: f"{x:02d}")

    df_filtered.loc[:, 'timestamp'] = pd.to_datetime(df_filtered[required_columns].apply(lambda row: ''.join(row.values), axis=1), format='%Y%m%d%H%M', errors='coerce')

    if df_filtered['timestamp'].isna().any():
        print("Warning: Some timestamps could not be parsed and will be NaN.")

    # Convert datetime to GPS time (seconds since January 6, 1980)
    gps_epoch = pd.Timestamp('1980-01-06 00:00:00')
    df_filtered['timestamp'] = (df_filtered['timestamp'] - gps_epoch).dt.total_seconds()

    df_filtered['WVHT'] = pd.to_numeric(df_filtered['WVHT'], errors='coerce')
    df_filtered['DPD']  = pd.to_numeric(df_filtered['DPD'], errors='coerce')
    df_filtered['WTMP'] = pd.to_numeric(df_filtered['WTMP'], errors='coerce')

    return df_filtered[['timestamp', 'WVHT', 'DPD', 'WTMP']]

def save_dict_to_hdf5(data_frame, filename):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    with h5py.File(os.path.join(data_dir, filename), 'w') as h5file:
        for column in data_frame.columns:
            data = data_frame[column].to_numpy(na_value=np.nan)
            h5file.create_dataset(column, data=data)

############################################################
if __name__ == "__main__":
    buoy_id = '46026'
    year    = 2022
    #months = [1, 2, 3]
    months  = list(range(1,12+1)) # get data for all 12 months
    data    = download_ndbc_historical_data(buoy_id, year)

    if data:
        df  = parse_data_to_dict(data, months)
        save_dict_to_hdf5(df, f'buoy_data_{buoy_id}_{year}.hdf5')
