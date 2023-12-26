#!/usr/bin/env python3

"""
Script Name: getNDBC.py
Description: This script downloads and processes historical oceanographic data 
             from the National Data Buoy Center (NDBC). It focuses on extracting 
             wave height, wave period, and water temperature data for specified 
             buoys and time periods, converting timestamps to GPS time. The buoy 
             IDs are read from a text file or a default buoy in the Gulf of Mexico
             is used for the year 2022.

Author: [Your Name]
Created on: [Date Created]
Last Modified: [Last Modification Date]

Usage:
    Run the script with the file containing buoy IDs and the year as arguments:
    python3 getNDBC.py --buoy_file buoys.txt --year 2022
    Or run without arguments to use the default buoy and year:
    python3 getNDBC.py

Dependencies:
    Requires Python 3 and the following libraries: 
    - requests
    - pandas
    - h5py
    - numpy
    - argparse
"""

import requests
import h5py
import pandas as pd
import io
import os
import numpy as np
import argparse

data_dir = 'Data/'

def download_ndbc_historical_data(buoy_id, year):
    url = f"https://www.ndbc.noaa.gov/view_text_file.php?filename={buoy_id}h{year}.txt.gz&dir=data/historical/stdmet/"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve data for buoy {buoy_id} for {year}")
        return None
    return response.content.decode('utf-8')

def parse_data_to_dict(data, months=list(range(1, 12 + 1))):
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

    gps_epoch = pd.Timestamp('1980-01-06 00:00:00')
    df_filtered['timestamp'] = (df_filtered['timestamp'] - gps_epoch).dt.total_seconds()

    df_filtered['WVHT'] = pd.to_numeric(df_filtered['WVHT'], errors='coerce')
    df_filtered['DPD'] = pd.to_numeric(df_filtered['DPD'], errors='coerce')
    df_filtered['WTMP'] = pd.to_numeric(df_filtered['WTMP'], errors='coerce')

    return df_filtered[['timestamp', 'WVHT', 'DPD', 'WTMP']]

def save_dict_to_hdf5(data_frame, buoy_id, year):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    filename = os.path.join(data_dir, f'BuoyData_{buoy_id}_{year}.hdf5')
    with h5py.File(filename, 'w') as h5file:
        for column in data_frame.columns:
            data = data_frame[column].to_numpy(na_value=np.nan)
            h5file.create_dataset(column, data=data)

def main(buoy_file, year):
    if buoy_file:
        with open(buoy_file, 'r') as file:
            buoy_ids = file.read().splitlines()
    else:
        # Default buoy ID from the Gulf of Mexico for the year 2022
        buoy_ids = ['42002']  # Replace with the actual buoy ID

    for buoy_id in buoy_ids:
        data = download_ndbc_historical_data(buoy_id, year)
        if data:
            df = parse_data_to_dict(data)
            save_dict_to_hdf5(df, buoy_id, year)
            print(f"Data processed and saved for buoy {buoy_id} for {year}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process NDBC buoy data.")
    parser.add_argument("--buoy_file", type=str, help="File containing list of buoy IDs.")
    parser.add_argument("--year", type=int, default=2022, help="The year for which to download data.")
    args = parser.parse_args()
    main(args.buoy_file, args.year)
