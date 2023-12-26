#!/usr/bin/env python3

"""
Script Name: getNDBC.py
Description: This script downloads and processes historical oceanographic data 
             from the National Data Buoy Center (NDBC). It focuses on extracting 
             specified data columns for specified buoys and time periods, as 
             defined in a YAML configuration file.

Author: [Your Name]
Created on: [Date Created]
Last Modified: [Last Modification Date]

Usage:
    Run the script with a YAML configuration file:
    python3 getNDBC.py --config_file config.yaml --year 2022
    Or run without specifying a config file to use the default 'config.yaml':
    python3 getNDBC.py --year 2022

Dependencies:
    Requires Python 3 and the following libraries: 
    - requests
    - pandas
    - argparse
    - pyyaml
"""

import requests
import pandas as pd
import io
import os
import argparse
import yaml

data_dir = 'Data/'

def download_ndbc_historical_data(buoy_id, year):
    url = f"https://www.ndbc.noaa.gov/view_text_file.php?filename={buoy_id}h{year}.txt.gz&dir=data/historical/stdmet/"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve data for buoy {buoy_id} for {year}")
        return None
    return response.content.decode('utf-8')

def parse_data_to_dict(data, columns, months=list(range(1, 12 + 1))):
    df = pd.read_fwf(io.StringIO(data), skiprows=[1], header=0)
    df_filtered = df[df['MM'].isin(months)].copy()

    required_columns = ['#YY', 'MM', 'DD', 'hh', 'mm'] + columns
    for column in required_columns:
        if column in df.columns:
            df_filtered.loc[:, column] = df_filtered[column].astype(str).str.zfill(2)

    df_filtered['timestamp'] = pd.to_datetime(df_filtered[['#YY', 'MM', 'DD', 'hh', 'mm']].agg(''.join, axis=1), format='%Y%m%d%H%M')
    return df_filtered[['timestamp'] + columns]

def save_data(df, buoy_id, year):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    filename_pickle = os.path.join(data_dir, f'BuoyData_{buoy_id}_{year}.pkl')
    df.to_pickle(filename_pickle, compression='gzip')

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main(config_file, year):
    config = load_config(config_file)
    buoy_ids = config['buoys']
    columns = config['columns']

    for buoy_id in buoy_ids:
        data = download_ndbc_historical_data(buoy_id, year)
        if data:
            df = parse_data_to_dict(data, columns)
            save_data(df, buoy_id, year)
            print(f"Data processed and saved for buoy {buoy_id} for {year}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process NDBC buoy data.")
    parser.add_argument("--config_file", type=str, default='config.yaml',
                         help="YAML configuration file. Default is 'config.yaml'.")
    parser.add_argument("--year", type=int, default=2022, 
                        help="The year for which to download data.")
    args = parser.parse_args()
    main(args.config_file, args.year)
