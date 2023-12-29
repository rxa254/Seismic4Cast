#!/usr/bin/env python3

"""
Script Name: getNDBC.py
Description: This script downloads and processes historical oceanographic data 
             from the National Data Buoy Center (NDBC). It focuses on extracting 
             data for specified buoys over a range of years, as defined in a YAML configuration file.

Author: [Your Name]
Created on: [Date Created]
Last Modified: [Last Modification Date]

Usage:
    Run the script with a YAML configuration file:
    python3 getNDBC.py --config_file config.yaml --start_year 2017 --end_year 2022
    Or run without specifying a config file to use the default 'config.yaml':
    python3 getNDBC.py --start_year 2017 --end_year 2022

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
    """
    Download historical data for a specific buoy and year from NDBC.

    Args:
    - buoy_id (str): Buoy ID.
    - year (int): Year of data.

    Returns:
    - str: Decoded content of the data file.
    """
    url = f"https://www.ndbc.noaa.gov/view_text_file.php?filename={buoy_id}h{year}.txt.gz&dir=data/historical/stdmet/"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve data for buoy {buoy_id} for {year}")
        return None
    return response.content.decode('utf-8')

def parse_data_to_dict(data, columns, months=list(range(1, 12 + 1))):
    """
    Parse data string to a DataFrame.

    Args:
    - data (str): Data string from NDBC.
    - columns (list): List of columns to include.
    - months (list): List of months to include in the data.

    Returns:
    - DataFrame: Parsed data.
    """
    df = pd.read_fwf(io.StringIO(data), skiprows=[1], header=0)
    df_filtered = df[df['MM'].isin(months)].copy()

    required_columns = ['#YY', 'MM', 'DD', 'hh', 'mm'] + columns
    for column in required_columns:
        if column in df.columns:
            df_filtered.loc[:, column] = df_filtered[column].astype(str).str.zfill(2)

    df_filtered['timestamp'] = pd.to_datetime(df_filtered[['#YY', 'MM', 'DD', 'hh', 'mm']].agg(''.join, axis=1), format='%Y%m%d%H%M')
    return df_filtered[['timestamp'] + columns]

def save_data(df, buoy_id, year):
    """
    Save data to a Pickle file.

    Args:
    - df (DataFrame): Data to save.
    - buoy_id (str): Buoy ID.
    - year (int): Year of data.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    filename_pickle = os.path.join(data_dir, f'BuoyData_{buoy_id}_{year}.pkl')
    df.to_pickle(filename_pickle, compression='gzip')

def load_config(config_file):
    """
    Load configuration from a YAML file.

    Args:
    - config_file (str): YAML configuration file path.

    Returns:
    - dict: Configuration data.
    """
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main(config_file, start_year, end_year):
    """
    Main function of the script.

    Args:
    - config_file (str): YAML configuration file path.
    - start_year (int): Start year for data retrieval.
    - end_year (int): End year for data retrieval.
    """
    config = load_config(config_file)
    buoy_ids = config['buoys']
    columns = config['columns']

    for year in range(start_year, end_year + 1):
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
    parser.add_argument("--start_year", type=int, default=2017, 
                        help="Start year for the data retrieval. Default is 2017.")
    parser.add_argument("--end_year", type=int, default=2022, 
                        help="End year for the data retrieval. Default is 2022.")
    args = parser.parse_args()
    main(args.config_file, args.start_year, args.end_year)
