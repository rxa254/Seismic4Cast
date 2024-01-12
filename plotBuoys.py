#!/usr/bin/env python3

"""
Script Name: plotBuoys.py
Description: This script generates plots of oceanographic data from Pickle files for multiple years. 
             It supports individual plots for each buoy and overlayed plots combining 
             multiple buoys' data. The script reads units for each data type and 
             buoy IDs from a YAML configuration file.

Usage:
    Run the script with optional arguments:
    - python3 plotBuoys.py --overlay to generate overlayed plots
    - python3 plotBuoys.py to generate individual plots
    - Add --start_year and --end_year to specify the range of years

Author       : Rana X Adhikari
Created on   : Dec - 2023
Last Modified: Jan - 2024
"""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import seaborn as sns
import numpy as np
import os
import glob
import argparse
import yaml

# Enhanced plot style settings
plt.style.use('bmh')
#sns.set_context('talk')  # Larger plot elements suitable for presentations

# Default data directory
data_dir = 'Data/'

def load_config(config_file):
    """
    Load configuration from a YAML file.
    
    Args:
    - config_file (str): Path to the YAML configuration file.

    Returns:
    - dict: Configuration data.
    """
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)
    

def find_pickle_files(directory, config, start_year, end_year):
    """
    Find all Pickle files in the specified directory within the given year range.
    
    Args:
    - directory (str): Directory to search for Pickle files.
    - buoy_ids (list): List of buoy IDs.
    - start_year (int): Start year of the data.
    - end_year (int): End year of the data.

    Returns:
    - list: List of file paths to Pickle files within the specified year range.
    """
    units_map = config['units_map']
    buoy_ids  = config['buoys']

    all_files = glob.glob(os.path.join(directory, '*.pkl'))
    filtered_files = [f for f in all_files if any(str(year) in f for year in range(start_year, end_year + 1))]
    return filtered_files


def load_pickle_data(filename):
    """
    Load and preprocess data from a Pickle file.
    
    Args:
    - filename (str): Path to the Pickle file.

    Returns:
    - DataFrame: Preprocessed data.
    """
    df = pd.read_pickle(filename, compression='gzip')
    for col in df.columns:
        if col != 'timestamp':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def filter_data(df, column):
    """
    Filter data based on a condition.
    
    Args:
    - df (DataFrame): Data to filter.
    - column (str): Column name to apply the filter on.

    Returns:
    - DataFrame: Filtered data.
    """
    if column == 'WVHT':
        ymin = 0
        ymax = 8.7
    elif column == 'DPD':
        ymin = 0
        ymax = 45
    elif column == 'APD':
        ymin = 0
        ymax = 45
    elif column == 'WTMP':
        ymin = 0
        ymax = 50 # very hot!
    else:
        ymin = 0
        ymax = 90
    

    x = df[df[column] <= ymax] # select the reasonable range of data
    filtered_data = x[x[column] > ymin]
    #filtered_data = df  # does no filtering
    
    return filtered_data

def determine_y_scales(pickle_files, column):
    """
    Determine global maximum and minimum y-scales for a specific data column.
    
    Args:
    - pickle_files (list): List of Pickle file paths.
    - column (str): Data column to determine scales for.

    Returns:
    - tuple: (max, min) values for the y-scale.
    """
    max_val = -np.inf
    min_val =  np.inf
    for file in pickle_files:
        df = load_pickle_data(file)
        if column in df.columns:
            filtered_df = filter_data(df, column)
            max_val = max(max_val, filtered_df[column].max())
            min_val = min(min_val, filtered_df[column].min())
    return max_val, min_val


def plot_individual_data(pickle_files, pdf, config, start_year, end_year):
    """
    Plot individual data for each buoy and column across the specified year range.

    Args:
    - pickle_files (list)  : List of Pickle file paths.
    - pdf (PdfPages object): PDF file to save plots to.
    - units_map (dict)     : Units mapping for each data type.
    - buoy_ids (list)      : List of buoy IDs.
    - start_year (int)     : Start year of data.
    - end_year (int)       : End year of data.
    """
    units_map = config['units_map']
    buoy_ids  = config['buoys']


    for buoy_id in buoy_ids:
        str_buoy_id = str(buoy_id)  # Convert buoy_id to string
        for column in units_map.keys():
            fig, ax = plt.subplots(figsize = (10, 6))
            max_val, min_val = determine_y_scales(pickle_files, column)
            legend_labels = []  # List to track labels for legend

            for file in pickle_files:
                if str_buoy_id in file:  # Use string representation of buoy_id
                    df = load_pickle_data(file)
                    if column in df.columns:
                        filtered_data = filter_data(df, column)
                        label = f"{os.path.basename(file).split('_')[1]} - {column}"
                        ax.plot(filtered_data['timestamp'], filtered_data[column], label=label)
                        legend_labels.append(label)

            if legend_labels:  # Only add legend if there are labels
                ax.legend()
            ax.set_ylabel(f"{column} ({units_map.get(column, 'Unknown Unit')})")
            ax.set_xlabel('Time')
            ax.set_title(f'{column} Data for Buoy {str_buoy_id} ({start_year}-{end_year})')
            if max_val != -np.inf and min_val != np.inf:
                ax.set_ylim(min_val, max_val)
            plt.xticks(rotation = 45)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)





def plot_overlayed_data(pickle_files, pdf, config, start_year, end_year):
    """
    Generate overlayed plots for multiple buoys and save to a PDF.

    Args:
    - pickle_files (list)  : List of Pickle file paths.
    - pdf (PdfPages object): PDF file to save plots to.
    - units_map (dict)     : Units mapping for each data type.
    """
    units_map = config['units_map'] # units for each sensor
    buoy_ids  = config['buoys']     # ID numbers of Buoys
    mycolumns = config['columns']   # list of sensors on each Buoy

    years     = range(start_year, end_year+1, 1)
    colors    = {41002: 'xkcd:purple',
                 41004: 'xkcd:turquoise',
                 41008: 'xkcd:azure',
                 41009: 'xkcd:neon blue',
                 41013: 'xkcd:soft blue',
                 41047: 'xkcd:purply blue',
                 42002: 'xkcd:brick',
                 42019: 'xkcd:neon pink',
                 42036: 'xkcd:jade',
                 42040: 'xkcd:lime green',
                 42055: 'xkcd:vomit',
                }

    # make 1 plot for each sensor, spanning all years
    # all buoys on the same plot, 1 color per buoy
    for column in mycolumns:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for year in years:
            for buoy in buoy_ids:
                filename = "BuoyData_" + str(buoy) + '_' + str(year) + '.pkl'
                try:
                    df = load_pickle_data(data_dir + filename)
                    df = filter_data(df, column)
                    ax.plot(df['timestamp'], df[column],
                            color=colors[buoy], ls='', marker='.', ms=1, alpha = 0.36,
                            label=f"Buoy {buoy}", rasterized=True)
                except:
                    print('File not found:' + filename)
        ax.set_ylabel(f"{column} ({units_map.get(column, 'Unknown Unit')})")
        ax.set_xlabel('Time')
        ax.set_title(f'{column} Data from Multiple Buoys')
        #ax.legend()
        plt.xticks(rotation = 45)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)


def main(directory, overlay, config_file, start_year, end_year):
    """
    Main function to execute the script.

    Args:
    - directory (str)  : Directory containing Pickle files.
    - overlay (bool)   : Flag to indicate overlay mode.
    - config_file (str): Path to the YAML configuration file.
    - start_year (int) : Start year for data to be plotted.
    - end_year (int)   : End year for data to be plotted.
    """
    config    = load_config(config_file)

    figures_directory = 'Figures'
    os.makedirs(figures_directory, exist_ok=True)
    pdf_path = os.path.join(figures_directory, 'buoy_data_plots.pdf')
    
    pickle_files = find_pickle_files(directory, config, start_year, end_year)
    with PdfPages(pdf_path) as pdf:
        if overlay:
            plot_overlayed_data( pickle_files, pdf, config, start_year, end_year)
        else:
            plot_individual_data(pickle_files, pdf, config, start_year, end_year)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot NDBC buoy data.")
    parser.add_argument("directory", type=str, nargs='?', default=data_dir,
                        help="Directory containing Pickle files. Default is 'Data/'.")
    #parser.add_argument("--overlay", action='store_true',
    #                    help="Overlay data from all buoys on single plots.")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to the configuration YAML file. Default is 'config.yaml'.")
    parser.add_argument("--start_year", type=int, default=2017,
                        help="Start year for data to be plotted.")
    parser.add_argument("--end_year", type=int, default=2022,
                        help="End year for data to be plotted.")
    args = parser.parse_args()

    main(args.directory, True, args.config, args.start_year, args.end_year)
