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

Author: [Your Name]
Created on: [Date Created]
Last Modified: [Last Modification Date]
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
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context('talk')  # Larger plot elements suitable for presentations

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

def find_pickle_files(directory, buoy_ids, start_year, end_year):
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
    return df[df[column] <= 90]

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
    min_val = np.inf
    for file in pickle_files:
        df = load_pickle_data(file)
        if column in df.columns:
            filtered_df = filter_data(df, column)
            max_val = max(max_val, filtered_df[column].max())
            min_val = min(min_val, filtered_df[column].min())
    return max_val, min_val

def plot_individual_data(pickle_files, pdf, units_map, buoy_ids, start_year, end_year):
    """
    Plot individual data for each buoy and column across the specified year range.

    Args:
    - pickle_files (list): List of Pickle file paths.
    - pdf (PdfPages object): PDF file to save plots to.
    - units_map (dict): Units mapping for each data type.
    - buoy_ids (list): List of buoy IDs.
    - start_year (int): Start year of data.
    - end_year (int): End year of data.
    """
    for buoy_id in buoy_ids:
        str_buoy_id = str(buoy_id)  # Convert buoy_id to string
        for column in units_map.keys():
            fig, ax = plt.subplots(figsize=(10, 6))
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
            plt.xticks(rotation=45)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)



def plot_overlayed_data(pickle_files, pdf, units_map):
    """
    Generate overlayed plots for multiple buoys and save to a PDF.

    Args:
    - pickle_files (list): List of Pickle file paths.
    - pdf (PdfPages object): PDF file to save plots to.
    - units_map (dict): Units mapping for each data type.
    """
    all_data = {}
    for file in pickle_files:
        df = load_pickle_data(file)
        buoy_id = os.path.basename(file).split('_')[1]
        for column in df.columns:
            if column != 'timestamp':
                filtered_data = filter_data(df, column)
                if column not in all_data:
                    all_data[column] = {}
                all_data[column][buoy_id] = filtered_data[['timestamp', column]]

    for column, data in all_data.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for buoy_id, df in data.items():
            ax.plot(df['timestamp'], df[column], label=f"Buoy {buoy_id}")
        ax.set_ylabel(f"{column} ({units_map.get(column, 'Unknown Unit')})")
        ax.set_xlabel('Time')
        ax.set_title(f'{column} Data from Multiple Buoys')
        ax.legend()
        plt.xticks(rotation=45)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

def main(directory, overlay, config_file, start_year, end_year):
    """
    Main function to execute the script.

    Args:
    - directory (str): Directory containing Pickle files.
    - overlay (bool): Flag to indicate overlay mode.
    - config_file (str): Path to the YAML configuration file.
    - start_year (int): Start year for data to be plotted.
    - end_year (int): End year for data to be plotted.
    """
    config = load_config(config_file)
    units_map = config['units_map']
    buoy_ids = config['buoys']

    figures_directory = 'Figures'
    os.makedirs(figures_directory, exist_ok=True)
    pdf_path = os.path.join(figures_directory, 'buoy_data_plots.pdf')
    
    pickle_files = find_pickle_files(directory, buoy_ids, start_year, end_year)
    with PdfPages(pdf_path) as pdf:
        if overlay:
            plot_overlayed_data(pickle_files, pdf, units_map)
        else:
            plot_individual_data(pickle_files, pdf, units_map, buoy_ids, start_year, end_year)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot NDBC buoy data.")
    parser.add_argument("directory", type=str, nargs='?', default=data_dir, help="Directory containing Pickle files. Default is 'Data/'.")
    parser.add_argument("--overlay", action='store_true', help="Overlay data from all buoys on single plots.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration YAML file. Default is 'config.yaml'.")
    parser.add_argument("--start_year", type=int, default=2017, help="Start year for data to be plotted.")
    parser.add_argument("--end_year", type=int, default=2022, help="End year for data to be plotted.")
    args = parser.parse_args()

    main(args.directory, args.overlay, args.config, args.start_year, args.end_year)
