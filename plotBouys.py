#!/usr/bin/env python3

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
import os
import glob

# Units mapping based on standard NDBC measurements
units_map = {
    'WDIR': 'degrees (True)',
    'WSPD': 'm/s',
    'GST': 'm/s',
    'WVHT': 'm',
    'DPD': 's',
    'APD': 's',
    'MWD': 'degrees (True)',
    'PRES': 'hPa',
    'ATMP': '°C',
    'WTMP': '°C',
    'DEWP': '°C',
    'VIS': 'nmi',
    'PTDY': 'hPa',
    'TIDE': 'ft',
    # Add more mappings as needed
}

def find_pickle_files(directory):
    return glob.glob(os.path.join(directory, '*.pkl'))

def load_pickle_data(filename):
    df = pd.read_pickle(filename, compression='gzip')
    for col in df.columns:
        if col != 'timestamp':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def determine_y_scales(pickle_files):
    global_max = {}
    global_min = {}
    for pickle_file in pickle_files:
        df = load_pickle_data(pickle_file)
        for column in df.columns:
            if column != 'timestamp':
                max_val = df[df[column] <= 90][column].max()
                min_val = df[df[column] <= 90][column].min()
                if not np.isnan(max_val) and not np.isinf(max_val):
                    global_max[column] = max(global_max.get(column, -np.inf), max_val)
                if not np.isnan(min_val) and not np.isinf(min_val):
                    global_min[column] = min(global_min.get(column, np.inf), min_val)
    return global_max, global_min

def plot_data(df, buoy_id, pdf, y_scales):
    global_max, global_min = y_scales
    plt.style.use('bmh')
    columns_to_plot = [col for col in df.columns if col != 'timestamp']
    num_plots = len(columns_to_plot)

    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 6*num_plots), sharex=True)

    for i, column in enumerate(columns_to_plot):
        ax = axes[i] if num_plots > 1 else axes
        filtered_data = df[df[column] <= 90]
        unit = units_map.get(column, "Unknown Unit")
        ax.plot(filtered_data['timestamp'], filtered_data[column], label=f"{column} over Time")
        ax.set_ylabel(f"{column} ({unit})")
        ax.legend()

        # Set y-axis limits if they are valid
        if column in global_max and column in global_min:
            if not np.isnan(global_max[column]) and not np.isinf(global_max[column]) and \
               not np.isnan(global_min[column]) and not np.isinf(global_min[column]):
                ax.set_ylim(global_min[column], global_max[column])

    plt.xlabel('Time')
    plt.suptitle(f'Meteorological Data Analysis for Buoy {buoy_id}')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    data_directory = 'Data'
    pickle_files = find_pickle_files(data_directory)
    global_max, global_min = determine_y_scales(pickle_files)

    with PdfPages('Figures/' + 'buoy_data_plots.pdf') as pdf:
        if pickle_files:
            for pickle_file in pickle_files:
                df = load_pickle_data(pickle_file)
                buoy_id = os.path.basename(pickle_file).split('_')[1]  # Extract buoy ID from filename
                plot_data(df, buoy_id, pdf, (global_max, global_min))
        else:
            print(f"No Pickle files found in {data_directory}.")
