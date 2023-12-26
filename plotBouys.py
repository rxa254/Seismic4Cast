#!/usr/bin/env python3

import h5py
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

def find_hdf5_files(directory):
    return glob.glob(os.path.join(directory, '*.hdf5'))

def load_hdf5_data(filename):
    with h5py.File(filename, 'r') as file:
        data = {key: np.array(file[key]) for key in file.keys()}
        df = pd.DataFrame(data)
        gps_epoch = pd.Timestamp('1980-01-06 00:00:00')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', origin=gps_epoch)
        return df

def determine_y_scales(hdf5_files):
    global_max = {}
    global_min = {}
    for hdf5_file in hdf5_files:
        df = load_hdf5_data(hdf5_file)
        for column in df.columns:
            if column != 'timestamp':
                max_val = df[df[column] <= 90][column].max()
                min_val = df[df[column] <= 90][column].min()
                global_max[column] = max(global_max.get(column, -np.inf), max_val)
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

        ax.set_ylim(global_min[column], global_max[column])

    plt.xlabel('Time')
    plt.suptitle(f'Meteorological Data Analysis for Buoy {buoy_id}')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    data_directory = 'Data'
    hdf5_files = find_hdf5_files(data_directory)
    global_max, global_min = determine_y_scales(hdf5_files)

    with PdfPages('buoy_data_plots.pdf') as pdf:
        if hdf5_files:
            for hdf5_file in hdf5_files:
                df = load_hdf5_data(hdf5_file)
                buoy_id = os.path.basename(hdf5_file).split('_')[0]  # Extract buoy ID from filename
                plot_data(df, buoy_id, pdf, (global_max, global_min))
        else:
            print(f"No HDF5 files found in {data_directory}.")
