#!/usr/bin/env python3

import h5py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.colors as mcolors

# Mapping from NDBC standard measurement names to their units
units_map = {
    'WDIR': 'degrees_true',
    'WSPD': 'm/s',
    'GST': 'm/s',
    'WVHT': 'm',
    'DPD': 'sec',
    'APD': 'sec',
    'MWD': 'degrees_true',
    'PRES': 'hPa',
    'ATMP': 'degC',
    'WTMP': 'degC',
    'DEWP': 'degC',
    'VIS': 'nmi',
    'PTDY': 'hPa',
    'TIDE': 'ft',
    # Add more mappings as needed
}

def find_latest_file(directory, extension):
    files = glob.glob(os.path.join(directory, f'*.{extension}'))
    latest_file = max(files, key=os.path.getmtime, default=None)
    return latest_file

def load_hdf5_data(filename):
    with h5py.File(filename, 'r') as file:
        data = {key: np.array(file[key]) for key in file.keys()}
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    return df

def plot_data(df):
    plt.style.use('ggplot')
    xkcd_colors = list(mcolors.XKCD_COLORS.values())
    columns_to_plot = [col for col in df.columns if col != 'timestamp']
    num_plots = len(columns_to_plot)

    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 6*num_plots), sharex=True)

    for i, column in enumerate(columns_to_plot):
        ax = axes[i] if num_plots > 1 else axes

        # Filter out values over 90
        filtered_data = df[df[column] <= 90]

        unit = units_map.get(column, "Unknown Unit")
        ax.plot(filtered_data['timestamp'], filtered_data[column], label=f"{column} over Time", color=xkcd_colors[i % len(xkcd_colors)])
        ax.set_ylabel(f"{column} ({unit})")
        ax.legend()

    plt.xlabel('Time')
    plt.suptitle('Meteorological Data Analysis')
    plt.show()

if __name__ == "__main__":
    data_directory = 'Data'
    latest_file = find_latest_file(data_directory, 'hdf5')

    if latest_file:
        df = load_hdf5_data(latest_file)
        plot_data(df)
    else:
        print(f"No HDF5 files found in {data_directory}.")
