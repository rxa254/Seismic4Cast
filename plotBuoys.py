#!/usr/bin/env python3

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import seaborn as sns
import numpy as np
import os
import glob
import argparse

# Enhanced plot style settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context('talk')  # Larger plot elements suitable for presentations

# Units mapping based on standard NDBC measurements
units_map = {
    # ... (your units mapping here)
}

# Default data directory
data_dir = 'Data/'

def find_pickle_files(directory):
    return glob.glob(os.path.join(directory, '*.pkl'))

def load_pickle_data(filename):
    df = pd.read_pickle(filename, compression='gzip')
    for col in df.columns:
        if col != 'timestamp':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def filter_data(df, column):
    return df[df[column] <= 90]

def determine_y_scales(pickle_files):
    global_max = {}
    global_min = {}
    for pickle_file in pickle_files:
        df = load_pickle_data(pickle_file)
        for column in df.columns:
            if column != 'timestamp':
                filtered_df = filter_data(df, column)
                max_val = filtered_df[column].max()
                min_val = filtered_df[column].min()
                global_max[column] = max(global_max.get(column, -np.inf), max_val)
                global_min[column] = min(global_min.get(column, np.inf), min_val)
    return global_max, global_min

def plot_individual_data(pickle_files, pdf, y_scales):
    global_max, global_min = y_scales
    color_palette = sns.color_palette('husl', len(pickle_files))  # Use a pleasing color palette

    for i, pickle_file in enumerate(pickle_files):
        df = load_pickle_data(pickle_file)
        buoy_id = os.path.basename(pickle_file).split('_')[1]
        columns_to_plot = [col for col in df.columns if col != 'timestamp']
        num_plots = len(columns_to_plot)

        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 6*num_plots), sharex=True)

        for j, column in enumerate(columns_to_plot):
            ax = axes[j] if num_plots > 1 else axes
            filtered_data = filter_data(df, column)
            unit = units_map.get(column, "Unknown Unit")
            ax.plot(filtered_data['timestamp'], filtered_data[column], label=f"{column} over Time", color=color_palette[i])
            ax.set_ylabel(f"{column} ({unit})")
            ax.legend()

            if column in global_max and column in global_min:
                ax.set_ylim(global_min[column], global_max[column])

        plt.xlabel('Time')
        plt.suptitle(f'Meteorological Data Analysis for Buoy {buoy_id}')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

def plot_overlayed_data(pickle_files, pdf, units_map):
    all_data = {}
    color_palette = sns.color_palette('husl', len(pickle_files))  # Use a pleasing color palette

    for pickle_file in pickle_files:
        buoy_id = os.path.basename(pickle_file).split('_')[1]
        df = load_pickle_data(pickle_file)
        for column in df.columns:
            if column != 'timestamp':
                filtered_data = filter_data(df, column)
                if column not in all_data:
                    all_data[column] = {}
                all_data[column][buoy_id] = filtered_data[['timestamp', column]]

    for column, data in all_data.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, (buoy_id, df) in enumerate(data.items()):
            ax.plot(df['timestamp'], df[column], label=f"Buoy {buoy_id}", color=color_palette[i])
        ax.set_ylabel(f"{column} ({units_map.get(column, 'Unknown Unit')})")
        ax.legend()
        plt.xlabel('Time')
        plt.title(f'{column} Data from Multiple Buoys')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

def main(directory, overlay):
    figures_directory = 'Figures'
    os.makedirs(figures_directory, exist_ok=True)
    pdf_path = os.path.join(figures_directory, 'buoy_data_plots.pdf')
    
    pickle_files = find_pickle_files(directory)
    with PdfPages(pdf_path) as pdf:
        if overlay:
            plot_overlayed_data(pickle_files, pdf, units_map)
        else:
            y_scales = determine_y_scales(pickle_files)
            plot_individual_data(pickle_files, pdf, y_scales)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot NDBC buoy data.")
    parser.add_argument("directory", type=str, nargs='?', default=data_dir, help="Directory containing Pickle files. Default is 'Data/'.")
    parser.add_argument("--overlay", action='store_true', help="Overlay data from all buoys on single plots.")
    args = parser.parse_args()

    main(args.directory, args.overlay)
