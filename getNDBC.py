import requests
import h5py
import pandas as pd
import io
import os
import numpy as np

data_dir = 'Data/'

def download_ndbc_historical_data(buoy_id, year, month):
    month = f"{month:02d}"
    url = f"https://www.ndbc.noaa.gov/view_text_file.php?filename={buoy_id}h{year}.txt.gz&dir=data/historical/stdmet/"

    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve data for buoy {buoy_id} for {year}-{month}")
        return None

    content = response.content.decode('utf-8')
    return content

def parse_data_to_dict(data):
    df = pd.read_fwf(io.StringIO(data), skiprows=[1], header=0)
    selected_data = df[['#YY', 'MM', 'DD', 'hh', 'mm', 'WVHT', 'DPD', 'WTMP']]
    data_dict = selected_data.to_dict(orient='list')
    return data_dict

def save_dict_to_hdf5(data_dict, filename):
    # Ensure the directory exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    with h5py.File(os.path.join(data_dir, filename), 'w') as h5file:
        for key, value in data_dict.items():
            h5file.create_dataset(key, data=np.array(value))

# Example usage
buoy_id = '46026'  # Replace with the buoy ID you're interested in
year = 2022  # Specify the year
months = [1, 2, 3]  # Specify the months

all_data = {}

for month in months:
    data = download_ndbc_historical_data(buoy_id, year, month)
    if data:
        data_dict = parse_data_to_dict(data)
        all_data[f"{year}_{month}"] = data_dict

save_dict_to_hdf5(all_data, 'buoy_data.hdf5')
