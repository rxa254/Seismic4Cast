import requests
import pandas as pd
import argparse
import h5py
import datetime  # For adding a timestamp to the file name

# Ethical disclaimer for data usage
print("**Disclaimer:** Please use this data responsibly and ethically, respecting privacy and intellectual property rights.")

# Create an argument parser with clear descriptions and defaults
parser = argparse.ArgumentParser(description="Download NOAA buoy data and save to HDF5")
parser.add_argument("buoy_ids", nargs="+", help="List of NOAA buoy IDs to download data for")
parser.add_argument("--output_file", default=f"buoy_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.hdf5",
                    help="Output HDF5 file name (defaults to timestamped format)")

args = parser.parse_args()

# Create an HDF5 file for storing data
with h5py.File(args.output_file, "w") as f:
    for buoy_id in args.buoy_ids:
        url = f"https://www.ndbc.noaa.gov/data/realtime2/{buoy_id}.txt"  # Construct the API URL

        try:
            response = requests.get(url)  # Send a GET request to the API
            response.raise_for_status()  # Raise an exception for error responses

            data = response.text.splitlines()  # Split the text data into lines
            header = data[0].split()  # Extract the header row
            buoy_data = pd.DataFrame(data[1:], columns=header)  # Create a DataFrame

            # Create a group for each buoy's data within the HDF5 file
            buoy_group = f.create_group(buoy_id)

            # Save the DataFrame as a dataset within the group
            buoy_group.create_dataset("data", data=buoy_data.to_numpy())

            print(f"Data for buoy {buoy_id} saved to HDF5 file.")

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for buoy {buoy_id}:", e)
