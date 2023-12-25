# use GWpy to grab some data and save it to a file

from gwpy.timeseries import TimeSeries, TimeSeriesDict
from gwpy.time import tconvert
import os
from timeit import default_timer as timer

import argparse

# ********** parse the input arguments 
parser = argparse.ArgumentParser(description='save some LIGO data')
parser.add_argument('-tstart','--start_time',
                    help     = 'Start Time UTC',
                    type     = str,
                    default  = '2023-12-01 00:00:00',
                    required = False)
parser.add_argument('-dur','--duration',
                    help     = 'data dur (hours)',
                    type     = float,
                    default  = 1,
                    required = False)
args = parser.parse_args()
# ******************************

data_dir = 'Data/'

if not os.path.exists(data_dir):
    os.makedirs(data_dir)


# Set the GPS time of interest
time_of_interest = args.start_time

# start the data 10 minutes before the start time of the test to get some baseline data
t0 = tconvert(time_of_interest)
#t0 = 1366807162

if __debug__:
    print("GPS start time = " + str(t0))

# Define the duration (in seconds) and the LIGO interferometer
dur = int(args.duration * 60 * 60)
ifo = 'L1'  # LIGO Obs prefix
myhost = "nds.ligo-la.caltech.edu"

# make a list of all the BLRMS channels
opts  = ['ETMX', 'ETMY', 'ITMX', 'ITMY']
dofs  = ['X', 'Y', 'Z']
bands = ['30M_100M', '100M_300M', '300M_1', '1_3', '3_10', '10_30']
channels = []

for opt in opts:
    for dof in dofs:
        for band in bands:
            chan_str = 'ISI-GND_STS_' + opt + '_' + dof + '_BLRMS_' + band + '.mean,m-trend'
            channels.append(f'{ifo}:' + chan_str)


t_start = timer()
# Fetch data and save it to an HDF5 file
if __debug__:
    print(f"Fetching data for channels: {channels}")
data = TimeSeriesDict.get(channels,
                          t0, t0 + dur, verbose=True, host=myhost)

t_elapsed = timer() - t_start
if __debug__:
    print('Elapsed time = {t:4.1f} seconds.'.format(t=t_elapsed))

# Save the data to an HDF5 file
output_file = f'{ifo}_seis_blrms_{time_of_interest.replace(":", "-").replace(" ", "_")}.hdf5'

fname = data_dir + output_file

TimeSeries.write(data, target = fname,
                 format='hdf5', overwrite=True)
print(f"Data saved to {fname}")
print(" ")

# that seems like enough for now
