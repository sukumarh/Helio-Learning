import numpy as np
import scipy.io as scio
import os.path
import numpy.ma as ma
import datetime as dt
import pickle

from Helper_Functions.File_Ops import save_data


def process_data(in_directory, out_directory, start_time, stop_time):

    print('Processing data files.')

    # Track number of days of data
    n_days = 0

    # Checking whether all files exist from start time to stop time.
    t = start_time

    while (t < stop_time):

        # Strip time into year, month and day
        t_amie = t.timetuple()

        # Increment time counter
        t = t + dt.timedelta(days=1)

        # Compute .sav file name
        if t_amie[0] > 2000:
            fname = ((in_directory +
                      '{0:4d}/{1:0=2d}/b{0:4d}{1:0=2d}{2:0=2d}n.save').format(t_amie[0],
                                                                              t_amie[1],
                                                                              t_amie[2]))
        else:
            fname = ((in_directory +
                      '{0:4d}/{1:0=2d}/b{0:4d}{1:0=2d}{2:0=2d}_all.save').format(t_amie[0],
                                                                                 t_amie[1],
                                                                                 t_amie[2]))

        if os.path.isfile(fname):
            n_days += 1
        else:
            print(f'WARNING: File {fname} not present in given directory.')

    a = scio.readsav(fname)  # Read any file to find the number of coordinates present
    nLat = len(a['data'][:, 0])
    nMLT = len(a['data'][0, :])
    n_mins = len(a['data'][0, 0, 0])

    # Leap Year or Not
    init_year = start_time.timetuple()[0]

    # Total number of days in a year
    n_DOY = 365
    if init_year % 4 != 0:
        n_DOY = 365
    else:
        n_DOY = 366

    print(f'Number of Days with Missing Data = {(n_DOY - n_days)}')
    print(f'Number of Days: {n_days}')
    print(f'Number of Minutes: {n_mins}')
    print(f'nLatitude = {nLat}; nMLT = {nMLT}')

    # Empty dataset to be filled by data from AMIE
    # Columns
    # Labels: sigma_h, sigma_p
    # Features: fac, lat, mlt
    dataset = np.zeros((n_days * n_mins * nLat * nMLT, 5))
    # dataset = np.array([])

    # Re-initialize Start Time
    t = start_time
    n = 0

    row_num = -1

    while t < stop_time:
        # Strip time into year, month and day
        t_amie = t.timetuple()
        # Increment time counter
        t = t + dt.timedelta(days=1)

        # Input .sav file from Folder
        if t_amie[0] > 2000:
            fname = ((in_directory +
                      '{0:4d}/{1:0=2d}/b{0:4d}{1:0=2d}{2:0=2d}n.save').format(t_amie[0],
                                                                              t_amie[1],
                                                                              t_amie[2]))
        else:
            fname = ((in_directory +
                      '{0:4d}/{1:0=2d}/b{0:4d}{1:0=2d}{2:0=2d}_all.save').format(t_amie[0],
                                                                                 t_amie[1],
                                                                                 t_amie[2]))
        print(f'Reading file {fname}')
        if os.path.isfile(fname):
            a = scio.readsav(fname)  # Read file

            # Finding indices of various features
            Fields = [i.decode("utf-8").strip() for i in a['fields']]
            for ix, x in enumerate(Fields):
                if 'pedersen conductance (aurora)' in x:
                    p = ix
                if 'hall conductance (aurora)' in x:
                    h = ix
                if 'field-aligned current' in x:
                    f = ix

            # Columns
            # Labels: sigma_h, sigma_p
            # Features: fac, lat, mlt
            for t_now in range(0, n_mins):
                for lat in range(nLat):
                    for mlt in range(nMLT):
                        # global_t = t_now + (n * n_mins) + lat
                        row_num += 1
                        dataset[row_num] = [a['data'][lat][mlt][h][t_now],
                                             a['data'][lat][mlt][p][t_now],
                                             a['data'][lat][mlt][f][t_now],
                                             a['lats'][lat],
                                             a['mlts'][mlt]]
                        # if dataset.shape[0] == 0:
                        #     dataset = np.array([[a['data'][lat][mlt][h][t_now],
                        #                          a['data'][lat][mlt][p][t_now],
                        #                          a['data'][lat][mlt][f][t_now],
                        #                          a['lats'][lat],
                        #                          a['mlts'][mlt]]])
                        #     continue
                        #
                        # dataset = np.append(dataset, [[a['data'][lat][mlt][h][t_now],
                        #                                a['data'][lat][mlt][p][t_now],
                        #                                a['data'][lat][mlt][f][t_now],
                        #                                a['lats'][lat],
                        #                                a['mlts'][mlt]]],
                        #                     axis=0)

        # n += 1

    data_sorted = dataset[dataset[:, 2].argsort()]
    save_data(out_directory, data_sorted)


def main():
    input_directory = 'Data/'
    processed_data_filename = 'Processed_Data/2004/01_01.p'

    start_time = dt.datetime(2004, 1, 1, 0, 0, 0)
    stop_time = dt.datetime(2004, 1, 2, 0, 0, 0)

    process_data(input_directory, processed_data_filename, start_time, stop_time)

if __name__ == '__main__':
    main()
