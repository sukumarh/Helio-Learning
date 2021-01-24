import numpy as np
import scipy.io as scio
import os.path
import numpy.ma as ma
import datetime as dt
import pickle


def read_entire_input(directory, start_time, stop_time, mode='Hall'):

    print('Reading data files.')

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
            fname = ((directory +
                      '{0:4d}/{1:0=2d}/b{0:4d}{1:0=2d}{2:0=2d}n.save').format(t_amie[0],
                                                                              t_amie[1],
                                                                              t_amie[2]))
        else:
            fname = ((directory +
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
    n_DOY = 365  # Total number of days in a year
    if init_year % 4 != 0:
        n_DOY = 365
    else:
        n_DOY = 366

    print('Number of Days with Missing Data = {}\n'.format(n_DOY - n_days))

    print('\n Number of Days: {}\n Number of Minutes: {}\n'
          .format(n_days, n_mins))
    print(' nLatitude = {}\n nMLT = {}\n\n'.format(nLat, nMLT))

    ####################### Input the Data from file ##########################

    # Empty Arrays to be filled by data from AMIE
    data = {}
    data['fac'] = np.zeros(n_days * n_mins)
    data['sigma_h'] = np.zeros(n_days * n_mins)
    data['sigma_p'] = np.zeros(n_days * n_mins)

    # Re-initialize Start Time
    t = start_time

    n = 0
    month = start_time.timetuple()[1]

    # Preparing dataset
    # Number of Latitudes = nLat

    # Columns
    # Labels: sigma_h, sigma_p
    # Features: fac, lat, mlt

    dataset = np.zeros((n_days * n_mins, 5))


    while (t < stop_time):
        # Strip time into year, month and day
        t_amie = t.timetuple()
        # Increment time counter
        t = t + dt.timedelta(days=1)

        # Input .sav file from Folder
        if t_amie[0] > 2000:
            fname = ((directory +
                      '{0:4d}/{1:0=2d}/b{0:4d}{1:0=2d}{2:0=2d}n.save').format(t_amie[0],
                                                                              t_amie[1],
                                                                              t_amie[2]))
        else:
            fname = ((directory +
                      '{0:4d}/{1:0=2d}/b{0:4d}{1:0=2d}{2:0=2d}_all.save').format(t_amie[0],
                                                                                 t_amie[1],
                                                                                 t_amie[2]))
        print('Reading file {}...'.format(fname))  # Header
        if os.path.isfile(fname) == False:  # If File Not present
            print('WARNING: File {} not present in given Directory!!!'
                  .format(fname))
            continue  # Warn User and continue with loop
        else:
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

        # Store data from file into the dictionary
        # for t_now in range(0, n_mins):
        #     # lat = la
        #     # mlt = lo
        #     lat = 0
        #     mlt = 16
        #     global_t = t_now + (n * n_mins)
        #     data['fac'][global_t] = a['data'][lat][mlt][f][t_now]
        #     data['sigma_h'][global_t] = a['data'][lat][mlt][h][t_now]
        #     data['sigma_p'][global_t] = a['data'][lat][mlt][p][t_now]

        # Columns
        # Labels: sigma_h, sigma_p
        # Features: fac, lat, mlt
        for t_now in range(0, n_mins):
            for lat in range(nLat):
                for mlt in range(nMLT):
                    global_t = t_now + (n * n_mins)
                    dataset[global_t] = [a['data'][lat][mlt][h][t_now],
                                         a['data'][lat][mlt][p][t_now],
                                         a['data'][lat][mlt][f][t_now],
                                         a['lats'][lat],
                                         a['mlts'][mlt]]
            # lat = 0
            # mlt = 16
            # global_t = t_now + (n * n_mins)
            # data['fac'][global_t] = a['data'][lat][mlt][f][t_now]
            # data['sigma_h'][global_t] = a['data'][lat][mlt][h][t_now]
            # data['sigma_p'][global_t] = a['data'][lat][mlt][p][t_now]

        n += 1

    f = data['fac']  # Convert 2D appended data into a single array

    if mode == 'Hall':
        h = data['sigma_h']
    elif mode == 'Pedersen':
        h = data['sigma_p']

    fac_ = ma.masked_invalid(f)
    hal_ = ma.masked_invalid(h)
    fac = np.array([fac_]).T
    hal = np.array(hal_)

    X = fac
    y = hal
    X_ = np.array([i[0] for i in X])

    X_y = np.vstack((X_, y)).T
    X_y_sorted = X_y[X_y[:, 0].argsort()]

    X_1_2D = np.array([X_y_sorted.T[0]]).T
    y_1 = X_y_sorted.T[1]

    return X_1_2D, y_1


def read_single_file(la, lo, file_name, mode='Hall'):

    # Dummy Counter Variable  to count number of days
    n_days = 1

    if os.path.isfile(file_name):
        print(f'File present in the given directory.')
        a = scio.readsav(file_name)  # Read file
        n_mins = len(a['data'][0, 0, 0])

        n_fields = len(a['fields'])

        # Empty Arrays to be filled by data from AMIE
        data = {}
        data['fac'] = np.zeros(n_days * n_mins)
        data['sigma_h'] = np.zeros(n_days * n_mins)
        data['sigma_p'] = np.zeros(n_days * n_mins)

        p, h, f, n = 0, 0, 0, 0

        for i in range(0, n_fields):  # The files are convoluted!
            if "pedersen conductance (aurora)" in str(a['fields'][i]):
                p = i
            if "hall conductance (aurora)" in str(a['fields'][i]):
                h = i
            if "field-aligned current" in str(a['fields'][i]):
                f = i

        # Store data from file into the dictionary
        lat = la
        mlt = lo
        for t_now in range(0, n_mins):
            global_t = t_now + (n * n_mins)
            data['fac'][global_t] = a['data'][lat][mlt][f][t_now]
            data['sigma_h'][global_t] = a['data'][lat][mlt][h][t_now]
            data['sigma_p'][global_t] = a['data'][lat][mlt][p][t_now]

        f = data['fac']  # Convert 2D appended data into a single array
        if mode == 'Hall':
            h = data['sigma_h']
        elif mode == 'Pedersen':
            h = data['sigma_p']

        fac_ = ma.masked_invalid(f)
        hal_ = ma.masked_invalid(h)
        fac = np.array([fac_]).T
        hal = np.array(hal_)

        return fac, hal

    else:
        print(f'Warning: File not present in the given directory.')
        return [], []


def read_multi_files(la, lo, directory, start_time, stop_time, mode='Hall'):

    # Print a Header to Screen to start Function
    print(('\n\n\n\n\n...........................start({} x {})...............'
           + '.................\n\n\n').format(la, lo))

    # Print a Header to Screen to start Function
    print('\n\n\n\n\n...........................start(CMEE)...............'
          + '.................\n\n\n')

    ####################### Find out how many files ###########################

    # Start Time
    t = start_time

    # Dummy Counter Variable  to count number of days
    n_days = 0

    # The Great Loop of our times....
    while (t < stop_time):  # Until time counter hasn't reached the end time
        # Strip time into year, month and day
        t_amie = t.timetuple()
        # Increment time counter
        t = t + dt.timedelta(days=1)

        # Input .sav file from Folder
        if t_amie[0] > 2000:
            fname = ((directory +
                      '{0:4d}/{1:0=2d}/b{0:4d}{1:0=2d}{2:0=2d}n.save').format(t_amie[0],
                                                                              t_amie[1],
                                                                              t_amie[2]))
        else:
            fname = ((directory +
                      '{0:4d}/{1:0=2d}/b{0:4d}{1:0=2d}{2:0=2d}_all.save').format(t_amie[0],
                                                                                 t_amie[1],
                                                                                 t_amie[2]))
        print('Counting file {}...'.format(fname))  # Header
        if os.path.isfile(fname) == False:  # If File Not present
            print('\tWARNING: File {} not present in given Directory!!!'
                  .format(fname))
            continue  # Warn User and continue with loop
        else:
            n_days += 1
    a = scio.readsav(fname)  # Read file
    nLat = len(a['data'][:, 0])
    nMLT = len(a['data'][0, :])
    n_mins = len(a['data'][0, 0, 0])

    # Leap Year or Not
    init_year = start_time.timetuple()[0]
    n_DOY = 365  # Total number of days in a year
    if init_year % 4 != 0:
        n_DOY = 365
    else:
        n_DOY = 366

    print('Number of Days with Missing Data = {}\n'.format(n_DOY - n_days))

    print('\n Number of Days: {}\n Number of Minutes: {}\n'
          .format(n_days, n_mins))
    print(' nLatitude = {}\n nMLT = {}\n\n'.format(nLat, nMLT))

    ####################### Input the Data from file ##########################

    # Empty Arrays to be filled by data from AMIE
    data = {}
    data['fac'] = np.zeros(n_days * n_mins)
    data['sigma_h'] = np.zeros(n_days * n_mins)
    data['sigma_p'] = np.zeros(n_days * n_mins)

    # Re-initialize Start Time
    t = start_time

    n = 0
    month = start_time.timetuple()[1]

    # The Great Loop of our times....
    while (t < stop_time):  # Until time counter hasn't reached the end time
        # Strip time into year, month and day
        t_amie = t.timetuple()
        # Increment time counter
        t = t + dt.timedelta(days=1)

        # Input .sav file from Folder
        if t_amie[0] > 2000:
            fname = ((directory +
                      '{0:4d}/{1:0=2d}/b{0:4d}{1:0=2d}{2:0=2d}n.save').format(t_amie[0],
                                                                              t_amie[1],
                                                                              t_amie[2]))
        else:
            fname = ((directory +
                      '{0:4d}/{1:0=2d}/b{0:4d}{1:0=2d}{2:0=2d}_all.save').format(t_amie[0],
                                                                                 t_amie[1],
                                                                                 t_amie[2]))
        print('Reading file {}...'.format(fname))  # Header
        if os.path.isfile(fname) == False:  # If File Not present
            print('WARNING: File {} not present in given Directory!!!'
                  .format(fname))
            continue  # Warn User and continue with loop
        else:
            a = scio.readsav(fname)  # Read file

        for i in range(0, len(a['fields'])):  # The files are convoluted!
            if "pedersen conductance (aurora)" in str(a['fields'][i]): p = i
            if "hall conductance (aurora)" in str(a['fields'][i]): h = i
            if "field-aligned current" in str(a['fields'][i]): f = i

        # Store data from file into the dictionary
        for t_now in range(0, n_mins):
            lat = la
            mlt = lo
            global_t = t_now + (n * n_mins)
            data['fac'][global_t] = a['data'][lat][mlt][f][t_now]
            data['sigma_h'][global_t] = a['data'][lat][mlt][h][t_now]
            data['sigma_p'][global_t] = a['data'][lat][mlt][p][t_now]
        n += 1

    f = data['fac']  # Convert 2D appended data into a single array

    if mode == 'Hall':
        h = data['sigma_h']
    elif mode == 'Pedersen':
        h = data['sigma_p']

    fac_ = ma.masked_invalid(f)
    hal_ = ma.masked_invalid(h)
    fac = np.array([fac_]).T
    hal = np.array(hal_)

    X = fac
    y = hal
    X_ = np.array([i[0] for i in X])

    X_y = np.vstack((X_, y)).T
    X_y_sorted = X_y[X_y[:, 0].argsort()]

    X_1_2D = np.array([X_y_sorted.T[0]]).T
    y_1 = X_y_sorted.T[1]

    return X_1_2D, y_1


def save_model(file_name, data):
    with open(file_name, "wb") as fp:
        pickle.dump(data, fp)


def read_model(file_name):
    if os.path.isfile(file_name):
        with open(file_name, "rb") as fp:
            data = pickle.load(fp)
            return data, True
    else:
        print('Model\'s pickle file not found')
        return [], False


def save_data(file_name, data):
    with open(file_name, "wb") as fp:
        pickle.dump(data, fp)


def read_data(file_name):
    if os.path.isfile(file_name):
        with open(file_name, "rb") as fp:
            data = pickle.load(fp)
            return data, True
    else:
        print('Data\'s pickle file not found')
        return [], False


def read_file_old(la, lo, start_time, stop_time, file_name, mode='Hall'):
    # Start Time
    t = start_time

    # Empty Arrays to be filled by data from AMIE
    fac_ = []
    hal_ = []
    ped = []

    # Dummy Counter Variable  to count number of days
    n_days = 1

    if os.path.isfile(file_name):
        print(f'File present in the given directory.')
        a = scio.readsav(file_name)  # Read file
        nLat = len(a['data'][:, 0])
        nMLT = len(a['data'][0, :])
        n_mins = len(a['data'][0, 0, 0])

        n_fields = len(a['fields'])

        # Empty Arrays to be filled by data from AMIE
        data = {}
        data['fac'] = np.zeros(n_days * n_mins)
        data['sigma_h'] = np.zeros(n_days * n_mins)
        data['sigma_p'] = np.zeros(n_days * n_mins)

        # n = 0
        p, h, f, n = 0, 0, 0, 0

        for i in range(0, n_fields):  # The files are convoluted!
            if "pedersen conductance (aurora)" in str(a['fields'][i]):
                p = i
            if "hall conductance (aurora)" in str(a['fields'][i]):
                h = i
            if "field-aligned current" in str(a['fields'][i]):
                f = i

        # Store data from file into the dictionary
        lat = la
        mlt = lo
        for t_now in range(0, n_mins):
            global_t = t_now + (n * n_mins)
            data['fac'][global_t] = a['data'][lat][mlt][f][t_now]
            data['sigma_h'][global_t] = a['data'][lat][mlt][h][t_now]
            data['sigma_p'][global_t] = a['data'][lat][mlt][p][t_now]

        f = data['fac']  # Convert 2D appended data into a single array
        if mode == 'Hall':
            h = data['sigma_h']
        elif mode == 'Pedersen':
            h = data['sigma_p']

        fac_ = ma.masked_invalid(f)
        hal_ = ma.masked_invalid(h)
        fac = np.array([fac_]).T
        hal = np.array(hal_)

        return fac, hal

    else:
        print(f'Warning: File not present in the given directory.')
        return [], []
