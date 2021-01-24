import numpy as np
import scipy.io as scio
import os.path
import numpy.ma as ma
import matplotlib.pyplot as plt
import datetime as dt


def amie_plot(la, lo, cond_txt, start_def, stop_def):
    """
    The function amie_plot() takes as input the auroral conductance (Hall and
    Pedersen) and the field aligned currents from AMIE data files stored in this
    directory. The fields in the AMIE files have 15 latitudes and 24 MLTs with
    ghost nodes. The redistribution takes place as follows.

    Inputs:
    -------
        la          I       Latitude

        lo          I       Longitude (in terms of magnetic local time (MLT))

        cond_txt    I       Type of Conductance ('Hall' or 'Pedersen')

        start_def   I       Start Date

        stop_def    I       Stop Date

    Outputs:
    --------
        No Returned Outputs

        1 Plot produced and saved. A Coefficient File is generated after each
        execution.

    """
    ############################# Initialization ##############################

    # Print a Header to Screen to start Function
    print(('\n\n\n\n\n...........................start({} x {})...............'
           + '.................\n\n\n').format(la, lo))

    # Start Time
    t = start_def

    # Empty Arrays to be filled by data from AMIE
    fac = []
    hal = []
    ped = []

    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################

    ###########################################################################
    ############################### S T A R T #################################
    ###########################################################################

    # Print a Header to Screen to start Function
    print('\n\n\n\n\n...........................start(CMEE)...............'
          + '.................\n\n\n')

    ####################### Find out how many files ###########################

    # Start Time
    t = start_def

    # Dummy Counter Variable  to count number of days
    n_days = 0

    # The Great Loop of our times....
    while (t < stop_def):  # Until time counter hasn't reached the end time
        # Strip time into year, month and day
        t_amie = t.timetuple()
        # Increment time counter
        t = t + dt.timedelta(days=1)

        # Input .sav file from Folder
        if t_amie[0] > 2000:
            fname = ('Data/{0:4d}/{1:0=2d}/b{0:4d}{1:0=2d}{2:0=2d}n.save'
                     .format(t_amie[0], t_amie[1], t_amie[2]))
        else:
            fname = ('Data/{0:4d}/{1:0=2d}/b{0:4d}{1:0=2d}{2:0=2d}_all.save'
                     .format(t_amie[0], t_amie[1], t_amie[2]))
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
    init_year = start_def.timetuple()[0]
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
    t = start_def

    n = 0
    month = start_def.timetuple()[1]

    # The Great Loop of our times....
    while (t < stop_def):  # Until time counter hasn't reached the end time
        # Strip time into year, month and day
        t_amie = t.timetuple()
        # Increment time counter
        t = t + dt.timedelta(days=1)

        # Input .sav file from Folder
        if t_amie[0] > 2000:
            fname = ('Data/{0:4d}/{1:0=2d}/b{0:4d}{1:0=2d}{2:0=2d}n.save'
                     .format(t_amie[0], t_amie[1], t_amie[2]))
        else:
            fname = ('Data/{0:4d}/{1:0=2d}/b{0:4d}{1:0=2d}{2:0=2d}_all.save'
                     .format(t_amie[0], t_amie[1], t_amie[2]))
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

        #        fac_0 = a['data'][la, lo, f]
        #
        #        upfac_max = np.max(fac_0)
        #        dofac_max = np.min(fac_0)
        #        #print(upfac_max, dofac_max)
        #        if UPFAC_MAX <= upfac_max:
        #            UPFAC_MAX = upfac_max
        #
        #        if DOFAC_MAX <= np.abs(dofac_max):
        #            DOFAC_MAX = np.abs(dofac_max)

        # Store data from file into the dictionary

        for t_now in range(0, n_mins):
            #            for lat in range(nLat):
            #                for mlt in range(nMLT):
            lat = la
            mlt = lo
            #                   t_now = 0
            global_t = t_now + (n * n_mins)
            #                    if global_t%10 == 0:
            #                        print(n, global_t)
            data['fac'][global_t] = a['data'][lat][mlt][f][t_now]
            data['sigma_h'][global_t] = a['data'][lat][mlt][h][t_now]
            data['sigma_p'][global_t] = a['data'][lat][mlt][p][t_now]
        n += 1

    ###########################################################################
    ############################### B R E A K #################################
    ###########################################################################

    ####################### Input the Data from file ##########################

    #    # The Great Loop of our times....
    #    while(t <= stop_def): # Until time counter hasn't reached the end time
    #        # Strip time into year, month and day
    #        t_amie = t.timetuple()
    #        # Increment time counter
    #        t = t + dt.timedelta(days = 1)
    #
    #        # Input .sav file from Folder
    #        fname = ('./AMIE_Data/{0:4d}/{1:0=2d}/b{0:4d}{1:0=2d}{2:0=2d}n.save'
    #                 .format(t_amie[0], t_amie[1], t_amie[2]))
    #        print('Reading file {}...'.format(fname)) # Header
    #        if os.path.isfile(fname) == False: # If File Not present
    #            print('WARNING: File {} not present in given Directory!!!'
    #                  .format(fname))
    #            continue # Warn User and continue with loop
    #        else:
    #            a = scio.readsav(fname) # Read file
    #
    #        for i in range(0, len(a['fields'])): # The files are convoluted!
    #            if "pedersen conductance (aurora)" in str(a['fields'][i]): p = i
    #            if "hall conductance (aurora)" in str(a['fields'][i]): h = i
    #            if "field-aligned current" in str(a['fields'][i]): f = i
    #
    #        # Store data from file into temporary arrays
    #        fac_0 = a['data'][la, lo, f]
    #        hal_0 = a['data'][la, lo, h]
    #        ped_0 = a['data'][la, lo, p]
    #
    #        # Append the temporary arrays into the main arrays intialized earlier
    #        # These arrays would then be used to formulate the empirical fits
    #        fac.append(fac_0)
    #        if cond_txt == 'Hall':
    #            hal.append(hal_0) # This function is to compute Hall Conductance only
    #        elif cond_txt == 'Pedersen':
    #            hal.append(ped_0)
    #        else:
    #            print("ERROR: Input Text is Misspelled/Wrong.")

    f = data['fac']  # Convert 2D appended data into a single array
    if cond_txt == 'Hall':
        h = data['sigma_h']
    elif cond_txt == 'Pedersen':
        h = data['sigma_p']

    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################

    #    fac_max = np.max(ma.masked_invalid(f)) # Mask NaNs and infs.
    #    fac_min = np.min(ma.masked_invalid(f))
    #    fac_zero_up = np.min(ma.masked_less(f, 0)) # Upward FACs only
    #    fac_zero_do = np.max(ma.masked_greater(f, 0)) # Downward FACs only

    fac = ma.masked_invalid(f)
    hal = ma.masked_invalid(h)

    plt.plot(fac, hal, '.')
    plt.show()


if __name__ == '__main__':
    # Time is initialized globally. However, it can be changed in loop
    start_time1 = dt.datetime(2003, 1, 1, 0, 0, 0)  # DEFINED Start Time
    stop_time1 = dt.datetime(2003, 1, 31, 0, 0, 0)  # Defined Stop Time

    amie_plot(16, 0, 'Hall', start_time1, stop_time1)