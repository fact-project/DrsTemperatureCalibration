import pandas as pd
import numpy as np
import h5py
import sys
import os

from astropy.io import fits

import logging


###################################################################################################
def initDrsDataBase(storeFilename_):
    if(os.path.isfile(storeFilename)):
        user_input = input("File ’"+str(storeFilename_)+"’ allready exist." +
                           " Type ’y’ to overwrite file\nYour input: ")
        if(user_input != 'y'):
            sys.exit()

    logging.basicConfig(filename=storeFilename_.split('.')[0]+'.log', filemode='w',
                        format='%(levelname)s:%(message)s', level=logging.DEBUG)

    with h5py.File(storeFilename_, 'w') as hf:
        hf.create_dataset('time_baseline', (0, 1), maxshape=(None, 1),
                          compression="gzip", fletcher32=True)
        hf.create_dataset('temp_baseline', (0, 160), maxshape=(None, 160),
                          compression="gzip", fletcher32=True)
        hf.create_dataset('tempStd_baseline', (0, 160), maxshape=(None, 160),
                          compression="gzip", fletcher32=True)
        hf.create_dataset('baselineMean', (0, 1024*1440), maxshape=(None, 1024*1440),
                          compression="gzip", fletcher32=True)
        hf.create_dataset('baselineMeanStd', (0, 1024*1440), maxshape=(None, 1024*1440),
                          compression="gzip", fletcher32=True)

        hf.create_dataset('time_gain', (0, 1), maxshape=(None, 1),
                          compression="gzip", fletcher32=True)
        hf.create_dataset('temp_gain', (0, 160), maxshape=(None, 160),
                          compression="gzip", fletcher32=True)
        hf.create_dataset('tempStd_gain', (0, 160), maxshape=(None, 160),
                          compression="gzip", fletcher32=True)
        hf.create_dataset('gainMean', (0, 1024*1440), maxshape=(None, 1024*1440),
                          compression="gzip", fletcher32=True)
        hf.create_dataset('gainMeanStd', (0, 1024*1440), maxshape=(None, 1024*1440),
                          compression="gzip", fletcher32=True)


###################################################################################################
def saveTupleOfAttribute(tempFilename, drsFilename, storeFilename):

    loggingFlag = True
    errorFlag = False
    try:
        tab_temp = fits.open(tempFilename, ignoremissing=True, ignore_missing_end=True)
        tab_drs = fits.open(drsFilename, ignoremissing=True, ignore_missing_end=True)

    except Exception as errInfos:
        errorFlag = True
        if(loggingFlag):
            errorStr = " LoadingError: in'"+drsFilename+"' or '"+tempFilename+"' ("+str(errInfos)+")"
            # print(errorStr)
            logging.critical(errorStr)
        return False

    tab_temp_time = None
    tab_temp_temp = None
    try:
        tab_temp_time = tab_temp[1].data['Time']
        tab_temp_temp = tab_temp[1].data['temp']

    except Exception as errInfos:
        errorFlag = True
        if(loggingFlag):
            errorStr = " In File '"+tempFilename+"': "+str(errInfos)
            # print(errorStr)
            logging.error(errorStr)

    if(tab_temp_temp is not None and tab_temp_temp.shape[1] != 160):
        errorFlag = True
        if(loggingFlag):
            errorStr = ("File not used: Just "+str(tab_temp_temp.shape[1]) +
                        " Temperature Values in File '"+tempFilename+"'")
            # print(errorStr)
            logging.error(errorStr)
        return True  # mark day as handeled (there is no Temperature data for all drsFiles of this day)

    beg_run_0 = None
    end_run_0 = None
    try:
        beg_run_0 = pd.to_datetime(tab_drs[1].header['RUN0-BEG'])

    except Exception as errInfos:
        errorFlag = True
        if(loggingFlag):
            errorStr = " In File '"+drsFilename+"': "+str(errInfos)
            # print(errorStr)
            logging.error(errorStr)

    try:
        end_run_0 = pd.to_datetime(tab_drs[1].header['RUN0-END'])

    except Exception as errInfos:
        errorFlag = True
        if(loggingFlag):
            errorStr = " In File '"+drsFilename+"': "+str(errInfos)
            # print(errorStr)
            logging.error(errorStr)

    if(errorFlag is False):
        baselineMean = tab_drs[1].data['BaselineMean'][0]
        baselineMeanStd = tab_drs[1].data['BaselineRms'][0]

        baselineMeanNulls = list(np.array(np.where(baselineMean == 0.))[0])
        if (len(baselineMeanNulls) != 0.):
            errorFlag = True
            if(loggingFlag):
                errorStr = (" File not used: Nulls of baselineMean in File '"+str(drsFilename) +
                            "' Nulls at Index:\n"+str(baselineMeanNulls))
                # print(errorStr)
                logging.error(errorStr)

        baselineMeanStdNulls = list(np.array(np.where(baselineMeanStd == 0.))[0])
        if (len(baselineMeanStdNulls) != 0.):
            errorFlag = True
            if(loggingFlag):
                errorStr = (" File not used: Nulls of baselineMeanStd in File '"+str(drsFilename) +
                            "' Nulls at Index:\n"+str(baselineMeanStdNulls))
                # print(errorStr)
                logging.error(errorStr)

    if(errorFlag is False):
        timeValuesRun_0 = []
        tempValuesRun_0 = []
        time_indexRun_0 = 0
        while (pd.to_datetime(tab_temp_time[time_indexRun_0] * 24 * 3600 * 1e9) < beg_run_0):
            time_indexRun_0 = time_indexRun_0+1

        while (pd.to_datetime(tab_temp_time[time_indexRun_0] * 24 * 3600 * 1e9) < end_run_0):
            timeValuesRun_0.append(tab_temp_time[time_indexRun_0])
            tempValuesRun_0.append(tab_temp_temp[time_indexRun_0])
            time_indexRun_0 = time_indexRun_0+1

        timeValuesRun_0 = np.array(timeValuesRun_0)
        tempValuesRun_0 = np.array(tempValuesRun_0)

        if(timeValuesRun_0.shape[0] > 1):
            time_baseline = np.mean(timeValuesRun_0, dtype='float64')
        else:
            time_baseline = timeValuesRun_0

        if(tempValuesRun_0.shape[0] > 1):
            temp_baseline = np.mean(tempValuesRun_0, dtype='float64', axis=0)
            tempStd_baseline = np.std(tempValuesRun_0, dtype='float64', axis=0, ddof=1)
        else:
            temp_baseline = tempValuesRun_0
            tempStd_baseline = np.zeros(tempValuesRun_0.shape[1])

    beg_run_1 = None
    end_run_1 = None
    try:
        beg_run_1 = pd.to_datetime(tab_drs[1].header['RUN1-BEG'])

    except Exception as errInfos:
        errorFlag = True
        if(loggingFlag):
            errorStr = " In File '"+drsFilename+"': "+str(errInfos)
            # print(errorStr)
            logging.error(errorStr)

    try:
        end_run_1 = pd.to_datetime(tab_drs[1].header['RUN1-END'])

    except Exception as errInfos:
        errorFlag = True
        if(loggingFlag):
            errorStr = " In File '"+drsFilename+"': "+str(errInfos)
            # print(errorStr)
            logging.error(errorStr)

    if(errorFlag is False):
        gainMean = tab_drs[1].data['GainMean'][0]
        gainMeanStd = tab_drs[1].data['GainRms'][0]

        gainMeanNulls = list(np.array(np.where(gainMean == 0.))[0])
        if (len(gainMeanNulls) != 0.):
            errorFlag = True
            if(loggingFlag):
                errorStr = (" File not used: Nulls of gainMean in File '"+str(drsFilename) +
                            "' Nulls at Index:\n"+str(gainMeanNulls))
                # print(errorStr)
                logging.error(errorStr)

        gainMeanStdNulls = list(np.array(np.where(gainMeanStd == 0.))[0])
        if (len(gainMeanStdNulls) != 0.):
            errorFlag = True
            if(loggingFlag):
                errorStr = (" File not used: Nulls of gainMeanStd in File '"+str(drsFilename) +
                            "' Nulls at Index:\n"+str(gainMeanStdNulls))
                # print(errorStr)
                logging.error(errorStr)

    if(errorFlag is False):
        timeValuesRun_1 = []
        tempValuesRun_1 = []
        time_indexRun_1 = 0
        while (pd.to_datetime(tab_temp_time[time_indexRun_1] * 24 * 3600 * 1e9) < beg_run_1):
            time_indexRun_1 = time_indexRun_1+1

        while (pd.to_datetime(tab_temp_time[time_indexRun_1] * 24 * 3600 * 1e9) < end_run_1):
            timeValuesRun_1.append(tab_temp_time[time_indexRun_1])
            tempValuesRun_1.append(tab_temp_temp[time_indexRun_1])
            time_indexRun_1 = time_indexRun_1+1

        timeValuesRun_1 = np.array(timeValuesRun_1)
        tempValuesRun_1 = np.array(tempValuesRun_1)

        if(timeValuesRun_1.shape[0] > 1):
            time_gain = np.mean(timeValuesRun_1, dtype='float64')
        else:
            time_gain = timeValuesRun_1

        if(tempValuesRun_1.shape[0] > 1):
            temp_gain = np.mean(tempValuesRun_1, dtype='float64', axis=0)
            tempStd_gain = np.std(tempValuesRun_1, dtype='float64', axis=0, ddof=1)
        else:
            temp_gain = tempValuesRun_1
            tempStd_gain = np.zeros(tempValuesRun_1.shape[1])

    with h5py.File(storeFilename) as store:
        if(errorFlag is False):
            data = store['time_baseline']
            data.resize((len(data)+1, data.maxshape[1]))
            data[len(data)-1, :] = time_baseline

            data = store['temp_baseline']
            data.resize((len(data)+1, data.maxshape[1]))
            data[len(data)-1, :] = temp_baseline

            data = store['tempStd_baseline']
            data.resize((len(data)+1, data.maxshape[1]))
            data[len(data)-1, :] = tempStd_baseline

            data = store['baselineMean']
            data.resize((len(data)+1, data.maxshape[1]))
            data[len(data)-1, :] = baselineMean

            data = store['baselineMeanStd']
            data.resize((len(data)+1, data.maxshape[1]))
            data[len(data)-1, :] = baselineMeanStd

            data = store['time_gain']
            data.resize((len(data)+1, data.maxshape[1]))
            data[len(data)-1, :] = time_gain

            data = store['temp_gain']
            data.resize((len(data)+1, data.maxshape[1]))
            data[len(data)-1, :] = temp_gain

            data = store['tempStd_gain']
            data.resize((len(data)+1, data.maxshape[1]))
            data[len(data)-1, :] = tempStd_gain

            data = store['gainMean']
            data.resize((len(data)+1, data.maxshape[1]))
            data[len(data)-1, :] = gainMean

            data = store['gainMeanStd']
            data.resize((len(data)+1, data.maxshape[1]))
            data[len(data)-1, :] = gainMeanStd

            return True
        else:
            return False


###################################################################################################

storeFilename = '/gpfs/scratch/schulz/drsData.h5'
drsFilname = 'drsFiles.txt'
initDrsDataBase(storeFilename)

previousDates = []

count = 0
countMax = sum(1 for line in open(drsFilname))
with open(drsFilname) as f:
    print("Check '", countMax, "' drsFiles'")
    for drsFilename in f:
        drsFilename = drsFilename.strip('\n')
        count = count + 1

        if(((count/countMax*100) % 1) < (((count-1)/countMax*100) % 1) and
           ((count/countMax*100) % 1) < (((count+1)/countMax*100) % 1)):
            print(count, ": ", int(count/countMax*100), '%')

        tempFilename = (str("/fact/aux") +
                        str(drsFilename.split('_')[0].split('raw')[-1]) +
                        str(".FAD_CONTROL_TEMPERATURE.fits"))

        date = drsFilename.split('_')[0].split('/')[-1]

        if(date not in previousDates and os.path.isfile(drsFilename) and os.path.isfile(tempFilename)):
            if(saveTupleOfAttribute(tempFilename, drsFilename, storeFilename)):
                previousDates.append(date)
