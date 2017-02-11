import pandas as pd
import numpy as np
import h5py
import sys
import os
import os.path

import logging

from fact.credentials import create_factdb_engine
from astropy.io import fits

####################################################################################################
####################################################################################################
# path to your local mounted isdc-in04-Folder (isdc-in04:/fact)
isdcPath = "/home/florian/Dokumente/Uni/Master/Masterarbeit/isdcFact/"
####################################################################################################
####################################################################################################


####################################################################################################
def filename(row):
    return os.path.join(
        "/fact/raw",
        str(row.date.year),
        "{:02d}".format(row.date.month),
        "{:02d}".format(row.date.day),
        "{}_{:03d}.drs.fits.gz".format(row.fNight, row.fRunID),
    )


####################################################################################################
def searchDrsFiles(storeFilename_="../data/drsFiles_.txt"):
    factdb = create_factdb_engine()

    drsinfos = pd.read_sql("RunInfo", factdb, columns=["fNight", "fRunID", "fRunTypeKey", "fDrsStep", "fNumEvents"])
    drsfiles = drsinfos.query("fDrsStep == 2 & fRunTypeKey == 2 & fNumEvents == 1000")
    # fNumEvents == 1000 prevent for unfinished/broken files
    drsfiles["date"] = pd.to_datetime(drsfiles.fNight.astype(str), format="%Y%m%d")

    drsfiles["filename"] = drsfiles.apply(filename, axis=1)
    drsfiles[["filename"]].to_csv(storeFilename_, index=False, header=False)


####################################################################################################
def filterDrsFiles(sourceFilname_="../data/drsFiles_.txt",
                   storeFilename_="../data/selectedDrsFiles2016.txt"):
    selectedFiles = []
    previousDates = []
    with open(sourceFilname_) as f:
        for drsFilename in f:
            drsFilename = drsFilename.strip("\n")

            date = drsFilename.split("_")[0].split("/")[-1]
            if(date not in previousDates and int(drsFilename.split("_")[0].split("/")[-4]) == 2016):

                previousDates.append(date)
                selectedFiles.append(drsFilename)

    pd.DataFrame(selectedFiles).to_csv(storeFilename_, index=False, header=False)


####################################################################################################
####################################################################################################
# ##############                         saveDrsAttributes                          ############## #
####################################################################################################
####################################################################################################

def saveDrsAttributes(drsFilname_="../data/drsFiles.txt",
                      storeFilename_="../data/drsData.h5"):

    if(os.path.isfile(storeFilename_)):
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

    previousDates = []

    count = 0
    countMax = sum(1 for line in open(drsFilname_))
    with open(drsFilname_) as f:
        print("Check '", countMax, "' drsFiles'")
        for drsFilename in f:
            drsFilename = drsFilename.strip('\n')
            count = count + 1

            if(((count/countMax*100) % 1) < (((count-1)/countMax*100) % 1) and
               ((count/countMax*100) % 1) < (((count+1)/countMax*100) % 1)):
                print(count, ": ", int(count/countMax*100), '%')

            tempFilename = (str(isdcPath+"aux") +
                            str(drsFilename.split('_')[0].split('raw')[-1]) +
                            str(".FAD_CONTROL_TEMPERATURE.fits"))

            date = drsFilename.split('_')[0].split('/')[-1]

            if(date not in previousDates and os.path.isfile(drsFilename) and os.path.isfile(tempFilename)):
                if(saveTupleOfAttribute(tempFilename, drsFilename, storeFilename_)):
                    previousDates.append(date)


####################################################################################################
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


####################################################################################################
####################################################################################################
# ##############                           saveFitValues                            ############## #
####################################################################################################
####################################################################################################

def saveFitValues(sourceFilname_="../data/drsData.h5",
                  storeFilename_="../data/fitValuesData.fits"):

    if(os.path.isfile(storeFilename_)):
        user_input = input("File ’"+str(storeFilename_)+"’ allready exist. Type ’y’ to overwrite file\n" +
                           "Your input: ")
        if(user_input != 'y'):
            sys.exit()

    logging.basicConfig(filename=storeFilename_.split('.')[0]+'.log', filemode='w',
                        format='%(levelname)s:%(message)s', level=logging.DEBUG)

    cutOffErrorFactor_baseline = 2
    cutOffErrorFactor_gain = 2

    baselineMeanSlope = []
    baselineMeanSlopeStd = []
    baselineMeanOffset = []
    baselineMeanOffsetStd = []
    gainMeanSlope = []
    gainMeanSlopeStd = []
    gainMeanOffset = []
    gainMeanOffsetStd = []

    for pixelNr in range(10):

        if(((pixelNr/1440*100) % 1) < (((pixelNr-1)/1440*100) % 1) and
           ((pixelNr/1440*100) % 1) < (((pixelNr+1)/1440*100) % 1)):
            print(pixelNr, ": ", int(pixelNr/1440*100), '%')

        with h5py.File(sourceFilname_, 'r') as dataSource:

            temp_baseline = np.array(dataSource['temp_baseline'][:, int(pixelNr/9)])
            baselineMean = np.array(dataSource['baselineMean'][:, pixelNr*1024: (pixelNr+1)*1024])
            baselineMeanStd = np.array(dataSource['baselineMeanStd'][:, pixelNr*1024: (pixelNr+1)*1024])

            temp_gain = np.array(dataSource['temp_gain'][:, int(pixelNr/9)])
            gainMean = np.array(dataSource['gainMean'][:, pixelNr*1024: (pixelNr+1)*1024])
            gainMeanStd = np.array(dataSource['gainMeanStd'][:, pixelNr*1024: (pixelNr+1)*1024])

        for condNr in range(1024):

            baselineMean_cond = baselineMean[:, condNr]
            baselineMeanStd_cond = baselineMeanStd[:, condNr]
            baselineMeanStd_cond_mean = np.mean(baselineMeanStd_cond, dtype='float')
            gainMean_cond = gainMean[:, condNr]
            gainMeanStd_cond = gainMeanStd[:, condNr]
            gainMeanStd_cond_mean = np.mean(gainMeanStd_cond, dtype='float')

            try:
                indices = [baselineMeanStd_cond < baselineMeanStd_cond_mean*cutOffErrorFactor_baseline]
                varBase, covBase = getLinearFitValues(temp_baseline[indices],
                                                      baselineMean_cond[indices],
                                                      baselineMeanStd_cond[indices])

                baselineMeanSlope.append(varBase[0])
                baselineMeanSlopeStd.append(np.sqrt(covBase[0][0]))
                baselineMeanOffset.append(varBase[1])
                baselineMeanOffsetStd.append(np.sqrt(covBase[1][1]))

            except Exception as errInfos:
                errorStr = ("Gain-Fit(PixelNr: "+str('{:04d}'.format(pixelNr)) +
                            ", condNr: "+str('{:04d}'.format(condNr)) +
                            ") with Slope="+str(varBase[0]) +
                            " and Offset='"+str(varBase[1])+": '"+str(errInfos)+"'")
                logging.warning(errorStr)

                baselineMeanSlope.append(varBase[0])
                baselineMeanSlopeStd.append(0)
                baselineMeanOffset.append(varBase[1])
                baselineMeanOffsetStd.append(0)

            try:
                indices = [gainMeanStd_cond < gainMeanStd_cond_mean*cutOffErrorFactor_gain]
                varGain, covGain = getLinearFitValues(temp_gain[indices],
                                                      gainMean_cond[indices],
                                                      gainMeanStd_cond[indices])

                gainMeanSlope.append(varGain[0])
                gainMeanSlopeStd.append(np.sqrt(covGain[0][0]))
                gainMeanOffset.append(varGain[1])
                gainMeanOffsetStd.append(np.sqrt(covGain[1][1]))

            except Exception as errInfos:
                errorStr = ("Gain-Fit(PixelNr: "+str('{:04d}'.format(pixelNr)) +
                            ", condNr: "+str('{:04d}'.format(condNr)) +
                            ") with Slope="+str(varGain[0]) +
                            " and Offset='"+str(varGain[1])+": '"+str(errInfos)+"'")
                logging.warning(errorStr)

                gainMeanSlope.append(varGain[0])
                gainMeanSlopeStd.append(0)
                gainMeanOffset.append(varGain[1])
                gainMeanOffsetStd.append(0)

    baselineMeanSlope = np.array([baselineMeanSlope])
    baselineMeanSlopeStd = np.array([baselineMeanSlopeStd])
    baselineMeanOffset = np.array([baselineMeanOffset])
    baselineMeanOffsetStd = np.array([baselineMeanOffsetStd])
    gainMeanSlope = np.array([gainMeanSlope])
    gainMeanSlopeStd = np.array([gainMeanSlopeStd])
    gainMeanOffset = np.array([gainMeanOffset])
    gainMeanOffsetStd = np.array([gainMeanOffsetStd])

    print("Write Data to Table")
    tbhdu_baseline = fits.BinTableHDU.from_columns(
            [fits.Column(name='slope',    format=str(baselineMeanSlope.shape[1])+'E',
                         unit='mV/celsius',  array=baselineMeanSlope),
             fits.Column(name='slopeStd', format=str(baselineMeanSlopeStd.shape[1])+'E',
                         unit='mV/celsius',  array=baselineMeanSlopeStd),
             fits.Column(name='offset',      format=str(baselineMeanOffset.shape[1])+'E',
                         unit='mV',          array=baselineMeanOffset),
             fits.Column(name='offsetStd',   format=str(baselineMeanOffsetStd.shape[1])+'E',
                         unit='mV',          array=baselineMeanOffsetStd)])
    tbhdu_baseline.header.insert("TFIELDS", ("EXTNAME", "Baseline"), after=True)
    commentStr = ("All baseline-values with a bigger error than the 'CutOff-ErrorFactor'" +
                  " multiplied with the mean of the error from all collected baseline-values" +
                  " will not used for the fit")
    tbhdu_baseline.header.insert("EXTNAME", ("comment", commentStr), after="True")
    tbhdu_baseline.header.insert("comment", ("CutOff", str(cutOffErrorFactor_gain), "Shortform of CutOffErrorFactor"),
                                 after=True)

    tbhdu_gain = fits.BinTableHDU.from_columns(
            [fits.Column(name='slope',    format=str(gainMeanSlope.shape[1])+'E',
                         unit='mV/celsius',  array=gainMeanSlope),
             fits.Column(name='slopeStd', format=str(gainMeanSlopeStd.shape[1])+'E',
                         unit='mV/celsius',  array=gainMeanSlopeStd),
             fits.Column(name='offset',      format=str(gainMeanOffset.shape[1])+'E',
                         unit='mV',          array=gainMeanOffset),
             fits.Column(name='offsetStd',   format=str(gainMeanOffsetStd.shape[1])+'E',
                         unit='mV',          array=gainMeanOffsetStd)])
    tbhdu_gain.header.insert("TFIELDS", ("EXTNAME", "Gain"), after=True)
    commentStr = ("All gain-values with a bigger error than the 'CutOffErrorFactor'" +
                  " multiplied with the mean of the error from all collected gain-values" +
                  " will not used for the fit")
    tbhdu_gain.header.insert("EXTNAME", ("comment", commentStr), after=True)
    tbhdu_gain.header.insert("comment", ("CutOff", str(cutOffErrorFactor_gain), "Shortform of CutOffErrorFactor"),
                             after=True)
    print("Save Table")
    thdulist = fits.HDUList([fits.PrimaryHDU(), tbhdu_baseline, tbhdu_gain])
    thdulist.writeto(storeFilename_, clobber=True, checksum=True)  # clobber/overwrite=True
    print("Verify Checksum")
    # Open the File verifying the checksum values for all HDUs
    try:
        hdul = fits.open(storeFilename_, checksum=True)
        print(hdul["Baseline"].header)
        print(hdul["Gain"].header)
        with open(storeFilename_.split('.')[0]+'.log', 'r') as logFile:
            if(logFile.readlines() == []):
                logging.info(" No errors occurred during the Fit-Value calculation.")
        logging.info(" Passed verifying Checksum")
    except Exception as errInfos:
        errorStr = str(errInfos)
        print(errorStr)
        logging.warning(errorStr)

    print("Done")


####################################################################################################
def getLinearFitValues(xValues_, yValues_, yValuesErrors_=[]):
    yWeighting = 1/pow(yValuesErrors_, 2)

    S_1 = np.sum(yWeighting)
    S_x = np.sum(yWeighting*xValues_)
    S_xx = np.sum(yWeighting*pow(xValues_, 2))

    S_y = np.sum(yWeighting*yValues_)
    S_xy = np.sum(yWeighting*xValues_*yValues_)

    D = S_1*S_xx - pow(S_x, 2)

    var = [(-S_x*S_y + S_1*S_xy)*(1/D), (S_xx*S_y - S_x*S_xy)*(1/D)]
    cov = [[S_1/D, -S_x/D], [-S_x/D, S_xx/D]]

    return(var, cov)
