import pandas as pd
import numpy as np
import h5py
import sys
import os
import os.path

import logging

from collections import namedtuple
from astropy.io import fits
from fact.credentials import create_factdb_engine


####################################################################################################
Constants = namedtuple("Constants", ["nrPix", "nrCap", "nrTempSenor"])
fact = Constants(nrPix=1440, nrCap=1024, nrTempSenor=160)


####################################################################################################
# 1. search through the fact-database and find all drsFiles
# 2. filter them, take just one drsFile per day
####################################################################################################
def searchDrsFiles(storeFilename_):
    print(">> Run 'SearchDrsFiles' <<")

    if(not os.path.isdir(storeFilename_[0:storeFilename_.rfind("/")])):
        print("Folder '", storeFilename_[0:storeFilename_.rfind("/")], "' does not exist")
        sys.exit()

    def filename(row_):
        return os.path.join(
            "/fact/raw",
            str(row_.date.year),
            "{:02d}".format(row_.date.month),
            "{:02d}".format(row_.date.day),
            "{}_{:03d}.drs.fits.gz".format(row_.fNight, row_.fRunID),
        )

    engine = create_factdb_engine()

    drsInfos = pd.read_sql("RunInfo", engine, columns=["fNight", "fRunID", "fRunTypeKey", "fDrsStep", "fNumEvents"])
    drsFileInfos = drsInfos.query("fRunTypeKey == 2 & fDrsStep == 2 & fNumEvents == 1000").copy()
    # fNumEvents == 1000 prevent for unfinished/broken files
    drsFileInfos["date"] = pd.to_datetime(drsFileInfos.fNight.astype(str), format="%Y%m%d")

    drsFiles = drsFileInfos.apply(filename, axis=1).tolist()

    ##############################
    # #### Filter Drs-Files #### #
    ##############################

    print(">> In 'SearchDrsFiles' start with filter Drs-Files <<")

    selectedFiles = []
    previousDates = []
    for drsFilename in drsFiles:
        drsFilename = drsFilename.strip("\n")

        date = drsFilename.split("_")[0].split("/")[-1]
        if(date not in previousDates and int(drsFilename.split("_")[0].split("/")[-4]) >= 2012):

            previousDates.append(date)
            selectedFiles.append(drsFilename)

    pd.DataFrame(selectedFiles).to_csv(storeFilename_, index=False, header=False)

    print(">> Finished 'SearchDrsFiles' <<")


####################################################################################################
####################################################################################################
# ##############                         saveDrsAttributes                          ############## #
####################################################################################################
####################################################################################################

# save Baseline and Gain of all drsfiles of the drsFileList
# together with the mean of Time and Temperature of taking
# into a .h5 File

def saveDrsAttributes(drsFileList_, storeFilename_):
    print(">> Run 'SaveDrsAttributes' <<")

    if(os.path.isfile(storeFilename_)):
        userInput = input("File ’"+str(storeFilename_)+"’ allready exist.\n" +
                          " Type ’y’ to overwrite File\nYour input: ")
        if(userInput != 'y'):
            sys.exit()

    elif(not os.path.isdir(storeFilename_[0:storeFilename_.rfind("/")])):
        print("Folder '", storeFilename_[0:storeFilename_.rfind("/")], "' does not exist")
        sys.exit()

    logging.basicConfig(filename=storeFilename_.split('.')[0]+".log", filemode='w',
                        format='%(levelname)s:%(message)s', level=logging.DEBUG)

    global fact
    nrFactValues = fact.nrPix*fact.nrCap

    with h5py.File(storeFilename_, 'w') as hf:
        hf.create_dataset('CreationDate', (1, 1), dtype='S19', maxshape=(1, 1),
                          compression="gzip", compression_opts=9, fletcher32=True)

        hf.create_dataset("TimeBaseline",    (0, 1), maxshape=(None, 1),
                          compression="gzip", compression_opts=9, fletcher32=True)
        hf.create_dataset("TempBaseline",    (0, fact.nrTempSenor), maxshape=(None, fact.nrTempSenor),
                          compression="gzip", compression_opts=9, fletcher32=True)
        hf.create_dataset("TempStdBaseline", (0, fact.nrTempSenor), maxshape=(None, fact.nrTempSenor),
                          compression="gzip", compression_opts=9, fletcher32=True)
        hf.create_dataset("BaselineMean",    (0, nrFactValues), maxshape=(None, nrFactValues),
                          compression="gzip", compression_opts=9, fletcher32=True)
        hf.create_dataset("BaselineMeanStd", (0, nrFactValues), maxshape=(None, nrFactValues),
                          compression="gzip", compression_opts=9, fletcher32=True)

        hf.create_dataset("TimeGain",        (0, 1), maxshape=(None, 1),
                          compression="gzip", compression_opts=9, fletcher32=True)
        hf.create_dataset("TempGain",        (0, fact.nrTempSenor), maxshape=(None, fact.nrTempSenor),
                          compression="gzip", compression_opts=9, fletcher32=True)
        hf.create_dataset("TempStdGain",     (0, fact.nrTempSenor), maxshape=(None, fact.nrTempSenor),
                          compression="gzip", compression_opts=9, fletcher32=True)
        hf.create_dataset("GainMean",        (0, nrFactValues), maxshape=(None, nrFactValues),
                          compression="gzip", compression_opts=9, fletcher32=True)
        hf.create_dataset("GainMeanStd",     (0, nrFactValues), maxshape=(None, nrFactValues),
                          compression="gzip", compression_opts=9, fletcher32=True)

    count = 0
    countMax = sum(1 for line in open(drsFileList_))
    with open(drsFileList_) as drsFileList:
        print("Check '", countMax, "' drsFiles'")
        for drsFilename in drsFileList:
            drsFilename = drsFilename.strip("\n")
            count = count + 1

            if(((count/countMax*100) % 1) < (((count-1)/countMax*100) % 1) and
               ((count/countMax*100) % 1) < (((count+1)/countMax*100) % 1)):
                print('{:4d}'.format(count), ":", '{:2d}'.format(int(count/countMax*100)), '%')

            tempFilename = (str("/fact/aux") +
                            str(drsFilename.split('_')[0].split("raw")[-1]) +
                            str(".FAD_CONTROL_TEMPERATURE.fits"))

            if(os.path.isfile(drsFilename) and os.path.isfile(tempFilename)):
                saveTupleOfAttribute(tempFilename, drsFilename, storeFilename_)

    print("Add CreationDate")
    creationDateStr = pd.datetime.now().strftime('%Y-%m-%d %H:%M:%S').encode("UTF-8", "ignore")
    with h5py.File(storeFilename_) as store:
        store['CreationDate'][0] = [creationDateStr]

    print(">> Finished 'SaveDrsAttributes' <<")


####################################################################################################
def saveTupleOfAttribute(tempFilename, drsFilename, storeFilename):

    global fact

    loggingFlag = True
    errorFlag = False
    try:
        tabTemp = fits.open(tempFilename, ignoremissing=True, ignore_missing_end=True)
        tabDrs = fits.open(drsFilename, ignoremissing=True, ignore_missing_end=True)

    except Exception as errInfos:
        errorFlag = True
        if(loggingFlag):
            errorStr = " LoadingError: in'"+drsFilename+"' or '"+tempFilename+"' ("+str(errInfos)+")"
            # print(errorStr)
            logging.critical(errorStr)
        return

    tabTemp_time = None
    tabTemp_temp = None
    tabTempDatetime = None
    try:
        tabTemp_time = tabTemp[1].data["Time"]
        tabTemp_temp = tabTemp[1].data["temp"]
        tabTempDatetime = pd.to_datetime(tabTemp_time * 24 * 3600 * 1e9)

    except Exception as errInfos:
        errorFlag = True
        if(loggingFlag):
            errorStr = " In File '"+tempFilename+"': "+str(errInfos)
            # print(errorStr)
            logging.error(errorStr)

    if(tabTemp_temp is not None and tabTemp_temp.shape[1] != fact.nrTempSenor):
        errorFlag = True
        if(loggingFlag):
            errorStr = ("File not used: Just "+str(tabTemp_temp.shape[1]) +
                        " Temperature Values in File '"+tempFilename+"'")
            # print(errorStr)
            logging.error(errorStr)
        return

    begRun_0 = None
    endRun_0 = None
    try:
        begRun_0 = pd.to_datetime(tabDrs[1].header["RUN0-BEG"])

    except Exception as errInfos:
        errorFlag = True
        if(loggingFlag):
            errorStr = " In File '"+drsFilename+"': "+str(errInfos)
            # print(errorStr)
            logging.error(errorStr)

    try:
        endRun_0 = pd.to_datetime(tabDrs[1].header["RUN0-END"])

    except Exception as errInfos:
        errorFlag = True
        if(loggingFlag):
            errorStr = " In File '"+drsFilename+"': "+str(errInfos)
            # print(errorStr)
            logging.error(errorStr)

    if(errorFlag is False):
        baselineMean = tabDrs[1].data["BaselineMean"][0]
        baselineMeanStd = tabDrs[1].data["BaselineRms"][0]

        baselineMeanNulls = list(np.array(np.where(baselineMean == 0.)[0]))
        if (len(baselineMeanNulls) != 0.):
            errorFlag = True
            if(loggingFlag):
                errorStr = (" File not used: Nulls of baselineMean in File '"+str(drsFilename) +
                            "' Nulls at Index:\n"+str(baselineMeanNulls))
                # print(errorStr)
                logging.error(errorStr)

        baselineMeanStdNulls = list(np.array(np.where(baselineMeanStd == 0.)[0]))
        if (len(baselineMeanStdNulls) != 0.):
            errorFlag = True
            if(loggingFlag):
                errorStr = (" File not used: Nulls of baselineMeanStd in File '"+str(drsFilename) +
                            "' Nulls at Index:\n"+str(baselineMeanStdNulls))
                # print(errorStr)
                logging.error(errorStr)

    if(errorFlag is False):
        indicesRun_0 = np.where((tabTempDatetime > begRun_0) & (tabTempDatetime < endRun_0))[0]
        timeValuesRun_0 = np.array(tabTemp_time[indicesRun_0])
        tempValuesRun_0 = np.array(tabTemp_temp[indicesRun_0])

        if(timeValuesRun_0.shape[0] > 1):
            timeBaseline = np.mean(timeValuesRun_0, dtype="float64")
        else:
            timeBaseline = timeValuesRun_0

        if(tempValuesRun_0.shape[0] > 1):
            tempBaseline = np.mean(tempValuesRun_0, dtype="float64", axis=0)
            tempStdBaseline = np.std(tempValuesRun_0, dtype="float64", axis=0, ddof=1)
        else:
            tempBaseline = tempValuesRun_0
            tempStdBaseline = np.zeros(tempValuesRun_0.shape[1])

    begRun_1 = None
    endRun_1 = None
    try:
        begRun_1 = pd.to_datetime(tabDrs[1].header["RUN1-BEG"])

    except Exception as errInfos:
        errorFlag = True
        if(loggingFlag):
            errorStr = " In File '"+drsFilename+"': "+str(errInfos)
            # print(errorStr)
            logging.error(errorStr)

    try:
        endRun_1 = pd.to_datetime(tabDrs[1].header["RUN1-END"])

    except Exception as errInfos:
        errorFlag = True
        if(loggingFlag):
            errorStr = " In File '"+drsFilename+"': "+str(errInfos)
            # print(errorStr)
            logging.error(errorStr)

    if(errorFlag is False):
        gainMean = tabDrs[1].data["GainMean"][0]
        gainMeanStd = tabDrs[1].data["GainRms"][0]

        gainMeanNulls = list(np.array(np.where(gainMean == 0.)[0]))
        if (len(gainMeanNulls) != 0.):
            errorFlag = True
            if(loggingFlag):
                errorStr = (" File not used: Nulls of gainMean in File '"+str(drsFilename) +
                            "' Nulls at Index:\n"+str(gainMeanNulls))
                # print(errorStr)
                logging.error(errorStr)

        gainMeanStdNulls = list(np.array(np.where(gainMeanStd == 0.)[0]))
        if (len(gainMeanStdNulls) != 0.):
            errorFlag = True
            if(loggingFlag):
                errorStr = (" File not used: Nulls of gainMeanStd in File '"+str(drsFilename) +
                            "' Nulls at Index:\n"+str(gainMeanStdNulls))
                # print(errorStr)
                logging.error(errorStr)

    if(errorFlag is False):
        indicesRun_1 = np.where((tabTempDatetime > begRun_1) & (tabTempDatetime < endRun_1))[0]
        timeValuesRun_1 = np.array(tabTemp_time[indicesRun_1])
        tempValuesRun_1 = np.array(tabTemp_temp[indicesRun_1])

        if(timeValuesRun_1.shape[0] > 1):
            timeGain = np.mean(timeValuesRun_1, dtype="float64")
        else:
            timeGain = timeValuesRun_1

        if(tempValuesRun_1.shape[0] > 1):
            tempGain = np.mean(tempValuesRun_1, dtype="float64", axis=0)
            tempStdGain = np.std(tempValuesRun_1, dtype="float64", axis=0, ddof=1)
        else:
            tempGain = tempValuesRun_1
            tempStdGain = np.zeros(tempValuesRun_1.shape[1])

    with h5py.File(storeFilename) as store:
        if(errorFlag is False):
            data = store["TimeBaseline"]
            data.resize((len(data)+1, data.maxshape[1]))
            data[len(data)-1, :] = timeBaseline

            data = store["TempBaseline"]
            data.resize((len(data)+1, data.maxshape[1]))
            data[len(data)-1, :] = tempBaseline

            data = store["TempStdBaseline"]
            data.resize((len(data)+1, data.maxshape[1]))
            data[len(data)-1, :] = tempStdBaseline

            data = store["BaselineMean"]
            data.resize((len(data)+1, data.maxshape[1]))
            data[len(data)-1, :] = baselineMean

            data = store["BaselineMeanStd"]
            data.resize((len(data)+1, data.maxshape[1]))
            data[len(data)-1, :] = baselineMeanStd

            data = store["TimeGain"]
            data.resize((len(data)+1, data.maxshape[1]))
            data[len(data)-1, :] = timeGain

            data = store["TempGain"]
            data.resize((len(data)+1, data.maxshape[1]))
            data[len(data)-1, :] = tempGain

            data = store["TempStdGain"]
            data.resize((len(data)+1, data.maxshape[1]))
            data[len(data)-1, :] = tempStdGain

            data = store["GainMean"]
            data.resize((len(data)+1, data.maxshape[1]))
            data[len(data)-1, :] = gainMean

            data = store["GainMeanStd"]
            data.resize((len(data)+1, data.maxshape[1]))
            data[len(data)-1, :] = gainMeanStd


####################################################################################################
####################################################################################################
# ##############                           saveFitValues                            ############## #
####################################################################################################
####################################################################################################

# Calculate the linear fitvalues of Basline and Gain of the .h5 source
# and store them into a .fits File
# All Basline/Gain-values with a bigger error than the 'CutOffErrorFactor'"
# multiplied with the mean of the error from all collected Baseline/Gain-values of the"
# Capacitor will not used for the fit

def saveFitValues(sourceFilename_, storeFilename_,
                  cutOffErrorFactorBaseline_, cutOffErrorFactorGain_,
                  firstDate_=None, lastDate_=None):

    print(">> Run 'SaveFitValues' <<")

    if(os.path.isfile(storeFilename_)):
        userInput = input("File ’"+str(storeFilename_)+"’ allready exist.\n" +
                          " Type ’y’ to overwrite File\nYour input: ")
        if(userInput != 'y'):
            sys.exit()

    elif(not os.path.isdir(storeFilename_[0:storeFilename_.rfind("/")])):
        print("Folder '", storeFilename_[0:storeFilename_.rfind("/")], "' does not exist")
        sys.exit()

    logging.basicConfig(filename=storeFilename_.split('.')[0]+".log", filemode='w',
                        format='%(levelname)s:%(message)s', level=logging.DEBUG)

    global fact

    # TODO check are dateBaseline and dateGain alwasy equal (drsFiles taken around 00:00)
    # ->just one indice-list needed
    with h5py.File(sourceFilename_, 'r') as dataSource:
        timeBaseline = np.array(dataSource["TimeBaseline"]).flatten()
        dateBaseline = pd.to_datetime(timeBaseline * 24 * 3600 * 1e9).date

        timeGain = np.array(dataSource["TimeGain"]).flatten()
        dateGain = pd.to_datetime(timeGain * 24 * 3600 * 1e9).date

    try:
        firstDate = pd.to_datetime(firstDate_).date()
    except Exception as errInfos:
        firstDate = dateBaseline[0]
        errorStr = "Used Data-startDate as firstDate. Input was '"+str(firstDate_)+"'\n("+str(errInfos)+")"
        print("Info:", errorStr)
        logging.warning(errorStr)

    try:
        lastDate = pd.to_datetime(lastDate_).date()
    except Exception as errInfos:
        lastDate = dateBaseline[-1]
        errorStr = "Used Data-endDate as lastDate. Input was '"+str(lastDate_)+"'\n("+str(errInfos)+")"
        print("Info:", errorStr)
        logging.warning(errorStr)

    intervalIndicesBaseline = np.where((dateBaseline >= firstDate) &
                                       (dateBaseline <= lastDate))[0]
    intervalIndicesGain = np.where((dateGain >= firstDate) &
                                   (dateGain <= lastDate))[0]

    print("Loading ...")
    with h5py.File(sourceFilename_, 'r') as dataSource:
        creationDate = dataSource["CreationDate"][0][0].decode("UTF-8")

        tempBaselineArray = np.array(dataSource["TempBaseline"][intervalIndicesBaseline, :])
        baselineMeanArray = np.array(dataSource["BaselineMean"][intervalIndicesBaseline, :])
        baselineMeanStdArray = np.array(dataSource["BaselineMeanStd"][intervalIndicesBaseline, :])

        tempGainArray = np.array(dataSource["TempGain"][intervalIndicesGain, :])
        gainMeanArray = np.array(dataSource["GainMean"][intervalIndicesGain, :])
        gainMeanStdArray = np.array(dataSource["GainMeanStd"][intervalIndicesGain, :])

    baselineMeanSlope = []
    baselineMeanSlopeStd = []
    baselineMeanOffset = []
    baselineMeanOffsetStd = []
    gainMeanSlope = []
    gainMeanSlopeStd = []
    gainMeanOffset = []
    gainMeanOffsetStd = []

    print("Calculate fitvalues for '"+str(fact.nrPix)+"' Pixel \n" +
          "for the period from "+str(firstDate)+" until "+str(lastDate))

    for pixelNr in range(fact.nrPix):

        if(((pixelNr/fact.nrPix*100) % 1) < (((pixelNr-1)/fact.nrPix*100) % 1) and
           ((pixelNr/fact.nrPix*100) % 1) < (((pixelNr+1)/fact.nrPix*100) % 1)):
            print("PixelNr:", str('{:4d}'.format(pixelNr+1)), ":", '{:2d}'.format(int(pixelNr/fact.nrPix*100)), '%')

        tempBaseline = tempBaselineArray[:, int(pixelNr/9)]
        tempGain = tempGainArray[:, int(pixelNr/9)]
        for capNr in range(fact.nrCap):
            baselineMeanCap = baselineMeanArray[:, pixelNr*fact.nrCap+capNr]
            baselineMeanStdCap = baselineMeanStdArray[:, pixelNr*fact.nrCap+capNr]
            baselineMeanStdCapMean = np.mean(baselineMeanStdCap, dtype="float")
            gainMeanCap = gainMeanArray[:, pixelNr*fact.nrCap+capNr]
            gainMeanStdCap = gainMeanStdArray[:, pixelNr*fact.nrCap+capNr]
            gainMeanStdCapMean = np.mean(gainMeanStdCap, dtype="float")

            try:
                indices = np.where(baselineMeanStdCap <
                                   baselineMeanStdCapMean*cutOffErrorFactorBaseline_)[0]
                varBase, covBase = getLinearFitValues(tempBaseline[indices],
                                                      baselineMeanCap[indices],
                                                      baselineMeanStdCap[indices])

                baselineMeanSlope.append(varBase[0])
                baselineMeanSlopeStd.append(np.sqrt(covBase[0][0]))
                baselineMeanOffset.append(varBase[1])
                baselineMeanOffsetStd.append(np.sqrt(covBase[1][1]))

            except Exception as errInfos:
                errorStr = ("Gain-Fit(PixelNr: "+str('{:04d}'.format(pixelNr)) +
                            ", capNr: "+str('{:04d}'.format(capNr)) +
                            ") with Slope="+str(varBase[0]) +
                            " and Offset='"+str(varBase[1])+": '"+str(errInfos)+"'")
                logging.warning(errorStr)

                baselineMeanSlope.append(varBase[0])
                baselineMeanSlopeStd.append(0)
                baselineMeanOffset.append(varBase[1])
                baselineMeanOffsetStd.append(0)

            try:
                indices = np.where(gainMeanStdCap <
                                   gainMeanStdCapMean*cutOffErrorFactorGain_)[0]
                varGain, covGain = getLinearFitValues(tempGain[indices],
                                                      gainMeanCap[indices],
                                                      gainMeanStdCap[indices])

                gainMeanSlope.append(varGain[0])
                gainMeanSlopeStd.append(np.sqrt(covGain[0][0]))
                gainMeanOffset.append(varGain[1])
                gainMeanOffsetStd.append(np.sqrt(covGain[1][1]))

            except Exception as errInfos:
                errorStr = ("Gain-Fit(PixelNr: "+str('{:04d}'.format(pixelNr)) +
                            ", capNr: "+str('{:04d}'.format(capNr)) +
                            ") with Slope="+str(varGain[0]) +
                            " and Offset='"+str(varGain[1])+": '"+str(errInfos)+"'")
                logging.warning(errorStr)

                gainMeanSlope.append(varGain[0])
                gainMeanSlopeStd.append(0)
                gainMeanOffset.append(varGain[1])
                gainMeanOffsetStd.append(0)

    print("Write Data to Table")
    primary = fits.PrimaryHDU()
    commentStr = "Datetime-String of the source .h5 creation."  # in the format 'yyyy-mm-dd HH:MM:SS'
    primary.header.insert("EXTEND", ("OrigDate", creationDate, commentStr), after=True)
    commentStr = "Date-String of the lower interval limit"  # [firstDate, lastDate]"  in the format 'yyyy-mm-dd'
    primary.header.insert("OrigDate", ("LowLimit", firstDate.strftime('%Y-%m-%d'), commentStr), after=True)
    commentStr = "Date-String of the upper interval limit"  # [firstDate, lastDate]"  in the format 'yyyy-mm-dd'
    primary.header.insert("LowLimit", ("UppLimit", lastDate.strftime('%Y-%m-%d'), commentStr), after=True)

    tbhdu_baseline = fits.BinTableHDU.from_columns(
            [fits.Column(name="Slope",       format=str(len(baselineMeanSlope))+'E',
                         unit="mV/celsius",  array=[baselineMeanSlope]),
             fits.Column(name="SlopeStd",    format=str(len(baselineMeanSlopeStd))+'E',
                         unit="mV/celsius",  array=[baselineMeanSlopeStd]),
             fits.Column(name="Offset",      format=str(len(baselineMeanOffset))+'E',
                         unit="mV",          array=[baselineMeanOffset]),
             fits.Column(name="OffsetStd",   format=str(len(baselineMeanOffsetStd))+'E',
                         unit="mV",          array=[baselineMeanOffsetStd])])
    tbhdu_baseline.header.insert("TFIELDS", ("EXTNAME", "Baseline"), after=True)
    commentStr = ("All baseline-values with a bigger error than the 'CutOff-ErrorFactor'" +
                  " multiplied with the mean of the error from all collected baseline-values" +
                  " will not used for the fit")
    tbhdu_baseline.header.insert("EXTNAME", ("Comment", commentStr), after="True")
    tbhdu_baseline.header.insert("Comment", ("CutOff", str(cutOffErrorFactorBaseline_),
                                             "Shortform of CutOffErrorFactor"), after=True)

    tbhdu_gain = fits.BinTableHDU.from_columns(
            [fits.Column(name="Slope",       format=str(len(gainMeanSlope))+'E',
                         unit="mV/celsius",  array=[gainMeanSlope]),
             fits.Column(name="SlopeStd",    format=str(len(gainMeanSlopeStd))+'E',
                         unit="mV/celsius",  array=[gainMeanSlopeStd]),
             fits.Column(name="Offset",      format=str(len(gainMeanOffset))+'E',
                         unit="mV",          array=[gainMeanOffset]),
             fits.Column(name="OffsetStd",   format=str(len(gainMeanOffsetStd))+'E',
                         unit="mV",          array=[gainMeanOffsetStd])])
    tbhdu_gain.header.insert("TFIELDS", ("EXTNAME", "Gain"), after=True)
    commentStr = ("All gain-values with a bigger error than the 'CutOffErrorFactor'" +
                  " multiplied with the mean of the error from all collected gain-values" +
                  " will not used for the fit")
    tbhdu_gain.header.insert("EXTNAME", ("Comment", commentStr), after=True)
    tbhdu_gain.header.insert("Comment", ("CutOff", str(cutOffErrorFactorGain_),
                                         "Shortform of CutOffErrorFactor"), after=True)
    print("Save Table")
    thdulist = fits.HDUList([primary, tbhdu_baseline, tbhdu_gain])
    thdulist.writeto(storeFilename_, overwrite=True, checksum=True)
    print("Verify Checksum")
    # Open the File verifying the checksum values for all HDUs
    try:
        hdul = fits.open(storeFilename_, checksum=True)
        print(hdul[0].header)
        print(hdul["Baseline"].header)
        print(hdul["Gain"].header)
        with open(storeFilename_.split('.')[0]+".log", 'r') as logFile:
            if(logFile.readlines() == []):
                logging.info(" No errors occurred during the Fit-Value calculation.")
        logging.info(" Passed verifying Checksum")
    except Exception as errInfos:
        errorStr = str(errInfos)
        print(errorStr)
        logging.warning(errorStr)

    print(">> Finished 'SaveFitValues' <<")


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
