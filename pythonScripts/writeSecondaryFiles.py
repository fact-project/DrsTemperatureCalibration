import drsCalibrationTool as tool
import subprocess as sp
import pandas as pd
import numpy as np
import h5py
import sys
import os

from astropy.io import fits
from collections import namedtuple

from fact.credentials import create_factdb_engine

####################################################################################################
Constants = namedtuple("Constants", ["nrPix", "nrCap", "nrTempSenor"])
fact = Constants(nrPix=1440, nrCap=1024, nrTempSenor=160)


####################################################################################################
def getMaxTempDiff(tempDiffFilename_, maxNr_=1):
    with fits.open(tempDiffFilename_) as store:
        date = store[1].data["date"]
        drsRunId = store[1].data["drsRunId"]
        tempDiffs = store[1].data["maxTempDiff"]
        tempDiffFlat = sorted(np.concatenate(tempDiffs))[-maxNr_]


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


# TODO FIX Checksum error
####################################################################################################
def temperatureMaxDifferencesPerPatch(storeFilename_, isdcRootPath_, startDate_, endDate_, freq_="D"):
    print(">> Run 'Temperature: MaxDifferencesPerPatch' <<")

    if(not os.path.isdir(isdcRootPath_)):
        print("Folder '", isdcRootPath_, "' does not exist")
        sys.exit()

    if(os.path.isfile(storeFilename_)):
        userInput = input("File ’"+str(storeFilename_)+"’ allready exist.\n" +
                          " Type ’y’ to overwrite File\nYour input: ")
        if(userInput != 'y'):
            sys.exit()

    elif(not os.path.isdir(storeFilename_[0:storeFilename_.rfind("/")])):
        print("Folder '", storeFilename_[0:storeFilename_.rfind("/")], "' does not exist")
        sys.exit()

    engine = create_factdb_engine
    dbTable = pd.read_sql("RunInfo", engine, columns=["fNight", "fRunID",
                                                      "fRunTypeKey", "fDrsStep",
                                                      "fNumEvents"])

    dateList = []
    drsRunIdList = []
    maxTempDiffList = []

    month_before = 0
    for date in pd.date_range(start=startDate_, end=endDate_, freq=freq_):
        if(month_before < date.month):
            month_before = date.month
            print("Month: ", date.month)
        # print("Date: ", date)

        datePath = date.strftime('%Y/%m/%d/')
        dateStr = date.strftime('%Y%m%d')

        tempFile = isdcRootPath_+"gpfs0/fact/fact-archive/rev_1/aux/"+datePath+dateStr+".FAD_CONTROL_TEMPERATURE.fits"
        drsDateStr = date.strftime('%Y-%m-%d')

        if(os.path.isfile(tempFile)):
            # print("found tempFile: ", tempFile)
            with fits.open(tempFile) as tabTemp:
                time = tabTemp[1].data["Time"]
                datetime = pd.to_datetime(time * 24 * 3600 * 1e9)
                temp = tabTemp[1].data["temp"]
        else:
            continue

        selectedDrsInfos = dbTable.query("fNight =="+str(dateStr)+"&" +
                                         "fRunTypeKey == 2 & fDrsStep == 2 & fNumEvents == 1000").copy()
        totalDrsRunIdList = selectedDrsInfos["fRunID"].tolist()

        drsTemp = None
        drsRunIdOfTheDay = np.array([])
        maxTempDiffOfTheDay = np.array([])
        for drsRunId in totalDrsRunIdList:
            drsFile = (isdcRootPath_+"gpfs0/fact/fact-archive/rev_1/raw/" +
                       datePath+dateStr+"_"+str("{:03d}".format(drsRunId))+".drs.fits.gz")
            if (os.path.isfile(drsFile)):
                # print("Use File: ", drsFile)
                with fits.open(drsFile) as drsTab:
                    drsRunStart = pd.to_datetime(drsTab[1].header["DATE-OBS"])
                    drsRunEnd = pd.to_datetime(drsTab[1].header["DATE-END"])

                drsRunIdOfTheDay = np.append(drsRunIdOfTheDay, drsRunId)
                drsRunIndices = np.where((datetime > drsRunStart) & (datetime < drsRunEnd))[0]
                # check first Drs-Run
                if(drsTemp is None):
                    # Save the Date where a mimimum of one drs-Run was found
                    dateList.append([drsDateStr])
                else:
                    tempInterval = temp[0:drsRunIndices[0]]
                    maxTempDiffOfTheDay = np.append(maxTempDiffOfTheDay, np.array(
                                                    np.amax([
                                                             np.amax(tempInterval, axis=0) - drsTemp,
                                                             drsTemp - np.amin(tempInterval, axis=0)
                                                             ], axis=0)))

                # Save Drs-Run Temperature
                drsTemp = np.mean(temp[drsRunIndices], axis=0)
                # cutoff the previous values of datetime and temp
                datetime = datetime[drsRunIndices[-1]+1:]
                temp = temp[drsRunIndices[-1]+1:]

        if(drsTemp is not None):
            # append values after last Drs-Run
            drsTemp = np.mean(temp, axis=0)
            maxTempDiffOfTheDay = np.append(maxTempDiffOfTheDay, np.array(
                                            np.amax([
                                                     np.amax(tempInterval, axis=0) - drsTemp,
                                                     drsTemp - np.amin(tempInterval, axis=0)
                                                     ], axis=0)))
            # append data of the day
            drsRunIdList.append(drsRunIdOfTheDay.astype("uint16"))
            maxTempDiffList.append(maxTempDiffOfTheDay)

    print("Write Data to Table")
    tbhduTempDiff = fits.BinTableHDU.from_columns(
            [fits.Column(name="date",        format="10A",
                         unit="yyyy-mm-dd",  array=dateList),
             fits.Column(name="drsRunId",    format="PB()",
                         unit="1",           array=drsRunIdList),
             fits.Column(name="maxTempDiff", format="PE()",
                         unit="degree C",    array=maxTempDiffList)])
    tbhduTempDiff.header.insert("TFIELDS", ("EXTNAME", "MaxDrsTempDiffs"), after=True)
    commentStr = "Maximum of the Temperature difference between two following Drs-Runs"
    tbhduTempDiff.header.insert("EXTNAME", ("comment", commentStr), after=True)

    print("Save Table")
    thdulist = fits.HDUList([fits.PrimaryHDU(), tbhduTempDiff])
    thdulist.writeto(storeFilename_, overwrite=True, checksum=True)

    print("Verify Checksum")
    # Open the File verifying the checksum values for all HDUs
    try:
        hdul = fits.open(storeFilename_, checksum=True)
        print(hdul["MaxDrsTempDiffs"].header)
        print("Passed verifying Checksum")
    except Exception as errInfos:
        errorStr = str(errInfos)
        print(errorStr)

    print(">> Finished 'Temperature: MaxDifferencesPerPatch'")


# TODO add checksum
####################################################################################################
def residumOfAllCapacitors(drsFilename_, fitFilname_, storeFilename_):
    print(">> Run 'ResidumOfAllCapacitors' <<")

    if(os.path.isfile(storeFilename_)):
        userInput = input("File ’"+str(storeFilename_)+"’ allready exist.\n" +
                          " Type ’y’ to overwrite File\nYour input: ")
        if(userInput != 'y'):
            sys.exit()

    elif(not os.path.isdir(storeFilename_[0:storeFilename_.rfind("/")])):
        print("Folder '", storeFilename_[0:storeFilename_.rfind("/")], "' does not exist")
        sys.exit()

    with h5py.File(drsFilename_, 'r') as store:
        creationDate = store["CreationDate"][0][0]

    with fits.open(fitFilname_, ignoremissing=True, ignore_missing_end=True) as fitValuesTab:
        # checksum = fitValuesTab[0].header["CHECKSUM"].encode("UTF-8", "ignore")
        intervalStart = fitValuesTab[0].header["LowLimit"].encode("UTF-8", "ignore")
        intervalEnd = fitValuesTab[0].header["UppLimit"].encode("UTF-8", "ignore")

        # print(type(fitValuesTab[0].header["LowLimit"]), type(fitValuesTab[0].header["CHECKSUM"]))
        # print(type(checksum), type(creationDate))

    interval = [intervalStart, intervalEnd]

    with h5py.File(storeFilename_, 'w') as hf:
        hf.create_dataset("CreationDate", (1, 1), dtype='S19', maxshape=(1, 1),
                          compression="gzip", compression_opts=9, fletcher32=True)

        hf.create_dataset("DrsCreationDate", (1, 1), dtype='S19', data=creationDate, maxshape=(1, 1),
                          compression="gzip", compression_opts=9, fletcher32=True)
        # hf.create_dataset('FitChecksum', (1, 1), dtype='S16', data=checksum, maxshape=(1, 1),
        #                   compression="gzip", compression_opts=9, fletcher32=True)
        hf.create_dataset("Interval", (2, 1), dtype='S19', data=interval, maxshape=(2, 1),
                          compression="gzip", compression_opts=9, fletcher32=True)

        hf.create_dataset("ResiduenBaseline", (0, 0), maxshape=(fact.nrPix*fact.nrCap, None),
                          compression="gzip", compression_opts=9, fletcher32=True)

        hf.create_dataset("ResiduenGain",     (0, 0), maxshape=(fact.nrPix*fact.nrCap, None),
                          compression="gzip", compression_opts=9, fletcher32=True)

    drsTypes = ["Baseline", "Gain"]
    for valueType in drsTypes:
        print("Loading '"+valueType+"-data' ...")
        with h5py.File(drsFilename_, 'r') as store:
            temp = np.array(store["Temp"+valueType])
            drsValue = np.array(store[valueType+"Mean"])

        with fits.open(fitFilname_, ignoremissing=True, ignore_missing_end=True) as fitValuesTab:
            slope = fitValuesTab[valueType].data["slope"][0]
            offset = fitValuesTab[valueType].data["offset"][0]

        with h5py.File(storeFilename_) as store:
            data = store["Residuen"+valueType]
            data.resize((fact.nrPix*fact.nrCap, drsValue.shape[0]))

        print("Calculate residuen for '"+str(fact.nrPix)+"' Pixel")

        for pixelNr in range(fact.nrPix):
            if(((pixelNr/fact.nrPix*100) % 1) < (((pixelNr-1)/fact.nrPix*100) % 1) and
               ((pixelNr/fact.nrPix*100) % 1) < (((pixelNr+1)/fact.nrPix*100) % 1)):
                print("PixelNr:", str('{:4d}'.format(pixelNr+1)), ":", '{:2d}'.format(int(pixelNr/fact.nrPix*100)), '%')

            drsTypeResiduenPixel = []
            tempPixel = temp[:, int(pixelNr/9)]
            for capNr in range(fact.nrCap):
                drsValueCap = drsValue[:, pixelNr*fact.nrCap+capNr]
                slopeCap = slope[pixelNr*fact.nrCap+capNr]
                offsetCap = offset[pixelNr*fact.nrCap+capNr]
                drsTypeResiduenPixel.append(np.array(drsValueCap-(slopeCap*tempPixel + offsetCap)).astype('float64'))

            with h5py.File(storeFilename_) as store:
                data = store["Residuen"+valueType]
                data[pixelNr*fact.nrCap:(pixelNr+1)*fact.nrCap] = drsTypeResiduenPixel

    print("Add CreationDate")
    creationDateStr = pd.datetime.now().strftime('%Y-%m-%d %H:%M:%S').encode("UTF-8", "ignore")
    with h5py.File(storeFilename_) as store:
        store['CreationDate'][0] = [creationDateStr]

    print(">> Finished 'ResidumOfAllCapacitors' <<")


####################################################################################################
def drsPedestalRunNoise(isdcRootPath_, sourcePath_, storePath_, factPath_, intervalFitValueDataFilename_,
                        startDate_, endDate_, freq_="D"):
    print(">> Run 'DrsPedestalRunNoise' <<")

    if(not os.path.isdir(storePath_)):
        print("Folder '", storePath_, "' does not exist")
        sys.exit()

    if(not os.path.isdir(isdcRootPath_)):
        print("Folder '", isdcRootPath_, "' does not exist")
        sys.exit()

    print("Loading Database ...")
    engine = create_factdb_engine
    dbTable = pd.read_sql("RunInfo", engine, columns=["fNight", "fRunID",
                                                      "fRunTypeKey", "fDrsStep",
                                                      "fNumEvents", "fBiasVoltageMedian"])

    month_before = 0
    for date in pd.date_range(start=startDate_, end=endDate_, freq=freq_):
        if(month_before < date.month):
            month_before = date.month
            print("Month: ", date.month)
        print("Date: ", date)

        dateStr = date.strftime('%Y%m%d')

        datePath = date.strftime('%Y/%m/%d/')
        auxFolder = isdcRootPath_+"gpfs0/fact/fact-archive/rev_1/aux/"+datePath
        tempFile = (auxFolder+dateStr+".FAD_CONTROL_TEMPERATURE.fits")

######################################
        # TODO select right one

        factToolsPath = factPath_+"target/fact-tools-0.17.2.jar"
        factToolsXmlPath = factPath_+"examples/studies/drsTemperatureCalibrationCheck.xml"
        fitValueDataFilename = sourcePath_+"fitValues/fitValuesData_Interval1of1.fits"
######################################

        if(not os.path.isfile(tempFile)):
            print("Date: ", date, " has no temp file")
            continue

        selectedDrsInfos = dbTable.query("fNight =="+str(dateStr)+"&" +
                                         "fRunTypeKey == 2 & fDrsStep == 2 & fNumEvents == 1000").copy()

        selectedDrsInfos["date"] = pd.to_datetime(selectedDrsInfos.fNight.astype(str), format="%Y%m%d")#
        drsRunIdList = selectedDrsInfos["fRunID"].tolist()

        # just take the one drs-Run of the middle of the night
        drsRunIndex = int(len(drsRunIdList)/2)-1
        drsRunId = drsRunIdList[drsRunIndex]
        drsFile = (isdcRootPath_+"gpfs0/fact/fact-archive/rev_1/raw/" +
                   datePath+dateStr+"_"+str("{:03d}".format(drsRunIdList[drsRunIndex]))+".drs.fits.gz")

        print(drsFile)#
        if(not os.path.isfile(drsFile)):
            print("Date: ", date, " has no drs-file", drsFile)
            continue

        selectedDrsInfos = dbTable.query("fNight =="+str(dateStr)+"&" +
                                         "fRunTypeKey == 2 &" +
                                         # fDrsStep == NaN   and fBiasVoltageMedian == NaN #  dosent work
                                         "fDrsStep != fDrsStep & fBiasVoltageMedian != fBiasVoltageMedian").copy()

        selectedDrsInfos["date"] = pd.to_datetime(selectedDrsInfos.fNight.astype(str), format="%Y%m%d")#
        pedestelRunIdList = selectedDrsInfos["fRunID"].tolist()

        usedPedestelRunIds = []
        pedestelRunFileList = []
        for runId in pedestelRunIdList:
            pedestelRunFilename = (isdcRootPath_+"gpfs0/fact/fact-archive/rev_1/raw/" +
                                   datePath+dateStr+"_"+str("{:03d}".format(runId))+".fits.fz")
            if(os.path.isfile(pedestelRunFilename)):
                usedPedestelRunIds.append(runId)
                pedestelRunFileList.append(pedestelRunFilename)
            else:
                print(pedestelRunFilename, " not found")

        with fits.open(tempFile) as tempTab:
            timeList = np.array(tempTab[1].data['Time'])
            tempList = np.array(tempTab[1].data['temp'])
            tempDatetime = pd.to_datetime(timeList * 24 * 3600 * 1e9)

        with fits.open(drsFile) as drsTab:
            drsStart = pd.to_datetime(drsTab[1].header["DATE-OBS"])
            drsEnd = pd.to_datetime(drsTab[1].header["DATE-END"])
            # mean ignore patches -->, axis=0 <--
            drsTempMean = np.mean(tempList[np.where((tempDatetime > drsStart) & (tempDatetime < drsEnd))])

        tempDiffList = []
        for runIndex in range(len(pedestelRunFileList)):
            runFile = pedestelRunFileList[runIndex]
            runId = usedPedestelRunIds[runIndex]

            with fits.open(runFile) as runTab:
                runStart = runTab[2].header["DATE-OBS"]
                runEnd = runTab[2].header["DATE-END"]

            runTemp = tempList[np.where((tempDatetime > runStart) & (tempDatetime < runEnd))[0]]

            if(len(runTemp) == 0):
                runTemp = tempList[np.where((tempDatetime < runStart))[0][-1]:
                                   np.where((tempDatetime > runEnd))[0][0]+1]

            tempDiffList.append(abs(np.mean(runTemp) - drsTempMean))
            #continue#
            print("run java ", runIndex+1, "/", len(pedestelRunFileList))
#pedestelNoise
            storeFilname = storePath_+"noise/data"+dateStr+"_{0:03d}".format(runId)+"_timeRow.fits"  # _spikes
            runFactTools(factToolsPath, factToolsXmlPath, runFile, drsFile, storeFilname, auxFolder,
                         fitValueDataFilename, intervalFitValueDataFilename_)
            return#
        print("Join Noise.fits of ", dateStr)

        drsCalibratedDataNoise = []
        drsCalibratedData_TempNoise = []
        drsCalibratedData_TempIntervalNoise = []

        for runId in usedPedestelRunIds:
            print("Add run ID: ", runId)
            sourceFile = storePath_+"noise/pedestelNoise"+dateStr+"_"+"{0:03d}".format(runId)+".fits"  # _spikes
            if(os.path.isfile(sourceFile)):
                with fits.open(sourceFile) as noiseTab:
                    drsCalibratedDataNoise.append(
                        noiseTab[1].data["DRSCalibratedDataNoise"].flatten())
                    #drsCalibratedData_TempNoise.append(
                    #    noiseTab[1].data["DRSCalibratedData_TempNoise"].flatten())
                    drsCalibratedData_TempIntervalNoise.append(
                        noiseTab[1].data["DRSCalibratedData_TempIntervalNoise"].flatten())
                #os.remove(sourceFile)

        # TODO add run ids
        print("Write Data to Table")
        tbhduNoise = fits.BinTableHDU.from_columns(
                [fits.Column(name="PedestelRunId",       format='1I',
                             unit="1",           array=usedPedestelRunIds),
                 fits.Column(name="TempDiff",    format='1E',
                             unit="Degree C",    array=tempDiffList),
                 fits.Column(name="drsCalibratedDataNoise",            format='PE()',
                             unit="mV",          array=drsCalibratedDataNoise),
                 #fits.Column(name="drsCalibratedData_TempNoise",       format='PE()',
                 #            unit="mV",          array=drsCalibratedData_TempNoise),
                 fits.Column(name="drsCalibratedData_TempIntervalNoise",   format='PE()',
                             unit="mV",          array=drsCalibratedData_TempIntervalNoise)])
        tbhduNoise.header.insert("TFIELDS", ("EXTNAME", "NoisePerPixel"), after=True)
        commentStr = ("-")
        tbhduNoise.header.insert("EXTNAME", ("comment", commentStr), after="True")
        tbhduNoise.header.insert("comment", ("Date", dateStr, "Date yyyy-mm-dd"), after=True)
        tbhduNoise.header.insert("Date", ("DrsRunId", drsRunId, "RunID of the based drsRunFile"), after=True)

        storeFilename_ = storePath_+"noise/pedestelNoise"+dateStr+".fits"  # _spikes
        print("Save Table")
        thdulist = fits.HDUList([fits.PrimaryHDU(), tbhduNoise])
        thdulist.writeto(storeFilename_, clobber=True, checksum=True)
        print("Verify Checksum")
        # Open the File verifying the checksum values for all HDUs
        try:
            hdul = fits.open(storeFilename_, checksum=True)
            print(hdul["NoisePerPixel"].header)
        except Exception as errInfos:
            errorStr = str(errInfos)
            print(errorStr)

    print(">> Finished 'DrsPedestalRunNoise' <<")


####################################################################################################
def runFactTools(factToolsPath_, factToolsXmlPath_, runFile_, drsFile_, storeFilname_, auxFolder_,
                 fitValueDataFilename_, intervalFitValueDataFilename_):

    print(factToolsPath_)
    print(factToolsXmlPath_)
    print(runFile_)
    print(drsFile_)
    print(storeFilname_)
    print(auxFolder_)
    print(fitValueDataFilename_)
    print(intervalFitValueDataFilename_)
    sp.run(["java", "-jar", "{}".format(factToolsPath_),
            "{}".format(factToolsXmlPath_),
            "-Dinfile=file:{}".format(runFile_),
            "-Ddrsfile=file:{}".format(drsFile_),
            "-Doutfile=file:{}".format(storeFilname_),
            "-DauxFolder=file:{}".format(auxFolder_),
            "-DfitValueFile=file:{}".format(fitValueDataFilename_),
            "-DfitValueFileInterval=file:{}".format(intervalFitValueDataFilename_),
            "-j8"])
