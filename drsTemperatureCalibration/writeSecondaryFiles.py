import subprocess as sp
import pandas as pd
import numpy as np
import h5py
import yaml
import click
import sys
import os
from tqdm import tqdm
from astropy.io import fits

#from fact.credentials import create_factdb_engine

from drsTemperatureCalibration.constants import NRPIX, NRCAP
from drsTemperatureCalibration.tools import safety_stuff, mem


@click.command()
@click.argument('drs_file_path',
                default="/net/big-tank/POOL/projects/fact/drs_temp_calib_data/" +
                        "calibration/drsData.h5",
                type=click.Path(exists=True))
@click.argument('fit_file_path',
                default="/net/big-tank/POOL/projects/fact/drs_temp_calib_data/" +
                        "calibration/fitValuesDataInterval.fits",
                type=click.Path(exists=True))
@click.argument('interval_array',
                default=[1, 2, 3])
@click.argument('store_file_path',
                default="/net/big-tank/POOL/projects/fact/drs_temp_calib_data/" +
                        "calibration/drsResiduals_new.h5",
                type=click.Path(exists=False))
@click.argument('drs_temp_calib_config',
                default="/home/fschulz/drsTemperatureCalibration/drsTemperatureCalibration/" +
                        "drsTempCalibConfig.yaml",
                type=click.Path(exists=True))
####################################################################################################
def residum_of_all_capacitors(drs_file_path, fit_file_path,
                              interval_array, store_file_path,
                              drs_temp_calib_config):
    # TODO add stuff
    '''

        Args:
            drs_file_path (str): Full path to the sourceParameter file with the extension '.h5'
            fit_file_path (str): Full path to the fit-value-file with the extension '.fits'
            interval_array (int-array): (Sorted) Array of ints/numbers of intervals for analysis
            store_file_path (str): Full path to the store-file with the extension '.h5'
            drs_temp_calib_config (str): Path to the drsCalibrationConfig-file with the extension '.yaml'
    '''

    # TODO check safety stuff. maybe remove
    safety_stuff(store_file_path)

    with h5py.File(store_file_path) as store:
        store.clear()

    with open(drs_temp_calib_config) as calbConfig:
        config = yaml.safe_load(calbConfig)

    drs_value_types = config["drsValueTypes"]

    NRCALIBVALUES = NRPIX*NRCAP

    for interval_nr in interval_array:
        with fits.open(fit_file_path, ignoremissing=True, ignore_missing_end=True) as fit_values_tab:
            header = fit_values_tab["Interval"+str(interval_nr)].header
            low_limit = header["LowLimit"]
            upp_limit = header["UppLimit"]
            # checksum = header["CHECKSUM"].encode("UTF-8", "ignore")

        groupname = "Interval"+str(interval_nr)
        with h5py.File(store_file_path) as store:
            drs_group = store.create_group(groupname)
            drs_group.attrs["LowLimit"] = low_limit
            drs_group.attrs["UppLimit"] = upp_limit
            # drs_group.attrs["SourceCS"] = checksum

        for drs_value_type in drs_value_types:
            print("interval_nr: ", interval_nr, "part: ", drs_value_type)
            with fits.open(fit_file_path, ignoremissing=True, ignore_missing_end=True) as fit_values_tab:
                data = fit_values_tab[interval_nr].data

                slope = data[drs_value_type+"Slope"][0]
                offset = data[drs_value_type+"Offset"][0]

            with h5py.File(drs_file_path, 'r') as data_source:
                time = np.array(data_source["Time"+drs_value_type]).flatten()

                datetime = pd.to_datetime(time * 24 * 3600 * 1e9)
                interval_indices = np.where((datetime >= low_limit) & (datetime <= upp_limit))[0]

                temp = np.array(data_source["Temp"+drs_value_type][interval_indices, :])
                drs_value = np.array(data_source[drs_value_type+"Mean"][interval_indices, :])

            with h5py.File(store_file_path) as store:
                store[groupname].create_dataset(drs_value_type+"Residuals",
                                                (NRCALIBVALUES, drs_value.shape[0]),
                                                maxshape=(NRCALIBVALUES, None),
                                                compression="gzip",
                                                compression_opts=9,
                                                fletcher32=True
                                                )

            print("Calculate residuals for '"+str(NRPIX)+"' Pixel")
            for pixel_indx in tqdm(range(NRPIX)):
                drs_type_residuen_pixel = []
                temp_pixel = temp[:, int(pixel_indx/9)]
                for cap_indx in range(NRCAP):
                    drs_value_cap = drs_value[:, pixel_indx*NRCAP+cap_indx]
                    slope_cap = slope[pixel_indx*NRCAP+cap_indx]
                    offset_cap = offset[pixel_indx*NRCAP+cap_indx]

                    drs_type_residuen_cap = drs_value_cap-(slope_cap*temp_pixel + offset_cap)
                    drs_type_residuen_pixel.append(drs_type_residuen_cap)

                with h5py.File(store_file_path) as store:
                    data = store[groupname][drs_value_type+"Residuals"]
                    data[pixel_indx*NRCAP:(pixel_indx+1)*NRCAP] = drs_type_residuen_pixel

            # dont trust the Garbage Collector, so force to free memory
            del drs_value_cap
            del slope_cap
            del offset_cap
            del temp_pixel
            del drs_value
            del slope
            del offset
            del temp

    # add creationDate to h5 file
    creation_date_str = pd.datetime.now().strftime('%Y-%m-%d %H:%M:%S').encode("UTF-8", "ignore")
    with h5py.File(store_file_path) as store:
        store.attrs['CreationDate'] = creation_date_str

    print(">> Finished 'ResidumOfAll_capacitors' <<")


# # TODO FIX Checksum error
# ####################################################################################################
# def temperatureMaxDifferencesPerPatch(store_file_path, isdcRootPath_, startDate_, endDate_, freq_="D"):
#     print(">> Run 'Temperature: MaxDifferencesPerPatch' <<")
#
#     if(not os.path.isdir(isdcRootPath_)):
#         print("Folder '", isdcRootPath_, "' does not exist")
#         sys.exit()
#
#     if(os.path.isfile(store_file_path)):
#         userInput = input("File ’"+str(store_file_path)+"’ allready exist.\n" +
#                           " Type ’y’ to overwrite File\nYour input: ")
#         if(userInput != 'y'):
#             sys.exit()
#
#     elif(not os.path.isdir(store_file_path[0:store_file_path.rfind("/")])):
#         print("Folder '", store_file_path[0:store_file_path.rfind("/")], "' does not exist")
#         sys.exit()
#
#     engine = create_factdb_engine
#     dbTable = pd.read_sql("RunInfo", engine, columns=["fNight", "fRunID",
#                                                       "fRunTypeKey", "fDrsStep",
#                                                       "fNumEvents"])
#
#     dateList = []
#     drsRunIdList = []
#     maxTempDiffList = []
#
#     month_before = 0
#     for date in pd.date_range(start=startDate_, end=endDate_, freq=freq_):
#         if(month_before < date.month):
#             month_before = date.month
#             print("Month: ", date.month)
#         # print("Date: ", date)
#
#         datePath = date.strftime('%Y/%m/%d/')
#         dateStr = date.strftime('%Y%m%d')
#
#         tempFile = isdcRootPath_+"gpfs0/fact/fact-archive/rev_1/aux/"+datePath+dateStr+".FAD_CONTROL_TEMPERATURE.fits"
#         drsDateStr = date.strftime('%Y-%m-%d')
#
#         if(os.path.isfile(tempFile)):
#             # print("found tempFile: ", tempFile)
#             with fits.open(tempFile) as tabTemp:
#                 time = tabTemp[1].data["Time"]
#                 datetime = pd.to_datetime(time * 24 * 3600 * 1e9)
#                 temp = tabTemp[1].data["temp"]
#         else:
#             continue
#
#         selectedDrsInfos = dbTable.query("fNight =="+str(dateStr)+"&" +
#                                          "fRunTypeKey == 2 & fDrsStep == 2 & fNumEvents == 1000").copy()
#         totalDrsRunIdList = selectedDrsInfos["fRunID"].tolist()
#
#         drsTemp = None
#         drsRunIdOfTheDay = np.array([])
#         maxTempDiffOfTheDay = np.array([])
#         for drsRunId in totalDrsRunIdList:
#             drsFile = (isdcRootPath_+"gpfs0/fact/fact-archive/rev_1/raw/" +
#                        datePath+dateStr+"_"+str("{:03d}".format(drsRunId))+".drs.fits.gz")
#             if (os.path.isfile(drsFile)):
#                 # print("Use File: ", drsFile)
#                 with fits.open(drsFile) as drsTab:
#                     drsRunStart = pd.to_datetime(drsTab[1].header["DATE-OBS"])
#                     drsRunEnd = pd.to_datetime(drsTab[1].header["DATE-END"])
#
#                 drsRunIdOfTheDay = np.append(drsRunIdOfTheDay, drsRunId)
#                 drsRunIndices = np.where((datetime > drsRunStart) & (datetime < drsRunEnd))[0]
#                 # check first Drs-Run
#                 if(drsTemp is None):
#                     # Save the Date where a mimimum of one drs-Run was found
#                     dateList.append([drsDateStr])
#                 else:
#                     tempInterval = temp[0:drsRunIndices[0]]
#                     maxTempDiffOfTheDay = np.append(maxTempDiffOfTheDay, np.array(
#                                                     np.amax([
#                                                              np.amax(tempInterval, axis=0) - drsTemp,
#                                                              drsTemp - np.amin(tempInterval, axis=0)
#                                                              ], axis=0)))
#
#                 # Save Drs-Run Temperature
#                 drsTemp = np.mean(temp[drsRunIndices], axis=0)
#                 # cutoff the previous values of datetime and temp
#                 datetime = datetime[drsRunIndices[-1]+1:]
#                 temp = temp[drsRunIndices[-1]+1:]
#
#         if(drsTemp is not None):
#             # append values after last Drs-Run
#             drsTemp = np.mean(temp, axis=0)
#             maxTempDiffOfTheDay = np.append(maxTempDiffOfTheDay, np.array(
#                                             np.amax([
#                                                      np.amax(tempInterval, axis=0) - drsTemp,
#                                                      drsTemp - np.amin(tempInterval, axis=0)
#                                                      ], axis=0)))
#             # append data of the day
#             drsRunIdList.append(drsRunIdOfTheDay.astype("uint16"))
#             maxTempDiffList.append(maxTempDiffOfTheDay)
#
#     print("Write Data to Table")
#     tbhduTempDiff = fits.BinTableHDU.from_columns(
#             [fits.Column(name="date",        format="10A",
#                          unit="yyyy-mm-dd",  array=dateList),
#              fits.Column(name="drsRunId",    format="PB()",
#                          unit="1",           array=drsRunIdList),
#              fits.Column(name="maxTempDiff", format="PE()",
#                          unit="degree C",    array=maxTempDiffList)])
#     tbhduTempDiff.header.insert("TFIELDS", ("EXTNAME", "MaxDrsTempDiffs"), after=True)
#     commentStr = "Maximum of the Temperature difference between two following Drs-Runs"
#     tbhduTempDiff.header.insert("EXTNAME", ("comment", commentStr), after=True)
#
#     print("Save Table")
#     thdulist = fits.HDUList([fits.PrimaryHDU(), tbhduTempDiff])
#     thdulist.writeto(store_file_path, overwrite=True, checksum=True)
#
#     print("Verify Checksum")
#     # Open the File verifying the checksum values for all HDUs
#     try:
#         hdul = fits.open(store_file_path, checksum=True)
#         print(hdul["MaxDrsTempDiffs"].header)
#         print("Passed verifying Checksum")
#     except Exception as errInfos:
#         errorStr = str(errInfos)
#         print(errorStr)
#
#     print(">> Finished 'Temperature: MaxDifferencesPerPatch'")


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

        store_file_path = storePath_+"noise/pedestelNoise"+dateStr+".fits"  # _spikes
        print("Save Table")
        thdulist = fits.HDUList([fits.PrimaryHDU(), tbhduNoise])
        thdulist.writeto(store_file_path, clobber=True, checksum=True)
        print("Verify Checksum")
        # Open the File verifying the checksum values for all HDUs
        try:
            hdul = fits.open(store_file_path, checksum=True)
            print(hdul["NoisePerPixel"].header)
        except Exception as errInfos:
            errorStr = str(errInfos)
            print(errorStr)


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


def add_value_to_h5py_table(h5pyTable, columnName, value):  # TODO remove double declaration
    data = h5pyTable[columnName]
    data.resize((len(data)+1, data.maxshape[1]))
    data[len(data)-1, :] = value
