import drsCalibrationTool as tool
import pandas as pd
import numpy as np
import h5py
import sys
import os
import os.path
import logging
from astropy.io import fits
from tqdm import tqdm

NRPIX = 1440
NRCAP = 1024
NRTEMPSENSOR = 160


def add_creation_date(storeFilename_):
    creationDateStr = pd.datetime.now().strftime('%Y-%m-%d %H:%M:%S').encode("UTF-8", "ignore")
    with h5py.File(storeFilename_) as store:
        store['CreationDate'][0] = [creationDateStr]


def searchDrsFiles(storeFilename_, dbConfigFile_=None):
    '''
        1. search through the fact-database and find all drsFiles
        2. filter them, take just one drsFile per day
    '''
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

    engine = tool.factDb.getEngine(dbConfigFile_)

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

def saveDrsAttributes(drsFileList_, storeFilename_):
    '''
        save Baseline and Gain of all drsfiles of the drsFileList
        together with the mean of Time and Temperature of taking
        into a .h5 File
    '''
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

    nrFactValues = NRPIX*NRCAP

    def create_my_dataset(file, name, shape=None, maxshape=None, dtype=None):
        if maxshape is None:
            maxshape = tuple(x if x!=0 else None for x in shape)

        file.create_dataset(
            name,
            shape,
            dtype=dtype,
            maxshape=maxshape,
            compression="gzip",
            compression_opts=9,
            fletcher32=True,
        )

    with h5py.File(storeFilename_, 'w') as hf:
        create_my_dataset(hf, 'CreationDate',    (1, 1), dtype='S19')
        create_my_dataset(hf, "TimeBaseline",    (0, 1))
        create_my_dataset(hf, "TempBaseline",    (0, NRTEMPSENSOR))
        create_my_dataset(hf, "TempStdBaseline", (0, NRTEMPSENSOR))
        create_my_dataset(hf, "BaselineMean",    (0, nrFactValues))
        create_my_dataset(hf, "BaselineMeanStd", (0, nrFactValues))
        create_my_dataset(hf, "TimeGain",        (0, 1))
        create_my_dataset(hf, "TempGain",        (0, NRTEMPSENSOR))
        create_my_dataset(hf, "TempStdGain",     (0, NRTEMPSENSOR))
        create_my_dataset(hf, "GainMean",        (0, nrFactValues))
        create_my_dataset(hf, "GainMeanStd",     (0, nrFactValues))

    drsFileList_ = open(drsFileList_).read().splitlines()
    for drsFilename in tqdm(drsFileList):
        drsFilename = drsFilename.strip("\n")

        path_part = drsFilename.split('_')[0].split("raw")[-1]
        tempFilename = "/fact/aux{0}.FAD_CONTROL_TEMPERATURE.fits".format(
            path_part)

        if(os.path.isfile(drsFilename) and os.path.isfile(tempFilename)):
            saveTupleOfAttribute(tempFilename, drsFilename, storeFilename_)

    add_creation_date(storeFilename_)

    print(">> Finished 'SaveDrsAttributes' <<")


def saveTupleOfAttribute(tempFilename, drsFilename, storeFilename):
    try:
        saveTupleOfAttribute_no_try(tempFilename, drsFilename, storeFilename)
    except:
        logging.exception()
        return


def read_temps_of_runs(path, runtimeslist):
    '''
    runtimeslist a list() of tuples of (start, end) times
    between which we want to read the "Time" and "temp" arrays
    from the fits file under `path`.
    '''
    table = fits.open(
        path,
        ignoremissing=True,
        ignore_missing_end=True)

    table_time = table[1].data["Time"]
    table_temperature = table[1].data["temp"]

    if table_temperature.shape[1] != NRTEMPSENSOR:
        message = (
            "File not used: Just "+str(table_temperature.shape[1]) +
            " Temperature Values in File '"+path+"'")
        raise Exception(message)

    results = []
    table_datetime = pd.to_datetime(table_time * 24 * 3600 * 1e9)
    for start, end in runtimeslist:
        idx = np.where(
            (table_datetime > start) &
            (table_datetime < end)
            )[0]
        timestamps_during_run = np.array(table_time[idx])
        temperature_during_run = np.array(table_temperature[idx])

        if timestamps_during_run.shape[0] > 1:
            mean_time = np.mean(timestamps_during_run, dtype="float64")
        else:
            mean_time = timestamps_during_run

        if temperature_during_run.shape[0] > 1:
            mean_temp = np.mean(temperature_during_run, dtype="float64", axis=0)
            std_temp = np.std(temperature_during_run, dtype="float64", axis=0, ddof=1)
        else:
            mean_temp = temperature_during_run
            std_temp = np.zeros(temperature_during_run.shape[1])

        results.append(dict(
            mean_time=mean_time,
            mean_temp=mean_temp,
            std_temp=std_temp,
            )
        )
    return results


def check_for_nulls(array, name, path):
    nulls = np.where(array == 0.)[0]
    if len(nulls):
        raise Exception(
            ("File not used: Nulls of {name} \n" +
             "in File {path} at index {idx}").format(
                name=name,
                path=path,
                idx=str(nulls)
            )
        )


def saveTupleOfAttribute_no_try(tempFilename, drsFilename, storeFilename):

    tab_drs = fits.open(
        drsFilename,
        ignoremissing=True,
        ignore_missing_end=True)
    header = tabDrs[1].header
    bintable = tabDrs[1].data

    baselineMean = tabDrs[1].data["BaselineMean"][0]
    baselineMeanStd = tabDrs[1].data["BaselineRms"][0]
    gainMean = tabDrs[1].data["GainMean"][0]
    gainMeanStd = tabDrs[1].data["GainRms"][0]

    check_for_nulls(baselineMean, "baselineMean", drsFilename)
    check_for_nulls(baselineMeanStd, "baselineMeanStd", drsFilename)
    check_for_nulls(gainMean, "gainMean", drsFilename)
    check_for_nulls(gainMeanStd, "gainMeanStd", drsFilename)

    temps_of_runs = read_temps_of_runs(
        tempFilename,
        runtimeslist=[
            (
                pd.to_datetime(header["RUN0-BEG"]),
                pd.to_datetime(header["RUN0-END"])
            ),
            (
                pd.to_datetime(header["RUN1-BEG"]),
                pd.to_datetime(header["RUN1-END"])
            ),
        ])

    def my_store(store, name, what):
        data = store[name]
        data.resize((len(data)+1, data.maxshape[1]))
        data[len(data)-1, :] = what

    with h5py.File(storeFilename) as store:
        my_store(store, "TimeBaseline", temps_of_runs[0]['mean_time'])
        my_store(store, "TempBaseline", temps_of_runs[0]['mean_temp'])
        my_store(store, "TempStdBaseline", temps_of_runs[0]['std_temp'])
        my_store(store, "BaselineMean", baselineMean)
        my_store(store, "BaselineMeanStd", baselineMeanStd)
        my_store(store, "TimeGain", temps_of_runs[1]['mean_time'])
        my_store(store, "TempGain", temps_of_runs[1]['mean_temp'])
        my_store(store, "TempStdGain", temps_of_runs[1]['std_temp'])
        my_store(store, "GainMean", gainMean)
        my_store(store, "GainMeanStd", gainMeanStd)

####################################################################################################
####################################################################################################
# ##############                           saveFitValues                            ############## #
####################################################################################################
####################################################################################################

def saveFitValues(sourceFilename_, storeFilename_,
                  cutOffErrorFactorBaseline_, cutOffErrorFactorGain_,
                  firstDate_=None, lastDate_=None):
    '''
        Calculate the linear fitvalues of Basline and Gain of the .h5 source
        and store them into a .fits File
        All Basline/Gain-values with a bigger error than the 'CutOffErrorFactor'"
        multiplied with the mean of the error from all collected Baseline/Gain-values of the"
        Capacitor will not used for the fit
    '''

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

    print("Calculate fitvalues for '"+str(NRPIX)+"' Pixel \n" +
          "for the period from "+str(firstDate)+" until "+str(lastDate))

    for pixelNr in range(NRPIX):

        if(((pixelNr/NRPIX*100) % 1) < (((pixelNr-1)/NRPIX*100) % 1) and
           ((pixelNr/NRPIX*100) % 1) < (((pixelNr+1)/NRPIX*100) % 1)):
            print("PixelNr:", str('{:4d}'.format(pixelNr+1)), ":", '{:2d}'.format(int(pixelNr/NRPIX*100)), '%')

        tempBaseline = tempBaselineArray[:, int(pixelNr/9)]
        tempGain = tempGainArray[:, int(pixelNr/9)]
        for capNr in range(NRCAP):
            baselineMeanCap = baselineMeanArray[:, pixelNr*NRCAP+capNr]
            baselineMeanStdCap = baselineMeanStdArray[:, pixelNr*NRCAP+capNr]
            baselineMeanStdCapMean = np.mean(baselineMeanStdCap, dtype="float")
            gainMeanCap = gainMeanArray[:, pixelNr*NRCAP+capNr]
            gainMeanStdCap = gainMeanStdArray[:, pixelNr*NRCAP+capNr]
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
