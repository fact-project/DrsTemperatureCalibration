import numpy as np
import pandas as pd
import os

from astropy.io import fits

####################################################################################################
####################################################################################################
# path to your local mounted isdc-in04-Folder (isdc-in04:/fact)
isdcPath = "/home/florian/Dokumente/Uni/Master/Masterarbeit/isdcFact/"
####################################################################################################
####################################################################################################


####################################################################################################
def maxPatchTempDrifts(startDate_="2016-10-01", endDate_="2016-10-03", freq_="D", save_=False):

    storeFilename = "../data/tempDiffs"+startDate_+"-"+endDate_+".fits"

    dates = []
    drsRunNr = []
    maxTempDiff = []

    month_before = 0
    for date in pd.date_range(start=startDate_, end=endDate_, freq=freq_):
        if(month_before < date.month):
            month_before = date.month
            print("Month: ", date.month)
        print("Date: ", date)

        dateStr = str(date.year)+"-"+str('{:02d}'.format(date.month))+"-"+str('{:02d}'.format(date.day))
        filename = (isdcPath+"aux/" +
                    str(date.year)+"/"+str('{:02d}'.format(date.month))+"/"+str('{:02d}'.format(date.day)) +
                    "/"+str(date.year)+str('{:02d}'.format(date.month))+str('{:02d}'.format(date.day)) +
                    ".FAD_CONTROL_TEMPERATURE.fits")

        folder = isdcPath+"raw/"+str(date.year)+"/"+str('{:02d}'.format(date.month))+"/"+str('{:02d}'.format(date.day))
        if(os.path.isfile(filename) and os.path.isdir(folder)):
            # print("found: ", filename, "and ", folder)
            with fits.open(filename) as tabTemp:
                time = tabTemp[1].data['Time']
                datetime = pd.to_datetime(time * 24 * 3600 * 1e9)
                temp = tabTemp[1].data['temp']

            intNr = 0
            drsRunNrList = []
            maxTempDiffList = []
            for filename in sorted(os.listdir(folder)):
                if filename.endswith("drs.fits.gz"):
                    # print("found: ", filename)
                    with fits.open(folder+"/"+filename) as tab_drs:
                        drsRunStart = pd.to_datetime(tab_drs[1].header["DATE-OBS"])
                        drsRunEnd = pd.to_datetime(tab_drs[1].header["DATE-END"])
                        runNr = tab_drs[1].header["STEP"]
                        roi = tab_drs[1].header["NROI"]
                        nrEvents0 = tab_drs[1].header["NBOFFSET"]
                        nrEvents1 = tab_drs[1].header["NBGAIN "]
                        nrEvents2 = tab_drs[1].header["NBTRGOFF"]

                    if(runNr == 2 and roi == 300 and nrEvents0 == nrEvents1 == nrEvents2 == 1000):
                        # print("Use File: ", filename)
                        drsRunNrList.append(int(filename.split(".")[0].split("_")[1]))
                        drsRunIndices = np.where((datetime > drsRunStart) & (datetime < drsRunEnd))[0]
                        if(intNr == 0):
                            intNr += 1
                            # cutoff the previous values of datetime and temp
                            # (before the first Drs-Run)
                            datetime = datetime[drsRunIndices[-1]+1:]
                            temp = temp[drsRunIndices[-1]+1:]

                            dates.append(dateStr)
                            continue

                        drsTemp = np.mean(temp[drsRunIndices], axis=0)
                        tempInterval = temp[0:drsRunIndices[0]]
                        maxTempDiffList.extend(np.amax([
                                                np.amax(tempInterval, axis=0) - drsTemp,
                                                drsTemp - np.amin(tempInterval, axis=0)
                                                ], axis=0))

                        # cut datetime and temp length
                        datetime = datetime[drsRunIndices[-1]+1:]
                        temp = temp[drsRunIndices[-1]+1:]

            # append values after last Drs-Run
            drsTemp = np.mean(temp, axis=0)
            maxTempDiffList.extend(np.amax([
                                    np.amax(temp, axis=0) - drsTemp,
                                    drsTemp - np.amin(temp, axis=0)
                                    ], axis=0))

        if(drsRunNrList != [] and maxTempDiffList != []):
            drsRunNr.append(drsRunNrList)
            maxTempDiff.append(maxTempDiffList)

    dates = np.array(dates)
    drsRunNr = np.array(drsRunNr)
    maxTempDiff = np.array(maxTempDiff)

    print(dates)
    print(drsRunNr)
    print(maxTempDiff)

    # print(np.array(maxTempDiff[np.where(dates == "2016-12-22")[0]])[interval/drsNr][patchNR])
    print("Write Data to Table")
    tbhduTempDiff = fits.BinTableHDU.from_columns(
            [fits.Column(name='date',        format=str(len(dates[0])*dates.shape[0])+'A',
                         unit='yyyy-mm-dd',  array=dates),
             fits.Column(name='drsRunNr',    format='PB()',
                         unit='1',           array=drsRunNr),
             fits.Column(name='maxTempDiff', format='PE()',
                         unit='degree C',    array=maxTempDiff)])
    tbhduTempDiff.header.insert("TFIELDS", ("EXTNAME", "MaxDrsTempDiffs"), after=True)
    # tbhduTempDiff.header.insert("EXTNAME", ("comment",
    #                                  "some text", after=True)

    print("Save Table")
    thdulist = fits.HDUList([fits.PrimaryHDU(), tbhduTempDiff])
    thdulist.writeto(storeFilename, clobber=True, checksum=True)  # clobber/overwrite=True

    print("Verify Checksum")
    # Open the File verifying the checksum values for all HDUs
    try:
        hdul = fits.open(storeFilename, checksum=True)
        print(hdul["MaxDrsTempDiffs"].header)
        print("Passed verifying Checksum")
    except Exception as errInfos:
        errorStr = str(errInfos)
        print(errorStr)

    print("Done")
