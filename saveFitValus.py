import numpy as np
import h5py
import os
import sys

import logging

from astropy.io import fits


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


####################################################################################################
loadFilename = '/gpfs/scratch/schulz/drsData.h5'
storeFilename = '/gpfs/scratch/schulz/fitValuesData_.fits'

if(os.path.isfile(storeFilename)):
    user_input = input("File ’"+str(storeFilename)+"’ allready exist. Type ’y’ to overwrite file\nYour input: ")
    if(user_input != 'y'):
        sys.exit()

logging.basicConfig(filename=storeFilename.split('.')[0]+'.log', filemode='w',
                    format='%(levelname)s:%(message)s', level=logging.DEBUG)

cutOffErrorFactor_baseline = 2
cutOffErrorFactor_gain = 2

baselineMeanGradient = []
baselineMeanGradientStd = []
baselineMeanOffset = []
baselineMeanOffsetStd = []
gainMeanGradient = []
gainMeanGradientStd = []
gainMeanOffset = []
gainMeanOffsetStd = []

for pixelNr in range(1):

    if(((pixelNr/1440*100) % 1) < (((pixelNr-1)/1440*100) % 1) and
       ((pixelNr/1440*100) % 1) < (((pixelNr+1)/1440*100) % 1)):
        print(pixelNr, ": ", int(pixelNr/1440*100), '%')

    with h5py.File(loadFilename, 'r') as dataSource:

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

        var_base = []
        cov_base = []
        try:
            indices = [baselineMeanStd_cond < baselineMeanStd_cond_mean*cutOffErrorFactor_baseline]
            var_base, cov_base = getLinearFitValues(temp_baseline[indices],
                                                    baselineMean_cond[indices],
                                                    baselineMeanStd_cond[indices])

            baselineMeanGradient.append(var_base[0])
            baselineMeanGradientStd.append(np.sqrt(cov_base[0][0]))
            baselineMeanOffset.append(var_base[1])
            baselineMeanOffsetStd.append(np.sqrt(cov_base[1][1]))

        except Exception as errInfos:
            errorStr = ("Gain-Fit(PixelNr: "+str('{:04d}'.format(pixelNr))+", condNr: "+str('{:04d}'.format(condNr)) +
                        ") with Gradient="+str(var_base[0])+" and Offset='"+str(var_base[1])+": '"+str(errInfos)+"'")
            logging.warning(errorStr)

            baselineMeanGradient.append(var_base[0])
            baselineMeanGradientStd.append(0)
            baselineMeanOffset.append(var_base[1])
            baselineMeanOffsetStd.append(0)

        try:
            indices = [gainMeanStd_cond < gainMeanStd_cond_mean*cutOffErrorFactor_gain]
            var_gain, cov_gain = getLinearFitValues(temp_gain[indices],
                                                    gainMean_cond[indices],
                                                    gainMeanStd_cond[indices])

            gainMeanGradient.append(var_gain[0])
            gainMeanGradientStd.append(np.sqrt(cov_gain[0][0]))
            gainMeanOffset.append(var_gain[1])
            gainMeanOffsetStd.append(np.sqrt(cov_gain[1][1]))

        except Exception as errInfos:
            errorStr = ("Gain-Fit(PixelNr: "+str('{:04d}'.format(pixelNr))+", condNr: "+str('{:04d}'.format(condNr)) +
                        ") with Gradient="+str(var_gain[0])+" and Offset='"+str(var_gain[1])+": '"+str(errInfos)+"'")
            logging.warning(errorStr)

            gainMeanGradient.append(var_gain[0])
            gainMeanGradientStd.append(0)
            gainMeanOffset.append(var_gain[1])
            gainMeanOffsetStd.append(0)

baselineMeanGradient = np.array([baselineMeanGradient])
baselineMeanGradientStd = np.array([baselineMeanGradientStd])
baselineMeanOffset = np.array([baselineMeanOffset])
baselineMeanOffsetStd = np.array([baselineMeanOffsetStd])
gainMeanGradient = np.array([gainMeanGradient])
gainMeanGradientStd = np.array([gainMeanGradientStd])
gainMeanOffset = np.array([gainMeanOffset])
gainMeanOffsetStd = np.array([gainMeanOffsetStd])

print("Write Data to Table")
# TODO maybe add the checksum and creation date of the drsData.h5 source-file to the table header
tbhdu_baseline = fits.BinTableHDU.from_columns(
        [fits.Column(name='gradient',    format=str(baselineMeanGradient.shape[1])+'E',
                     unit='mV/celsius',  array=baselineMeanGradient),
         fits.Column(name='gradientStd', format=str(baselineMeanGradientStd.shape[1])+'E',
                     unit='mV/celsius',  array=baselineMeanGradientStd),
         fits.Column(name='offset',      format=str(baselineMeanOffset.shape[1])+'E',
                     unit='mV',          array=baselineMeanOffset),
         fits.Column(name='offsetStd',   format=str(baselineMeanOffsetStd.shape[1])+'E',
                     unit='mV',          array=baselineMeanOffsetStd)])
tbhdu_baseline.update_ext_name("Baseline")
tbhdu_baseline.header.set("CutOffErrorFactor", str(cutOffErrorFactor_baseline),
                          "All baseline-values with a bigger error than the 'CutOffErrorFactor'" +
                          "multiplied with the mean of the error from all collected baseline-values" +
                          "it will not used for the fit")

tbhdu_gain = fits.BinTableHDU.from_columns(
        [fits.Column(name='gradient',    format=str(gainMeanGradient.shape[1])+'E',
                     unit='mV/celsius',  array=gainMeanGradient),
         fits.Column(name='gradientStd', format=str(gainMeanGradientStd.shape[1])+'E',
                     unit='mV/celsius',  array=gainMeanGradientStd),
         fits.Column(name='offset',      format=str(gainMeanOffset.shape[1])+'E',
                     unit='mV',          array=gainMeanOffset),
         fits.Column(name='offsetStd',   format=str(gainMeanOffsetStd.shape[1])+'E',
                     unit='mV',          array=gainMeanOffsetStd)])
tbhdu_gain.update_ext_name("Gain")
tbhdu_gain.header.set("CutOffErrorFactor", str(cutOffErrorFactor_gain),
                      "All gain-values with a bigger error than the 'CutOffErrorFactor'" +
                      "multiplied with the mean of the error from all collected gain-values" +
                      "it will not used for the fit")

print("Save Table")
thdulist = fits.HDUList([fits.PrimaryHDU(), tbhdu_baseline, tbhdu_gain])
thdulist.writeto(storeFilename, clobber=True, checksum=True)

print("Verify Checksum")
# Open the File verifying the checksum values for all HDUs
try:
    hdul = fits.open(storeFilename, checksum=True)
    print(hdul["Baseline"].header)
    print(hdul["Gain"].header)
    with open(storeFilename.split('.')[0]+'.log', 'r') as logFile:
        if(logFile.readlines() == []):
            logging.info(" No errors occurred during the Fit-Value calculation.")
    logging.info(" Passed verifying Checksum")
except Exception as errInfos:
    errorStr = str(errInfos)
    print(errorStr)
    logging.warning(errorStr)

print("Done")
