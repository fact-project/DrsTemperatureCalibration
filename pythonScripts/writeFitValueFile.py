import drsFitTool as fitTool

# "/scratch/schulz/"->"/gpfs0/scratch/schulz/"

# 1. search through the fact-database and find all drsFiles
# 2. filter them, take just one drsFile per day
fitTool.searchDrsFiles(storeFilename_="/scratch/schulz/drsFiles.txt",
                       dbConfigFile_="../config/factDbConfig.yaml")

# save Baseline and Gain of all drsfiles of the drsFileList
# together with the mean of Time and Temperature of taking
# into a .h5 File
fitTool.saveDrsAttributes(drsFileList_="/scratch/schulz/drsFiles.txt",
                          storeFilename_="/scratch/schulz/drsData.h5")

# Calculate the linear fitvalues of Basline and Gain of the .h5 source
# and store them into a .fits File
# All Basline/Gain-values with a bigger error than the 'CutOffErrorFactor'"
# multiplied with the mean of the error from all collected Baseline/Gain-values of the"
# Capacitor will not used for the fit
fitTool.saveFitValues(sourceFilename_="/scratch/schulz/drsData.h5",
                      storeFilename_="/scratch/schulz/fitValues/fitValuesDataInterval.fits",
                      cutOffErrorFactorBaseline_=2,
                      cutOffErrorFactorGain_=2,
                      firstDate_=None,
                      lastDate_=None)
