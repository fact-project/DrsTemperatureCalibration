import writeSecondaryFiles
import plotDrsAttributes

# "/gpfs0/scratch/schulz/" -> "/scratch/schulz/"
# "/fact/ ->

# writeSecondaryFiles.temperatureMaxDifferencesPerPatch(
#     storeFilename_="/gpfs0/scratch/schulz/temperature/tempDiffs2014.fits",
#     isdcRootPath_="/",
#     startDate_="2014-01-01",
#     endDate_="2014-12-31",
#     freq_="D")

# writeSecondaryFiles.runFactTools(
#     factToolsPath_="/home/florian/Dokumente/Uni/Master/Masterarbeit/FACT/git_fact-tools/target/fact-tools-0.17.2.jar",
#     factToolsXmlPath_="/home/florian/Dokumente/Uni/Master/Masterarbeit/FACT/git_fact-tools/examples/studies/drsTemperatureCalibrationCheck.xml",
#     runFile_="/home/florian/Dokumente/Uni/Master/Masterarbeit/isdcRoot/gpfs0/fact/fact-archive/rev_1/raw/2016/10/19/20161019_015.fits.fz",
#     drsFile_="/home/florian/Dokumente/Uni/Master/Masterarbeit/isdcRoot/gpfs0/fact/fact-archive/rev_1/raw/2016/10/19/20161019_007.drs.fits.gz",
#     storeFilname_="/home/florian/Dokumente/Uni/Master/Masterarbeit/isdcRoot/gpfs0/scratch/schulz/noise/test_15.fits",
#     auxFolder_="/home/florian/Dokumente/Uni/Master/Masterarbeit/isdcRoot/gpfs0/fact/fact-archive/rev_1/aux/2016/10/19/",
#     fitValueDataFilename_="/home/florian/Dokumente/Uni/Master/Masterarbeit/isdcRoot/gpfs0/scratch/schulz/fitValues/fitValuesData_Interval1of1.fits",
#     intervalFitValueDataFilename_="/home/florian/Dokumente/Uni/Master/Masterarbeit/isdcRoot/gpfs0/scratch/schulz/fitValues/fitValuesData_Interval3of3.fits")

# writeSecondaryFiles.drsPedestalRunNoise(
#     isdcRootPath_="/home/florian/Dokumente/Uni/Master/Masterarbeit/isdcRoot/",
#     sourcePath_="/home/florian/Dokumente/Uni/Master/Masterarbeit/isdcRoot/gpfs0/scratch/schulz/",
#     storePath_="/home/florian/Dokumente/Uni/Master/Masterarbeit/isdcRoot/gpfs0/scratch/schulz/",
#     factPath_="/home/florian/Dokumente/Uni/Master/Masterarbeit/FACT/git_fact-tools/",
#     intervalFitValueDataFilename_="/home/florian/Dokumente/Uni/Master/Masterarbeit/isdcRoot/gpfs0/scratch/schulz/fitValues/fitValuesData_Interval3of3.fits",
#     startDate_="2016-09-01",
#     endDate_="2016-09-01",
#     freq_="D")

# Hardware changes cut: "2014-05-01", "2015-05-01" (no data taken between 2014-04-21 until 2014-05-22
#                                                             and between 2015-04-30 until 2015-05-06
#                                                                         2015-05-20 until 2015-05-26
#                                            missing data on isdc between 2014-11-01 until 2015-10-01

# Check Performances behavior for cuts on "2013-10-06" or "2013-12-18" Ints3_1/Ints3_2
# Performances cuts: "2014-08-10" (inaccurate cut iscd has not for every date data although this data exists)
# Performances cuts: "2015-10-27" (Also full moon, no data of surrounding days)

# "2015-08-08" (no data taken bad wether and ..)(Logbookentry: "power cut There was a power cut
# at around 22:20 UTC yesterday and none of the computers were reachable anymore.
# Since the weather was bad and the telescope was not operated tonight,
# I informed the MAGIC crew about the power cut and
# they restarted the computers at around 19:00 UTC today."

# ["2013-08-17", "2014-05-20", "2014-08-11", "2015-05-01", "2015-08-08",
# "2015-09-05", "2015-11-04", "2015-12-18", "2016-04-10"],

# writeSecondaryFiles.residumOfAllCapacitors(
#     drsFilename_="/gpfs0/scratch/schulz/drsData.h5",
#     fitFilname_="/gpfs0/scratch/schulz/fitValues/fitValuesData_Interval1of1.fits",
#     storeFilename_="/gpfs0/scratch/schulz/residuen/residuen_Interval1of1.h5")


# writeSecondaryFiles.getMaxTempDiff(
#    tempDiffFilename_="../../isdcRoot/gpfs0/scratch/schulz/temperature/tempDiffs2016.fits",
#    maxNr_=2)

# plotDrsAttributes.temperatureEraPerPatch(
#     patchNr_=100,
#     start_date_="2016-01-01",
#     end_date_="2016-12-31",
#     freq_="D",
#     storeFilename_="/gpfs0/scratch/schulz/temperature/temperatureEra2016_ofPatch100_sDate20161019.jpg")

# plotDrsAttributes.temperaturePerPatch(
#     patchNr_=100,
#     date_="2016-10-01",
#     storeFilename_="../plots/temperature/temperature_2016-10-01_ofPatch100.pdf")

#plotDrsAttributes.example()

# plotDrsAttributes.temperaturePatchMean(
#     start_date_="2016-01-01",
#     end_date_="2016-12-31",
#     freq_="D",
#     storeFilename_="../plots/temperature/temperaturePatchMean2016.pdf")

# plotDrsAttributes.temperatureMaxDifferencesPerPatch(
#     tempDiffFilename_="../../isdcRoot/gpfs0/scratch/schulz/temperature/tempDiffs2016.fits",
#     patchNr_=100,
#     nrBins_=40,
#     storeFilename_="../plots/temperature/temperatureMaxDifferencesPerPatch100.pdf")


# plotDrsAttributes.drsValueStdHist(
#     drsFilename_="/gpfs0/scratch/schulz/drsData.h5",
#     valueType_="Baseline", nrBins_=150,
#     storeFilename_="/gpfs0/scratch/schulz/baselineStdHist.pdf")

# plotDrsAttributes.drsValueStdHist(
#     drsFilename_="/gpfs0/scratch/schulz/drsData.h5",
#     valueType_="Gain", nrBins_=150,
#     storeFilename_="/gpfs0/scratch/schulz/gainStdHist.pdf")


# plotDrsAttributes.pixelCapacitorDrsValue_test(
#     drsFilename_="../../isdcRoot/gpfs0/scratch/schulz/drsData.h5",
#     valueType_="Baseline",
#     pixelNr_=1000, capNr_=500, errFactor_=2.0, showStd_=False,
#     subTimeInterval_=None,
#     storeFilename_="../plots/drsValues/fit_pix1000_cap500_baseline+trigger.jpg")

# plotDrsAttributes.pixelCapacitorDrsValueInPEA(
#     drsFilename_="../../isdcRoot/gpfs0/scratch/schulz/drsData.h5",
#     valueType_="Baseline",
#     pixelNr_=1000, capNr_=500, errFactor_=2.0, showStd_=False,
#     subTimeInterval_=None,
#     storeFilename_="../plots/drsValues/fit_pix1000_cap500_baseline.pdf")

# plotDrsAttributes.residuenPerPixelCapacitor(
#     drsFilename_="../../isdcRoot/gpfs0/scratch/schulz/drsData.h5",
#     residuenFilenameArray_=["../../isdcRoot/gpfs0/scratch/schulz/residuen/residuen_Interval1of1.h5"],
#     valueType_="Baseline",
#     pixelNr_=1000, capNr_=500,
#     restrictResiduen_=True,
#     storeFilename_="../plots/residuen/residuenBaseline_pix1000_cap500_Interval1of1.pdf")

# plotDrsAttributes.residuenPerPixelCapacitor(
#     drsFilename_="../../isdcRoot/gpfs0/scratch/schulz/drsData.h5",
#     residuenFilenameArray_=["../../isdcRoot/gpfs0/scratch/schulz/residuen/residuen_Interval1of3.h5",
#                             "../../isdcRoot/gpfs0/scratch/schulz/residuen/residuen_Interval2of3.h5",
#                             "../../isdcRoot/gpfs0/scratch/schulz/residuen/residuen_Interval3of3.h5"],
#     valueType_="Gain",
#     pixelNr_=1000, capNr_=500,
#     restrictResiduen_=True,
#     storeFilename_="../plots/residuen/residuenGain_pix1000_cap500_3Intervals_RestrictParts.pdf")

# plotDrsAttributes.residumMeanOfAllCapacitors(
#     drsFilename_="/gpfs0/scratch/schulz/drsData.h5",
#     residuenFilenameArray_=["../../isdcRoot/gpfs0/scratch/schulz/residuen/residuen_Interval1of3.h5",
#                             "../../isdcRoot/gpfs0/scratch/schulz/residuen/residuen_Interval2of3.h5",
#                             "../../isdcRoot/gpfs0/scratch/schulz/residuen/residuen_Interval3of3.h5"],
#     valueType_="Gain",
#     restrictResiduen_=True,
#     storeFilename_="/gpfs0/scratch/schulz/residuen/residuenMeanGain_3Intervals_RestrictParts.pdf")
#
# plotDrsAttributes.residumMeanOfAllCapacitors(
#     drsFilename_="/gpfs0/scratch/schulz/drsData.h5",
#     residuenFilenameArray_=["../../isdcRoot/gpfs0/scratch/schulz/residuen/residuen_Interval1of3.h5",
#                             "../../isdcRoot/gpfs0/scratch/schulz/residuen/residuen_Interval2of3.h5",
#                             "../../isdcRoot/gpfs0/scratch/schulz/residuen/residuen_Interval3of3.h5"],
#     valueType_="Baseline",
#     restrictResiduen_=True,
#     storeFilename_="/gpfs0/scratch/schulz/residuen/residuenMeanBaseline_3Intervals_RestrictParts.pdf")

# plotDrsAttributes.residumMeanPerPatchAndInterval(
#     drsFilename_="../../isdcRoot//gpfs0/scratch/schulz/drsData.h5",
#     residuenFilenameArray_=["../../isdcRoot//gpfs0/scratch/schulz/residuen/residuen_Interval1of3.h5",
#                             "../../isdcRoot//gpfs0/scratch/schulz/residuen/residuen_Interval2of3.h5",
#                             "../../isdcRoot//gpfs0/scratch/schulz/residuen/residuen_Interval3of3.h5"],
#     valueType_="Gain",
#     storeFilename_="/gpfs0/scratch/schulz/residumMeanPerPatchAndInterval_gain.pdf"
#     )

# plotDrsAttributes.residumMeanOfAllCapacitorsPerCrates(
#     drsFilename_="/gpfs0/scratch/schulz/drsData.h5",
#     residuenFilenameArray_=["/gpfs0/scratch/schulz/residuen/residuen_Interval1of1.h5"],
#     valueType_="Gain",
#     restrictResiduen_=True,
#     storeFilename_="/gpfs0/scratch/schulz/residuen/residuenMeanGainPerCrates_Interval1of1.pdf")

# plotDrsAttributes.residumMeanOfAllCapacitorsPerCrates(
#     drsFilename_="/gpfs0/scratch/schulz/drsData.h5",
#     residuenFilenameArray_=["/gpfs0/scratch/schulz/residuen/residuen_Interval1of3.h5",
#                             "/gpfs0/scratch/schulz/residuen/residuen_Interval2of3.h5",
#                             "/gpfs0/scratch/schulz/residuen/residuen_Interval3of3.h5"],
#     valueType_="Gain",
#     restrictResiduen_=True,
#     storeFilename_="/gpfs0/scratch/schulz/residuen/residuenGainPerCrates_pix1000_cap500_3Intervals_RestrictParts.pdf")

# plotDrsAttributes.noise(
#     drsFileCalibrated_=None,
#     drsModelCalibrated_=None,
#     drsModelIntervalCalibrated_=None,
#     titleStr_="Standard deviation 2017-01-01 \n"+r" runID: 66, temperature difference: ? $^\circ C$",
#     maxNoise_=18,
#     storeFilename_="../plots/noise/noise20170101_runId66.jpg",
#     sourceFile_=("/home/florian/Dokumente/Uni/Master/Masterarbeit/isdcRoot/" +
#                  "gpfs0/scratch/schulz/noise/test_15.fits"))
# #
# plotDrsAttributes.noiseFactCam(
#     drsFileCalibrated_=None,
#     drsModelCalibrated_=None,
#     drsModelIntervalCalibrated_=None,
#     storeFilename_="../plots/noise/noiseFactCam20170101_runId66",
#     sourceFile_=("/home/florian/Dokumente/Uni/Master/Masterarbeit/isdcRoot/" +
#                  "gpfs0/scratch/schulz/noise/test_15.fits"))

# plotDrsAttributes.pedestialNoise(
#     filename_=("/home/florian/Dokumente/Uni/Master/Masterarbeit/isdcRoot/" +
#                "gpfs0/scratch/schulz/noise/pedestelNoise20160901.fits"),
#     save_=True)

plotDrsAttributes.plot()
