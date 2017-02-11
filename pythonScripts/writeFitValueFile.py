import drsFitValueHandler as handler
import subprocess as sp

sp.Popen([handler.searchDrsFiles(storeFilename_="../data/drsFiles.txt")])
sp.Popen([handler.filterDrsFiles(sourceFilname_="../data/drsFiles.txt",
                                 storeFilename_="../data/selectedDrsFiles.txt")])
sp.Popen([handler.saveDrsAttributes(drsFilname_="../data/selectedDrsFiles.txt",
                                    storeFilename_="../data/drsData.h5")])
sp.Popen([handler.saveFitValues(sourceFilname_="../data/drsData.h5",
                                storeFilename_="../data/fitValuesData.fits")])
