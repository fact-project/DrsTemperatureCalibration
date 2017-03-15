import drsCalibrationTool as tool
import matplotlib.pyplot as plt

import matplotlib.dates as dates

import numpy as np
import pandas as pd
import math
import h5py
import os

import pathHandler as path

from fact.pixels import non_standard_pixel_chids as nonStandardPixel
from fact.plotting import camera
from astropy.io import fits
from matplotlib import gridspec
from matplotlib import patches
from matplotlib.cm import hot
from collections import namedtuple


def example():
    storeFilename_ = "../plots/temperature/temperatureExampleTrend.pdf"
    date = pd.to_datetime("20000101")
    startDate = date+pd.DateOffset(hours=12)
    endDate = date+pd.DateOffset(hours=14)
    datetime = pd.date_range(start=startDate, end=endDate, freq="S")
    x = np.linspace(0, 1, len(datetime))

    y = 2*np.cos(x*np.pi)+17.5
    yIndives = [np.where((y >= 18.999) & (y <= 19.001))[0][0], np.where((y >= 15.999) & (y <= 16.001))[0][0]]
    y2 = 4*np.cos(x*np.pi+3*np.pi/2)+15.5
    y2Indives = [np.where((y2 >= 16.999) & (y2 <= 17.001))[0][0], np.where((y2 >= 17.999) & (y2 <= 18.001))[0][-1]]

    plt.figure(1)
    plt.subplot(211)
    plt.title("Exemplary temperature trend", fontsize=18, y=1.02)
    plt.plot(datetime, y, 'g.')
    plt.plot(datetime[yIndives], y[yIndives], "r|", ms=15)
    plt.plot([datetime[0], datetime[-1]], [19, 19], 'k-')
    plt.plot([datetime[0], datetime[-1]], [16, 16], 'k-')
    plt.text(endDate-pd.DateOffset(minutes=20), 19, r"$\Delta T = 2 ^\circ C$",
             fontdict={'family': 'serif',
                       'color':  'black',
                       'weight': 'bold',
                       'size': 14,
                       },
             bbox=dict(boxstyle="round", facecolor="grey", ec="k"))
    plt.ylabel(r'Temperature /$\mathrm{^\circ C}$')
    plt.grid()
    plt.ylim(15, 20)
    plt.gca().xaxis.set_major_formatter(dates.DateFormatter("%H:%M"))

    plt.subplot(212)
    plt.plot(datetime, y2, 'g.')
    plt.plot(datetime[y2Indives], y2[y2Indives], "r|", ms=18)
    plt.plot([datetime[0], datetime[-1]], [17, 17], 'k-')
    plt.plot([datetime[0], datetime[-1]], [max(y2), max(y2)], 'k-')
    plt.text(endDate-pd.DateOffset(minutes=20), 19, r"$\Delta T = 2.5 ^\circ C$",
             fontdict={'family': 'serif',
                       'color':  'black',
                       'weight': 'bold',
                       'size': 14,
                       },
             bbox=dict(boxstyle="round", facecolor="grey", ec="k"))
    plt.ylabel(r'Temperature /$\mathrm{^\circ C}$')
    plt.grid()
    plt.ylim(14, 20)
    plt.gca().xaxis.set_major_formatter(dates.DateFormatter("%H:%M"))

    plt.plot([], [], "g.", ms=15, label="Temperature measurement point")
    plt.plot([], [], "r|", ms=15, label="DRS-Runs")
    plt.xlabel("Hour"+" /$h$")
    plt.legend(bbox_to_anchor=(0.01, 0.1, 0.98, .102), ncol=2,
               mode="expand", borderaxespad=0., scatterpoints=1, numpoints=1)
    if(storeFilename_ is not None):
        plt.savefig(storeFilename_)
    plt.show()
    plt.close()


def plot():
    table = fits.open("/home/florian/Dokumente/Uni/Master/Masterarbeit/isdcRoot/gpfs0/scratch/schulz/noise/data20160901_008_timeRow.fits")

    eventNr = 200
    pixelNr = 250

    mV2Photon = 0.1
    x = np.linspace(1, 300, 300, dtype="int")
    # Data, DRSCalibratedData, DRSCalibratedData_TempInterval
    a = np.array(table[1].data["Data"][eventNr]).reshape(1440, 300)
    b = np.array(table[1].data["DRSCalibratedData"][eventNr]).reshape(1440, 300)
    c = np.array(table[1].data["DRSCalibratedData_TempInterval"][eventNr]).reshape(1440, 300)

    timeRowData = a[pixelNr]*mV2Photon
    timeRowData = (timeRowData-np.mean(timeRowData))
    timeRowDataStd = np.std(timeRowData, ddof=1)
    timeRowFile = b[pixelNr]*mV2Photon
    timeRowFileStd = np.std(timeRowFile[10:250+1], ddof=1)
    timeRowTempInterval = c[pixelNr]*mV2Photon
    timeRowTempIntervalStd = np.std(timeRowTempInterval[10:250+1], ddof=1)

    braun = (139/255.0, 69/255.0, 19/255.0, 1)
    #plt.plot(x, timeRowData, color=braun)
    plt.plot(x, timeRowFile, color=braun, lw=2, label="Drs-File \nStd: "+str(round(timeRowFileStd, 2)))
    #plt.plot(x, timeRowTempInterval, "b", lw=2, label="Model \nStd: "+str(round(timeRowTempIntervalStd, 2)))
    #plt.annotate(s='', xy=(10, 1.8), xytext=(250, 1.8),
    #             arrowprops=dict(arrowstyle='<->', lw=2, color="k"))
    #plt.legend(loc="lower center", scatterpoints=1, numpoints=1, fontsize=18)#, boxstyle="fancy")
    #plt.axvline(x=10, lw=2, color="k")
    #plt.axvline(x=250, lw=2, color="k")
    plt.ylim(-8, 8)
    plt.xlabel("Slices")
    plt.ylabel(r"Noise /$\mathrm{PEA}$")
    plt.text(200, 6.5, r"Std: "+str(round(np.std(timeRowFile, ddof=1), 2))+r"$\,\mathrm{PEA}$ ",
             fontdict={'family': 'serif',
                       'color':  'black',
                       'weight': 'bold',
                       'size': 18,
                       },
             bbox=dict(boxstyle="round", facecolor="white", ec="k"),
             multialignment="center")
    # calibratedPedestelRun20160901_008_E200_P250
    # pedestelRunRawData20160901_008_E200_P250
    plt.savefig("../plots/noise/calibratedPedestelRun20160901_008_E200_P250.jpg")
    plt.show()
    plt.close()


####################################################################################################
# ############## ##############                Helper                ############## ############## #
####################################################################################################
font = {'family': 'serif',
        'color':  'grey',
        'weight': 'bold',
        'size': 16,
        'alpha': 0.5,
        }


####################################################################################################
Constants = namedtuple("Constants", ["nrPix", "nrCap", "nrPatch"])
fact = Constants(nrPix=1440, nrCap=1024, nrPatch=160)


####################################################################################################
def linearerFit(x, m, b):
    return (m*x+b)


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
def getDateString(date1_, date2_, sep_=["-", " "]):
    if (date1_.year != date2_.year):
        return date1_.strftime('%d.%m.%Y')+sep_[0]+date2_.strftime('%d.%m.%Y')
    elif (date1_.month != date2_.month):
        return date1_.strftime('%d'+sep_[1]+'%b')+sep_[0]+date2_.strftime('%d'+sep_[1]+'%b'+sep_[1]+'%Y')
    elif (date1_.day != date2_.day):
        return date1_.strftime('%d')+sep_[0]+date2_.strftime('%d'+sep_[1]+'%b'+sep_[1]+'%Y')


####################################################################################################
# ############## ##############          Temperature  Plots          ############## ############## #
####################################################################################################
# TODO update selected date
def temperatureEraPerPatch(patchNr_, start_date_, end_date_, freq_="D", storeFilename_=None):

    time = np.array([])
    temp = np.array([])

    month_before = 0
    for date in pd.date_range(start=start_date_, end=end_date_, freq=freq_):
        if(month_before < date.month):
            month_before = date.month
            print("Month: ", date.month)

        filename = (path.getIsdcRootPath()+"/gpfs0/fact/fact-archive/rev_1/aux/" +
                    str(date.year)+"/"+str('{:02d}'.format(date.month))+"/"+str('{:02d}'.format(date.day)) +
                    "/"+str(date.year)+str('{:02d}'.format(date.month))+str('{:02d}'.format(date.day)) +
                    ".FAD_CONTROL_TEMPERATURE.fits")
        if(os.path.isfile(filename)):
            # print("found: ", filename)
            with fits.open(filename) as tab_temp:
                time = np.append(time, tab_temp[1].data['Time'])
                temp = np.append(temp, tab_temp[1].data['temp'][::, patchNr_-1])

    datetime = pd.to_datetime(time * 24 * 3600 * 1e9)

    # mark selected Date with ellipse
    if(True):
        selectedDatestr_ = "20161001"
        selectedDatetime = pd.to_datetime(selectedDatestr_)
        selectedDatetimeInterval = [selectedDatetime+pd.DateOffset(hours=12),
                                    selectedDatetime+pd.DateOffset(hours=12)+pd.DateOffset(days=1)]

        selectedDateIndices = np.where((datetime > selectedDatetimeInterval[0]) &
                                       (datetime < selectedDatetimeInterval[1]))[0]

        startDate = dates.date2num(selectedDatetimeInterval[0])
        endDate = dates.date2num(selectedDatetimeInterval[-1])

        ySelected = temp[selectedDateIndices]

        width = endDate-startDate
        height = max(ySelected)-min(ySelected)

        rectangle = patches.Rectangle((startDate, min(ySelected)-1), width, height+1,
                                      facecolor="None", linewidth=2., zorder=10)

        plt.gca().add_patch(rectangle)
        plt.text(selectedDatetime, max(ySelected)+1.5, selectedDatetime.strftime('%Y-%m-%d'),
                 fontdict={'family': 'serif',
                           'color':  'black',
                           'weight': 'bold',
                           'size': 18,
                           },
                 bbox=dict(boxstyle="round", facecolor="white", ec="k"),
                 multialignment="center")

    #plt.title("Temperature trend of Patch: "+str(patchNr_), fontsize=18, y=1.02)
    plt.plot(datetime, temp, "g.", ms=8, label="Temperature measurement point")
    timeLabel = pd.date_range(start=start_date_, end=end_date_, freq="M") - pd.offsets.MonthBegin(1)
    plt.xticks(timeLabel, timeLabel, rotation=30)
    plt.gca().xaxis.set_major_formatter(dates.DateFormatter("%d.%m.%y"))
    plt.ylabel(r'Temperature /$\mathrm{^\circ C}$')
    plt.grid()
    plt.ylim(8, 45)#
    plt.legend(bbox_to_anchor=(0.01, -0.002, 0.98, .102), ncol=1,
               mode="expand", borderaxespad=0., scatterpoints=1, numpoints=1)
    if(storeFilename_ is not None):
        print(storeFilename_)
        plt.savefig(storeFilename_)
    plt.show()
    plt.close()


# TODO update ->find drs per database
####################################################################################################
def temperaturePerPatch(patchNr_, date_, storeFilename_=None, dbConfigFile_=None):

    date = pd.to_datetime(date_)

    dateStr = date.strftime('%Y%m%d')
    datePath = date.strftime('%Y/%m/%d/')

    time = np.array([])
    temp = np.array([])

    filename = (path.getIsdcRootPath()+"/gpfs0/fact/fact-archive/rev_1/aux/" +
                datePath+dateStr+".FAD_CONTROL_TEMPERATURE.fits")
    if(os.path.isfile(filename)):
        # print("found: ", filename)
        with fits.open(filename) as tab_temp:
            time = np.append(time, tab_temp[1].data['Time'])
            temp = np.append(temp, tab_temp[1].data['temp'][::, patchNr_-1])

        datetime = pd.to_datetime(time * 24 * 3600 * 1e9)
        dateStrLabel = getDateString(datetime[0], datetime[-1])

        #plt.title("Temperature trend of Patch: "+str(patchNr_), fontsize=18, y=1.02)
        plt.plot(datetime, temp, "b.", ms=15, label="Temperature measurement point")

        print("Loading Database ...")
        engine = tool.factDb.getEngine(dbConfigFile_)
        dbTable = pd.read_sql("RunInfo", engine, columns=["fNight", "fRunID",
                                                          "fRunTypeKey", "fDrsStep",
                                                          "fNumEvents"])
        selectedDrsInfos = dbTable.query("fNight =="+str(dateStr)+"&" +
                                         "fRunTypeKey == 2 & fDrsStep == 2 & fNumEvents == 1000").copy()

        drsRunIdList = selectedDrsInfos["fRunID"].tolist()

        for drsRunId in drsRunIdList:
            drsFiles = (path.getIsdcRootPath()+"/gpfs0/fact/fact-archive/rev_1/raw/" +
                        datePath+dateStr+"_"+str("{:03d}".format(drsRunId))+".drs.fits.gz")
            with fits.open(drsFiles) as tab_drs:
                drsStart = pd.to_datetime(tab_drs[1].header["DATE-OBS"])
                drsEnd = pd.to_datetime(tab_drs[1].header["DATE-END"])
                tempMean = np.mean(temp[np.where((datetime > drsStart) & (datetime < drsEnd))])
                plt.plot([drsStart, drsEnd], [tempMean, tempMean], linestyle="--", marker="|", color="r", lw=40, ms=40)

        plt.plot([], [], "r|", linestyle="-", color="r", lw=15, label="DRS-Runs")

        # timeLabel = pd.date_range(start=datetime[0], end=datetime[-1], freq="H")
        # plt.xticks(timeLabel, timeLabel, rotation=30)
        plt.gca().xaxis.set_major_formatter(dates.DateFormatter("%H:%M"))
        plt.ylim(26.5, 29)#
        plt.xlabel(dateStrLabel+r" /$\mathrm{h}$")
        plt.ylabel(r'Temperature /$\mathrm{^\circ C}$')
        plt.grid()
        plt.legend(bbox_to_anchor=(0.01, -0.002, 0.98, .102), ncol=2,
                   mode="expand", borderaxespad=0., scatterpoints=1, numpoints=1)
        if(storeFilename_ is not None):
            plt.savefig(storeFilename_)
        plt.show()
        plt.close()
    else:
        print("File '", filename, "' or Folder '", folder, "' does not exist")


####################################################################################################
def temperaturePatchMean(start_date_, end_date_, freq_="D", storeFilename_=None):

    # One Temperature sensor per Patch (nrTempSenor == nrPatch)
    temp = np.array([]).reshape(0, fact.nrPatch)
    for date in pd.date_range(start=start_date_, end=end_date_, freq=freq_):
        filename = (path.getIsdcRootPath()+"/gpfs0/fact/fact-archive/rev_1/aux/" +
                    str(date.year)+"/"+str('{:02d}'.format(date.month))+"/"+str('{:02d}'.format(date.day)) +
                    "/"+str(date.year)+str('{:02d}'.format(date.month))+str('{:02d}'.format(date.day)) +
                    ".FAD_CONTROL_TEMPERATURE.fits")
        if(os.path.isfile(filename)):
            # print("found: ", filename)
            with fits.open(filename, ignoremissing=True, ignore_missing_end=True) as tab_temp:
                tempValue = tab_temp[1].data['temp']
                nanIndices = np.array(np.where(np.isnan(tempValue)))
                if(nanIndices != []):
                    print(filename, "contains nans with index: ", nanIndices)
                    tempValue = np.delete(tempValue, nanIndices[0], axis=0)
                temp = np.append(temp, tempValue, axis=0)

    tempMean_list = np.mean(temp, dtype='float64', axis=0)
    tempMeanStd_list = np.std(temp, dtype='float64', axis=0, ddof=1)

    width = 1   # the width of the bars
    xRange = np.linspace(0, fact.nrPatch-1, fact.nrPatch)
    dateStr = getDateString(pd.to_datetime(start_date_), pd.to_datetime(end_date_), [" until ", " "])
    plt.title("Mean Patch temperature\n "+dateStr, fontsize=16, y=1.00)
    plt.bar(xRange-width/2, tempMean_list, width=width, color='r', yerr=tempMeanStd_list)
    plt.xlabel('Patch Nr')
    plt.ylabel(r'Temperature /$\mathrm{^\circ C}$')
    plt.xlim(-1, fact.nrPatch)
    if(storeFilename_ is not None):
        plt.savefig(storeFilename_)
    plt.show()
    plt.close()


####################################################################################################
def temperatureMaxDifferencesPerPatch(tempDiffFilename_, patchNr_, nrBins_, storeFilename_=None):

    # One Temperature sensor per Patch (nrTempSenor == nrPatch)
    with fits.open(tempDiffFilename_) as store:
        tempDiffs = np.concatenate(store[1].data["maxTempDiff"]).ravel().reshape(-1, fact.nrPatch)[::, patchNr_-1]
        print(len(tempDiffs))

    weights = np.full(len(tempDiffs), 100/len(tempDiffs))
    plt.hist(tempDiffs, weights=weights, bins=nrBins_, range=(0, max(tempDiffs)))
    #plt.title(r"Histogram of the maximal temperature differences $\Delta t$\n"+" of Patch "+str(patchNr_), fontsize=16, y=1.0)
    plt.title(r"Histogram of the maximal temperature differences $\Delta t$", fontsize=18, y=1.05)
    plt.xlabel(r'Maximal temperature differences $\Delta t  /^\circ C$')
    plt.ylabel(r'Frequency /$\mathrm{\%}$')
    if(storeFilename_ is not None):
        plt.savefig(storeFilename_)
    plt.show()
    plt.close()


####################################################################################################
# def temperatureMaxDifferencesPerPatch(tempDiffFilename_, patchNr_, nrBins_, storeFilename_=None, dbConfigFile_=""):
#
#     engine = tool.factDb.getEngine(dbConfigFile_)
#     dbTable = pd.read_sql("RunInfo", engine, columns=["fNight", "fRunID",
#                                                       "fRunTypeKey", "fDrsStep",
#                                                       "fNumEvents"])
#
#     tempDiffs
#     weights = np.full(len(tempDiffs), 100/len(tempDiffs))
#     plt.hist(tempDiffs, weights=weights, bins=nrBins_, range=(0, max(tempDiffs)))
#     #plt.title(r"Histogram of the maximal temperature differences $\Delta t$\n"+" of Patch "+str(patchNr_), fontsize=16, y=1.0)
#     plt.title(r"Histogram of the maximal temperature differences $\Delta t$", fontsize=18, y=1.05)
#     plt.xlabel(r'Maximal temperature differences $\Delta t  /^\circ C$')
#     plt.ylabel(r'Frequency /$\mathrm{\%}$')
#     if(storeFilename_ is not None):
#         plt.savefig(storeFilename_)
#     plt.show()
#     plt.close()

####################################################################################################
# ############## ##############           Drs-Value  Plots           ############## ############## #
####################################################################################################
def drsValueStdHist(drsFilename_, valueType_, nrBins_=150, storeFilename_=None):

    print("Loading '"+valueType_+"-data' ...")
    with h5py.File(drsFilename_, 'r') as store:
        drsValueMeanStd = np.array(store[valueType_+"MeanStd"]).flatten()

    print(np.mean(drsValueMeanStd))
    return
    weights = np.full(len(drsValueMeanStd), 100/len(drsValueMeanStd))
    hist = plt.hist(drsValueMeanStd, weights=weights, bins=nrBins_, range=(0, max(drsValueMeanStd)))
    print(np.where(hist[1] > 0.1), hist(max(np.where(hist[1] > 0.1)[0])))
    plt.title(r"Histogram of "+valueType_+"Std", fontsize=16, y=1.01)
    plt.xlabel(r'Std /$\mathrm{mv}$')
    plt.ylabel(r'Frequency /$\mathrm{\%}$')
    plt.xlim(0)
    if(storeFilename_ is not None):
        plt.savefig(storeFilename_)
    plt.show()
    plt.close()




####################################################################################################
def pixelCapacitorDrsValue_test(drsFilename_, valueType_, pixelNr_, capNr_, errFactor_=2.0, showStd_=False,
                                subTimeInterval_=None, storeFilename_=None):

    with h5py.File(drsFilename_, 'r') as store:
        time = np.array(store["Time"+valueType_]).flatten()
        temp = np.array(store["Temp"+valueType_][:, int(pixelNr_/9)])
        tempStd = np.array(store["TempStd"+valueType_][:, int(pixelNr_/9)])
        drsValue = np.array(store[valueType_+"Mean"][:, pixelNr_*fact.nrCap + capNr_])
        drsValueStd = np.array(store[valueType_+"MeanStd"][:, pixelNr_*fact.nrCap + capNr_])

        trigger = np.array(store["TriggerOffsetMean"][:, pixelNr_*fact.nrCap + capNr_])
        drsValue = drsValue+trigger

    drsValueStdMean = np.mean(drsValueStd, dtype='float64')
    indices = np.where(drsValueStd < drsValueStdMean*errFactor_)[0]
    indices_ = np.where(drsValueStd > drsValueStdMean*errFactor_)[0]

    sc_all = plt.scatter(temp, drsValue, c=time)
    plt.close()  # Dont show sc_all, just use it to create the colorbar

    fig, img = plt.subplots()

    intervalMonth = 3
    datetime = pd.to_datetime(time * 24 * 3600 * 1e9)
    timeLabel = pd.date_range(start=datetime[0], end=datetime[-1], freq=str(intervalMonth)+"M")
    cbar = fig.colorbar(sc_all, ticks=dates.MonthLocator(interval=intervalMonth))
    cbar.ax.set_yticklabels(timeLabel.strftime("%b %Y"))
    timeColor = cbar.to_rgba(time)

    sc = img.scatter(temp[indices], drsValue[indices], s=50, marker="+",
                     c=timeColor[indices], label="Normal")

    sc_ = None
    if(len(indices_) > 0):
        sc_ = img.scatter(temp[indices_], drsValue[indices_], s=50, marker="*",
                          c=timeColor[indices_], label="Not used")

    if(showStd_ is True):
        for i in range(len(temp)):
            plt.errorbar(temp[i], drsValue[i],
                         xerr=tempStd[i], yerr=drsValueStd[i],
                         color=timeColor[i], marker='', ls='', zorder=0)

    temperature = np.linspace(min(temp)-1, max(temp)+1, 1000)
    var, cov = getLinearFitValues(temp[indices], drsValue[indices], drsValueStd[indices])
    fit = linearerFit(temperature, var[0], var[1])

    fitPlot, = plt.plot(temperature, fit, "k-",
                        label=("    All: a="+str('{:0.2f}'.format(round(var[0], 2)))+r"$\pm$" +
                               str('{:0.2f}'.format(round(np.sqrt(cov[0][0]), 2))) +
                               ", b="+str('{:0.1f}'.format(round(var[1], 1)))+r"$\pm$" +
                               str('{:0.1f}'.format(round(np.sqrt(cov[1][1]), 1)))))

    sc2 = None
    fitPlot2 = None
    if (subTimeInterval_ is not None and len(subTimeInterval_) == 2):
        startDatetime, endDatetime = pd.to_datetime(subTimeInterval_[0]), pd.to_datetime(subTimeInterval_[1])
        indicesSub = np.intersect1d(indices, np.where((datetime > startDatetime) & (datetime < endDatetime))[0])

        var2, cov2 = getLinearFitValues(temp[indicesSub], drsValue[indicesSub], drsValueStd[indicesSub])
        fit2 = linearerFit(temperature, var2[0], var2[1])
        sc2 = img.scatter(temp[indicesSub], drsValue[indicesSub], s=50, marker="o",
                          c=timeColor[indicesSub], label="Interval")

        fitPlot2, = plt.plot(temperature, fit2, "r-",
                             label=("Interval: a="+str('{:0.3f}'.format(round(var2[0], 3)))+r"$\pm$" +
                                    str('{:0.3f}'.format(round(np.sqrt(cov2[0][0]), 3))) +
                                    ", b="+str('{:0.3f}'.format(round(var2[1], 3)))+r"$\pm$" +
                                    str('{:0.3f}'.format(round(np.sqrt(cov2[1][1]), 3)))))

    plt.title(valueType_+"Mean\n")# Pixel: "+str(pixelNr_)+", Capacitor: "+str(capNr_) +
    #          ", ErrFactor: "+str('{:0.1f}'.format(errFactor_)), fontsize=15, y=1.02)

    plt.xlabel(r'Temperature /$\mathrm{^\circ C}$')
    plt.ylabel(valueType_+r'Mean +TriggerMean /$\mathrm{mv}$')
    plt.xlim(min(temp)-1, max(temp)+1)
    plt.grid()
    handles1 = [sc]
    handles2 = [fitPlot]
    if(sc_ is not None):
        handles1.append(sc_)
    if(sc2 is not None and fitPlot2 is not None):
        handles1.append(sc2)
        handles2.append(fitPlot2)
    first_legend = plt.legend(handles=handles1, loc="upper right", ncol=2, scatterpoints=1,
                              title=valueType_+"Mean with averaged Temperature")
    plt.gca().add_artist(first_legend)
    plt.legend(handles=handles2, loc="lower left", numpoints=1,
               title="Linearerfit: "+r"$f(x) = a\cdot x + b$")
    plt.gca().ticklabel_format(useOffset=False)
    plt.text(0.02, 0.19, "preliminary", fontdict=font, transform=plt.gca().transAxes)
    if(storeFilename_ is not None):
        plt.savefig(storeFilename_)
    plt.show()
    plt.close()


####################################################################################################
def pixelCapacitorDrsValueInPEA(drsFilename_, valueType_, pixelNr_, capNr_, errFactor_=2.0, showStd_=False,
                                subTimeInterval_=None, storeFilename_=None):

    with h5py.File(drsFilename_, 'r') as store:
        time = np.array(store["Time"+valueType_]).flatten()
        temp = np.array(store["Temp"+valueType_][:, int(pixelNr_/9)])
        tempStd = np.array(store["TempStd"+valueType_][:, int(pixelNr_/9)])
        drsValue = np.array(store[valueType_+"Mean"][:, pixelNr_*fact.nrCap + capNr_])
        drsValueStd = np.array(store[valueType_+"MeanStd"][:, pixelNr_*fact.nrCap + capNr_])

    drsValueStdMean = np.mean(drsValueStd, dtype='float64')
    indices = np.where(drsValueStd < drsValueStdMean*errFactor_)[0]
    indices_ = np.where(drsValueStd > drsValueStdMean*errFactor_)[0]

    drsValueMean = np.mean(drsValue, dtype='float64')
    drsValue = (drsValue-drsValueMean)/10
    drsValueStd = drsValueStd/10

    sc_all = plt.scatter(temp, drsValue, c=time)
    plt.close()  # Dont show sc_all, just use it to create the colorbar

    fig, img = plt.subplots()

    intervalMonth = 3
    datetime = pd.to_datetime(time * 24 * 3600 * 1e9)
    timeLabel = pd.date_range(start=datetime[0], end=datetime[-1], freq=str(intervalMonth)+"M")
    cbar = fig.colorbar(sc_all, ticks=dates.MonthLocator(interval=intervalMonth))
    cbar.ax.set_yticklabels(timeLabel.strftime("%b %Y"))
    timeColor = cbar.to_rgba(time)

    sc = img.scatter(temp[indices], drsValue[indices], s=50, marker="+",
                     c=timeColor[indices], label="paternNoiseMean with averaged Temperature")

    sc_ = None
    if(len(indices_) > 0):
        sc_ = img.scatter(temp[indices_], drsValue[indices_], s=50, marker="*",
                          c=timeColor[indices_], label="Not used")

    if(showStd_ is True):
        for i in range(len(temp)):
            plt.errorbar(temp[i], drsValue[i],
                         xerr=tempStd[i], yerr=drsValueStd[i],
                         color=timeColor[i], marker='', ls='', zorder=0)

    temperature = np.linspace(min(temp)-1, max(temp)+1, 1000)







    # fitfile = "../../isdcRoot/gpfs0/scratch/schulz/fitValues/fitValuesData_Interval1of1.fits"
    # fitValuesTab = fits.open(fitfile, ignoremissing=True, ignore_missing_end=True)
    # slopeBase = fitValuesTab[valueType_].data["slope"][0][fact.nrCap*pixelNr_+capNr_]
    # offsetBase = fitValuesTab[valueType_].data["offset"][0][fact.nrCap*pixelNr_+capNr_]
    # print(slopeBase, offsetBase)
    # temperature = np.linspace(min(temp)-1, max(temp)+1, 1000)
    # fit = linearerFit(temperature, slopeBase, offsetBase)
    # plt.plot(temperature, fit, "k-")
    #
    # fitfile = "../../isdcRoot/gpfs0/scratch/schulz/fitValues/fitValuesData_Interval1of3.fits"
    # fitValuesTab = fits.open(fitfile, ignoremissing=True, ignore_missing_end=True)
    # slopeBase = fitValuesTab[valueType_].data["slope"][0][fact.nrCap*pixelNr_+capNr_]
    # offsetBase = fitValuesTab[valueType_].data["offset"][0][fact.nrCap*pixelNr_+capNr_]
    # print(slopeBase, offsetBase)
    # temperature = np.linspace(min(temp)-1, max(temp)+1, 1000)
    # fit = linearerFit(temperature, slopeBase, offsetBase)
    # plt.plot(temperature, fit, "r--")
    #
    # fitfile = "../../isdcRoot/gpfs0/scratch/schulz/fitValues/fitValuesData_Interval2of3.fits"
    # fitValuesTab = fits.open(fitfile, ignoremissing=True, ignore_missing_end=True)
    # slopeBase = fitValuesTab[valueType_].data["slope"][0][fact.nrCap*pixelNr_+capNr_]
    # offsetBase = fitValuesTab[valueType_].data["offset"][0][fact.nrCap*pixelNr_+capNr_]
    # print(slopeBase, offsetBase)
    # temperature = np.linspace(min(temp)-1, max(temp)+1, 1000)
    # fit = linearerFit(temperature, slopeBase, offsetBase)
    # plt.plot(temperature, fit, "b--")
    #
    # fitfile = "../../isdcRoot/gpfs0/scratch/schulz/fitValues/fitValuesData_Interval3of3.fits"
    # fitValuesTab = fits.open(fitfile, ignoremissing=True, ignore_missing_end=True)
    # slopeBase = fitValuesTab[valueType_].data["slope"][0][fact.nrCap*pixelNr_+capNr_]
    # offsetBase = fitValuesTab[valueType_].data["offset"][0][fact.nrCap*pixelNr_+capNr_]
    # print(slopeBase, offsetBase)
    # temperature = np.linspace(min(temp)-1, max(temp)+1, 1000)
    # fit = linearerFit(temperature, slopeBase, offsetBase)
    # plt.plot(temperature, fit, "g--")










    var, cov = getLinearFitValues(temp[indices], drsValue[indices], drsValueStd[indices])
    fit = linearerFit(temperature, var[0], var[1])

    fitPlot, = plt.plot(temperature, fit, "k-",
                        # label=("    All: a="+str('{:0.2f}'.format(round(var[0], 2)))+r"$\pm$" +
                        #        str('{:0.2f}'.format(round(np.sqrt(cov[0][0]), 2))) +
                        #        ", b="+str('{:0.1f}'.format(round(var[1], 1)))+r"$\pm$" +
                        #        str('{:0.1f}'.format(round(np.sqrt(cov[1][1]), 1)))))
                        label=("a=("+str('{:0.3f}'.format(round(var[0], 3)))+r"$\pm$" +
                               str('{:0.3f}'.format(round(np.sqrt(cov[0][0]), 3)))+r")$\,\frac{\mathrm{PEA}}{\mathrm{^\circ C}}$" +
                               ", b=("+str('{:0.2f}'.format(round(var[1], 2)))+r"$\pm$" +
                               str('{:0.2f}'.format(round(np.sqrt(cov[1][1]), 2)))+r")$\,\mathrm{PEA}$"))

    sc2 = None
    fitPlot2 = None
    if (subTimeInterval_ is not None and len(subTimeInterval_) == 2):
        startDatetime, endDatetime = pd.to_datetime(subTimeInterval_[0]), pd.to_datetime(subTimeInterval_[1])
        indicesSub = np.intersect1d(indices, np.where((datetime > startDatetime) & (datetime < endDatetime))[0])

        var2, cov2 = getLinearFitValues(temp[indicesSub], drsValue[indicesSub], drsValueStd[indicesSub])
        fit2 = linearerFit(temperature, var2[0], var2[1])
        sc2 = img.scatter(temp[indicesSub], drsValue[indicesSub], s=50, marker="o",
                          c=timeColor[indicesSub], label="Interval")

        fitPlot2, = plt.plot(temperature, fit2, "r-",
                             label=("Interval: a="+str('{:0.3f}'.format(round(var2[0], 3)))+r"$\pm$" +
                                    str('{:0.3f}'.format(round(np.sqrt(cov2[0][0]), 3))) +
                                    ", b="+str('{:0.3f}'.format(round(var2[1], 3)))+r"$\pm$" +
                                    str('{:0.3f}'.format(round(np.sqrt(cov2[1][1]), 3)))))

    #plt.title(valueType_+"Mean\n", fontsize=20, y=0.95)# Pixel: "+str(pixelNr_)+", Capacitor: "+str(capNr_) +
    #          ", ErrFactor: "+str('{:0.1f}'.format(errFactor_)), fontsize=15, y=1.02)

    plt.xlabel(r'Temperature /$\mathrm{^\circ C}$')
    #plt.ylabel(valueType_+r'Mean /$\mathrm{mv}$')
    plt.ylabel(r'paternNoiseMean /$\mathrm{PEA}$')
    plt.ylim(-0.65, 0.4)
    plt.xlim(min(temp)-1, max(temp)+1)
    plt.grid()
    handles1 = [sc]
    handles2 = [fitPlot]
    if(sc_ is not None):
        handles1.append(sc_)
    if(sc2 is not None and fitPlot2 is not None):
        handles1.append(sc2)
        handles2.append(fitPlot2)
    #first_legend = plt.legend(handles=handles1, loc="upper right", ncol=2, scatterpoints=1)
    #plt.gca().add_artist(first_legend)
    plt.legend(handles=handles2, loc="lower left", numpoints=1,
               title="Linearerfit: "+r"$f(x) = a\cdot x + b$")
    plt.gca().ticklabel_format(useOffset=False)
    plt.text(0.02, 0.21, "preliminary", fontdict=font, transform=plt.gca().transAxes)
    if(storeFilename_ is not None):
        plt.savefig(storeFilename_)
    plt.show()
    plt.close()




# TODO update
####################################################################################################
def pixelCapacitorDrsValue(drsFilename_, valueType_, pixelNr_, capNr_, errFactor_=2.0, showStd_=False,
                           subTimeInterval_=None, storeFilename_=None):

    with h5py.File(drsFilename_, 'r') as store:
        time = np.array(store["Time"+valueType_]).flatten()
        temp = np.array(store["Temp"+valueType_][:, int(pixelNr_/9)])
        tempStd = np.array(store["TempStd"+valueType_][:, int(pixelNr_/9)])
        drsValue = np.array(store[valueType_+"Mean"][:, pixelNr_*fact.nrCap + capNr_])
        drsValueStd = np.array(store[valueType_+"MeanStd"][:, pixelNr_*fact.nrCap + capNr_])

    drsValueStdMean = np.mean(drsValueStd, dtype='float64')
    indices = np.where(drsValueStd < drsValueStdMean*errFactor_)[0]
    indices_ = np.where(drsValueStd > drsValueStdMean*errFactor_)[0]

    sc_all = plt.scatter(temp, drsValue, c=time)
    plt.close()  # Dont show sc_all, just use it to create the colorbar

    fig, img = plt.subplots()

    intervalMonth = 3
    datetime = pd.to_datetime(time * 24 * 3600 * 1e9)
    timeLabel = pd.date_range(start=datetime[0], end=datetime[-1], freq=str(intervalMonth)+"M")
    cbar = fig.colorbar(sc_all, ticks=dates.MonthLocator(interval=intervalMonth))
    cbar.ax.set_yticklabels(timeLabel.strftime("%b %Y"))
    timeColor = cbar.to_rgba(time)

    sc = img.scatter(temp[indices], drsValue[indices], s=50, marker="+",
                     c=timeColor[indices], label="Normal")

    sc_ = None
    if(len(indices_) > 0):
        sc_ = img.scatter(temp[indices_], drsValue[indices_], s=50, marker="*",
                          c=timeColor[indices_], label="Not used")

    if(showStd_ is True):
        for i in range(len(temp)):
            plt.errorbar(temp[i], drsValue[i],
                         xerr=tempStd[i], yerr=drsValueStd[i],
                         color=timeColor[i], marker='', ls='', zorder=0)

    temperature = np.linspace(min(temp)-1, max(temp)+1, 1000)







    # fitfile = "../../isdcRoot/gpfs0/scratch/schulz/fitValues/fitValuesData_Interval1of1.fits"
    # fitValuesTab = fits.open(fitfile, ignoremissing=True, ignore_missing_end=True)
    # slopeBase = fitValuesTab[valueType_].data["slope"][0][fact.nrCap*pixelNr_+capNr_]
    # offsetBase = fitValuesTab[valueType_].data["offset"][0][fact.nrCap*pixelNr_+capNr_]
    # print(slopeBase, offsetBase)
    # temperature = np.linspace(min(temp)-1, max(temp)+1, 1000)
    # fit = linearerFit(temperature, slopeBase, offsetBase)
    # plt.plot(temperature, fit, "k-")
    #
    # fitfile = "../../isdcRoot/gpfs0/scratch/schulz/fitValues/fitValuesData_Interval1of3.fits"
    # fitValuesTab = fits.open(fitfile, ignoremissing=True, ignore_missing_end=True)
    # slopeBase = fitValuesTab[valueType_].data["slope"][0][fact.nrCap*pixelNr_+capNr_]
    # offsetBase = fitValuesTab[valueType_].data["offset"][0][fact.nrCap*pixelNr_+capNr_]
    # print(slopeBase, offsetBase)
    # temperature = np.linspace(min(temp)-1, max(temp)+1, 1000)
    # fit = linearerFit(temperature, slopeBase, offsetBase)
    # plt.plot(temperature, fit, "r--")
    #
    # fitfile = "../../isdcRoot/gpfs0/scratch/schulz/fitValues/fitValuesData_Interval2of3.fits"
    # fitValuesTab = fits.open(fitfile, ignoremissing=True, ignore_missing_end=True)
    # slopeBase = fitValuesTab[valueType_].data["slope"][0][fact.nrCap*pixelNr_+capNr_]
    # offsetBase = fitValuesTab[valueType_].data["offset"][0][fact.nrCap*pixelNr_+capNr_]
    # print(slopeBase, offsetBase)
    # temperature = np.linspace(min(temp)-1, max(temp)+1, 1000)
    # fit = linearerFit(temperature, slopeBase, offsetBase)
    # plt.plot(temperature, fit, "b--")
    #
    # fitfile = "../../isdcRoot/gpfs0/scratch/schulz/fitValues/fitValuesData_Interval3of3.fits"
    # fitValuesTab = fits.open(fitfile, ignoremissing=True, ignore_missing_end=True)
    # slopeBase = fitValuesTab[valueType_].data["slope"][0][fact.nrCap*pixelNr_+capNr_]
    # offsetBase = fitValuesTab[valueType_].data["offset"][0][fact.nrCap*pixelNr_+capNr_]
    # print(slopeBase, offsetBase)
    # temperature = np.linspace(min(temp)-1, max(temp)+1, 1000)
    # fit = linearerFit(temperature, slopeBase, offsetBase)
    # plt.plot(temperature, fit, "g--")










    var, cov = getLinearFitValues(temp[indices], drsValue[indices], drsValueStd[indices])
    fit = linearerFit(temperature, var[0], var[1])

    fitPlot, = plt.plot(temperature, fit, "k-",
                        # label=("    All: a="+str('{:0.2f}'.format(round(var[0], 2)))+r"$\pm$" +
                        #        str('{:0.2f}'.format(round(np.sqrt(cov[0][0]), 2))) +
                        #        ", b="+str('{:0.1f}'.format(round(var[1], 1)))+r"$\pm$" +
                        #        str('{:0.1f}'.format(round(np.sqrt(cov[1][1]), 1)))))
                        label=("a="+str('{:0.2f}'.format(round(var[0], 2)))+r"$\pm$" +
                               str('{:0.2f}'.format(round(np.sqrt(cov[0][0]), 2))) +
                               ", b="+str('{:0.1f}'.format(round(var[1], 1)))+r"$\pm$" +
                               str('{:0.1f}'.format(round(np.sqrt(cov[1][1]), 1)))))

    sc2 = None
    fitPlot2 = None
    if (subTimeInterval_ is not None and len(subTimeInterval_) == 2):
        startDatetime, endDatetime = pd.to_datetime(subTimeInterval_[0]), pd.to_datetime(subTimeInterval_[1])
        indicesSub = np.intersect1d(indices, np.where((datetime > startDatetime) & (datetime < endDatetime))[0])

        var2, cov2 = getLinearFitValues(temp[indicesSub], drsValue[indicesSub], drsValueStd[indicesSub])
        fit2 = linearerFit(temperature, var2[0], var2[1])
        sc2 = img.scatter(temp[indicesSub], drsValue[indicesSub], s=50, marker="o",
                          c=timeColor[indicesSub], label="Interval")

        fitPlot2, = plt.plot(temperature, fit2, "r-",
                             label=("Interval: a="+str('{:0.3f}'.format(round(var2[0], 3)))+r"$\pm$" +
                                    str('{:0.3f}'.format(round(np.sqrt(cov2[0][0]), 3))) +
                                    ", b="+str('{:0.3f}'.format(round(var2[1], 3)))+r"$\pm$" +
                                    str('{:0.3f}'.format(round(np.sqrt(cov2[1][1]), 3)))))

    #plt.title(valueType_+"Mean\n", fontsize=20, y=0.95)# Pixel: "+str(pixelNr_)+", Capacitor: "+str(capNr_) +
    #          ", ErrFactor: "+str('{:0.1f}'.format(errFactor_)), fontsize=15, y=1.02)

    plt.xlabel(r'Temperature /$\mathrm{^\circ C}$')
    #plt.ylabel(valueType_+r'Mean /$\mathrm{mv}$')
    plt.ylabel(r'patternNoise mean /$\mathrm{mv}$')
    plt.xlim(min(temp)-1, max(temp)+1)
    plt.grid()
    handles1 = [sc]
    handles2 = [fitPlot]
    if(sc_ is not None):
        handles1.append(sc_)
    if(sc2 is not None and fitPlot2 is not None):
        handles1.append(sc2)
        handles2.append(fitPlot2)
    first_legend = plt.legend(handles=handles1, loc="upper right", ncol=2, scatterpoints=1,
                              title=valueType_+"Mean with averaged Temperature")
    plt.gca().add_artist(first_legend)
    plt.legend(handles=handles2, loc="lower left", numpoints=1,
               title="Linearerfit: "+r"$f(x) = a\cdot x + b$")
    plt.gca().ticklabel_format(useOffset=False)
    plt.text(0.02, 0.19, "preliminary", fontdict=font, transform=plt.gca().transAxes)
    if(storeFilename_ is not None):
        plt.savefig(storeFilename_)
    plt.show()
    plt.close()


# TODO update
####################################################################################################
def pixelFitValues(sourceFilename_, pixelNr_, valueType_, part_, showStd_=False, save_=False):

    fitValuesTab = fits.open(sourceFilename_, ignoremissing=True, ignore_missing_end=True)
    slopeBase = fitValuesTab["Baseline"].data["slope"][0][fact.nrCap*pixelNr_:fact.nrCap*(pixelNr_+1)]
    slopeStdBase = fitValuesTab["Baseline"].data["slopeStd"][0][fact.nrCap*pixelNr_:fact.nrCap*(pixelNr_+1)]
    offsetBase = fitValuesTab["Baseline"].data["offset"][0][fact.nrCap*pixelNr_:fact.nrCap*(pixelNr_+1)]
    offsetStdBase = fitValuesTab["Baseline"].data["offsetStd"][0][fact.nrCap*pixelNr_:fact.nrCap*(pixelNr_+1)]

    slopeGain = fitValuesTab["Gain"].data["slope"][0][fact.nrCap*pixelNr_:fact.nrCap*(pixelNr_+1)]
    slopeStdGain = fitValuesTab["Gain"].data["slopeStd"][0][fact.nrCap*pixelNr_:fact.nrCap*(pixelNr_+1)]
    offsetGain = fitValuesTab["Gain"].data["offset"][0][fact.nrCap*pixelNr_:fact.nrCap*(pixelNr_+1)]
    offsetStdGain = fitValuesTab["Gain"].data["offsetStd"][0][fact.nrCap*pixelNr_:fact.nrCap*(pixelNr_+1)]

    valueTypeTerms = [["Baseline", "baseline"], ["Gain", "gain"]]
    partTerms = [["Slope", "slope", "m"], ["Offset", "offset", "b"]]

    labelStr = ''
    variableStr = ''
    partStr = ''
    errStr = ''

    colors = hot(np.linspace(0, 0, fact.nrCap))
    for i in range(33):
        colors[i*32-1] = [1., 0., 0., 1.]

    Capacitor = np.linspace(0, fact.nrCap-1, fact.nrCap)
    if(valueType_ in valueTypeTerms[0]):
        variableStr = "BaselineMean"
        if(part_ in partTerms[0]):
            partStr = "Slope"
            plt.scatter(Capacitor, slopeBase, s=50, marker="+",
                        color=colors)
            labelStr = partStr+r" [$\frac{mV}{^\circ C}$]"
        elif(part_ in partTerms[1]):
            partStr = "Offset"
            plt.scatter(Capacitor, offsetBase, s=50, marker="+",
                        color=colors)
            labelStr = partStr+r" /$\mathrm{mv}$"
        else:
            print("Unknown part '", part_, "' - expected '",
                  partTerms[0][0], "' or '", partTerms[1][0], "'")
            return

        if(showStd_ is True):
            errStr = "withError"
            if(part_ in partTerms[0]):
                for i in range(len(Capacitor)):
                    plt.errorbar(Capacitor[i], slopeBase[i], xerr=None, yerr=slopeStdBase[i],
                                 color=colors[i], marker='', ls='', zorder=0)
            elif(part_ in partTerms[1]):
                for i in range(len(Capacitor)):
                    plt.errorbar(Capacitor[i], offsetBase[i], xerr=None, yerr=offsetStdBase[i],
                                 color=colors[i], marker='', ls='', zorder=0)

    elif(valueType_ in valueTypeTerms[1]):
        variableStr = "GainMean"
        if(part_ in partTerms[0]):
            partStr = "Slope"
            plt.scatter(Capacitor, slopeGain, s=50, marker="+",
                        color=colors)
            labelStr = partStr+r" [$\frac{mV}{^\circ C}$]"
        elif(part_ in partTerms[1]):
            partStr = "Offset"
            plt.scatter(Capacitor, offsetGain, s=50, marker="+",
                        color=colors)
            labelStr = partStr+r" /$\mathrm{mv}$"
        else:
            print("Unknown part '", part_, "' - expected '",
                  partTerms[0][0], "' or '", partTerms[1][0], "'")
            return

        if(showStd_ is True):
            errStr = "withError"
            if(part_ in partTerms[0]):
                for i in range(len(Capacitor)):
                    plt.errorbar(Capacitor[i], slopeGain[i], xerr=None, yerr=slopeStdGain[i],
                                 color=colors[i], marker='', ls='', zorder=0)
            elif(part_ in partTerms[1]):
                for i in range(len(Capacitor)):
                    plt.errorbar(Capacitor[i], offsetGain[i], xerr=None, yerr=offsetStdGain[i],
                                 color=colors[i], marker='', ls='', zorder=0)
    else:
        print("Unknown variable '", valueType_, "' - expected '",
              valueTypeTerms[0][0], "' or '", valueTypeTerms[1][0], "'")
        return

    plt.title(variableStr+r" Fit $f(x) = m \cdot x + b$"+"\n"+partStr+" Pixel:"+str(pixelNr_), fontsize=16, y=1.01)
    plt.xlabel('Capacitor [1]')
    plt.ylabel(labelStr)
    plt.xlim(-1, fact.nrCap)
    plt.grid()
    # plt.legend(loc='upper right', scatterpoints=1, numpoints=1)
    if(save_ is True):
        plt.savefig("../plots/"+variableStr+partStr+"_P"+str('{:04d}'.format(pixelNr_))+errStr+".pdf")
    plt.show()
    plt.close()


# TODO update
####################################################################################################
def fitValuesStd(bar_=150, save_=False):

    fitValuesTab = fits.open(path.getIsdcRootPath()+"fitValuesData.fits", ignoremissing=True, ignore_missing_end=True)
    slopeBaseStd = np.array(fitValuesTab["Baseline"].data["slopeStd"][0])
    offsetBaseStd = np.array(fitValuesTab["Baseline"].data["offsetStd"][0])
    slopeGainStd = np.array(fitValuesTab["Gain"].data["slopeStd"][0])
    offsetGainStd = np.array(fitValuesTab["Gain"].data["offsetStd"][0])

    xRange = np.linspace(0, fact.nrPix-1, fact.nrPix)

    slopeBaseStdPerPixel = []
    offsetBaseStdPerPixel = []
    slopeGainStdPerPixel = []
    offsetGainStdPerPixel = []

    for i in range(fact.nrPix):
        slopeBaseStdPerPixel.append(sum(slopeBaseStd[i*fact.nrCap:(i+1)*fact.nrCap])/fact.nrCap)
        offsetBaseStdPerPixel.append(sum(offsetBaseStd[i*fact.nrCap:(i+1)*fact.nrCap])/fact.nrCap)
        slopeGainStdPerPixel.append(sum(slopeGainStd[i*fact.nrCap:(i+1)*fact.nrCap])/fact.nrCap)
        offsetGainStdPerPixel.append(sum(offsetGainStd[i*fact.nrCap:(i+1)*fact.nrCap])/fact.nrCap)

    barlist = plt.bar(xRange-0.5, slopeBaseStdPerPixel, width=1., color='b', edgecolor='none')
    for i in range(fact.nrPix):
        if(i % 9 == 0):
            barlist[i-1].set_color('r')
    plt.title(r"BaselineMean Fit $f(x) = m \cdot x + b$"+"\n"+"Mean SlopeStd per Pixel", fontsize=16, y=1.01)
    plt.ylabel(r'slopeStd U/$\mathrm{mv}$')
    plt.xlabel('PixelNr')
    plt.xlim(-1, fact.nrPix)
    plt.show()
    plt.close()

    barlist = plt.bar(xRange-0.5, offsetBaseStdPerPixel, width=1., color='b', edgecolor='none')
    for i in range(fact.nrPix):
        if(i % 9 == 0):
            barlist[i-1].set_color('r')
    plt.title(r"BaselineMean Fit $f(x) = m \cdot x + b$"+"\n"+"Mean OffsetStd per Pixel", fontsize=16, y=1.01)
    plt.ylabel(r'offsetStd U/$\mathrm{mv}$')
    plt.xlabel('PixelNr')
    plt.xlim(-1, fact.nrPix)
    plt.show()
    plt.close()

    barlist = plt.bar(xRange-0.5, slopeGainStdPerPixel, width=1., color='b', edgecolor='none')
    for i in range(fact.nrPix):
        if(i % 9 == 0):
            barlist[i-1].set_color('r')
    plt.title(r"GainMean Fit $f(x) = m \cdot x + b$"+"\n"+"Mean slopeStd per Pixel", fontsize=16, y=1.01)
    plt.ylabel(r'slopeStd U/$\mathrm{mv}$')
    plt.xlabel('PixelNr')
    plt.xlim(-1, fact.nrPix)
    plt.show()
    plt.close()

    barlist = plt.bar(xRange-0.5, offsetGainStdPerPixel, width=1., color='b', edgecolor='none')
    for i in range(fact.nrPix):
        if(i % 9 == 0):
            barlist[i-1].set_color('r')
    plt.title(r"GainMean Fit $f(x) = m \cdot x + b$"+"\n"+"Mean offsetStd per Pixel", fontsize=16, y=1.01)
    plt.ylabel(r'offsetStd U/$\mathrm{mv}$')
    plt.xlabel('PixelNr')
    plt.xlim(-1, fact.nrPix)
    plt.show()
    plt.close()


####################################################################################################
def residuenPerPixelCapacitor(drsFilename_, residuenFilenameArray_, valueType_,
                              pixelNr_, capNr_, restrictResiduen_, storeFilename_=None):

    print("Loading '"+valueType_+"-data' ...")
    with h5py.File(drsFilename_, 'r') as store:
        time = np.array(store["Time"+valueType_]).flatten()
        datetime = pd.to_datetime(time * 24 * 3600 * 1e9)

    date = datetime.date
    datetime = np.array(datetime)

    nrOfIntervals = len(residuenFilenameArray_)

    intervalList = []
    residuenPairList = []
    minRes, maxRes = 0, 0
    minRes_, maxRes_ = 0, 0
    for residuenFilename in residuenFilenameArray_:
        with h5py.File(residuenFilename, 'r') as residuenTab:
            residuen = np.array(residuenTab["Residuen"+valueType_][pixelNr_*fact.nrCap+capNr_, :])
            interval_b = np.array(residuenTab["Interval"])

            interval = []
            for intervalLimit in interval_b:
                interval.append(pd.to_datetime(intervalLimit[0].decode("UTF-8")).date())

            if(restrictResiduen_):
                intervalIndices = np.where((date >= interval[0]) & (date <= interval[1]))[0]
                residuenPair = [datetime[intervalIndices], residuen[intervalIndices]]
            else:

                residuenPair = [datetime, residuen]

            minRes_, maxRes_ = min(residuenPair[1]), max(residuenPair[1])
            if(minRes_ < minRes):
                minRes = minRes_
            if(maxRes_ > maxRes):
                maxRes = maxRes_

        intervalList.append(interval)
        residuenPairList.append(residuenPair)

    intervalList = np.array(intervalList)
    residuenPairList = np.array(residuenPairList)

    offset = min([abs(minRes*0.1), abs(maxRes*0.1)])
    if(nrOfIntervals > 1):
        for intervalIndex in range(nrOfIntervals):
            c = [float(intervalIndex)/float(nrOfIntervals-1),
                 0.0,
                 float(nrOfIntervals-1-intervalIndex)/float(nrOfIntervals-1)]
            plt.plot([intervalList[intervalIndex][0], intervalList[intervalIndex][0]],
                     [minRes-offset, maxRes+offset], "k-")
            plt.plot([intervalList[intervalIndex][1], intervalList[intervalIndex][1]],
                     [minRes-offset, maxRes+offset], "k-")

            plt.annotate(s='', xy=(intervalList[intervalIndex][0], 0), xytext=(intervalList[intervalIndex][1], 0),
                         arrowprops=dict(arrowstyle='<->', color=c))
            plt.plot(residuenPairList[intervalIndex][0], residuenPairList[intervalIndex][1], "x", color=c,)
        plt.plot(residuenPairList[0][0], residuenPairList[0][1], "kx", label=("Residuen per interval"))
        plt.plot([date[0], date[0]], [minRes, maxRes], "k-", label="Interval boundary")
        plt.legend(loc='lower left', numpoints=1)
    else:
        plt.plot(residuenPair[0], residuenPair[1], "kx")
    # plt.errorbar(datetime, residuenMean, yerr=residuenMeanStd, fmt="x",
    #              label=(valueType_+r"Mean - f(t)"))
    # plt.plot(datetime, np.mean(residuen[0:1080, :], dtype="float64", axis=0), "gx", label="Crate 1-3")

    plt.title(valueType_+"Residum \n Pixel: "+str(pixelNr_)+" Capacitor: "+str(capNr_))
    plt.ylabel("(f(t)-"+valueType_+"Mean)/$\mathrm{mv}$")
    plt.gcf().autofmt_xdate()
    plt.xlim(min(date)-pd.DateOffset(days=7), max(date)+pd.DateOffset(days=7))
    plt.ylim(minRes-offset, maxRes+offset)
    plt.grid()
    plt.text(0.02, 0.2, "preliminary", fontdict=font, transform=plt.gca().transAxes)
    if(storeFilename_ is not None):
        plt.savefig(storeFilename_)
    plt.show()
    plt.close()


####################################################################################################
def residumMeanOfAllCapacitors(drsFilename_, residuenFilenameArray_, valueType_,
                               restrictResiduen_, storeFilename_=None):

    nrOfIntervals = len(residuenFilenameArray_)

    print("Loading '"+valueType_+"-data' ...")
    with h5py.File(drsFilename_, 'r') as store:
        time = np.array(store["Time"+valueType_]).flatten()
        datetime = pd.to_datetime(time * 24 * 3600 * 1e9)

    date = datetime.date
    datetime = np.array(datetime)

    intervalList = []
    residuenPairList = []
    minRes, maxRes = 0, 0
    minRes_, maxRes_ = 0, 0
    for residuenFilename in residuenFilenameArray_:
        with h5py.File(residuenFilename, 'r') as residuenTab:
            residuen = np.array(residuenTab["Residuen"+valueType_])
            residuenMean = np.mean(residuen, dtype="float64", axis=0)
            # residuenMeanStd = np.std(residuen, dtype="float64", axis=0, ddof=1)
            # print("max residuenMeanStd: ", max(residuenMeanStd))

            interval_b = np.array(residuenTab["Interval"])

            interval = []
            for intervalLimit in interval_b:
                interval.append(pd.to_datetime(intervalLimit[0].decode("UTF-8")).date())

            if(restrictResiduen_):
                intervalIndices = np.where((date >= interval[0]) & (date <= interval[1]))[0]
                residuenPair = [datetime[intervalIndices], residuenMean[intervalIndices]]
            else:

                residuenPair = [datetime, residuenMean]

            minRes_, maxRes_ = min(residuenPair[1]), max(residuenPair[1])
            if(minRes_ < minRes):
                minRes = minRes_
            if(maxRes_ > maxRes):
                maxRes = maxRes_

        intervalList.append(interval)
        residuenPairList.append(residuenPair)

    offset = min([abs(minRes*0.1), abs(maxRes*0.1)])
    if(nrOfIntervals > 1):
        for intervalIndex in range(nrOfIntervals):
            c = [float(intervalIndex)/float(nrOfIntervals-1),
                 0.0,
                 float(nrOfIntervals-1-intervalIndex)/float(nrOfIntervals-1)]
            plt.plot([intervalList[intervalIndex][0], intervalList[intervalIndex][0]],
                     [minRes-offset, maxRes+offset], "k-")
            plt.plot([intervalList[intervalIndex][1], intervalList[intervalIndex][1]],
                     [minRes-offset, maxRes+offset], "k-")

            plt.annotate(s='', xy=(intervalList[intervalIndex][0], 0), xytext=(intervalList[intervalIndex][1], 0),
                         arrowprops=dict(arrowstyle='<->', color=c))
            plt.plot(residuenPairList[intervalIndex][0], residuenPairList[intervalIndex][1], "x", color=c,)
        plt.plot(residuenPairList[0][0], residuenPairList[0][1], "kx", label=("ResiduenMean per Interval"))
        plt.plot([date[0], date[0]], [minRes, maxRes], "k-", label="Interval boundary")
        plt.legend(loc='lower left', numpoints=1)
    else:
        plt.plot(residuenPair[0], residuenPair[1], "kx")

    # plt.errorbar(datetime, residuenMean, yerr=residuenMeanStd, fmt="x",
    #              label=(valueType_+r"Mean - f(t)"))
    plt.title(valueType_+"Residum Mean:")
    plt.ylabel("(f(t)-"+valueType_+"Mean)/$\mathrm{mv}$")
    plt.gcf().autofmt_xdate()
    plt.xlim(min(date)-pd.DateOffset(days=7), max(date)+pd.DateOffset(days=7))
    plt.ylim(minRes-offset, maxRes+offset)
    plt.grid()
    plt.text(0.02, 0.2, "preliminary", fontdict=font, transform=plt.gca().transAxes)
    if(storeFilename_ is not None):
        plt.savefig(storeFilename_)
    plt.show()
    plt.close()


####################################################################################################
def residumMeanOfAllCapacitorsPerCrates(drsFilename_, residuenFilenameArray_, valueType_,
                                        restrictResiduen_, storeFilename_=None):

    nrOfIntervals = len(residuenFilenameArray_)

    print("Loading '"+valueType_+"-data' ...")
    with h5py.File(drsFilename_, 'r') as store:
        time = np.array(store["Time"+valueType_]).flatten()
        datetime = pd.to_datetime(time * 24 * 3600 * 1e9)

    date = datetime.date
    datetime = np.array(datetime)

    intervalList = []
    residuenPairList = []
    minRes, maxRes = 0, 0
    minRes_, maxRes_ = 0, 0
    for residuenFilename in residuenFilenameArray_:
        with h5py.File(residuenFilename, 'r') as residuenTab:
            residuen = np.array(residuenTab["Residuen"+valueType_])
            residuenMean = np.mean(residuen, dtype="float64", axis=0)
            residuenMeanC1 = np.mean(residuen[0*9*fact.nrCap:40*9*fact.nrCap, :], dtype="float64", axis=0)
            residuenMeanC2 = np.mean(residuen[40*9*fact.nrCap:80*9*fact.nrCap, :], dtype="float64", axis=0)
            residuenMeanC3 = np.mean(residuen[80*9*fact.nrCap:120*9*fact.nrCap, :], dtype="float64", axis=0)
            residuenMeanC4 = np.mean(residuen[120*9*fact.nrCap:160*9*fact.nrCap, :], dtype="float64", axis=0)

            residuenMeanPerCrates = np.array([residuenMeanC1, residuenMeanC2,
                                              residuenMeanC3, residuenMeanC4,
                                              residuenMean])

            interval_b = np.array(residuenTab["Interval"])

            interval = []
            for intervalLimit in interval_b:
                interval.append(pd.to_datetime(intervalLimit[0].decode("UTF-8")).date())

            if(restrictResiduen_):
                intervalIndices = np.where((date >= interval[0]) & (date <= interval[1]))[0]

                residuenPair = [datetime[intervalIndices], residuenMeanPerCrates[:, intervalIndices]]
            else:
                residuenPair = [datetime, residuenMeanPerCrates]

            minRes_, maxRes_ = np.amin(residuenPair[1]), np.amax(residuenPair[1])
            if(minRes_ < minRes):
                minRes = minRes_
            if(maxRes_ > maxRes):
                maxRes = maxRes_

        intervalList.append(interval)
        residuenPairList.append(residuenPair)

    offset = min([abs(minRes*0.1), abs(maxRes*0.1)])
    if(nrOfIntervals > 1):
        for intervalIndex in range(nrOfIntervals):
            plt.plot([intervalList[intervalIndex][0], intervalList[intervalIndex][0]],
                     [minRes-offset, maxRes+offset], "k-")
            plt.plot([intervalList[intervalIndex][1], intervalList[intervalIndex][1]],
                     [minRes-offset, maxRes+offset], "k-")

            print(residuenPairList[intervalIndex][0].shape, residuenPairList[intervalIndex][1].shape)

            print(residuenPairList[intervalIndex][0].shape, residuenPairList[intervalIndex][1][0].shape)
            print(residuenPairList[intervalIndex][0].shape, residuenPairList[intervalIndex][1][1].shape)
            print(residuenPairList[intervalIndex][0].shape, residuenPairList[intervalIndex][1][2].shape)

            plt.plot(residuenPairList[intervalIndex][0], residuenPairList[intervalIndex][1][0])
            plt.plot(residuenPairList[intervalIndex][0], residuenPairList[intervalIndex][1][1])
            plt.plot(residuenPairList[intervalIndex][0], residuenPairList[intervalIndex][1][2])
            plt.plot(residuenPairList[intervalIndex][0], residuenPairList[intervalIndex][1][3])
            plt.plot(residuenPairList[intervalIndex][0], residuenPairList[intervalIndex][1][4])

        plt.plot(residuenPairList[0][0], residuenPairList[0][1][0], "bx", label="Crate 1")
        plt.plot(residuenPairList[0][0], residuenPairList[0][1][1], "gx", label="Crate 2")
        plt.plot(residuenPairList[0][0], residuenPairList[0][1][2], "yx", label="Crate 3")
        plt.plot(residuenPairList[0][0], residuenPairList[0][1][3], "rx", label="Crate 4")
        plt.plot(residuenPairList[0][0], residuenPairList[0][1][4], "ko", label="Crate 1-4")
        plt.plot([date[0], date[0]], [minRes, maxRes], "k-", label="Interval boundary")

    else:
        plt.plot(datetime, residuenMeanC1, "bx", label="Crate 1")
        plt.plot(datetime, residuenMeanC2, "gx", label="Crate 2")
        plt.plot(datetime, residuenMeanC3, "yx", label="Crate 3")
        plt.plot(datetime, residuenMeanC4, "rx", label="Crate 4")
        plt.plot(datetime, residuenMean, "ko", label="Crate 1-4")

    plt.title(valueType_+"Residum Mean per Crate:")
    plt.ylabel("(f(t)-"+valueType_+r"Mean)/$\mathrm{mv}$")
    plt.gcf().autofmt_xdate()
    plt.xlim(min(date)-pd.DateOffset(days=7), max(date)+pd.DateOffset(days=7))
    plt.ylim(minRes-offset, maxRes+offset)
    plt.grid()
    plt.legend(loc='lower left', numpoints=1)
    plt.text(0.02, 0.2, "preliminary", fontdict=font, transform=plt.gca().transAxes)
    if(storeFilename_ is not None):
        plt.savefig(storeFilename_)
    plt.show()
    plt.close()


####################################################################################################
def residumMeanPerPatchAndInterval(drsFilename_, residuenFilenameArray_, valueType_, storeFilename_=None):

    print("Loading '"+valueType_+"-data' ...")
    with h5py.File(drsFilename_, 'r') as store:
        time = np.array(store["Time"+valueType_]).flatten()
        date = pd.to_datetime(time * 24 * 3600 * 1e9).date

    residumMeanPerPatchAndInterval = []
    for residuenFilename in residuenFilenameArray_:
        intervalResiduenMeanPerPatch = []
        with h5py.File(residuenFilename, 'r') as residuenTab:
            interval_b = np.array(residuenTab["Interval"])

            interval = []
            for intervalLimit in interval_b:
                interval.append(pd.to_datetime(intervalLimit[0].decode("UTF-8")).date())

            intervalIndices = np.where((date >= interval[0]) & (date <= interval[1]))[0]

            residuen = np.array(residuenTab["Residuen"+valueType_])[:, intervalIndices]
        for patchNr in range(fact.nrPatch):
            intervalResiduenMeanPerPatch.append(
                np.mean(abs(
                            residuen[patchNr*9*fact.nrCap:(patchNr+1)*9*fact.nrCap].flatten()
                            ), dtype="float64"))
        residumMeanPerPatchAndInterval.append(intervalResiduenMeanPerPatch)

    residumMeanPerPatchAndInterval = np.array(residumMeanPerPatchAndInterval).transpose()

    plt.matshow(residumMeanPerPatchAndInterval, interpolation="None", aspect='auto')
    # plt.title(r"Mean of the absolute "+str(valueType_)+"Residuen-value \n per Interval ", fontsize=25, y=1.02)
    cbar = plt.colorbar()
    resMax = residumMeanPerPatchAndInterval.shape
    for x in range(resMax[1]+1):
        plt.plot([x-0.5, x-0.5], [-0.5, resMax[0]-0.5], "k:")
    for y in range(4+1):
        y = y*40
        plt.plot([-0.5, resMax[1]-0.5], [y-0.5, y-0.5], "k:")
    plt.xlabel("IntervalNr", fontsize=20)
    plt.ylabel("PatchNr", fontsize=20)
    plt.tick_params(axis='both', which='major', direction='out', labelsize=16)
    cbar.ax.tick_params(labelsize=16)
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.xticks(np.arange(0, 160, 20))
    plt.gca().invert_yaxis()
    DefaultSize = plt.gcf().get_size_inches()
    plt.gcf().set_size_inches(DefaultSize[0]*2.5, DefaultSize[1], forward=True)
    plt.xlim(-0.5, resMax[1]-0.5)
    plt.ylim(-0.5, resMax[0]-0.5)
    if(storeFilename_ is not None):
        plt.savefig(storeFilename_)
    plt.show()
    plt.close()


# TODO update
####################################################################################################
def noise(drsFileCalibrated_, drsModelCalibrated_, drsModelIntervalCalibrated_,
          titleStr_, maxNoise_=None, storeFilename_=None, sourceFile_=None):

    if(sourceFile_ is not None):
        print("load new File '", sourceFile_, "'")
        with fits.open(sourceFile_) as noiseTab:
            drsFileCalibrated_ = noiseTab[1].data["DRSCalibratedDataNoise"]
            drsModelCalibrated_ = noiseTab[1].data["DRSCalibratedData_TempNoise"]
            drsModelIntervalCalibrated_ = noiseTab[1].data["DRSCalibratedData_TempIntervalNoise"]

    pixelList = np.linspace(0, fact.nrPix-1, fact.nrPix, dtype="int")
    usefulPixel = pixelList[np.setdiff1d(pixelList, np.array([nonStandardPixel["crazy"], nonStandardPixel["dead"]])-1)]

    # TODO update maybe use repeat
    drsFileCalibrated = np.array(drsFileCalibrated_).reshape(-1, fact.nrPix)[:, usefulPixel].flatten()/10
    drsModelCalibrated = np.array(drsModelCalibrated_).reshape(-1, fact.nrPix)[:, usefulPixel].flatten()/10
    drsModelIntervalCalibrated = np.array(drsModelIntervalCalibrated_).reshape(-1, fact.nrPix)[:, usefulPixel].flatten()/10

    if(maxNoise_ is None):
        maxNoise_ = math.ceil(np.amax([drsFileCalibrated, drsModelCalibrated, drsModelIntervalCalibrated]))

    nrBins = int(maxNoise_/0.01)
    weights = np.full(len(drsFileCalibrated), 100/len(drsFileCalibrated))

    drsFileCalibratedMean = np.mean(drsFileCalibrated)
    drsModelIntervalCalibratedMean = np.mean(drsModelIntervalCalibrated)

    drsFileCalibratedStd = np.std(drsFileCalibrated, dtype="float64", ddof=1)
    drsModelIntervalCalibratedStd = np.std(drsModelIntervalCalibrated, dtype="float64", ddof=1)

    gs = gridspec.GridSpec(4, 1)
    #ax0 = plt.subplot(gs[0:3, :])
    ax0 = plt.subplot(gs[0:4, :])
    plt.title(titleStr_, y=1.0)
    hist1 = ax0.hist(drsFileCalibrated, weights=weights, bins=nrBins, histtype='step',
                     range=(0.0, maxNoise_), lw=1, edgecolor="r", label="Drs-File Noise\nMean: "+str(format(round(drsFileCalibratedMean, 3), '.3f'))+", Std: "+str(format(round(drsFileCalibratedStd, 3), '.3f')))
    #hist2 = ax0.hist(drsModelCalibrated, weights=weights, bins=nrBins, histtype='step',
    #                 range=(0.0, maxNoise_), lw=1, edgecolor="g", label="Model Noise")
    hist3 = ax0.hist(drsModelIntervalCalibrated, weights=weights, bins=nrBins, histtype='step',
                     range=(0.0, maxNoise_), lw=1, edgecolor="b", label="Model Noise\nMean: "+str(format(round(drsModelIntervalCalibratedMean, 3), '.3f'))+", Std: "+str(format(round(drsModelIntervalCalibratedStd, 3), '.3f')))# label="Model Interval Noise")

    plt.ylabel(r'frequency /$\mathrm{\%}$')
    plt.legend(loc='upper right', numpoints=1, title="")

    #ax1 = plt.subplot(gs[3, :])
    #ax1.step(hist1[1][0:-1], hist2[0]-hist1[0], "g")
    #ax1.step(hist1[1][0:-1], hist3[0]-hist1[0], "b")
    plt.xlabel(r'Noise /$\mathrm{PEA}$')
    plt.ylabel(r'$\Delta$ frequency /$\mathrm{\%}$')
    #plt.setp(ax0.get_xticklabels(), visible=False)
    plt.text(0.7, 0.15, "preliminary", fontdict=font, transform=plt.gca().transAxes)
    if(storeFilename_ is not None):
        plt.savefig(storeFilename_)
    plt.show()
    plt.close()


####################################################################################################
def noiseFactCam(drsFileCalibrated_, drsModelCalibrated_, drsModelIntervalCalibrated_,
                 storeFilename_=None, sourceFile_=None):

    if(sourceFile_ is not None):
        print("load new File '", sourceFile_, "'")
        with fits.open(sourceFile_) as noiseTab:
            drsFileCalibrated_ = noiseTab[1].data["DRSCalibratedDataNoise"].flatten()
            drsModelCalibrated_ = noiseTab[1].data["DRSCalibratedData_TempNoise"].flatten()
            drsModelIntervalCalibrated_ = noiseTab[1].data["DRSCalibratedData_TempIntervalNoise"].flatten()

    pixelList = np.linspace(0, fact.nrPix-1, fact.nrPix, dtype="int")
    usefulPixel = pixelList[np.setdiff1d(pixelList, np.array([nonStandardPixel["crazy"], nonStandardPixel["dead"]])-1)]

    drsFileCalibratedPixelMean = np.mean(np.array(drsFileCalibrated_).reshape(-1, fact.nrPix), axis=0)
    drsModelCalibratedPixelMean = np.mean(np.array(drsModelCalibrated_).reshape(-1, fact.nrPix), axis=0)
    drsModelIntervalCalibratedPixelMean = np.mean(np.array(drsModelIntervalCalibrated_).reshape(-1, fact.nrPix), axis=0)

    nonStandardPixelIndices = np.array([nonStandardPixel["crazy"], nonStandardPixel["dead"]]).flatten()-1
    drsFileCalibratedPixelMean[nonStandardPixelIndices] = 0.

    plot = camera(drsFileCalibratedPixelMean, cmap='hot')
    plt.colorbar(plot, label="Noise /$\mathrm{mv}$")
    if(storeFilename_ is not None):
        plt.savefig(storeFilename_+"_org.jpg")
    plt.show()
    plt.close()

    plot = camera(drsModelCalibratedPixelMean, cmap='hot')
    plt.colorbar(plot, label="Noise /$\mathrm{mv}$")
    if(storeFilename_ is not None):
        plt.savefig(storeFilename_+"_temp.jpg")
    plt.show()
    plt.close()

    plot = camera(drsModelIntervalCalibratedPixelMean, cmap='hot')
    plt.colorbar(plot, label="Noise /$\mathrm{mv}$")
    if(storeFilename_ is not None):
        plt.savefig(storeFilename_+"_tempInt.jpg")
    plt.show()
    plt.close()


# TODO update
####################################################################################################
def pedestialNoise(filename_, save_=False):

    with fits.open(filename_) as tabNoise:
        date = pd.to_datetime(tabNoise[1].header["Date"]).date()
        tempDiff = tabNoise[1].data["TempDiff"]
        runIds = tabNoise[1].data["PedestelRunId"]
        drsCalibratedDataNoise = tabNoise[1].data["drsCalibratedDataNoise"]
        drsCalibratedData_TempNoise = tabNoise[1].data["drsCalibratedDataNoise"]#tabNoise[1].data["drsCalibratedData_TempNoise"]
        drsCalibratedData_TempIntervalNoise = tabNoise[1].data["drsCalibratedData_TempIntervalNoise"]

    runIds = [runIds for
              (tempDiff, runIds) in
              sorted(zip(tempDiff, runIds),
                     key=lambda pair: pair[0])]

    drsCalibratedDataNoise = [drsCalibratedDataNoise for
                              (tempDiff, drsCalibratedDataNoise) in
                              sorted(zip(tempDiff, drsCalibratedDataNoise),
                                     key=lambda pair: pair[0])]

    drsCalibratedData_TempNoise = [drsCalibratedData_TempNoise for
                                   (tempDiff, drsCalibratedData_TempNoise) in
                                   sorted(zip(tempDiff, drsCalibratedData_TempNoise),
                                          key=lambda pair: pair[0])]

    drsCalibratedData_TempIntervalNoise = [drsCalibratedData_TempIntervalNoise for
                                           (tempDiff, drsCalibratedData_TempIntervalNoise) in
                                           sorted(zip(tempDiff, drsCalibratedData_TempIntervalNoise),
                                                  key=lambda pair: pair[0])]

    tempDiff = sorted(tempDiff)

    print("date ", date)
    print("runIds ", runIds)
    print("tempDiff ", tempDiff)

    dateStr = date.strftime('%Y-%m-%d')
    dateStr2 = date.strftime('%Y%m%d')
    maxNoise = math.ceil(max(np.array([list(drsCalibratedDataNoise),
                                       list(drsCalibratedData_TempNoise),
                                       list(drsCalibratedData_TempIntervalNoise)]).flatten()))

    print("maxNoise: ", maxNoise)
    maxNoise = 0.5
    for i in range(len(drsCalibratedDataNoise)):
        storeFilename = "../plots/noise/"+dateStr2+"/pedestelNoise"+dateStr+"_runId"+str(runIds[i])+".jpg"
        #titleStr = ("Standard deviation "+dateStr+"\n " +
        #            "runID: "+str(runIds[i])+", Temperature difference "+str(round(tempDiff[i], 3))+r"$^\circ C$")
        titleStr = ("Temperature difference "+str(round(tempDiff[i], 3))+r"$^\circ C$")
        noise(drsCalibratedDataNoise[i], drsCalibratedData_TempNoise[i], drsCalibratedData_TempIntervalNoise[i],
              titleStr, maxNoise, storeFilename)
        storeFilename = "../plots/noise/"+dateStr2+"/pedestelNoiseFactCam"+dateStr+"_runId"+str(runIds[i])
        #noiseFactCam(drsCalibratedDataNoise[i], drsCalibratedData_TempNoise[i], drsCalibratedData_TempIntervalNoise[i],
        #             storeFilename)
    return#
    data = abs(np.sum(tempHistVec, axis=0)+np.sum(tempHistIntervalVec, axis=0)-2*np.sum(orgHistVec, axis=0))

    y = 0
    sum = 0
    sumData = np.sum(data)
    while (sum < sumData*0.92):
        sum += data[y]
        y += 1

    nrOfHists = len(orgHistVec)

    plt.title("Standard deviation between slice 10 and 250", fontsize=16, y=1.02)
    plotList = []
    labelList = []
    for i in range(nrOfHists):
        alpha = (i+1)/nrOfHists*(3/4) + 0.25
        plot, = plt.plot([-1, -1], [-1, -1], 'k--', alpha=alpha, dashes=(5, 1+(i*10)))
        plotList.append(plot)
        labelList.append(str(round(tempDiff[i], 1))+r" $^\circ$ C")
        plt.plot(pos, tempHistVec[i]-orgHistVec[i], 'b--', alpha=alpha, dashes=(5, 1+(i*10)))
        plt.plot(pos, tempHistIntervalVec[i]-orgHistVec[i], 'r--', alpha=alpha, dashes=(5, 1+(i*10)))

    l_1, = plt.plot([], [], "bo")
    l_2, = plt.plot([], [], "ro")
    plt.ylabel(r'TemperatureHistogram /$\mathrm{\%}$ - FileHistogram /$\mathrm{\%}$')
    plt.xlabel(r'standard deviation /$\mathrm{mv}$')
    plt.xlim(0, y*step)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 3))
    legend1 = plt.legend([l_1, l_2], ["All TempDiff", "Interval TempDiff"], loc='upper right', numpoints=1)
    plt.legend(plotList, labelList, loc='lower right', title="Temperature difference:")
    plt.gca().add_artist(legend1)
    plt.text(0.02, 0.05, "preliminary", fontdict=font, transform=plt.gca().transAxes)
    if(save_ is True):
        plt.savefig("../plots/noise/pedestelNoise"+dateStr+".pdf")
    plt.show()
    plt.close()
