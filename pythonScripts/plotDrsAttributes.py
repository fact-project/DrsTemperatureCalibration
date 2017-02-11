import matplotlib.pyplot as plt
import matplotlib.dates as dates
import numpy as np
import pandas as pd
import h5py
import os

from astropy.io import fits
from matplotlib.cm import hot

####################################################################################################
####################################################################################################
# path to your local mounted isdc-in04-Folder (isdc-in04:/fact)
isdcPath = "/home/florian/Dokumente/Uni/Master/Masterarbeit/isdcFact/"
####################################################################################################
####################################################################################################


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
def baselineStdHist(dateNr_, pixelNr_=5,  nrBins_=150):

    with h5py.File('/gpfs/scratch/schulz/drsData.h5', 'r') as store:
        baselineMeanStd = np.array(store['baselineMeanStd'][dateNr_])

    plt.hist(baselineMeanStd, bins=nrBins_, range=(0, max(baselineMeanStd)))
    plt.title(r"Histogram of baselineStd", fontsize=16, y=1.01)
    plt.ylabel('Count [1]')
    plt.xlabel(r'baselineStd U [$mV$]')
    plt.show()
    plt.close()


####################################################################################################
def gainStdHist(dateNr_, pixelNr_=5, nrBins_=150):

    with h5py.File('/gpfs/scratch/schulz/drsData.h5', 'r') as store:
        gainMeanStd = np.array(store['gainMeanStd'][dateNr_])

        plt.hist(gainMeanStd, bins=nrBins_, range=(0, max(gainMeanStd)))
        plt.title(r"Histogram of gainStd", fontsize=16, y=1.01)
        plt.ylabel('Count [1]')
        plt.xlabel(r'gainStd U [$mV$]')
        plt.show()
        plt.close()


####################################################################################################
def pixelCapacitorBaseline(pixelNr_, condNr_, errFactor_=2.0, showStd_=False,
                           subTimeInterval_=None, save_=False):

    with h5py.File('/gpfs/scratch/schulz/drsData.h5', 'r') as store:
        time = np.array(store['time_baseline']).flatten()
        temp = np.array(store['temp_baseline'][:, int(pixelNr_/9)])
        tempStd = np.array(store['tempStd_baseline'][:, int(pixelNr_/9)])
        baseline = np.array(store['baselineMean'][:, pixelNr_*1024 + condNr_])
        baselineStd = np.array(store['baselineMeanStd'][:, pixelNr_*1024 + condNr_])

    baselineStdMean = np.mean(baselineStd, dtype='float64')
    indices = np.where(baselineStd < baselineStdMean*errFactor_)[0]
    indices_ = np.where(baselineStd > baselineStdMean*errFactor_)[0]

    sc_all = plt.scatter(temp, baseline, c=time)
    plt.close()  # Dont show sc_all, just use it to create the colorbar

    fig, img = plt.subplots()

    intervalMonth = 3
    datetime = pd.to_datetime(time * 24 * 3600 * 1e9)
    timeLabel = pd.date_range(start=datetime[0], end=datetime[-1], freq=str(intervalMonth)+"M")
    cbar = fig.colorbar(sc_all, ticks=dates.MonthLocator(interval=intervalMonth))
    cbar.ax.set_yticklabels(timeLabel.strftime("%b %Y"))
    timeColor = cbar.to_rgba(time)

    sc = img.scatter(temp[indices], baseline[indices], s=50, marker="+",
                     c=timeColor[indices], label="Normal")

    sc_ = None
    if(len(indices_) > 0):
        sc_ = img.scatter(temp[indices_], baseline[indices_], s=50, marker="*",
                          c=timeColor[indices_], label="Not used")

    if(showStd_ is True):
        for i in range(len(temp)):
            plt.errorbar(temp[i], baseline[i],
                         xerr=tempStd[i], yerr=baselineStd[i],
                         color=timeColor[i], marker='', ls='', zorder=0)

    temperature = np.linspace(min(temp)-1, max(temp)+1, 1000)
    var, cov = getLinearFitValues(temp[indices], baseline[indices], baselineStd[indices])
    fit = linearerFit(temperature, var[0], var[1])

    fitPlot, = plt.plot(temperature, fit, "k-",
                        label=("    All: a="+str('{:0.3f}'.format(round(var[0], 3)))+r"$\pm$" +
                               str('{:0.3f}'.format(round(np.sqrt(cov[0][0]), 3))) +
                               ", b="+str('{:0.3f}'.format(round(var[1], 3)))+r"$\pm$" +
                               str('{:0.3f}'.format(round(np.sqrt(cov[1][1]), 3)))))

    sc2 = None
    fitPlot2 = None
    if (subTimeInterval_ is not None and len(subTimeInterval_) == 2):
        startDatetime, endDatetime = pd.to_datetime(subTimeInterval_[0]), pd.to_datetime(subTimeInterval_[1])
        indicesSub = np.intersect1d(indices, np.where((datetime > startDatetime) & (datetime < endDatetime))[0])

        var2, cov2 = getLinearFitValues(temp[indicesSub], baseline[indicesSub], baselineStd[indicesSub])
        fit2 = linearerFit(temperature, var2[0], var2[1])
        sc2 = img.scatter(temp[indicesSub], baseline[indicesSub], s=50, marker="o",
                          c=timeColor[indicesSub], label="2016")

        fitPlot2, = plt.plot(temperature, fit2, "r-",
                             label=("2016: a="+str('{:0.3f}'.format(round(var2[0], 3)))+r"$\pm$" +
                                    str('{:0.3f}'.format(round(np.sqrt(cov2[0][0]), 3))) +
                                    ", b="+str('{:0.3f}'.format(round(var2[1], 3)))+r"$\pm$" +
                                    str('{:0.3f}'.format(round(np.sqrt(cov2[1][1]), 3)))))

    plt.title("Baselinemean: Pixel: "+str(pixelNr_)+", Capacitor: "+str(condNr_) +
              ", ErrFactor: "+str('{:0.1f}'.format(errFactor_)), fontsize=15, y=1.02)

    plt.xlabel(r'Temperature [$^\circ C$]')
    plt.ylabel(r'BaselineMean [$mV$]')
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
                              title="BaselineMean with averaged Temperature")
    plt.gca().add_artist(first_legend)
    plt.legend(handles=handles2, loc="lower left", numpoints=1,
               title="Linearerfit: "+r"$f(x) = a\cdot x + b$")
    if(save_ is True):
        if not os.path.exists('plots'):
            os.mkdir('plots')
        dateStr = ""
        if(sc2 is not None and fitPlot2 is not None):
            dateStr = dateStr+"Sub"+getDateString(pd.to_datetime(subTimeInterval_[0]),
                                                  pd.to_datetime(subTimeInterval_[1]), ["_", ""])
        plt.savefig("plots/baselineMean_P"+str('{:04d}'.format(pixelNr_))+"_C"+str('{:04d}'.format(condNr_)) +
                    dateStr+".png")
    plt.show()
    plt.close()


####################################################################################################
def pixelCapacitorGain(pixelNr_, condNr_, errFactor_=2.0, showStd_=False,
                       subTimeInterval_=None, save_=False):

    with h5py.File('/gpfs/scratch/schulz/drsData.h5', 'r') as store:
        time = np.array(store['time_gain']).flatten()
        temp = np.array(store['temp_gain'][:, int(pixelNr_/9)])
        tempStd = np.array(store['tempStd_gain'][:, int(pixelNr_/9)])
        gain = np.array(store['gainMean'][:, pixelNr_*1024 + condNr_])
        gainStd = np.array(store['gainMeanStd'][:, pixelNr_*1024 + condNr_])

    gainStdMean = np.mean(gainStd, dtype='float64')
    indices = np.where(gainStd < gainStdMean*errFactor_)[0]
    indices_ = np.where(gainStd > gainStdMean*errFactor_)[0]

    sc_all = plt.scatter(temp, gain, c=time)
    plt.close()  # Dont show sc_all, just use it to create the colorbar

    fig, img = plt.subplots()

    intervalMonth = 3
    datetime = pd.to_datetime(time * 24 * 3600 * 1e9)
    timeLabel = pd.date_range(start=datetime[0], end=datetime[-1], freq=str(intervalMonth)+"M")
    cbar = fig.colorbar(sc_all, ticks=dates.MonthLocator(interval=intervalMonth))
    cbar.ax.set_yticklabels(timeLabel.strftime("%b %Y"))
    timeColor = cbar.to_rgba(time)

    sc = img.scatter(temp[indices], gain[indices], s=50, marker="+",
                     c=timeColor[indices], label="Normal")

    sc_ = None
    if(len(indices_) > 0):
        sc_ = img.scatter(temp[indices_], gain[indices_], s=50, marker="*",
                          c=timeColor[indices_], label="Not used")

    if(showStd_ is True):
        for i in range(len(temp)):
            plt.errorbar(temp[i], gain[i],
                         xerr=tempStd[i], yerr=gainStd[i],
                         color=timeColor[i], marker='', ls='', zorder=0)

    temperature = np.linspace(min(temp)-1, max(temp)+1, 1000)
    var, cov = getLinearFitValues(temp[indices], gain[indices], gainStd[indices])
    fit = linearerFit(temperature, var[0], var[1])

    fitPlot, = plt.plot(temperature, fit, "g-",
                        label=("    All: a="+str('{:0.3f}'.format(round(var[0], 3)))+r"$\pm$" +
                               str('{:0.3f}'.format(round(np.sqrt(cov[0][0]), 3))) +
                               ", b="+str('{:0.3f}'.format(round(var[1], 3)))+r"$\pm$" +
                               str('{:0.3f}'.format(round(np.sqrt(cov[1][1]), 3)))))

    sc2 = None
    fitPlot2 = None
    if (subTimeInterval_ is not None and len(subTimeInterval_) == 2):
        startDatetime, endDatetime = pd.to_datetime(subTimeInterval_[0]), pd.to_datetime(subTimeInterval_[1])
        indicesSub = np.intersect1d(indices, np.where((datetime > startDatetime) & (datetime < endDatetime))[0])

        var2, cov2 = getLinearFitValues(temp[indicesSub], gain[indicesSub], gainStd[indicesSub])
        fit2 = linearerFit(temperature, var2[0], var2[1])
        sc2 = img.scatter(temp[indicesSub], gain[indicesSub], s=50, marker="o",
                          c=timeColor[indicesSub], label="2016")

        fitPlot2, = plt.plot(temperature, fit2, "r-",
                             label=("2016: a="+str('{:0.3f}'.format(round(var2[0], 3)))+r"$\pm$" +
                                    str('{:0.3f}'.format(round(np.sqrt(cov2[0][0]), 3))) +
                                    ", b="+str('{:0.3f}'.format(round(var2[1], 3)))+r"$\pm$" +
                                    str('{:0.3f}'.format(round(np.sqrt(cov2[1][1]), 3)))))

    plt.title("GainMean: Pixel: "+str(pixelNr_)+", Capacitor: "+str(condNr_) +
              ", ErrFactor: "+str(errFactor_), fontsize=15, y=1.02)

    plt.xlabel(r'Temperature [$^\circ C$]')
    plt.ylabel(r'GainMean [$mV$]')
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
                              title="BaselineMean with averaged Temperature")
    plt.gca().add_artist(first_legend)
    plt.legend(handles=handles2, loc="lower left", numpoints=1,
               title="Linearerfit: "+r"$f(x) = a\cdot x + b$")
    if(save_ is True):
        if not os.path.exists('plots'):
            os.mkdir('plots')
        dateStr = ""
        if(sc2 is not None and fitPlot2 is not None):
            dateStr = dateStr+"Sub"+getDateString(pd.to_datetime(subTimeInterval_[0]),
                                                  pd.to_datetime(subTimeInterval_[1]), ["_", ""])
        plt.savefig("plots/gainMean_P"+str('{:04d}'.format(pixelNr_))+"_C"+str('{:04d}'.format(condNr_)) +
                    dateStr+".png")
    plt.show()
    plt.close()


####################################################################################################
def pixelFitValues(pixelNr_, variable_, part_, showStd_=False, save_=False):

    fitValuesTab = fits.open('/gpfs/scratch/schulz/fitValuesData.fits', ignoremissing=True, ignore_missing_end=True)
    slopeBase = fitValuesTab["baseline"].data["slope"][0][1024*pixelNr_:1024*(pixelNr_+1)]
    slopeStdBase = fitValuesTab["baseline"].data["slopeStd"][0][1024*pixelNr_:1024*(pixelNr_+1)]
    offsetBase = fitValuesTab["baseline"].data["offset"][0][1024*pixelNr_:1024*(pixelNr_+1)]
    offsetStdBase = fitValuesTab["baseline"].data["offsetStd"][0][1024*pixelNr_:1024*(pixelNr_+1)]

    slopeGain = fitValuesTab["gain"].data["slope"][0][1024*pixelNr_:1024*(pixelNr_+1)]
    slopeStdGain = fitValuesTab["gain"].data["slopeStd"][0][1024*pixelNr_:1024*(pixelNr_+1)]
    offsetGain = fitValuesTab["gain"].data["offset"][0][1024*pixelNr_:1024*(pixelNr_+1)]
    offsetStdGain = fitValuesTab["gain"].data["offsetStd"][0][1024*pixelNr_:1024*(pixelNr_+1)]

    variableTerms = [["Baseline", "baseline"], ["Gain", "gain"]]
    partTerms = [["Slope", "slope", "m"], ["Offset", "offset", "b"]]

    labelStr = ''
    variableStr = ''
    partStr = ''
    errStr = ''

    colors = hot(np.linspace(0, 0, 1024))
    for i in range(33):
        colors[i*32-1] = [1., 0., 0., 1.]

    Capacitor = np.linspace(0, 1023, 1024)
    if(variable_ in variableTerms[0]):
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
            labelStr = partStr+r" [$mV$]"
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

    elif(variable_ in variableTerms[1]):
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
            labelStr = partStr+r" [$mV$]"
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
        print("Unknown variable '", variable_, "' - expected '",
              variableTerms[0][0], "' or '", variableTerms[1][0], "'")
        return

    plt.title(variableStr+r" Fit $f(x) = m \cdot x + b$"+"\n"+partStr+" Pixel:"+str(pixelNr_), fontsize=16, y=1.01)
    plt.xlabel('Capacitor [1]')
    plt.ylabel(labelStr)
    plt.xlim(-1, 1024)
    plt.grid()
    # plt.legend(loc='upper right', scatterpoints=1, numpoints=1)
    if(save_ is True):
        if not os.path.exists('plots'):
            os.mkdir('plots')
        plt.savefig("plots/"+variableStr+partStr+"_P"+str('{:04d}'.format(pixelNr_))+errStr+".png")
    plt.show()
    plt.close()


####################################################################################################
def fitValuesStd(bar_=150, save_=False):

    fitValuesTab = fits.open('/gpfs/scratch/schulz/fitValuesData.fits', ignoremissing=True, ignore_missing_end=True)
    slopeBaseStd = np.array(fitValuesTab["baseline"].data["slopeStd"][0])
    offsetBaseStd = np.array(fitValuesTab["baseline"].data["offsetStd"][0])
    slopeGainStd = np.array(fitValuesTab["gain"].data["slopeStd"][0])
    offsetGainStd = np.array(fitValuesTab["gain"].data["offsetStd"][0])

    xRange = np.linspace(0, 1440-1, 1440)

    slopeBaseStdPerPixel = []
    offsetBaseStdPerPixel = []
    slopeGainStdPerPixel = []
    offsetGainStdPerPixel = []

    for i in range(1440):
        slopeBaseStdPerPixel.append(sum(slopeBaseStd[i*1024:(i+1)*1024])/1024)
        offsetBaseStdPerPixel.append(sum(offsetBaseStd[i*1024:(i+1)*1024])/1024)
        slopeGainStdPerPixel.append(sum(slopeGainStd[i*1024:(i+1)*1024])/1024)
        offsetGainStdPerPixel.append(sum(offsetGainStd[i*1024:(i+1)*1024])/1024)

    barlist = plt.bar(xRange-0.5, slopeBaseStdPerPixel, width=1., color='b', edgecolor='none')
    for i in range(1440):
        if(i % 9 == 0):
            barlist[i-1].set_color('r')
    plt.title(r"BaselineMean Fit $f(x) = m \cdot x + b$"+"\n"+"Mean SlopeStd per Pixel", fontsize=16, y=1.01)
    plt.ylabel(r'slopeStd U[$mV$]')
    plt.xlabel('PixelNr')
    plt.xlim(-1, 1440)
    plt.show()
    plt.close()

    barlist = plt.bar(xRange-0.5, offsetBaseStdPerPixel, width=1., color='b', edgecolor='none')
    for i in range(1440):
        if(i % 9 == 0):
            barlist[i-1].set_color('r')
    plt.title(r"BaselineMean Fit $f(x) = m \cdot x + b$"+"\n"+"Mean OffsetStd per Pixel", fontsize=16, y=1.01)
    plt.ylabel(r'offsetStd U[$mV$]')
    plt.xlabel('PixelNr')
    plt.xlim(-1, 1440)
    plt.show()
    plt.close()

    barlist = plt.bar(xRange-0.5, slopeGainStdPerPixel, width=1., color='b', edgecolor='none')
    for i in range(1440):
        if(i % 9 == 0):
            barlist[i-1].set_color('r')
    plt.title(r"GainMean Fit $f(x) = m \cdot x + b$"+"\n"+"Mean slopeStd per Pixel", fontsize=16, y=1.01)
    plt.ylabel(r'slopeStd U[$mV$]')
    plt.xlabel('PixelNr')
    plt.xlim(-1, 1440)
    plt.show()
    plt.close()

    barlist = plt.bar(xRange-0.5, offsetGainStdPerPixel, width=1., color='b', edgecolor='none')
    for i in range(1440):
        if(i % 9 == 0):
            barlist[i-1].set_color('r')
    plt.title(r"GainMean Fit $f(x) = m \cdot x + b$"+"\n"+"Mean offsetStd per Pixel", fontsize=16, y=1.01)
    plt.ylabel(r'offsetStd U[$mV$]')
    plt.xlabel('PixelNr')
    plt.xlim(-1, 1440)
    plt.show()
    plt.close()


####################################################################################################
def pixelCapacitorFitResidumBaseline(pixelNr_, condNr_, save_=False):

    with h5py.File('/gpfs/scratch/schulz/drsData2016.h5', 'r') as store:
        time = np.array(store['time_baseline']).flatten()
        temp = np.array(store['temp_baseline'][:, int(pixelNr_/9)])
        baselineMean = np.array(store['baselineMean'][:, pixelNr_*1024 + condNr_])

    fitValuesTab = fits.open('/gpfs/scratch/schulz/fitValuesData2016.fits', ignoremissing=True, ignore_missing_end=True)

    datetime = pd.to_datetime(time * 24 * 3600 * 1e9)
    slope = fitValuesTab["baseline"].data["slope"][0][pixelNr_*1024 + condNr_]
    offset = fitValuesTab["baseline"].data["offset"][0][pixelNr_*1024 + condNr_]

    baselineMeanDiff = baselineMean-(slope*temp + offset)

    plt.plot(np.array(datetime), np.array(baselineMeanDiff).astype('float64'), "kx",
             label=(r"baselineMean - f(t)"))
    plt.title("BaselineMeanResidum: \n"+"Pixel: "+str(pixelNr_)+" Capacitor: "+str(condNr_))
    plt.ylabel('Difference [mV]')
    plt.gcf().autofmt_xdate()
    plt.xlim(min(datetime)-pd.DateOffset(days=1), max(datetime)+pd.DateOffset(days=1))
    plt.ylim(min(baselineMeanDiff), max(baselineMeanDiff))
    plt.grid()
    plt.legend(loc='lower left', numpoints=1)
    if(save_ is True):
        if not os.path.exists('plots'):
            os.mkdir('plots')
        plt.savefig("plots/baselineMeanResidum_P"+str('{:04d}'.format(pixelNr_)) +
                    "_C"+str('{:04d}'.format(condNr_))+"2016.png")
    plt.show()
    plt.close()


####################################################################################################
def pixelCapacitorFitResidumGain(pixelNr_, condNr_, showStd_=False, save_=False):

    with h5py.File('/gpfs/scratch/schulz/drsData2016.h5', 'r') as store:
        time = np.array(store['time_gain']).flatten()
        temp = np.array(store['temp_gain'][:, int(pixelNr_/9)])
        gainMean = np.array(store['gainMean'][:, pixelNr_*1024 + condNr_])

    fitValuesTab = fits.open('/gpfs/scratch/schulz/fitValuesData2016.fits', ignoremissing=True, ignore_missing_end=True)

    datetime = pd.to_datetime(time * 24 * 3600 * 1e9)

    slope = fitValuesTab["gain"].data["slope"][0][pixelNr_*1024 + condNr_]
    offset = fitValuesTab["gain"].data["offset"][0][pixelNr_*1024 + condNr_]

    gainMeanDiff = gainMean-(slope*temp + offset)

    plt.plot(np.array(datetime), np.array(gainMeanDiff).astype('float64'), "kx",
             label=(r"baselineMean - f(t)"))
    plt.title("GainMeanResidum: \n"+"Pixel: "+str(pixelNr_)+" Capacitor: "+str(condNr_))
    plt.ylabel('Diff [mV]')
    plt.gcf().autofmt_xdate()
    plt.xlim(min(datetime)-pd.DateOffset(days=1), max(datetime)+pd.DateOffset(days=1))
    plt.ylim(min(gainMeanDiff), max(gainMeanDiff))
    plt.grid()
    plt.legend(loc='lower left', numpoints=1)
    if(save_ is True):
        if not os.path.exists('plots'):
            os.mkdir('plots')
        plt.savefig("plots/gainMeanResidum_P"+str('{:04d}'.format(pixelNr_)) +
                    "_C"+str('{:04d}'.format(condNr_))+"2016.png")
    plt.show()
    plt.close()


####################################################################################################
def noise(path_="/home/florian/Dokumente/Uni/Master/Masterarbeit/noise/",
          filename_=["DrsCalibratedDataStd_40.fit", "DrsCalibratedDataStd_32.fit", "DrsCalibratedDataStd_10.fit"],
          limits_=[0.0, 40.0], nrBins_=200, save_=False):

    date = pd.to_datetime("2016-10-19")

    time_list = None
    temp_list = None
    drsTempMean = None
    filename = (isdcPath+"aux/" +
                str(date.year)+"/"+str('{:02d}'.format(date.month))+"/"+str('{:02d}'.format(date.day)) +
                "/"+str(date.year)+str('{:02d}'.format(date.month))+str('{:02d}'.format(date.day)) +
                ".FAD_CONTROL_TEMPERATURE.fits")
    folder = isdcPath+"raw/"+str(date.year)+"/"+str('{:02d}'.format(date.month))+"/"+str('{:02d}'.format(date.day))
    if(os.path.isfile(filename) and os.path.isdir(folder)):
        # print("found: ", filename, "and", folder)
        with fits.open(filename) as tab_temp:
            time_list = np.array(tab_temp[1].data['Time'])
            temp_list = np.array(tab_temp[1].data['temp'])

        temp_datetime = pd.to_datetime(time_list * 24 * 3600 * 1e9)
        filename = "20161019_"+'{:03d}'.format(7)+".drs.fits.gz"
        with fits.open(folder+"/"+filename) as tab_drs:
            drsStart = pd.to_datetime(tab_drs[1].header["DATE-OBS"])
            drsEnd = pd.to_datetime(tab_drs[1].header["DATE-END"])
            # mean ignore patches -->, axis=0 <--
            drsTempMean = np.mean(temp_list[np.where((temp_datetime > drsStart) & (temp_datetime < drsEnd))])

    sourceTempMeanVec = []
    nr = [40, 32, 10]
    for i in range(len(nr)):
        filename = "20161019_"+'{:03d}'.format(nr[i])+".fits.fz"
        with fits.open(folder+"/"+filename) as sourceTab:
            sourceStart = sourceTab[2].header["DATE-OBS"]
            sourceEnd = sourceTab[2].header["DATE-END"]
            temp = temp_list[np.where((temp_datetime > sourceStart) & (temp_datetime < sourceEnd))[0]]
            if(len(temp) == 0):
                temp = temp_list[np.where((temp_datetime > sourceStart))[0][0],
                                 np.where((temp_datetime < sourceEnd))[0][-1]]
            sourceTempMeanVec.append(np.mean(temp))

    events = []
    orgHistVec = []
    tempHistVec = []
    tempHist2016Vector = []
    for filename in filename_:
        with fits.open(path_+filename) as store:
            org = store[1].data['DrsCalibratedDataStd']
            temp = store[1].data['DrsTemperatureCalibratedDataStd']
            temp2016 = store[1].data['DrsTemperatureCalibratedDataStd2016']

        events.append(org.shape[0])
        print("Events: ", events[-1])

        orgData = np.reshape(org, org.shape[0]*org.shape[1])
        tempData = np.reshape(temp, temp.shape[0]*temp.shape[1])
        temp2016Data = np.reshape(temp2016, temp2016.shape[0]*temp2016.shape[1])

        print("mean org: ", np.mean(orgData))
        print("temp org: ", np.mean(tempData))
        print("temp2016 org: ", np.mean(temp2016Data))

        orgHistVec.append(plt.hist(orgData, bins=nrBins_, range=(limits_[0], limits_[1]), normed=True)[0])
        tempHistVec.append(plt.hist(tempData, bins=nrBins_, range=(limits_[0], limits_[1]), normed=True)[0])
        tempHist2016Vector.append(plt.hist(temp2016Data, bins=nrBins_, range=(limits_[0], limits_[1]), normed=True)[0])
        plt.close()

    step = (limits_[1]-limits_[0])/nrBins_/2
    pos = np.linspace(limits_[0]-step, limits_[1]-step, nrBins_+1)[:-1]

    plt.plot(pos, orgHistVec[0], 'g--', dashes=(5, 1), label="Org: ")
    plt.plot(pos, tempHistVec[0], 'b--', dashes=(5, 1), label="temp")
    plt.plot(pos, tempHist2016Vector[0], 'r--', dashes=(5, 1), label="2016")
    plt.ylabel('Counts [1]')
    plt.xlabel(r'standard deviation [$mV$]')
    plt.xlim(limits_[0], 20)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 3))
    plt.legend(loc='upper right', scatterpoints=1, numpoints=1, title="")
    plt.savefig("plots/000.png")
    plt.show()
    plt.close()

    plt.title("Standard deviation between slice 10 and 250", fontsize=16, y=1.02)
    for i in range(len(orgHistVec)):
        plt.plot(pos, tempHistVec[i]-orgHistVec[i], 'b--', dashes=(5, 1+(i*5)),
                 label="All TempDiff: "+str(round(abs(sourceTempMeanVec[i]-drsTempMean), 1)))
        plt.plot(pos, tempHist2016Vector[i]-orgHistVec[i], 'r--', dashes=(5, 1+(i*5)),
                 label="2016 TempDiff: "+str(round(abs(sourceTempMeanVec[i]-drsTempMean), 1)))
    plt.ylabel('TemperatureHistogram - FileHistogram (Counts [1])')
    plt.xlabel(r'standard deviation [$mV$]')
    plt.xlim(limits_[0], 20)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 3))
    plt.legend(loc='upper right', scatterpoints=1, numpoints=1, title="")
    if(save_ is True):
        if not os.path.exists('plots'):
            os.mkdir('plots')
        plt.savefig("plots/"+filename_[0]+".png")
    plt.show()
    plt.close()


####################################################################################################
####################################################################################################
def meanPatchTemperature(start_date_="2016-01-01", end_date_="2016-12-31", freq_="D", save_=False):

    temp_list = np.array([]).reshape(0, 160)
    for date in pd.date_range(start=start_date_, end=end_date_, freq=freq_):
        filename = (isdcPath+"aux/" +
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
                temp_list = np.append(temp_list, tempValue, axis=0)

    tempMean_list = np.mean(temp_list, dtype='float64', axis=0)
    tempMeanStd_list = np.std(temp_list, dtype='float64', axis=0, ddof=1)

    width = 1   # the width of the bars
    xRange = np.linspace(0, 159, 160)
    dateStr = getDateString(pd.to_datetime(start_date_), pd.to_datetime(end_date_), [" until ", " "])
    plt.title("Mean Patch temperature\n "+dateStr, fontsize=16, y=1.00)
    plt.bar(xRange-width/2, tempMean_list, width=width, color='r', yerr=tempMeanStd_list)
    plt.xlabel('Patch Nr')
    plt.ylabel(r'Temperature [$^\circ C$]')
    plt.xlim(-1, 160)
    if(save_ is True):
        if not os.path.exists('plots'):
            os.mkdir('plots')
        dateStr = getDateString(pd.to_datetime(start_date_), pd.to_datetime(end_date_), ["_", ""])
        plt.savefig("plots/meanPatchTemperature"+dateStr+".png")
    plt.show()
    plt.close()


####################################################################################################
def patchTemperatureEra(patchNr_, start_date_="2016-01-01", end_date_="2016-12-31", freq_="D", save_=False):

    time_list = np.array([])
    temp_list = np.array([])

    month_before = 0
    for date in pd.date_range(start=start_date_, end=end_date_, freq=freq_):
        if(month_before < date.month):
            month_before = date.month
            print("Month: ", date.month)

        filename = (isdcPath+"aux/" +
                    str(date.year)+"/"+str('{:02d}'.format(date.month))+"/"+str('{:02d}'.format(date.day)) +
                    "/"+str(date.year)+str('{:02d}'.format(date.month))+str('{:02d}'.format(date.day)) +
                    ".FAD_CONTROL_TEMPERATURE.fits")
        if(os.path.isfile(filename)):
            # print("found: ", filename)
            with fits.open(filename) as tab_temp:
                time_list = np.append(time_list, tab_temp[1].data['Time'])
                temp_list = np.append(temp_list, tab_temp[1].data['temp'][::, patchNr_])

    temp_datetime = pd.to_datetime(time_list * 24 * 3600 * 1e9)
    dateStr = getDateString(temp_datetime[0], temp_datetime[-1])

    plt.plot(temp_datetime, temp_list, "g.", ms=10, label="Temperature measurement point")
    plt.title("Temperature trend of Patch: "+str(patchNr_), fontsize=18, y=1.02)

    timeLabel = pd.date_range(start=start_date_, end=end_date_, freq="M") - pd.offsets.MonthBegin(1)
    plt.xticks(timeLabel, timeLabel, rotation=30)
    plt.gca().xaxis.set_major_formatter(dates.DateFormatter("%d.%m.%y"))
    plt.ylabel(r'Temperature [$^\circ C$]')
    plt.grid()
    plt.legend(bbox_to_anchor=(0.01, -0.002, 0.98, .102), ncol=1,
               mode="expand", borderaxespad=0., scatterpoints=1, numpoints=1)
    if(save_ is True):
        if not os.path.exists('plots'):
            os.mkdir('plots')
        dateStr = getDateString(temp_datetime[0], temp_datetime[-1], ["_", ""])
        plt.savefig("plots/patch"+str('{:02d}'.format(patchNr_)) +
                    "Temperature"+dateStr+".png")
    plt.show()
    plt.close()


####################################################################################################
def patchTemperature(patchNr_, date_="2016-06-16", save_=False):

    date = pd.to_datetime(date_)

    time_list = np.array([])
    temp_list = np.array([])

    filename = (isdcPath+"aux/" +
                str(date.year)+"/"+str('{:02d}'.format(date.month))+"/"+str('{:02d}'.format(date.day)) +
                "/"+str(date.year)+str('{:02d}'.format(date.month))+str('{:02d}'.format(date.day)) +
                ".FAD_CONTROL_TEMPERATURE.fits")
    folder = isdcPath+"raw/"+str(date.year)+"/"+str('{:02d}'.format(date.month))+"/"+str('{:02d}'.format(date.day))
    if(os.path.isfile(filename) and os.path.isdir(folder)):
        # print("found: ", filename, "and", folder)
        with fits.open(filename) as tab_temp:
            time_list = np.append(time_list, tab_temp[1].data['Time'])
            temp_list = np.append(temp_list, tab_temp[1].data['temp'][::, patchNr_])

        temp_datetime = pd.to_datetime(time_list * 24 * 3600 * 1e9)
        dateStr = getDateString(temp_datetime[0], temp_datetime[-1])

        plt.title("Temperature trend of Patch: "+str(patchNr_), fontsize=18, y=1.02)
        plt.plot(temp_datetime, temp_list, "g.", ms=15, label="Temperature measurement point")

        for filename in os.listdir(folder):
            if filename.endswith("drs.fits.gz"):
                with fits.open(folder+"/"+filename) as tab_drs:
                    drsStart = pd.to_datetime(tab_drs[1].header["DATE-OBS"])
                    drsEnd = pd.to_datetime(tab_drs[1].header["DATE-END"])
                    tempMean = np.mean(temp_list[np.where((temp_datetime > drsStart) & (temp_datetime < drsEnd))])
                    plt.plot([drsStart, drsEnd], [tempMean, tempMean], linestyle="--", marker="|", color="r", ms=25)

        plt.plot([], [], "r|", ms=15, label="DRS-Runs")

        # timeLabel = pd.date_range(start=temp_datetime[0], end=temp_datetime[-1], freq="H")
        # plt.xticks(timeLabel, timeLabel, rotation=30)
        plt.gca().xaxis.set_major_formatter(dates.DateFormatter("%H:%M"))

        plt.xlabel(dateStr+" [h]")
        plt.ylabel(r'Temperature [$^\circ C$]')
        plt.grid()
        plt.legend(bbox_to_anchor=(0.01, -0.002, 0.98, .102), ncol=2,
                   mode="expand", borderaxespad=0., scatterpoints=1, numpoints=1)
        if(save_ is True):
            if not os.path.exists('plots'):
                os.mkdir('plots')
            dateStr = getDateString(temp_datetime[0], temp_datetime[-1], ["_", ""])
            plt.savefig("plots/patch"+str('{:02d}'.format(patchNr_))+"Temperature"+dateStr+".png")
        plt.show()
        plt.close()
    else:
        print("File '", filename, "' does not exist")


####################################################################################################
def maxPatchTemperatureDifferences(patchNr_, nrBins_=200, save_=False):
    filename = "../data/tempDiffs2016-10-01-2016-10-03.fits"
    with fits.open(filename) as store:
        tempDiffs = np.array(list(store[1].data["maxTempDiff"])).reshape(-1, 160)[::, patchNr_]
        print(tempDiffs)

    plt.hist(tempDiffs, bins=nrBins_, range=(0, max(tempDiffs)), normed=False)
    plt.title(r"Histogram of tempDiffs", fontsize=16, y=1.01)
    plt.ylabel('Count [1]')
    plt.xlabel(r'tempDiffs U [$^\circ C$]')
    if(save_ is True):
        if not os.path.exists('plots'):
            os.mkdir('plots')
        plt.savefig("plots/maxTemperatureDifferencesPatch"+str('{:02d}'.format(patchNr_))+".png")
    plt.show()
    plt.close()
