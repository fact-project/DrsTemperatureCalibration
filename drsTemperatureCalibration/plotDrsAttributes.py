import matplotlib.pyplot as plt
import matplotlib.dates as dates

import os
import numpy as np
import pandas as pd
import math
import h5py
import click
# import yaml

from tqdm import tqdm
from fact.pixels import non_standard_pixel_chids as non_standard_chids
from fact.plotting import camera
from astropy.io import fits
from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.cm import hot

from drsTemperatureCalibration.tools import check_file_match
from drsTemperatureCalibration.constants import NRCHID, NRCELL, NRPATCH, PEAFACTOR


###############################################################################
# ##############                    Helper                     ############## #
###############################################################################
font = {'family': 'serif',
        'color':  'grey',
        'weight': 'bold',
        'size': 16,
        'alpha': 0.5,
        }


###############################################################################
def linearerFit(x, m, b):
    return (m*x+b)


###############################################################################
# ##############               Drs-Value  Plots                ############## #
###############################################################################
@click.command()
@click.argument('drs_file_path',
                default="/net/big-tank/POOL/projects/fact/drs_temp_calib_data/" +
                        "calibration/drsSourceCollection/drsData.h5",
                type=click.Path(exists=True))
@click.argument('interval_file_path',
                default="/net/big-tank/POOL/projects/fact/drs_temp_calib_data/" +
                        "calibration/drsSourceCollection/intervalIndices.h5",
                type=click.Path(exists=True))
@click.argument('store_file_path',
                default="/home/fschulz/plots/camera_drs_values_range_-10_10.mp4",
                type=click.Path(exists=False))
def plot_camera_video(drs_file_path, interval_file_path, store_file_path):
    source_table = h5py.File(drs_file_path, 'r')
    interval_table = h5py.File(interval_file_path, 'r')

    nr_of_intervals = 3

    nr_of_dates = source_table['BaselineMean'].shape[0]

    gs = gridspec.GridSpec(10, 11)
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    fig.suptitle('-')

    ax0b = plt.subplot(gs[0:9, 0:5])
    ax0b.set_axis_off()
    ax0b.set_xlabel('')
    ax0b.set_ylabel('')
    ax0b.set_title('Baseline', fontsize=18)

    camera_plot_b = camera(np.zeros(1440), vmin=-10, vmax=10)

    ax0g = plt.subplot(gs[0:9, 5:11])
    ax0g.set_axis_off()
    ax0g.set_xlabel('')
    ax0g.set_ylabel('')
    ax0g.set_title('Gain', fontsize=18)

    camera_plot_g = camera(np.zeros(1440), vmin=-10, vmax=10)

    cbar = fig.colorbar(camera_plot_g, ax=ax0g)
    cbar.set_label(r"Delta drsValue / $\mathrm{mV}$")

    ax1 = plt.subplot(gs[9, 1:9])
    ax1.set_ylim(-0.45, 0.45)
    ax1.set_xlim(0, source_table['BaselineMean'].shape[0])
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.set_xlabel('')
    ax1.set_ylabel('')

    progressbar = ax1.barh(0, 0, height=1.0, align='center')
    fig.tight_layout()

    def data_gen(source_table, interval_table):
        for interval_nr in range(1, nr_of_intervals+1):
            groupname = "Interval"+str(interval_nr)
            print(groupname)
            interval_indices = interval_table[groupname]["IntervalIndices"]
            low_limit = interval_table[groupname].attrs["LowLimit"][0:-3]
            upp_limit = interval_table[groupname].attrs["UppLimit"][0:-3]
            time = source_table['TimeBaseline'][interval_indices, :]

            drs_value_b = source_table['BaselineMean'][interval_indices, :]
            drs_value_g = source_table['GainMean'][interval_indices, :]
            drs_value_chid_cells_mean_b = np.empty(NRCHID*NRCELL)
            drs_value_chid_cells_mean_g = np.empty(NRCHID*NRCELL)

            for chid in tqdm(range(NRCHID)):
                drs_value_chid_b = drs_value_b[:, chid*NRCELL:(chid+1)*NRCELL]
                drs_value_chid_cells_mean_b[chid*NRCELL:(chid+1)*NRCELL] = np.mean(drs_value_chid_b, axis=0)
                drs_value_chid_g = drs_value_g[:, chid*NRCELL:(chid+1)*NRCELL]
                drs_value_chid_cells_mean_g[chid*NRCELL:(chid+1)*NRCELL] = np.mean(drs_value_chid_g, axis=0)

            i_name = "Interval "+str(interval_nr)+"\n["+low_limit+", "+upp_limit+"]"
            for date_index in tqdm(range(time.shape[0])):
                delta_drs_value_b = np.array(drs_value_b[date_index, :] - drs_value_chid_cells_mean_b)
                delta_drs_value_g = np.array(drs_value_g[date_index, :] - drs_value_chid_cells_mean_g)
                yield (time[date_index, ],
                       delta_drs_value_b.reshape(1440, 1024).mean(axis=1),
                       delta_drs_value_g.reshape(1440, 1024).mean(axis=1),
                       date_index,
                       i_name)

    def update(data):
        t, b, g, i, i_name = data
        t = pd.to_datetime(t * 24 * 3600 * 1e9)

        offset_str = "            "
        fig.suptitle("Date"+offset_str+"\n"+t.strftime('%Y-%m-%d')+offset_str, y=0.92, fontsize=20)

        camera_plot_b.set_array(b)
        camera_plot_g.set_array(g)

        progressbar[0].set_width(i)
        ax1.set_title(i_name, fontsize=18)
        return camera_plot_b, camera_plot_g, progressbar, ax1.get_title()

    ani = FuncAnimation(
        fig,
        update,
        frames=data_gen(source_table, interval_table),
        interval=1,
        save_count=nr_of_dates,
    )
    writer = FFMpegWriter(fps=10, bitrate=18000)
    ani.save(store_file_path, writer=writer)


@click.command()
@click.argument('drs_file_path',
                default="/net/big-tank/POOL/projects/fact/drs_temp_calib_data/" +
                        "calibration/drsSourceCollection/drsData.h5",
                type=click.Path(exists=True))
@click.argument('interval_file_path',
                default="/net/big-tank/POOL/projects/fact/drs_temp_calib_data/" +
                        "calibration/drsSourceCollection/intervalIndices.h5",
                type=click.Path(exists=True))
@click.argument('store_file_path',
                default="/home/fschulz/plots/baseline_std_hist.jpg",
                type=click.Path(exists=False))
@click.argument("interval_array",
                default=[1, 2, 3])
@click.argument('drs_value_type',
                default="Gain")
@click.argument('nr_bins',
                default=150)
def drs_value_std_hist(drs_file_path, interval_file_path,
                       store_file_path, interval_array,
                       drs_value_type, nr_bins):
    # Cecking wether the intervalIndices are based on the given drsData
    check_file_match(drs_file_path, interval_file_path=interval_file_path)

    upper_limit = 20
    for interval_nr in interval_array:
        groupname = "Interval"+str(interval_nr)
        print(groupname)
        with h5py.File(interval_file_path, 'r') as interval_source:
            data = interval_source[groupname]
            interval_indices = np.array(data["IntervalIndices"])
        with h5py.File(drs_file_path, 'r') as store:
            drs_value_std = np.array(store[drs_value_type+"MeanStd"][interval_indices, :]).flatten()

        drs_value_std_mean = np.mean(drs_value_std)

        weights = np.full(len(drs_value_std), 100/len(drs_value_std))
        hist = plt.hist(drs_value_std, weights=weights, bins=nr_bins,
                        range=(0, upper_limit),
                        label=groupname+" mean: "+str(drs_value_std_mean))

    plt.title(r"Histogram of "+drs_value_type+"Std", fontsize=16, y=1.01)
    plt.xlabel(r'Std /$\mathrm{mV}$')
    plt.ylabel(r'Frequency /$\mathrm{\%}$')
    plt.xlim(0)
    plt.legend(loc="upper right")
    if(store_file_path is not None):
        plt.savefig(store_file_path)
    plt.show()
    plt.close()


@click.command()
@click.argument('chi2_file_path',
                default="/net/big-tank/POOL/projects/fact/drs_temp_calib_data/" +
                        "calibration/residuals/drsChiSquare.h5",
                type=click.Path(exists=True))
@click.argument('store_file_path',
                default="/home/fschulz/plots/chi2_fact_cam_gain.jpg",
                type=click.Path(exists=False))
@click.argument("interval_array",
                default=[1])
@click.argument('drs_value_type',
                default="Gain")
###############################################################################
def chi2_fact_cam(chi2_file_path, store_file_path,
                  interval_array, drs_value_type):

    for interval_nr in interval_array:
        groupname = "Interval"+str(interval_nr)
        with h5py.File(chi2_file_path, 'r') as chi2_tab:
            data = chi2_tab[groupname]
            chi2 = np.mean(np.array(data[drs_value_type+"Chi2"]).reshape(1440, 1024), axis=1)
            #p = np.mean(np.array(data[drs_value_type+"P"]).reshape(1440, 1024), axis=1)

    plot = camera(abs(chi2), cmap='hot')
    plt.colorbar(plot, label=r"$| \mathrm{Chi}^2 |$")
    if(store_file_path is not None):
        plt.savefig(store_file_path)
    plt.show()
    plt.close()


@click.command()
@click.argument('drs_file_path',
                default="/net/big-tank/POOL/projects/fact/drs_temp_calib_data/" +
                        "calibration/drsSourceCollection/drsData.h5",
                type=click.Path(exists=True))
@click.argument('interval_file_path',
                default="/net/big-tank/POOL/projects/fact/drs_temp_calib_data/" +
                        "calibration/drsSourceCollection/intervalIndices.h5",
                type=click.Path(exists=True))
@click.argument('fit_file_path',
                default="/net/big-tank/POOL/projects/fact/drs_temp_calib_data/" +
                        "calibration/fitParameter/drsFitParameter.fits",
                type=click.Path(exists=True))
@click.argument('store_file_path',
                default="/home/fschulz/plots/chid565_cell279_gain.jpg",
                type=click.Path(exists=False))
@click.argument("interval_array",
                default=[2])
@click.argument('drs_value_type',
                default="Gain")
# @click.option('--drs_value_type', '-vt',
#               default="Baseline",
#               type=click.Choice(['Baseline', 'Gain']))
@click.argument('chid',
                default=565)
@click.argument('cell',
                default=279)
# @click.option('--show_std_dev', '-std',
#               is_flag=False)
###############################################################################
def drs_value_cell(drs_file_path, interval_file_path, fit_file_path,
                   store_file_path, interval_array, drs_value_type,
                   chid, cell):

    value_index = chid*NRCELL + cell
    border = 2.0  # mV
    # Cecking wether the intervalIndices and the fitvalues are based on the given drsData
    check_file_match(drs_file_path,
                     interval_file_path=interval_file_path,
                     fit_file_path=fit_file_path)

    # loading source data
    with h5py.File(drs_file_path, 'r') as store:
        time = np.array(store["Time"+drs_value_type]).flatten()

    use_mask = True
    in_PEA = False
    mask_collection = []
    time_collection = []
    temp_collection = []
    drs_value_collection = []
    fit_value_collection = []
    for interval_nr in interval_array:
        groupname = "Interval"+str(interval_nr)
        with h5py.File(interval_file_path, 'r') as interval_source:
            data = interval_source[groupname]
            cut_off_error_factor = interval_source.attrs["CutOff"+drs_value_type]
            interval_indices = np.array(data["IntervalIndices"])
            if(use_mask):
                mask = np.array(data[drs_value_type+"Mask"])
                mask_collection.append(mask)
        with h5py.File(drs_file_path, 'r') as store:
            temp = np.array(store["Temp"+drs_value_type][interval_indices, int(chid/9)])
            drs_value = np.array(store[drs_value_type+"Mean"][interval_indices, value_index])
        with fits.open(fit_file_path, ignoremissing=True, ignore_missing_end=True) as fit_value_tab:
            data = fit_value_tab[groupname].data
            slope = data[drs_value_type+"Slope"][0][value_index]
            offset = data[drs_value_type+"Offset"][0][value_index]

        time_interval = time[interval_indices]
        time_collection.append(time_interval)
        temp_collection.append(temp)
        drs_value_collection.append(drs_value)
        fit_value_collection.append([slope, offset])

        ylabel_str = drs_value_type+r'Mean /$\mathrm{mV}$'
        if(in_PEA):
            border = border*PEAFACTOR
            ylabel_str = r'paternNoiseMean /$\mathrm{PEA}$'
            for drs_value_index in range(len(drs_value_collection)):
                drs_value = drs_value_collection[drs_value_index]
                drs_valueMean = np.mean(drs_value, dtype='float64')
                drs_value_collection[drs_value_index] = (drs_value-drs_valueMean)*PEAFACTOR
                fit_value_collection[drs_value_index][0] *= PEAFACTOR
                offset = fit_value_collection[drs_value_index][1]
                fit_value_collection[drs_value_index][1] = (offset-drs_valueMean)*PEAFACTOR

    temp_list = np.concatenate(temp_collection).ravel()
    drs_value_list = np.concatenate(drs_value_collection).ravel()
    time_list = np.concatenate(time_collection).ravel()

    sc_all = plt.scatter(temp_list, drs_value_list, c=time_list)
    plt.close()  # Dont show sc_all, just use it to create the colorbar

    fig, img = plt.subplots()

    intervalMonth = 3
    start_date = pd.to_datetime(time_list[0] * 24 * 3600 * 1e9).date()
    end_date = pd.to_datetime(time_list[-1] * 24 * 3600 * 1e9).date()
    timeLabel = pd.date_range(start=start_date, end=end_date, freq=str(intervalMonth)+"M")
    cbar = fig.colorbar(sc_all, ticks=dates.MonthLocator(interval=intervalMonth))
    cbar.ax.set_yticklabels(timeLabel.strftime("%b %Y"))
    timeColor = cbar.to_rgba(time_list)

    i_min, i_max = 0, 0
    temp_range = np.linspace(min(temp_list)-1, max(temp_list)+1, 10000)
    for interval_index in range(len(interval_array)):
        i_max = i_max+len(temp_collection[interval_index])
        color = timeColor[i_min: i_max]
        temp = temp_collection[interval_index]
        drs_value = drs_value_collection[interval_index]
        if(use_mask):
            mask_u = mask_collection[interval_index][:, value_index]
            mask_nu = np.logical_not(mask_u)
            sc = img.scatter(temp[mask_u], drs_value[mask_u], s=50, marker="+",
                             c=color, label="paternNoiseMean with averaged Temperature")
            sc = img.scatter(temp[mask_nu], drs_value[mask_nu], s=50, marker="*",
                             c=color, label="paternNoiseMean with averaged Temperature")
        else:
            sc = img.scatter(temp, drs_value, s=50, marker="+",
                             c=color, label="paternNoiseMean with averaged Temperature")
        i_min = i_min+i_max

        slope, offset = fit_value_collection[interval_index]
        fit = linearerFit(temp_range, slope, offset)

        color_mean = np.mean(color, axis=0)
        fitPlot, = plt.plot(temp_range, fit, "-", color=color_mean)

        fitPlot, = plt.plot(temp_range, fit-border, "--", color=color_mean)
        fitPlot, = plt.plot(temp_range, fit+border, "--", color=color_mean)

    plt.title(drs_value_type+"Mean\nChid: "+str(chid)+", Cell: "+str(cell) +  # , fontsize=20, y=0.95
              ", ErrFactor: "+str('{:0.1f}'.format(cut_off_error_factor)), fontsize=15, y=1.02)

    plt.xlabel(r'Temperature /$\mathrm{^\circ C}$')
    plt.ylabel(ylabel_str)
    plt.xlim(min(temp_list)-1, max(temp_list)+1)
    plt.grid()
    plt.gca().ticklabel_format(useOffset=False)
    plt.text(0.02, 0.21, "preliminary", fontdict=font, transform=plt.gca().transAxes)
    if(store_file_path is not None):
        plt.savefig(store_file_path)
    plt.show()
    plt.close()


@click.command()
@click.argument('drs_file_path',
                default="/net/big-tank/POOL/projects/fact/drs_temp_calib_data/" +
                        "calibration/drsSourceCollection/drsData.h5",
                type=click.Path(exists=True))
@click.argument('interval_file_path',
                default="/net/big-tank/POOL/projects/fact/drs_temp_calib_data/" +
                        "calibration/drsSourceCollection/intervalIndices.h5",
                type=click.Path(exists=True))
@click.argument('store_file_path',
                default="/home/fschulz/plots/chid700_cell0_gain.jpg",
                type=click.Path(exists=False))
@click.argument("interval_array",
                default=[2])
@click.argument('drs_value_type',
                default="Gain")
# @click.option('--drs_value_type', '-vt',
#               default="Baseline",
#               type=click.Choice(['Baseline', 'Gain']))
@click.argument('chid',
                default=700)
@click.argument('cell',
                default=0)
# @click.option('--show_std_dev', '-std',
#               is_flag=False)
###############################################################################
def chid_cell_drs_values_time(drs_file_path, interval_file_path,
                              store_file_path, interval_array,
                              drs_value_type, chid, cell):

    value_index = chid*NRCELL + cell

    # Cecking wether the intervalIndices and the fitvalues are based on the given drsData
    with h5py.File(drs_file_path, 'r') as data_source:
        source_creation_date = data_source.attrs['CreationDate']

    with h5py.File(interval_file_path) as interval_source:
        used_source_creation_date_i = interval_source.attrs["SCDate"]

    if(source_creation_date != used_source_creation_date_i):
        error_str = ("'interval_file_path' is not based on the given 'source_file_path'")
        print(error_str)
        return

    with h5py.File(drs_file_path, 'r') as store:
        time = np.array(store["Time"+drs_value_type]).flatten()

    use_mask = True
    in_PEA = False
    mask_collection = []
    time_collection = []
    temp_collection = []
    drs_value_collection = []
    drs_value_std_collection = []
    for interval_nr in interval_array:
        groupname = "Interval"+str(interval_nr)
        with h5py.File(interval_file_path, 'r') as interval_source:
            data = interval_source[groupname]
            cut_off_error_factor = interval_source.attrs["CutOff"+drs_value_type]
            interval_indices = np.array(data["IntervalIndices"])
            if(use_mask):
                mask = np.array(data[drs_value_type+"Mask"][:, value_index])
                mask_collection.append(mask)
        with h5py.File(drs_file_path, 'r') as store:
            temp = np.array(store["Temp"+drs_value_type][interval_indices, int(chid/9)])
            drs_value = np.array(store[drs_value_type+"Mean"][interval_indices, value_index])
            drs_value_std = np.array(store[drs_value_type+"MeanStd"][interval_indices, value_index])

        time_interval = pd.to_datetime(time[interval_indices] * 24 * 3600 * 1e9)
        time_collection.append(time_interval)
        temp_collection.append(temp)
        drs_value_collection.append(drs_value)
        drs_value_std_collection.append(drs_value_std)

        single_photon_limit = 2.1  # mV
        ylabel_str = drs_value_type+r'Mean /$\mathrm{mV}$'
        if(in_PEA):
            single_photon_limit *= PEAFACTOR
            ylabel_str = r'paternNoiseMean /$\mathrm{PEA}$'
            for interval_index in range(len(drs_value_collection)):
                drs_value = drs_value_collection[interval_index]
                drs_valueMean = np.mean(drs_value, dtype='float64')
                drs_value_collection[interval_index] = (drs_value-drs_valueMean)*PEAFACTOR

    for interval_index in range(len(interval_array)):
        mask = mask_collection[interval_index]
        drs_value = drs_value_collection[interval_index]
        drs_value_std = drs_value_std_collection[interval_index]
        time = time_collection[interval_index]
        temp = temp_collection[interval_index]

        mask0 = np.where(drs_value_std == 0.)[0]
        mask2 = np.where(drs_value_std > np.mean(drs_value_std)*2.0)[0]
        # mask2 = np.where(drs_value_std < np.mean(drs_value_std)*0.4)[0]
        #ylabel_str = r'Temperature /$\mathrm{^\circ C}$'
        plt.errorbar(time, drs_value, yerr=None, color="k", marker='*', ls='')
        plt.errorbar(time[mask], drs_value[mask], yerr=None, color="g", marker='*', ls='')
        #plt.errorbar(time[mask], drs_value[mask], yerr=None, color="r", marker='*', ls='')
        #plt.errorbar(time[mask2], drs_value[mask2], yerr=None, color="b", marker='*', ls='')
    plt.title(drs_value_type+"Mean\nChid: "+str(chid)+", Cell: "+str(cell) +  # , fontsize=20, y=0.95
              ", ErrFactor: "+str('{:0.1f}'.format(cut_off_error_factor)), fontsize=15, y=1.02)

    plt.xlabel(r'Time')
    plt.ylabel(ylabel_str)
    # plt.xlim(min(temp_list)-1, max(temp_list)+1)
    plt.grid()
    #plt.gca().ticklabel_format(useOffset=False)
    #timeLabel = pd.date_range(start=start_date_, end=end_date_, freq="M") - pd.offsets.MonthBegin(1)
    #plt.xticks(timeLabel, timeLabel, rotation=30)
    #plt.gca().xaxis.set_major_formatter(time.DateFormatter("%d.%m.%y"))
    plt.text(0.02, 0.21, "preliminary", fontdict=font, transform=plt.gca().transAxes)
    if(store_file_path is not None):
        plt.savefig(store_file_path)
    plt.show()
    plt.close()


@click.command()
@click.argument('fit_file_path',
                default="/net/big-tank/POOL/projects/fact/drs_temp_calib_data/" +
                        "calibration/fitParameter/drsFitParameter.fits",
                type=click.Path(exists=True))
@click.argument('store_file_path',
                default="/home/fschulz/plots/baselineSlopeChid_500.jpg",
                type=click.Path(exists=False))
@click.argument("interval_array",
                default=[2])
@click.argument('drs_value_type',
                default="Baseline")
@click.argument('fit_parameter_type',
                default="Slope")
@click.argument('chid',
                default=500)
###############################################################################
def drs_value_chid_fit_parameter(fit_file_path, chid, store_file_path, interval_array, drs_value_type, fit_parameter_type,):

    for interval_nr in interval_array:
        groupname = "Interval"+str(interval_nr)
        with fits.open(fit_file_path, ignoremissing=True, ignore_missing_end=True) as fit_value_tab:
            data = fit_value_tab[groupname].data
            fit_parameter = data[drs_value_type+fit_parameter_type][0][chid*NRCELL:(chid+1)*NRCELL]

    colors = hot(np.linspace(0, 0, NRCELL))
    for i in range(33):
        colors[i*32-1] = [1., 0., 0., 1.]

    cell = np.linspace(0, NRCELL-1, NRCELL)
    plt.scatter(cell, fit_parameter, s=50, marker="+", color=colors)

    labelStr = ""
    if(fit_parameter_type == "Offset"):
        labelStr = fit_parameter_type+r" /$\mathrm{mV}$"
    elif(fit_parameter_type == "Slope"):
        labelStr = fit_parameter_type+r" /$\frac{\mathrm{mV}}{\mathrm{^\circ C}}$"

    plt.title((drs_value_type+" "+fit_parameter_type+" CHID:"+str(chid)+"\n" +
               r" Fit $f(x) = m \cdot x + b$"), fontsize=16, y=1.00)
    plt.xlabel('Cell [1]')
    plt.ylabel(labelStr)
    plt.xlim(-1, NRCELL)
    plt.grid()
    # plt.legend(loc='upper right', scatterpoints=1, numpoints=1)
    plt.savefig(store_file_path)
    plt.show()
    plt.close()


@click.command()
@click.argument("residuals_file_path",
                default="/net/big-tank/POOL/projects/fact/drs_temp_calib_data/" +
                        "calibration/residuals/drsResiduals.h5",
                type=click.Path(exists=True))
@click.argument('interval_file_path',
                default="/net/big-tank/POOL/projects/fact/drs_temp_calib_data/" +
                        "calibration/drsSourceCollection/intervalIndices.h5",
                type=click.Path(exists=True))
@click.argument("interval_array",
                default=[1])
@click.argument("drs_value_type",
                default="Baseline")
@click.argument("store_file_path",
                default="/home/fschulz/plots/residualsBaseline_his_I1.jpg",
                type=click.Path(exists=False))
####################################################################################################
def residuals_hist(residuals_file_path, interval_file_path, interval_array, drs_value_type, store_file_path):

    with h5py.File(interval_file_path) as interval_source:
        used_source_creation_date_i = interval_source.attrs["SCDate"]

    with h5py.File(residuals_file_path) as residual_source:
        used_source_creation_date_r = residual_source.attrs["SCDate"]

    if(used_source_creation_date_i != used_source_creation_date_r):
        error_str = ("'interval_file_path' and 'residuals_file_path' dont belong together")
        print(error_str)

    for interval_nr in interval_array:
        groupname = "Interval"+str(interval_nr)
        with h5py.File(interval_file_path, 'r') as interval_source:
            data = interval_source[groupname]
            mask = np.array(data[drs_value_type+"Mask"])

        with h5py.File(residuals_file_path, 'r') as residuals_tab:
            data = residuals_tab[groupname]
            residuals = np.array(data[drs_value_type+"Residuals"])
            residuals = residuals[mask].flatten()

    outlier = np.where(abs(residuals) > 2.1)[0]
    print("outlier: ", len(outlier)/len(residuals)*100, " %")

    nr_bins = 40
    title_str = "Hist"
    plt.title(title_str, y=1.0)
    weights = np.full(len(residuals), 100/len(residuals))
    hist1 = plt.hist(residuals, weights=weights, bins=nr_bins, histtype='step',
                     range=(-1, 1), lw=1, edgecolor="r", label="test")
    if(store_file_path is not None):
        plt.savefig(store_file_path)
    plt.show()
    plt.close()


@click.command()
@click.argument("residuals_file_path",
                default="/net/big-tank/POOL/projects/fact/drs_temp_calib_data/" +
                        "calibration/residuals/drsResiduals.h5",
                type=click.Path(exists=True))
@click.argument('interval_file_path',
                default="/net/big-tank/POOL/projects/fact/drs_temp_calib_data/" +
                        "calibration/drsSourceCollection/intervalIndices.h5",
                type=click.Path(exists=True))
@click.argument("interval_array",
                default=[1])
@click.argument("drs_value_type",
                default="Baseline")
@click.argument("store_file_path",
                default="/home/fschulz/plots/residualsBaseline_hisout_I1.jpg",
                type=click.Path(exists=False))
###############################################################################
def residuals_hist_outlier(residuals_file_path, interval_file_path,
                           interval_array, drs_value_type,
                           store_file_path):

    with h5py.File(interval_file_path) as interval_source:
        used_source_creation_date_i = interval_source.attrs["SCDate"]

    with h5py.File(residuals_file_path) as residual_source:
        used_source_creation_date_r = residual_source.attrs["SCDate"]

    if(used_source_creation_date_i != used_source_creation_date_r):
        error_str = ("'interval_file_path' and 'residuals_file_path' dont belong together")
        print(error_str)

    single_photon_limit = 2.1  # mV
    for interval_nr in interval_array:
        groupname = "Interval"+str(interval_nr)
        with h5py.File(interval_file_path, 'r') as interval_source:
            data = interval_source[groupname]
            mask = np.array(data[drs_value_type+"Mask"])

        with h5py.File(residuals_file_path, 'r') as residuals_tab:
            data = residuals_tab[groupname]
            residuals = np.array(data[drs_value_type+"Residuals"])

    chid_residuen = np.zeros((NRCHID, 1))
    for chid in range(NRCHID):
        sub_mask = mask[:, chid*NRCELL:(chid+1)*NRCELL]
        residuals_chid = residuals[:, chid*NRCELL:(chid+1)*NRCELL][sub_mask].flatten()
        value = len(residuals_chid[residuals_chid > single_photon_limit])/len(residuals_chid)*100
        chid_residuen[chid] = value

    plt.title(("Frequency of "+drs_value_type+"residuals \n" +
               "over the limit of "+str(single_photon_limit)+r" $\mathrm{mV}$"))
    plt.step(range(1, NRCHID+1), chid_residuen, where="mid")
    max_ = np.amax(chid_residuen)
    # for chid in range(NRCHID):
    #     if(chid % 9 == 8):
    #         plt.plot([chid+1, chid+1], [0, max_], "r-")
    plt.xlabel("CHID")
    plt.ylabel(r'Frequency /$\mathrm{\%}$')
    plt.xlim(1, NRCHID+1)
    plt.ylim(0, )
    if(store_file_path is not None):
        plt.savefig(store_file_path)
    plt.show()
    plt.close()


@click.command()
@click.argument('chi2_file_path',
                default="/net/big-tank/POOL/projects/fact/drs_temp_calib_data/" +
                        "calibration/residuals/drsChiSquare.h5",
                type=click.Path(exists=True))
@click.argument('store_file_path',
                default="/home/fschulz/plots/chi2_cell_mean_per_chid_gain.pdf",
                type=click.Path(exists=False))
@click.argument("interval_array",
                default=[1, 2, 3])
@click.argument('drs_value_type',
                default="Gain")
###############################################################################
def drs_values_chi2_cell_mean_per_chid(chi2_file_path, store_file_path,
                                       interval_array, drs_value_type):

    y1_prop, y2_prop = 3, 1
    gs = gridspec.GridSpec(2, 1, height_ratios=[y2_prop, y1_prop])
    plt.figure(figsize=(10, 8))


    y_split = 1
    with PdfPages(store_file_path) as pdf:
        for interval_nr in interval_array:
            groupname = "Interval"+str(interval_nr)
            print(groupname)
            with h5py.File(chi2_file_path, 'r') as chi2_tab:
                data = chi2_tab[groupname]
                chid_chi2 = np.mean(abs(np.array(data[drs_value_type+"Chi2"])).reshape(NRCHID, NRCELL), axis=1)
            plt.close()
            plt.figure(figsize=(10, 8))
            plt.ylabel(r'mean $\left(|CHI2|\right)$ /$\mathrm{1}$')

            ax0 = plt.subplot(gs[1, 0])
            ax1 = plt.subplot(gs[0, 0], sharex=ax0)
            plt.subplots_adjust(hspace=0.1)
            plt.setp(ax1.get_xticklabels(), visible=False)
            ax0.spines['top'].set_visible(False)
            ax1.spines['bottom'].set_visible(False)
            ax1.xaxis.tick_top()
            ax1.tick_params(labeltop='off')  # don't put tick labels at the top

            plt.title(drs_value_type+"\n"+groupname)
            ax0.step(range(1, NRCHID+1), chid_chi2, where="mid")
            ax1.step(range(1, NRCHID+1), chid_chi2, where="mid")

            x_0, x_1 = -10, 1449
            d = .015
            scale1 = (y1_prop+y2_prop)/y1_prop
            scale2 = (y1_prop+y2_prop)/y2_prop
            kwargs = dict(transform=ax0.transAxes, color='k', clip_on=False)
            ax0.plot((-d, d), (1-d*scale1, 1+d*scale1), **kwargs)
            ax0.plot((1-d, 1+d), (1-d*scale1, 1+d*scale1), **kwargs)
            kwargs.update(transform=ax1.transAxes)
            ax1.plot((-d, d), (-d*scale2, d*scale2), **kwargs)
            ax1.plot((1-d, 1+d), (-d*scale2, d*scale2), **kwargs)

            ax0.set_xlim(x_0, x_1)
            ax0.set_ylim(0, y_split)
            ax1.set_ylim(y_split,)
            ax0.set_xlabel("CHID")
            pdf.savefig()
            plt.close()


@click.command()
@click.argument("residuals_file_path",
                default="/net/big-tank/POOL/projects/fact/drs_temp_calib_data/" +
                        "calibration/residuals/drsResiduals.h5",
                type=click.Path(exists=True))
@click.argument('store_file_path',
                default="/home/fschulz/plots/residuals_cell_mean_per_chid_gain.pdf",
                type=click.Path(exists=False))
@click.argument("interval_array",
                default=[1, 2, 3])
@click.argument('drs_value_type',
                default="Gain")
###############################################################################
def drs_values_residuals_cell_mean_per_chid(residuals_file_path, store_file_path,
                                            interval_array, drs_value_type):

    with PdfPages(store_file_path) as pdf:
        for interval_nr in interval_array:
            groupname = "Interval"+str(interval_nr)
            print(groupname)
            with h5py.File(residuals_file_path, 'r') as residuals_tab:
                data = residuals_tab[groupname]
                residuals = np.mean(np.mean(abs(np.array(data[drs_value_type+"Residuals"])), axis=0).reshape(NRCHID, NRCELL), axis=1)
            plt.title(drs_value_type+"\n"+groupname)
            plt.step(range(1, NRCHID+1), residuals, where="mid")
            plt.xlabel("CHID")
            plt.ylabel(r'mean $\left($|residuals|$\right)$ /$\mathrm{mV}$')
            pdf.savefig()
            plt.close()


@click.command()
@click.argument('drs_file_path',
                default="/net/big-tank/POOL/projects/fact/drs_temp_calib_data/" +
                        "calibration/drsSourceCollection/drsData.h5",
                type=click.Path(exists=True))
@click.argument('interval_file_path',
                default="/net/big-tank/POOL/projects/fact/drs_temp_calib_data/" +
                        "calibration/drsSourceCollection/intervalIndices.h5",
                type=click.Path(exists=True))
@click.argument('fit_file_path',
                default="/net/big-tank/POOL/projects/fact/drs_temp_calib_data/" +
                        "calibration/fitParameter/drsFitParameter.fits",
                type=click.Path(exists=True))
@click.argument('chi2_file_path',
                default="/net/big-tank/POOL/projects/fact/drs_temp_calib_data/" +
                        "calibration/residuals/drsChiSquare.h5",
                type=click.Path(exists=True))
@click.argument('store_file_path',
                default="/home/fschulz/plots/outlier_cell_collection_I2_baseline_chi2_limit_2.5.pdf",
                type=click.Path(exists=False))
@click.argument("interval_array",
                default=[2])
@click.argument('drs_value_type',
                default="Baseline")
###############################################################################
def drs_values_outlier_cell_collection(drs_file_path, interval_file_path,
                                       chi2_file_path, fit_file_path,
                                       store_file_path,
                                       interval_array, drs_value_type):

    # Cecking wether the intervalIndices and the fitvalues are based on the given drsData
    check_file_match(drs_file_path,
                     chi2_file_path=chi2_file_path,
                     interval_file_path=interval_file_path,
                     fit_file_path=fit_file_path)

    chi2_limit = 2.5

    groupname = "Interval"+str(interval_array[0])
    with h5py.File(chi2_file_path, 'r') as chi2_tab:
        data = chi2_tab[groupname]
        chi2 = abs(np.array(data[drs_value_type+"Chi2"]).flatten())

    interval_source = h5py.File(interval_file_path, 'r')
    cut_off_error_factor = interval_source.attrs["CutOff"+drs_value_type]
    interval_source = interval_source[groupname]
    interval_indices = np.array(interval_source["IntervalIndices"])

    fit_value_tab = fits.open(fit_file_path, ignoremissing=True, ignore_missing_end=True)[groupname].data

    count = 0
    with PdfPages(store_file_path) as pdf:
        for chid in tqdm(range(NRCHID)):
            if (chid >= 720 and chid <= 755):
                continue
            for cell in range(NRCELL):
                value_index = chid*NRCELL + cell
                if chi2[value_index] > chi2_limit:
                    print(chid, cell, chi2[value_index])
                    count +=1
                    mask = np.array(interval_source[drs_value_type+"Mask"][:, value_index])
                    with h5py.File(drs_file_path, 'r') as store:
                        time = np.array(store["Time"+drs_value_type][interval_indices, :]).flatten()
                        temp = store["Temp"+drs_value_type][interval_indices, int(chid/9)]
                        drs_value = store[drs_value_type+"Mean"][interval_indices, value_index]

                    sc_all = plt.scatter(temp, drs_value, c=np.array(time))
                    plt.close()  # Dont show sc_all, just use it to create the colorbar

                    fig, img = plt.subplots()

                    intervalMonth = 3
                    start_date = pd.to_datetime(time[0] * 24 * 3600 * 1e9).date()
                    end_date = pd.to_datetime(time[-1] * 24 * 3600 * 1e9).date()
                    timeLabel = pd.date_range(start=start_date, end=end_date, freq=str(intervalMonth)+"M")
                    cbar = fig.colorbar(sc_all, ticks=dates.MonthLocator(interval=intervalMonth))
                    cbar.ax.set_yticklabels(timeLabel.strftime("%b %Y"))
                    timeColor = cbar.to_rgba(time)

                    i_min, i_max = 0, len(temp)
                    temp_range = np.linspace(min(temp)-1, max(temp)+1, 10000)
                    color = timeColor[i_min: i_max]
                    mask_u = mask
                    mask_nu = np.logical_not(mask_u)
                    sc = img.scatter(temp[mask_u], drs_value[mask_u], s=50, marker="+",
                                     c=color, label="paternNoiseMean with averaged Temperature")
                    sc = img.scatter(temp[mask_nu], drs_value[mask_nu], s=50, marker="*",
                                     c=color, label="paternNoiseMean with averaged Temperature")

                    slope = fit_value_tab[drs_value_type+"Slope"][0][value_index]
                    offset = fit_value_tab[drs_value_type+"Offset"][0][value_index]
                    fit = linearerFit(temp_range, slope, offset)

                    color_mean = np.mean(color, axis=0)
                    fitPlot, = plt.plot(temp_range, fit, "-", color=color_mean)

                    #fitPlot, = plt.plot(temp_range, fit-single_photon_limit, "--", color=color_mean)
                    #fitPlot, = plt.plot(temp_range, fit+single_photon_limit, "--", color=color_mean)

                    plt.title((drs_value_type+"Mean, Chi2: "+str('{:0.1f}'.format(chi2[value_index])) +
                              "\nChid: "+str(chid)+", Cell: "+str(cell)), fontsize=15, y=1.00)  # , fontsize=20, y=0.95

                    plt.xlabel(r'Temperature /$\mathrm{^\circ C}$')
                    plt.ylabel(drs_value_type+r'Mean /$\mathrm{mV}$')
                    plt.xlim(min(temp)-1, max(temp)+1)
                    plt.grid()
                    plt.gca().ticklabel_format(useOffset=False)
                    pdf.savefig()
    print(count)


@click.command()
@click.argument('drs_file_path',
                default="/net/big-tank/POOL/projects/fact/drs_temp_calib_data/" +
                        "calibration/drsSourceCollection/drsData.h5",
                type=click.Path(exists=True))
@click.argument('interval_file_path',
                default="/net/big-tank/POOL/projects/fact/drs_temp_calib_data/" +
                        "calibration/drsSourceCollection/intervalIndices.h5",
                type=click.Path(exists=True))
@click.argument("residuals_file_path",
                default="/net/big-tank/POOL/projects/fact/drs_temp_calib_data/" +
                        "calibration/residuals/drsResiduals.h5",
                type=click.Path(exists=True))
@click.argument("interval_array",
                default=[1, 2, 3])
@click.argument("drs_value_type",
                default="Baseline")
@click.argument("chid",
                default=161)
@click.argument("cell",
                default=494)
@click.argument("restrict_residuals",
                default=True,
                type=click.BOOL)
@click.argument("store_file_path",
                default="/home/fschulz/plots/residualsBaseline_chid721_cap500_3Intervals_RestrictParts.jpg",
                type=click.Path(exists=False))
###############################################################################
def residuals_per_chid_cell(drs_file_path, interval_file_path,
                            residuals_file_path, store_file_path,
                            interval_array, drs_value_type,
                            chid, cell,
                            restrict_residuals):

    value_index = chid*NRCELL + cell

    # Cecking wether the intervalIndices and the fitvalues are based on the given drsData
    with h5py.File(drs_file_path, 'r') as data_source:
        source_creation_date = data_source.attrs['CreationDate']

    with h5py.File(interval_file_path) as interval_source:
        used_source_creation_date_i = interval_source.attrs["SCDate"]

    with h5py.File(residuals_file_path) as residual_source:
        used_source_creation_date_r = residual_source.attrs["SCDate"]

    if((source_creation_date != used_source_creation_date_i) or
       (source_creation_date != used_source_creation_date_r)):
        error_str = ("'interval_file_path' or 'residuals_file_path' is not based on the given 'source_file_path'")
        print(error_str)
        return

    # loading source data
    with h5py.File(drs_file_path, 'r') as store:
        time = np.array(store["Time"+drs_value_type]).flatten()
    datetime = pd.to_datetime(time * 24 * 3600 * 1e9)

    use_mask = True
    in_PEA = False
    offset = 0.1  # TODO maybe ask/ magic number
    mask_collection = []
    datetime_collection = []
    residuals_collection = []
    boundarie_collection = []
    for interval_nr in interval_array:
        groupname = "Interval"+str(interval_nr)
        print(groupname)
        with h5py.File(interval_file_path, 'r') as interval_source:
            data = interval_source[groupname]
            low_limit = pd.to_datetime(data.attrs["LowLimit"])
            upp_limit = pd.to_datetime(data.attrs["UppLimit"])
            interval_indices = np.array(data["IntervalIndices"])
            print(interval_indices.shape)
            if(use_mask):
                mask = np.array(data[drs_value_type+"Mask"][:, value_index])
                print(mask.shape)
                mask_collection.append(mask)
        with h5py.File(residuals_file_path, 'r') as residuals_tab:
            data = residuals_tab[groupname]
            residuals = np.array(data[drs_value_type+"Residuals"][:, value_index])
            print(residuals.shape)
        datetime_interval = datetime[interval_indices]

        ylabel_str = "(f(t)-"+drs_value_type+"Mean) /$\mathrm{mV}$"
        if(in_PEA):
            ylabel_str = "(f(t)-"+drs_value_type+"Mean) /$\mathrm{PEA}$"
            for drs_value_index in range(len(residuals_collection)):
                residuals *= PEAFACTOR

        datetime_collection.append(datetime_interval)
        residuals_collection.append(residuals)
        boundarie_collection.append([low_limit, upp_limit])

    nr_of_intervals = len(interval_array)
    datetime_collection_ = np.concatenate(datetime_collection)
    residuals_collection_ = np.concatenate(residuals_collection)
    min_res, max_res = np.amin(residuals_collection_), np.amax(residuals_collection_)
    for interval_index in range(nr_of_intervals):
        datetime = datetime_collection[interval_index]
        residuals = residuals_collection[interval_index]
        low_limit, upp_limit = boundarie_collection[interval_index]
        c = [float(interval_index)/float(nr_of_intervals),
             0.0,
             float(nr_of_intervals-1-interval_index)/float(nr_of_intervals)]
        plt.plot([low_limit, low_limit], [min_res-offset, max_res+offset], "k-")
        plt.plot([upp_limit, upp_limit], [min_res-offset, max_res+offset], "k-")

        plt.annotate(s='', xy=(low_limit, 0), xytext=(upp_limit, 0),
                     arrowprops=dict(arrowstyle='<->', color=c))

        if(use_mask):
            mask_u = mask_collection[interval_index]
            mask_nu = np.logical_not(mask_u)
            plt.plot(datetime[mask_u], residuals[mask_u], "x", color=c)
            plt.plot(datetime[mask_nu], residuals[mask_nu], "*", color=c)
        else:
            plt.plot(datetime, residuals, "x", color=c)

    plt.title(drs_value_type+"residuals \n CHID: "+str(chid)+" Cell: "+str(cell))
    plt.ylabel(ylabel_str)
    plt.gcf().autofmt_xdate()
    plt.xlim(pd.to_datetime(np.amin(datetime_collection_)).date()-pd.DateOffset(days=7),
             pd.to_datetime(np.amax(datetime_collection_)).date()+pd.DateOffset(days=7))
    plt.ylim(min_res-offset, max_res+offset)
    plt.grid()
    plt.text(0.02, 0.2, "preliminary", fontdict=font, transform=plt.gca().transAxes)
    if(store_file_path is not None):
        plt.savefig(store_file_path)
    plt.show()
    plt.close()


@click.command()
@click.argument('drs_file_path',
                default="/net/big-tank/POOL/projects/fact/drs_temp_calib_data/" +
                        "calibration/drsSourceCollection/drsData.h5",
                type=click.Path(exists=True))
@click.argument('interval_file_path',
                default="/net/big-tank/POOL/projects/fact/drs_temp_calib_data/" +
                        "calibration/drsSourceCollection/intervalIndices.h5",
                type=click.Path(exists=True))
@click.argument("residuals_file_path",
                default="/net/big-tank/POOL/projects/fact/drs_temp_calib_data/" +
                        "calibration/residuals/drsResiduals_.h5",
                type=click.Path(exists=True))
@click.argument("interval_array",
                default=[1])
@click.argument("drs_value_type",
                default="Baseline")
@click.argument("chid",
                default=1400)
@click.argument("restrict_residuals",
                default=True,
                type=click.BOOL)
@click.argument("store_file_path",
                default="/home/fschulz/plots/residualsBaseline_chid1000_cap500_3Intervals_RestrictParts.jpg",
                type=click.Path(exists=False))
####################################################################################################
def residuals_mean_per_chid(drs_file_path, interval_file_path, residuals_file_path, interval_array, drs_value_type,
                             chid, restrict_residuals, store_file_path):

    with h5py.File(drs_file_path, 'r') as store:
        time = np.array(store["Time"+drs_value_type]).flatten()
        datetime = pd.to_datetime(time * 24 * 3600 * 1e9)

    date = datetime.date

    offset = 0.1  # TODO maybe ask/ magic number
    time_collection = []
    residuals_collection = []
    for interval_nr in interval_array:
        groupname = "Interval"+str(interval_nr)
        with h5py.File(residuals_file_path, 'r') as residuals_tab:
            data = residuals_tab[groupname]

            low_limit = pd.to_datetime(data.attrs["LowLimit"])
            upp_limit = pd.to_datetime(data.attrs["UppLimit"])

            residuals = np.array(data[drs_value_type+"Residuals"][chid*NRCELL: (chid+1)*NRCELL])

        interval_indices = np.where((datetime >= low_limit) & (datetime <= upp_limit))[0]
        datetime_interval = datetime[interval_indices]
        with h5py.File(drs_file_path, 'r') as store:
            drs_value_std = np.array(store[drs_value_type+"MeanStd"][interval_indices, chid*NRCELL: (chid+1)*NRCELL])

        if(False):  # TODO fix
            print(drs_value_std.shape)
            drs_value_std_mean = np.mean(drs_value_std, dtype='float64', axis=1)
            print(drs_value_std_mean.shape)
            drs_value_std_limit = drs_value_std_mean*2  # cut_off_factor[drs_value_type]
            indices_used_values = np.where(drs_value_std_mean < drs_value_std_limit)[0]
            indices_not_used_values = np.where(drs_value_std_mean >= drs_value_std_limit)[0]
            print(residuals.shape)
            print(indices_used_values)
            residuals = residuals[: , indices_used_values]
            datetime_interval = datetime_interval[indices_used_values]

        residuals_mean_per_chid = np.mean(residuals, dtype='float64', axis=0)
        residuals_collection.append(residuals_mean_per_chid)
        time_collection.append(datetime_interval)

        nr_of_intervals = len(interval_array)
        interval_nr =1  # TODO fix
        c = [float(interval_nr-1)/float(nr_of_intervals),
             0.0,
             float(nr_of_intervals-1-(interval_nr-1))/float(nr_of_intervals)]
        min_res, max_res = min(residuals_mean_per_chid), max(residuals_mean_per_chid)  # TODO set lower
        plt.plot([low_limit, low_limit], [min_res-offset, max_res+offset], "k-")
        plt.plot([upp_limit, upp_limit], [min_res-offset, max_res+offset], "k-")

        plt.annotate(s='', xy=(low_limit, 0), xytext=(upp_limit, 0),
                     arrowprops=dict(arrowstyle='<->', color=c))

        plt.plot(datetime_interval, residuals_mean_per_chid, "x", color=c)

    residuals_list = np.concatenate(residuals_collection).ravel()
    time_list = np.concatenate(time_collection).ravel()

    min_res, max_res = min(residuals_list), max(residuals_list)

    # plt.errorbar(datetime, residualsMean, yerr=residualsMeanStd, fmt="x",
    #              label=(drs_value_type+r"Mean - f(t)"))
    plt.title(drs_value_type+"residuals Mean:")
    plt.ylabel("(f(t)-"+drs_value_type+"Mean)/$\mathrm{mV}$")
    plt.gcf().autofmt_xdate()
    #plt.xlim(min(date)-pd.DateOffset(days=7), max(date)+pd.DateOffset(days=7))
    plt.ylim(min_res-offset, max_res+offset)
    plt.grid()
    plt.text(0.02, 0.2, "preliminary", fontdict=font, transform=plt.gca().transAxes)
    if(store_file_path is not None):
        plt.savefig(store_file_path)
    plt.show()
    plt.close()


@click.command()
@click.argument('drs_file_path',
                default="/net/big-tank/POOL/projects/fact/drs_temp_calib_data/" +
                        "calibration/drsSourceCollection/drsData.h5",
                type=click.Path(exists=True))
@click.argument('interval_file_path',
                default="/net/big-tank/POOL/projects/fact/drs_temp_calib_data/" +
                        "calibration/drsSourceCollection/intervalIndices.h5",
                type=click.Path(exists=True))
@click.argument("residuals_file_path",
                default="/net/big-tank/POOL/projects/fact/drs_temp_calib_data/" +
                        "calibration/residuals/drsResiduals.h5",
                type=click.Path(exists=True))
@click.argument("interval_array",
                default=[3])
@click.argument("drs_value_type",
                default="Baseline")
@click.argument("crate_nr",
                default=1)
@click.argument("restrict_residuals",
                default=True,
                type=click.BOOL)
@click.argument("store_file_path",
                default="/home/fschulz/plots/residualsBaseline_chid1000_cap500_3Intervals_RestrictParts.jpg",
                type=click.Path(exists=False))
####################################################################################################
def residuals_mean_per_crate(drs_file_path, interval_file_path, residuals_file_path, interval_array, drs_value_type,
                             crate_nr, restrict_residuals, store_file_path):

    crate_index = crate_nr-1
    with h5py.File(drs_file_path, 'r') as store:
        time = np.array(store["Time"+drs_value_type]).flatten()
        datetime = pd.to_datetime(time * 24 * 3600 * 1e9)

    date = datetime.date

    offset = 0.1  # TODO maybe ask/ magic number
    time_collection = []
    residuals_collection = []
    for interval_nr in interval_array:
        groupname = "Interval"+str(interval_nr)
        with h5py.File(residuals_file_path, 'r') as residuals_tab:
            data = residuals_tab[groupname]

            low_limit = pd.to_datetime(data.attrs["LowLimit"])
            upp_limit = pd.to_datetime(data.attrs["UppLimit"])

            residuals = np.array(data[drs_value_type+"Residuals"][crate_index*int(NRCHID/4)*NRCELL: (crate_index+1)*int(NRCHID/4)*NRCELL])

        interval_indices = np.where((datetime >= low_limit) & (datetime <= upp_limit))[0]
        datetime_interval = datetime[interval_indices]
        with h5py.File(drs_file_path, 'r') as store:
            drs_value_std = np.array(store[drs_value_type+"MeanStd"][crate_index*int(NRCHID/4)*NRCELL: (crate_index+1)*int(NRCHID/4)*NRCELL])

        if(False):  # TODO fix
            print(drs_value_std.shape)
            drs_value_std_mean = np.mean(drs_value_std, dtype='float64', axis=1)
            print(drs_value_std_mean.shape)
            drs_value_std_limit = drs_value_std_mean*2  # cut_off_factor[drs_value_type]
            indices_used_values = np.where(drs_value_std_mean < drs_value_std_limit)[0]
            indices_not_used_values = np.where(drs_value_std_mean >= drs_value_std_limit)[0]
            print(residuals.shape)
            print(indices_used_values)
            residuals = residuals[: , indices_used_values]
            datetime_interval = datetime_interval[indices_used_values]

        residuals_mean_per_chid = np.mean(residuals, dtype='float64', axis=0)
        residuals_collection.append(residuals_mean_per_chid)
        time_collection.append(datetime_interval)

        nr_of_intervals = len(interval_array)
        interval_nr =1  # TODO fix
        c = [float(interval_nr-1)/float(nr_of_intervals),
             0.0,
             float(nr_of_intervals-1-(interval_nr-1))/float(nr_of_intervals)]
        min_res, max_res = min(residuals_mean_per_chid), max(residuals_mean_per_chid)  # TODO set lower
        plt.plot([low_limit, low_limit], [min_res-offset, max_res+offset], "k-")
        plt.plot([upp_limit, upp_limit], [min_res-offset, max_res+offset], "k-")

        plt.annotate(s='', xy=(low_limit, 0), xytext=(upp_limit, 0),
                     arrowprops=dict(arrowstyle='<->', color=c))

        plt.plot(datetime_interval, residuals_mean_per_chid, "x", color=c)

    residuals_list = np.concatenate(residuals_collection).ravel()
    time_list = np.concatenate(time_collection).ravel()

    min_res, max_res = min(residuals_list), max(residuals_list)

    # plt.errorbar(datetime, residualsMean, yerr=residualsMeanStd, fmt="x",
    #              label=(drs_value_type+r"Mean - f(t)"))
    plt.title(drs_value_type+"residuals Mean:")
    plt.ylabel("(f(t)-"+drs_value_type+"Mean)/$\mathrm{mV}$")
    plt.gcf().autofmt_xdate()
    #plt.xlim(min(date)-pd.DateOffset(days=7), max(date)+pd.DateOffset(days=7))
    plt.ylim(min_res-offset, max_res+offset)
    plt.grid()
    plt.text(0.02, 0.2, "preliminary", fontdict=font, transform=plt.gca().transAxes)
    if(store_file_path is not None):
        plt.savefig(store_file_path)
    plt.show()
    plt.close()


####################################################################################################
def residualsMeanOfAllCellsPerCrates(drs_file_path, residualsFilenameArray_, drs_value_type,
                                     restrict_residuals, store_file_path=None):

    nr_of_intervals = len(residualsFilenameArray_)

    print("Loading '"+drs_value_type+"-data' ...")
    with h5py.File(drs_file_path, 'r') as store:
        time = np.array(store["Time"+drs_value_type]).flatten()
        datetime = pd.to_datetime(time * 24 * 3600 * 1e9)

    date = datetime.date
    datetime = np.array(datetime)

    intervalList = []
    residualsPairList = []
    min_res, max_res = 0, 0
    min_res_, max_res_ = 0, 0
    for residualsFilename in residualsFilenameArray_:
        with h5py.File(residualsFilename, 'r') as residuals_tab:
            residuals = np.array(residuals_tab["residuals"+drs_value_type])
            residualsMean = np.mean(residuals, dtype="float64", axis=0)
            residualsMeanC1 = np.mean(residuals[0*9*NRCELL:40*9*NRCELL, :], dtype="float64", axis=0)
            residualsMeanC2 = np.mean(residuals[40*9*NRCELL:80*9*NRCELL, :], dtype="float64", axis=0)
            residualsMeanC3 = np.mean(residuals[80*9*NRCELL:120*9*NRCELL, :], dtype="float64", axis=0)
            residualsMeanC4 = np.mean(residuals[120*9*NRCELL:160*9*NRCELL, :], dtype="float64", axis=0)

            residualsMeanPerCrates = np.array([residualsMeanC1, residualsMeanC2,
                                              residualsMeanC3, residualsMeanC4,
                                              residualsMean])

            interval_b = np.array(residuals_tab["Interval"])

            interval = []
            for intervalLimit in interval_b:
                interval.append(pd.to_datetime(intervalLimit[0].decode("UTF-8")).date())

            if(restrict_residuals):
                intervalIndices = np.where((date >= interval[0]) & (date <= interval[1]))[0]

                residualsPair = [datetime[intervalIndices], residualsMeanPerCrates[:, intervalIndices]]
            else:
                residualsPair = [datetime, residualsMeanPerCrates]

            min_res_, max_res_ = np.amin(residualsPair[1]), np.amax(residualsPair[1])
            if(min_res_ < min_res):
                min_res = min_res_
            if(max_res_ > max_res):
                max_res = max_res_

        intervalList.append(interval)
        residualsPairList.append(residualsPair)

    offset = min([abs(min_res*0.1), abs(max_res*0.1)])
    if(nr_of_intervals > 1):
        for intervalIndex in range(nr_of_intervals):
            plt.plot([intervalList[intervalIndex][0], intervalList[intervalIndex][0]],
                     [min_res-offset, max_res+offset], "k-")
            plt.plot([intervalList[intervalIndex][1], intervalList[intervalIndex][1]],
                     [min_res-offset, max_res+offset], "k-")

            print(residualsPairList[intervalIndex][0].shape, residualsPairList[intervalIndex][1].shape)

            print(residualsPairList[intervalIndex][0].shape, residualsPairList[intervalIndex][1][0].shape)
            print(residualsPairList[intervalIndex][0].shape, residualsPairList[intervalIndex][1][1].shape)
            print(residualsPairList[intervalIndex][0].shape, residualsPairList[intervalIndex][1][2].shape)

            plt.plot(residualsPairList[intervalIndex][0], residualsPairList[intervalIndex][1][0])
            plt.plot(residualsPairList[intervalIndex][0], residualsPairList[intervalIndex][1][1])
            plt.plot(residualsPairList[intervalIndex][0], residualsPairList[intervalIndex][1][2])
            plt.plot(residualsPairList[intervalIndex][0], residualsPairList[intervalIndex][1][3])
            plt.plot(residualsPairList[intervalIndex][0], residualsPairList[intervalIndex][1][4])

        plt.plot(residualsPairList[0][0], residualsPairList[0][1][0], "bx", label="Crate 1")
        plt.plot(residualsPairList[0][0], residualsPairList[0][1][1], "gx", label="Crate 2")
        plt.plot(residualsPairList[0][0], residualsPairList[0][1][2], "yx", label="Crate 3")
        plt.plot(residualsPairList[0][0], residualsPairList[0][1][3], "rx", label="Crate 4")
        plt.plot(residualsPairList[0][0], residualsPairList[0][1][4], "ko", label="Crate 1-4")
        plt.plot([date[0], date[0]], [min_res, max_res], "k-", label="Interval boundary")

    else:
        plt.plot(datetime, residualsMeanC1, "bx", label="Crate 1")
        plt.plot(datetime, residualsMeanC2, "gx", label="Crate 2")
        plt.plot(datetime, residualsMeanC3, "yx", label="Crate 3")
        plt.plot(datetime, residualsMeanC4, "rx", label="Crate 4")
        plt.plot(datetime, residualsMean, "ko", label="Crate 1-4")

    plt.title(drs_value_type+"residuals Mean per Crate:")
    plt.ylabel("(f(t)-"+drs_value_type+r"Mean)/$\mathrm{mV}$")
    plt.gcf().autofmt_xdate()
    plt.xlim(min(date)-pd.DateOffset(days=7), max(date)+pd.DateOffset(days=7))
    plt.ylim(min_res-offset, max_res+offset)
    plt.grid()
    plt.legend(loc='lower left', numpoints=1)
    plt.text(0.02, 0.2, "preliminary", fontdict=font, transform=plt.gca().transAxes)
    if(store_file_path is not None):
        plt.savefig(store_file_path)
    plt.show()
    plt.close()


####################################################################################################
def residualsMeanPerPatchAndInterval(drs_file_path, residualsFilenameArray_, drs_value_type, store_file_path=None):

    print("Loading '"+drs_value_type+"-data' ...")
    with h5py.File(drs_file_path, 'r') as store:
        time = np.array(store["Time"+drs_value_type]).flatten()
        date = pd.to_datetime(time * 24 * 3600 * 1e9).date

    residualsMeanPerPatchAndInterval = []
    for residualsFilename in residualsFilenameArray_:
        intervalresidualsMeanPerPatch = []
        with h5py.File(residualsFilename, 'r') as residuals_tab:
            interval_b = np.array(residuals_tab["Interval"])

            interval = []
            for intervalLimit in interval_b:
                interval.append(pd.to_datetime(intervalLimit[0].decode("UTF-8")).date())

            intervalIndices = np.where((date >= interval[0]) & (date <= interval[1]))[0]

            residuals = np.array(residuals_tab["residuals"+drs_value_type])[:, intervalIndices]
        for patchNr in range(NRPATCH):
            intervalresidualsMeanPerPatch.append(
                np.mean(abs(
                            residuals[patchNr*9*NRCELL:(patchNr+1)*9*NRCELL].flatten()
                            ), dtype="float64"))
        residualsMeanPerPatchAndInterval.append(intervalresidualsMeanPerPatch)

    residualsMeanPerPatchAndInterval = np.array(residualsMeanPerPatchAndInterval).transpose()

    plt.matshow(residualsMeanPerPatchAndInterval, interpolation="None", aspect='auto')
    # plt.title(r"Mean of the absolute "+str(drs_value_type)+"residuals-value \n per Interval ", fontsize=25, y=1.02)
    cbar = plt.colorbar()
    resMax = residualsMeanPerPatchAndInterval.shape
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
    if(store_file_path is not None):
        plt.savefig(store_file_path)
    plt.show()
    plt.close()


# TODO update
###############################################################################
def noise(drs_file_calibrated, drs_model_calibrated,
          title_str, store_file_path=None, source_file=None):

    if(source_file is not None):  # TODO remove
        print("load new File '", source_file, "'")
        with fits.open(source_file) as noise_tab:
            drs_file_calibrated = noise_tab[1].data["DrsCalibratedDataNoise"]
            drs_model_calibrated = noise_tab[1].data["DrsCalibratedDataNoiseTemp"]

    chid_list = np.linspace(0, NRCHID-1, NRCHID, dtype="int")
    useful_chid = chid_list[np.setdiff1d(chid_list,
                                         np.array([
                                            non_standard_chids["crazy"],
                                            non_standard_chids["dead"]])-1)]

    # TODO update maybe use repeat
    drs_file_calibrated = np.array(drs_file_calibrated).reshape(-1, NRCHID)[:, useful_chid].flatten()
    drs_model_calibrated = np.array(drs_model_calibrated).reshape(-1, NRCHID)[:, useful_chid].flatten()

    xlim = 5
    xlabel = r'Noise /$\mathrm{mV}$'
    in_pea = True
    if(in_pea):
        drs_file_calibrated = drs_file_calibrated*PEAFACTOR
        drs_model_calibrated = drs_model_calibrated*PEAFACTOR
        xlim = xlim*PEAFACTOR
        xlabel = r'Noise /$\mathrm{PEA}$'

    nr_bins = int(xlim*100)
    weights = np.full(len(drs_file_calibrated),
                      len(drs_file_calibrated*0.01))

    drs_file_calibrated_mean = np.mean(drs_file_calibrated)
    drs_model_calibrated_mean = np.mean(drs_model_calibrated)

    drs_file_calibrated_std = np.std(drs_file_calibrated, dtype="float64", ddof=1)
    drs_model_calibrated_std = np.std(drs_model_calibrated, dtype="float64", ddof=1)

    gs = gridspec.GridSpec(4, 1)
    #ax0 = plt.subplot(gs[0:3, :])
    ax0 = plt.subplot(gs[0:4, :])
    plt.title(title_str, y=1.0)
    hist1 = ax0.hist(drs_file_calibrated, weights=weights, bins=nr_bins, histtype='step',
                     range=(0.0, xlim), lw=1, edgecolor="r", label="Drs-File Noise\nMean: "+str(format(round(drs_file_calibrated_mean, 3), '.3f'))+", Std: "+str(format(round(drs_file_calibrated_std, 3), '.3f')))
    hist2 = ax0.hist(drs_model_calibrated, weights=weights, bins=nr_bins, histtype='step',
                     range=(0.0, xlim), lw=1, edgecolor="g", label="Model Noise\nMean: "+str(format(round(drs_model_calibrated_mean, 3), '.3f'))+", Std: "+str(format(round(drs_model_calibrated_std, 3), '.3f')))

    plt.ylabel(r'frequency /$\mathrm{\%}$')
    plt.legend(loc='upper right', numpoints=1, title="")

    #ax1 = plt.subplot(gs[3, :])
    #ax1.step(hist1[1][0:-1], hist2[0]-hist1[0], "g")
    #ax1.step(hist1[1][0:-1], hist3[0]-hist1[0], "b")
    plt.xlabel(xlabel)
    plt.ylabel(r'$\Delta$ frequency /$\mathrm{\%}$')
    #plt.setp(ax0.get_xticklabels(), visible=False)
    plt.text(0.7, 0.15, "preliminary", fontdict=font, transform=plt.gca().transAxes)
    if(store_file_path is not None):
        plt.savefig(store_file_path)
    plt.show()
    plt.close()


###############################################################################
def noise_fact_cam(drs_file_calibrated, drs_model_calibrated,
                   store_file_path=None, source_file=None):

    if(source_file is not None):
        print("load new File '", source_file, "'")
        with fits.open(source_file) as noise_tab:
            drs_file_calibrated = noise_tab[1].data["DrsCalibratedDataNoise"].flatten()
            drs_model_calibrated = noise_tab[1].data["DrsCalibratedDataNoiseTemp"].flatten()

    drs_file_calibrated_chid_mean = np.mean(np.array(drs_file_calibrated).reshape(-1, NRCHID), axis=0)
    drs_model_calibrated_chid_mean = np.mean(np.array(drs_model_calibrated).reshape(-1, NRCHID), axis=0)

    non_standard_chids_indices = np.array([non_standard_chids["crazy"], non_standard_chids["dead"]]).flatten()-1
    drs_file_calibrated_chid_mean[non_standard_chids_indices] = 0.
    drs_model_calibrated_chid_mean[non_standard_chids_indices] = 0.

    gs = gridspec.GridSpec(10, 11)
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)

    max_noise = math.ceil(np.amax([drs_file_calibrated_chid_mean,
                                   drs_model_calibrated_chid_mean]))
    vmin, vmax = 0, max_noise

    ax0b = plt.subplot(gs[0:9, 0:5])
    ax0b.set_axis_off()
    ax0b.set_xlabel('')
    ax0b.set_ylabel('')
    ax0b.set_title('file calibrated', fontsize=18)

    camera_plot_f = camera(drs_file_calibrated_chid_mean,
                           vmin=vmin, vmax=vmax, cmap='hot')

    ax0g = plt.subplot(gs[0:9, 5:11])
    ax0g.set_axis_off()
    ax0g.set_xlabel('')
    ax0g.set_ylabel('')
    ax0g.set_title('model calibrated', fontsize=18)

    camera_plot_m = camera(drs_model_calibrated_chid_mean,
                           vmin=vmin, vmax=vmax, cmap='hot')

    cbar = fig.colorbar(camera_plot_m, ax=ax0g)
    cbar.set_label(r"Noise /$\mathrm{mV}$")
    if(store_file_path is not None):
        plt.savefig(store_file_path+".jpg")
    plt.show()
    plt.close()


# TODO update
@click.command()
@click.argument("filename",
                default="/net/big-tank/POOL/projects/fact/" +
                        "drs_temp_calib_data/calibration/validating/noise/" +
                        "pedestelNoise_20140911.fits",
                type=click.Path(exists=True))
@click.argument("save",
                default=True,
                type=click.BOOL)
###############################################################################
def pedestial_noise(filename, save):

    with fits.open(filename) as noise_tab:
        date = pd.to_datetime(noise_tab[1].header["Date"]).date()
        temp_diff = noise_tab[1].data["TempDiff"]
        run_ids = noise_tab[1].data["PedestelRunId"]
        drs_file_noise = noise_tab[1].data["DrsCalibratedDataNoise"]
        drs_model_noise = noise_tab[1].data["DrsCalibratedDataNoiseTemp"]

    print("date ", date)
    print("run_ids ", run_ids)
    print("temp_diff ", temp_diff)
    print("drs_file_noise ", np.array(drs_file_noise[0]).shape)
    print("drs_model_noise ", np.array(drs_model_noise[0]).shape)
    run_ids = [run_ids_ for
               (temp_diff_, run_ids_) in
               sorted(zip(temp_diff, run_ids),
                      key=lambda pair: pair[0])]

    drs_file_noise = [drs_file_noise_ for
                      (temp_diff_, drs_file_noise_) in
                      sorted(zip(temp_diff, drs_file_noise),
                             key=lambda pair: pair[0])]

    drs_model_noise = [drs_model_noise_ for
                       (temp_diff_, drs_model_noise_) in
                       sorted(zip(temp_diff, drs_model_noise),
                              key=lambda pair: pair[0])]

    temp_diff = sorted(temp_diff)

    print("date ", date)
    print("run_ids ", run_ids)
    print("temp_diff ", temp_diff)
    print("drs_file_noise ", np.array(drs_file_noise[0]).shape)
    print("drs_model_noise ", np.array(drs_model_noise[0]).shape)

    dateStr = date.strftime('%Y-%m-%d')
    dateStr2 = date.strftime('%Y%m%d')

    for i in range(len(drs_file_noise)):
        store_filename = ("/home/fschulz/plots/noise/pedestelNoise" +
                          dateStr+"_runId"+str(run_ids[i])+".jpg")
        #title_str = ("Standard deviation "+dateStr+"\n " +
        #            "runID: "+str(run_ids[i])+", Temperature difference "+str(round(temp_diff[i], 3))+r"$^\circ C$")
        title_str = ("Temperature difference "+str(round(temp_diff[i], 3))+r"$^\circ C$")
        noise(drs_file_noise[i], drs_model_noise[i], title_str, store_filename)
        store_filename = ("/home/fschulz/plots/noise/pedestelnoise_fact_cam" +
                          dateStr+"_runId"+str(int(run_ids[i])))
        noise_fact_cam(drs_file_noise[i], drs_model_noise[i], store_filename)


@click.command()
@click.argument("folder",
                default="/net/big-tank/POOL/projects/fact/" +
                        "drs_temp_calib_data/calibration/validating/noise/",
                type=click.Path(exists=True))
@click.argument("save",
                default=True,
                type=click.BOOL)
###############################################################################
def noise_mean_hist(folder, save):
    print("hist")
    store_file_path = "/home/fschulz/plots/noise/pedestelNoiseDistribution.jpg"
    noise_file_list = sorted([file for file in
                              os.listdir(folder)
                              if (file.startswith('pedestelNoise_') and
                                  file.endswith('.fits'))])

    chid_list = np.linspace(0, NRCHID-1, NRCHID, dtype="int")
    useful_chid = chid_list[np.setdiff1d(chid_list,
                                         np.array([
                                            non_standard_chids["crazy"],
                                            non_standard_chids["dead"]])-1)]
    drs_file_calibrated_mean_list = []
    drs_model_calibrated_mean_list = []
    for noise_file_path in tqdm(noise_file_list):
        with fits.open(folder+noise_file_path) as noise_tab:
            nr_runs = len(noise_tab[1].data["PedestelRunId"])
            drs_file_calibrated_of_the_day = noise_tab[1].data["DrsCalibratedDataNoise"]
            drs_model_calibrated_of_the_day = noise_tab[1].data["DrsCalibratedDataNoiseTemp"]

        for run_index in range(nr_runs):
            drs_file_calibrated = np.array(drs_file_calibrated_of_the_day[run_index]).reshape(-1, NRCHID)[:, useful_chid].flatten()
            drs_model_calibrated = np.array(drs_model_calibrated_of_the_day[run_index]).reshape(-1, NRCHID)[:, useful_chid].flatten()

            drs_file_calibrated_mean = np.mean(drs_file_calibrated)
            drs_model_calibrated_mean = np.mean(drs_model_calibrated)

            in_pea = True
            if(in_pea):
                drs_file_calibrated_mean = drs_file_calibrated_mean*PEAFACTOR
                drs_model_calibrated_mean = drs_model_calibrated_mean*PEAFACTOR

            drs_file_calibrated_mean_list.append(drs_file_calibrated_mean)
            drs_model_calibrated_mean_list.append(drs_model_calibrated_mean)

    drs_file_calibrated_collection_mean = np.mean(drs_file_calibrated_mean_list)
    drs_model_calibrated_collection_mean = np.mean(drs_model_calibrated_mean_list)

    drs_file_calibrated_collection_std = np.std(drs_file_calibrated_mean_list, dtype="float64", ddof=1)
    drs_model_calibrated_collection_std = np.std(drs_model_calibrated_mean_list, dtype="float64", ddof=1)

    hist_range = [1.8, 5.0]
    nr_bins = int((hist_range[1]-hist_range[0])*100)
    weights = np.full(len(drs_file_calibrated_mean_list),
                      100/len(drs_file_calibrated_mean_list))
    xlabel = r'Noise /$\mathrm{mV}$'
    in_pea = True
    if(in_pea):
        hist_range[0] = hist_range[0]*PEAFACTOR
        hist_range[1] = hist_range[1]*PEAFACTOR
        xlabel = r'Noise /$\mathrm{PEA}$'

    label_str = "Drs-File Noise\nMean: "+str(format(round(drs_file_calibrated_collection_mean, 3), '.3f'))+", Std: "+str(format(round(drs_file_calibrated_collection_std, 3), '.3f'))
    plt.hist(drs_file_calibrated_mean_list, weights=weights, bins=nr_bins, histtype='step',
             range=(hist_range[0], hist_range[1]), lw=1, edgecolor="r", label=label_str)
    label_str = "Model Noise\nMean: "+str(format(round(drs_model_calibrated_collection_mean, 3), '.3f'))+", Std: "+str(format(round(drs_model_calibrated_collection_std, 3), '.3f'))
    plt.hist(drs_model_calibrated_mean_list, weights=weights, bins=nr_bins, histtype='step',
             range=(hist_range[0], hist_range[1]), lw=1, edgecolor="g", label=label_str)

    plt.xlabel(xlabel)
    plt.ylabel(r'frequency /$\mathrm{\%}$')
    plt.legend(loc='upper right', numpoints=1, title="")
    plt.text(0.7, 0.15, "preliminary", fontdict=font, transform=plt.gca().transAxes)
    if(store_file_path is not None):
        plt.savefig(store_file_path)
    plt.show()
    plt.close()


@click.command()
@click.argument("folder",
                default="/net/big-tank/POOL/projects/fact/" +
                        "drs_temp_calib_data/calibration/validating/noise/",
                type=click.Path(exists=True))
@click.argument("save",
                default=True,
                type=click.BOOL)
###############################################################################
def noise_mean_vs_temp(folder, save):
    print("temp")
    datetime_limits = [pd.to_datetime("2014"), pd.to_datetime("2017")]
    store_file_path = "/home/fschulz/plots/noise/pedestelNoiseTempDistribution.jpg"
    noise_file_list = sorted([file for file in
                              os.listdir(folder)
                              if (file.startswith('pedestelNoise_') and
                                  file.endswith('.fits'))])

    chid_list = np.linspace(0, NRCHID-1, NRCHID, dtype="int")
    useful_chid = chid_list[np.setdiff1d(chid_list,
                                         np.array([
                                            non_standard_chids["crazy"],
                                            non_standard_chids["dead"]])-1)]

    temp_diff_list = []
    time_list = []
    drs_file_calibrated_mean_list = []
    drs_model_calibrated_mean_list = []
    for noise_file_path in tqdm(noise_file_list):
        datetime = pd.to_datetime(noise_file_path.split('_')[-1].split('.')[0])
        if(datetime < datetime_limits[0] or datetime > datetime_limits[1]):
            continue
        with fits.open(folder+noise_file_path) as noise_tab:
            nr_runs = len(noise_tab[1].data["PedestelRunId"])
            temp_diff = noise_tab[1].data["TempDiff"]
            drs_file_calibrated_of_the_day = noise_tab[1].data["DrsCalibratedDataNoise"]
            drs_model_calibrated_of_the_day = noise_tab[1].data["DrsCalibratedDataNoiseTemp"]

        for run_index in range(nr_runs):
            drs_file_calibrated = np.array(drs_file_calibrated_of_the_day[run_index]).reshape(-1, NRCHID)[:, useful_chid].flatten()
            drs_model_calibrated = np.array(drs_model_calibrated_of_the_day[run_index]).reshape(-1, NRCHID)[:, useful_chid].flatten()

            drs_file_calibrated_mean = np.mean(drs_file_calibrated)
            drs_model_calibrated_mean = np.mean(drs_model_calibrated)

            in_pea = True
            if(in_pea):
                drs_file_calibrated_mean = drs_file_calibrated_mean*PEAFACTOR
                drs_model_calibrated_mean = drs_model_calibrated_mean*PEAFACTOR

            time_list.append(datetime.value / 24 / 3600 / 1e9)
            temp_diff_list.append(temp_diff[run_index])
            drs_file_calibrated_mean_list.append(drs_file_calibrated_mean)
            drs_model_calibrated_mean_list.append(drs_model_calibrated_mean)

    drs_file_calibrated_collection_mean = np.mean(drs_file_calibrated_mean_list)
    drs_model_calibrated_collection_mean = np.mean(drs_model_calibrated_mean_list)

    drs_file_calibrated_collection_std = np.std(drs_file_calibrated_mean_list, dtype="float64", ddof=1)
    drs_model_calibrated_collection_std = np.std(drs_model_calibrated_mean_list, dtype="float64", ddof=1)

    ylabel = r'Noise /$\mathrm{mV}$'
    in_pea = True
    if(in_pea):
        ylabel = r'Noise /$\mathrm{PEA}$'

    fig, img = plt.subplots()

    label_str = ("Drs-File Noise\nMean: "+str(format(round(drs_file_calibrated_collection_mean, 3), '.3f')) +
                 ", Std: "+str(format(round(drs_file_calibrated_collection_std, 3), '.3f')))
    sc_f = plt.scatter(temp_diff_list, drs_file_calibrated_mean_list,
                       s=50, marker="+", c=time_list, label=label_str)
    label_str = ("Model Noise\nMean: "+str(format(round(drs_model_calibrated_collection_mean, 3), '.3f')) +
                 ", Std: "+str(format(round(drs_model_calibrated_collection_std, 3), '.3f')))
    sc_m = plt.scatter(temp_diff_list, drs_model_calibrated_mean_list,
                       s=50, marker="*", c=time_list, label=label_str)


    intervalMonth = 3
    start_date = pd.to_datetime(time_list[0] * 24 * 3600 * 1e9).date()
    end_date = pd.to_datetime(time_list[-1] * 24 * 3600 * 1e9).date()
    timeLabel = pd.date_range(start=start_date, end=end_date, freq=str(intervalMonth)+"M")
    cbar = fig.colorbar(sc_f, ticks=dates.MonthLocator(interval=intervalMonth))
    cbar.ax.set_yticklabels(timeLabel.strftime("%b %Y"))
    timeColor = cbar.to_rgba(time_list)

    plt.xlabel(r'Temperatur /$\mathrm{C\degree}$')
    plt.ylabel(ylabel)
    plt.legend(loc='upper right', numpoints=1, title="")
    plt.text(0.6, 0.15, "preliminary", fontdict=font, transform=plt.gca().transAxes)
    if(store_file_path is not None):
        plt.savefig(store_file_path)
    plt.show()
    plt.close()
