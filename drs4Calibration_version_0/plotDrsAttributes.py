import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import sys

import os
import numpy as np
import pandas as pd
import math
import h5py
import click
# import yaml

from scipy.stats import linregress
from tqdm import tqdm
from fact.instrument.camera import non_standard_pixel_chids as non_standard_chids
import fact.plotting as factPlots
from astropy.io import fits
from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import hot

import config as config
from constants import NRCHID, NRCELL, NRPATCH, PEAFACTOR, DACfactor


###############################################################################
# ##############                    Helper                     ############## #
###############################################################################

font = {'family': 'serif',
        'color': 'grey',
        'weight': 'bold',
        'size': 16,
        'alpha': 0.5,
        }


###############################################################################
def get_best_limits(value, scale=0.02):
    min, max = np.amin(value), np.amax(value)
    range = max-min
    offset = range*scale
    return [min-offset, max+offset]


###############################################################################
def linearerFit(x, m, b):
    return (m*x+b)


###############################################################################
def get_not_useful_chids(interval_nr):

    # default values
    not_useful_chids = np.array([non_standard_chids['crazy'],
                                 non_standard_chids['dead']]).flatten()

    if(interval_nr == 2):
        not_useful_chids = np.append(not_useful_chids, np.arange(720, 755+1, 1))
    elif(interval_nr == 3):
        not_useful_chids = np.append(not_useful_chids, np.arange(1296, 1299+1, 1))
        not_useful_chids = np.append(not_useful_chids, np.arange(697, 701+1, 1))
    # print(np.sort(not_useful_chids))
    return np.unique(not_useful_chids)


###############################################################################
def get_useful_chids(interval_nr):

    chid_list = np.linspace(0, NRCHID-1, NRCHID, dtype='int')

    not_useful_chids = get_not_useful_chids(interval_nr)
    useful_chids = chid_list[np.setdiff1d(chid_list, not_useful_chids)]
    return useful_chids


###############################################################################
# ##############               Drs-Value  Plots                ############## #
###############################################################################

###############################################################################
@click.command()
@click.argument('data_collection_file_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_0/dataCollection.h5',
                type=click.Path(exists=True))
@click.argument('store_file_path',
                default='/home/fschulz/plots/version_0/drsValues/baseline/chid250_cell250_20160101_20171001.png',
                type=click.Path(exists=False))
@click.argument('date_range',
                default=['2016-01-01', '2017-10-01'])
@click.option('--drs_value_type', '-type',
              default='Baseline',
              type=click.Choice(['Baseline', 'Gain', 'ROIOffset']))
@click.argument('chid',
                default=250)
@click.argument('cell',
                default=250)
###############################################################################
def general_drs_value_cell(data_collection_file_path, store_file_path,
                           date_range, drs_value_type,
                           chid, cell):

    general_drs_value_cell_(data_collection_file_path, store_file_path,
                            date_range, drs_value_type,
                            chid, cell)


###############################################################################
def general_drs_value_cell_(data_collection_file_path, store_file_path,
                            date_range, drs_value_type,
                            chid, cell):

    NRCELLSPERCHID = config.nrCellsPerChid[drs_value_type]
    if(cell > NRCELLSPERCHID):
        print('ERROR: cell > '+str(NRCELLSPERCHID))
        return

    value_index = chid*NRCELLSPERCHID + cell

    # loading source data
    with h5py.File(data_collection_file_path, 'r') as store:
        time = np.array(store['Time'+drs_value_type]).flatten()

    datetime = pd.to_datetime(time * 24 * 3600 * 1e9)

    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])

    indices = np.where((datetime > start_date) & (datetime < end_date))[0]
    time = time[indices]
    datetime = datetime[indices]
    with h5py.File(data_collection_file_path, 'r') as store:
        temp = np.array(store['Temp'+drs_value_type][indices, int(chid/9)])
        drs_value = np.array(store[drs_value_type][indices, value_index])
        drs_value_std = np.sqrt(np.array(store[drs_value_type+'Var'][indices, value_index]))

    if(drs_value_type == 'Gain'):
        drs_value /= DACfactor
        drs_value_std /= DACfactor

    print(np.mean(drs_value_std))
    intervalMonth = 3

    sc_all = plt.scatter(temp, drs_value, c=time)
    plt.close()  # Dont show sc_all, just use it to create the colorbar

    fig, img = plt.subplots()

    start_date = datetime[0].date()
    end_date = datetime[-1].date()
    time_label = pd.date_range(start=start_date, end=end_date, freq=str(intervalMonth)+'M')
    cbar = fig.colorbar(sc_all, ticks=mdates.MonthLocator(interval=intervalMonth))
    cbar.ax.set_yticklabels(time_label.strftime('%b %Y'))
    #cbar.ax.minorticks_on()
    time_color = cbar.to_rgba(time)

    sc_all = plt.scatter(temp, drs_value, marker='.', s=25, c=time, alpha=0.45)

    slope, offset, r_value, p_value, var_err = linregress(temp, drs_value)
    print(slope, offset)
    print(var_err)
    temp_range = np.linspace(10, 45, 10000)
    y_fit = linearerFit(temp_range, slope, offset)

    ylabel_str = drs_value_type+r' /$\mathrm{mV}$'
    slope_unit_str = r'$\,\frac{\mathrm{mV}}{°C}$'
    offset_unit_str = r'$\,\mathrm{mV}$'
    if(drs_value_type == 'Gain'):
        drs_value *= pow(10, 3)
        y_fit *= pow(10, 3)
        slope *= pow(10, 3)
        offset *= pow(10, 3)
        ylabel_str = drs_value_type+r' /$10^{-3}$'
        slope_unit_str = r'$\,\frac{10^{-3}}{°C}$'
        offset_unit_str = r'10^{-3}'

    color_mean = np.mean(time_color, axis=0)
    label_str = (r'm={:0.3f}'.format(slope) + slope_unit_str + ', ' +
                 r'b={:0.3f}'.format(offset) + offset_unit_str)
    fit_plot, = plt.plot(temp_range, y_fit, '-', color=color_mean,
                         label=label_str)
    plt.legend(handles=[fit_plot], loc='lower center', title=r'Linearer Fit: f(x) = m $\cdot$ x + b')

    plt.xlabel(r'Temperatur /$\mathrm{^\circ C}$')
    plt.ylabel(ylabel_str)
    plt.xlim(min(temp)-1, max(temp)+1)
    plt.ylim(get_best_limits(drs_value, 0.05))
    ymin, ymax = plt.gca().get_ylim()
    plt.ylim(get_best_limits(drs_value, 0.15)[0], ymax)
    plt.grid()
    plt.gca().ticklabel_format(useOffset=False)
    if(store_file_path is not None):
        plt.savefig(store_file_path)
    plt.show()
    plt.close()


###############################################################################
@click.command()
@click.argument('data_collection_file_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_0/dataCollection.h5',
                type=click.Path(exists=True))
@click.argument('store_file_path',
                default='/home/fschulz/plots/version_0/drsValues/baseline/res_chid500_cell500_20130607_20171001.png',
                type=click.Path(exists=False))
@click.argument('date_range',
                default=['2013-06-07', '2017-10-01'])
@click.option('--drs_value_type', '-type',
              default='Baseline',
              type=click.Choice(['Baseline', 'Gain', 'ROIOffset']))
@click.argument('chid',
                default=500)
@click.argument('cell',
                default=500)
####################################################################################################
def general_drs_value_cell_residual(data_collection_file_path, store_file_path,
                                    date_range, drs_value_type,
                                    chid, cell):

    general_drs_value_cell_residual(data_collection_file_path, store_file_path,
                                    date_range, drs_value_type,
                                    chid, cell)


####################################################################################################
def general_drs_value_cell_residual_(data_collection_file_path, store_file_path,
                                     date_range, drs_value_type,
                                     chid, cell):

    NRCELLSPERCHID = config.nrCellsPerChid[drs_value_type]
    if(cell > NRCELLSPERCHID):
        print('ERROR: cell > '+str(NRCELLSPERCHID))
        return

    value_index = chid*NRCELLSPERCHID + cell

    # loading source data
    with h5py.File(data_collection_file_path, 'r') as store:
        time = np.array(store['Time'+drs_value_type]).flatten()

    datetime = pd.to_datetime(time * 24 * 3600 * 1e9)

    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])

    indices = np.where((datetime > start_date) & (datetime < end_date))[0]
    time = time[indices]
    datetime = datetime[indices]

    with h5py.File(data_collection_file_path, 'r') as store:
        temp = np.array(store['Temp'+drs_value_type][indices, int(chid/9)])
        drs_value = np.array(store[drs_value_type][indices, value_index])

    ylabel_str = drs_value_type+r' /$\mathrm{mV}$'
    if(drs_value_type == 'Gain'):
        drs_value /= DACfactor
        ylabel_str = drs_value_type+r' /$\mathrm{1}$'

    slope, offset, r_value, p_value, var_err = linregress(temp, drs_value)

    model_value = slope*temp + offset
    residual = drs_value - model_value

    sc_all = plt.scatter(datetime, residual, marker='.', c=temp, alpha=1)
    cbar = plt.colorbar(sc_all)
    cbar.set_label(r'Temperatur /$\mathrm{^\circ C}$')

    plt.xticks(rotation=30)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator(interval=3))

    plt.axvline(x=pd.to_datetime('2014-05-20 12'), linewidth=2, color='k')
    plt.axvline(x=pd.to_datetime('2015-05-26 12'), linewidth=2, color='k')

    plt.ylabel(ylabel_str)
    #plt.xlim(min(temp)-1, max(temp)+1)
    plt.ylim(get_best_limits(residual, 0.05))
    plt.grid()
    if(store_file_path is not None):
        plt.savefig(store_file_path)
    plt.show()
    plt.close()


###############################################################################
@click.command()
@click.argument('data_collection_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_0/dataCollection.h5',
                type=click.Path(exists=True))
@click.argument('interval_file_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_0/intervalIndices.h5',
                type=click.Path(exists=True))
@click.argument('store_file_path',
                default='/home/fschulz/plots/version_0/drsValues/baseline/std_hist_baseline.png',
                type=click.Path(exists=False))
@click.argument('interval_array',
                default=[1, 2, 3])
@click.option('--drs_value_type', '-type',
              default='Baseline',
              type=click.Choice(['Baseline', 'Gain', 'ROIOffset']))
@click.argument('x_lim',
                default=[0.0, 4.2])
###############################################################################
def drs_value_std_hist(data_collection_path, interval_file_path,
                       store_file_path, interval_array,
                       drs_value_type, x_lim):

    drs_value_std_hist_(data_collection_path, interval_file_path,
                        store_file_path, interval_array,
                        drs_value_type, x_lim)


###############################################################################
def drs_value_std_hist_(data_collection_path, interval_file_path,
                        store_file_path, interval_array,
                        drs_value_type, x_lim):

    NRCELLSPERCHID = config.nrCellsPerChid[drs_value_type]

    factor_str = ''
    unit_str = r' $\mathrm{mV}$'

    for interval_nr in interval_array:
        groupname = 'Interval'+str(interval_nr)
        title_str = 'Intervall '+str(interval_nr)
        print(groupname)
        with h5py.File(interval_file_path, 'r') as interval_source:
            data = interval_source[groupname]
            interval_indices = np.array(data['IntervalIndices'])
        print('loading')
        with h5py.File(data_collection_path, 'r') as store:
            drs_value_var = store[drs_value_type+'Var'][interval_indices, :]

        useful_chids = get_useful_chids(interval_nr)
        useful_chids_ = np.arange(8, 1440, 9)
        drs_value_var_ = drs_value_var.reshape(-1, NRCHID, NRCELLSPERCHID)[:, useful_chids_, :].flatten()

        drs_value_var = drs_value_var.reshape(-1, NRCHID, NRCELLSPERCHID)[:, useful_chids, :].flatten()
        drs_value_std = np.sqrt(drs_value_var)
        drs_value_std_ = np.sqrt(drs_value_var_)
        del drs_value_var

        if(drs_value_type == 'Gain'):
            drs_value_std /= DACfactor/pow(10, 3)
            drs_value_std_ /= DACfactor/pow(10, 3)
            factor_str = r' $\cdot$'
            unit_str = r' $10^{-3}$'

        drs_value_std_mean = np.mean(drs_value_std)
        drs_value_std_std = np.std(drs_value_std)
        drs_value_std_max = max(drs_value_std)
        drs_value_std_min = min(drs_value_std)

        plot = plt.plot([], [])
        color = plot[0].get_color()
        label = (title_str+':' +
                 '\n'+r' $\overline{Hist}$: '+str(round(drs_value_std_mean, 2))+factor_str+unit_str +
                 '\n'+r' $\sigma_\mathrm{Hist}$: '+str(round(drs_value_std_std, 2))+factor_str+unit_str +
                 '\n'+r' $x_\mathrm{Max}$: '+str(round(drs_value_std_max, 2)))+factor_str+unit_str+'\n'
        weights = np.full(len(drs_value_std), 100/len(drs_value_std))
        nr_bins = int(abs((drs_value_std_max-drs_value_std_min))/abs(x_lim[1]-x_lim[0])*100)
        plt.hist(drs_value_std, bins=nr_bins, weights=weights,
                 histtype='step', label=label, color=color)

        plt.hist(drs_value_std, bins=nr_bins, weights=weights,
                 histtype='step', label=label, color=color, linestyle='dashed')

        del drs_value_std

    plt.xlabel(r'Standardabweichung /'+unit_str)
    plt.ylabel(r'Häufigkeit /$\mathrm{\%}$')
    plt.xlim(x_lim)
    plt.legend(loc='upper right')
    plt.tight_layout()
    if(store_file_path is not None):
        plt.savefig(store_file_path)
    plt.show()
    plt.close()


###############################################################################
@click.command()
@click.argument('data_collection_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_0/dataCollection.h5',
                type=click.Path(exists=True))
@click.argument('interval_file_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_0/intervalIndices.h5',
                type=click.Path(exists=True))
@click.argument('store_file_path',
                default='/home/fschulz/plots/version_0/drsValues/roiOffset/std_hist_chid250_cell250.png',
                type=click.Path(exists=False))
@click.argument('interval_array',
                default=[1, 2, 3])
@click.option('--drs_value_type', '-type',
              default='ROIOffset',
              type=click.Choice(['Baseline', 'Gain', 'ROIOffset']))
@click.argument('chid',
                default=250)
@click.argument('cell',
                default=250)
@click.argument('cut_off_error_factor',
                default=2)
@click.argument('x_lim',
                default=[0.0, 4.2])
###############################################################################
def drs_value_std_hist_per_chid_cell(data_collection_path, interval_file_path,
                                     store_file_path, interval_array,
                                     drs_value_type, chid,
                                     cell, cut_off_error_factor,
                                     x_lim):

    drs_value_std_hist_per_chid_cell_(data_collection_path, interval_file_path,
                                      store_file_path, interval_array,
                                      drs_value_type, chid,
                                      cell, cut_off_error_factor,
                                      x_lim)


###############################################################################
def drs_value_std_hist_per_chid_cell_(data_collection_path, interval_file_path,
                                      store_file_path, interval_array,
                                      drs_value_type, chid,
                                      cell, cut_off_error_factor,
                                      x_lim):

    NRCELLSPERCHID = config.nrCellsPerChid[drs_value_type]
    if(cell > NRCELLSPERCHID):
        print('ERROR: cell > '+str(NRCELLSPERCHID))
        return

    value_index = chid*NRCELLSPERCHID + cell

    factor_str = ''
    unit_str = r' $\mathrm{mV}$'
    for interval_nr in interval_array:
        groupname = 'Interval'+str(interval_nr)
        title_str = 'Intervall '+str(interval_nr)
        print(groupname)
        with h5py.File(interval_file_path, 'r') as interval_source:
            data = interval_source[groupname]
            interval_indices = np.array(data['IntervalIndices'])
        print('loading')
        with h5py.File(data_collection_path, 'r') as store:
            drs_value_var = store[drs_value_type+'Var'][interval_indices, value_index]
            drs_value_std = np.sqrt(drs_value_var)

        if(drs_value_type == 'Gain'):
            drs_value_std /= DACfactor/pow(10, 3)
            factor_str = r' $\cdot$'
            unit_str = r' $10^{-3}$'
        drs_value_std_mean = np.mean(drs_value_std)
        drs_value_std_std = np.std(drs_value_std)
        drs_value_std_max = max(drs_value_std)
        drs_value_std_min = min(drs_value_std)

        drs_value_std_limit = drs_value_std_mean*cut_off_error_factor

        n = len(drs_value_std)
        n_ = len(drs_value_std[drs_value_std > drs_value_std_limit])
        print(n_/n*100, '%')

        plot = plt.plot([], [])
        color = plot[0].get_color()
        label = (title_str+':' +
                 '\n'+r' $\overline{Hist}$: '+str(round(drs_value_std_mean, 2))+factor_str+unit_str +
                 '\n'+r' $\sigma_\mathrm{Hist}$: '+str(round(drs_value_std_std, 2))+factor_str+unit_str +
                 '\n'+r' $x_\mathrm{Max}$: '+str(round(drs_value_std_max, 2)))+factor_str+unit_str+'\n'
        weights = np.full(len(drs_value_std), 100/len(drs_value_std))
        nr_bins = int(abs((drs_value_std_max-drs_value_std_min))/abs(x_lim[1]-x_lim[0])*100)
        plt.hist(drs_value_std, bins=nr_bins, weights=weights,
                 histtype='step', label=label, color=color)

        plt.axvline(x=drs_value_std_limit, linewidth=2, ls='--', color=color)

    plt.xlabel(r'Standardabweichung /'+unit_str)
    plt.ylabel(r'Häufigkeit /$\mathrm{\%}$')
    plt.xlim(x_lim)
    plt.legend(loc='upper right')
    if(store_file_path is not None):
        plt.savefig(store_file_path)
    plt.show()
    plt.close()


###############################################################################
@click.command()
@click.argument('data_collection_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_0/dataCollection.h5',
                type=click.Path(exists=True))
@click.argument('interval_file_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_0/intervalIndices.h5',
                type=click.Path(exists=True))
@click.argument('fit_file_path_array',
                default=['/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                         'calibration/calculation/version_0/drsFitParameter_interval1.fits',
                         '/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                         'calibration/calculation/version_0/drsFitParameter_interval2.fits',
                         '/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                         'calibration/calculation/version_0/drsFitParameter_interval3.fits'],
                )
@click.argument('store_file_path',
                default='/home/fschulz/plots/version_0/drsValues/gain/chid1101_cell240_interval1_2_3.jpg',
                type=click.Path(exists=False))
@click.argument('interval_array',
                default=[1])
@click.option('--drs_value_type', '-type',
              default='Gain',
              type=click.Choice(['Baseline', 'Gain', 'ROIOffset']))
@click.argument('chid',
                default=1101)
@click.argument('cell',
                default=240)
###############################################################################
def drs_value_cell(data_collection_path, interval_file_path, fit_file_path_array,
                   store_file_path, interval_array, drs_value_type,
                   chid, cell):

    drs_value_cell_(data_collection_path, interval_file_path, fit_file_path_array,
                    store_file_path, interval_array, drs_value_type,
                    chid, cell)


###############################################################################
def drs_value_cell_(data_collection_path, interval_file_path, fit_file_path_array,
                    store_file_path, interval_array, drs_value_type,
                    chid, cell):

    NRCELLSPERCHID = config.nrCellsPerChid[drs_value_type]
    if(cell > NRCELLSPERCHID):
        print('ERROR: cell > '+str(NRCELLSPERCHID))
        return

    value_index = chid*NRCELLSPERCHID + cell

    # loading source data
    with h5py.File(data_collection_path, 'r') as store:
        time = np.array(store['Time'+drs_value_type]).flatten()

    ylabel_str = drs_value_type+r' /$\mathrm{mV}$'
    slope_unit_str = r'$\,\frac{\mathrm{mV}}{°C}$'
    offset_unit_str = r'$\,\mathrm{mV}$'
    if(drs_value_type == 'Gain'):
        ylabel_str = drs_value_type+r' /$10^{-3}$'
        slope_unit_str = r'$\,\frac{10^{-6}}{°C}$'
        offset_unit_str = r'$\times 10^{-3}$'

    use_mask = True
    mask_collection = []
    time_collection = []
    temp_collection = []
    drs_value_collection = []
    fit_value_collection = []
    for interval_nr in interval_array:
        groupname = 'Interval'+str(interval_nr)
        with h5py.File(interval_file_path, 'r') as interval_source:
            data = interval_source[groupname]
            interval_indices = np.array(data['IntervalIndices'])
            if(use_mask):
                mask = np.array(data[drs_value_type+'Mask'])
                mask_collection.append(mask)
        with h5py.File(data_collection_path, 'r') as store:
            temp = np.array(store['Temp'+drs_value_type][interval_indices, int(chid/9)])
            drs_value = np.array(store[drs_value_type][interval_indices, value_index])

        with fits.open(fit_file_path_array[np.where(np.array(interval_array) == interval_nr)[0][0]], ignoremissing=True, ignore_missing_end=True) as fit_value_tab:
            data = fit_value_tab['FitParameter'].data
            slope = data[drs_value_type+'Slope'][0][value_index]
            offset = data[drs_value_type+'Offset'][0][value_index]

        time_interval = time[interval_indices]
        time_collection.append(time_interval)
        temp_collection.append(temp)
        drs_value_collection.append(drs_value)
        fit_value_collection.append([slope, offset])

        if(drs_value_type == 'Gain'):
            drs_value /= DACfactor/pow(10, 3)

    temp_list = np.concatenate(temp_collection).ravel()
    drs_value_list = np.concatenate(drs_value_collection).ravel()
    time_list = np.concatenate(time_collection).ravel()

    sc_all = plt.scatter(temp_list, drs_value_list, c=time_list)
    plt.close()  # Dont show sc_all, just use it to create the colorbar

    fig, img = plt.subplots()

    intervalMonth = 6
    start_date = pd.to_datetime(time_list[0] * 24 * 3600 * 1e9).date()
    end_date = pd.to_datetime(time_list[-1] * 24 * 3600 * 1e9).date()
    time_label = pd.date_range(start=start_date, end=end_date, freq=str(intervalMonth)+'M')
    cbar = fig.colorbar(sc_all, ticks=mdates.MonthLocator(interval=intervalMonth))
    cbar.ax.set_yticklabels(time_label.strftime('%b %Y'))
    time_color = cbar.to_rgba(time_list)

    i_min, i_max = 0, 0
    temp_range = np.linspace(min(temp_list)-1, max(temp_list)+1, 10000)
    fit_plots = []
    for interval_index in range(len(interval_array)):
        i_max = i_max+len(temp_collection[interval_index])
        color = time_color[i_min: i_max]
        temp = temp_collection[interval_index]
        drs_value = drs_value_collection[interval_index]
        if(use_mask):
            mask_u = mask_collection[interval_index][:, value_index]
            mask_nu = np.logical_not(mask_u)
            img.scatter(temp[mask_u], drs_value[mask_u], marker='.', s=50, alpha=0.45,
                        c=color)
            img.scatter(temp[mask_nu], drs_value[mask_nu], marker='*', s=50, alpha=0.75,
                        c=color, edgecolors='k', linewidth=0.5)

            l_1 = img.scatter([], [], s=50, marker='.', c='r')
            l_2 = img.scatter([], [], s=50, marker='*', c='r', edgecolors='k')

            first_legend = plt.legend([l_1, l_2], ['Verwendete Werte', 'Nicht verwendete Werte'],
                                      loc='upper right', scatterpoints=1, numpoints=1)
            plt.gca().add_artist(first_legend)
        else:
            img.scatter(temp, drs_value, marker='.', s=50, alpha=0.45, c=color)
        i_min = i_min+i_max

        slope, offset = fit_value_collection[interval_index]
        fit = linearerFit(temp_range, slope, offset)

        if(drs_value_type == 'Gain'):
            fit *= pow(10, 3)
            slope *= pow(10, 6)
            offset *= pow(10, 3)

        color_mean = np.mean(color, axis=0)
        label_str = (r'm={:0.2f}'.format(slope) + slope_unit_str + ', ' +
                     r'b={:0.2f}'.format(offset) + offset_unit_str)
        fit_plot, = plt.plot(temp_range, fit, '-', color=color_mean,
                             label=label_str)
        fit_plots.append(fit_plot)
    plt.legend(handles=fit_plots, loc='lower center', title=r'Linearer Fit: f(x) = m $\cdot$ x + b')

    plt.xlabel(r'Temperatur /$\mathrm{^\circ C}$')
    plt.ylabel(ylabel_str)
    plt.xlim(min(temp_list)-1, max(temp_list)+1)
    ymin, ymax = get_best_limits(drs_value, 0.36)
    plt.ylim(ymin*0.9994, ymax)
    plt.grid()
    plt.tight_layout()
    if(store_file_path is not None):
        plt.savefig(store_file_path)
    plt.show()
    plt.close()


###############################################################################
@click.command()
@click.argument('drs_fit_file_path_array',
                default=['/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                         'calibration/calculation/version_0/drsFitParameter_interval1.fits',
                         '/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                         'calibration/calculation/version_0/drsFitParameter_interval2.fits',
                         '/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                         'calibration/calculation/version_0/drsFitParameter_interval3.fits'])
@click.argument('store_file_path',
                default='/home/fschulz/plots/version_0/model/gain/gain_slope_hist.png',
                type=click.Path(exists=False))
@click.option('--drs_value_type', '-type',
              default='Gain',
              type=click.Choice(['Baseline', 'Gain', 'ROIOffset']))
@click.option('--fit_parameter_type', '-type',
              default='Slope',
              type=click.Choice(['Slope', 'Offset']))
@click.argument('x_lim',
                default=[-1, 1])  # 0.75, b[-840, -980], 1, b[.85, .975],0.4, b[-5, 20]
###############################################################################
def drs_model_parameter_hist(drs_fit_file_path_array, store_file_path,
                             drs_value_type, fit_parameter_type,
                             x_lim):

    drs_model_parameter_hist_(drs_fit_file_path_array, store_file_path,
                              drs_value_type, fit_parameter_type,
                              x_lim)


###############################################################################
def drs_model_parameter_hist_(drs_fit_file_path_array, store_file_path,
                              drs_value_type, fit_parameter_type,
                              x_lim):

    NRCELLSPERCHID = config.nrCellsPerChid[drs_value_type]

    factor_str = ''
    unit_str_1 = r' $\mathrm{mV}$'
    unit_str_2 = unit_str_1
    for fit_file_path_array in drs_fit_file_path_array:
        fit_file = fits.open(fit_file_path_array, ignoremissing=True, ignore_missing_end=True)
        interval_nr = int(fit_file[0].header['INTNR'])

        useful_chids = get_useful_chids(interval_nr)

        fit_value = fit_file['FitParameter'].data[drs_value_type+fit_parameter_type][0]
        if(drs_value_type == 'Gain'):
            if(fit_parameter_type == 'Slope'):
                fit_value *= pow(10, 3)
                factor_str = r' $\cdot$'
                unit_str_1 = r' $10^{-3}$'
                unit_str_2 = unit_str_1
            else:
                unit_str_1 = ''
                unit_str_2 = '1'

        fit_value_ = fit_value.reshape(NRCHID, NRCELLSPERCHID)[useful_chids, :].flatten()

        fit_value_mean = np.mean(fit_value_)
        fit_value_std = np.std(fit_value_)
        fit_value_min = min(fit_value_)
        fit_value_max = max(fit_value_)

        plot = plt.plot([], [])
        color = plot[0].get_color()
        title_str = 'Intervall '+str(interval_nr)
        label = (title_str+':' +
                 '\n'+r' $\overline{Hist}$: '+str(round(fit_value_mean, 2))+factor_str+unit_str_1 +
                 '\n'+r' $\sigma_\mathrm{Hist}$: '+str(round(fit_value_std, 2))+factor_str+unit_str_1 +
                 '\n'+r' $x_\mathrm{Min}$: '+str(round(fit_value_min, 2))+factor_str+unit_str_1 +
                 '\n'+r' $x_\mathrm{Max}$: '+str(round(fit_value_max, 2))+factor_str+unit_str_1+'\n')
        weights = np.full(len(fit_value_), 100/len(fit_value_))
        nr_bins = int(abs((fit_value_max-fit_value_min))/abs(x_lim[1]-x_lim[0])*100)
        plt.hist(fit_value_, bins=nr_bins, weights=weights,
                 histtype='step', label=label, color=color)
        del fit_value_

        # abs(x_lim[1]-x_lim[0])
        # fit_value = fit_value.flatten()
        # fit_value_min = min(fit_value)
        # fit_value_max = max(fit_value)
        # weights = np.full(len(fit_value), 100/len(fit_value))
        # nr_bins = int(abs((fit_value_max-fit_value_min))/abs(x_lim[1]-x_lim[0])*100)
        # plt.hist(fit_value, bins=nr_bins, weights=weights,
        #          histtype='step', linestyle=('dashed'),
        #          color=color)
        # del fit_value

    x_label_str = 'Steigung a'
    if(fit_parameter_type == 'Offset'):
        x_label_str = 'y-Achsenabschnitt b'

    plt.xlabel(x_label_str+r' /'+unit_str_2)
    plt.ylabel(r'Häufigkeit /$\mathrm{\%}$')
    plt.xlim(x_lim)
    plt.legend(loc='upper right')
    plt.tight_layout()
    if(store_file_path is not None):
        plt.savefig(store_file_path)
    #plt.show()
    plt.close()


###############################################################################
@click.command()
@click.argument('fit_file_path_array',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_0/drsFitParameter_interval1.fits')
@click.argument('store_file_path',
                default='/home/fschulz/plots/version_0/model/gain/gain_offset_fact_cam.png',
                type=click.Path(exists=False))
@click.option('--drs_value_type', '-type',
              default='Gain',
              type=click.Choice(['Baseline', 'Gain', 'ROIOffset']))
@click.option('--fit_parameter_type', '-type',
              default='Offset',
              type=click.Choice(['Slope', 'Offset']))
@click.argument('worst_chids',
                default=[])
###############################################################################
def drs_model_parameter_fact_cam(fit_file_path_array, store_file_path,
                                 drs_value_type, fit_parameter_type,
                                 worst_chids):

    drs_model_parameter_fact_cam_(fit_file_path_array, store_file_path,
                                  drs_value_type, fit_parameter_type,
                                  worst_chids)


###############################################################################
def drs_model_parameter_fact_cam_(fit_file_path_array, store_file_path,
                                  drs_value_type, fit_parameter_type,
                                  worst_chids=[]):

    NRCELLSPERCHID = config.nrCellsPerChid[drs_value_type]

    fig, ax = plt.subplots(1, len(fit_file_path_array), figsize=(20, 10), dpi=100)

    for i in range(len(fit_file_path_array)):
        with fits.open(fit_file_path_array[i], ignoremissing=True, ignore_missing_end=True) as fit_file:
            interval_nr = int(fit_file[0].header['INTNR'])
            print('interval_nr: ', interval_nr)

            fit_value = fit_file['FitParameter'].data[drs_value_type+fit_parameter_type][0].reshape(NRCHID, NRCELLSPERCHID)

        unit_str = r'$\mathrm{mV}$'
        if(drs_value_type == 'Gain'):
            if(fit_parameter_type == 'Slope'):
                fit_value *= pow(10, 3)
                unit_str = r'$\times 10^{-3}$'

        not_useful_chids = get_not_useful_chids(interval_nr)

        fit_value_per_chid = np.mean(fit_value, axis=1)

        chid_list = np.linspace(0, NRCHID-1, NRCHID, dtype='int')

        bad_chids = np.unique(np.append(worst_chids, not_useful_chids)).astype('int')
        useful_chids = np.setdiff1d(chid_list, bad_chids)

        fit_value_per_chid[bad_chids] = np.max(fit_value_per_chid[useful_chids])

        plot = factPlots.camera(fit_value_per_chid, cmap='hot')
        factPlots.mark_pixel(not_useful_chids, color='b', linewidth=1)
        factPlots.mark_pixel(worst_chids, color='r', linewidth=1)
    #factPlots.pixelids(size=10)

        fig.colorbar(plot, ax=ax[i], label=r'Mittelwert des Betrags der Residuen eines Pixels/'+unit_str)
    plt.tight_layout()
    if(store_file_path is not None):
        plt.savefig(store_file_path)
    #plt.show()
    plt.close()


###############################################################################
@click.command()
@click.argument('drs_fit_file_path_array',
                default=['/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                         'calibration/calculation/version_0/drsFitParameter_interval1.fits',
                         '/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                         'calibration/calculation/version_0/drsFitParameter_interval2.fits',
                         '/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                         'calibration/calculation/version_0/drsFitParameter_interval3.fits'])
@click.argument('store_file_path',
                default='/home/fschulz/plots/version_0/model/roiOffset/roiOffset_residual_hist.png',
                type=click.Path(exists=False))
@click.option('--drs_value_type', '-type',
              default='ROIOffset',
              type=click.Choice(['Baseline', 'Gain', 'ROIOffset']))
###############################################################################
def drs_fit_value_residual_hist(drs_fit_file_path_array, store_file_path,
                                drs_value_type, x_lim):

    drs_fit_value_residual_hist_(drs_fit_file_path_array, store_file_path,
                                 drs_value_type, x_lim)


###############################################################################
def drs_fit_value_residual_hist_(drs_fit_file_path_array, store_file_path,
                                 drs_value_type, x_lim):

    NRCELLSPERCHID = config.nrCellsPerChid[drs_value_type]

    factor_str = ''
    unit_str = r' $\mathrm{mV}$'
    for fit_file_path_array in drs_fit_file_path_array:
        fit_file = fits.open(fit_file_path_array, ignoremissing=True, ignore_missing_end=True)
        interval_nr = int(fit_file[0].header['INTNR'])

        useful_chids = get_useful_chids(interval_nr)

        res_value = fit_file['FitParameter'].data[drs_value_type+'Residual'][0]
        if(drs_value_type == 'Gain'):
                res_value *= pow(10, 3)
                factor_str = r' $\cdot$'
                unit_str = r' $10^{-3}$'

        res_value_ = res_value.reshape(NRCHID, NRCELLSPERCHID)[useful_chids, :].flatten()

        res_value_mean = np.mean(res_value_)
        res_value_std = np.std(res_value_)
        res_value_min = 0
        res_value_max = max(res_value_)

        plot = plt.plot([], [])
        color = plot[0].get_color()
        title_str = 'Intervall '+str(interval_nr)
        label = (title_str+':' +
                 '\n'+r' $\overline{Hist}$: '+str(round(res_value_mean, 2))+factor_str+unit_str +
                 '\n'+r' $\sigma_\mathrm{Hist}$: '+str(round(res_value_std, 2))+factor_str+unit_str +
                 '\n'+r' $x_\mathrm{Max}$: '+str(round(res_value_max, 2))+factor_str+unit_str+'\n')
        weights = np.full(len(res_value_), 100/len(res_value_))
        nr_bins = int(abs((res_value_max-res_value_min))/abs(x_lim[1]-x_lim[0])*100)
        plt.hist(res_value_, bins=nr_bins, weights=weights,
                 histtype='step', label=label, color=color)
        del res_value_

        abs(x_lim[1]-x_lim[0])
        res_value = res_value.flatten()
        res_value_min = 0
        res_value_max = max(res_value)
        weights = np.full(len(res_value), 100/len(res_value))
        nr_bins = int(abs((res_value_max-res_value_min))/abs(x_lim[1]-x_lim[0])*100)
        plt.hist(res_value, bins=nr_bins, weights=weights,
                 histtype='step', linestyle=('dashed'),
                 color=color)
        del res_value

    plt.xlabel(r'Residuenmittelwert /'+unit_str)
    plt.ylabel(r'Häufigkeit /$\mathrm{\%}$')
    plt.xlim(x_lim)
    plt.legend(loc='upper right')
    plt.tight_layout()
    if(store_file_path is not None):
        plt.savefig(store_file_path)
    #plt.show()
    plt.close()


###############################################################################
@click.command()
@click.argument('fit_file_path_array',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_0/drsFitParameter_interval1.fits')
@click.argument('store_file_path',
                default='/home/fschulz/plots/version_0/model/gain/fact_cam_residual_int2.jpg',
                type=click.Path(exists=False))
@click.option('--drs_value_type', '-type',
              default='Baseline',
              type=click.Choice(['Baseline', 'Gain', 'ROIOffset']))
@click.argument('worst_chids',
                default=[])
###############################################################################
def residual_fact_cam(fit_file_path_array, store_file_path,
                      drs_value_type, worst_chids):

    residual_fact_cam_(fit_file_path_array, store_file_path,
                       drs_value_type, worst_chids)


###############################################################################
def residual_fact_cam_(fit_file_path_array, store_file_path,
                       drs_value_type, worst_chids=[]):

    NRCELLSPERCHID = config.nrCellsPerChid[drs_value_type]

    with fits.open(fit_file_path_array, ignoremissing=True, ignore_missing_end=True) as fit_file:
        interval_nr = int(fit_file[0].header['INTNR'])
        print('interval_nr: ', interval_nr)

        res_value = fit_file['FitParameter'].data[drs_value_type+'Residual'][0].reshape(NRCHID, NRCELLSPERCHID)

    unit_str = r'$\mathrm{mV}$'
    if(drs_value_type == 'Gain'):
        res_value /= pow(10, -3)
        unit_str = r'$\times 10^{-3}$'

    not_useful_chids = get_not_useful_chids(interval_nr)

    residual_per_chid = np.mean(res_value, axis=1)

    chid_list = np.linspace(0, NRCHID-1, NRCHID, dtype='int')
    #bad_chids = np.unique(np.append(worst_chids, not_useful_chids)).astype('int')
    #useful_chids = np.setdiff1d(chid_list, bad_chids)
    #residual_per_chid[bad_chids] = np.max(residual_per_chid[useful_chids])

    useful_chids = np.setdiff1d(chid_list, not_useful_chids)
    residual_per_chid[not_useful_chids] = np.max(residual_per_chid[useful_chids])

    plot = factPlots.camera(residual_per_chid, cmap='hot')
    factPlots.mark_pixel(worst_chids, color='b', linewidth=2)
    factPlots.mark_pixel(not_useful_chids, color='darkgreen', linewidth=2)
    #factPlots.pixelids(size=10)

    # max_all = max(res_value[np.setdiff1d(chid_list, not_useful_chids)].flatten())
    # mean_all = np.mean(res_value[np.setdiff1d(chid_list, not_useful_chids)].flatten())
    # std_all = np.std(res_value[np.setdiff1d(chid_list, not_useful_chids)].flatten())
    #
    # max_sel = max(res_value[useful_chids].flatten())
    # mean_sel = np.mean(res_value[useful_chids].flatten())
    # std_sel = np.std(res_value[useful_chids].flatten())

    # print(r'Mean: {:0.2f} $\pm$ {:0.1f} % , Max: {:0.2f}'.format(mean_all, (std_all/mean_all*100), max_all))
    # print(r'Mean: {:0.2f} $\pm$ {:0.1f} %, Max: {:0.2f}'.format(mean_sel, (std_sel/mean_sel*100), max_sel))
    # print('\SI{'+'{:0.2f}'.format(mean_all)+'}{\mV} $\pm$ \SI{'+'{:0.1f}'.format(std_all/mean_all*100)+'}{\percent}')
    # print('\SI{'+'{:0.2f}'.format(max_all)+'}{\mV}')
    # print('\SI{'+'{:0.2f}'.format(mean_sel)+'}{\mV} $\pm$ \SI{'+'{:0.1f}'.format(std_sel/mean_sel*100)+'}{\percent}')
    # print('\SI{'+'{:0.2f}'.format(max_sel)+'}{\mV}')
    plt.colorbar(plot, label=r'Mittelwert des Betrags der Residuen eines Pixels/'+unit_str)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.tight_layout()
    if(store_file_path is not None):
        plt.savefig(store_file_path)
    plt.show()
    plt.close()


###############################################################################
@click.command()
@click.argument('data_collection_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_0/dataCollection.h5',
                type=click.Path(exists=True))
@click.argument('interval_file_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_0/intervalIndices.h5',
                type=click.Path(exists=True))
@click.argument('fit_file_path_array',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_0/drsFitParameter_interval1.fits',
                type=click.Path(exists=True))
@click.argument('store_file_path',
                default=('/home/fschulz/plots/version_0/residual/' +
                         'gain/worst_100_cell_collection_I1.pdf'),
                type=click.Path(exists=False))
@click.argument('interval_nr',
                default=1)
@click.option('--drs_value_type', '-type',
              default='Baseline',
              type=click.Choice(['Baseline', 'Gain', 'ROIOffset']))
@click.argument('worst_chids',
                default=[])
@click.argument('selected_values',
                default=100)
###############################################################################
def worst_cell_collection_selected_by_residual(
                                    data_collection_path, interval_file_path,
                                    fit_file_path_array, store_file_path,
                                    interval_nr, drs_value_type,
                                    worst_chids, selected_values):

    worst_cell_collection_selected_by_residual_(
                                    data_collection_path, interval_file_path,
                                    fit_file_path_array, store_file_path,
                                    interval_nr, drs_value_type,
                                    worst_chids, selected_values)



###############################################################################
# ##############                  Helper Plots                 ############## #
###############################################################################
def worst_cell_collection_selected_by_residual_(
                                    data_collection_path, interval_file_path,
                                    fit_file_path_array, store_file_path,
                                    interval_nr, drs_value_type,
                                    worst_chids, selected_values=100):

    NRCELLSPERCHID = config.nrCellsPerChid[drs_value_type]

    groupname = 'Interval'+str(interval_nr)

    interval_source = h5py.File(interval_file_path, 'r')[groupname]
    # cut_off_error_factor = interval_source.attrs['CutOff'+drs_value_type]
    interval_indices = np.array(interval_source['IntervalIndices'])

    chid_list = np.linspace(0, NRCHID-1, NRCHID, dtype='int')
    not_useful_chids = np.array([non_standard_chids['crazy'],
                                 non_standard_chids['dead']]).flatten()

    not_useful_chids = np.unique(np.sort(np.append(not_useful_chids, worst_chids)))

    useful_chids = chid_list[np.setdiff1d(chid_list, not_useful_chids)]
    fit_value_tab = fits.open(fit_file_path_array, ignoremissing=True, ignore_missing_end=True)['FitParameter'].data

    residual_array = fit_value_tab[drs_value_type+'Residual'][0].reshape(NRCHID, NRCELLSPERCHID)[useful_chids, :].flatten()

    indices = np.argsort(residual_array)[-selected_values:]
    print(max(residual_array))

    with PdfPages(store_file_path) as pdf:
        for index in tqdm(reversed(indices)):
            residual = residual_array[index]
            print(residual, index, np.where(residual_array == max(residual_array)))
            return
            y_unit_str = r'$\mathrm{mV}$'
            unit_power_str = ''
            if(drs_value_type == 'Gain'):
                y_unit_str = r'$\mathrm{1}$'
                residual /= pow(10, -3)
                unit_power_str = r'$\times 10^{-3}$'

            chid = int(index / NRCELLSPERCHID)
            i = 0
            while (i < len(not_useful_chids)) and (chid+i >= not_useful_chids[i]):
                i += 1
            chid += i
            cell = index % NRCELLSPERCHID
            value_index = chid*NRCELLSPERCHID+cell

            #print('Chid: ', chid, 'Cell: ', cell, 'Residual: ', residual)
            mask = np.array(interval_source[drs_value_type+'Mask'][:, value_index])
            with h5py.File(data_collection_path, 'r') as store:
                time = np.array(store['Time'+drs_value_type][interval_indices, :]).flatten()
                temp = store['Temp'+drs_value_type][interval_indices, int(chid/9)]
                drs_value = store[drs_value_type][interval_indices, value_index]

            if(drs_value_type == 'Gain'):
                drs_value /= DACfactor

            sc_all = plt.scatter(temp, drs_value, c=np.array(time))
            plt.close()  # Dont show sc_all, just use it to create the colorbar

            fig, img = plt.subplots()

            intervalMonth = 3
            start_date = pd.to_datetime(time[0] * 24 * 3600 * 1e9).date()
            end_date = pd.to_datetime(time[-1] * 24 * 3600 * 1e9).date()
            time_label = pd.date_range(start=start_date, end=end_date, freq=str(intervalMonth)+'M')
            cbar = fig.colorbar(sc_all, ticks=mdates.MonthLocator(interval=intervalMonth))
            cbar.ax.set_yticklabels(time_label.strftime('%b %Y'))
            time_color = cbar.to_rgba(time)

            i_min, i_max = 0, len(temp)
            temp_range = np.linspace(min(temp)-1, max(temp)+1, 10000)
            color = time_color[i_min: i_max]
            mask_u = mask
            mask_nu = np.logical_not(mask_u)
            img.scatter(temp[mask_u], drs_value[mask_u], s=50, marker='+',
                        c=color, label='paternNoiseMean with averaged Temperature')
            img.scatter(temp[mask_nu], drs_value[mask_nu], s=50, marker='*',
                        c=color, label='paternNoiseMean with averaged Temperature')

            slope = fit_value_tab[drs_value_type+'Slope'][0][value_index]
            offset = fit_value_tab[drs_value_type+'Offset'][0][value_index]
            fit = linearerFit(temp_range, slope, offset)

            color_mean = np.mean(color, axis=0)
            fit_plot, = plt.plot(temp_range, fit, '-', color=color_mean)

            plt.title((drs_value_type+', absolute residual mean: '+str('{:0.1f}'.format(residual)) +
                      unit_power_str +
                      '\nChid: '+str(chid)+', Cell: '+str(cell)), fontsize=15, y=1.00)  # , fontsize=20, y=0.95

            plt.xlabel(r'Temperatur /$\mathrm{^\circ C}$')
            plt.ylabel(drs_value_type+' /'+y_unit_str)
            plt.xlim(min(temp)-1, max(temp)+1)
            plt.grid()
            plt.gca().ticklabel_format(useOffset=False)
            pdf.savefig()


###############################################################################
@click.command()
@click.argument('data_folder_path',
                default='/net/big-tank/POOL/projects/fact/' +
                        'drs4_calibration_data/calibration/validating/version_0/' +
                        'meanAndStd/interval3/',
                type=click.Path(exists=True))
@click.argument('store_file_path',
                default='/home/fschulz/plots/version_0/meanAndStd/interval3/' +
                        'calibratedPedestelDataDistribution_mean.png', #_2016-04-01
                type=click.Path(exists=False))
@click.option('--calibrated_type', '-type',
              default='Mean',
              type=click.Choice(['Mean', 'Std']))
@click.argument('interval_nr',
                default=3)
@click.argument('datetime_limits_string',
                default=['2015-05-26', '2017-10-01'])
###############################################################################
def pedestel_mean_or_std_vs_temp(data_folder_path, store_file_path,
                                 calibrated_type, interval_nr,
                                 datetime_limits_string):

    pedestel_mean_or_std_vs_temp_(data_folder_path, store_file_path,
                                  calibrated_type, interval_nr,
                                  datetime_limits_string)


###############################################################################
def pedestel_mean_or_std_vs_temp_(data_folder_path, store_file_path,
                                  calibrated_type, interval_nr,
                                  datetime_limits_string):

    in_pea = True
    noise_file_list = sorted([file for file in
                              os.listdir(data_folder_path)
                              if (file.startswith('pedestelStats') and
                                  file.endswith('.fits'))])

    useful_chids = get_useful_chids('Interval'+str(interval_nr))

    datetime_limits = [pd.to_datetime(datetime_limits_string[0]),
                       pd.to_datetime(datetime_limits_string[1])]

    temp_diff_list = []
    time_list = []
    drs_file_calibrated_mean_list = []
    drs_model_calibrated_mean_list = []
    for noise_file_path in tqdm(noise_file_list):
        datetime = pd.to_datetime(noise_file_path.split('_')[-1].split('.')[0])
        if(datetime < datetime_limits[0] or datetime > datetime_limits[1]):
            continue
        #print(data_folder_path+noise_file_path)
        with fits.open(data_folder_path+noise_file_path) as noise_tab:
            nr_runs = len(noise_tab[1].data["PedestelRunId"])
            temp_diff = noise_tab[1].data["TempDiff"]

            drs_file_calibrated_of_the_day = noise_tab[1].data["DRSCalibratedData_"+calibrated_type]
            drs_model_calibrated_of_the_day = noise_tab[1].data["DRSCalibratedData_Temp_"+calibrated_type]

        for run_index in range(nr_runs):
            drs_file_calibrated = np.array(drs_file_calibrated_of_the_day[run_index]).reshape(-1, NRCHID)[:, useful_chids].flatten()
            drs_model_calibrated = np.array(drs_model_calibrated_of_the_day[run_index]).reshape(-1, NRCHID)[:, useful_chids].flatten()

            # calculate the mean over all events(1000) and chids per run
            drs_file_calibrated_mean = np.mean(drs_file_calibrated)
            drs_model_calibrated_mean = np.mean(drs_model_calibrated)
            if(in_pea):
                drs_file_calibrated_mean = drs_file_calibrated_mean*PEAFACTOR
                drs_model_calibrated_mean = drs_model_calibrated_mean*PEAFACTOR

            if(temp_diff[run_index] > 5):
                print(noise_file_path, run_index+1)
                continue

            if (calibrated_type == "Std" and drs_file_calibrated_mean > 0.6):
                print(noise_file_path, run_index+1)
                continue

            time_list.append(datetime)
            temp_diff_list.append(temp_diff[run_index])
            drs_file_calibrated_mean_list.append(drs_file_calibrated_mean)
            drs_model_calibrated_mean_list.append(drs_model_calibrated_mean)

    drs_file_calibrated_collection_mean = np.mean(drs_file_calibrated_mean_list)
    drs_model_calibrated_collection_mean = np.mean(drs_model_calibrated_mean_list)

    drs_file_calibrated_collection_std = np.std(drs_file_calibrated_mean_list, dtype="float64", ddof=1)
    drs_model_calibrated_collection_std = np.std(drs_model_calibrated_mean_list, dtype="float64", ddof=1)

    unit_str = r'/$\mathrm{mV}$'
    calibrated_type_str = 'Mittelwert'
    if(calibrated_type == 'Std'):
        calibrated_type_str = 'Standartabweichung'

    if(in_pea):
        unit_str = r'$\mathrm{PEA}$'
    ylabel = calibrated_type_str+'/'+unit_str

    gs = gridspec.GridSpec(1, 11)
    fig = plt.figure(figsize=(12, 10), dpi=100)

    ax0 = plt.subplot(gs[0, 0:5])
    ax1 = plt.subplot(gs[0, 5:11], sharey=ax0)
    fig.subplots_adjust(wspace=0.5)
    ax1.tick_params(labelleft='off')

    label_str = ("Mittelwert: "+str(format(round(drs_file_calibrated_collection_mean, 3), '.3f'))+unit_str +
                 ",\nStandartabweichung: "+str(format(round(drs_file_calibrated_collection_std, 3), '.3f')))+unit_str
    sc = ax0.scatter(time_list, drs_file_calibrated_mean_list,
                     s=50, marker=".", c=temp_diff_list, label=label_str)
    label_str = ("Mittelwert: "+str(format(round(drs_model_calibrated_collection_mean, 3), '.3f'))+unit_str +
                 ",\nStandartabweichung: "+str(format(round(drs_model_calibrated_collection_std, 3), '.3f')))+unit_str
    sc = ax1.scatter(time_list, drs_model_calibrated_mean_list,
                     s=50, marker=".", c=temp_diff_list, label=label_str)

    cbar = fig.colorbar(sc)
    cbar.set_label(r'|Pedestal-Run Temperatur - DRS-Run Temperatur| /$\mathrm{C\degree}$')
    ax0.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax0.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
    ax0.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax0.fmt_xdata = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax1.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.fmt_xdata = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
    plt.gcf().autofmt_xdate()

    ax0.set_title('DRS-File', fontsize=16, y=0.92)
    ax0.text(0.6, 0.15, "DRS-File")
    ax0.yaxis.grid(linestyle=':')
    ax0.set_ylabel(ylabel)
    ax0.legend(loc='lower left', numpoints=1, title="")
    ax1.set_title('Model', fontsize=16, y=0.92)
    ax1.yaxis.grid(linestyle=':')
    ax1.legend(loc='lower left', numpoints=1, title="")
    plt.tight_layout()
    if(store_file_path is not None):
        plt.savefig(store_file_path)
    #plt.show()
    plt.close()


###############################################################################
@click.command()
@click.argument('fit_file_path_array',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/drsFitParameter.fits',
                type=click.Path(exists=True))
@click.argument('store_file_path',
                default='/home/fschulz/plots/baselineSlopeChid_500.jpg',
                type=click.Path(exists=False))
@click.argument('interval_array',
                default=[2])
@click.option('--drs_value_type', '-type',
              default='ROIOffset',
              type=click.Choice(['Baseline', 'Gain', 'ROIOffset']))
@click.option('--fit_parameter_type', '-fitType',
              default='Slope',
              type=click.Choice(['Slope', 'Offset']))
@click.argument('chid',
                default=500)
###############################################################################
def drs_value_chid_fit_parameter(fit_file_path_array, store_file_path, interval_array,
                                 drs_value_type, fit_parameter_type, chid):

    for interval_nr in interval_array:
        groupname = 'Interval'+str(interval_nr)
        NRCELLSPERCHID = config.nrCellsPerChid[drs_value_type]
        with fits.open(fit_file_path_array, ignoremissing=True, ignore_missing_end=True) as fit_value_tab:
            data = fit_value_tab[groupname].data
            fit_parameter = data[drs_value_type+fit_parameter_type][0][chid*NRCELLSPERCHID:(chid+1)*NRCELLSPERCHID]

    colors = hot(np.linspace(0, 0, NRCELLSPERCHID))
    for i in range(33):
        colors[i*32-1] = [1., 0., 0., 1.]

    cell = np.linspace(0, NRCELLSPERCHID-1, NRCELLSPERCHID)
    plt.scatter(cell, fit_parameter, s=50, marker='+', color=colors)

    labelStr = ''
    if(fit_parameter_type == 'Offset'):
        labelStr = fit_parameter_type+r' /$\mathrm{mV}$'
    elif(fit_parameter_type == 'Slope'):
        labelStr = fit_parameter_type+r' /$\frac{\mathrm{mV}}{\mathrm{^\circ C}}$'

    plt.title((drs_value_type+' '+fit_parameter_type+' CHID:'+str(chid)+'\n' +
               r' Fit $f(x) = m \cdot x + b$'), fontsize=16, y=1.00)
    plt.xlabel('Cell [1]')
    plt.ylabel(labelStr)
    plt.xlim(-1, NRCELLSPERCHID)
    plt.grid()
    # plt.legend(loc='upper right', scatterpoints=1, numpoints=1)
    plt.savefig(store_file_path)
    plt.show()
    plt.close()


@click.command()
@click.argument('data_collection_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/dataCollection.h5',
                type=click.Path(exists=True))
@click.argument('interval_file_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/intervalIndices.h5',
                type=click.Path(exists=True))
@click.argument('store_file_path',
                default='/home/fschulz/plots/chid700_cell0_gain.jpg',
                type=click.Path(exists=False))
@click.argument('interval_array',
                default=[2])
@click.argument('drs_value_type',
                default='Gain')
# @click.option('--drs_value_type', '-vt',
#               default='Baseline',
#               type=click.Choice(['Baseline', 'Gain']))
@click.argument('chid',
                default=700)
@click.argument('cell',
                default=0)
# @click.option('--show_var_dev', '-var',
#               is_flag=False)
###############################################################################
def chid_cell_drs_values_time(data_collection_path, interval_file_path,
                              store_file_path, interval_array,
                              drs_value_type, chid, cell):

    value_index = chid*NRCELL + cell

    # Cecking wether the intervalIndices and the fitvalues are based on the given drsData
    with h5py.File(data_collection_path, 'r') as data_source:
        source_creation_date = data_source.attrs['CreationDate']

    with h5py.File(interval_file_path) as interval_source:
        used_source_creation_date_i = interval_source.attrs['SCDate']

    if(source_creation_date != used_source_creation_date_i):
        error_str = ("'interval_file_path' is not based on the given 'source_file_path'")
        print(error_str)
        return

    with h5py.File(data_collection_path, 'r') as store:
        time = np.array(store['Time'+drs_value_type]).flatten()

    use_mask = True
    in_PEA = False
    mask_collection = []
    time_collection = []
    temp_collection = []
    drs_value_collection = []
    drs_value_var_collection = []
    for interval_nr in interval_array:
        groupname = 'Interval'+str(interval_nr)
        with h5py.File(interval_file_path, 'r') as interval_source:
            data = interval_source[groupname]
            cut_off_error_factor = interval_source.attrs['CutOff'+drs_value_type]
            interval_indices = np.array(data['IntervalIndices'])
            if(use_mask):
                mask = np.array(data[drs_value_type+'Mask'][:, value_index])
                mask_collection.append(mask)
        with h5py.File(data_collection_path, 'r') as store:
            temp = np.array(store['Temp'+drs_value_type][interval_indices, int(chid/9)])
            drs_value = np.array(store[drs_value_type][interval_indices, value_index])
            drs_value_var = np.array(store[drs_value_type+'Std'][interval_indices, value_index])

        time_interval = pd.to_datetime(time[interval_indices] * 24 * 3600 * 1e9)
        time_collection.append(time_interval)
        temp_collection.append(temp)
        drs_value_collection.append(drs_value)
        drs_value_var_collection.append(drs_value_var)

        single_photon_limit = 2.1  # mV
        ylabel_str = drs_value_type+r' /$\mathrm{mV}$'
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
        drs_value_var = drs_value_var_collection[interval_index]
        time = time_collection[interval_index]
        temp = temp_collection[interval_index]

        mask0 = np.where(drs_value_var == 0.)[0]
        mask2 = np.where(drs_value_var > np.mean(drs_value_var)*2.0)[0]
        # mask2 = np.where(drs_value_var < np.mean(drs_value_var)*0.4)[0]
        #ylabel_str = r'Temperatur /$\mathrm{^\circ C}$'
        plt.errorbar(time, drs_value, yerr=None, color='k', marker='*', ls='')
        plt.errorbar(time[mask], drs_value[mask], yerr=None, color='g', marker='*', ls='')
        #plt.errorbar(time[mask], drs_value[mask], yerr=None, color='r', marker='*', ls='')
        #plt.errorbar(time[mask2], drs_value[mask2], yerr=None, color='b', marker='*', ls='')
    plt.title(drs_value_type+'Mean\nChid: '+str(chid)+', Cell: '+str(cell) +  # , fontsize=20, y=0.95
              ', ErrFactor: '+str('{:0.1f}'.format(cut_off_error_factor)), fontsize=15, y=1.02)

    plt.xlabel(r'Time')
    plt.ylabel(ylabel_str)
    # plt.xlim(min(temp_list)-1, max(temp_list)+1)
    plt.grid()
    #plt.gca().ticklabel_format(useOffset=False)
    #time_label = pd.date_range(start=start_date_, end=end_date_, freq='M') - pd.offsets.MonthBegin(1)
    #plt.xticks(time_label, time_label, rotation=30)
    #plt.gca().xaxis.set_major_formatter(time.DateFormatter('%d.%m.%y'))
    plt.text(0.02, 0.21, 'preliminary', fontdict=font, transform=plt.gca().transAxes)
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
            drs_file_calibrated = noise_tab[1].data['DrsCalibratedDataNoise']
            drs_model_calibrated = noise_tab[1].data['DrsCalibratedDataNoiseTemp']

    chid_list = np.linspace(0, NRCHID-1, NRCHID, dtype='int')
    useful_chid = chid_list[np.setdiff1d(chid_list,
                                         np.array([
                                            non_standard_chids['crazy'],
                                            non_standard_chids['dead']])-1)]

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

    drs_file_calibrated_var = np.var(drs_file_calibrated, dtype='float64', ddof=1)
    drs_model_calibrated_var = np.var(drs_model_calibrated, dtype='float64', ddof=1)

    gs = gridspec.GridSpec(4, 1)
    #ax0 = plt.subplot(gs[0:3, :])
    ax0 = plt.subplot(gs[0:4, :])
    plt.title(title_str, y=1.0)
    hist1 = ax0.hist(drs_file_calibrated, weights=weights, bins=nr_bins, histtype='step',
                     range=(0.0, xlim), lw=1, edgecolor='r', label='Drs-File Noise\nMean: '+str(format(round(drs_file_calibrated_mean, 3), '.3f'))+', Std: '+str(format(round(drs_file_calibrated_var, 3), '.3f')))
    hist2 = ax0.hist(drs_model_calibrated, weights=weights, bins=nr_bins, histtype='step',
                     range=(0.0, xlim), lw=1, edgecolor='g', label='Model Noise\nMean: '+str(format(round(drs_model_calibrated_mean, 3), '.3f'))+', Std: '+str(format(round(drs_model_calibrated_var, 3), '.3f')))

    plt.ylabel(r'frequency /$\mathrm{\%}$')
    plt.legend(loc='upper right', numpoints=1, title='')

    #ax1 = plt.subplot(gs[3, :])
    #ax1.step(hist1[1][0:-1], hist2[0]-hist1[0], 'g')
    #ax1.step(hist1[1][0:-1], hist3[0]-hist1[0], 'b')
    plt.xlabel(xlabel)
    plt.ylabel(r'$\Delta$ frequency /$\mathrm{\%}$')
    #plt.setp(ax0.get_xticklabels(), visible=False)
    plt.text(0.7, 0.15, 'preliminary', fontdict=font, transform=plt.gca().transAxes)
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
            drs_file_calibrated = noise_tab[1].data['DrsCalibratedDataNoise'].flatten()
            drs_model_calibrated = noise_tab[1].data['DrsCalibratedDataNoiseTemp'].flatten()

    drs_file_calibrated_chid_mean = np.mean(np.array(drs_file_calibrated).reshape(-1, NRCHID), axis=0)
    drs_model_calibrated_chid_mean = np.mean(np.array(drs_model_calibrated).reshape(-1, NRCHID), axis=0)

    non_standard_chids_indices = np.array([non_standard_chids['crazy'], non_standard_chids['dead']]).flatten()-1
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

    camera_plot_f = factPlots.camera(drs_file_calibrated_chid_mean,
                           vmin=vmin, vmax=vmax, cmap='hot')

    ax0g = plt.subplot(gs[0:9, 5:11])
    ax0g.set_axis_off()
    ax0g.set_xlabel('')
    ax0g.set_ylabel('')
    ax0g.set_title('model calibrated', fontsize=18)

    camera_plot_m = factPlots.camera(drs_model_calibrated_chid_mean,
                           vmin=vmin, vmax=vmax, cmap='hot')

    cbar = fig.colorbar(camera_plot_m, ax=ax0g)
    cbar.set_label(r'Noise /$\mathrm{mV}$')
    if(store_file_path is not None):
        plt.savefig(store_file_path+'.jpg')
    plt.show()
    plt.close()


# TODO update
@click.command()
@click.argument('filename',
                default='/net/big-tank/POOL/projects/fact/' +
                        'drs4_calibration_data/calibration/validating/noise/' +
                        'pedestelNoise_20140911.fits',
                type=click.Path(exists=True))
@click.argument('save',
                default=True,
                type=click.BOOL)
###############################################################################
def pedestial_noise(filename, save):

    with fits.open(filename) as noise_tab:
        date = pd.to_datetime(noise_tab[1].header['Date']).date()
        temp_diff = noise_tab[1].data['TempDiff']
        run_ids = noise_tab[1].data['PedestelRunId']
        drs_file_noise = noise_tab[1].data['DrsCalibratedDataNoise']
        drs_model_noise = noise_tab[1].data['DrsCalibratedDataNoiseTemp']

    print('date ', date)
    print('run_ids ', run_ids)
    print('temp_diff ', temp_diff)
    print('drs_file_noise ', np.array(drs_file_noise[0]).shape)
    print('drs_model_noise ', np.array(drs_model_noise[0]).shape)
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

    print('date ', date)
    print('run_ids ', run_ids)
    print('temp_diff ', temp_diff)
    print('drs_file_noise ', np.array(drs_file_noise[0]).shape)
    print('drs_model_noise ', np.array(drs_model_noise[0]).shape)

    dateStr = date.strftime('%Y-%m-%d')
    dateStr2 = date.strftime('%Y%m%d')

    for i in range(len(drs_file_noise)):
        store_filename = ('/home/fschulz/plots/noise/pedestelNoise' +
                          dateStr+'_runId'+str(run_ids[i])+'.jpg')
        #title_str = ('Standard deviation '+dateStr+'\n ' +
        #            'runID: '+str(run_ids[i])+', Temperature difference '+str(round(temp_diff[i], 3))+r'$^\circ C$')
        title_str = ('Temperature difference '+str(round(temp_diff[i], 3))+r'$^\circ C$')
        noise(drs_file_noise[i], drs_model_noise[i], title_str, store_filename)
        store_filename = ('/home/fschulz/plots/noise/pedestelnoise_fact_cam' +
                          dateStr+'_runId'+str(int(run_ids[i])))
        noise_fact_cam(drs_file_noise[i], drs_model_noise[i], store_filename)


@click.command()
@click.argument('folder',
                default='/net/big-tank/POOL/projects/fact/' +
                        'drs4_calibration_data/calibration/validating/noise/',
                type=click.Path(exists=True))
@click.argument('save',
                default=True,
                type=click.BOOL)
###############################################################################
def noise_mean_hist(folder, save):
    print('hist')
    store_file_path = '/home/fschulz/plots/noise/pedestelNoiseDistribution.jpg'
    noise_file_list = sorted([file for file in
                              os.livarir(folder)
                              if (file.startswith('pedestelNoise_') and
                                  file.endswith('.fits'))])

    chid_list = np.linspace(0, NRCHID-1, NRCHID, dtype='int')
    useful_chid = chid_list[np.setdiff1d(chid_list,
                                         np.array([
                                            non_standard_chids['crazy'],
                                            non_standard_chids['dead']])-1)]
    drs_file_calibrated_mean_list = []
    drs_model_calibrated_mean_list = []
    for noise_file_path in tqdm(noise_file_list):
        with fits.open(folder+noise_file_path) as noise_tab:
            nr_runs = len(noise_tab[1].data['PedestelRunId'])
            drs_file_calibrated_of_the_day = noise_tab[1].data['DrsCalibratedDataNoise']
            drs_model_calibrated_of_the_day = noise_tab[1].data['DrsCalibratedDataNoiseTemp']

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

    drs_file_calibrated_collection_var = np.var(drs_file_calibrated_mean_list, dtype='float64', ddof=1)
    drs_model_calibrated_collection_var = np.var(drs_model_calibrated_mean_list, dtype='float64', ddof=1)

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

    label_str = 'Drs-File Noise\nMean: '+str(format(round(drs_file_calibrated_collection_mean, 3), '.3f'))+', Std: '+str(format(round(drs_file_calibrated_collection_var, 3), '.3f'))
    plt.hist(drs_file_calibrated_mean_list, weights=weights, bins=nr_bins, histtype='step',
             range=(hist_range[0], hist_range[1]), lw=1, edgecolor='r', label=label_str)
    label_str = 'Model Noise\nMean: '+str(format(round(drs_model_calibrated_collection_mean, 3), '.3f'))+', Std: '+str(format(round(drs_model_calibrated_collection_var, 3), '.3f'))
    plt.hist(drs_model_calibrated_mean_list, weights=weights, bins=nr_bins, histtype='step',
             range=(hist_range[0], hist_range[1]), lw=1, edgecolor='g', label=label_str)

    plt.xlabel(xlabel)
    plt.ylabel(r'frequency /$\mathrm{\%}$')
    plt.legend(loc='upper right', numpoints=1, title='')
    plt.text(0.7, 0.15, 'preliminary', fontdict=font, transform=plt.gca().transAxes)
    if(store_file_path is not None):
        plt.savefig(store_file_path)
    plt.show()
    plt.close()




    @click.command()
    @click.argument('folder',
                    default='/net/big-tank/POOL/projects/fact/' +
                            'drs4_calibration_data/calibration/validating/noiseAndMean/',
                    type=click.Path(exists=True))
    @click.argument('store_file_path',
                    default='/home/fschulz/plots/noiseAndMean/' +
                            'calibratedPedestelDataDistribution_newBaselineNoise.jpg',
                            # CalibratedStdDev_Interval3 CalibratedBaseline, CalibratedStdDev
                    type=click.Path(exists=False))
    @click.option('--calibrated_type', '-type',
                  default='Noise',
                  type=click.Choice(['Mean', 'Noise']))
    @click.argument('datetime_limits_string',
                    default=['2015-05-26 12', '2017-10-01 12'])#['2014-05-20 12', '2015-05-26 12'])
    ###############################################################################
    def noise_mean_vs_temp(folder, store_file_path, calibrated_type, datetime_limits_string):
        print('noise_mean_vs_temp')

        noise_file_list = sorted([file for file in
                                  os.livarir(folder)
                                  if (file.startswith('pedestelNoise') and
                                      file.endswith('.fits'))])

        chid_list = np.linspace(0, NRCHID-1, NRCHID, dtype='int')
        useful_chid = chid_list[np.setdiff1d(chid_list,
                                             np.array([
                                                       non_standard_chids['crazy'],
                                                       non_standard_chids['dead']])-1)]

        datetime_limits = [pd.to_datetime(datetime_limits_string[0]),
                           pd.to_datetime(datetime_limits_string[1])]

        temp_diff_list = []
        time_list = []
        drs_file_calibrated_mean_list = []
        drs_model_calibrated_mean_list = []
        for noise_file_path in tqdm(noise_file_list):
            datetime = pd.to_datetime(noise_file_path.split('_')[-1].split('.')[0])
            if(datetime < datetime_limits[0] or datetime > datetime_limits[1]):
                continue
            #print(folder+noise_file_path)
            with fits.open(folder+noise_file_path) as noise_tab:
                nr_runs = len(noise_tab[1].data['PedestelRunId'])
                temp_diff = noise_tab[1].data['TempDiff']

                drs_file_calibrated_of_the_day = noise_tab[1].data['DRSCalibratedData_'+calibrated_type]
                drs_model_calibrated_of_the_day = noise_tab[1].data['DRSCalibratedData_Temp_'+calibrated_type]

            for run_index in range(nr_runs):
                drs_file_calibrated = np.array(drs_file_calibrated_of_the_day[run_index]).reshape(-1, NRCHID)[:, useful_chid].flatten()
                drs_model_calibrated = np.array(drs_model_calibrated_of_the_day[run_index]).reshape(-1, NRCHID)[:, useful_chid].flatten()

                # calculate the mean over all events(1000) and chids per run
                drs_file_calibrated_mean = np.mean(drs_file_calibrated)
                drs_model_calibrated_mean = np.mean(drs_model_calibrated)

                in_pea = True
                if(in_pea):
                    drs_file_calibrated_mean = drs_file_calibrated_mean*PEAFACTOR
                    drs_model_calibrated_mean = drs_model_calibrated_mean*PEAFACTOR

                if(temp_diff[run_index] > 5):
                    print(noise_file_path, run_index+1)
                    continue

                time_list.append(datetime.value)
                temp_diff_list.append(temp_diff[run_index])
                drs_file_calibrated_mean_list.append(drs_file_calibrated_mean)
                drs_model_calibrated_mean_list.append(drs_model_calibrated_mean)

        drs_file_calibrated_collection_mean = np.mean(drs_file_calibrated_mean_list)
        drs_model_calibrated_collection_mean = np.mean(drs_model_calibrated_mean_list)

        drs_file_calibrated_collection_var = np.var(drs_file_calibrated_mean_list, dtype='float64', ddof=1)
        drs_model_calibrated_collection_var = np.var(drs_model_calibrated_mean_list, dtype='float64', ddof=1)

        ylabel = calibrated_type+r'/$\mathrm{mV}$'
        if(in_pea):
            ylabel = calibrated_type+r'/$\mathrm{PEA}$'

        fig, img = plt.subplots()

        label_str = ('Drs-File '+calibrated_type+'\nMean: '+str(format(round(drs_file_calibrated_collection_mean, 3), '.3f')) +
                     ', Std: '+str(format(round(drs_file_calibrated_collection_var, 3), '.3f')))
        sc_f = plt.scatter(temp_diff_list, drs_file_calibrated_mean_list,
                           s=50, marker='+', c=time_list, label=label_str)
        label_str = ('Model '+calibrated_type+'\nMean: '+str(format(round(drs_model_calibrated_collection_mean, 3), '.3f')) +
                     ', Std: '+str(format(round(drs_model_calibrated_collection_var, 3), '.3f')))
        plt.scatter(temp_diff_list, drs_model_calibrated_mean_list,
                    s=50, marker='*', c=time_list, label=label_str)

        intervalMonth = 1
        start_date = pd.to_datetime(time_list[0]).date()
        end_date = pd.to_datetime(time_list[-1]).date()
        time_label = pd.date_range(start=start_date, end=end_date, freq=str(intervalMonth)+'M')
        cbar = fig.colorbar(sc_f, ticks=mdates.MonthLocator(interval=intervalMonth))
        cbar.ax.set_yticklabels(time_label.strftime('%b %Y'))
        # time_color = cbar.to_rgba(time_list)

        plt.title(r'Calibrated PedestelRun '+calibrated_type, fontsize=16, y=1.02)
        plt.xlabel(r'Temperatur /$\mathrm{C\degree}$')
        plt.ylabel(ylabel)
        plt.legend(loc='upper right', numpoints=1, title='')
        plt.text(0.6, 0.15, 'preliminary', fontdict=font, transform=plt.gca().transAxes)
        if(store_file_path is not None):
            plt.savefig(store_file_path)
        plt.show()
        plt.close()
