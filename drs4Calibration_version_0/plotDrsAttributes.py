import matplotlib.pyplot as plt
import matplotlib.dates as dates

import sys

import os
import numpy as np
import pandas as pd
import math
import h5py
import click
# import yaml

from tqdm import tqdm
from fact.instrument.camera import non_standard_pixel_chids as non_standard_chids
import fact.plotting as factPlots
from astropy.io import fits
from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import hot

import config as config
from constants import NRCHID, NRCELL, NRPATCH, PEAFACTOR


###############################################################################
# ##############                    Helper                     ############## #
###############################################################################
def check_file_match(drs_file_path,
                     interval_file_path=None, fit_file_path=None,
                     residual_file_path=None, chi2_file_path=None):

    match_flag = True
    with h5py.File(drs_file_path, 'r') as data_source:
        source_creation_date = data_source.attrs['CreationDate']

    if (interval_file_path is not None):
        with h5py.File(interval_file_path) as interval_source:
            used_source_creation_date_i = interval_source.attrs['SCDate']
        if(source_creation_date != used_source_creation_date_i):
            print("'interval_file_path' is not based on the given 'source_file_path'")
            match_flag = False

    if (fit_file_path is not None):
        with fits.open(fit_file_path, ignoremissing=True, ignore_missing_end=True) as fit_values_tab:
            used_source_creation_date_f = fit_values_tab[0].header['SCDate']
        if(source_creation_date != used_source_creation_date_f):
            print("'fit_file_path' is not based on the given 'source_file_path'")
            match_flag = False

    if (residual_file_path is not None):
        with h5py.File(residual_file_path) as residual_source:
            used_source_creation_date_r = residual_source.attrs['SCDate']
        if(source_creation_date != used_source_creation_date_r):
            print("'residual_file_path' is not based on the given 'source_file_path'")
            match_flag = False

    if (chi2_file_path is not None):
        with h5py.File(chi2_file_path) as residual_source:
            used_source_creation_date_c = residual_source.attrs['SCDate']
        if(source_creation_date != used_source_creation_date_c):
            print("'chi2_file_path' is not based on the given 'source_file_path'")
            match_flag = False

    if(not match_flag):
        sys.exit()


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
# ##############               Drs-Value  Plots                ############## #
###############################################################################

@click.command()
@click.argument('drs_file_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_0/drsData.h5',
                type=click.Path(exists=True))
@click.argument('interval_file_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_0/intervalIndices.h5',
                type=click.Path(exists=True))
@click.argument('store_file_path',
                default='/home/fschulz/plots/version_0/drsValues/gain/std_hist.jpg',
                type=click.Path(exists=False))
@click.argument('interval_array',
                default=[1, 2, 3])
@click.option('--drs_value_type', '-type',
              default='Gain',
              type=click.Choice(['Baseline', 'Gain', 'ROIOffset']))
@click.argument('nr_bins',
                default=100)
###############################################################################
def drs_value_std_hist(drs_file_path, interval_file_path,
                       store_file_path, interval_array,
                       drs_value_type, nr_bins):
    drs_value_std_hist_(drs_file_path, interval_file_path,
                        store_file_path, interval_array,
                        drs_value_type, nr_bins)


###############################################################################
def drs_value_std_hist_(drs_file_path, interval_file_path,
                        store_file_path, interval_array,
                        drs_value_type, nr_bins=100):
    # Cecking wether the intervalIndices are based on the given drsData
    check_file_match(drs_file_path, interval_file_path=interval_file_path)

    NRCELLSPERCHID = config.nrCellsPerChid[drs_value_type]

    chid_list = np.linspace(0, NRCHID-1, NRCHID, dtype='int')
    not_useful_chids = np.array([non_standard_chids['crazy'],
                                 non_standard_chids['dead']]).flatten()

    useful_chids = chid_list[np.setdiff1d(chid_list, not_useful_chids)]

    upper_limit = 10
    for interval_nr in interval_array:
        groupname = 'Interval'+str(interval_nr)
        print(groupname)
        with h5py.File(interval_file_path, 'r') as interval_source:
            data = interval_source[groupname]
            interval_indices = np.array(data['IntervalIndices'])
        print('loading')
        with h5py.File(drs_file_path, 'r') as store:
            drs_value_var = store[drs_value_type+'Var'][interval_indices, :]
            drs_value_var = drs_value_var.reshape(-1, NRCHID, NRCELLSPERCHID)[:, useful_chids, :].flatten()
            drs_value_std = np.sqrt(drs_value_var)
        drs_value_std_mean = np.mean(drs_value_std)
        drs_value_std_std = np.std(drs_value_std)
        drs_value_std_max = max(drs_value_std)

        label = (groupname+':' +
                 '\n Hist:' +
                 '\n   mean: '+str(round(drs_value_std_mean, 2))+r' $\mathrm{mV}$' +
                 '\n   std dev: '+str(round(drs_value_std_std, 2))+r' $\mathrm{mV}$' +
                 '\n max value: '+str(round(drs_value_std_max, 2)))+r' $\mathrm{mV}$'
        weights = np.full(len(drs_value_std), 100/len(drs_value_std))
        plt.hist(drs_value_std, bins=nr_bins, weights=weights,
                 range=(0, upper_limit), histtype='step', label=label)

    plt.title(r'Histogram of '+drs_value_type+' Standart deviation', fontsize=16, y=1.01)
    plt.xlabel(r'Standart deviation /$\mathrm{mV}$')
    plt.ylabel(r'Frequency /$\mathrm{\%}$')
    plt.xlim(0)
    plt.legend(loc='upper right')
    if(store_file_path is not None):
        plt.savefig(store_file_path)
    # plt.show()
    plt.close()


###############################################################################
@click.command()
@click.argument('fit_file_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_0/drsFitParameter_.fits',
                type=click.Path(exists=True))
@click.argument('store_file_path',
                default='/home/fschulz/plots/version_0/residual/gain/fact_cam_residual_int2.jpg',
                type=click.Path(exists=False))
@click.argument('interval_nr',
                default=2)
@click.option('--drs_value_type', '-type',
              default='Gain',
              type=click.Choice(['Baseline', 'Gain', 'ROIOffset']))
@click.argument('worse_chids',
                default=[])
###############################################################################
def residual_fact_cam(fit_file_path, store_file_path,
                      interval_nr, drs_value_type,
                      worse_chids):
    residual_fact_cam_(fit_file_path, store_file_path,
                       interval_nr, drs_value_type,
                       worse_chids)


###############################################################################
def residual_fact_cam_(fit_file_path, store_file_path,
                       interval_nr, drs_value_type,
                       worse_chids=[]):

    NRCELLSPERCHID = config.nrCellsPerChid[drs_value_type]

    groupname = 'Interval'+str(interval_nr)
    with fits.open(fit_file_path, ignoremissing=True, ignore_missing_end=True) as fit_value_tab:
        residual = fit_value_tab[groupname].data[drs_value_type+'Residual'][0].reshape(NRCHID, NRCELLSPERCHID)

    unit_str = r'$\mathrm{mV}$'
    if(drs_value_type == 'Gain'):
        residual /= pow(10, -3)
        unit_str = r'$\times 10^{-3}$'

    residual_per_chid = np.mean(residual, axis=1)

    not_useful_chids = np.array([non_standard_chids['crazy'],
                                 non_standard_chids['dead']]).flatten()

    residual_per_chid[worse_chids] = 0
    residual_per_chid[worse_chids] = np.max(residual_per_chid)

    plt.title(drs_value_type+'-residual mean per Pixel\n Interval: {}'.format(interval_nr), fontsize=16, y=1.01)
    plot = factPlots.camera(residual_per_chid, cmap='hot')
    factPlots.mark_pixel(not_useful_chids, color='b', linewidth=1)
    factPlots.mark_pixel(worse_chids, color='r', linewidth=1)
    # factPlots.pixelids(size=10)

    plt.colorbar(plot, label=r'Residual mean per pixel /'+unit_str)
    if(store_file_path is not None):
        plt.savefig(store_file_path)
    # plt.show()
    plt.close()


###############################################################################
@click.command()
@click.argument('fit_file_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_0/drsFitParameter_.fits',
                type=click.Path(exists=True))
@click.argument('store_file_path',
                default='/home/fschulz/plots/version_0/residual/residual_roi_hist.jpg',
                type=click.Path(exists=False))
@click.argument('interval_array',
                default=[1, 2, 3])
@click.option('--drs_value_type', '-type',
              default='ROIOffset',
              type=click.Choice(['Baseline', 'Gain', 'ROIOffset']))
@click.argument('nr_bins',
                default=100)
###############################################################################
def drs_fit_value_residual_hist(fit_file_path, store_file_path,
                                interval_array, drs_value_type,
                                nr_bins):
    drs_fit_value_residual_hist_(fit_file_path, store_file_path,
                                 interval_array, drs_value_type,
                                 nr_bins)


###############################################################################
def drs_fit_value_residual_hist_(fit_file_path, store_file_path,
                                 interval_array, drs_value_type,
                                 nr_bins=100):
    upper_limit = 1  # mV or 1
    for interval_nr in interval_array:
        groupname = 'Interval'+str(interval_nr)
        print(groupname)
        with fits.open(fit_file_path, ignoremissing=True, ignore_missing_end=True) as fit_value_tab:
            residual = fit_value_tab[groupname].data[drs_value_type+'Residual'][0]

        unit_str = r'$\mathrm{mV}$'
        if(drs_value_type == 'Gain'):
            unit_str = r'$\times 10^{-3}$'
            residual /= pow(10, -3)

        drs_value_residual_mean = np.mean(residual)
        drs_value_residual_std = np.std(residual)
        drs_value_residual_max = max(residual)

        label = (groupname+':' +
                 '\n Hist:' +
                 '\n   mean: '+str(round(drs_value_residual_mean, 2))+unit_str +
                 '\n   std dev: '+str(round(drs_value_residual_std, 2))+unit_str +
                 '\n max value: '+str(round(drs_value_residual_max, 2)))+unit_str
        weights = np.full(len(residual), 100/len(residual))
        plt.hist(residual, bins=nr_bins, weights=weights,
                 range=(0, upper_limit), histtype='step', label=label)

    plt.title(r'Histogram of the absolute '+drs_value_type+'-residual mean', fontsize=16, y=1.01)
    plt.xlabel(r'Absolute residual mean /'+unit_str)
    plt.ylabel(r'Frequency /$\mathrm{\%}$')
    plt.xlim(0)
    plt.legend(loc='upper right')
    if(store_file_path is not None):
        plt.savefig(store_file_path)
    # plt.show()
    plt.close()


###############################################################################
@click.command()
@click.argument('drs_file_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_0/drsData.h5',
                type=click.Path(exists=True))
@click.argument('interval_file_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_0/intervalIndices.h5',
                type=click.Path(exists=True))
@click.argument('fit_file_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_0/drsFitParameter_.fits',
                type=click.Path(exists=True))
@click.argument('store_file_path',
                default=('/home/fschulz/plots/version_0/residual/' +
                         'gain/worst_100_cell_collection_I1.pdf'),
                type=click.Path(exists=False))
@click.argument('interval_nr',
                default=1)
@click.option('--drs_value_type', '-type',
              default='Gain',
              type=click.Choice(['Baseline', 'Gain', 'ROIOffset']))
@click.argument('worse_chids',
                default=[])
@click.argument('selected_values',
                default=100)
###############################################################################
def worst_cell_collection_selected_by_residual(
                                    drs_file_path, interval_file_path,
                                    fit_file_path, store_file_path,
                                    interval_nr, drs_value_type,
                                    worse_chids, selected_values):

    worst_cell_collection_selected_by_residual_(
                                    drs_file_path, interval_file_path,
                                    fit_file_path, store_file_path,
                                    interval_nr, drs_value_type,
                                    worse_chids, selected_values)


###############################################################################
def worst_cell_collection_selected_by_residual_(
                                    drs_file_path, interval_file_path,
                                    fit_file_path, store_file_path,
                                    interval_nr, drs_value_type,
                                    worse_chids, selected_values=100):

    NRCELLSPERCHID = config.nrCellsPerChid[drs_value_type]

    groupname = 'Interval'+str(interval_nr)

    interval_source = h5py.File(interval_file_path, 'r')[groupname]
    # cut_off_error_factor = interval_source.attrs['CutOff'+drs_value_type]
    interval_indices = np.array(interval_source['IntervalIndices'])

    chid_list = np.linspace(0, NRCHID-1, NRCHID, dtype='int')
    not_useful_chids = np.array([non_standard_chids['crazy'],
                                 non_standard_chids['dead']]).flatten()

    not_useful_chids = np.unique(np.sort(np.append(not_useful_chids, worse_chids)))

    useful_chids = chid_list[np.setdiff1d(chid_list, not_useful_chids)]
    fit_value_tab = fits.open(fit_file_path, ignoremissing=True, ignore_missing_end=True)[groupname].data

    residual_array = fit_value_tab[drs_value_type+'Residual'][0].reshape(NRCHID, NRCELLSPERCHID)[useful_chids, :].flatten()

    indices = np.argsort(residual_array)[-selected_values:]

    with PdfPages(store_file_path) as pdf:
        for index in tqdm(reversed(indices)):
            residual = residual_array[index]

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
            with h5py.File(drs_file_path, 'r') as store:
                time = np.array(store['Time'+drs_value_type][interval_indices, :]).flatten()
                temp = store['Temp'+drs_value_type][interval_indices, int(chid/9)]
                drs_value = store[drs_value_type][interval_indices, value_index]

            if(drs_value_type == 'Gain'):
                drs_value /= config.DACfactor

            sc_all = plt.scatter(temp, drs_value, c=np.array(time))
            plt.close()  # Dont show sc_all, just use it to create the colorbar

            fig, img = plt.subplots()

            intervalMonth = 3
            start_date = pd.to_datetime(time[0] * 24 * 3600 * 1e9).date()
            end_date = pd.to_datetime(time[-1] * 24 * 3600 * 1e9).date()
            timeLabel = pd.date_range(start=start_date, end=end_date, freq=str(intervalMonth)+'M')
            cbar = fig.colorbar(sc_all, ticks=dates.MonthLocator(interval=intervalMonth))
            cbar.ax.set_yticklabels(timeLabel.strftime('%b %Y'))
            timeColor = cbar.to_rgba(time)

            i_min, i_max = 0, len(temp)
            temp_range = np.linspace(min(temp)-1, max(temp)+1, 10000)
            color = timeColor[i_min: i_max]
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
            fitPlot, = plt.plot(temp_range, fit, '-', color=color_mean)

            plt.title((drs_value_type+', absolute residual mean: '+str('{:0.1f}'.format(residual)) +
                      unit_power_str +
                      '\nChid: '+str(chid)+', Cell: '+str(cell)), fontsize=15, y=1.00)  # , fontsize=20, y=0.95

            plt.xlabel(r'Temperature /$\mathrm{^\circ C}$')
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
                        'calibratedPedestelDataDistribution_mean.jpg', #_2016-04-01
                type=click.Path(exists=False))
@click.option('--calibrated_type', '-type',
              default='Mean',
              type=click.Choice(['Mean', 'Std']))
@click.argument('worse_chids',
                default=[])
@click.argument('datetime_limits_string',
                default=['2015-05-26', '2017-10-01'])# ['2014-05-20 12', '2015-05-26 12'] # '2015-05-26'
###############################################################################
def pedestel_mean_or_std_vs_temp_(data_folder_path, store_file_path,
                                  calibrated_type, worse_chids,
                                  datetime_limits_string):
    print('pedestel_mean_or_std_vs_temp_')
    in_pea = True

    noise_file_list = sorted([file for file in
                              os.listdir(data_folder_path)
                              if (file.startswith('pedestelStats') and
                                  file.endswith('.fits'))])

    chid_list = np.linspace(0, NRCHID-1, NRCHID, dtype='int')
    not_useful_chids = np.array([non_standard_chids['crazy'],
                                 non_standard_chids['dead']]).flatten()

    not_useful_chids = np.unique(np.sort(np.append(not_useful_chids, worse_chids)))

    useful_chids = chid_list[np.setdiff1d(chid_list, not_useful_chids)]

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

    ylabel = calibrated_type+r'/$\mathrm{mV}$'
    if(in_pea):
        ylabel = calibrated_type+r'/$\mathrm{PEA}$'

    gs = gridspec.GridSpec(1, 11)
    fig = plt.figure(figsize=(12, 10), dpi=100)
    fig.suptitle(r"Calibrated PedestelRun "+calibrated_type, fontsize=24, y=0.95)

    ax0 = plt.subplot(gs[0, 0:5])
    ax1 = plt.subplot(gs[0, 5:11], sharey=ax0)
    fig.subplots_adjust(wspace=0.5)
    ax1.tick_params(labelleft='off')

    label_str = ("Distribution: Mean:"+str(format(round(drs_file_calibrated_collection_mean, 3), '.3f')) +
                 ", Std: "+str(format(round(drs_file_calibrated_collection_std, 3), '.3f')))
    sc = ax0.scatter(time_list, drs_file_calibrated_mean_list,
                     s=50, marker=".", c=temp_diff_list, label=label_str)
    label_str = ("Distribution: Mean:"+str(format(round(drs_model_calibrated_collection_mean, 3), '.3f')) +
                 ", Std: "+str(format(round(drs_model_calibrated_collection_std, 3), '.3f')))
    sc = ax1.scatter(time_list, drs_model_calibrated_mean_list,
                     s=50, marker=".", c=temp_diff_list, label=label_str)

    cbar = fig.colorbar(sc)
    cbar.set_label(r'Delta Temperatur /$\mathrm{C\degree}$')
    ax0.xaxis.set_major_locator(dates.MonthLocator(interval=3))
    ax0.xaxis.set_minor_locator(dates.MonthLocator(interval=1))
    ax0.xaxis.set_major_formatter(dates.DateFormatter('%b %Y'))
    ax0.fmt_xdata = dates.DateFormatter('%Y-%m-%d %H:%M:%S')
    ax1.xaxis.set_major_locator(dates.MonthLocator(interval=3))
    ax1.xaxis.set_minor_locator(dates.MonthLocator(interval=1))
    ax1.xaxis.set_major_formatter(dates.DateFormatter('%b %Y'))
    ax1.fmt_xdata = dates.DateFormatter('%Y-%m-%d %H:%M:%S')
    plt.gcf().autofmt_xdate()

    ax0.set_title('DRS-File', fontsize=16, y=0.92)
    ax0.text(0.6, 0.15, "DRS-File")
    ax0.yaxis.grid(linestyle=':')
    ax0.set_xlabel(r'Time')
    ax0.set_ylabel(ylabel)
    ax0.legend(loc='lower left', numpoints=1, title="")
    ax1.set_title('Model', fontsize=16, y=0.92)
    ax1.yaxis.grid(linestyle=':')
    ax1.set_xlabel(r'Time')
    ax1.legend(loc='lower left', numpoints=1, title="")
    #plt.text(0.6, 0.15, "preliminary", fontdict=font, transform=plt.gca().transAxes)
    if(store_file_path is not None):
        plt.savefig(store_file_path)
    plt.show()
    plt.close()


###############################################################################
@click.command()
@click.argument('drs_file_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_0/drsData.h5',
                type=click.Path(exists=True))
@click.argument('interval_file_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_0/intervalIndices.h5',
                type=click.Path(exists=True))
@click.argument('fit_file_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_0/drsFitParameter_.fits',
                type=click.Path(exists=True))
@click.argument('store_file_path',
                default='/home/fschulz/plots/version_0/drsValues/gain/chid100_cell500_interval3.jpg',
                type=click.Path(exists=False))
@click.argument('interval_array',
                default=[3])
@click.option('--drs_value_type', '-type',
              default='Gain',
              type=click.Choice(['Baseline', 'Gain', 'ROIOffset']))
@click.argument('chid',
                default=100)
@click.argument('cell',
                default=500)
# @click.option('--show_var_dev', '-var',
#               is_flag=False)
###############################################################################
def drs_value_cell(drs_file_path, interval_file_path, fit_file_path,
                   store_file_path, interval_array, drs_value_type,
                   chid, cell):

    NRCELLSPERCHID = config.nrCellsPerChid[drs_value_type]
    value_index = chid*NRCELLSPERCHID + cell
    border = 2.0  # mV
    # Cecking wether the intervalIndices and the fitvalues are based on the given drsData
    # check_file_match(drs_file_path,
    #                  interval_file_path=interval_file_path,
    #                  fit_file_path=fit_file_path)

    # loading source data
    with h5py.File(drs_file_path, 'r') as store:
        time = np.array(store['Time'+drs_value_type]).flatten()

    use_mask = True
    in_PEA = False
    mask_collection = []
    time_collection = []
    temp_collection = []
    drs_value_collection = []
    fit_value_collection = []
    for interval_nr in interval_array:
        groupname = 'Interval'+str(interval_nr)
        with h5py.File(interval_file_path, 'r') as interval_source:
            data = interval_source[groupname]
            cut_off_error_factor = interval_source.attrs['CutOff'+drs_value_type]
            interval_indices = np.array(data['IntervalIndices'])
            if(use_mask):
                mask = np.array(data[drs_value_type+'Mask'])
                mask_collection.append(mask)
        with h5py.File(drs_file_path, 'r') as store:
            temp = np.array(store['Temp'+drs_value_type][interval_indices, int(chid/9)])
            drs_value = np.array(store[drs_value_type][interval_indices, value_index])
        with fits.open(fit_file_path, ignoremissing=True, ignore_missing_end=True) as fit_value_tab:
            data = fit_value_tab[groupname].data
            slope = data[drs_value_type+'Slope'][value_index]
            offset = data[drs_value_type+'Offset'][value_index]

        time_interval = time[interval_indices]
        time_collection.append(time_interval)
        temp_collection.append(temp)
        drs_value_collection.append(drs_value)
        fit_value_collection.append([slope, offset])

        ylabel_str = drs_value_type+r' /$\mathrm{mV}$'
        if(drs_value_type == 'Gain'):
            drs_value /= config.DACfactor
            ylabel_str = drs_value_type+r' /$\mathrm{1}$'
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
    timeLabel = pd.date_range(start=start_date, end=end_date, freq=str(intervalMonth)+'M')
    cbar = fig.colorbar(sc_all, ticks=dates.MonthLocator(interval=intervalMonth))
    cbar.ax.set_yticklabels(timeLabel.strftime('%b %Y'))
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
            img.scatter(temp[mask_u], drs_value[mask_u], s=50, marker='+',
                        c=color)
            img.scatter(temp[mask_nu], drs_value[mask_nu], s=50, marker='*',
                        c=color, edgecolors='k', linewidth=0.5)

            l_1 = img.scatter([], [], s=50, marker='+', c='r')
            l_2 = img.scatter([], [], s=50, marker='*', c='r', edgecolors='k')

            first_legend = plt.legend([l_1, l_2], ['Used values', 'Not used values'],
                                      loc='upper right', scatterpoints=1, numpoints=1)
            plt.gca().add_artist(first_legend)
        else:
            img.scatter(temp, drs_value, s=50, marker='+', c=color)
        i_min = i_min+i_max

        slope, offset = fit_value_collection[interval_index]
        fit = linearerFit(temp_range, slope, offset)

        color_mean = np.mean(color, axis=0)
        fitPlot, = plt.plot(temp_range, fit, '-', color=color_mean,
                            label='Slope: {:0.3f}, Offset: {:0.3f}'.format(slope, offset))
        plt.legend(handles=[fitPlot], loc='lower center')

    plt.title('    Chid: '+str(chid)+', Cell: '+str(cell) +
              ', ErrFactor: '+str('{:0.1f}'.format(cut_off_error_factor)), fontsize=18, y=1.04)

    plt.xlabel(r'Temperature /$\mathrm{^\circ C}$')
    plt.ylabel(ylabel_str)
    plt.xlim(min(temp_list)-1, max(temp_list)+1)
    plt.ylim(get_best_limits(drs_value, 0.25))
    plt.grid()
    plt.gca().ticklabel_format(useOffset=False)
    plt.text(0.02, 0.21, 'preliminary', fontdict=font, transform=plt.gca().transAxes)
    if(store_file_path is not None):
        plt.savefig(store_file_path)
    plt.show()
    plt.close()


@click.command()
@click.argument('fit_file_path',
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
def drs_value_chid_fit_parameter(fit_file_path, store_file_path, interval_array,
                                 drs_value_type, fit_parameter_type, chid):

    for interval_nr in interval_array:
        groupname = 'Interval'+str(interval_nr)
        NRCELLSPERCHID = config.nrCellsPerChid[drs_value_type]
        with fits.open(fit_file_path, ignoremissing=True, ignore_missing_end=True) as fit_value_tab:
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
@click.argument('chi2_file_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/validating/chiSquare/drsChiSquare.h5',
                type=click.Path(exists=True))
@click.argument('store_file_path',
                default='/home/fschulz/plots/chi2_cell_mean_per_chid_gain.pdf',
                type=click.Path(exists=False))
@click.argument('interval_array',
                default=[1, 2, 3])
@click.option('--drs_value_type', '-type',
              default='Gain',
              type=click.Choice(['Baseline', 'Gain', 'ROIOffset']))
###############################################################################
def drs_values_chi2_cell_mean_per_chid(chi2_file_path, store_file_path,
                                       interval_array, drs_value_type):

    y1_prop, y2_prop = 3, 1
    gs = gridspec.GridSpec(2, 1, height_ratios=[y2_prop, y1_prop])
    plt.figure(figsize=(10, 8))

    y_split = 1
    with PdfPages(store_file_path) as pdf:
        for interval_nr in interval_array:
            groupname = 'Interval'+str(interval_nr)
            print(groupname)
            NRCELLSPERCHID = config.nrCellsPerChid[drs_value_type]
            with h5py.File(chi2_file_path, 'r') as chi2_tab:
                data = chi2_tab[groupname]
                chid_chi2 = np.mean(np.array(data[drs_value_type+'Chi2']).reshape(NRCHID, NRCELLSPERCHID), axis=1)
            plt.close()
            plt.figure(figsize=(10, 8))
            plt.ylabel(r' $\left(|CHI2|\right)$ /$\mathrm{1}$')

            ax0 = plt.subplot(gs[1, 0])
            ax1 = plt.subplot(gs[0, 0], sharex=ax0)
            plt.subplots_adjust(hspace=0.1)
            plt.setp(ax1.get_xticklabels(), visible=False)
            ax0.spines['top'].set_visible(False)
            ax1.spines['bottom'].set_visible(False)
            ax1.xaxis.tick_top()
            ax1.tick_params(labeltop='off')  # don't put tick labels at the top

            plt.title(drs_value_type+'\n'+groupname)
            ax0.step(range(1, NRCHID+1), chid_chi2, where='mid')
            ax1.step(range(1, NRCHID+1), chid_chi2, where='mid')

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
            ax0.set_xlabel('CHID')
            pdf.savefig()
            plt.close()


@click.command()
@click.argument('chi2_file_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/validating/chiSquare/drsChiSquare.h5',
                type=click.Path(exists=True))
@click.argument('store_file_path',
                default='/home/fschulz/plots/chi2_fact_cam_gain.jpg',
                type=click.Path(exists=False))
@click.argument('interval_array',
                default=[1])
@click.option('--drs_value_type', '-type',
              default='Gain',
              type=click.Choice(['Baseline', 'Gain', 'ROIOffset']))
###############################################################################
def chi2_fact_cam(chi2_file_path, store_file_path,
                  interval_array, drs_value_type):

    for interval_nr in interval_array:
        groupname = 'Interval'+str(interval_nr)
        with h5py.File(chi2_file_path, 'r') as chi2_tab:
            data = chi2_tab[groupname]
            chi2 = np.mean(np.array(data[drs_value_type+'Chi2']).reshape(1440, 1024), axis=1)

    plot = factPlots.camera(abs(chi2), cmap='hot')
    plt.colorbar(plot, label=r'$| \mathrm{Chi}^2 |$')
    if(store_file_path is not None):
        plt.savefig(store_file_path)
    plt.show()
    plt.close()


@click.command()
@click.argument('drs_file_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_0/drsData.h5',
                type=click.Path(exists=True))
@click.argument('interval_file_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_0/intervalIndices.h5',
                type=click.Path(exists=True))
@click.argument('fit_file_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_0/drsFitParameter.fits',
                type=click.Path(exists=True))
@click.argument('store_file_path',
                default='/home/fschulz/plots/version_0/chi2/outlier_cell_collection_I3_gain_limit_65.pdf',
                type=click.Path(exists=False))
@click.argument('interval_nr',
                default=3)
@click.option('--drs_value_type', '-type',
              default='Gain',
              type=click.Choice(['Baseline', 'Gain', 'ROIOffset']))
@click.argument('chi2_limit',
                default=65)
###############################################################################
def drs_values_chi2_outlier_cell_collection(drs_file_path, interval_file_path,
                                            fit_file_path,
                                            store_file_path, interval_nr,
                                            drs_value_type, chi2_limit):

    # Cecking wether the intervalIndices and the fitvalues are based on the given drsData
    # check_file_match(drs_file_path,
    #                  interval_file_path=interval_file_path,
    #                  fit_file_path=fit_file_path)

    NRCELLSPERCHID = config.nrCellsPerChid[drs_value_type]

    groupname = 'Interval'+str(interval_nr)

    interval_source = h5py.File(interval_file_path, 'r')
    # cut_off_error_factor = interval_source.attrs['CutOff'+drs_value_type]
    interval_source = interval_source[groupname]
    interval_indices = np.array(interval_source['IntervalIndices'])

    fit_value_tab = fits.open(fit_file_path, ignoremissing=True, ignore_missing_end=True)[groupname].data
    chi2 = fit_value_tab[drs_value_type+'Chi2']
    count = 0
    with PdfPages(store_file_path) as pdf:
        for chid in tqdm(range(NRCHID)):
            chi2_chid = chi2[chid*NRCELLSPERCHID:(chid+1)*NRCELLSPERCHID]
            nr_of_incrased_chi2_values = len(chi2_chid[chi2_chid > chi2_limit])
            if nr_of_incrased_chi2_values > 10:
                print(chid, nr_of_incrased_chi2_values)
                continue
            for cell in range(NRCELLSPERCHID):
                value_index = chid*NRCELLSPERCHID + cell
                if chi2_chid[cell] > chi2_limit:
                    print('Chid: ', chid, 'Cell: ', cell, 'Chi2: ', chi2_chid[cell])
                    count += 1
                    mask = np.array(interval_source[drs_value_type+'Mask'][:, value_index])
                    with h5py.File(drs_file_path, 'r') as store:
                        time = np.array(store['Time'+drs_value_type][interval_indices, :]).flatten()
                        temp = store['Temp'+drs_value_type][interval_indices, int(chid/9)]
                        drs_value = store[drs_value_type][interval_indices, value_index]

                    if(drs_value_type == 'Gain'):
                        drs_value /= config.DACfactor
                    sc_all = plt.scatter(temp, drs_value, c=np.array(time))
                    plt.close()  # Dont show sc_all, just use it to create the colorbar

                    fig, img = plt.subplots()

                    intervalMonth = 3
                    start_date = pd.to_datetime(time[0] * 24 * 3600 * 1e9).date()
                    end_date = pd.to_datetime(time[-1] * 24 * 3600 * 1e9).date()
                    timeLabel = pd.date_range(start=start_date, end=end_date, freq=str(intervalMonth)+'M')
                    cbar = fig.colorbar(sc_all, ticks=dates.MonthLocator(interval=intervalMonth))
                    cbar.ax.set_yticklabels(timeLabel.strftime('%b %Y'))
                    timeColor = cbar.to_rgba(time)

                    i_min, i_max = 0, len(temp)
                    temp_range = np.linspace(min(temp)-1, max(temp)+1, 10000)
                    color = timeColor[i_min: i_max]
                    mask_u = mask
                    mask_nu = np.logical_not(mask_u)
                    img.scatter(temp[mask_u], drs_value[mask_u], s=50, marker='+',
                                c=color, label='paternNoiseMean with averaged Temperature')
                    img.scatter(temp[mask_nu], drs_value[mask_nu], s=50, marker='*',
                                c=color, label='paternNoiseMean with averaged Temperature')

                    slope = fit_value_tab[drs_value_type+'Slope'][value_index]
                    offset = fit_value_tab[drs_value_type+'Offset'][value_index]
                    fit = linearerFit(temp_range, slope, offset)

                    color_mean = np.mean(color, axis=0)
                    fitPlot, = plt.plot(temp_range, fit, '-', color=color_mean)

                    #fitPlot, = plt.plot(temp_range, fit-single_photon_limit, '--', color=color_mean)
                    #fitPlot, = plt.plot(temp_range, fit+single_photon_limit, '--', color=color_mean)

                    plt.title((drs_value_type+', Chi2: '+str('{:0.1f}'.format(chi2[value_index])) +
                              '\nChid: '+str(chid)+', Cell: '+str(cell)), fontsize=15, y=1.00)  # , fontsize=20, y=0.95

                    plt.xlabel(r'Temperature /$\mathrm{^\circ C}$')
                    plt.ylabel(drs_value_type+r' /$\mathrm{mV}$')
                    plt.xlim(min(temp)-1, max(temp)+1)
                    plt.grid()
                    plt.gca().ticklabel_format(useOffset=False)
                    pdf.savefig()
    print(count)


@click.command()
@click.argument('drs_file_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/drsData.h5',
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
def chid_cell_drs_values_time(drs_file_path, interval_file_path,
                              store_file_path, interval_array,
                              drs_value_type, chid, cell):

    value_index = chid*NRCELL + cell

    # Cecking wether the intervalIndices and the fitvalues are based on the given drsData
    with h5py.File(drs_file_path, 'r') as data_source:
        source_creation_date = data_source.attrs['CreationDate']

    with h5py.File(interval_file_path) as interval_source:
        used_source_creation_date_i = interval_source.attrs['SCDate']

    if(source_creation_date != used_source_creation_date_i):
        error_str = ("'interval_file_path' is not based on the given 'source_file_path'")
        print(error_str)
        return

    with h5py.File(drs_file_path, 'r') as store:
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
        with h5py.File(drs_file_path, 'r') as store:
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
        #ylabel_str = r'Temperature /$\mathrm{^\circ C}$'
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
    #timeLabel = pd.date_range(start=start_date_, end=end_date_, freq='M') - pd.offsets.MonthBegin(1)
    #plt.xticks(timeLabel, timeLabel, rotation=30)
    #plt.gca().xaxis.set_major_formatter(time.DateFormatter('%d.%m.%y'))
    plt.text(0.02, 0.21, 'preliminary', fontdict=font, transform=plt.gca().transAxes)
    if(store_file_path is not None):
        plt.savefig(store_file_path)
    plt.show()
    plt.close()


@click.command()
@click.argument('residual_file_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/residual/drsresidual.h5',
                type=click.Path(exists=True))
@click.argument('interval_file_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/intervalIndices.h5',
                type=click.Path(exists=True))
@click.argument('interval_array',
                default=[1])
@click.argument('drs_value_type',
                default='Baseline')
@click.argument('store_file_path',
                default='/home/fschulz/plots/residualBaseline_his_I1.jpg',
                type=click.Path(exists=False))
####################################################################################################
def residual_hist(residual_file_path, interval_file_path, interval_array, drs_value_type, store_file_path):

    with h5py.File(interval_file_path) as interval_source:
        used_source_creation_date_i = interval_source.attrs['SCDate']

    with h5py.File(residual_file_path) as residual_source:
        used_source_creation_date_r = residual_source.attrs['SCDate']

    if(used_source_creation_date_i != used_source_creation_date_r):
        error_str = ("'interval_file_path' and 'residual_file_path' dont belong together")
        print(error_str)

    for interval_nr in interval_array:
        groupname = 'Interval'+str(interval_nr)
        with h5py.File(interval_file_path, 'r') as interval_source:
            data = interval_source[groupname]
            mask = np.array(data[drs_value_type+'Mask'])

        with h5py.File(residual_file_path, 'r') as residual_tab:
            data = residual_tab[groupname]
            residual = np.array(data[drs_value_type+'residual'])
            residual = residual[mask].flatten()

    outlier = np.where(abs(residual) > 2.1)[0]
    print('outlier: ', len(outlier)/len(residual)*100, ' %')

    nr_bins = 40
    title_str = 'Hist'
    plt.title(title_str, y=1.0)
    weights = np.full(len(residual), 100/len(residual))
    hist1 = plt.hist(residual, weights=weights, bins=nr_bins, histtype='step',
                     range=(-1, 1), lw=1, edgecolor='r', label='test')
    if(store_file_path is not None):
        plt.savefig(store_file_path)
    plt.show()
    plt.close()


@click.command()
@click.argument('residual_file_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/residual/drsresidual.h5',
                type=click.Path(exists=True))
@click.argument('interval_file_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/intervalIndices.h5',
                type=click.Path(exists=True))
@click.argument('interval_array',
                default=[1])
@click.argument('drs_value_type',
                default='Baseline')
@click.argument('store_file_path',
                default='/home/fschulz/plots/residualBaseline_hisout_I1.jpg',
                type=click.Path(exists=False))
###############################################################################
def residual_hist_outlier(residual_file_path, interval_file_path,
                           interval_array, drs_value_type,
                           store_file_path):

    with h5py.File(interval_file_path) as interval_source:
        used_source_creation_date_i = interval_source.attrs['SCDate']

    with h5py.File(residual_file_path) as residual_source:
        used_source_creation_date_r = residual_source.attrs['SCDate']

    if(used_source_creation_date_i != used_source_creation_date_r):
        error_str = ("'interval_file_path' and 'residual_file_path' dont belong together")
        print(error_str)

    single_photon_limit = 2.1  # mV
    for interval_nr in interval_array:
        groupname = 'Interval'+str(interval_nr)
        with h5py.File(interval_file_path, 'r') as interval_source:
            data = interval_source[groupname]
            mask = np.array(data[drs_value_type+'Mask'])

        with h5py.File(residual_file_path, 'r') as residual_tab:
            data = residual_tab[groupname]
            residual = np.array(data[drs_value_type+'residual'])

    chid_residuen = np.zeros((NRCHID, 1))
    for chid in range(NRCHID):
        sub_mask = mask[:, chid*NRCELL:(chid+1)*NRCELL]
        residual_chid = residual[:, chid*NRCELL:(chid+1)*NRCELL][sub_mask].flatten()
        value = len(residual_chid[residual_chid > single_photon_limit])/len(residual_chid)*100
        chid_residuen[chid] = value

    plt.title(('Frequency of '+drs_value_type+'residual \n' +
               'over the limit of '+str(single_photon_limit)+r' $\mathrm{mV}$'))
    plt.step(range(1, NRCHID+1), chid_residuen, where='mid')
    max_ = np.amax(chid_residuen)
    # for chid in range(NRCHID):
    #     if(chid % 9 == 8):
    #         plt.plot([chid+1, chid+1], [0, max_], 'r-')
    plt.xlabel('CHID')
    plt.ylabel(r'Frequency /$\mathrm{\%}$')
    plt.xlim(1, NRCHID+1)
    plt.ylim(0, )
    if(store_file_path is not None):
        plt.savefig(store_file_path)
    plt.show()
    plt.close()


@click.command()
@click.argument('residual_file_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/residual/drsresidual.h5',
                type=click.Path(exists=True))
@click.argument('store_file_path',
                default='/home/fschulz/plots/residual_cell_mean_per_chid_gain.pdf',
                type=click.Path(exists=False))
@click.argument('interval_array',
                default=[1, 2, 3])
@click.argument('drs_value_type',
                default='Gain')
###############################################################################
def drs_values_residual_cell_mean_per_chid(residual_file_path, store_file_path,
                                            interval_array, drs_value_type):

    with PdfPages(store_file_path) as pdf:
        for interval_nr in interval_array:
            groupname = 'Interval'+str(interval_nr)
            print(groupname)
            with h5py.File(residual_file_path, 'r') as residual_tab:
                data = residual_tab[groupname]
                residual = np.mean(np.mean(abs(np.array(data[drs_value_type+'residual'])), axis=0).reshape(NRCHID, NRCELL), axis=1)
            plt.title(drs_value_type+'\n'+groupname)
            plt.step(range(1, NRCHID+1), residual, where='mid')
            plt.xlabel('CHID')
            plt.ylabel(r' $\left($|residual|$\right)$ /$\mathrm{mV}$')
            pdf.savefig()
            plt.close()


@click.command()
@click.argument('drs_file_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/drsData.h5',
                type=click.Path(exists=True))
@click.argument('interval_file_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/intervalIndices.h5',
                type=click.Path(exists=True))
@click.argument('residual_file_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/residual/drsresidual.h5',
                type=click.Path(exists=True))
@click.argument('interval_array',
                default=[1, 2, 3])
@click.argument('drs_value_type',
                default='Baseline')
@click.argument('chid',
                default=161)
@click.argument('cell',
                default=494)
@click.argument('restrict_residual',
                default=True,
                type=click.BOOL)
@click.argument('store_file_path',
                default='/home/fschulz/plots/residualBaseline_chid721_cap500_3Intervals_RestrictParts.jpg',
                type=click.Path(exists=False))
###############################################################################
def residual_per_chid_cell(drs_file_path, interval_file_path,
                           residual_file_path, store_file_path,
                           interval_array, drs_value_type,
                           chid, cell,
                           restrict_residual):

    value_index = chid*NRCELL + cell

    # Cecking wether the intervalIndices and the fitvalues are based on the given drsData
    with h5py.File(drs_file_path, 'r') as data_source:
        source_creation_date = data_source.attrs['CreationDate']

    with h5py.File(interval_file_path) as interval_source:
        used_source_creation_date_i = interval_source.attrs['SCDate']

    with h5py.File(residual_file_path) as residual_source:
        used_source_creation_date_r = residual_source.attrs['SCDate']

    if((source_creation_date != used_source_creation_date_i) or
       (source_creation_date != used_source_creation_date_r)):
        error_str = ("'interval_file_path' or 'residual_file_path' is not based on the given 'source_file_path'")
        print(error_str)
        return

    # loading source data
    with h5py.File(drs_file_path, 'r') as store:
        time = np.array(store['Time'+drs_value_type]).flatten()
    datetime = pd.to_datetime(time * 24 * 3600 * 1e9)

    use_mask = True
    in_PEA = False
    offset = 0.1  # TODO maybe ask/ magic number
    mask_collection = []
    datetime_collection = []
    residual_collection = []
    boundarie_collection = []
    for interval_nr in interval_array:
        groupname = 'Interval'+str(interval_nr)
        print(groupname)
        with h5py.File(interval_file_path, 'r') as interval_source:
            data = interval_source[groupname]
            low_limit = pd.to_datetime(data.attrs['LowLimit'])
            upp_limit = pd.to_datetime(data.attrs['UppLimit'])
            interval_indices = np.array(data['IntervalIndices'])
            print(interval_indices.shape)
            if(use_mask):
                mask = np.array(data[drs_value_type+'Mask'][:, value_index])
                print(mask.shape)
                mask_collection.append(mask)
        with h5py.File(residual_file_path, 'r') as residual_tab:
            data = residual_tab[groupname]
            residual = np.array(data[drs_value_type+'residual'][:, value_index])
            print(residual.shape)
        datetime_interval = datetime[interval_indices]

        ylabel_str = '(f(t)-'+drs_value_type+'Mean) /$\mathrm{mV}$'
        if(in_PEA):
            ylabel_str = '(f(t)-'+drs_value_type+'Mean) /$\mathrm{PEA}$'
            for drs_value_index in range(len(residual_collection)):
                residual *= PEAFACTOR

        datetime_collection.append(datetime_interval)
        residual_collection.append(residual)
        boundarie_collection.append([low_limit, upp_limit])

    nr_of_intervals = len(interval_array)
    datetime_collection_ = np.concatenate(datetime_collection)
    residual_collection_ = np.concatenate(residual_collection)
    min_res, max_res = np.amin(residual_collection_), np.amax(residual_collection_)
    for interval_index in range(nr_of_intervals):
        datetime = datetime_collection[interval_index]
        residual = residual_collection[interval_index]
        low_limit, upp_limit = boundarie_collection[interval_index]
        c = [float(interval_index)/float(nr_of_intervals),
             0.0,
             float(nr_of_intervals-1-interval_index)/float(nr_of_intervals)]
        plt.plot([low_limit, low_limit], [min_res-offset, max_res+offset], 'k-')
        plt.plot([upp_limit, upp_limit], [min_res-offset, max_res+offset], 'k-')

        plt.annotate(s='', xy=(low_limit, 0), xytext=(upp_limit, 0),
                     arrowprops=dict(arrowstyle='<->', color=c))

        if(use_mask):
            mask_u = mask_collection[interval_index]
            mask_nu = np.logical_not(mask_u)
            plt.plot(datetime[mask_u], residual[mask_u], 'x', color=c)
            plt.plot(datetime[mask_nu], residual[mask_nu], '*', color=c)
        else:
            plt.plot(datetime, residual, 'x', color=c)

    plt.title(drs_value_type+'residual \n CHID: '+str(chid)+' Cell: '+str(cell))
    plt.ylabel(ylabel_str)
    plt.gcf().autofmt_xdate()
    plt.xlim(pd.to_datetime(np.amin(datetime_collection_)).date()-pd.DateOffset(days=7),
             pd.to_datetime(np.amax(datetime_collection_)).date()+pd.DateOffset(days=7))
    plt.ylim(min_res-offset, max_res+offset)
    plt.grid()
    plt.text(0.02, 0.2, 'preliminary', fontdict=font, transform=plt.gca().transAxes)
    if(store_file_path is not None):
        plt.savefig(store_file_path)
    plt.show()
    plt.close()


@click.command()
@click.argument('drs_file_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/drsData.h5',
                type=click.Path(exists=True))
@click.argument('interval_file_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/intervalIndices.h5',
                type=click.Path(exists=True))
@click.argument('residual_file_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/residual/drsresidual_.h5',
                type=click.Path(exists=True))
@click.argument('interval_array',
                default=[1])
@click.argument('drs_value_type',
                default='Baseline')
@click.argument('chid',
                default=1400)
@click.argument('restrict_residual',
                default=True,
                type=click.BOOL)
@click.argument('store_file_path',
                default='/home/fschulz/plots/residualBaseline_chid1000_cap500_3Intervals_RestrictParts.jpg',
                type=click.Path(exists=False))
####################################################################################################
def residual_mean_per_chid(drs_file_path, interval_file_path, residual_file_path, interval_array, drs_value_type,
                             chid, restrict_residual, store_file_path):

    with h5py.File(drs_file_path, 'r') as store:
        time = np.array(store['Time'+drs_value_type]).flatten()
        datetime = pd.to_datetime(time * 24 * 3600 * 1e9)

    date = datetime.date

    offset = 0.1  # TODO maybe ask/ magic number
    time_collection = []
    residual_collection = []
    for interval_nr in interval_array:
        groupname = 'Interval'+str(interval_nr)
        with h5py.File(residual_file_path, 'r') as residual_tab:
            data = residual_tab[groupname]

            low_limit = pd.to_datetime(data.attrs['LowLimit'])
            upp_limit = pd.to_datetime(data.attrs['UppLimit'])

            residual = np.array(data[drs_value_type+'residual'][chid*NRCELL: (chid+1)*NRCELL])

        interval_indices = np.where((datetime >= low_limit) & (datetime <= upp_limit))[0]
        datetime_interval = datetime[interval_indices]
        with h5py.File(drs_file_path, 'r') as store:
            drs_value_var = np.array(store[drs_value_type+'Std'][interval_indices, chid*NRCELL: (chid+1)*NRCELL])

        if(False):  # TODO fix
            print(drs_value_var.shape)
            drs_value_var_mean = np.mean(drs_value_var, dtype='float64', axis=1)
            print(drs_value_var_mean.shape)
            drs_value_var_limit = drs_value_var_mean*2  # cut_off_factor[drs_value_type]
            indices_used_values = np.where(drs_value_var_mean < drs_value_var_limit)[0]
            indices_not_used_values = np.where(drs_value_var_mean >= drs_value_var_limit)[0]
            print(residual.shape)
            print(indices_used_values)
            residual = residual[: , indices_used_values]
            datetime_interval = datetime_interval[indices_used_values]

        residual_mean_per_chid = np.mean(residual, dtype='float64', axis=0)
        residual_collection.append(residual_mean_per_chid)
        time_collection.append(datetime_interval)

        nr_of_intervals = len(interval_array)
        interval_nr =1  # TODO fix
        c = [float(interval_nr-1)/float(nr_of_intervals),
             0.0,
             float(nr_of_intervals-1-(interval_nr-1))/float(nr_of_intervals)]
        min_res, max_res = min(residual_mean_per_chid), max(residual_mean_per_chid)  # TODO set lower
        plt.plot([low_limit, low_limit], [min_res-offset, max_res+offset], 'k-')
        plt.plot([upp_limit, upp_limit], [min_res-offset, max_res+offset], 'k-')

        plt.annotate(s='', xy=(low_limit, 0), xytext=(upp_limit, 0),
                     arrowprops=dict(arrowstyle='<->', color=c))

        plt.plot(datetime_interval, residual_mean_per_chid, 'x', color=c)

    residual_list = np.concatenate(residual_collection).ravel()
    time_list = np.concatenate(time_collection).ravel()

    min_res, max_res = min(residual_list), max(residual_list)

    # plt.errorbar(datetime, residualMean, yerr=residualMeanStd, fmt='x',
    #              label=(drs_value_type+r'Mean - f(t)'))
    plt.title(drs_value_type+'residual Mean:')
    plt.ylabel('(f(t)-'+drs_value_type+'Mean)/$\mathrm{mV}$')
    plt.gcf().autofmt_xdate()
    #plt.xlim(min(date)-pd.DateOffset(days=7), max(date)+pd.DateOffset(days=7))
    plt.ylim(min_res-offset, max_res+offset)
    plt.grid()
    plt.text(0.02, 0.2, 'preliminary', fontdict=font, transform=plt.gca().transAxes)
    if(store_file_path is not None):
        plt.savefig(store_file_path)
    plt.show()
    plt.close()


@click.command()
@click.argument('drs_file_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/drsData.h5',
                type=click.Path(exists=True))
@click.argument('interval_file_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/intervalIndices.h5',
                type=click.Path(exists=True))
@click.argument('residual_file_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/residual/drsresidual.h5',
                type=click.Path(exists=True))
@click.argument('interval_array',
                default=[3])
@click.argument('drs_value_type',
                default='Baseline')
@click.argument('crate_nr',
                default=1)
@click.argument('restrict_residual',
                default=True,
                type=click.BOOL)
@click.argument('store_file_path',
                default='/home/fschulz/plots/residualBaseline_chid1000_cap500_3Intervals_RestrictParts.jpg',
                type=click.Path(exists=False))
####################################################################################################
def residual_mean_per_crate(drs_file_path, interval_file_path, residual_file_path, interval_array, drs_value_type,
                             crate_nr, restrict_residual, store_file_path):

    crate_index = crate_nr-1
    with h5py.File(drs_file_path, 'r') as store:
        time = np.array(store['Time'+drs_value_type]).flatten()
        datetime = pd.to_datetime(time * 24 * 3600 * 1e9)

    date = datetime.date

    offset = 0.1  # TODO maybe ask/ magic number
    time_collection = []
    residual_collection = []
    for interval_nr in interval_array:
        groupname = 'Interval'+str(interval_nr)
        with h5py.File(residual_file_path, 'r') as residual_tab:
            data = residual_tab[groupname]

            low_limit = pd.to_datetime(data.attrs['LowLimit'])
            upp_limit = pd.to_datetime(data.attrs['UppLimit'])

            residual = np.array(data[drs_value_type+'residual'][crate_index*int(NRCHID/4)*NRCELL: (crate_index+1)*int(NRCHID/4)*NRCELL])

        interval_indices = np.where((datetime >= low_limit) & (datetime <= upp_limit))[0]
        datetime_interval = datetime[interval_indices]
        with h5py.File(drs_file_path, 'r') as store:
            drs_value_var = np.array(store[drs_value_type+'Std'][crate_index*int(NRCHID/4)*NRCELL: (crate_index+1)*int(NRCHID/4)*NRCELL])

        if(False):  # TODO fix
            print(drs_value_var.shape)
            drs_value_var_mean = np.mean(drs_value_var, dtype='float64', axis=1)
            print(drs_value_var_mean.shape)
            drs_value_var_limit = drs_value_var_mean*2  # cut_off_factor[drs_value_type]
            indices_used_values = np.where(drs_value_var_mean < drs_value_var_limit)[0]
            indices_not_used_values = np.where(drs_value_var_mean >= drs_value_var_limit)[0]
            print(residual.shape)
            print(indices_used_values)
            residual = residual[: , indices_used_values]
            datetime_interval = datetime_interval[indices_used_values]

        residual_mean_per_chid = np.mean(residual, dtype='float64', axis=0)
        residual_collection.append(residual_mean_per_chid)
        time_collection.append(datetime_interval)

        nr_of_intervals = len(interval_array)
        interval_nr =1  # TODO fix
        c = [float(interval_nr-1)/float(nr_of_intervals),
             0.0,
             float(nr_of_intervals-1-(interval_nr-1))/float(nr_of_intervals)]
        min_res, max_res = min(residual_mean_per_chid), max(residual_mean_per_chid)  # TODO set lower
        plt.plot([low_limit, low_limit], [min_res-offset, max_res+offset], 'k-')
        plt.plot([upp_limit, upp_limit], [min_res-offset, max_res+offset], 'k-')

        plt.annotate(s='', xy=(low_limit, 0), xytext=(upp_limit, 0),
                     arrowprops=dict(arrowstyle='<->', color=c))

        plt.plot(datetime_interval, residual_mean_per_chid, 'x', color=c)

    residual_list = np.concatenate(residual_collection).ravel()
    time_list = np.concatenate(time_collection).ravel()

    min_res, max_res = min(residual_list), max(residual_list)

    # plt.errorbar(datetime, residualMean, yerr=residualMeanStd, fmt='x',
    #              label=(drs_value_type+r'Mean - f(t)'))
    plt.title(drs_value_type+'residual Mean:')
    plt.ylabel('(f(t)-'+drs_value_type+'Mean)/$\mathrm{mV}$')
    plt.gcf().autofmt_xdate()
    #plt.xlim(min(date)-pd.DateOffset(days=7), max(date)+pd.DateOffset(days=7))
    plt.ylim(min_res-offset, max_res+offset)
    plt.grid()
    plt.text(0.02, 0.2, 'preliminary', fontdict=font, transform=plt.gca().transAxes)
    if(store_file_path is not None):
        plt.savefig(store_file_path)
    plt.show()
    plt.close()


####################################################################################################
def residualMeanOfAllCellsPerCrates(drs_file_path, residualFilenameArray_, drs_value_type,
                                     restrict_residual, store_file_path=None):

    nr_of_intervals = len(residualFilenameArray_)

    print("Loading '"+drs_value_type+"-data' ...")
    with h5py.File(drs_file_path, 'r') as store:
        time = np.array(store['Time'+drs_value_type]).flatten()
        datetime = pd.to_datetime(time * 24 * 3600 * 1e9)

    date = datetime.date
    datetime = np.array(datetime)

    intervalList = []
    residualPairList = []
    min_res, max_res = 0, 0
    min_res_, max_res_ = 0, 0
    for residualFilename in residualFilenameArray_:
        with h5py.File(residualFilename, 'r') as residual_tab:
            residual = np.array(residual_tab['residual'+drs_value_type])
            residualMean = np.mean(residual, dtype='float64', axis=0)
            residualMeanC1 = np.mean(residual[0*9*NRCELL:40*9*NRCELL, :], dtype='float64', axis=0)
            residualMeanC2 = np.mean(residual[40*9*NRCELL:80*9*NRCELL, :], dtype='float64', axis=0)
            residualMeanC3 = np.mean(residual[80*9*NRCELL:120*9*NRCELL, :], dtype='float64', axis=0)
            residualMeanC4 = np.mean(residual[120*9*NRCELL:160*9*NRCELL, :], dtype='float64', axis=0)

            residualMeanPerCrates = np.array([residualMeanC1, residualMeanC2,
                                              residualMeanC3, residualMeanC4,
                                              residualMean])

            interval_b = np.array(residual_tab['Interval'])

            interval = []
            for intervalLimit in interval_b:
                interval.append(pd.to_datetime(intervalLimit[0].decode('UTF-8')).date())

            if(restrict_residual):
                intervalIndices = np.where((date >= interval[0]) & (date <= interval[1]))[0]

                residualPair = [datetime[intervalIndices], residualMeanPerCrates[:, intervalIndices]]
            else:
                residualPair = [datetime, residualMeanPerCrates]

            min_res_, max_res_ = np.amin(residualPair[1]), np.amax(residualPair[1])
            if(min_res_ < min_res):
                min_res = min_res_
            if(max_res_ > max_res):
                max_res = max_res_

        intervalList.append(interval)
        residualPairList.append(residualPair)

    offset = min([abs(min_res*0.1), abs(max_res*0.1)])
    if(nr_of_intervals > 1):
        for intervalIndex in range(nr_of_intervals):
            plt.plot([intervalList[intervalIndex][0], intervalList[intervalIndex][0]],
                     [min_res-offset, max_res+offset], 'k-')
            plt.plot([intervalList[intervalIndex][1], intervalList[intervalIndex][1]],
                     [min_res-offset, max_res+offset], 'k-')

            print(residualPairList[intervalIndex][0].shape, residualPairList[intervalIndex][1].shape)

            print(residualPairList[intervalIndex][0].shape, residualPairList[intervalIndex][1][0].shape)
            print(residualPairList[intervalIndex][0].shape, residualPairList[intervalIndex][1][1].shape)
            print(residualPairList[intervalIndex][0].shape, residualPairList[intervalIndex][1][2].shape)

            plt.plot(residualPairList[intervalIndex][0], residualPairList[intervalIndex][1][0])
            plt.plot(residualPairList[intervalIndex][0], residualPairList[intervalIndex][1][1])
            plt.plot(residualPairList[intervalIndex][0], residualPairList[intervalIndex][1][2])
            plt.plot(residualPairList[intervalIndex][0], residualPairList[intervalIndex][1][3])
            plt.plot(residualPairList[intervalIndex][0], residualPairList[intervalIndex][1][4])

        plt.plot(residualPairList[0][0], residualPairList[0][1][0], 'bx', label='Crate 1')
        plt.plot(residualPairList[0][0], residualPairList[0][1][1], 'gx', label='Crate 2')
        plt.plot(residualPairList[0][0], residualPairList[0][1][2], 'yx', label='Crate 3')
        plt.plot(residualPairList[0][0], residualPairList[0][1][3], 'rx', label='Crate 4')
        plt.plot(residualPairList[0][0], residualPairList[0][1][4], 'ko', label='Crate 1-4')
        plt.plot([date[0], date[0]], [min_res, max_res], 'k-', label='Interval boundary')

    else:
        plt.plot(datetime, residualMeanC1, 'bx', label='Crate 1')
        plt.plot(datetime, residualMeanC2, 'gx', label='Crate 2')
        plt.plot(datetime, residualMeanC3, 'yx', label='Crate 3')
        plt.plot(datetime, residualMeanC4, 'rx', label='Crate 4')
        plt.plot(datetime, residualMean, 'ko', label='Crate 1-4')

    plt.title(drs_value_type+'residual Mean per Crate:')
    plt.ylabel('(f(t)-'+drs_value_type+r'Mean)/$\mathrm{mV}$')
    plt.gcf().autofmt_xdate()
    plt.xlim(min(date)-pd.DateOffset(days=7), max(date)+pd.DateOffset(days=7))
    plt.ylim(min_res-offset, max_res+offset)
    plt.grid()
    plt.legend(loc='lower left', numpoints=1)
    plt.text(0.02, 0.2, 'preliminary', fontdict=font, transform=plt.gca().transAxes)
    if(store_file_path is not None):
        plt.savefig(store_file_path)
    plt.show()
    plt.close()


####################################################################################################
def residualMeanPerPatchAndInterval(drs_file_path, residualFilenameArray_, drs_value_type, store_file_path=None):

    print("Loading '"+drs_value_type+"-data' ...")
    with h5py.File(drs_file_path, 'r') as store:
        time = np.array(store['Time'+drs_value_type]).flatten()
        date = pd.to_datetime(time * 24 * 3600 * 1e9).date

    residualMeanPerPatchAndInterval = []
    for residualFilename in residualFilenameArray_:
        intervalresidualMeanPerPatch = []
        with h5py.File(residualFilename, 'r') as residual_tab:
            interval_b = np.array(residual_tab['Interval'])

            interval = []
            for intervalLimit in interval_b:
                interval.append(pd.to_datetime(intervalLimit[0].decode('UTF-8')).date())

            intervalIndices = np.where((date >= interval[0]) & (date <= interval[1]))[0]

            residual = np.array(residual_tab['residual'+drs_value_type])[:, intervalIndices]
        for patchNr in range(NRPATCH):
            intervalresidualMeanPerPatch.append(
                np.mean(abs(
                            residual[patchNr*9*NRCELL:(patchNr+1)*9*NRCELL].flatten()
                            ), dtype='float64'))
        residualMeanPerPatchAndInterval.append(intervalresidualMeanPerPatch)

    residualMeanPerPatchAndInterval = np.array(residualMeanPerPatchAndInterval).transpose()

    plt.matshow(residualMeanPerPatchAndInterval, interpolation='None', aspect='auto')
    # plt.title(r'Mean of the absolute '+str(drs_value_type)+'residual-value \n per Interval ', fontsize=25, y=1.02)
    cbar = plt.colorbar()
    resMax = residualMeanPerPatchAndInterval.shape
    for x in range(resMax[1]+1):
        plt.plot([x-0.5, x-0.5], [-0.5, resMax[0]-0.5], 'k:')
    for y in range(4+1):
        y = y*40
        plt.plot([-0.5, resMax[1]-0.5], [y-0.5, y-0.5], 'k:')
    plt.xlabel('IntervalNr', fontsize=20)
    plt.ylabel('PatchNr', fontsize=20)
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
        timeLabel = pd.date_range(start=start_date, end=end_date, freq=str(intervalMonth)+'M')
        cbar = fig.colorbar(sc_f, ticks=dates.MonthLocator(interval=intervalMonth))
        cbar.ax.set_yticklabels(timeLabel.strftime('%b %Y'))
        # timeColor = cbar.to_rgba(time_list)

        plt.title(r'Calibrated PedestelRun '+calibrated_type, fontsize=16, y=1.02)
        plt.xlabel(r'Temperatur /$\mathrm{C\degree}$')
        plt.ylabel(ylabel)
        plt.legend(loc='upper right', numpoints=1, title='')
        plt.text(0.6, 0.15, 'preliminary', fontdict=font, transform=plt.gca().transAxes)
        if(store_file_path is not None):
            plt.savefig(store_file_path)
        plt.show()
        plt.close()
