import matplotlib.pyplot as plt
import matplotlib.dates as dates

import sys

import os
import numpy as np
import pandas as pd
import math
import h5py
import click
from zfits import FactFits
# import yaml

from tqdm import tqdm
from fact.instrument.camera import non_standard_pixel_chids as non_standard_chids
import fact.plotting as factPlots
from astropy.io import fits
from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import hot

NRCELL = 1024
NRCHID = 1440
ROI = 300
DACfactor = 2500/pow(2, 16)*50000

nrCellsPerChid = {'Baseline': NRCELL,
                  'Gain': NRCELL,
                  'ROIOffset': ROI}

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
@click.command()
@click.argument('list_of_needed_files_doc_path',
                default='/net/big-tank/POOL/' +
                        'projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_1/list_of_needed_files.txt',
                type=click.Path(exists=False))
@click.argument('source_folder_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/',
                type=click.Path(exists=False))
@click.argument('store_file_path',
                default='/home/fschulz/plots/version_1/' +
                        'hist_20141003_161.jpg',
                type=click.Path(exists=False))
###############################################################################
def chid_startcell_distribution(list_of_needed_files_doc_path, source_folder_path,
                                store_file_path):
    chid = 1439
    calibration_file_list = open(list_of_needed_files_doc_path).read().splitlines()
    value_sum = 0
    startcell_counter = np.zeros((NRCELL), dtype='uint32')
    for file_collection_of_the_day in tqdm(calibration_file_list[690:691]):
        file_collection_of_the_day = file_collection_of_the_day.split('|')
        #for run_serie in file_collection_of_the_day[1:]:
        run_serie = file_collection_of_the_day[1:][0]

        run_serie = run_serie.split(',')

        pedestal_run2_path = source_folder_path+run_serie[2]
        if not os.path.isfile(pedestal_run2_path):
            continue

        fits_stream = FactFits(pedestal_run2_path)
        print(pedestal_run2_path)
        for event in fits_stream:
            start_cell = event['StartCellData'][chid]
            if start_cell > 1024:
                print(start_cell)
            startcell_counter[start_cell] += 1
            value_sum += 1

    plt.title('    Chid: '+str(chid), fontsize=18, y=1.04)
    nr_of_empty_cells = len(startcell_counter[startcell_counter == 0])
    text = ('Number of empty cells: '+str(nr_of_empty_cells) +
            ' ('+str(round(nr_of_empty_cells/NRCELL*100, 2))+' %)')
    plt.text(0.025, 0.925, text, fontdict=font_black, transform=plt.gca().transAxes)
    range = np.arange(0, NRCELL, dtype='int16')
    print(startcell_counter, value_sum)
    plt.bar(range, startcell_counter*10000/value_sum,
            width=1, align='center', linewidth=0)
    plt.axhline(10000/NRCELL, color='r', label='mean')
    plt.xlim(0, NRCELL)
    plt.legend(loc='upper right', scatterpoints=1, numpoints=1)
    plt.xlabel('CELL')
    plt.ylabel(r'Frequency /'+u'\u2030')  # Unicode for \permil
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
                default='/home/fschulz/plots/version_0/drsValues/roiOffset/std_hist_outlier_dist_roi.png',
                type=click.Path(exists=False))
@click.argument('interval_nr',
                default=1)
@click.option('--drs_value_type', '-type',
              default='ROIOffset',
              type=click.Choice(['Baseline', 'Gain', 'ROIOffset']))
@click.argument('value_range',
                default=[2.4, 8.0]) # [2.25, 4.0], [2.0, 6.5]
###############################################################################
def drs_value_std_pixel_distribution(data_collection_path, interval_file_path,
                                     store_file_path, interval_nr,
                                     drs_value_type, value_range):

    drs_value_std_pixel_distribution_(data_collection_path, interval_file_path,
                                      store_file_path, interval_nr,
                                      drs_value_type, value_range)


###############################################################################
def drs_value_std_pixel_distribution_(data_collection_path, interval_file_path,
                                      store_file_path, interval_nr,
                                      drs_value_type, value_range):

    NRCELLSPERCHID = nrCellsPerChid[drs_value_type]

    chid_range = np.arange(0, NRCHID, dtype='int16')

    groupname = 'Interval'+str(interval_nr)
    print(groupname)
    with h5py.File(interval_file_path, 'r') as interval_source:
        data = interval_source[groupname]
        interval_indices = np.array(data['IntervalIndices'])
    print('loading')
    with h5py.File(data_collection_path, 'r') as store:
        drs_value_var = store[drs_value_type+'Var'][interval_indices, :]

    not_useful_chids = get_not_useful_chids(interval_nr)

    #drs_value_var = drs_value_var.reshape(-1, NRCHID, NRCELLSPERCHID)[:, useful_chids, :].flatten()
    drs_value_std = np.sqrt(drs_value_var)
    del drs_value_var

    if(drs_value_type == 'Gain'):
        drs_value_std /= DACfactor/pow(10, 3)

    print(value_range)
    selected_value_indices = np.where(np.logical_and(drs_value_std > value_range[0],
                                                     drs_value_std < value_range[1]))[1]

    selected_chids = (selected_value_indices/NRCELLSPERCHID).astype('int')

    counter = np.bincount(selected_chids)
    value_sum = len(selected_chids)

    chid_9 = len(np.where(selected_chids % 9 == 8)[0])

    plt.title('Verteilung der {}Std-Werte\nzwischen [{}, {}]\n Chid9: {} %'.format(drs_value_type, value_range[0], value_range[1], chid_9/value_sum*100),
              fontsize=17, y=1.00)
    plt.bar(chid_range, counter*10000/value_sum,
            width=1, align='center', linewidth=0)
    del drs_value_std

    for i in range(160):
        plt.axvline(i*9+8, color='g', linestyle='dashed')
    for chid in not_useful_chids:
        plt.axvline(chid, color='r', linestyle='dashed')
    plt.axhline(10000/NRCHID, color='r', label='Mean')
    plt.ylim(0, 10000/NRCHID*2)
    plt.xlim(0, NRCHID)
    plt.legend(loc='upper right', scatterpoints=1, numpoints=1)
    plt.xlabel('Chid')
    plt.ylabel(r'Frequency /'+u'\u2030')  # Unicode for \permil
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
                         'calibration/calculation/version_0/drsFitParameter_interval3.fits'],
                )
@click.argument('store_file_path',
                default='/home/fschulz/plots/version_0/model/gain/slope_hist_outlier_dist_gain.png',
                type=click.Path(exists=False))
@click.argument('interval_nr',
                default=3)
@click.option('--drs_value_type', '-type',
              default='Gain',
              type=click.Choice(['Baseline', 'Gain', 'ROIOffset']))
@click.option('--fit_parameter_type', '-type',
              default='Slope',
              type=click.Choice(['Slope', 'Offset']))
@click.argument('value_range',
                default=[-11, -0.3])
###############################################################################
def drs_model_pixel_distribution(drs_fit_file_path_array, store_file_path, interval_nr,
                                 drs_value_type, fit_parameter_type, value_range):

    drs_model_pixel_distribution_(drs_fit_file_path_array, store_file_path, interval_nr,
                                  drs_value_type, fit_parameter_type, value_range)


###############################################################################
def drs_model_pixel_distribution_(drs_fit_file_path_array, store_file_path, interval_nr,
                                  drs_value_type, fit_parameter_type, value_range):

    NRCELLSPERCHID = nrCellsPerChid[drs_value_type]

    chid_range = np.arange(0, NRCHID, dtype='int16')

    groupname = 'Interval'+str(interval_nr)
    print(groupname)

    fit_file = fits.open(drs_fit_file_path_array[interval_nr-1], ignoremissing=True, ignore_missing_end=True)
    interval_nr = int(fit_file[0].header['INTNR'])

    fit_value = fit_file['FitParameter'].data[drs_value_type+fit_parameter_type][0]
    print(fit_value.shape)
    if(drs_value_type == 'Gain'):
        if(fit_parameter_type == 'Slope'):
            fit_value *= pow(10, 3)

    selected_value_indices = np.where(np.logical_and(fit_value > value_range[0],
                                                     fit_value < value_range[1]))[0]

    chids = (selected_value_indices/NRCELLSPERCHID).astype('int')
    cells = (selected_value_indices % NRCELLSPERCHID)
    print(chids, cells)
    selected_chids = (selected_value_indices/NRCELLSPERCHID).astype('int')

    counter = np.bincount(selected_chids)
    print(counter[1415:-1])
    while len(counter) < NRCHID:
        counter = np.append(counter, 0)
    print(counter[1415:-1])

    value_sum = len(selected_chids)
    print(selected_chids.shape, counter.shape)

    chid_9 = len(np.where(selected_chids % 9 == 8)[0])
    print('Chid9: ', chid_9/value_sum*100, ' %')

    plt.title('Verteilung der {}Std-Werte\nzwischen [{}, {}]'.format(drs_value_type, value_range[0], value_range[1]),
              fontsize=17, y=1.00)
    print(len(chid_range), len(counter))
    plt.bar(chid_range, counter*10000/value_sum,
            width=1, align='center', linewidth=0)
    del fit_value

    for i in range(160):
        plt.axvline(i*9+8, color='r', linestyle='dashed')
    plt.axhline(10000/NRCHID, color='r', label='Mean')
    plt.ylim(0, 10000/NRCHID*2)
    plt.xlim(0, NRCHID)
    plt.legend(loc='upper right', scatterpoints=1, numpoints=1)
    plt.xlabel('Chid')
    plt.ylabel(r'Frequency /'+u'\u2030')  # Unicode for \permil
    plt.savefig(store_file_path)
    plt.show()
    plt.close()


###############################################################################
@click.command()
@click.argument('data_collection_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_1/dataCollection.h5',
                type=click.Path(exists=True))
@click.argument('interval_file_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_1/intervalIndices.h5',
                type=click.Path(exists=True))
@click.argument('store_file_path',
                default='/home/fschulz/plots/version_1/drsValues/gain/std_hist_outlier_dist_gain.png',
                type=click.Path(exists=False))
@click.argument('interval_nr',
                default=3)
@click.option('--drs_value_type', '-type',
              default='Gain',
              type=click.Choice(['Gain']))
@click.argument('value_range',
                default=[3.5, 4.4]) # [5.7, 7.0]
###############################################################################
def drs_value_std_pixel_distribution_v1(data_collection_path, interval_file_path,
                                        store_file_path, interval_nr,
                                        drs_value_type, value_range):

    drs_value_std_pixel_distribution_v1_(data_collection_path, interval_file_path,
                                         store_file_path, interval_nr,
                                         drs_value_type, value_range)


###############################################################################
def drs_value_std_pixel_distribution_v1_(data_collection_path, interval_file_path,
                                         store_file_path, interval_nr,
                                         drs_value_type, value_range):

    NRCELLSPERCHID = nrCellsPerChid[drs_value_type]

    chid_range = np.arange(0, NRCHID, dtype='int16')

    groupname = 'Interval'+str(interval_nr)
    print(groupname)
    with h5py.File(interval_file_path, 'r') as interval_source:
        data = interval_source[groupname]
        interval_indices = np.array(data['IntervalIndices'])
    print('loading')
    with h5py.File(data_collection_path, 'r') as store:
        drs_value_std = store[drs_value_type+'Std'][interval_indices, :]

    not_useful_chids = get_not_useful_chids(interval_nr)

    if(drs_value_type == 'Gain'):
        drs_value_std /= DACfactor/pow(10, 3)

    print(value_range)
    selected_value_indices = np.where(np.logical_and(drs_value_std > value_range[0],
                                                     drs_value_std < value_range[1]))[1]

    selected_chids = (selected_value_indices/NRCELLSPERCHID).astype('int')

    counter = np.bincount(selected_chids)
    value_sum = len(selected_chids)

    chid_9 = len(np.where(selected_chids % 9 == 8)[0])

    plt.title('Verteilung der {}Std-Werte\nzwischen [{}, {}]\n Chid9: {} %'.format(drs_value_type, value_range[0], value_range[1], chid_9/value_sum*100),
              fontsize=17, y=1.00)
    plt.bar(chid_range, counter*10000/value_sum,
            width=1, align='center', linewidth=0)
    del drs_value_std

    for i in range(160):
        plt.axvline(i*9+8, color='g', linestyle='dashed')
    for chid in not_useful_chids:
        plt.axvline(chid, color='r', linestyle='dashed')
    plt.axhline(10000/NRCHID, color='r', label='Mean')
    plt.ylim(0, 10000/NRCHID*2)
    plt.xlim(0, NRCHID)
    plt.legend(loc='upper right', scatterpoints=1, numpoints=1)
    plt.xlabel('Chid')
    plt.ylabel(r'Frequency /'+u'\u2030')  # Unicode for \permil
    plt.savefig(store_file_path)
    plt.show()
    plt.close()
