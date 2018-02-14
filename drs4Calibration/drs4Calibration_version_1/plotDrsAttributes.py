import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import sys

import os
import numpy as np
import pandas as pd
import math
import h5py
import click

import gc

from zfits import FactFits
from tqdm import tqdm
from fact.instrument.camera import non_standard_pixel_chids as non_standard_chids
import fact.plotting as factPlots
from astropy.io import fits
from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as colors
from matplotlib.cm import hot, seismic

import config as config
from constants import NRCHID, NRCELL, ROI, PEAFACTOR, DACfactor, ADCCOUNTSTOMILIVOLT

###############################################################################
# ##############                    Helper                     ############## #
###############################################################################

interval_color = ['k', 'royalblue', 'orange', 'forestgreen']

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
        not_useful_chids = np.append(not_useful_chids, np.arange(1080, 1439+1))
    return np.unique(not_useful_chids)


###############################################################################
def get_useful_chids(interval_nr, bad_chids=[]):

    chid_list = np.linspace(0, NRCHID-1, NRCHID, dtype='int')

    not_useful_chids = get_not_useful_chids(interval_nr)
    not_useful_chids = np.unique(np.append(not_useful_chids, bad_chids)).astype('int')
    # print(np.sort(not_useful_chids))
    useful_chids = chid_list[np.setdiff1d(chid_list, not_useful_chids)]
    return useful_chids


###############################################################################
# See https://matplotlib.org/users/colormapnorms.html
###############################################################################
class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


###############################################################################
# ##############               Drs-Value  Plots                ############## #
###############################################################################

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
                default='/home/fschulz/plots/version_1/drsValues/gain/std_hist_gain.png',
                type=click.Path(exists=False))
@click.argument('interval_array',
                default=[3])
@click.option('--drs_value_type', '-type',
              default='Gain',
              type=click.Choice(['Baseline', 'Gain']))
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
            drs_value_std = store[drs_value_type+'Std'][interval_indices, :].astype('float32')

        useful_chids = get_useful_chids(interval_nr)
        drs_value_std = drs_value_std.reshape(-1, NRCHID, NRCELLSPERCHID)[:, useful_chids, :].flatten()

        if(drs_value_type == 'Gain'):
            drs_value_std /= pow(10, 3)
            factor_str = r' $\cdot$'
            unit_str = r' $10^{-3}$'

        drs_value_std_mean = np.mean(drs_value_std)
        drs_value_std_std = np.std(drs_value_std)
        drs_value_std_max = max(drs_value_std)
        drs_value_std_min = min(drs_value_std)

        color = 'g'
        weights = np.full(len(drs_value_std), 100/len(drs_value_std))
        nr_bins = int(abs((drs_value_std_max-drs_value_std_min))/abs(x_lim[1]-x_lim[0])*100)
        plt.hist(drs_value_std, bins=nr_bins, weights=weights,
                 histtype='step', label=title_str+':', color=color)

        info_str = (r' $\overline{x}$: '+'({:0.2f}'.format(drs_value_std_mean) +
                    ' $\pm$ '+'{:0.2f})'.format(drs_value_std_std)+factor_str+unit_str +
                    '\n'+r' $x_\mathrm{Max}$: '+'{:0.2f}'.format(drs_value_std_max)+factor_str+unit_str)
        plt.plot([], [], label=info_str)
        del drs_value_std
        gc.collect()

    plt.xlabel(r'Standardabweichung /'+unit_str)
    plt.ylabel(r'Häufigkeit /$\mathrm{\%}$')
    plt.xlim(x_lim)
    plt.legend(loc='upper right')
    plt.tight_layout()
    if(store_file_path is not None):
        plt.savefig(store_file_path, dpi=200)
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
                default='/home/fschulz/plots/version_1/drsValues/gain/std_hist_chid250_cell250_gain.png',
                type=click.Path(exists=False))
@click.argument('interval_array',
                default=[3])
@click.option('--drs_value_type', '-type',
              default='Gain',
              type=click.Choice(['Baseline', 'Gain']))
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
    unit_str = r'$\mathrm{mV}$'
    for interval_nr in interval_array:
        groupname = 'Interval'+str(interval_nr)
        title_str = 'Intervall '+str(interval_nr)
        print(groupname)
        with h5py.File(interval_file_path, 'r') as interval_source:
            data = interval_source[groupname]
            interval_indices = np.array(data['IntervalIndices'])
        print('loading')
        with h5py.File(data_collection_path, 'r') as store:
            drs_value_std = store[drs_value_type+'Std'][interval_indices, value_index].astype('float32')

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
                 '\n'+r' $\overline{x}$: '+str(round(drs_value_std_mean, 2))+factor_str+unit_str +
                 '\n'+r' $\sigma_\mathrm{Hist}$: '+str(round(drs_value_std_std, 2))+factor_str+unit_str +
                 '\n'+r' $x_\mathrm{Max}$: '+str(round(drs_value_std_max, 2)))+factor_str+unit_str+'\n'
        weights = np.full(len(drs_value_std), 100/len(drs_value_std))
        nr_bins = int(abs((drs_value_std_max-drs_value_std_min))/abs(x_lim[1]-x_lim[0])*100)
        plt.hist(drs_value_std, bins=nr_bins, weights=weights,
                 histtype='step', label=label, color=color)

        plt.axvline(x=drs_value_std_limit, linewidth=2, ls='--', color=color)

    plt.xlabel(r'Standardabweichung / '+unit_str)
    plt.ylabel(r'Häufigkeit / $\mathrm{\%}$')
    plt.xlim(x_lim)
    plt.legend(loc='upper right')
    if(store_file_path is not None):
        plt.savefig(store_file_path, dpi=200)
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
@click.argument('fit_file_path_array',
                default=['/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                         'calibration/calculation/version_1/drsFitParameter_interval3.fits'],
                )
@click.argument('store_file_path',
                default='/home/fschulz/plots/version_1/drsValues/gain/chid1101_cell240_interval3.jpg',
                type=click.Path(exists=False))
@click.argument('interval_array',
                default=[3])
@click.option('--drs_value_type', '-type',
              default='Gain',
              type=click.Choice(['Baseline', 'Gain']))
@click.argument('chid',
                default=1101)
@click.argument('cell',
                default=240)
@click.argument('ylimits',
                default=[])
###############################################################################
def drs_value_cell(data_collection_path, interval_file_path, fit_file_path_array,
                   store_file_path, interval_array, drs_value_type,
                   chid, cell, ylimits):

    drs_value_cell_(data_collection_path, interval_file_path, fit_file_path_array,
                    store_file_path, interval_array, drs_value_type,
                    chid, cell, ylimits)


###############################################################################
def drs_value_cell_(data_collection_path, interval_file_path, fit_file_path_array,
                    store_file_path, interval_array, drs_value_type,
                    chid, cell, ylimits=[]):

    sampleID = 200
    NRCELLSPERCHID = config.nrCellsPerChid[drs_value_type]
    if(cell > NRCELLSPERCHID):
        print('ERROR: cell > '+str(NRCELLSPERCHID))
        return

    value_index = chid*NRCELLSPERCHID + cell
    value_index_ = value_index

    # loading source data
    with h5py.File(data_collection_path, 'r') as store:
        time = np.array(store['Time'+drs_value_type]).flatten()

    ylabel_str = drs_value_type+r' / $\mathrm{mV}$'
    slope_unit_str = r'$\,\frac{\mathrm{mV}}{°C}$'
    offset_unit_str = r'$\,\mathrm{mV}$'
    if(drs_value_type == 'Gain'):
        ylabel_str = drs_value_type+r' / $10^{-3}$'
        slope_unit_str = r'$\,\frac{10^{-6}}{°C}$'
        offset_unit_str = r'$\cdot 10^{-3}$'

    use_mask = True
    if(drs_value_type == 'Baseline'):
        value_index_ = chid*NRCELLSPERCHID*ROI + cell*ROI+sampleID
        use_mask = False
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
            drs_value = np.array(store[drs_value_type][interval_indices, value_index_]).astype('float32')

        with fits.open(fit_file_path_array[0], ignoremissing=True, ignore_missing_end=True) as fit_value_tab:
            data = fit_value_tab['FitParameter'].data
            slope = data[drs_value_type+'Slope'][value_index]
            offset = data[drs_value_type+'Offset'][value_index]

        time_interval = time[interval_indices]

        if(drs_value_type == 'Baseline'):
            time_interval = time_interval[drs_value != 0]
            temp = temp[drs_value != 0]
            drs_value = drs_value[drs_value != 0]*ADCCOUNTSTOMILIVOLT
            slope = slope[sampleID]*ADCCOUNTSTOMILIVOLT
            offset = offset[sampleID]*ADCCOUNTSTOMILIVOLT

        print(np.mean(drs_value), ", min: ", min(drs_value), ", max: ", max(drs_value))
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
        label_str = (r'm={: 0.2f}'.format(slope) + slope_unit_str + ', ' +
                     r'b={:0.2f}'.format(offset) + offset_unit_str)
        fit_plot, = plt.plot(temp_range, fit, '-', color=color_mean,
                             label=label_str)
        fit_plots.append(fit_plot)
    plt.legend(handles=fit_plots, loc='lower center', title=r'Linearer Fit: f(x) = m $\cdot$ x + b')

    plt.xlabel(r'Temperatur / $\mathrm{^\circ C}$')
    plt.ylabel(ylabel_str)
    plt.xlim(min(temp_list)-1, max(temp_list)+1)

    if (len(ylimits) == 0):
        ymin, ymax = get_best_limits(drs_value, 0.36)
        plt.ylim(ymin*0.9994, ymax)
    else:
        plt.ylim(ylimits)

    plt.grid()
    plt.tight_layout()
    if(store_file_path is not None):
        plt.savefig(store_file_path, dpi=200)
    plt.show()
    plt.close()


###############################################################################
@click.command()
@click.argument('drs_fit_file_path_array',
                default=['/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                         'calibration/calculation/version_1/drsFitParameter_interval3.fits'])
@click.argument('store_file_path',
                default='/home/fschulz/plots/version_1/model/gain/gain_slope_hist.png',
                type=click.Path(exists=False))
@click.option('--drs_value_type', '-type',
              default='Gain',
              type=click.Choice(['Baseline', 'Gain']))
@click.option('--fit_parameter_type', '-type',
              default='Slope',
              type=click.Choice(['Slope', 'Offset']))
@click.argument('x_lim',
                default=[-1, 1])
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

    factor_str = '$\,$'
    unit_str = r'$\mathrm{mV}$'
    unit_y_label_str = unit_str
    for fit_file_path in drs_fit_file_path_array:
        fit_file = fits.open(fit_file_path, ignoremissing=True, ignore_missing_end=True)
        interval_nr = int(fit_file[0].header['INTNR'])

        fit_value = fit_file['FitParameter'].data[drs_value_type+fit_parameter_type]

        useful_chids = get_useful_chids(interval_nr)
        if(drs_value_type == 'Baseline'):
            fit_value = fit_value*ADCCOUNTSTOMILIVOLT
            fit_value = fit_value.reshape(NRCHID, NRCELL, ROI)[useful_chids, :, :].flatten()
        elif(drs_value_type == 'Gain'):
            fit_value = fit_value.reshape(NRCHID, NRCELL)[useful_chids, :].flatten()
            if(fit_parameter_type == 'Slope'):
                fit_value *= pow(10, 3)
                factor_str = r'$\cdot$'
                unit_str = r'$10^{-3}$'
                unit_y_label_str = unit_str
            else:
                factor_str = ''
                unit_str = ''
                unit_y_label_str = '1'

        fit_value_mean = np.mean(fit_value)
        fit_value_std = np.std(fit_value)
        fit_value_min = min(fit_value)
        fit_value_max = max(fit_value)

        color = interval_color[interval_nr]
        title_str = 'Intervall '+str(interval_nr)+':'
        weights = np.full(len(fit_value.flatten()), 100/len(fit_value.flatten()))
        nr_bins = int(abs((fit_value_max-fit_value_min))/abs(x_lim[1]-x_lim[0])*100)
        plt.hist(fit_value, bins=nr_bins, weights=weights,
                 histtype='step', label=title_str, color=color)

        info_str = (r' $\overline{x}$: '+'({:0.2f}'.format(fit_value_mean) +
                    ' $\pm$ '+'{:0.2f})'.format(fit_value_std)+factor_str+unit_str +
                    '\n'+r' $x_\mathrm{Min}$: '+'{:0.2f}'.format(fit_value_min)+factor_str+unit_str +
                    '\n'+r' $x_\mathrm{Max}$: '+'{:0.2f}'.format(fit_value_max)+factor_str+unit_str)
        plt.plot([], [], label=info_str)
        del fit_value
        gc.collect()

    x_label_str = 'Steigung m'
    if(fit_parameter_type == 'Offset'):
        x_label_str = 'Y-Achsenabschnitt b'

    plt.xlabel(x_label_str+r' / '+unit_y_label_str)
    plt.ylabel(r'Häufigkeit / $\mathrm{\%}$')
    plt.xlim(x_lim)

    # legendstuff BEG
    handles, labels = plt.gca().get_legend_handles_labels()
    order = []
    intervals = int(len(handles)/2)
    for i in range(intervals):
        order.append(intervals+i)
        order.append(i)
    legend = plt.legend([handles[idx] for idx in order],
                        [labels[idx] for idx in order],
                        loc='upper right')
    for item in legend.legendHandles[1::2]:
        item.set_visible(False)
    # legendstuff END
    plt.tight_layout()
    if(store_file_path is not None):
        plt.savefig(store_file_path, dpi=200)
    #plt.show()
    plt.close()


###############################################################################
@click.command()
@click.argument('drs_fit_file_path_array',
                default=['/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                         'calibration/calculation/version_1/drsFitParameter_interval3.fits'])
@click.argument('store_file_path',
                default='/home/fschulz/plots/version_1/model/baseline/residual_hist_baseline.png',
                type=click.Path(exists=False))
@click.option('--drs_value_type', '-type',
              default='Baseline',
              type=click.Choice(['Baseline', 'Gain']))
###############################################################################
def drs_fit_value_residual_hist(drs_fit_file_path_array, store_file_path,
                                drs_value_type, x_lim):

    drs_fit_value_residual_hist_(drs_fit_file_path_array, store_file_path,
                                 drs_value_type, x_lim)


###############################################################################
def drs_fit_value_residual_hist_(drs_fit_file_path_array, store_file_path,
                                 drs_value_type, x_lim):

    factor_str = ''
    unit_str = r' $\mathrm{mV}$'
    for fit_file_path in drs_fit_file_path_array:
        fit_file = fits.open(fit_file_path, ignoremissing=True, ignore_missing_end=True)
        interval_nr = int(fit_file[0].header['INTNR'])

        useful_chids = get_useful_chids(interval_nr)

        res_value = fit_file['FitParameter'].data[drs_value_type+'Residual']
        if(drs_value_type == 'Baseline'):
            res_value = res_value*ADCCOUNTSTOMILIVOLT
            res_value_ = res_value.reshape(NRCHID, NRCELL, ROI)[useful_chids, :, :].flatten()
        elif(drs_value_type == 'Gain'):
            res_value *= pow(10, 3)
            factor_str = r' $\cdot$'
            unit_str = r' $10^{-3}$'

            res_value_ = res_value.reshape(NRCHID, NRCELL)[useful_chids, :].flatten()

        res_value_mean = np.mean(res_value_)
        res_value_std = np.std(res_value_)
        res_value_min = 0
        res_value_max = max(res_value_)

        #get default green(3rd color) for interval 3
        plt.plot([], [])
        plt.plot([], [])
        plot = plt.plot([], [])
        color = plot[0].get_color()
        title_str = 'Intervall '+str(interval_nr)
        label = (title_str+':' +
                 '\n'+r' $\overline{x}$: '+str(round(res_value_mean, 2))+factor_str+unit_str +
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
        plt.savefig(store_file_path, dpi=200)
    #plt.show()
    plt.close()


###############################################################################
###############################################################################
@click.command()
@click.argument('fit_file_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_1/drsFitParameter_interval1.fits')
@click.argument('store_file_path',
                default='/home/fschulz/plots/version_1/model/gain/fact_cam_residual_int2_gain.jpg',
                type=click.Path(exists=False))
@click.option('--drs_value_type', '-type',
              default='Baseline',
              type=click.Choice(['Baseline', 'Gain']))
@click.argument('worst_chids',
                default=[])
###############################################################################
def residual_fact_cam(fit_file_path, store_file_path,
                      drs_value_type, worst_chids):

    residual_fact_cam_(fit_file_path, store_file_path,
                       drs_value_type, worst_chids)


###############################################################################
def residual_fact_cam_(fit_file_path, store_file_path,
                       drs_value_type, worst_chids=[]):

    with fits.open(fit_file_path, ignoremissing=True, ignore_missing_end=True) as fit_file:
        interval_nr = int(fit_file[0].header['INTNR'])
        print('interval_nr: ', interval_nr)

        res_value = fit_file['FitParameter'].data[drs_value_type+'Residual']

    unit_str = r'$\mathrm{mV}$'
    if(drs_value_type == 'Baseline'):
        res_value = res_value*ADCCOUNTSTOMILIVOLT
        res_value = res_value.reshape(NRCHID, NRCELL*ROI)
    elif(drs_value_type == 'Gain'):
        res_value *= pow(10, 3)
        unit_str = r'$10^{-3}$'
        res_value = res_value.reshape(NRCHID, NRCELL)

    residual_per_chid = np.mean(res_value, axis=1)

    not_useful_chids = get_not_useful_chids(interval_nr)
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

    plt.colorbar(plot, label=r'Mittelwert des Betrags der Residuen eines Pixels / '+unit_str)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.tight_layout()
    if(store_file_path is not None):
        plt.savefig(store_file_path, dpi=200)
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
@click.argument('fit_file_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_1/drsFitParameter_interval3.fits',
                type=click.Path(exists=True))
@click.argument('store_file_path',
                default=('/home/fschulz/plots/version_1/residual/' +
                         'gain/worst_100_cell_collection_I3.pdf'),
                type=click.Path(exists=False))
@click.argument('interval_nr',
                default=3)
@click.option('--drs_value_type', '-type',
              default='Gain',
              type=click.Choice(['Baseline', 'Gain']))
@click.argument('worse_chids',
                default=[])
@click.argument('selected_values',
                default=100)
###############################################################################
def worst_cell_collection_selected_by_residual(
                                    data_collection_path, interval_file_path,
                                    fit_file_path, store_file_path,
                                    interval_nr, drs_value_type,
                                    worse_chids, selected_values):

    worst_cell_collection_selected_by_residual_(
                                    data_collection_path, interval_file_path,
                                    fit_file_path, store_file_path,
                                    interval_nr, drs_value_type,
                                    worse_chids, selected_values)


###############################################################################
def worst_cell_collection_selected_by_residual_(
                                    data_collection_path, interval_file_path,
                                    fit_file_path, store_file_path,
                                    interval_nr, drs_value_type,
                                    worse_chids, selected_values=100):

    NRCELLSPERCHID = NRCELL #config.nrCellsPerChid[drs_value_type]

    groupname = 'Interval'+str(interval_nr)

    interval_source = h5py.File(interval_file_path, 'r')[groupname]
    # cut_off_error_factor = interval_source.attrs['CutOff'+drs_value_type]
    interval_indices = np.array(interval_source['IntervalIndices'])

    chid_list = np.linspace(0, NRCHID-1, NRCHID, dtype='int')
    not_useful_chids = np.array([non_standard_chids['crazy'],
                                 non_standard_chids['dead']]).flatten()

    not_useful_chids = np.unique(np.sort(np.append(not_useful_chids, worse_chids)))

    useful_chids = chid_list[np.setdiff1d(chid_list, not_useful_chids)]
    fit_value_tab = fits.open(fit_file_path, ignoremissing=True, ignore_missing_end=True)[1].data

    residual_array = fit_value_tab[drs_value_type+'Residual'].reshape(NRCHID, NRCELLSPERCHID)[useful_chids, :].flatten()

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
            with h5py.File(data_collection_path, 'r') as store:
                time = np.array(store['Time'+drs_value_type][interval_indices, :]).flatten()
                temp = store['Temp'+drs_value_type][interval_indices, int(chid/9)]
                drs_value = store[drs_value_type][interval_indices, value_index]

            if(drs_value_type == 'Gain'):
                drs_value = drs_value.astype('float64')/DACfactor

            sc_all = plt.scatter(temp, drs_value, c=np.array(time))
            plt.close()  # Dont show sc_all, just use it to create the colorbar

            fig, img = plt.subplots()

            intervalMonth = 3
            start_date = pd.to_datetime(time[0] * 24 * 3600 * 1e9).date()
            end_date = pd.to_datetime(time[-1] * 24 * 3600 * 1e9).date()
            timeLabel = pd.date_range(start=start_date, end=end_date, freq=str(intervalMonth)+'M')
            cbar = fig.colorbar(sc_all, ticks=mdates.MonthLocator(interval=intervalMonth))
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
                        'drs4_calibration_data/calibration/validating/version_1/' +
                        'meanAndStd/interval3/',
                type=click.Path(exists=True))
@click.argument('store_file_path',
                default='/home/fschulz/plots/version_1/meanAndStd/interval3/' +
                        'calibratedPedestelDataDistribution_mean.png', #_2016-04-01
                type=click.Path(exists=False))
@click.option('--calibrated_type', '-type',
              default='Mean',
              type=click.Choice(['Mean', 'Std']))
@click.argument('interval_nr',
                default=3)
@click.argument('datetime_limits_string',
                default=['2015-05-26', '2017-10-01'])
@click.argument('month_lim',
                default=[3, 1])
@click.argument('ylim',
                default=0.17)
###############################################################################
def pedestel_mean_or_std_vs_temp(data_folder_path, store_file_path,
                                 calibrated_type, interval_nr,
                                 datetime_limits_string, month_lim,
                                 ylim):

    pedestel_mean_or_std_vs_temp_(data_folder_path, store_file_path,
                                  calibrated_type, interval_nr,
                                  datetime_limits_string, month_lim,
                                  ylim)


###############################################################################
def pedestel_mean_or_std_vs_temp_(data_folder_path, store_file_path,
                                  calibrated_type, interval_nr,
                                  datetime_limits_string, month_lim,
                                  ylim):

    in_pea = True
    noise_file_list = sorted([file for file in
                              os.listdir(data_folder_path)
                              if (file.startswith('pedestelStats') and
                                  file.endswith('_.fits'))])

    useful_chids = get_useful_chids(interval_nr)

    datetime_limits = [pd.to_datetime(datetime_limits_string[0]),
                       pd.to_datetime(datetime_limits_string[1])]

    counter = 0  # number of calibrated pedestel-runs
    # helper_key_list = []  # store
    helper_key_list = open('pedestel_std_vs_temp_helper_v0_v1.txt').read().splitlines()

    temp_diff_list = []
    time_list = []
    drs_file_calibrated_mean_list = []
    drs_model_calibrated_mean_list = []
    for noise_file_path in tqdm(noise_file_list):
        datetime = pd.to_datetime(noise_file_path.split('_')[-2].split('.')[0])
        #datetime = pd.to_datetime(noise_file_path.split('_')[-1].split('.')[0])
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

            key = datetime.strftime('%Y%m%d')+'_{:0.5f}'.format(abs(temp_diff[run_index]))
            if(key not in helper_key_list):
                print('Key not found: ', key)
                continue

            if(temp_diff[run_index] > 5):
                print(noise_file_path, run_index+1, ', temp')
                continue

            if (calibrated_type == "Std" and drs_file_calibrated_mean > 0.6):
                print(noise_file_path, run_index+1, ', std')
                continue

            counter += 1
            # helper_key_list.append(key)

            time_list.append(datetime)
            temp_diff_list.append(temp_diff[run_index])
            drs_file_calibrated_mean_list.append(drs_file_calibrated_mean)
            drs_model_calibrated_mean_list.append(drs_model_calibrated_mean)

    # pd.DataFrame(helper_key_list).to_csv('pedestel_std_vs_temp_helper_v1.txt', index=False, header=False)

    print('number of calibrated pedestel-runs: ', counter)
    drs_file_calibrated_collection_mean = np.mean(drs_file_calibrated_mean_list)
    drs_model_calibrated_collection_mean = np.mean(drs_model_calibrated_mean_list)

    drs_file_calibrated_collection_std = np.std(drs_file_calibrated_mean_list, dtype="float64", ddof=1)
    drs_model_calibrated_collection_std = np.std(drs_model_calibrated_mean_list, dtype="float64", ddof=1)

    unit_str = r'$\mathrm{mV}$'
    calibrated_type_str = 'Mittelwert'
    if(calibrated_type == 'Std'):
        calibrated_type_str = 'Standardabweichung'

    if(in_pea):
        unit_str = r'Spe'
    ylabel = calibrated_type_str+' / '+unit_str

    sc_all = plt.scatter(time_list, drs_model_calibrated_mean_list, c=temp_diff_list,
                         cmap=seismic, norm=MidpointNormalize(midpoint=0.))
    plt.close()  # Dont show sc_all, just use it to create the colorbar

    gs = gridspec.GridSpec(1, 11)
    fig = plt.figure(figsize=(10, 7), dpi=100)

    ax0 = plt.subplot(gs[0, 0:5])
    ax1 = plt.subplot(gs[0, 5:11], sharey=ax0)
    fig.subplots_adjust(wspace=0.5)
    ax0.tick_params(labelsize=12)
    ax1.tick_params(labelleft='off', labelsize=12)

    ax0.scatter(time_list, drs_file_calibrated_mean_list,
                s=50, lw=0.5, edgecolors='gray', marker=".", alpha=0.75,
                c=temp_diff_list, cmap=seismic, norm=MidpointNormalize(midpoint=0.))
    ax1.scatter(time_list, drs_model_calibrated_mean_list,
                s=50, lw=0.5, edgecolors='gray', marker=".", alpha=0.75,
                c=temp_diff_list, cmap=seismic, norm=MidpointNormalize(midpoint=0.))

    cbar = fig.colorbar(sc_all)
    cbar.set_label(r'Pedestal-Run Temperatur - DRS-Run Temperatur / $\mathrm{C\degree}$', fontsize=12)
    cbar.ax.tick_params(labelsize=14)

    ax0.xaxis.set_major_locator(mdates.MonthLocator(interval=month_lim[0]))
    ax0.xaxis.set_minor_locator(mdates.MonthLocator(interval=month_lim[1]))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=month_lim[0]))
    ax1.xaxis.set_minor_locator(mdates.MonthLocator(interval=month_lim[1]))

    ax0.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    ax0.fmt_xdata = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
    ax1.fmt_xdata = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
    fig.autofmt_xdate()

    font = {'family': 'serif',
            'color': 'black',
            'size': 16,
            'alpha': 0.85
            }

    print(calibrated_type)
    print('file: ', drs_file_calibrated_collection_mean)
    print('model: ', drs_model_calibrated_collection_mean)

    label_str = (r' $\overline{x}$: '+'({:0.3f}'.format(drs_file_calibrated_collection_mean) +
                 r' $\pm$ {:0.2f})'.format(drs_file_calibrated_collection_std)+r'$\,$'+unit_str)
    ax0.text(0.5, 0.06, label_str, fontdict=font, transform=ax0.transAxes,
             bbox=dict(facecolor='none', edgecolor='gray', boxstyle='round, pad=0.5'),
             ha='center', va='center')
    label_str = (r' $\overline{x}$: '+'({:0.3f}'.format(drs_model_calibrated_collection_mean) +
                 r' $\pm$ {:0.2f})'.format(drs_model_calibrated_collection_std)+r'$\,$'+unit_str)
    ax1.text(0.5, 0.06, label_str, fontdict=font, transform=ax1.transAxes,
             bbox=dict(facecolor='none', edgecolor='gray', boxstyle='round, pad=0.5'),
             ha='center', va='center')
    ax0.set_title('DRS-File', fontsize=18, y=0.92)
    ax0.yaxis.grid(linestyle=':')
    ax0.set_ylabel(ylabel, fontsize=13)
    ax0.set_ylim(ylim,)
    ax1.set_title('Modell', fontsize=18, y=0.92)
    ax1.set_ylim(ylim,)
    ax1.yaxis.grid(linestyle=':')

    plt.tight_layout()
    if(store_file_path is not None):
        plt.savefig(store_file_path, dpi=200)
    #plt.show()
    plt.close()


###############################################################################
###############################################################################
@click.command()
@click.argument('data_folder_path',
                default='/net/big-tank/POOL/projects/fact/' +
                        'drs4_calibration_data/calibration/validating/version_1/' +
                        'meanAndStd/interval3/',
                type=click.Path(exists=True))
@click.argument('store_file_path',
                default='/home/fschulz/plots/version_1/meanAndStd/interval3/' +
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
def pedestel_mean_or_std_fact_cam(data_folder_path, store_file_path,
                                  calibrated_type, interval_nr,
                                  datetime_limits_string):

    pedestel_mean_or_std_fact_cam_(data_folder_path, store_file_path,
                                   calibrated_type, interval_nr,
                                   datetime_limits_string)


###############################################################################
def pedestel_mean_or_std_fact_cam_(data_folder_path, store_file_path,
                                   calibrated_type, interval_nr,
                                   datetime_limits_string):

    in_pea = True
    noise_file_list = sorted([file for file in
                              os.listdir(data_folder_path)
                              if (file.startswith('pedestelStats') and
                                  file.endswith('.fits'))])

    useful_chids = get_useful_chids(interval_nr)

    datetime_limits = [pd.to_datetime(datetime_limits_string[0]),
                       pd.to_datetime(datetime_limits_string[1])]

    counter = 0  # number of calibrated pedestel-runs
    # helper_key_list = []  # store
    helper_key_list = open('pedestel_std_vs_temp_helper_v0_v1.txt').read().splitlines()

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
            drs_file_calibrated = np.array(drs_file_calibrated_of_the_day[run_index]).reshape(-1, NRCHID)
            drs_model_calibrated = np.array(drs_model_calibrated_of_the_day[run_index]).reshape(-1, NRCHID)

            # calculate the mean over all events(1000)
            drs_file_calibrated_mean = np.mean(drs_file_calibrated, axis=0)
            drs_model_calibrated_mean = np.mean(drs_model_calibrated, axis=0)

            if(in_pea):
                drs_file_calibrated_mean = drs_file_calibrated_mean*PEAFACTOR
                drs_model_calibrated_mean = drs_model_calibrated_mean*PEAFACTOR

            key = datetime.strftime('%Y%m%d')+'_{:0.5f}'.format(temp_diff[run_index])
            if(key not in helper_key_list):
                print('not found: ', key)
                continue

            counter += 1
            drs_file_calibrated_mean_list.append(drs_file_calibrated_mean)
            drs_model_calibrated_mean_list.append(drs_model_calibrated_mean)

    print('number of calibrated pedestel-runs: ', counter)
    drs_file_calibrated_collection_mean = np.mean(drs_file_calibrated_mean_list, axis=0)
    drs_model_calibrated_collection_mean = np.mean(drs_model_calibrated_mean_list, axis=0)

    drs_file_calibrated_collection_std = np.std(drs_file_calibrated_mean_list, axis=0, dtype="float64", ddof=1)
    drs_model_calibrated_collection_std = np.std(drs_model_calibrated_mean_list, axis=0, dtype="float64", ddof=1)

    not_useful_chids = get_not_useful_chids(interval_nr)
    plot = factPlots.camera(drs_file_calibrated_collection_mean, cmap='hot')
    factPlots.mark_pixel(not_useful_chids, color='darkgreen', linewidth=2)
    plt.colorbar(plot, label=r'drs_file_calibrated_collection_mean: {}'.format(np.mean(drs_file_calibrated_collection_mean)))
    #factPlots.pixelids(size=10)
    plt.show()
    plt.close()
    plot = factPlots.camera(drs_model_calibrated_collection_mean, cmap='hot')
    factPlots.mark_pixel(not_useful_chids, color='darkgreen', linewidth=2)
    plt.colorbar(plot, label=r'drs_model_calibrated_collection_mean: {}'.format(np.mean(drs_model_calibrated_collection_mean)))
    plt.show()
    plt.close()
    plot = factPlots.camera(drs_file_calibrated_collection_std, cmap='hot')
    factPlots.mark_pixel(not_useful_chids, color='darkgreen', linewidth=2)
    plt.colorbar(plot, label=r'drs_file_calibrated_collection_std: {}'.format(np.mean(drs_file_calibrated_collection_std)))
    plt.show()
    plt.close()
    plot = factPlots.camera(drs_model_calibrated_collection_std, cmap='hot')
    factPlots.mark_pixel(not_useful_chids, color='darkgreen', linewidth=2)
    plt.colorbar(plot, label=r'drs_model_calibrated_collection_std: {}'.format(np.mean(drs_model_calibrated_collection_std)))
    plt.show()
    plt.close()
    plt.tight_layout()
    # if(store_file_path is not None):
    #     plt.savefig(store_file_path, dpi=200)
    plt.show()
    plt.close()


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
    plt.savefig(store_file_path, dpi=200)
    plt.show()
    plt.close()


###############################################################################
@click.command()
@click.argument('data_folder_path',
                default='/net/big-tank/POOL/projects/fact/' +
                        'drs4_calibration_data/calibration/validating/version_1/' +
                        'meanAndStd/outside/',
                type=click.Path(exists=True))
@click.argument('store_file_path',
                default='/home/fschulz/plots/version_1/meanAndStd/outside/' +
                        'calibratedPedestelDataDistribution_mean.png',
                type=click.Path(exists=False))
@click.option('--calibrated_type', '-type',
              default='Mean',
              type=click.Choice(['Mean', 'Std']))
@click.argument('interval_nr',
                default=3)
@click.argument('datetime_limits_string',
                default=['2018-01-01', '2018-01-10'])
###############################################################################
def pedestel_mean_or_std_vs_temp_outside(data_folder_path, store_file_path,
                                         calibrated_type, interval_nr,
                                         datetime_limits_string):

    pedestel_mean_or_std_vs_temp_outside_(data_folder_path, store_file_path,
                                          calibrated_type, interval_nr,
                                          datetime_limits_string)


###############################################################################
def pedestel_mean_or_std_vs_temp_outside_(data_folder_path, store_file_path,
                                          calibrated_type, interval_nr,
                                          datetime_limits_string):

    in_pea = True
    noise_file_list = sorted([file for file in
                              os.listdir(data_folder_path)
                              if (file.startswith('pedestelStats') and
                                  file.endswith('.fits'))])

    useful_chids = get_useful_chids(interval_nr)

    datetime_limits = [pd.to_datetime(datetime_limits_string[0]),
                       pd.to_datetime(datetime_limits_string[1])]

    time_list = []
    drs_model_calibrated_mean_list = []
    fig, ax = plt.subplots()
    for noise_file_path in tqdm(noise_file_list):
        datetime = pd.to_datetime(noise_file_path.split('_')[-1].split('.')[0])
        if(datetime < datetime_limits[0] or datetime > datetime_limits[1]):
            continue
        #print(data_folder_path+noise_file_path)
        with fits.open(data_folder_path+noise_file_path) as noise_tab:
            nr_runs = len(noise_tab[1].data["PedestelRunId"])
            drs_model_calibrated_of_the_day = noise_tab[1].data["DRSCalibratedData_Temp_"+calibrated_type]

        for run_index in range(nr_runs):
            drs_model_calibrated = np.array(drs_model_calibrated_of_the_day[run_index]).reshape(-1, NRCHID)[:, useful_chids].flatten()
            print(drs_model_calibrated)
            # calculate the mean over all events(1000) and chids per run
            drs_model_calibrated_mean = np.nanmean(drs_model_calibrated)

            if(in_pea):
                drs_model_calibrated_mean = drs_model_calibrated_mean*PEAFACTOR

            if (calibrated_type == "Std" and drs_model_calibrated_mean > 0.6):
                print(noise_file_path, run_index+1, ', std')
                continue

            time_list.append(datetime)
            drs_model_calibrated_mean_list.append(drs_model_calibrated_mean)

    drs_model_calibrated_collection_mean = np.mean(drs_model_calibrated_mean_list)
    print(drs_model_calibrated_collection_mean)
    drs_model_calibrated_collection_std = np.std(drs_model_calibrated_mean_list, dtype="float64", ddof=1)

    unit_str = r'$\mathrm{mV}$'
    calibrated_type_str = 'Mittelwert'
    if(calibrated_type == 'Std'):
        calibrated_type_str = 'Standardabweichung'

    if(in_pea):
        unit_str = r'Spe'
    ylabel = calibrated_type_str+' / '+unit_str

    label_str = ("Mittelwert: "+str(format(round(drs_model_calibrated_collection_mean, 3), '.3f'))+unit_str +
                 ",\nStandardabweichung: "+str(format(round(drs_model_calibrated_collection_std, 3), '.3f')))+unit_str
    plt.scatter(time_list, drs_model_calibrated_mean_list,
                s=50, marker=".", alpha=0.75, label=label_str)

    #ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    #ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    #ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
    #plt.gcf().autofmt_xdate()

    #plt.ylabel(ylabel, fontsize=13)
    #ax.yaxis.grid(linestyle=':')
    plt.legend(loc='lower left', numpoints=1, title="", fontsize=12)
    #plt.tight_layout()
    if(store_file_path is not None):
        plt.savefig(store_file_path, dpi=200)
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
              default='Baseline',
              type=click.Choice(['Baseline', 'Gain']))
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
    plt.savefig(store_file_path, dpi=200)
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

    # Cecking wether the intervalIndices and the fitvalues are based on the given dataCollection
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
        plt.savefig(store_file_path, dpi=200)
    plt.show()
    plt.close()
