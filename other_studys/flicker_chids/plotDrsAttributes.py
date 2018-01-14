import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import sys

import numpy as np
import pandas as pd


import h5py
import click

###############################################################################
# ##############                    Helper                     ############## #
###############################################################################

NRCHID = 1440
NRCELL = 1024
ROI = 300

DACfactor = 2500/pow(2, 16)*50000  # ca. 1907.35 mV

nrCellsPerChid = {'Baseline': NRCELL,
                  'Gain': NRCELL,
                  'ROIOffset': ROI}

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
@click.command()
@click.argument('data_collection_file_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_0/drsData.h5',
                type=click.Path(exists=True))
@click.argument('store_file_path',
                default='/home/fschulz/plots/flicker_chids/gain_chid1265_cell500.png',
                type=click.Path(exists=False))
@click.argument('date_range',
                default=['2014-01-01', '2017-10-01'])
@click.option('--drs_value_type', '-type',
              default='Gain',
              type=click.Choice(['Baseline', 'Gain', 'ROIOffset']))
@click.argument('chid',
                default=1265) #1260-1271
@click.argument('cell',
                default=500)
###############################################################################
def drs_value_cell_timeline(data_collection_file_path, store_file_path,
                            date_range, drs_value_type,
                            chid, cell):

    drs_value_cell_timeline_(data_collection_file_path, store_file_path,
                             date_range, drs_value_type,
                             chid, cell)


###############################################################################
def drs_value_cell_timeline_(data_collection_file_path, store_file_path,
                             date_range, drs_value_type,
                             chid, cell):

    NRCELLSPERCHID = nrCellsPerChid[drs_value_type]
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

    ylabel_str = drs_value_type+r' /$\mathrm{mV}$'
    if(drs_value_type == 'Gain'):
        drs_value /= DACfactor
        drs_value_std /= DACfactor
        ylabel_str = drs_value_type+r' /$\mathrm{1}$'

    print(np.mean(drs_value_std))

    sc_all = plt.scatter(datetime, drs_value, marker='.', c=temp, alpha=1)

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
    plt.ylim(get_best_limits(drs_value, 0.05))
    plt.grid()
    if(store_file_path is not None):
        plt.savefig(store_file_path)
    plt.show()
    plt.close()
