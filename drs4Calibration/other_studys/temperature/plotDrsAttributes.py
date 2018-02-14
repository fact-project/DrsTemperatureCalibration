import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import sys

import os
import numpy as np
import pandas as pd
import click

from matplotlib.ticker import MultipleLocator
from astropy.table import Table
from tqdm import tqdm
from fact.factdb import connect_database, RunInfo
from astropy.io import fits


###############################################################################
def get_best_limits(value, scale=0.02):
    min, max = np.amin(value), np.amax(value)
    range = max-min
    offset = range*scale
    return [min-offset, max+offset]


###############################################################################
# ##############                General  Plots                 ############## #
###############################################################################
# based on older datafiles (need an update)
@click.command()
@click.argument('drs_file_path',
                default="/net/big-tank/POOL/projects/fact/drs4_calibration_data/" +
                        "calibration/calculation/version_0/dataCollection.h5",
                type=click.Path(exists=True))
@click.argument('interval_file_path',
                default="/net/big-tank/POOL/projects/fact/drs4_calibration_data/" +
                        "calibration/calculation/version_0/itervalIndices.h5",
                type=click.Path(exists=True))
@click.argument('store_file_path',
                default="/home/fschulz/plots/camera_drs_values_range_-10_10.mp4",
                type=click.Path(exists=False))
###############################################################################
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


###############################################################################
@click.command()
@click.argument('data_file_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                        'aux/',
                type=click.Path(exists=True))
@click.argument('start_date_str',
                default='20160101')
@click.argument('end_date_str',
                default='20161231')
@click.argument('patch_nr',
                default=10)
@click.argument('store_file_path',
                default='temperatureTrendOf2016_patch10.png')
###############################################################################
def temperature_trend_per_patch(data_file_path, start_date_str, end_date_str,
                                patch_nr, store_file_path):

    time = np.array([])
    temp = np.array([])

    month_before = 0
    for date in tqdm(pd.date_range(start=start_date_str, end=end_date_str, freq='D')):
        if(month_before < date.month):
            month_before = date.month

        filename = (data_file_path + '{}/{:02d}/{:02d}/'.format(date.year, date.month, date.day) +
                    '{}{:02d}{:02d}.FAD_CONTROL_TEMPERATURE.fits'.format(date.year, date.month, date.day))
        if(os.path.isfile(filename)):
            #print("found: ", filename)
            with fits.open(filename) as tab_temp:
                time = np.append(time, tab_temp[1].data['Time'])
                temp = np.append(temp, tab_temp[1].data['temp'][::, patch_nr])

    datetime = pd.to_datetime(time * 24 * 3600 * 1e9)

    plt.plot(datetime, temp, "b.", ms=3, label='')
    plt.xticks(rotation=30)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator(interval=1))

    # plt.gca().yaxis.set_major_locator(MultipleLocator(5))
    # plt.gca().yaxis.set_minor_locator(MultipleLocator(1))

    plt.ylabel(r'Temperatur / $\mathrm{^\circ C}$')
    plt.grid()
    plt.ylim(get_best_limits(temp))
    plt.tight_layout()
    plt.savefig(store_file_path, dpi=200)
    plt.show()
    plt.close()


@click.command()
@click.argument('data_file_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/',
                type=click.Path(exists=True))
@click.argument('date_str',
                default='20160901')
@click.argument('patch_nr',
                default=10)
@click.argument('store_file_path',
                default='temperatureTrendOf20160901_patch10.png')
####################################################################################################
def temperature_per_patch(data_file_path, date_str, patch_nr, store_file_path):
    temperature_per_patch_(data_file_path, date_str, patch_nr, store_file_path)


####################################################################################################
def temperature_per_patch_(data_file_path, date_str, patch_nr, store_file_path):

    date = pd.to_datetime(date_str)

    time = np.array([])
    temp = np.array([])

    filename = (data_file_path + 'aux/{}/{:02d}/{:02d}/'.format(date.year, date.month, date.day) +
                '{}{:02d}{:02d}.FAD_CONTROL_TEMPERATURE.fits'.format(date.year, date.month, date.day))
    folder = (data_file_path + 'raw/{}/{:02d}/{:02d}/'.format(date.year, date.month, date.day))
    if(os.path.isfile(filename) and os.path.isdir(folder)):
        # print("found: ", filename, "and", folder)

        with fits.open(filename) as tab_temp:
            time = np.append(time, tab_temp[1].data['Time'])
            temp = np.append(temp, tab_temp[1].data['temp'][::, patch_nr])

        datetime = pd.to_datetime(time * 24 * 3600 * 1e9)
        plt.plot(datetime, temp, "b.", ms=5,)

        for filename in os.listdir(folder):
            if filename.endswith("drs.fits.gz"):
                with fits.open(folder+"/"+filename) as tab_drs:
                    drsStart = pd.to_datetime(tab_drs[1].header["DATE-OBS"])
                    drsEnd = pd.to_datetime(tab_drs[1].header["DATE-END"])
                    tempMean = np.mean(temp[np.where((datetime > drsStart) & (datetime < drsEnd))])
                    plt.plot([drsStart, drsEnd], [tempMean, tempMean], linestyle="--", marker="|", color="r", ms=20)

        plt.plot([], [], "r|", ms=8, label="DRS-Runs")

        plt.xticks(rotation=30)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

        plt.gca().yaxis.set_major_locator(MultipleLocator(0.5))
        plt.gca().yaxis.set_minor_locator(MultipleLocator(0.25))

        plt.ylim(get_best_limits(temp, 0.02))
        plt.xlabel('Uhrzeit')
        plt.ylabel(r'Temperatur / $^\circ C$')
        plt.grid()
        plt.grid(b=True, which='minor')
        plt.legend(bbox_to_anchor=(0.01, -0.002, 0.98, 0.975), ncol=2, scatterpoints=1, numpoints=1)
        plt.tight_layout()
        plt.savefig(store_file_path, dpi=200)
        plt.show()
        plt.close()
    else:
        print("File '", filename, "' or Folder '", folder, "' does not exist")


@click.command()
@click.argument('data_file_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/aux/',
                type=click.Path(exists=True))
@click.argument('date_str',
                default='20160901')#20160701
@click.argument('board_nr',
                default=2)
@click.argument('store_file_path',
                default='temperatureTrendOf20160901_board2.png')
####################################################################################################
def temperature_per_board(data_file_path, date_str, board_nr, store_file_path):
    temperature_per_board_(data_file_path, date_str, board_nr, store_file_path)


####################################################################################################
def temperature_per_board_(data_file_path, date_str, board_nr, store_file_path,
                           ylim=[25, 31]):
    date = pd.to_datetime(date_str)

    patch_indices = np.array([board_nr*4+0,
                              board_nr*4+1,
                              board_nr*4+2,
                              board_nr*4+3])
    temp = np.array([[]]*4).T
    time = np.array([])

    day, month, year = date.day, date.month, date.year
    try:
        table = Table.read(data_file_path+'{}/{:02d}/{:02d}/{}{:02d}{:02d}.FAD_CONTROL_TEMPERATURE.fits'.format(year, month, day, year, month, day))
        temp = np.concatenate((temp, np.array(table['temp'])[:, patch_indices]))
        time = np.concatenate((time, np.array(table['Time'])))

    except Exception as errInfos:
        print(str(errInfos))
        sys.exit()

    datetime = pd.to_datetime(time * 24 * 3600 * 1e9)

    for index in range(4):
        plt.plot(datetime, temp[:, index], ".", ms=5, label="patch "+str(patch_indices[index]))

    plt.xticks(rotation=30)
    plt.gca().xaxis.set_major_locator(mdates.HourLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    # plt.gca().yaxis.set_major_locator(MultipleLocator(2))
    # plt.gca().yaxis.set_minor_locator(MultipleLocator(1))
    plt.ylim(ylim)#plt.ylim(18, 31)
    plt.xlabel("Uhrzeit")
    plt.ylabel(r'Temperatur / $^\circ C$')
    plt.grid()
    plt.legend(bbox_to_anchor=(0.01, -0.002, 0.98, 1.002), ncol=4,
               mode="expand", scatterpoints=1, numpoints=1)
    plt.tight_layout()
    plt.savefig(store_file_path, dpi=200)
    plt.show()
    plt.close()


####################################################################################################
@click.command()
@click.argument('data_file_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/aux/',
                type=click.Path(exists=True))
@click.argument('date_range',
                default=['2013-06-01', '2017-10-01']) # 2013-06-07
@click.argument('board_nr',
                default=0)
@click.argument('store_file_path',
                default='temperatureDiff20130607_20171001_board0.png')
####################################################################################################
def temperature_diff_per_board(data_file_path, date_range, board_nr, store_file_path):
    temperature_diff_per_board_(data_file_path, date_range, board_nr, store_file_path)


####################################################################################################
def temperature_diff_per_board_(data_file_path, date_range, board_nr, store_file_path,
                                ylim_min=-2.1):
    start_date = pd.to_datetime(date_range[0]).date()
    end_date = pd.to_datetime(date_range[1]).date()

    patch_indices = np.array([board_nr*4+0,
                              board_nr*4+1,
                              board_nr*4+2,
                              board_nr*4+3])

    temp = np.array([[]]*4).T
    time = np.array([])
    for date in tqdm(pd.date_range(start_date, end_date)):
        day, month, year = date.day, date.month, date.year
        try:
            table = Table.read(data_file_path+'{}/{:02d}/{:02d}/{}{:02d}{:02d}.FAD_CONTROL_TEMPERATURE.fits'.format(year, month, day, year, month, day))
            temp = np.concatenate((temp, np.array(table['temp'])[:, patch_indices]))
            time = np.concatenate((time, np.array(table['Time'])))

        except Exception as errInfos:
            # print(str(errInfos))
            continue

    total_mask = np.ones((time.shape), dtype=bool)
    for i in range(4):
        mask = (temp[:, i] != 0.0) & (abs(temp[:, i]) < 100.0)
        total_mask = total_mask & mask

    #print(len(time[~total_mask]), pd.to_datetime(time[~total_mask]), temp[~total_mask, :])

    temp = temp[total_mask]
    time = time[total_mask]

    datetime = pd.to_datetime(time * 24 * 3600 * 1e9)

    for i in range(4):
        for j in range(i+1, 4):
            delta_value = abs(temp[:, i] - temp[:, j])
            label_str = r'$\Delta$ P{}-P{}'.format(patch_indices[i],
                                                   patch_indices[j])
            plt.plot(datetime, delta_value, '.', ms=1, label=label_str)

    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator(interval=2))

    plt.gca().yaxis.set_major_locator(MultipleLocator(2))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(1))
    plt.xticks(rotation=30)
    lgnd = plt.legend(bbox_to_anchor=(0.01, 0.05, 0.98, .102), ncol=3,
                      mode="expand", borderaxespad=0., scatterpoints=1, numpoints=1)
    # change the marker size manually
    for i in range(len(lgnd.legendHandles)):
        lgnd.legendHandles[i]._legmarker.set_markersize(8)

    plt.ylabel(r'Betrag der Temperaturdifferenz / $^\circ C$')
    plt.grid()
    plt.ylim(ylim_min, ) # -0.8
    plt.savefig(store_file_path, dpi=200)
    plt.show()
    plt.close()


####################################################################################################
@click.command()
@click.argument('data_file_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/',
                type=click.Path(exists=True))
@click.argument('date_str',
                default='20160901')
@click.argument('patch_nr',
                default=(10, 15))
@click.argument('store_file_path',
                default='TemperaturTrendOf20160901_patch10_15.png')
####################################################################################################
def temperature_diff_per_patch(data_file_path, date_str, patch_nr, store_file_path):

    date = pd.to_datetime(date_str)

    time = np.array([])
    temp_diff = np.array([])

    filename = (data_file_path + 'aux/{}/{:02d}/{:02d}/'.format(date.year, date.month, date.day) +
                '{}{:02d}{:02d}.FAD_CONTROL_TEMPERATURE.fits'.format(date.year, date.month, date.day))
    folder = (data_file_path + 'raw/{}/{:02d}/{:02d}/'.format(date.year, date.month, date.day))
    if(os.path.isfile(filename) and os.path.isdir(folder)):
        # print("found: ", filename, "and", folder)
        with fits.open(filename) as tab_temp:
            time = np.append(time, tab_temp[1].data['Time'])
            temp_1 = tab_temp[1].data['temp'][::, patch_nr[0]]
            temp_2 = tab_temp[1].data['temp'][::, patch_nr[1]]
            temp_diff = np.append(temp_diff, temp_2-temp_1)

        datetime = pd.to_datetime(time * 24 * 3600 * 1e9)

        plt.plot(datetime, temp_diff, ".", ms=5, label="temp diff patch 10 und 15")

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        plt.ylim(get_best_limits(temp_diff))
        plt.xlabel(date_str+" / $h$")
        plt.ylabel(r'Temperatur / $^\circ C$')
        plt.grid()
        plt.legend(bbox_to_anchor=(0.01, -0.002, 0.98, .102), ncol=2,
                   mode="expand", borderaxespad=0., scatterpoints=1, numpoints=1)
        plt.tight_layout()
        plt.savefig(store_file_path, dpi=200)
        plt.show()
        plt.close()
    else:
        print("File '", filename, "' or Folder '", folder, "' does not exist")


@click.command()
@click.argument('time_range',
                default=['20120101', '20171231'])
@click.argument('store_file_path',
                default='TemperaturDiffHist20120101_20171231.png')
####################################################################################################
def temperature_difference_hist(time_range, store_file_path):

    connect_database()
    drs_runs = pd.DataFrame(list(
        RunInfo
        .select(
            RunInfo.frunstart.alias('start'),
            RunInfo.fdrstempmaxmean.alias('Temperatur'),
        )
        .where(RunInfo.fnight >= time_range[0])
        .where(RunInfo.fnight <= time_range[1])
        .where(RunInfo.fruntypekey == 2)
        .where(RunInfo.fdrsstep == 2)
        .dicts()
    ))
    data_runs = pd.DataFrame(list(
        RunInfo
        .select(
            RunInfo.frunstart.alias('start'),
            RunInfo.fdrstempmaxmean.alias('Temperatur'),
        )
        .where(RunInfo.fnight >= time_range[0])
        .where(RunInfo.fnight <= time_range[1])
        .where(RunInfo.fruntypekey == 1)
        .dicts()
    ))

    diffs = []
    for run1, run2 in tqdm(zip(drs_runs.iloc[:-1].itertuples(), drs_runs.iloc[1:].itertuples())):
        runs = data_runs.query('(start >= @run1.start) & (start < @run2.start)')
        diffs.extend(
            runs.Temperatur.values - run1.Temperatur
        )

    diffs = np.array(diffs)
    mask = np.isfinite(diffs)

    values = diffs[mask]
    values_mean = np.mean(values)
    values_std = np.std(values)
    values_min, values_max = np.min(values), np.max(values)

    text_str = 'Wertebereich: [{:0.2f}$\,°C$, {:0.2f}$\,°C$], Mittelwert: {:0.2f}$\,°C \pm $ {:0.1f}$\,\%$'.format(values_min, values_max, values_mean, abs(values_std/values_mean*100))
    plt.text(-2.4425, -0.77, text_str,
             fontdict={'family': 'serif',
                       'color':  'black',
                       #'weight': 'bold',
                       'size': 12,
                       },
             bbox=dict(boxstyle="round", facecolor="white", ec="k"),
             multialignment="left")
    #plt.axhline(0.0, color='k')

    weights = np.full(len(values), 100/len(values))
    N, bins, patches = plt.hist(values, bins=100, weights=weights)

    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    plt.gca().xaxis.set_minor_locator(MultipleLocator(0.5))

    plt.gca().yaxis.set_major_locator(MultipleLocator(2))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(1))

# vor calc. use plt.hist(abs(values), bins=100, weights=weights)
# vorher bins=10000 setzen
    # print(N, bins)
    # print(sum(N[0:len(bins[bins<(0.5+0.00043/2)])]))
    # print(sum(N[0:len(bins[bins<(0.75+0.00043/2)])]))
    # print(sum(N[0:len(bins[bins<(1.+0.00043/2)])]))
    # print(np.percentile(values, 50), np.percentile(values, 60), np.percentile(values, 75), np.percentile(values, 90), np.percentile(values, 95))
    # print(np.percentile(abs(values), 50), np.percentile(abs(values), 60), np.percentile(abs(values), 75), np.percentile(abs(values), 90), np.percentile(abs(values), 95))
    # plt.xlim(0, 3)

    plt.xlim(-2.5, 2.5)
    plt.ylim(-1.16,)
    plt.grid()
    plt.ylabel(r'Häufigkeit / $\%$ ')
    plt.xlabel(r'($T_{\mathrm{Data-Run}} - T_{\mathrm{DRS-Run}}$) / °C ')
    plt.tight_layout()
    plt.savefig(store_file_path, dpi=300)
    plt.show()
    plt.close()
