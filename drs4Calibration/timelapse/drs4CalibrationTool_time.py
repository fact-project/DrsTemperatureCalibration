import pandas as pd
import numpy as np
import click
import h5py
import os
import logging

from array import array
from copy import deepcopy

from tqdm import tqdm
from astropy.io import fits
from fact.credentials import create_factdb_engine
from zfits import FactFits
from scipy.optimize import curve_fit
from joblib import Parallel, delayed

import drs4Calibration.config as config
from drs4Calibration.constants import NRCHID, NRCELL, NRTEMPSENSOR, ROI, ADCCOUNTSTOMILIVOLT
from drs4Calibration.tools import safety_stuff

import matplotlib.pyplot as plt
from time import time


def print_delta_time(time, string=""):
    hours = int(time / 3600)
    rest = time % 3600
    minutes = int(rest / 60)
    seconds = round(rest % 60, 2)
    print(string+" deltaTime: ", hours, minutes, seconds)


@click.command()
@click.argument('drs_file_list_doc_path',
                default="/net/big-tank/POOL/" +
                        "projects/fact/drs4_calibration_data/" +
                        "calibration/calculation/drsFitsFiles.txt",
                type=click.Path(exists=False))
def search_drs_fits_files(drs_file_list_doc_path: str):
    '''
        Search through the fact-database and store the path of all drsFiles
        under the given storePath

        Args:
            drs_file_list_doc_path (str):
                Full path to the storeFile
                with the extension '.txt'
    '''
    # TODO check safety stuff. maybe remove
    #safety_stuff(drs_file_list_doc_path)

    def filename(row):
        return os.path.join(
            str(row.date.year),
            "{:02d}".format(row.date.month),
            "{:02d}".format(row.date.day),
            "{}_{:03d}.fits.fz".format(row.fNight, row.fRunID),
        )
    # 40drs4320Bias
    drs_infos = pd.read_sql(
                   "RunInfo",
                   create_factdb_engine(),
                   columns=[
                       "fNight", "fRunID",
                       "fRunTypeKey", "fDrsStep",
                       "fNumEvents"])
    drs_file_infos = drs_infos.query("fRunTypeKey == 2 &" +
                                     "fDrsStep == 2 &" +
                                     "fNumEvents == 1000").copy()

    # fNumEvents == 1000 prevent for unfinished/broken files
    drs_file_infos["date"] = pd.to_datetime(drs_file_infos.fNight.astype(str),
                                            format="%Y%m%d")

    drs_files = drs_file_infos.apply(filename, axis=1).tolist()
    pd.DataFrame(drs_files).to_csv(drs_file_list_doc_path, index=False,
                                   header=False)


@click.command()
@click.argument('drs_file_list_doc_path',
                default="/net/big-tank/POOL/" +
                        "projects/fact/drs4_calibration_data/" +
                        "calibration/calculation/selectedDrsFitsFiles.txt",
                type=click.Path(exists=True))
@click.argument('store_file_path',
                default="/net/big-tank/POOL/" +
                        "projects/fact/drs4_calibration_data/" +
                        "calibration/calculation/newBaseline_timeTest.h5",
                type=click.Path(exists=False))
@click.argument('source_folder_path',
                default="/net/big-tank/POOL/projects/fact/drs4_calibration_data/",
                type=click.Path(exists=False))
def store_drs_values(drs_file_list_doc_path, store_file_path, source_folder_path):

    with h5py.File(store_file_path, 'w') as hf:
        hf.create_dataset(
            name="Time", dtype="float32",
            shape=(0, 1), maxshape=(None, 1),
            compression="gzip", compression_opts=9,
            fletcher32=True)
        hf.create_dataset(
            name="Temperature", dtype="float32",
            shape=(0, NRTEMPSENSOR), maxshape=(None, NRTEMPSENSOR),
            compression="gzip", compression_opts=9,
            fletcher32=True)
        hf.create_dataset(
            name="NewBaseline", dtype="float32",
            shape=(0, NRCHID*NRCELL*ROI), maxshape=(None, NRCHID*NRCELL*ROI),
            compression="gzip", compression_opts=9,
            fletcher32=True)

    class SourceDataSet:
        # @resettable

        run_begin = pd.to_datetime("")
        run_end = pd.to_datetime("")

        def __init__(self):
            type(self).run_begin = pd.to_datetime("")
            type(self).run_end = pd.to_datetime("")


    source_data_set = SourceDataSet()
    drs_file_list = open(drs_file_list_doc_path).read().splitlines()
    for drs_fits_file_path in tqdm(drs_file_list):
        drs_fits_file_path = drs_file_list[700] # care!!
        date_path_part = drs_fits_file_path.split('_')[0]

        drs_fits_file_path = (source_folder_path+"raw/" +
                              drs_fits_file_path.strip("\n"))
        drs_file_path = (drs_fits_file_path.strip("fits.fz") +
                         ".drs.fits.gz")
        temp_file_path = (source_folder_path+"aux/" +
                          date_path_part+".FAD_CONTROL_TEMPERATURE.fits")

        if(os.path.isfile(drs_fits_file_path) and os.path.isfile(temp_file_path)):
            time_marker1 = time()
            with fits.open(drs_file_path,
                           ignoremissing=True,
                           ignore_missing_end=True) as drs_table:

                source_data_set.run_begin = pd.to_datetime(drs_table[1].header["RUN2-BEG"])
                source_data_set.run_end = pd.to_datetime(drs_table[1].header["RUN2-END"])

            print(type(source_data_set.run_begin), type(source_data_set.run_end))
            time_marker2 = time()
            print_delta_time(time_marker2 - time_marker1, "open drs_file_path")
            time_marker3 = time()
            with fits.open(temp_file_path,
                           mmap=True,
                           mode='denywrite',
                           ignoremissing=True,
                           ignore_missing_end=True) as table:

                table_time = table[1].data["Time"]
                table_temperature = table[1].data["temp"]
            time_marker4 = time()
            print_delta_time(time_marker4 - time_marker3, "open temp_file_path")
            print(type(table_time), table_time.shape, type(table_temperature), table_temperature.shape)
            time_marker5 = time()
            if table_temperature.shape[1] != NRTEMPSENSOR:
                temp_filename = temp_file_path.split('/')[-1]
                message = (
                    " File not used: Just "+str(table_temperature.shape[1]) +
                    " Temperature Values in File '"+temp_filename+"'")
                raise Exception(message)

            table_datetime = pd.to_datetime(table_time * 24 * 3600 * 1e9)
            data_len = len(table_datetime)

            lower_mask = np.where(table_datetime > source_data_set.run_begin)[0]
            upper_mask = np.where(table_datetime < source_data_set.run_end)[0]

            mask = []
            if(len(lower_mask) is not 0 and
               len(upper_mask) is not 0):

                lower_boundarie_idx = lower_mask[0]
                upper_boundarie_idx = upper_mask[-1]

                if(lower_boundarie_idx > 0):
                    lower_boundarie_idx = lower_boundarie_idx - 1
                if(upper_boundarie_idx < data_len):
                    upper_boundarie_idx = upper_boundarie_idx + 1

                mask = np.arange(lower_boundarie_idx, upper_boundarie_idx+1, 1, dtype="int")

            if len(mask) == 0:
                message = ("Cant use drs file," +
                           " runs out of range of temperature data taking")
                raise Exception(message)
            timestamps_during_run = np.array(table_time[mask])
            temperature_during_run = np.array(table_temperature[mask])

            if timestamps_during_run.shape[0] > 1:
                time_mean = np.mean(timestamps_during_run, dtype="float32")
            else:
                time_mean = timestamps_during_run

            if temperature_during_run.shape[0] > 1:
                temp_mean = np.mean(temperature_during_run, dtype="float32",
                                    axis=0)
            else:
                temp_mean = temperature_during_run
            time_marker6 = time()
            print_delta_time(time_marker6 - time_marker5, "calc temp/time")
            print_delta_time(time_marker6 - time_marker1, "complete")

            time_marker7 = time()
            fits_stream = FactFits(drs_fits_file_path)
            time_marker8 = time()
            print_delta_time(time_marker8 - time_marker7, "load  fits_stream")

            cell_sample_value_mean_default = array("f", [np.NaN] * (NRCELL*ROI))
            chid_cell_sample_value_mean_default = array("f", [np.NaN] * (NRCHID*NRCELL*ROI))
            chid_cell_sample_value_mean = deepcopy(chid_cell_sample_value_mean_default)
            for chid in tqdm(range(NRCHID)):
                #time_marker9 = time()
                cell_sample_values = [x[:] for x in [[]] * (1024*300)]
                #time_marker10 = time()
                #print_delta_time(time_marker10 - time_marker9, "init  cell_sample_values")
                fits_stream = FactFits(drs_fits_file_path)
                for event in tqdm(fits_stream):
                    start_cell = event["StartCellData"][chid]
                    data = event["Data"]
                    for sample in range(ROI):
                        cell = (start_cell + sample) % NRCELL
                        value = data[chid][sample]
                        cell_sample_values[cell*ROI+sample].append(value)
                        #print(type(event["Data"]), event["Data"].shape)

                # print(cell_sample_values[5*300+150])
                # print(cell_sample_values[15*300+150])
                # print(cell_sample_values[100*300+150])
                cell_sample_value_mean = deepcopy(cell_sample_value_mean_default)
                for index in tqdm(range(len(cell_sample_values))):
                    #print(type(cell_sample_values[index]), cell_sample_values[index])
                    values = cell_sample_values[index]
                    if(len(values) == 1):
                        cell_sample_value_mean[index] = values[0]
                    elif (len(values) > 1):
                        cell_sample_value_mean[index] = np.mean(values)

                chid_cell_sample_value_mean[chid*NRCELL*ROI:(chid+1)*NRCELL*ROI] = cell_sample_value_mean
                #print(cell_sample_value_mean)
            return
            #fits_stream.close()
            with h5py.File(store_file_path) as h5pyTable:
                add_value_to_h5py_table(
                    h5pyTable,
                    "Time",
                    time_mean)
                add_value_to_h5py_table(
                    h5pyTable,
                    "Temperature",
                    temp_mean)
                add_value_to_h5py_table(
                    h5pyTable,
                    "NewBaseline",
                    chid_cell_sample_value_mean)
        else:
            drs_filename = drs_fits_file_path.split('/')[-1]
            temp_filename = temp_file_path.split('/')[-1]
            print(" Pair of drs file '"+drs_filename+"'" +
                  " and temp file '"+temp_filename+"' does not exist")



def add_value_to_h5py_table(h5pyTable, columnName, value):
    data = h5pyTable[columnName]
    data.resize((len(data)+1, data.maxshape[1]))
    data[len(data)-1, :] = value


@click.command()
@click.argument('path',
                default="/net/big-tank/POOL/" +
                        "projects/fact/drs4_calibration_data/" +
                        "calibration/calculation/newBaseline.h5",
                type=click.Path(exists=False))
def plot(path):
    chid = 0
    cell = 0
    sample = 10
    with h5py.File(path) as h5pyTable:
        time = h5pyTable["Time"][:]
        temp = h5pyTable["Temperature"][:, int(chid/9)]
        value = h5pyTable["NewBaseline"][:, (chid*NRCELL+cell)*ROI+sample]

    print(type(time), len(time))
    print(type(temp), len(temp))
    print(type(value), len(value))
    mask = np.where(value == value)
    temp = temp[mask]
    value = value[mask]*ADCCOUNTSTOMILIVOLT

    temp_matrix = np.vstack([temp, np.ones(len(temp))]).T
    slope, offset = np.linalg.lstsq(temp_matrix, value)[0]

    plt.plot(temp, value, ".")

    temp_range = np.linspace(10, 40, 10000)
    y_fit = slope*temp_range+offset
    plt.plot(temp_range, y_fit)

    print(slope, offset)
    print(y_fit)

    #fitPlot, = plt.plot(temp_range, fit-single_photon_limit, "--", color=color_mean)
    #fitPlot, = plt.plot(temp_range, fit+single_photon_limit, "--", color=color_mean)

    plt.title("new Baseline \nChid: "+str(chid)+", Cell: "+str(cell)+", Sample: "+str(sample), fontsize=15, y=1.00)  # , fontsize=20, y=0.95

    plt.xlabel(r'Temperature /$\mathrm{^\circ C}$')
    plt.ylabel("Baseline"+r' /$\mathrm{mV}$')
    plt.xlim(min(temp)-1, max(temp)+1)
    plt.grid()
    plt.gca().ticklabel_format(useOffset=False)
    plt.savefig("test.jpg")
    plt.show()
    plt.close()



    # def value(temp, chid, cell, sample):
    # ADCCOUNTSTOMILIVOLT = 2000.0 / 4096.0
    # NRCELL = 1024
    # ROI = 300
    # f = fits.open("drsFitParameter.fits")
    # bs = f[1].data["BaselineSlope"][0][chid*NRCELL+cell]
    # bo = f[1].data["BaselineOffset"][0][chid*NRCELL+cell]
    # ts = f[1].data["TriggerOffsetSlope"][0][chid*ROI+sample]
    # to = f[1].data["TriggerOffsetOffset"][0][chid*ROI+sample]
    # oldValue = bs*temp+bo+ts*temp+to
    # h5pyTable = h5py.File("newBaseline.h5")
    # time = h5pyTable["Time"][:]
    # temperature = h5pyTable["Temperature"][:, int(chid/9)]
    # value = h5pyTable["NewBaseline"][:, (chid*NRCELL+cell)*ROI+sample]
    # mask = np.where(value == value)
    # temperature = temperature[mask]
    # value = value[mask]*ADCCOUNTSTOMILIVOLT
    # temp_matrix = np.vstack([temperature, np.ones(len(temperature))]).T
    # slope, offset = np.linalg.lstsq(temp_matrix, value)[0]
    # newValue = slope*temp+offset
    # return [oldValue, newValue, oldValue-newValue]


    for cell in range(10):
        print("cell", cell)
        delta = []
        for sample in range(300):
            delta.append(value(20, 0, cell, sample)[2])
        plt.plot(np.arange(300), delta)
        plt.xlabel(r'Sample')
        plt.ylabel("Delta Baseline"+r' /$\mathrm{mV}$')
        plt.grid()
        plt.savefig("deltaBaseline_temp20_chid0_cell"+str(cell)+".jpg")
        plt.show()
        plt.close()


# @click.command()
# @click.argument('source_file_path',
#                 default="/net/big-tank/POOL/" +
#                         "projects/fact/drs4_calibration_data/" +
#                         "calibration/calculation/time/temp/timeCalibrationData20160817_017.fits",
#                 type=click.Path(exists=True))
# @click.argument('store_file_path',
#                 default="/net/big-tank/POOL/" +
#                         "projects/fact/drs4_calibration_data/" +
#                         "calibration/calculation/time/temp/timeCalibrationData.h5",
#                 type=click.Path(exists=False))
# def store_new(source_file_path, store_file_path):
#     # TODO check safety stuff. maybe remove
#     safety_stuff(store_file_path)
#
#     with h5py.File(store_file_path, 'w') as hf:
#         hf.create_dataset(
#             name="delta_t", dtype="float32",
#             shape=(0, NRCHID*NRCELL*ROI), maxshape=(None, NRCHID*NRCELL*ROI),
#             compression="gzip", compression_opts=5,
#             fletcher32=True)
#         hf.create_dataset(
#             name="voltage", dtype="float32",
#             shape=(0, NRCHID*NRCELL*ROI), maxshape=(None, NRCHID*NRCELL*ROI),
#             compression="gzip", compression_opts=5,
#             fletcher32=True)
#
#     chid_array_offset = np.linspace(0, (NRCHID-1)*NRCELL, NRCHID, dtype='uint32')
#     chid_array_offset = np.repeat(chid_array_offset, repeats=ROI)
#     with fits.open(source_file_path,
#                    mmap=True,
#                    mode='denywrite',
#                    ignoremissing=True,
#                    ignore_missing_end=True) as table:
#         nr_rows = 0
#         max_counter = 0
#         counter = np.zeros((NRCHID*NRCELL), dtype='uint32')
#         nr_events = table[1].data["Data"].shape[0]
#         for chid in range(NRCHID):
#             chid_array_offset = chid*NRCELL*RIO
#             cell_ids = table[1].data["cellIDs"][:, chid]
#             array_indices = np.add(np.multiply(cell_ids, 300), chid_array_offset)
#             voltage = table[1].data["Data"]
#             delta_t = table[1].data["deltaT"]
#
#
# def add_value_to_h5py_table(h5pyTable, columnName, value):
#     data = h5pyTable[columnName]
#     data.resize(len(data)+1, axis=0)
#     data[-1, :] = value


@click.command()
@click.argument('source_file_path',
                default="/net/big-tank/POOL/" +
                        "projects/fact/drs4_calibration_data/" +
                        "calibration/calculation/time/temp/timeCalibrationData20160817_017.fits",
                type=click.Path(exists=True))
@click.argument('store_file_path',
                default="/net/big-tank/POOL/" +
                        "projects/fact/drs4_calibration_data/" +
                        "calibration/calculation/time/temp/timeCalibrationData20160817_017_newVersion.h5",
                type=click.Path(exists=False))
def store(source_file_path, store_file_path):

    # TODO check safety stuff. maybe remove
    safety_stuff(store_file_path)

    with fits.open(source_file_path,
                   mmap=True,
                   mode='denywrite',
                   ignoremissing=True,
                   ignore_missing_end=True) as table:
        cell_ids = table[1].data["cellIDs"]
        voltage = table[1].data["Data"]
        delta_t = table[1].data["deltaT"]

    sorted_delta_t = []
    sorted_voltage = []
    for chid in tqdm(range(NRCHID)):
        cell_ids_chid = cell_ids[:, chid*ROI:(chid+1)*ROI]
        delta_t_chid = delta_t[:, chid*ROI:(chid+1)*ROI]
        voltage_chid = voltage[:, chid*ROI:(chid+1)*ROI]
        for cell in tqdm(range(NRCELL)):
            mask_cell = np.where(cell_ids_chid.ravel() == cell)
            sorted_delta_t.append([delta_t_chid.ravel()[mask_cell]])
            sorted_voltage.append([voltage_chid.ravel()[mask_cell]])


def read_chid(fits_file, chid):
    num_events = fits_file[1].data.shape[0]

    data = pd.DataFrame()
    for key in ('cellIDs', 'deltaT', 'Data'):
        data[key] = (
            fits_file[1].data[key][:, chid * 300: (chid + 1) * 300]
            .ravel()
            .byteswap()
            .newbyteorder()
        )

    data['sample'] = np.tile(np.arange(300), num_events)
    data.rename(
        columns={
            'cellIDs': 'cell',
            'Data': 'adc_counts',
            'deltaT': 'delta_t',
        },
        inplace=True,
    )
    data.dropna(inplace=True)

    return data


def time_function(x, a, b, c):
    return a * x ** b + c


def fit(df, cell, plot=False):
    big_time = df.delta_t.quantile(0.75)
    p0 = [
        0.3,
        -0.66,
        df.adc_counts[df.delta_t >= big_time].mean(),
    ]
    try:
        (a, b, c), cov = curve_fit(
            f,
            df['delta_t'],
            df['adc_counts'],
            p0=p0,
            maxfev=100000,
        )
    except RuntimeError:
        logging.error('Could not fit cell {}'.format(cell))
        return np.full(4, np.nan)

    ndf = len(df.index) - 3
    residuals = df['adc_counts'] - f(df['delta_t'], a, b, c)

    model_value = slope_cell*temp_cell + offset_cell

    residuals = drs_value_cell - model_value
    chisquare = np.sum(pow(residuals[submask], 2)/abs(model_value[submask]))
    chisquare = np.sum(residuals**2) / ndf

    return a, b, c, chisquare


@click.command()
@click.argument('source_file_path',
                default="/net/big-tank/POOL/" +
                        "projects/fact/drs4_calibration_data/" +
                        "calibration/calculation/drsFiles.txt",
                type=click.Path(exists=True))
@click.argument('store_file_path',
                default="/net/big-tank/POOL/" +
                        "projects/fact/drs4_calibration_data/" +
                        "calibration/calculation/drsFiles.txt",
                type=click.Path(exists=False))
@click.argument('jobs',
                default=1)
@click.argument('verbosity',
                default=0)
def calculate_time_fitvalues(source_file_path: str, store_file_path: str,
                             jobs: int, verbosity: int):

    """
    Fit raw data with powerlaw a*x**b+c and calculate chisquare for every fit.
    data is contained in a pandas data frame.

    Args:
        source_file_path (str):
            Full path to the sourceParameter file with the extension '.h5'
        store_file_path (str):
            Full path to the storeFile with the extension '.h5'
        jobs (int):
            The maximum number of concurrently running jobs,
            or the size of the thread-pool. -Nr of CPUs used
        verbosity (int):
            The verbosity level: if non zero, progress messages are printed.
            Above 50, the output is sent to stdout.
            The frequency of the messages increases with the verbosity level.
            If it more than 10, all iterations are reported.
    """

    logging.basicConfig(
        filename=store_file_path.split('.')[0]+".log", filemode='w',
        format='%(levelname)s:%(message)s', level=logging.DEBUG)

    # TODO check safety stuff. maybe remove
    safety_stuff(store_file_path)

    slope = []
    exponent = []
    offset = []

    sample_limits = [10, 290]
    for chid in range(1440):
        logging.info('%s', chid)
        if chid % 9 == 8:
            sample_limits[1] = 240
        for cell in range(NRCELLSPERCHID):
            voltage_cell = voltage[chid*NRCELLSPERCHID]

        data = read_chid(f, chid)

        data = data[(data['sample'] > lower_limit) &
                    (data['sample'] < upper_limit)]


    big_time = df.delta_t.quantile(0.75)
    p0 = [
        0.3,
        -0.66,
        df.adc_counts[df.delta_t >= big_time].mean(),
    ]
    try:
        (a, b, c), cov = curve_fit(
            f,
            df['delta_t'],
            df['adc_counts'],
            p0=p0,
            maxfev=100000,
        )
    except RuntimeError:
        logging.error('Could not fit cell {}'.format(cell))
        return np.full(4, np.nan)


        # new_columns = fits.ColDefs(
        #     [fits.Column(
        #         name="Slope", format=str(NRCELLSPERCHID)+'E',
        #         unit="mV/second", dim=[1, NRCELLSPERCHID],
        #         array=[slope]),
        #      fits.Column(
        #         name="exponent", format=str(NRCELLSPERCHID)+'E',
        #         unit="1", dim=[1, NRCELLSPERCHID],
        #         array=[exponent]),
        #      fits.Column(
        #         name="Offset", format=str(NRCELLSPERCHID)+'E',
        #         unit="mV", dim=[1, NRCELLSPERCHID],
        #         array=[offset])])
