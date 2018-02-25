import pandas as pd
import numpy as np
import click
import h5py
import os
import logging
import gc
from zfits import FactFits
from joblib import Parallel, delayed

from tqdm import tqdm

from scipy.stats import linregress
import multiprocessing

from astropy.io import fits
from fact.credentials import create_factdb_engine
from drs4Calibration.drs4Calibration_version_1.config import data_collection_config, fit_value_config
from drs4Calibration.drs4Calibration_version_1.constants import NRCHID, NRCELL, ROI, NRTEMPSENSOR, DACfactor

import tempfile
import shutil

import sys


###############################################################################
###############################################################################
@click.command()
@click.argument('list_of_needed_files_doc_path',
                default='/net/big-tank/POOL/' +
                        'projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_1/list_of_needed_files.txt',
                type=click.Path(exists=False))
###############################################################################
def search_drs_run_files(list_of_needed_files_doc_path: str):
    '''
        Search through the fact-database and store the path of all needed
        drs-pedestal-runs und temperature files under the given storePath

        Args:
            list_of_needed_files_doc_path (str):
                Full path to the storeFile
                with the extension '.txt'
    '''

    db_table = pd.read_sql(
                   'RunInfo',
                   create_factdb_engine(),
                   columns=[
                       'fNight', 'fRunID',
                       'fRunTypeKey', 'fDrsStep',
                       'fNumEvents'])
    selected_db_table = db_table.query('fNight > 20120000 &' +
                                       'fNight < 20170801 &' +
                                       '((fRunTypeKey == 3 &' + 'fDrsStep == 0) |' +
                                       ' (fRunTypeKey == 4 &' + 'fDrsStep == 1) |' +
                                       ' (fRunTypeKey == 2 &' + 'fDrsStep == 2)) &' +
                                       'fNumEvents == 1000').copy()

    # fNumEvents == 1000 prevent for unfinished/broken files
    selected_db_table['date'] = pd.to_datetime(selected_db_table.fNight.astype(str),
                                               format='%Y%m%d')

    def get_files_of_the_day(data_frame_of_a_day):
        datetime = data_frame_of_a_day.iloc[0]['date']

        path = os.path.join(
            '{:04d}'.format(datetime.year),
            '{:02d}'.format(datetime.month),
            '{:02d}'.format(datetime.day),
            '',
        )

        date_str = str(data_frame_of_a_day.iloc[0]['fNight'])
        temperature_file = 'aux/'+path+date_str+'.FAD_CONTROL_TEMPERATURE.fits'

        def get_file_path(path, date, run_id, extension):
            filename = '{}_{:03d}'.format(date, run_id)+extension
            return (path+filename)

        array_of_run_ids = data_frame_of_a_day.groupby('fDrsStep')['fRunID'].apply(list).as_matrix()
        if (array_of_run_ids.shape[0] != 3):
            print(date_str, 'No completed pedestal run serie taken')

            return ''

        if (len(array_of_run_ids[0]) >= len(array_of_run_ids[1]) and
            len(array_of_run_ids[1]) >= len(array_of_run_ids[2])):
            files_of_the_day = temperature_file

            for nr_serie in range(1, len(array_of_run_ids[2])+1):
                id_serie = [-1, -1, array_of_run_ids[2][-nr_serie]]
                indices = np.where(array_of_run_ids[1] < id_serie[2])[0]
                if(len(indices) == 0):
                    continue
                id_serie[1] = array_of_run_ids[1][indices[-1]]
                indices = np.where(array_of_run_ids[0] < id_serie[1])[0]
                if(len(indices) == 0):
                    continue
                id_serie[0] = array_of_run_ids[0][indices[-1]]
                files_of_the_day += (
                    '|'+get_file_path('raw/'+path, date_str, id_serie[0], '.fits.fz') +
                    ','+get_file_path('raw/'+path, date_str, id_serie[1], '.fits.fz') +
                    ','+get_file_path('raw/'+path, date_str, id_serie[2], '.fits.fz')
                )

            return files_of_the_day

        else:
            print(date_str)
            print(array_of_run_ids[0])
            print(array_of_run_ids[1])
            print(array_of_run_ids[2])

            return ''

    file_collection = selected_db_table.groupby('fNight').apply(get_files_of_the_day).tolist()
    file_collection = list(filter(None, file_collection))

    pd.DataFrame(file_collection).to_csv(list_of_needed_files_doc_path,
                                         sep=';',  # use a char not contained in the path
                                         index=False,
                                         header=False)


###############################################################################
###############################################################################
@click.command()
@click.argument('list_of_needed_files_doc_path',
                default='/net/big-tank/POOL/' +
                        'projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_1/list_of_needed_files.txt',
                type=click.Path(exists=False))
@click.argument('store_file_path',
                default='/net/big-tank/POOL/' +
                        'projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_1/dataCollection_.h5',
                type=click.Path(exists=False))
@click.argument('source_folder_path',
                default='/net/big-tank/POOL/projects/fact/drs4_calibration_data/',
                type=click.Path(exists=False))
@click.argument('jobs',
                default=15)
###############################################################################
def store_drs_attributes(list_of_needed_files_doc_path: str,
                         store_file_path: str,
                         source_folder_path: str,
                         jobs: int):
    '''
        Calculate and store Baseline and Gain from all drs pedestal runs
        of the given 'list_of_needed_files' together with the Temperature and
        mean of Time of data taking into a .h5 File.
        The stored gain is the unnormed difference between 'baseline' and
        'headline' in ADC-counts so the stored gain values still needs
        to get normend by divide with the DAC/ADC-factor of 3906.25 ADC-counts.
        Given by the used 16 Bit DAC with 2.5V range, and the
        input of 50000 DAC-counts and the 12 Bit ADC with 2.0V range.
        Note: The value Pairs are not stored ordered in time.


        Args:
            list_of_needed_files_doc_path (str):
                Full path to the pedestalFileList-file with the extension '.txt'
            store_file_path (str):
                Full path to the store-file with the extension '.h5'
            source_folder_path (str):
                Path to the raw- and aux-folder containing
                the drs pedestal- and temperature-files
            jobs (int):
                Number of processes
    '''

    column_names = data_collection_config.column_names
    column_dtype = data_collection_config.column_dtype
    column_length = data_collection_config.column_length

    # unable to give callback args like store_file_path
    # therefore define method here
    ############################################################
    def store_result(result):
        if len(result) == 0:
            return

        with h5py.File(store_file_path) as h5py_table:
            for column_name in column_names:
                add_value_to_h5py_table(
                    h5py_table,
                    column_name,
                    result[column_name])
    ############################################################

    def raise_exception(error):
        print(str(error))
        raise Exception(str(error))
    ############################################################
    ############################################################

    logging.basicConfig(
        filename=store_file_path.split('.')[0]+".log", filemode='w',
        format='%(levelname)s:%(message)s', level=logging.DEBUG)

    init_empty_h5_table(store_file_path, column_names, column_dtype, column_length)
    pool = multiprocessing.Pool(processes=jobs)

    calibration_file_list = open(list_of_needed_files_doc_path).read().splitlines()
    # main loop: tqdm dosen work with pool.apply_async
    print(calibration_file_list[1100:][0])
    for file_collection_of_the_day in calibration_file_list[1100:]:
        file_collection_of_the_day = file_collection_of_the_day.split('|')

        temperature_file_path = source_folder_path+file_collection_of_the_day[0]

        if not os.path.isfile(temperature_file_path):
            logging.info(' Temperature file not found: '+file_collection_of_the_day[0])
            continue

        for run_serie in file_collection_of_the_day[1:]:
            run_serie = run_serie.split(',')
            handle_run_serie(run_serie, temperature_file_path, source_folder_path)

            # pool.apply_async(handle_run_serie,
            #                  args=(run_serie, temperature_file_path, source_folder_path),
            #                  callback=store_result, error_callback=raise_exception)

    pool.close()
    pool.join()

    # add creationDate to h5 file
    creation_date_str = pd.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with h5py.File(store_file_path) as store:
        store.attrs['CreationDate'] = creation_date_str


# ########################################################################### #
def init_empty_h5_table(store_file_path, column_names, column_dtype, column_length):
    with h5py.File(store_file_path, 'w') as store:
        comment_str = ("The stored gain is the unnormed difference " +
                       "between 'baseline' and 'headline' in ADC-counts " +
                       "so the stored gain values still needs " +
                       "to get normend by divide with the DAC/ADC-factor " +
                       "of 3906.25 ADC-counts. Given by the used " +
                       "16 Bit DAC with 2.5V range, and the " +
                       "input of 50000 DAC-counts and " +
                       "the 12 Bit ADC with 2.0V range")

        store.attrs['Info'] = comment_str
        # add empty columns to h5py table
        for column_name in column_names:
            dtype = column_dtype[column_name]
            length = column_length[column_name]
            store.create_dataset(
                name=column_name, dtype=dtype,
                shape=(0, length), maxshape=(None, length),
                compression='gzip', compression_opts=5,
                fletcher32=True)


# ########################################################################### #
def add_value_to_h5py_table(h5py_table, column_name, value):
    data = h5py_table[column_name]
    data.resize((len(data)+1, data.maxshape[1]))
    data[len(data)-1, :] = value


# ########################################################################### #
def handle_run_serie(run_serie, temperature_file_path, source_folder_path):
    try:
        run_serie_result = {}
        pedestal_run0_path = source_folder_path+run_serie[0]
        pedestal_run1_path = source_folder_path+run_serie[1]
        pedestal_run2_path = source_folder_path+run_serie[2]
        if(not os.path.isfile(pedestal_run0_path) or
           not os.path.isfile(pedestal_run1_path) or
           not os.path.isfile(pedestal_run2_path)):
            info_str = (' Run serie of files [' +
                        run_serie[0]+', ' +
                        run_serie[1]+', ' +
                        run_serie[2]+'] is not complete')
            logging.info(info_str)
            return run_serie_result

        print('GO ', temperature_file_path.split('/')[-1].split('.')[0])

        fits_stream_run0 = FactFits(pedestal_run0_path)
        fits_stream_run1 = FactFits(pedestal_run1_path)
        fits_stream_run2 = FactFits(pedestal_run2_path)
        if (fits_stream_run0.header()['DRSSTEP'] != 0 or
           fits_stream_run1.header()['DRSSTEP'] != 1 or
           fits_stream_run2.header()['DRSSTEP'] != 2):
            info_str = (' Run serie of files [' +
                        run_serie[0]+', ' +
                        run_serie[1]+', ' +
                        run_serie[2]+'] ' +
                        'is not a valid drs-serie' +
                        'with the correct drs-steps 0, 1, 2')
            logging.info(info_str)
            return run_serie_result

        # for baseline
        run_beg_baseline300 = pd.to_datetime(fits_stream_run2.header()['DATE-OBS'])
        run_end_baseline300 = pd.to_datetime(fits_stream_run2.header()['DATE-END'])

        # for gain
        run_beg_baseline1024 = pd.to_datetime(fits_stream_run0.header()['DATE-OBS'])
        run_end_headline1024 = pd.to_datetime(fits_stream_run1.header()['DATE-END'])

        temp_time_collection = get_mean_of_temp_and_time(
                                    temperature_file_path,
                                    [[run_beg_baseline300, run_end_baseline300],
                                     [run_beg_baseline1024, run_end_headline1024]]
                                )

        run_serie_result['TimeBaseline'] = temp_time_collection['run_0']['time']
        run_serie_result['TempBaseline'] = temp_time_collection['run_0']['temperature']

        run_serie_result['Baseline'] = get_mean_for_ROI_300(fits_stream_run2)

        run_serie_result['TimeGain'] = temp_time_collection['run_1']['time']
        run_serie_result['TempGain'] = temp_time_collection['run_1']['temperature']

        baseline1024_mean, baseline1024_std = get_mean_and_std_for_ROI_1024(fits_stream_run0)
        headline1024_mean, headline1024_std = get_mean_and_std_for_ROI_1024(fits_stream_run1)

        run_serie_result['Gain'] = np.subtract(headline1024_mean, baseline1024_mean)
        # error propagation f = a-b
        print('baseline1024_mean: ', baseline1024_mean)
        print(np.mean(headline1024_std), np.mean(baseline1024_std))
        print(np.mean(np.sqrt(pow(headline1024_std, 2) + pow(baseline1024_std, 2))))
        run_serie_result['GainStd'] = np.sqrt(pow(headline1024_std, 2) + pow(baseline1024_std, 2)).astype('float16')
        sys.quit()
        return run_serie_result

    except Exception as error:
        logging.error(str(error))


# ########################################################################### #
def get_mean_of_temp_and_time(temperature_file_path, run_duration):
    with fits.open(temperature_file_path,
                   mmap=True,
                   mode='denywrite',
                   ignoremissing=True,
                   ignore_missing_end=True) as table:

        table_time = table[1].data['Time']
        table_temperature = table[1].data['temp']

    if table_temperature.shape[1] != NRTEMPSENSOR:
        temp_filename = temperature_file_path.split('/')[-1]
        message = (
            " File not used: Just "+str(table_temperature.shape[1]) +
            " Temperature Values in File '"+temp_filename+"'")
        raise Exception(message)

    table_datetime = pd.to_datetime(table_time * 24 * 3600 * 1e9)
    data_len = len(table_datetime)

    run_nr = 0
    temp_time_collection = {}
    for run_begin, run_end in run_duration:
        lower_mask = np.where(table_datetime > run_begin)[0]
        upper_mask = np.where(table_datetime < run_end)[0]

        if len(lower_mask) > 0 and len(upper_mask) > 0:

            lower_boundarie_idx = lower_mask[0]
            upper_boundarie_idx = upper_mask[-1]

            if(lower_boundarie_idx > 0):
                lower_boundarie_idx = lower_boundarie_idx - 1
            if(upper_boundarie_idx < data_len):
                upper_boundarie_idx = upper_boundarie_idx + 1

            indices = np.arange(lower_boundarie_idx, upper_boundarie_idx + 1, 1, dtype='uint32')

        else:
            raise Exception(
                'Cant use drs file,' +
                ' runs out of range of temperature data taking'
            )
        temperature_during_run = np.array(table_temperature[indices], dtype='float32')
        timestamps_during_run = np.array(table_time[indices], dtype='float32')

        temp_time_pair = {}
        if temperature_during_run.shape[0] > 1:
            temp_time_pair['temperature'] = np.mean(temperature_during_run,
                                                    dtype='float32',
                                                    axis=0)
        else:
            temp_time_pair['temperature'] = temperature_during_run

        if timestamps_during_run.shape[0] > 1:
            temp_time_pair['time'] = np.mean(timestamps_during_run,
                                             dtype='float32')
        else:
            temp_time_pair['time'] = timestamps_during_run

        temp_time_collection['run_'+str(run_nr)] = temp_time_pair
        run_nr += 1

    return temp_time_collection


# TODO check int faster than uint
# ########################################################################### #
def get_mean_for_ROI_300(fits_stream):
    # sum up for ints is faster than building the running-mean
    value_sum = np.zeros((NRCHID*NRCELL, ROI), dtype='int32')
    chid_array_offset = np.linspace(0, (NRCHID-1)*NRCELL, NRCHID, dtype='int32')
    start_cell_index = np.zeros((NRCHID), dtype='int32')
    count = np.zeros((NRCHID*NRCELL), dtype='int32')
    for event in fits_stream:
        start_cell = event['StartCellData'].astype('int32')
        data = event['Data'].astype('int32')

        np.add(start_cell, chid_array_offset, out=start_cell_index, dtype='int32')
        count[start_cell_index] += 1
        value_sum[start_cell_index] = np.add(value_sum[start_cell_index], data, dtype='int32')

    # np.seterr: ignore divide by 0 (count=0)
    # mean_values[count==0] will automatic set to nan from divide
    np.seterr(invalid='ignore')
    mean_values = np.divide(value_sum.astype('float64'), count[:, None].astype('float64'), dtype='float64')

    return mean_values.reshape(-1)


# ########################################################################### #
def get_mean_and_std_for_ROI_1024(fits_stream):
    dtype_data = 'float64'  # float64 will return the exact values
    mean_values = np.zeros((NRCHID, NRCELL), dtype=dtype_data)
    var_values = np.zeros((NRCHID, NRCELL), dtype=dtype_data)

    delta = np.zeros((NRCHID, NRCELL), dtype=dtype_data)
    delta_new = np.zeros((NRCHID, NRCELL), dtype=dtype_data)
    diff = np.zeros((NRCHID, NRCELL), dtype=dtype_data)
    prod = np.zeros((NRCHID, NRCELL), dtype=dtype_data)

    indices = np.zeros((NRCHID, NRCELL), dtype='int16')

    cell_indices = np.arange(NRCELL, dtype='int16')
    chid_indices = np.arange(NRCHID, dtype='int16')[:, None]

    count = 0
    for event in fits_stream:
        all_ = []
        start_cell = event['StartCellData'][:, None].astype('int16')
        # subtract because we have to go the StartCell back
        np.subtract(cell_indices, start_cell, out=indices, dtype='int16')
        np.mod(indices, NRCELL, out=indices, dtype='int16')
        # rolle values for every row about the start_cell offset
        data = event['Data'][chid_indices, indices]
        all_.append(data)
        count += 1
        # calculate running mean
        np.subtract(data, mean_values, out=delta, dtype=dtype_data)
        np.divide(delta, count, out=diff, dtype=dtype_data)
        np.add(mean_values, diff, out=mean_values, dtype=dtype_data)
        #print(np.mean(diff))
        # calculate running var
        #print("var: ",var_values[0][0])
        np.subtract(data, mean_values, out=delta_new, dtype=dtype_data)
        np.multiply(delta, delta_new, out=prod, dtype=dtype_data)
        #print(delta[0][0], delta_new[0][0], "\n delta: ", delta[0][0]*delta_new[0][0], prod[0][0])
        np.add(var_values, prod, out=var_values, dtype=dtype_data)

    var_values = var_values/(count-1)
    print(np.mean(np.array(all_).flatten()), np.mean(mean_values.reshape(-1)))
    print(np.var(np.array(all_).flatten()), np.mean(var_values.flatten()))
    print(np.std(np.array(all_).flatten()), np.mean(np.sqrt(var_values/(count-1)).flatten()))
    return (mean_values.reshape(-1), np.sqrt(var_values/(count-1)).reshape(-1))


###############################################################################
###############################################################################
@click.command()
@click.argument('source_file_path',
                default='/net/big-tank/POOL/' +
                        'projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_1/dataCollection.h5',
                type=click.Path(exists=True))
@click.argument('store_file_path',
                default='/net/big-tank/POOL/' +
                        'projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_1/intervalIndices.h5',
                type=click.Path(exists=False))
###############################################################################
def store_source_based_interval_indices(source_file_path: str,
                                        store_file_path: str):
    '''
    Save the interval limits and the associated interval-indices,
    based on the given '.h5-file' source and
    the with the 'config.py' given hardware boundaries.
    The result for every drs_value_type should be the same.
    H5py cant shuffle data with unsorted(not increasing number) indices,
    therfore the interval-indices will not sort the drs-value data.
    Also save for the drs_value_type Gain per interval a mask,
    based of the with the 'config.py' given 'CutOffErrorFactor'
    drsValue selection.
    There are two reasons for calculate no mask for Baseline values.
    1. No sufficient standard deviation of the Baseline mean exist.
    2. The Baseline mask does not fit in ram.

    Args:
        source_file_path (str):
            Full path to the sourceParameter file with the extension '.h5'
        store_file_path (str):
            Full path to the storeFile with the extension '.h5'
    '''

    drs_value_types = fit_value_config.drs_value_types
    hardware_boundaries = fit_value_config.interval_indice_config.hardware_boundaries
    cut_off_error_factor = fit_value_config.interval_indice_config.cut_off_error_factor

    with h5py.File(source_file_path, 'r') as data_source:
        source_creation_date = data_source.attrs['CreationDate']

    with h5py.File(store_file_path) as store:
        store.clear()
        store.attrs['SCDate'] = source_creation_date

    interval_limits, list_of_interval_indices = get_source_and_boundarie_based_interval_limits_and_indices(
            source_file_path, drs_value_types, hardware_boundaries)

    nr_of_intervals = len(list_of_interval_indices)
    for interval_nr in range(1, nr_of_intervals+1):
        interval_indices = list_of_interval_indices[interval_nr-1]

        low_limit = interval_limits[interval_nr-1].strftime('%Y-%m-%d %H')
        upp_limit = interval_limits[interval_nr].strftime('%Y-%m-%d %H')

        groupname = 'Interval'+str(interval_nr)
        with h5py.File(store_file_path) as store:
            drs_group = store.create_group(groupname)
            drs_group.attrs['LowLimit'] = low_limit
            drs_group.attrs['UppLimit'] = upp_limit
            drs_group.create_dataset('IntervalIndices',
                                     data=interval_indices,
                                     dtype='uint32',
                                     maxshape=(interval_indices.shape),
                                     compression='gzip',
                                     compression_opts=4,
                                     fletcher32=True
                                     )

        if len(interval_indices) == 0:
            continue

        drs_value_type = 'Gain'
        print('Loading ...', drs_value_type, ' : ', interval_nr)
        indiceMask = get_indice_mask(source_file_path,
                                     drs_value_type,
                                     interval_indices,
                                     cut_off_error_factor[drs_value_type])
        print(indiceMask.shape)
        with h5py.File(store_file_path) as store:
            drs_group = store[groupname]
            drs_group.attrs['CutOff'+drs_value_type] = cut_off_error_factor[drs_value_type]
            drs_group.create_dataset(drs_value_type+'Mask',
                                     data=indiceMask,
                                     dtype='bool',
                                     maxshape=(indiceMask.shape),
                                     compression='gzip',
                                     compression_opts=4,
                                     fletcher32=True
                                     )


# ########################################################################### #
def get_source_and_boundarie_based_interval_limits_and_indices(
        source_file_path, drs_value_types, hardware_boundaries_str):
    '''Split the from the source_file_path loaded 'list of dates'
       into intervals, based on the given boundaries.
       The result for every drs_value_type should be the same.
    '''
    interval_dict = {}
    value_dict = {}
    hardware_boundaries = pd.to_datetime(hardware_boundaries_str)
    # Calculate for every drs_value_type the interval limits and
    # interval indices(based on the source array)
    for drs_value_type in drs_value_types:
        with h5py.File(source_file_path, 'r') as data_source:
            time = np.array(data_source['Time'+drs_value_type]).flatten()

        datetime = pd.to_datetime(time * 24 * 3600 * 1e9)

        lower_boundarie = min(datetime).date() + pd.DateOffset(hours=12)
        if(lower_boundarie > hardware_boundaries[0]):
            lower_boundarie = hardware_boundaries[0]
        interval_limits = [lower_boundarie]
        list_of_interval_indices = []
        for boundarie in hardware_boundaries:
            interval_indices = np.where(
                                    (datetime >= lower_boundarie) &
                                    (datetime < boundarie))[0]
            list_of_interval_indices.append(interval_indices)
            lower_boundarie = boundarie
            interval_limits.append(boundarie)
        list_of_interval_indices.append(np.where(datetime >= lower_boundarie)[0])
        upper_boundarie = max(datetime).date() + pd.DateOffset(hours=12)
        if(upper_boundarie < hardware_boundaries[-1]):
            upper_boundarie = hardware_boundaries[-1]
        interval_limits.append(upper_boundarie)
        value_dict['Limits'] = interval_limits
        value_dict['Indices'] = list_of_interval_indices
        interval_dict[drs_value_type] = value_dict

    # Checking whether for every drs_value_type the interval limits and
    # interval indices are the same
    for drs_value_index in range(1, len(interval_dict)):
        if(interval_dict[drs_value_types[0]] != interval_dict[drs_value_types[drs_value_index]]):
            error_str = ('There are differences between the interval boundaries' +
                         'of differend drs_value_types')
            raise Exception(error_str)

    return (interval_dict[drs_value_types[0]]['Limits'],
            interval_dict[drs_value_types[0]]['Indices'])


# ########################################################################### #
def get_indice_mask(source_file_path, drs_value_type,
                    interval_indices, cut_off_error_factor):

    with h5py.File(source_file_path, 'r') as data_source:
        drs_value_std_array = data_source[drs_value_type+'Std'][interval_indices, :]

    drs_value_std_mean_per_cell = np.mean(drs_value_std_array, axis=0)
    drs_value_std_limit = np.multiply(drs_value_std_mean_per_cell,
                                      cut_off_error_factor)

    mask_limit = np.array(drs_value_std_array < drs_value_std_limit[None, :])

    return mask_limit


###############################################################################
###############################################################################
@click.command()
@click.argument('source_file_path',
                default='/net/big-tank/POOL/' +
                        'projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_1/dataCollection.h5',
                type=click.Path(exists=True))
@click.argument('interval_file_path',
                default='/net/big-tank/POOL/' +
                        'projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_1/intervalIndices.h5',
                type=click.Path(exists=True))
@click.argument('interval_nr',
                default=3)
@click.argument('store_file_path',
                default='/net/big-tank/POOL/' +
                        'projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_1/drsFitParameter_interval3.fits',
                type=click.Path(exists=False))
###############################################################################
def calculate_fit_values(source_file_path: str, interval_file_path: str,
                         interval_nr: int, store_file_path: str):
    '''
        Calculate the linear fitvalues of Baseline and Gain
        based on the .h5 source data for the by the hardware boundaries
        given itervals and store them into a .fits File.
        All Gain-values with a larger error (std dev of the mean)
        than the 'CutOffErrorFactor' multiplied with the mean of the error
        from all collected Gain-values for one capacitor will not used for the fit
        Args:
            source_file_path (str):
                Full path to the sourceParameter file
                with the extension '.h5'
            interval_file_path (str):
                Full path to the sourceParameter based intervalndices file
                with the extension '.h5'
            interval_nr (int):
                number of the selected interval
            store_file_path (str):
                Full path to the storeFile
                with the extension '.fits'
    '''

    logging.basicConfig(
        filename=store_file_path.split('.')[0]+".log", filemode='w',
        format='%(levelname)s:%(message)s', level=logging.DEBUG)

    jobs = 20
    verbosity = 10
    pool = Parallel(n_jobs=jobs, verbose=verbosity, max_nbytes="50G")  # batch_size=1,

    groupname = 'Interval'+str(interval_nr)
    drs_value_types = fit_value_config.drs_value_types
    drs_values_per_cell = fit_value_config.drs_values_per_cell
    value_units = fit_value_config.value_units

    column_collection = fits.ColDefs([])
    for drs_value_type in drs_value_types:
        print('Loading ...', drs_value_type)

        drs_value_per_cell = drs_values_per_cell[drs_value_type]
        drs_value_shape = NRCHID*NRCELL*drs_value_per_cell

        temp_folder = tempfile.mkdtemp()

        memmap_paths_slope = os.path.join(temp_folder, 'slope.map')
        memmap_paths_offset = os.path.join(temp_folder, 'offset.map')
        memmap_paths_residual_mean = os.path.join(temp_folder, 'residual_mean.map')

        try:
            mask = np.array([[]])
            with h5py.File(interval_file_path, 'r') as interval_source:
                data = interval_source[groupname]
                low_limit = data.attrs['LowLimit']
                upp_limit = data.attrs['UppLimit']
                interval_indices = np.array(data['IntervalIndices'])
                if (drs_value_type == 'Gain'):
                    cut_off_error_factor = data.attrs['CutOff'+drs_value_type]
                    mask = np.array(data[drs_value_type+'Mask'])

            slope = np.memmap(memmap_paths_slope, mode='w+',
                              shape=drs_value_shape, dtype='float32')
            offset = np.memmap(memmap_paths_offset, mode='w+',
                               shape=drs_value_shape, dtype='float32')
            residual_mean = np.memmap(memmap_paths_residual_mean, mode='w+',
                                      shape=drs_value_shape, dtype='float32')
            slope[:] = np.nan
            offset[:] = np.nan
            residual_mean[:] = np.nan
            del slope
            del offset
            del residual_mean

            split_factor = 12  # split data in smaller pices to avoid run out of memory
            if drs_value_shape % (NRTEMPSENSOR*split_factor) != 0:
                raise Exception('Bad split factor: remaining cells')

            chunk = int(drs_value_shape/NRTEMPSENSOR/split_factor)
            with h5py.File(source_file_path, 'r') as data_source:
                pool(delayed(calculate_fit_values_and_more)(
                     chunk,
                     data_source['Temp'+drs_value_type][interval_indices, int(pice_nr/split_factor)],
                     data_source[drs_value_type][interval_indices, pice_nr*chunk:(pice_nr+1)*chunk],
                     mask[:, pice_nr*chunk:(pice_nr+1)*chunk],
                     np.memmap(memmap_paths_slope, mode='r+',
                               shape=chunk, dtype='float32',
                               offset=int((pice_nr*chunk)*32/8)),
                     np.memmap(memmap_paths_offset, mode='r+',
                               shape=chunk, dtype='float32',
                               offset=int((pice_nr*chunk)*32/8)),
                     np.memmap(memmap_paths_residual_mean, mode='r+',
                               shape=chunk, dtype='float32',
                               offset=int((pice_nr*chunk)*32/8))
                     ) for pice_nr in tqdm(range(NRTEMPSENSOR*split_factor)))

            print('Done')
            # reload fit results
            drs_value_slope = np.memmap(memmap_paths_slope, mode='r+',
                                        shape=drs_value_shape, dtype='float32')
            drs_value_offset = np.memmap(memmap_paths_offset, mode='r+',
                                         shape=drs_value_shape, dtype='float32')
            drs_value_residual_mean = np.memmap(memmap_paths_residual_mean, mode='r+',
                                                shape=drs_value_shape, dtype='float32')

            # catch up gain standardization to 1
            # (The stored gain is the unnormed difference
            # between 'baseline' and 'headline' in ADC-counts
            # so the stored gain values still needs
            # to get normend by divide with the DAC/ADC-factor
            # of 3906.25 ADC-counts)
            if(drs_value_type == 'Gain'):
                drs_value_slope /= DACfactor
                drs_value_offset /= DACfactor
                drs_value_residual_mean /= DACfactor

            drs_value_slope = drs_value_slope.reshape(-1, drs_value_per_cell)
            drs_value_offset = drs_value_offset.reshape(-1, drs_value_per_cell)
            drs_value_residual_mean = drs_value_residual_mean.reshape(-1, drs_value_per_cell)

            value_unit = value_units[drs_value_type]
            drs_value_format = '{}E'.format(drs_value_per_cell)
            drs_value_format_str = '{}*[{}]'.format(NRCHID*NRCELL, drs_value_per_cell)

            new_columns = fits.ColDefs(
                [fits.Column(
                    name=drs_value_type+'Slope',
                    format=drs_value_format,
                    unit=value_unit+'/celsius',
                    dim=drs_value_format_str,
                    array=drs_value_slope),
                 fits.Column(
                    name=drs_value_type+'Offset',
                    format=drs_value_format,
                    unit=value_unit,
                    dim=drs_value_format_str,
                    array=drs_value_offset),
                 fits.Column(
                    name=drs_value_type+'Residual',
                    format=drs_value_format,
                    unit=value_unit,
                    dim=drs_value_format_str,
                    array=drs_value_residual_mean)])
            column_collection = column_collection + new_columns

        finally:
            try:
                shutil.rmtree(temp_folder)
            except:
                print("Failed to delete: " + temp_folder)

    print('write Data to Table')
    hdu = fits.BinTableHDU.from_columns(column_collection)
    hdu.header.insert('TFIELDS', ('EXTNAME', 'FitParameter'), after=True)

    primary = fits.PrimaryHDU()
    comment_str = "Number of the interval"
    primary.header.insert("EXTEND", ("IntNR", str(interval_nr), comment_str), after=True)
    comment_str = "Date-String of the lower interval limit"  # in the format 'yyyy-mm-dd hh'
    primary.header.insert("IntNR", ("LowLimit", low_limit, comment_str), after=True)
    comment_str = "Date-String of the upper interval limit"  # in the format 'yyyy-mm-dd hh'
    primary.header.insert("LowLimit", ("UppLimit", upp_limit, comment_str), after=True)
    comment_str = "'CutOff-ErrorFactor' for the Gain values"
    primary.header.insert("UppLimit", ("CutOff", cut_off_error_factor, comment_str), after=True)

    with h5py.File(source_file_path, 'r') as data_source:
        source_creation_date = data_source.attrs['CreationDate']

    comment_str = "Datetime-String of the source .h5 creation."  # in the format 'yyyy-mm-dd HH:MM:SS'
    primary.header.insert('UppLimit', ('SCDate', source_creation_date, comment_str), after=True)

    print('Save Table')
    thdulist = fits.HDUList([primary, hdu])
    thdulist.writeto(store_file_path, overwrite=True, checksum=True)
    print('Verify Checksum')
    # Open the File verifying the checksum values for all HDUs
    try:
        hdul = fits.open(store_file_path, checksum=True)
        print(hdul[0].header)
        print(hdul[1].header)
        with open(store_file_path.split('.')[0]+'.log', 'r') as logFile:
            logging.info(' Passed verifying Checksum')
            if(logFile.readlines() == [' Passed verifying Checksum']):
                logging.info(' No errors occurred during the Fit-Value calculation.')
    except Exception as err_infos:
        error_str = str(err_infos)
        print(error_str)
        logging.warning(error_str)


# ########################################################################### #
def calculate_fit_values_and_more(indice_range, temperature, drs_value_array,
                                  mask, slope_array, offset_array,
                                  residual_mean_array):

    # tryed to avoide if in for-loop
    if(mask.shape[1] == 0):
        # for Baseline no mask exist jet
        for index in range(indice_range):
            value = drs_value_array[:, index]
            nan_mask = (value != 0)
            value = value[nan_mask]
            temp = temperature[nan_mask]
            apply_calculation(index, temp, value,
                              slope_array, offset_array, residual_mean_array)
            del nan_mask
    else:
        for index in range(indice_range):
            sub_mask = mask[:, index]
            temp = temperature[sub_mask]
            value = drs_value_array[:, index][sub_mask]
            apply_calculation(index, temp, value,
                              slope_array, offset_array, residual_mean_array)
            del sub_mask

    # dont trust the Garbage Collector, so force to free memory
    del temperature
    del drs_value_array
    del slope_array
    del offset_array
    del residual_mean_array
    gc.collect()


# ########################################################################### #
def apply_calculation(index, temp, value, slope_array, offset_array, residual_mean_array):
    # catch RuntimeWarning: invalid value encountered in double_scalars
    with np.errstate(invalid='raise'):
        try:
            slope, offset, r_value, p_value, var_err = linregress(temp, value)

            slope_array[index] = slope
            offset_array[index] = offset

            model_value = slope*temp + offset
            residual = value - model_value

            residual_mean_array[index] = np.mean(abs(residual))
        except Exception as err_infos:
            error_str = str(err_infos)
            logging.error(index, value, error_str)

        finally:
            # dont trust the Garbage Collector, so force to free memory
            del temp
            del value
