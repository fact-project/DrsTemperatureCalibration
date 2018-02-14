import pandas as pd
import numpy as np
import click
import h5py
import os
import logging
import gc

import tempfile
import shutil

from joblib import Parallel, delayed
from scipy.stats import linregress
from tqdm import tqdm
from astropy.io import fits
from fact.credentials import create_factdb_engine

import drs4Calibration.drs4Calibration_version_0.config as config
from drs4Calibration.drs4Calibration_version_0.constants import NRCHID, NRTEMPSENSOR, DACfactor


###############################################################################
@click.command()
@click.argument('drs_file_list_doc_path',
                default='/net/big-tank/POOL/' +
                        'projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_0/drsFiles.txt',
                type=click.Path(exists=False))
###############################################################################
def search_drs_files(drs_file_list_doc_path: str):
    '''
        Search through the fact-database and store the path of all drsFiles
        under the given storePath taken after 2012

        Args:
            drs_file_list_doc_path (str):
                Full path to the storeFile
                with the extension '.txt'
    '''

    def filename(row):
        return os.path.join(
            str(row.date.year),
            '{:02d}'.format(row.date.month),
            '{:02d}'.format(row.date.day),
            '{}_{:03d}.drs.fits.gz'.format(row.fNight, row.fRunID),
        )

    drs_infos = pd.read_sql(
                   'RunInfo',
                   create_factdb_engine(),
                   columns=[
                       'fNight', 'fRunID',
                       'fRunTypeKey', 'fDrsStep',
                       'fNumEvents'])
    drs_file_infos = drs_infos.query(
                                     'fNight > 20120000 &' +
                                     'fNight < 20170801 &' +
                                     'fRunTypeKey == 2 &' +
                                     'fDrsStep == 2 &' +
                                     'fNumEvents == 1000').copy()
    # fNumEvents == 1000 prevent for unfinished/broken files
    drs_file_infos['date'] = pd.to_datetime(drs_file_infos.fNight.astype(str),
                                            format='%Y%m%d')

    drs_files = drs_file_infos.apply(filename, axis=1).tolist()
    pd.DataFrame(drs_files).to_csv(drs_file_list_doc_path, index=False,
                                   header=False)


###############################################################################
@click.command()
@click.argument('drs_file_list_doc_path',
                default='/net/big-tank/POOL/' +
                        'projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_0/drsFiles.txt',
                type=click.Path(exists=True))
@click.argument('store_file_path',
                default='/net/big-tank/POOL/' +
                        'projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_0/drsData.h5',
                type=click.Path(exists=False))
@click.argument('source_folder_path',
                default=('/net/big-tank/POOL/' +
                         'projects/fact/drs4_calibration_data/'),
                type=click.Path(exists=True))
###############################################################################
def store_drs_attributes(drs_file_list_doc_path: str, store_file_path: str,
                         source_folder_path: str):
    '''
        Save Baseline, Gain and TriggerOffset from all drsfiles
        of the given drsFileList together with the Temperature and
        mean of Time of taking into a .h5 File

        Args:
            drs_file_list_doc_path (str):
                Full path to the drsFileList-file with the extension '.txt'
            store_file_path (str):
                Full path to the store-file with the extension '.h5'
            source_folder_path (str):
                Path to the raw- and aux-folder containing
                the drs- and temperature-files
    '''
    print('store')
    logging.basicConfig(
        filename=store_file_path.split('.')[0]+'.log', filemode='w',
        format='%(levelname)s:%(message)s', level=logging.DEBUG)

    renamed_drs_value_types = config.renamedDrsValueTypes

    nrCellsPerChid = config.nrCellsPerChid

    # add empty columns to h5py table
    with h5py.File(store_file_path, 'w') as hf:
        for drs_value_type in renamed_drs_value_types:
            NRCALIBVALUES = nrCellsPerChid[drs_value_type]
            hf.create_dataset(
                name='Time'+drs_value_type, dtype='float32',
                shape=(0, 1), maxshape=(None, 1),
                compression='gzip', compression_opts=4,
                fletcher32=True)
            hf.create_dataset(
                name='Temp'+drs_value_type, dtype='float32',
                shape=(0, NRTEMPSENSOR), maxshape=(None, NRTEMPSENSOR),
                compression='gzip', compression_opts=4,
                fletcher32=True)
            hf.create_dataset(
                name=drs_value_type, dtype='float32',
                shape=(0, NRCHID*NRCALIBVALUES), maxshape=(None, NRCHID*NRCALIBVALUES),
                compression='gzip', compression_opts=4,
                fletcher32=True)
            # used but check is necessary
            hf.create_dataset(
                name=drs_value_type+'Var', dtype='float32',
                shape=(0, NRCHID*NRCALIBVALUES), maxshape=(None, NRCHID*NRCALIBVALUES),
                compression='gzip', compression_opts=4, fletcher32=True)

    # This loop basically iterate over the drsFiles of the drsFileList and
    # check if there are all needed information/ tuple of attributes
    # for later calculations based on the selected drsFile
    # in case they are there, they will simply stored into a h5py table
    # in the case it is impossible to to collect all needed attributes
    # we will continue with the next drsFile

    drs_file_list = open(drs_file_list_doc_path).read().splitlines()
    for drs_file_path in tqdm(drs_file_list):

        date_path_part = drs_file_path.split('_')[0]

        drs_file_path = (source_folder_path+'raw/' +
                         drs_file_path.strip('\n'))
        temp_file_path = (source_folder_path+'aux/' +
                          date_path_part+'.FAD_CONTROL_TEMPERATURE.fits')

        if(os.path.isfile(drs_file_path) and os.path.isfile(temp_file_path)):
            try:
                save_tuple_of_attribute_if_possible(
                    temp_file_path, drs_file_path,
                    store_file_path)
            except Exception as exc:
                drs_filename = drs_file_path.split('/')[-1]
                temp_filename = temp_file_path.split('/')[-1]
                logging.info('In drs file ''+drs_filename+''' +
                             ' or temp file ''+temp_filename+'': '+str(exc))
        else:
            drs_filename = drs_file_path.split('/')[-1]
            temp_filename = temp_file_path.split('/')[-1]
            logging.info(" Pair of drs file '"+drs_filename+"'" +
                         " and temp file '"+temp_filename+"' does not exist")

    # add creationDate to h5 file
    creation_date_str = pd.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with h5py.File(store_file_path) as store:
        store.attrs['CreationDate'] = creation_date_str


###############################################################################
def save_tuple_of_attribute_if_possible(temp_file_path, drs_file_path,
                                        store_file_path):
    drs_value_types = config.drsValueTypes
    renamed_drs_value_types = config.renamedDrsValueTypes
    with fits.open(drs_file_path,
                   mmap=True,
                   mode='denywrite',
                   ignoremissing=True,
                   ignore_missing_end=True) as drs_table:

        header = drs_table[1].header
        bintable = drs_table[1].data

    run_times_list = []
    for i in range(len(drs_value_types)):
        run_times_list.append((
                               pd.to_datetime(header['RUN'+str(i)+'-BEG']),
                               pd.to_datetime(header['RUN'+str(i)+'-END'])
                              ))

    temp_and_time_pairs = get_temp_and_time_pairs_per_drs_value_type(
                            temp_file_path,
                            run_times_list)

    drs_value_mean = []
    drs_value_mean_var = []
    for drs_value_type in drs_value_types:
        drs_value_mean.append(bintable[drs_value_type+'Mean'][0])
        drs_value_mean_var.append(bintable[drs_value_type+'Rms'][0])

    with h5py.File(store_file_path) as h5pyTable:
        for i in range(len(renamed_drs_value_types)):
            add_value_to_h5py_table(
                h5pyTable,
                'Time'+renamed_drs_value_types[i],
                temp_and_time_pairs[i]['time_mean'])
            add_value_to_h5py_table(
                h5pyTable,
                'Temp'+renamed_drs_value_types[i],
                temp_and_time_pairs[i]['temp_mean'])
            add_value_to_h5py_table(
                h5pyTable,
                renamed_drs_value_types[i],
                drs_value_mean[i])
            add_value_to_h5py_table(
                h5pyTable,
                renamed_drs_value_types[i]+'Var',
                drs_value_mean_var[i])


###############################################################################
def add_value_to_h5py_table(h5pyTable, columnName, value):
    data = h5pyTable[columnName]
    data.resize((len(data)+1, data.maxshape[1]))
    data[len(data)-1, :] = value


###############################################################################
def get_temp_and_time_pairs_per_drs_value_type(temp_path, run_times_list):
    '''
    run_times_list a list() of tuples of (start, end) times per
    drs run/drs value type, between which we want to build the mean
    of 'time' and 'temp' from the fits file under 'temp_path'.
    '''

    with fits.open(temp_path,
                   mmap=True,
                   mode='denywrite',
                   ignoremissing=True,
                   ignore_missing_end=True) as table:

        table_time = table[1].data['Time']
        table_temperature = table[1].data['temp']

    if table_temperature.shape[1] != NRTEMPSENSOR:
        temp_filename = temp_path.split('/')[-1]
        message = (
            " File not used: Just "+str(table_temperature.shape[1]) +
            " Temperature Values in File '"+temp_filename+"'")
        raise Exception(message)

    results = []
    table_datetime = pd.to_datetime(table_time * 24 * 3600 * 1e9)
    data_len = len(table_datetime)
    for start, end in run_times_list:  # one pair per drs_value_type
        lower_mask = np.where(table_datetime > start)[0]
        upper_mask = np.where(table_datetime < end)[0]

        mask = []
        if(len(lower_mask) is not 0 and
           len(upper_mask) is not 0):

            lower_boundarie_idx = lower_mask[0]
            upper_boundarie_idx = upper_mask[-1]

            if(lower_boundarie_idx > 0):
                lower_boundarie_idx = lower_boundarie_idx - 1
            if(upper_boundarie_idx < data_len):
                upper_boundarie_idx = upper_boundarie_idx + 1

            mask = np.arange(lower_boundarie_idx, upper_boundarie_idx+1, 1, dtype='int')

        if len(mask) == 0:
            message = ('Cant use drs file,' +
                       ' runs out of range of temperature data taking')
            raise Exception(message)
        timestamps_during_run = np.array(table_time[mask])
        temperature_during_run = np.array(table_temperature[mask])

        if timestamps_during_run.shape[0] > 1:
            time_mean = np.mean(timestamps_during_run, dtype='float32')
        else:
            time_mean = timestamps_during_run

        if temperature_during_run.shape[0] > 1:
            temp_mean = np.mean(temperature_during_run, dtype='float32',
                                axis=0)
        else:
            temp_mean = temperature_during_run

        time_mean = np.array([time_mean])  # ->[float]
        results.append(dict(time_mean=time_mean,
                            temp_mean=temp_mean))

    return results


###############################################################################
@click.command()
@click.argument('source_file_path',
                default='/net/big-tank/POOL/' +
                        'projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_0/drsData.h5',
                type=click.Path(exists=True))
@click.argument('store_file_path',
                default='/net/big-tank/POOL/' +
                        'projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_0/intervalIndices.h5',
                type=click.Path(exists=False))
###############################################################################
def store_source_based_interval_indices(source_file_path: str,
                                        store_file_path: str):
    '''
    Save the interval limits and the associated intervalindices,
    based on the given '.h5-file' source and
    the with the 'config.py' given boundaries.
    The result for every drs_value_type should be the same.
    Also save for every drs_value_type and interval a mask
    of the with the 'config.py' given 'CutOffErrorFactor'
    based drsValue selection.

    Args:
        source_file_path (str):
            Full path to the sourceParameter file with the extension '.h5'
        store_file_path (str):
            Full path to the storeFile with the extension '.h5'
    '''

    drs_value_types = config.renamedDrsValueTypes
    cut_off_error_factor = config.cutOffErrorFactor
    hardware_boundaries = pd.to_datetime(config.hardwareBoundaries)

    with h5py.File(source_file_path, 'r') as data_source:
        source_creation_date = data_source.attrs['CreationDate']

    with h5py.File(store_file_path) as store:
        store.clear()
        store.attrs['SCDate'] = source_creation_date
        for key, value in cut_off_error_factor.items():
            store.attrs['CutOff'+key] = int(value)

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

        for drs_value_type in drs_value_types:
            print('Loading ...', drs_value_type, ' : ', interval_nr)
            indiceMask = getIndiceMask(source_file_path,
                                       drs_value_type,
                                       interval_indices,
                                       cut_off_error_factor[drs_value_type])

            with h5py.File(store_file_path) as store:
                store[groupname].create_dataset(drs_value_type+'Mask',
                                                data=indiceMask,
                                                dtype='bool',
                                                maxshape=(indiceMask.shape),
                                                compression='gzip',
                                                compression_opts=4,
                                                fletcher32=True
                                                )


###############################################################################
def get_source_and_boundarie_based_interval_limits_and_indices(
        source_file_path, drs_value_types, hardware_boundaries):
    '''Split the from the source_file_path loaded 'list of dates'
       into intervals, based on the given boundaries.
       The result for every drs_value_type should be the same.
    '''
    interval_dict = {}
    value_dict = {}
    # Calculate for every drs_value_type the interval limits and
    # interval indices(based on the source array)
    for drs_value_type in drs_value_types:
        with h5py.File(source_file_path, 'r') as data_source:
            time = np.array(data_source['Time'+drs_value_type]).flatten()

        datetime = pd.to_datetime(time * 24 * 3600 * 1e9)

        lower_boundarie = datetime[0].date() + pd.DateOffset(hours=12)
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
        interval_limits.append(datetime[-1].date() + pd.DateOffset(hours=12))
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


###############################################################################
def getIndiceMask(source_file_path, drs_value_type,
                  interval_indices, cut_off_error_factor):

    with h5py.File(source_file_path, 'r') as data_source:
        drs_value_var_array = data_source[drs_value_type+'Var'][interval_indices, :]

    NRCELLSPERCHID = config.nrCellsPerChid[drs_value_type]
    indiceMask = np.full(
                    (interval_indices.shape[0], NRCHID*NRCELLSPERCHID),
                    True, dtype=bool)
    for chid in tqdm(range(NRCHID)):
        for cell in range(NRCELLSPERCHID):
            drs_value_var_cell = drs_value_var_array[:, chid*NRCELLSPERCHID+cell]
            mask_var_0 = np.array(drs_value_var_cell != 0.)

            drs_value_var_cell_mean = np.mean(drs_value_var_cell[mask_var_0], dtype='float32')
            drs_value_var_limit = (drs_value_var_cell_mean * cut_off_error_factor)

            mask_limit = np.array(drs_value_var_cell < drs_value_var_limit)

            mask_total = mask_limit & mask_var_0
            indiceMask[:, chid*NRCELLSPERCHID+cell] = mask_total

    return indiceMask


###############################################################################
@click.command()
@click.argument('source_file_path',
                default='/net/big-tank/POOL/' +
                        'projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_0/drsData.h5',
                type=click.Path(exists=True))
@click.argument('interval_file_path',
                default='/net/big-tank/POOL/' +
                        'projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_0/intervalIndices.h5',
                type=click.Path(exists=True))
@click.argument('interval_nr',
                default=3)
@click.argument('store_file_path',
                default='/net/big-tank/POOL/' +
                        'projects/fact/drs4_calibration_data/' +
                        'calibration/calculation/version_0/drsFitParameter_interval3.fits',
                type=click.Path(exists=False))
###############################################################################
def calculate_fit_values(source_file_path: str, interval_file_path: str,
                         interval_nr: int, store_file_path: str):
    '''
        Calculate the linear fitvalues of Baseline, Gain and TriggerOffset
        based on the .h5 source data for the by the hardware boundaries
        given itervals and store them into a .fits File
        All Baseline, Gain or TriggerOffset-values with a larger error
        than the 'CutOffErrorFactor' multiplied with the mean of the error
        from all collected Baseline, Gain or TriggerOffset-values of the capacitor
        will not used for the fit

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
        filename=store_file_path.split('.')[0]+'.log', filemode='w',
        format='%(levelname)s:%(message)s', level=logging.DEBUG)

    # Cecking wether the intervalIndices are based on the given drsData
    check_dependency(source_file_path, interval_file_path)

    jobs = 15
    verbosity = 10
    pool = Parallel(n_jobs=jobs, verbose=verbosity, max_nbytes="50G")  # batch_size=1,

    # for safety reasons take all needed infos from the interval_file
    # and not from the config
    with h5py.File(interval_file_path, 'r') as interval_source:
        groupnames = list(interval_source.keys())

    if(interval_nr > len(groupnames)):
        error_str = ("interval_nr {} is larger than the number " +
                     "of known intervals{}").format(interval_nr, len(groupnames))
        logging.error(error_str)
        raise Exception(error_str)

    groupname = groupnames[interval_nr-1]
    # for safety reasons take all needed infos from the interval_file
    # and not from the config
    with h5py.File(interval_file_path, 'r') as interval_source:
        keys = list(interval_source[groupname].keys())
        drs_value_types = [key.strip('Mask') for key in keys if "Mask" in key]

    cut_off_error_factor_values = []
    column_collection = fits.ColDefs([])
    for drs_value_type in drs_value_types:
        print('Loading ...', drs_value_type)
        temp_folder = tempfile.mkdtemp()

        memmap_paths_slope = os.path.join(temp_folder, 'slope.map')
        memmap_paths_offset = os.path.join(temp_folder, 'offset.map')
        memmap_paths_residual_mean = os.path.join(temp_folder, 'residual_mean.map')
        NRCELLSPERCHID = config.nrCellsPerChid[drs_value_type]

        try:
            with h5py.File(interval_file_path, 'r') as interval_source:
                data = interval_source[groupname]
                low_limit = data.attrs['LowLimit']
                upp_limit = data.attrs['UppLimit']
                interval_indices = np.array(data['IntervalIndices'])
                cut_off_error_factor = interval_source.attrs['CutOff'+drs_value_type]
                cut_off_error_factor_values.append(cut_off_error_factor)
                mask = np.array(interval_source[groupname][drs_value_type+'Mask'])

            slope = np.memmap(memmap_paths_slope, mode='w+',
                              shape=NRCHID*NRCELLSPERCHID, dtype='float32')
            offset = np.memmap(memmap_paths_offset, mode='w+',
                               shape=NRCHID*NRCELLSPERCHID, dtype='float32')
            residual_mean = np.memmap(memmap_paths_residual_mean, mode='w+',
                                      shape=NRCHID*NRCELLSPERCHID, dtype='float32')
            slope[:] = np.nan
            offset[:] = np.nan
            residual_mean[:] = np.nan
            del slope
            del offset
            del residual_mean

            chunk = int(NRCHID*NRCELLSPERCHID/NRTEMPSENSOR)
            with h5py.File(source_file_path, 'r') as data_source:
                pool(delayed(fit)(
                     chunk,
                     data_source['Temp'+drs_value_type][interval_indices, int(pice_nr)],
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
                     ) for pice_nr in range(NRTEMPSENSOR))

            # reload fit results
            drs_value_slope = np.memmap(memmap_paths_slope, mode='r+',
                                        shape=NRCHID*NRCELLSPERCHID, dtype='float32')
            drs_value_offset = np.memmap(memmap_paths_offset, mode='r+',
                                         shape=NRCHID*NRCELLSPERCHID, dtype='float32')
            drs_value_residual_mean = np.memmap(memmap_paths_residual_mean, mode='r+',
                                                shape=NRCHID*NRCELLSPERCHID, dtype='float32')

            value_unit = 'mV'
            # catch up gain standardization to 1
            if(drs_value_type == "Gain"):
                drs_value_slope /= DACfactor
                drs_value_offset /= DACfactor
                drs_value_residual_mean /= DACfactor
                value_unit = '1'

            new_columns = fits.ColDefs(
                [fits.Column(
                    name=drs_value_type+'Slope',
                    format='{}E'.format(NRCHID*NRCELLSPERCHID),
                    unit=value_unit+'/celsius',
                    dim='[[{}*{}]]'.format(NRCHID, NRCELLSPERCHID),
                    array=[drs_value_slope]),
                 fits.Column(
                    name=drs_value_type+'Offset',
                    format='{}E'.format(NRCHID*NRCELLSPERCHID),
                    unit=value_unit,
                    dim='[[{}*{}]]'.format(NRCHID, NRCELLSPERCHID),
                    array=[drs_value_offset]),
                 fits.Column(
                    name=drs_value_type+'Residual',
                    format='{}E'.format(NRCHID*NRCELLSPERCHID),
                    unit=value_unit,
                    dim='[[{}*{}]]'.format(NRCHID, NRCELLSPERCHID),
                    array=[drs_value_residual_mean])])
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
    comment_str = ("All "+str(drs_value_types)+"-values with a bigger error than the 'CutOff-ErrorFactor'" +
                   " multiplied with the mean of the error from all collected values of this type" +
                   " will not used for the fit")
    primary.header.insert('EXTEND', ('Comment', comment_str), after='True')
    primary.header.insert('Comment', ('CutOff', str(cut_off_error_factor_values),
                                      'Shortform of CutOffErrorFactor'), after=True)
    comment_str = "Number of the interval"
    primary.header.insert("CutOff", ("IntNR", str(interval_nr), comment_str), after=True)
    comment_str = "Date-String of the lower interval limit"  # in the format 'yyyy-mm-dd hh'
    primary.header.insert("IntNR", ("LowLimit", low_limit, comment_str), after=True)
    comment_str = "Date-String of the upper interval limit"  # in the format 'yyyy-mm-dd hh'
    primary.header.insert("LowLimit", ("UppLimit", upp_limit, comment_str), after=True)

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


###############################################################################
def check_dependency(source_file_path: str, interval_file_path: str):
    with h5py.File(source_file_path, 'r') as data_source:
        source_creation_date = data_source.attrs['CreationDate']

    with h5py.File(interval_file_path) as interval_source:
        used_source_creation_date = interval_source.attrs['SCDate']

    if(source_creation_date != used_source_creation_date):
        error_str = ("'interval_file_path' is not based" +
                     "on the given 'source_file_path'")
        logging.error(error_str)
        raise Exception(error_str)


###############################################################################
def fit(indice_range, temperature, drs_value_array, mask,
        slope_array, offset_array, residual_mean_array):
    for index in range(indice_range):
        sub_mask = mask[:, index]
        temp = temperature[sub_mask]
        value = drs_value_array[:, index][sub_mask]

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
                logging.error(index, drs_value_array[:, index], error_str)

            finally:
                # dont trust the Garbage Collector, so force to free memory
                del sub_mask
                del temp
                del value

    del temperature
    del drs_value_array
    del mask
    del slope_array
    del offset_array
    del residual_mean_array
    gc.collect()
