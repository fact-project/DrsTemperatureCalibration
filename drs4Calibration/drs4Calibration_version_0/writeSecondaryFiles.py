import subprocess as sp
import pandas as pd
import numpy as np
import h5py
import click
import os
import sys

from tqdm import tqdm
from astropy.io import fits
from joblib import Parallel, delayed

from fact.credentials import create_factdb_engine

import config as config
from constants import NRCHID


@click.command()
@click.argument('source_folder_path',
                default=('/net/big-tank/POOL/projects/fact/' +
                         'drs4_calibration_data/'),
                type=click.Path(exists=True))
@click.argument('store_folder_path',
                default=('/net/big-tank/POOL/projects/fact/' +
                         'drs4_calibration_data/calibration/validating/version_0/meanAndStd/interval2/'),
                type=click.Path(exists=True))
@click.argument('facttools_file_path',
                default=('/home/fschulz/git/fact-tools_drs/' +
                         'target/fact-tools-0.18.1-SNAPSHOT.jar'),
                type=click.Path(exists=True))
@click.argument('facttools_xml_path',
                default=('/home/fschulz/git/fact-tools_drs/' +
                         'examples/drsCalibration/drsCalibrationMeanAndStd.xml'),
                type=click.Path(exists=True))
@click.argument('fitparameter_file_path_temp',
                default=('/net/big-tank/POOL/projects/fact/' +
                         'drs4_calibration_data/calibration/calculation/' +
                         'version_0/drsFitParameter_interval2.fits'),
                type=click.Path(exists=True))
@click.argument('time_interval',
                default=['2014-05-20', '2015-05-26'])  # ['2014-05-20', '2015-05-26'], ['2015-05-26', '2017-10-01']
###############################################################################
def drs_pedestal_run_mean_and_std(source_folder_path, store_folder_path,
                                  facttools_file_path, facttools_xml_path,
                                  fitparameter_file_path_temp, time_interval):
    jobs = 15
    verbosity = 10

    pool = Parallel(n_jobs=jobs, verbose=verbosity, max_nbytes='50G')
    with fits.open(fitparameter_file_path_temp) as tab:
        interval_limits = [tab[0].header['LowLimit'], tab[0].header['UppLimit']]

    # if ((pd.to_datetime(time_interval[0]) < pd.to_datetime(interval_limits[0]).replace(hour=0)) or
    #     (pd.to_datetime(time_interval[1]) > pd.to_datetime(interval_limits[1]).replace(hour=0))):
    #     print('Input range [{}, {}] '.format(time_interval[0], time_interval[1]) +
    #           'is out of interval range [{}, {}]'.format(interval_limits[0],
    #                                                      interval_limits[1]))
    #     sys.exit()

    print('Loading Database ...')
    # TODO maybe query start_date an end_date with fNight
    db_table = pd.read_sql(
                    'RunInfo',
                    create_factdb_engine(),
                    columns=[
                        'fNight', 'fRunID',
                        'fRunTypeKey', 'fDrsStep',
                        'fNumEvents', 'fBiasVoltageMedian'])

    # loop over the start_date to end_date interval
    pre_filename = 'pedestelStats_'
    for date in tqdm(pd.date_range(start=time_interval[0], end=time_interval[1], freq='D')):

        date_str = date.strftime('%Y%m%d')

        date_path = date.strftime('%Y/%m/%d/')
        pre_aux_path = source_folder_path+'aux/'+date_path
        pre_drs_path = source_folder_path+'raw/'+date_path

        temp_file = (pre_aux_path+date_str+'.FAD_CONTROL_TEMPERATURE.fits')

        # skip calculation if no temperatur file exist
        if(not os.path.isfile(temp_file)):
            print('Date: ', date_str, ' has no temp file')  # TODO maybe log
            continue

        selected_drs_infos = db_table.query('fNight =='+str(date_str)+'&' +
                                            'fRunTypeKey == 2 &' +
                                            'fDrsStep == 2 &' +
                                            'fNumEvents == 1000').copy()

        selected_drs_infos['date'] = pd.to_datetime(
                                        selected_drs_infos.fNight.astype(str),
                                        format='%Y%m%d')

        drs_run_id_list = selected_drs_infos['fRunID'].tolist()
        existing_drs_run_ids = []
        existing_drs_run_files = []
        for run_id in drs_run_id_list:
            drs_run_filename = (pre_drs_path+date_str +
                                '_'+str('{:03d}'.format(run_id))+'.drs.fits.gz')
            if(os.path.isfile(drs_run_filename)):
                existing_drs_run_ids.append(run_id)
                existing_drs_run_files.append(drs_run_filename)
            else:
                print(drs_run_filename, ' not found')

        if(len(existing_drs_run_files) == 0):
            print('Date: ', date_str, ' no drs files found')  # TODO maybe log
            continue

        # just use one drs-Run for the calculations
        # to afford larger temperature differences
        # (all pedestal-runs follow right after
        # the drs-run taking, so there are just small
        # temperature differences)
        # so we take the drs-Run of the middle of the night
        # keep im mind other influences in time can now appear
        # and distort the results

        drs_run_index = int(len(existing_drs_run_files)/2)
        drs_run_id = existing_drs_run_ids[drs_run_index]
        drs_file = existing_drs_run_files[drs_run_index]

        # Searching pedestal_runs

        # fDrsStep == NaN and fBiasVoltageMedian == NaN dosent work
        nr_of_pedestal_events = 1000
        selected_drs_infos = db_table.query(
                                'fNight == '+str(date_str)+'&' +
                                'fRunTypeKey == 2 &' +
                                'fDrsStep != fDrsStep &' +
                                'fBiasVoltageMedian != fBiasVoltageMedian &'
                                'fNumEvents == '+str(nr_of_pedestal_events)
                             ).copy()

        selected_drs_infos['date'] = pd.to_datetime(selected_drs_infos.fNight.astype(str), format='%Y%m%d')
        pedestal_run_id_list = selected_drs_infos['fRunID'].tolist()

        existing_pedestal_run_ids = []
        existing_pedestal_run_files = []
        for run_id in pedestal_run_id_list:
            pedestal_run_filename = (pre_drs_path+date_str+'_'+str('{:03d}'.format(run_id))+'.fits.fz')
            if(os.path.isfile(pedestal_run_filename)):
                existing_pedestal_run_ids.append(run_id)
                existing_pedestal_run_files.append(pedestal_run_filename)
            else:
                print(pedestal_run_filename, ' not found')

        if(existing_pedestal_run_files == []):
            continue

        with fits.open(temp_file) as tempTab:
            timeList = np.array(tempTab[1].data['Time'])
            temp_list = np.array(tempTab[1].data['temp'])
            tempDatetime = pd.to_datetime(timeList * 24 * 3600 * 1e9)

        with fits.open(drs_file) as drsTab:
            drsStart = pd.to_datetime(drsTab[1].header['DATE-OBS'])
            drsEnd = pd.to_datetime(drsTab[1].header['DATE-END'])
            # mean ignore patches -->, axis=0 <--
            drsTempMean = np.mean(temp_list[np.where((tempDatetime > drsStart) & (tempDatetime < drsEnd))])

        store_folder_path_tmp = store_folder_path+date_str+'/'
        if not os.path.exists(store_folder_path_tmp):
            os.makedirs(store_folder_path_tmp)

        nr_runs_of_the_day = len(existing_pedestal_run_files)
        temp_diff_list = [np.nan] * nr_runs_of_the_day
        store_file_list = [np.nan] * nr_runs_of_the_day
        for run_index in range(nr_runs_of_the_day):
            run_file = existing_pedestal_run_files[run_index]
            run_id = existing_pedestal_run_ids[run_index]

            with fits.open(run_file) as run_tab:
                run_start = run_tab[2].header['DATE-OBS']
                run_end = run_tab[2].header['DATE-END']

            run_temp = temp_list[np.where((tempDatetime > run_start) & (tempDatetime < run_end))[0]]

            if(len(run_temp) == 0):
                run_temp = temp_list[np.where((tempDatetime < run_start))[0][-1]:
                                     np.where((tempDatetime > run_end))[0][0]+1]

            temp_diff_list[run_index] = np.mean(run_temp) - drsTempMean

            store_file_path = (store_folder_path_tmp+pre_filename +
                               date_str+'_{0:03d}'.format(run_id)+'_tmp.fits')
            store_file_list[run_index] = store_file_path

        # pool(delayed(run_fact_tools)(
        #      facttools_file_path,
        #      facttools_xml_path,
        #      existing_pedestal_run_files[run_index],
        #      store_file_list[run_index],
        #      drs_file,
        #      pre_aux_path,
        #      fitparameter_file_path_temp
        #      ) for run_index in range(nr_runs_of_the_day))

        print('Join stats.fits of ', date_str)

        drs_calibrated_data_mean = []
        drs_calibrated_data_std = []
        drs_calibrated_data_mean_temp = []
        drs_calibrated_data_std_temp = []

        used_pedestal_run_ids = []
        used_temperature_differences = []
        data_shape = NRCHID * nr_of_pedestal_events
        for run_id in existing_pedestal_run_ids:
            print('Try to add run ID: ', run_id)
            source_file = (store_folder_path_tmp+pre_filename +
                           date_str+'_{0:03d}'.format(run_id)+'_tmp.fits')
            try:
                with fits.open(source_file) as stats_tab:
                    data_mean = stats_tab[1].data['DRSCalibratedData_Mean'].flatten()
                    data_std = stats_tab[1].data['DRSCalibratedData_Std'].flatten()
                    data_mean_temp = stats_tab[1].data['DRSCalibratedData_Temp_Mean'].flatten()
                    data_std_temp = stats_tab[1].data['DRSCalibratedData_Temp_Std'].flatten()

                if((len(data_mean) == data_shape) and (len(data_std) == data_shape) and
                   (len(data_mean_temp) == data_shape) and (len(data_std_temp) == data_shape)):
                        drs_calibrated_data_mean.append(data_mean)
                        drs_calibrated_data_std.append(data_std)
                        drs_calibrated_data_mean_temp.append(data_mean_temp)
                        drs_calibrated_data_std_temp.append(data_std_temp)
                        used_pedestal_run_ids.append(run_id)
                        run_id_index = existing_pedestal_run_ids.index(run_id)
                        used_temperature_differences.append(temp_diff_list[run_id_index])
                else:
                    error_str = ("Incomplete run")
                    #logging.error(error_str)
                    raise Exception(error_str)
            except Exception as errInfos:
                print('Unable to add run ID: ', run_id, '-', str(errInfos))

                # os.remove(source_file)
        # os.rmdir(store_folder_path_tmp)
        # if(len(used_pedestal_run_ids) == 0):
        #     continue

        print('Write Data to Table')
        tbhduStats = fits.BinTableHDU.from_columns(
                [fits.Column(
                    name='PedestelRunId', format='1E', # format='1I' for int dosent work
                    unit='1', array=used_pedestal_run_ids),
                 fits.Column(
                    name='TempDiff', format='1E',
                    unit='Degree C', array=used_temperature_differences),
                 fits.Column(
                    name='DRSCalibratedData_Mean', format='{}E'.format(data_shape),
                    unit='mV', array=drs_calibrated_data_mean),
                 fits.Column(
                    name='DrsCalibratedData_Std', format='{}E'.format(data_shape),
                    unit='mV', array=drs_calibrated_data_std),
                 fits.Column(
                    name='DrsCalibratedData_Temp_Mean', format='{}E'.format(data_shape),
                    unit='mV', array=drs_calibrated_data_mean_temp),
                 fits.Column(
                    name='DrsCalibratedData_Temp_Std', format='{}E'.format(data_shape),
                    unit='mV', array=drs_calibrated_data_std_temp)])
        tbhduStats.header.insert('TFIELDS', ('EXTNAME', 'StatsPerChid'), after=True)
        commentStr = ('-')
        tbhduStats.header.insert('EXTNAME', ('comment', commentStr), after='True')
        tbhduStats.header.insert('comment', ('Date', date_str, 'Date yyyy-mm-dd'), after=True)
        tbhduStats.header.insert('Date', ('DrsRunId', drs_run_id, 'RunID of the based drsrun_file'), after=True)

        store_file_path = store_folder_path+pre_filename+date_str+'_.fits'
        print('Save Table')
        thdulist = fits.HDUList([fits.PrimaryHDU(), tbhduStats])
        thdulist.writeto(store_file_path, overwrite=True, checksum=True)
        print('Verify Checksum')
        # Open the File verifying the checksum values for all HDUs
        try:
            hdul = fits.open(store_file_path, checksum=True)
            print(hdul['StatsPerChid'].header)
        except Exception as errInfos:
            errorStr = str(errInfos)
            print(errorStr)


###############################################################################
def run_fact_tools(facttools_file_path, facttools_xml_path, run_file,
                   store_file_path, drs_file, pre_aux_path,
                   fitparameter_file_path_temp):

    sp.run(['java', '-jar', '{}'.format(facttools_file_path),
            '{}'.format(facttools_xml_path),
            '-Dinfile=file:{}'.format(run_file),
            '-Doutfile=file:{}'.format(store_file_path),
            '-Ddrsfile=file:{}'.format(drs_file),
            '-DauxFolder=file:{}'.format(pre_aux_path),
            '-DfitParameterFile_Temp=file:{}'.format(fitparameter_file_path_temp),
            '-j8'])
