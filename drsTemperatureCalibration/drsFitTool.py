import pandas as pd
import numpy as np
import yaml
import click
import h5py
import os
import logging

from astropy.io import fits
from tqdm import tqdm
from fact.credentials import create_factdb_engine

from drsTemperatureCalibration.tools import get_linear_fit_values, safety_stuff
from drsTemperatureCalibration.constants import NRPIX, NRCAP, NRTEMPSENSOR

# TODO overwrite @click.argument defaultValues


####################################################################################################
####################################################################################################
@click.command()
@click.argument('drs_file_list_doc_path',
                default="/gpfs0/scratch/schulz/drsFiles.txt",
                type=click.Path(exists=False))
def search_drs_files(drs_file_list_doc_path: str):
    '''
        Search through the fact-database and store the path of all drsFiles under the given storePath

        Args:
            drs_file_list_doc_path (str): Full path to the storeFile with the extension '.txt'
    '''

    print(">> Run 'SearchDrsFiles' <<")

    # TODO check safety stuff. maybe remove
    safety_stuff(drs_file_list_doc_path)

    def filename(row):
        return os.path.join(
            str(row.date.year),
            "{:02d}".format(row.date.month),
            "{:02d}".format(row.date.day),
            "{}_{:03d}.drs.fits.gz".format(row.fNight, row.fRunID),
        )

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
    drs_file_infos["date"] = pd.to_datetime(drs_file_infos.fNight.astype(str), format="%Y%m%d")

    drs_files = drs_file_infos.apply(filename, axis=1).tolist()
    pd.DataFrame(drs_files).to_csv(drs_file_list_doc_path, index=False, header=False)

    # ##############################
    # # #### Filter Drs-Files #### #
    # ##############################
    #
    # print(">> In 'SearchDrsFiles' start with filter Drs-Files <<")
    #
    # selected_files = []
    # previous_dates = []
    # for drs_filename in drs_files:
    #     drs_filename = drs_filename.strip("\n")
    #
    #     date = drs_filename.split("_")[0].split("/")[-1]
    #     year = drs_filename.split("_")[0].split("/")[-4]
    #     if(date not in previous_dates and int(year) >= 2012):
    #
    #         previous_dates.append(date)
    #         selected_files.append(drs_filename)
    #
    # pd.DataFrame(selected_files).to_csv(store_filename_path, index=False, header=False)

    print(">> Finished 'SearchDrsFiles' <<")


####################################################################################################
####################################################################################################
# ##############                         saveDrsAttributes                          ############## #
####################################################################################################
####################################################################################################
@click.command()
@click.argument('drs_file_list_doc_path',
                default="/gpfs0/scratch/schulz/drsFiles.txt",
                type=click.Path(exists=True))
@click.argument('store_filename_path',
                default="/gpfs0/scratch/schulz/drsData.h5",
                type=click.Path(exists=False))
@click.argument('source_pre_path',
                default=("/home/florian/Dokumente/Uni/Master/Masterarbeit/" +
                         "vollmondHome/net/big-tank/POOL/projects/fact/drs_temp_calib_data"),
                type=click.Path(exists=True))
@click.argument('drs_temp_calib_config',
                default="drsTempCalibConfig.yaml",
                type=click.Path(exists=True))
def save_drs_attributes(drs_file_list_doc_path: str, store_filename_path: str,
                        source_pre_path: str, drs_temp_calib_config: str):
    '''
        Save Baseline and Gain of all drsfiles of the given drsFileList
        together with the Temperature and mean of Time of taking
        into a .h5 File

        Args:
            drs_file_list_doc_path (str): Full path to the drsFileList-file with the extension '.txt'
            store_filename_path (str): Full path to the store-file with the extension '.h5'
            source_pre_path (str): Path to the raw- and aux-folder containing the drs- and temperature-files
            drs_temp_calib_config (str): Path to the drsCalibrationConfig-file with the extension '.yaml'
    '''
    print(">> Run 'SaveDrsAttributes' <<")

    # TODO check safety stuff. maybe remove
    safety_stuff(store_filename_path)

    logging.basicConfig(filename=store_filename_path.split('.')[0]+".log", filemode='w',
                        format='%(levelname)s:%(message)s', level=logging.DEBUG)

    NRCALIBVALUES = NRPIX*NRCAP

    with open(drs_temp_calib_config) as calbConfig:
        config = yaml.safe_load(calbConfig)

    drs_value_types = config["drsValueTypes"]

    # add columns to h5py table
    with h5py.File(store_filename_path, 'w') as hf:
        hf.create_dataset('CreationDate', (1, 1), dtype='S19', maxshape=(1, 1),
                          compression="gzip", compression_opts=9, fletcher32=True)

        for drs_value_type in drs_value_types:
            hf.create_dataset("Time"+drs_value_type,    (0, 1), maxshape=(None, 1),
                              compression="gzip", compression_opts=9, fletcher32=True)
            hf.create_dataset("Temp"+drs_value_type,    (0, NRTEMPSENSOR), maxshape=(None, NRTEMPSENSOR),
                              compression="gzip", compression_opts=9, fletcher32=True)
            # not necessary for drs calibration
            hf.create_dataset("TempStd"+drs_value_type, (0, NRTEMPSENSOR), maxshape=(None, NRTEMPSENSOR),
                              compression="gzip", compression_opts=9, fletcher32=True)
            hf.create_dataset(drs_value_type+"Mean",    (0, NRCALIBVALUES), maxshape=(None, NRCALIBVALUES),
                              compression="gzip", compression_opts=9, fletcher32=True)
            # used but check is necessary
            hf.create_dataset(drs_value_type+"MeanStd", (0, NRCALIBVALUES), maxshape=(None, NRCALIBVALUES),
                              compression="gzip", compression_opts=9, fletcher32=True)

    # TODO update text
    # This loop basically iterate over the drsFiles of the drsFileList and
    # check if there are all needed information/ tuple of attributes for later calculations
    # based on the selected drsFile
    # in case they are there, they will simply stored into a h5py table
    # in the case it is impossible to to collect all needed attributes
    # we will continue with the next drsFile and if there is a diagnosticFilname given
    # all .h5 File ...

    drs_file_list = open(drs_file_list_doc_path).read().splitlines()
    for drs_filename in tqdm(drs_file_list):
        drs_filename = source_pre_path+"raw/"+drs_filename.strip("\n")

        date_path_part = drs_filename.split('_')[0]
        temp_filename = source_pre_path+"aux/{}.FAD_CONTROL_TEMPERATURE.fits".format(date_path_part)

        if(os.path.isfile(drs_filename) and os.path.isfile(temp_filename)):
            try_to_save_tuple_of_attributes(drs_filename, temp_filename, store_filename_path, drs_value_types)

    # update h5py creationDate
    creation_date_str = pd.datetime.now().strftime('%Y-%m-%d %H:%M:%S').encode("UTF-8", "ignore")
    with h5py.File(store_filename_path) as store:
        store['CreationDate'][0] = [creation_date_str]

    print(">> Finished 'SaveDrsAttributes' <<")


def try_to_save_tuple_of_attributes(temp_filename, drs_filename, store_filename_path, drs_value_types):
    try:
        save_tuple_of_attribute_if_possible(temp_filename, drs_filename, store_filename_path, drs_value_types)
    except:
        logging.exception()
        return


def save_tuple_of_attribute_if_possible(temp_filename, drs_filename, store_filename_path, drs_value_types):

    tabDrs = fits.open(
        drs_filename,
        ignoremissing=True,
        ignore_missing_end=True)
    header = tabDrs[1].header
    bintable = tabDrs[1].data

    temps_of_runs = read_temps_of_runs(
        temp_filename,
        runTimesList=[
            (
                pd.to_datetime(header["RUN0-BEG"]),
                pd.to_datetime(header["RUN0-END"])
            ),
            (
                pd.to_datetime(header["RUN1-BEG"]),
                pd.to_datetime(header["RUN1-END"])
            ),
        ])

    for drs_value_type in drs_value_types:
        drs_value_mean = bintable[drs_value_type+"Mean"][0]
        drs_value_mean_std = bintable[drs_value_type+"Rms"][0]

        check_for_nulls(drs_value_mean, drs_value_type+"Mean", drs_filename)
        check_for_nulls(drs_value_mean_std, drs_value_type+"MeanStd", drs_filename)

    with h5py.File(store_filename_path) as table:
        for drs_value_type in drs_value_types:
            add_value_to_h5py_table(table, "Time"+drs_value_type, temps_of_runs[0]['mean_time'])
            add_value_to_h5py_table(table, "Temp"+drs_value_type, temps_of_runs[0]['mean_temp'])
            add_value_to_h5py_table(table, "TempStd"+drs_value_type, temps_of_runs[0]['std_temp'])
            add_value_to_h5py_table(table, drs_value_type+"Mean", drs_value_mean)
            add_value_to_h5py_table(table, drs_value_type+"MeanStd", drs_value_mean_std)


def add_value_to_h5py_table(h5pyTable, columnName, value):
    data = h5pyTable[columnName]
    data.resize((len(data)+1, data.maxshape[1]))
    data[len(data)-1, :] = value


def read_temps_of_runs(path, runTimesList):
    '''
    RunTimesList a list() of tuples of (start, end) times
    between which we want to read the 'time' and 'temp' arrays
    from the fits file under `path`.
    '''

    table = fits.open(
        path,
        ignoremissing=True,
        ignore_missing_end=True)

    table_time = table[1].data["Time"]
    table_temperature = table[1].data["temp"]

    if table_temperature.shape[1] != NRTEMPSENSOR:
        message = (
            "File not used: Just "+str(table_temperature.shape[1]) +
            " Temperature Values in File '"+path+"'")
        raise Exception(message)

    results = []
    table_datetime = pd.to_datetime(table_time * 24 * 3600 * 1e9)
    for start, end in runTimesList:
        idx = np.where(
            (table_datetime > start) &
            (table_datetime < end)
            )[0]
        timestamps_during_run = np.array(table_time[idx])
        temperature_during_run = np.array(table_temperature[idx])

        if timestamps_during_run.shape[0] > 1:
            mean_time = np.mean(timestamps_during_run, dtype="float64")
        else:
            mean_time = timestamps_during_run

        if temperature_during_run.shape[0] > 1:
            mean_temp = np.mean(temperature_during_run, dtype="float64", axis=0)
            std_temp = np.std(temperature_during_run, dtype="float64", axis=0, ddof=1)
        else:
            mean_temp = temperature_during_run
            std_temp = np.zeros(temperature_during_run.shape[1])

        results.append(dict(
            mean_time=mean_time,
            mean_temp=mean_temp,
            std_temp=std_temp,
            )
        )
    return results


def check_for_nulls(array, name, path):
    nulls = np.where(array == 0.)[0]
    if len(nulls):
        raise Exception(
            ("File not used: Nulls of {name} \n" +
             "in File {path} at index {idx}").format(
                name=name,
                path=path,
                idx=str(nulls)
            )
        )


####################################################################################################
####################################################################################################
# ##############                           saveFitValues                            ############## #
####################################################################################################
####################################################################################################
@click.command()
@click.argument('source_filename_path',
                default="/gpfs0/scratch/schulz/drsData.h5",
                type=click.Path(exists=True))
@click.argument('store_filename_path',
                default="/gpfs0/scratch/schulz/fitValues/fitValuesDataInterval.fits",
                type=click.Path(exists=False))
@click.argument('drs_temp_calib_config',
                default="drsTempCalibConfig.yaml",
                type=click.Path(exists=True))
def save_fit_values(source_filename_path: str, store_filename_path: str,
                    drs_temp_calib_config: str):
    '''
        Calculate the linear fitvalues of Basline and Gain based on the .h5 source data for the
        by the hardware boundaries given itervals and store them into a .fits File
        All Basline/Gain-values with a bigger error than the 'CutOffErrorFactor'
        multiplied with the mean of the error from all collected Baseline/Gain-values of the
        Capacitor will not used for the fit

        Args:
            source_filename_path (str): Full path to the sourceParameter file with the extension '.h5'
            store_filename_path (str): Full path to the storeFile with the extension '.fits'
            drs_temp_calib_config (str): Path to the drsCalibrationConfig-file with the extension '.yaml'
    '''

    print(">> Run 'SaveFitValues' <<")

    # TODO check safety stuff. maybe remove
    safety_stuff(store_filename_path)

    logging.basicConfig(filename=store_filename_path.split('.')[0]+".log", filemode='w',
                        format='%(levelname)s:%(message)s', level=logging.DEBUG)

    with open(drs_temp_calib_config) as calbConfig:
        config = yaml.safe_load(calbConfig)

    drs_value_types = config["drsValueTypes"]
    cut_off_error_factor = config["cutOffErrorFactor"]
    hardware_boundaries = config["hardwareBoundaries"]

    # TODO write comment +12h
    hardware_boundaries = pd.to_datetime(hardware_boundaries)+pd.DateOffset(hours=12)

    for drs_value_type in drs_value_types:
        with h5py.File(source_filename_path, 'r') as dataSource:
            time = np.array(dataSource["Time"+drs_value_type]).flatten()
            date = pd.to_datetime(time * 24 * 3600 * 1e9)

        list_of_interval_indices = get_boundarie_based_interval_indices(date, hardware_boundaries)

        print("Loading ...")
        tbhdus = []
        # iterate over the number of hardware based intervals
        for interval_indices in list_of_interval_indices:
            with h5py.File(source_filename_path, 'r') as dataSource:
                creation_date = dataSource["CreationDate"][0][0].decode("UTF-8")

                drs_value_temp_array = np.array(dataSource["Temp"+drs_value_type][interval_indices, :])
                drs_value_mean_array = np.array(dataSource[drs_value_type+"Mean"][interval_indices, :])
                drs_value_mean_std_array = np.array(dataSource[drs_value_type+"MeanStd"][interval_indices, :])

            drs_value_mean_slope = []
            drs_value_mean_slope_std = []
            drs_value_mean_offset = []
            drs_value_mean_offset_std = []

            print("Calculate fitvalues of "+drs_value_type+" for '"+str(NRPIX)+"' Pixel \n")  # +
                  # "for the period from "+str(firstDate)+" until "+str(lastDate))

            for pixelNr in tqdm(range(NRPIX)):

                drs_value_temp = drs_value_temp_array[:, int(pixelNr/9)]
                for capNr in range(NRCAP):
                    drs_value_mean_cap = drs_value_mean_array[:, pixelNr*NRCAP+capNr]
                    drs_value_mean_std_cap = drs_value_mean_std_array[:, pixelNr*NRCAP+capNr]
                    drs_value_mean_std_cap_mean = np.mean(drs_value_mean_std_cap, dtype="float")

                    try:
                        indices = np.where(drs_value_mean_std_cap <
                                           drs_value_mean_std_cap_mean*cut_off_error_factor[drs_value_type])[0]
                        var_drs_value, cov_drs_value = get_linear_fit_values(drs_value_temp[indices],
                                                                             drs_value_mean_cap[indices],
                                                                             drs_value_mean_std_cap[indices])

                        drs_value_mean_slope.append(var_drs_value[0])
                        drs_value_mean_slope_std.append(np.sqrt(cov_drs_value[0][0]))
                        drs_value_mean_offset.append(var_drs_value[1])
                        drs_value_mean_offset_std.append(np.sqrt(cov_drs_value[1][1]))

                    except Exception as err_infos:
                        error_str = (drs_value_type+"-Fit(PixelNr: "+str('{:04d}'.format(pixelNr)) +
                                     ", capNr: "+str('{:04d}'.format(capNr)) +
                                     ") with Slope="+str(var_drs_value[0]) +
                                     " and Offset='"+str(var_drs_value[1])+": '"+str(err_infos)+"'")
                        logging.warning(error_str)

                        drs_value_mean_slope.append(var_drs_value[0])
                        drs_value_mean_slope_std.append(0)
                        drs_value_mean_offset.append(var_drs_value[1])
                        drs_value_mean_offset_std.append(0)

            tbhdu = fits.BinTableHDU.from_columns(
                    [fits.Column(name="Slope",       format=str(len(drs_value_mean_slope))+'E',
                                 unit="mV/celsius",  array=[drs_value_mean_slope]),
                     # not necessary for drs calibration
                     fits.Column(name="SlopeStd",    format=str(len(drs_value_mean_slope_std))+'E',
                                 unit="mV/celsius",  array=[drs_value_mean_slope_std]),
                     fits.Column(name="Offset",      format=str(len(drs_value_mean_offset))+'E',
                                 unit="mV",          array=[drs_value_mean_offset]),
                     # not necessary for drs calibration
                     fits.Column(name="OffsetStd",   format=str(len(drs_value_mean_offset_std))+'E',
                                 unit="mV",          array=[drs_value_mean_offset_std])])
            tbhdu.header.insert("TFIELDS", ("EXTNAME", drs_value_type), after=True)
            commentStr = ("All "+drs_value_type+"-values with a bigger error than the 'CutOff-ErrorFactor'" +
                          " multiplied with the mean of the error from all collected "+drs_value_type+"-values" +
                          " will not used for the fit")
            tbhdu.header.insert("EXTNAME", ("Comment", commentStr), after="True")
            tbhdu.header.insert("Comment", ("CutOff", str(cut_off_error_factor[drs_value_type]),
                                            "Shortform of CutOffErrorFactor"), after=True)

            tbhdus.append(tbhdu)

    print("Write Data to Table")
    primary = fits.PrimaryHDU()
    commentStr = "Datetime-String of the source .h5 creation."  # in the format 'yyyy-mm-dd HH:MM:SS'
    primary.header.insert("EXTEND", ("OrigDate", creation_date, commentStr), after=True)
    #commentStr = "Date-String of the lower interval limit"  # [firstDate, lastDate]"  in the format 'yyyy-mm-dd'
    #primary.header.insert("OrigDate", ("LowLimit", firstDate.strftime('%Y-%m-%d'), commentStr), after=True)
    #commentStr = "Date-String of the upper interval limit"  # [firstDate, lastDate]"  in the format 'yyyy-mm-dd'
    #primary.header.insert("LowLimit", ("UppLimit", lastDate.strftime('%Y-%m-%d'), commentStr), after=True)

    print("Save Table")
    thdulist = fits.HDUList([primary, tbhdus[0], tbhdus[1]])
    thdulist.writeto(store_filename_path, overwrite=True, checksum=True)
    print("Verify Checksum")
    # Open the File verifying the checksum values for all HDUs
    try:
        hdul = fits.open(store_filename_path, checksum=True)
        print(hdul[0].header)
        print(hdul["Baseline"].header)
        print(hdul["Gain"].header)
        with open(store_filename_path.split('.')[0]+".log", 'r') as logFile:
            if(logFile.readlines() == []):
                logging.info(" No errors occurred during the Fit-Value calculation.")
        logging.info(" Passed verifying Checksum")
    except Exception as err_infos:
        error_str = str(err_infos)
        print(error_str)
        logging.warning(error_str)

    print(">> Finished 'SaveFitValues' <<")


def get_boundarie_based_interval_indices(list_of_dates, list_of_boundaries):
    """Split the given list_of_dates into intervals based on the given boundaries
    """
    lower_boundarie = list_of_dates[0]
    list_of_interval_indices = []
    for boundarie in list_of_boundaries:
        interval_indices = np.where((list_of_dates > lower_boundarie) & (list_of_dates <= boundarie))[0]
        list_of_interval_indices.append(interval_indices)
        lower_boundarie = list_of_dates[interval_indices[-1]]
    list_of_interval_indices.append(np.where(list_of_dates > lower_boundarie)[0])

    return list_of_interval_indices
