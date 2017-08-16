import os
import sys
import h5py
import resource

from astropy.io import fits


###############################################################################
def mem():
    print('Memory usage         : % 2.2f MB' % round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0, 1)
    )


# TODO check safety stuff. maybe remove
###############################################################################
def safety_stuff(store_filename):
    if(os.path.isfile(store_filename)):
        userInput = input("File '"+str(store_filename)+"' allready exist.\n" +
                          " Type 'y' to overwrite File\nYour input: ")
        if(userInput != 'y'):
            sys.exit()

    elif(not os.path.isdir(store_filename[0:store_filename.rfind("/")])):
        print("Folder '", store_filename[0:store_filename.rfind("/")], "' does not exist")
        sys.exit()


###############################################################################
def check_file_match(drs_file_path,
                     interval_file_path=None, fit_file_path=None,
                     residuals_file_path=None, chi2_file_path=None):

    match_flag = True
    with h5py.File(drs_file_path, 'r') as data_source:
        source_creation_date = data_source.attrs['CreationDate']

    if (interval_file_path is not None):
        with h5py.File(interval_file_path) as interval_source:
            used_source_creation_date_i = interval_source.attrs["SCDate"]
        if(source_creation_date != used_source_creation_date_i):
            print("'interval_file_path' is not based on the given 'source_file_path'")
            match_flag = False

    if (fit_file_path is not None):
        with fits.open(fit_file_path, ignoremissing=True, ignore_missing_end=True) as fit_values_tab:
            used_source_creation_date_f = fit_values_tab[0].header["SCDate"]
        if(source_creation_date != used_source_creation_date_f):
            print("'fit_file_path' is not based on the given 'source_file_path'")
            match_flag = False

    if (residuals_file_path is not None):
        with h5py.File(residuals_file_path) as residual_source:
            used_source_creation_date_r = residual_source.attrs["SCDate"]
        if(source_creation_date != used_source_creation_date_r):
            print("'residuals_file_path' is not based on the given 'source_file_path'")
            match_flag = False

    if (chi2_file_path is not None):
        with h5py.File(chi2_file_path) as residual_source:
            used_source_creation_date_c = residual_source.attrs["SCDate"]
        if(source_creation_date != used_source_creation_date_c):
            print("'chi2_file_path' is not based on the given 'source_file_path'")
            match_flag = False

    if(not match_flag):
        sys.exit()
