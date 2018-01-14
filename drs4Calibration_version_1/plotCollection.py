import plotDrsAttributes as plot
import numpy as np

pre_path = ('/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
            'calibration/calculation/')

pre_store_path = '/home/fschulz/plots/version_1/'

data_collection_path = pre_path+'/version_1/dataCollection.h5'

interval_file_path = pre_path+'version_1/intervalIndices.h5'

fit_file_path_array = [pre_path+'version_1/drsFitParameter_interval3.fits']


################################################################################
# ############## ###############   Version 1   ############### ############### #
################################################################################

def plot_drs_value_std_hist():
    print('Version 1: drs_value_std_hist')

    versions = [['gain/std_hist_gain.png', [3], 'Gain', [0.0, 8.0]]]

    for store_path, interval_array, drs_value_type, x_lim in versions:
        plot.drs_value_std_hist_(data_collection_path,
                                 interval_file_path,
                                 (pre_store_path+'drsValues/'+store_path),
                                 interval_array,
                                 drs_value_type,
                                 x_lim)


################################################################################
def plot_drs_value_std_hist_per_chid_cell():
    print('Version 1: drs_value_std_hist_per_chid_cell')
    versions = [['gain/std_hist_gain_chid250_cell250.png', [3], 'Gain', 250, 250, 2, [0.0, 10.0]]]

    for store_path, interval_array, drs_value_type, chid, cell, cut_off_error_factor, x_lim in versions:
        plot.drs_value_std_hist_per_chid_cell_(data_collection_path,
                                               interval_file_path,
                                               (pre_store_path+'drsValues/'+store_path),
                                               interval_array,
                                               drs_value_type,
                                               chid,
                                               cell,
                                               cut_off_error_factor,
                                               x_lim)


################################################################################
def plot_drs_value_cell():
    print('Version 1: drs_value_cell')
    versions = [['gain/fitparameter_gain_chid250_cell250_interval1_2_3.png', [3], 'Gain', 250, 250]]

    for store_path, interval_array, drs_value_type, chid, cell, in versions:
        plot.drs_value_cell_(data_collection_path,
                             interval_file_path,
                             fit_file_path_array,
                             (pre_store_path+'drsValues/'+store_path),
                             interval_array,
                             drs_value_type,
                             chid,
                             cell)


################################################################################
def plot_drs_model_parameter_hist():
    print('Version 1: drs_model_parameter_hist')

    versions = [['baseline/fit_parameter_offset_hist_baseline.png', 'Baseline', 'Offset', [-840, -980]],
                ['baseline/fit_parameter_slope_hist_baseline.png', 'Baseline', 'Slope', [-0.6, 0.2]],
                ['gain/fit_parameter_offset_hist_gain.png', 'Gain', 'Offset', [0.85, 0.98]],
                ['gain/fit_parameter_slope_hist_gain.png', 'Gain', 'Slope', [-1.15, 0.75]]]

    for store_path, drs_value_type, fit_parameter_type, x_lim in versions:
        plot.drs_model_parameter_hist_(fit_file_path_array,
                                       (pre_store_path+'model/'+store_path),
                                       drs_value_type,
                                       fit_parameter_type,
                                       x_lim)


################################################################################
def plot_drs_fit_value_residual_hist():
    print('Version 1: drs_fit_value_residual_hist')

    versions = [['baseline/residual_hist_baseline.png', 'Baseline', [0.0, 1.5]],
                ['gain/residual_hist_gain.png', 'Gain', [0.0, 0.4]]]

    for store_path, drs_value_type, x_lim in versions:
        plot.drs_fit_value_residual_hist_(fit_file_path_array,
                                          (pre_store_path+'model/'+store_path),
                                          drs_value_type,
                                          x_lim)


################################################################################
def plot_residual_fact_cam():
    print('Version 1: residual_fact_cam')

    versions = [['baseline/residual_fact_cam_int3_baseline.png', 3, 'Baseline', []],
                ['gain/residual_fact_cam_int3_gain.png', 3, 'Gain', []]]

    for store_path, interval_nr, drs_value_type, worst_chids in versions:
        plot.residual_fact_cam_(fit_file_path_array[0],
                                (pre_store_path+'model/'+store_path),
                                drs_value_type,
                                worst_chids)


################################################################################
def plot_pedestel_mean_or_std_vs_temp():
    print('Version 0: pedestel_mean_or_std_vs_temp')

    versions = [['calibratedPedestelDataDistribution_mean.png', 'Mean', 3, ['2015-05-26', '2017-10-01']],
                ['calibratedPedestelDataDistribution_std.png', 'Std', 3, ['2015-05-26', '2017-10-01']]]

    for store_path, calibrated_type, interval_nr, datetime_limits in versions:
        plot.pedestel_mean_or_std_vs_temp_((pre_path+'../validating/version_0/meanAndStd/interval{}/'.format(interval_nr)),
                                           (pre_store_path+'meanAndStd/interval{}/'.format(interval_nr)+store_path),
                                           calibrated_type,
                                           interval_nr,
                                           datetime_limits)


################################################################################
if __name__ == '__main__':
    plot_drs_value_std_hist()
    #plot_drs_value_std_hist_per_chid_cell()
    #plot_drs_model_parameter_hist()
    #plot_drs_fit_value_residual_hist()
    #plot_residual_fact_cam()

    #plot_pedestel_mean_or_std_vs_temp()
