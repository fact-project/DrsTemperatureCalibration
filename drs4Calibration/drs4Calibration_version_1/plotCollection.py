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
    versions = [[[3], 'Gain', 250, 250, []],
                [[3], 'Baseline', 98, 93, []],
                [[3], 'Gain', 98, 93, []],
                [[3], 'Baseline', 1275, 31, []],
                [[3], 'Gain', 1275, 31, []]]
    for interval_array, drs_value_type, chid, cell, ylimits in versions:
        array_str = '3'
        filename = (pre_store_path+'drsValues/{}/'.format(drs_value_type.lower()) +
                    'fit_parameter_{}_chid{}_cell_{}_interval{}.png'.format(drs_value_type.lower(),
                                                                            chid,
                                                                            cell,
                                                                            array_str))
        plot.drs_value_cell_(data_collection_path,
                             interval_file_path,
                             fit_file_path_array,
                             filename,
                             interval_array,
                             drs_value_type,
                             chid,
                             cell,
                             ylimits)


################################################################################
def plot_drs_model_parameter_hist():
    print('Version 1: drs_model_parameter_hist')

    versions = [['Baseline', 'Offset', [-825, -1025]],
                ['Baseline', 'Slope', [-0.6, 0.3]],
                ['Gain', 'Offset', [0.85, 1.0]],
                ['Gain', 'Slope', [-0.95, 0.95]]]

    for drs_value_type, fit_parameter_type, x_lim in versions:
        filename = (pre_store_path+'model/{}/'.format(drs_value_type.lower()) +
                    'fit_parameter_{}_hist_{}.png'.format(fit_parameter_type.lower(),
                                                          drs_value_type.lower()))
        plot.drs_model_parameter_hist_(fit_file_path_array,
                                       filename,
                                       drs_value_type,
                                       fit_parameter_type,
                                       x_lim)


################################################################################
def plot_drs_fit_value_residual_hist():
    print('Version 1: drs_fit_value_residual_hist')

    versions = [['Baseline', [0.0, 1.5]],
                ['Gain', [0.0, 0.4]]]

    for drs_value_type, x_lim in versions:
        filename = (pre_store_path+'model/{}/'.format(drs_value_type.lower()) +
                    'residual_hist_{}.png'.format(drs_value_type.lower()))
        plot.drs_fit_value_residual_hist_(fit_file_path_array,
                                          filename,
                                          drs_value_type,
                                          x_lim)


################################################################################
def plot_residual_fact_cam():
    print('Version 1: residual_fact_cam')

    versions = [[3, 'Baseline', [928]],
                [3, 'Gain', np.arange(288, 299+1)]]

    for interval_nr, drs_value_type, worst_chids in versions:
        filename = (pre_store_path+'model/{}/'.format(drs_value_type.lower()) +
                    'residual_fact_cam_int{}_{}.png'.format(interval_nr,
                                                            drs_value_type.lower()))
        plot.residual_fact_cam_(fit_file_path_array[0],
                                filename,
                                drs_value_type,
                                worst_chids)


################################################################################
def plot_pedestel_mean_or_std_vs_temp():
    print('Version 1: pedestel_mean_or_std_vs_temp')

    versions = [['', 'Mean', 3, ['2015-05-26', '2017-10-01'], [6, 2], -0.24],
                ['', 'Std', 3, ['2015-05-26', '2017-10-01'], [6, 2], 0.17],
                ['_2016-01-01', 'Mean', 3, ['2016-01-01', '2017-10-01'], [6, 2], -0.12],
                ['_2016-01-01', 'Std', 3, ['2016-01-01', '2017-10-01'], [6, 2], 0.17]]

    for add, calibrated_type, interval_nr, datetime_limits, month_lim, ylim in versions:
        filename = (pre_store_path+'meanAndStd/interval{}/'.format(interval_nr) +
                    'calibratedPedestelDataDistribution_{}{}.png'.format(calibrated_type.lower(),
                                                                         add))
        plot.pedestel_mean_or_std_vs_temp_((pre_path+'../validating/version_1/meanAndStd/interval{}/'.format(interval_nr)),
                                           filename,
                                           calibrated_type,
                                           interval_nr,
                                           datetime_limits,
                                           month_lim,
                                           ylim)


################################################################################
def plot_pedestel_mean_or_std_fact_cam():
    print('Version 1: pedestel_mean_or_std_vs_temp')

    versions = [['calibratedPedestelDataFactcam_mean_2016-01-01.png', 'Mean', 3, ['2015-05-26', '2016-01-01']]]

    for store_path, calibrated_type, interval_nr, datetime_limits in versions:
        plot.pedestel_mean_or_std_fact_cam_((pre_path+'../validating/version_1/meanAndStd/interval{}/'.format(interval_nr)),
                                            (pre_store_path+'meanAndStd/interval{}/'.format(interval_nr)+store_path),
                                            calibrated_type,
                                            interval_nr,
                                            datetime_limits)



################################################################################
if __name__ == '__main__':
    #plot_drs_value_std_hist()
    #plot_drs_value_std_hist_per_chid_cell()
    #plot_drs_value_cell()
    #plot_drs_model_parameter_hist()
    #plot_drs_fit_value_residual_hist()
    #plot_residual_fact_cam()
    plot_pedestel_mean_or_std_vs_temp()
    #plot_pedestel_mean_or_std_fact_cam()  # just for infos
