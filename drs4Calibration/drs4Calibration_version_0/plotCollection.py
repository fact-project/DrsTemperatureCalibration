import plotDrsAttributes as plot
import numpy as np

pre_path = ('/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
            'calibration/calculation/')

pre_store_path = '/home/fschulz/plots/version_0/'

data_collection_path = pre_path+'/version_0/dataCollection.h5'


interval_file_path = pre_path+'version_0/intervalIndices.h5'


fit_file_path_array = [pre_path+'version_0/drsFitParameter_interval1.fits',
                       pre_path+'version_0/drsFitParameter_interval2.fits',
                       pre_path+'version_0/drsFitParameter_interval3.fits']


################################################################################
# ############## ###############   Version 0   ############### ############### #
################################################################################

def plot_drs_value_std_hist():
    print('Version 0: drs_value_std_hist')

    versions = [#['baseline/std_hist_baseline_without_9er_chids.png', [1, 2, 3], 'Baseline', [1.0, 3.75]],
                ['gain/std_hist_gain_without_9er_chids.png', [1, 2, 3], 'Gain', [0.5, 4.2]],]
                #['roiOffset/std_hist_roiOffset.png', [1, 2, 3], 'ROIOffset', [1.0, 5.0]]]
    # '_without_9er_chids'

    for store_path, interval_array, drs_value_type, x_lim in versions:
        plot.drs_value_std_hist_(data_collection_path,
                                 interval_file_path,
                                 (pre_store_path+'drsValues/'+store_path),
                                 interval_array,
                                 drs_value_type,
                                 x_lim)


################################################################################
def plot_drs_value_std_hist_per_chid_cell():
    print('Version 0: drs_value_std_hist_per_chid_cell')
    versions = [['gain/std_hist_gain_chid250_cell250.png', [1, 2, 3], 'Gain', 250, 250, 2, [0.5, 4.2]]]

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
    print('Version 0: drs_value_cell')
    versions = [[[1, 2, 3], 'Gain', 250, 250, []],
                [[1, 2, 3], 'Baseline', 1124, 794, []],
                [[1, 2, 3], 'Gain', 1124, 794, []],
                [[1, 2, 3], 'Baseline', 746, 561, []],
                [[1, 2, 3], 'Gain', 746, 561, []],
                [[1, 2, 3], 'ROIOffset', 590, 0, []],
                [[1, 2, 3], 'Baseline', 566, 40, []],
                [[1, 2, 3], 'Gain', 566, 40, []],
                [[1, 2, 3], 'ROIOffset', 750, 20, []],
                [[1, 2, 3], 'ROIOffset', 750, 150, []],
                [[1, 2, 3], 'ROIOffset', 750, 280, []],
                [[1, 2, 3], 'Baseline', 98, 93, []],
                [[1, 2, 3], 'Gain', 98, 93, []],
                [[1, 2, 3], 'Baseline', 1275, 31, [-1040, -930]],
                [[1, 2, 3], 'Gain', 1275, 31, []],
                [[1, 2, 3], 'Gain', 7, 543, []],
                [[1, 2, 3], 'Gain', 15, 753, []]]

    for interval_array, drs_value_type, chid, cell, ylimits in versions:
        array_str = '1_2_3'
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
    print('Version 0: drs_model_parameter_hist')

    versions = [['Baseline', 'Offset', [-830, -1030]],
                ['Baseline', 'Slope', [-0.65, 0.4]],
                ['Gain', 'Offset', [0.85, 1.0]],
                ['Gain', 'Slope', [-0.95, 0.95]],
                ['ROIOffset', 'Offset', [-4.0, 18.0]],
                ['ROIOffset', 'Slope', [-0.25, 0.3]]]

    for drs_value_type, fit_parameter_type, x_lim in versions:
        filename = (pre_store_path+'model/{}/'.format(drs_value_type.lower()) +
                    'fit_parameter_{}_hist_{}.png'.format(fit_parameter_type.lower(),
                                                          drs_value_type.lower()))
        # filename = (pre_store_path+'model/{}/'.format(drs_value_type.lower()) +
        #             'fit_parameter_{}_hist_{}_int3_with_create4.png'.format(fit_parameter_type.lower(),
        #                                                                     drs_value_type.lower()))

        plot.drs_model_parameter_hist_(fit_file_path_array,
                                       filename,
                                       drs_value_type,
                                       fit_parameter_type,
                                       x_lim)


################################################################################
def plot_drs_model_parameter_fact_cam():
    print('Version 0: drs_model_parameter_fact_cam')

    versions = [['Baseline', 'Offset'],
                ['Baseline', 'Slope'],
                ['Gain', 'Offset'],
                ['Gain', 'Slope'],
                ['ROIOffset', 'Offset'],
                ['ROIOffset', 'Slope']]

    for drs_value_type, fit_parameter_type in versions:
        filename = (pre_store_path+'model/{}/'.format(drs_value_type.lower()) +
                    'fit_parameter_{}_fact_cam_{}.png'.format(fit_parameter_type.lower(),
                                                              drs_value_type.lower()))
        plot.drs_model_parameter_fact_cam_(fit_file_path_array,
                                           filename,
                                           drs_value_type,
                                           fit_parameter_type)


################################################################################
def plot_drs_fit_value_residual_hist():
    print('Version 0: drs_fit_value_residual_hist')

    versions = [['Baseline', [0.0, 1.0]],
                ['Gain', [0.0, 0.4]],
                ['ROIOffset', [0.0, 1.0]]]

    for drs_value_type, x_lim in versions:
        filename = (pre_store_path+'model/{}/'.format(drs_value_type.lower()) +
                    'residual_hist_{}.png'.format(drs_value_type.lower()))
        plot.drs_fit_value_residual_hist_(fit_file_path_array,
                                          filename,
                                          drs_value_type,
                                          x_lim)


################################################################################
def plot_residual_fact_cam():
    print('Version 0: residual_fact_cam')

    versions = [[1, 'Baseline', np.append([966], np.arange(432, 440+1))],
                [2, 'Baseline', np.append([966], np.arange(432, 440+1))],
                [3, 'Baseline', np.append([738, 966], np.arange(1080, 1439+1))],
                [1, 'Gain', [566]],
                [2, 'Gain', [566]],
                [3, 'Gain', np.arange(1080, 1439+1)],
                [1, 'ROIOffset', [566]],
                [2, 'ROIOffset', [566]],
                [3, 'ROIOffset', np.arange(1080, 1439+1)]]

    for interval_nr, drs_value_type, worst_chids in versions:
        filename = (pre_store_path+'model/{}/'.format(drs_value_type.lower()) +
                    'residual_fact_cam_int{}_{}.png'.format(interval_nr,
                                                            drs_value_type.lower()))
        plot.residual_fact_cam_(fit_file_path_array[interval_nr-1],
                                filename,
                                drs_value_type,
                                worst_chids)
      # np.append([966], np.arange(432, 440+1)) np.arange(1080, 1439+1)


################################################################################
def plot_pedestel_mean_or_std_vs_temp():
    print('Version 0: pedestel_mean_or_std_vs_temp')

    versions = [['', 'Mean', 2, ['2014-05-20 12', '2015-05-26 12'], [3, 1], -0.24],
                ['', 'Std', 2, ['2014-05-20 12', '2015-05-26 12'], [3, 1], 0.17],]
                # ['', 'Mean', 3, ['2015-05-26', '2017-10-01'], [6, 2], -0.24],
                # ['', 'Std', 3, ['2015-05-26', '2017-10-01'], [6, 2], 0.17],
                # ['_2016-01-01', 'Mean', 3, ['2016-01-01', '2017-10-01'], [6, 2], -0.12],
                # ['_2016-01-01', 'Std', 3, ['2016-01-01', '2017-10-01'], [6, 2], 0.17]]

    for add, calibrated_type, interval_nr, datetime_limits, month_lim, ylim in versions:
        filename = (pre_store_path+'meanAndStd/interval{}/'.format(interval_nr) +
                    'calibratedPedestelDataDistribution_{}{}.png'.format(calibrated_type.lower(),
                                                                         add))
        plot.pedestel_mean_or_std_vs_temp_((pre_path+'../validating/version_0/meanAndStd/interval{}/'.format(interval_nr)),
                                           filename,
                                           calibrated_type,
                                           interval_nr,
                                           datetime_limits,
                                           month_lim,
                                           ylim)


################################################################################
if __name__ == '__main__':

    plot_drs_value_std_hist()
    #plot_drs_value_std_hist_per_chid_cell()
    #plot_drs_value_cell()
    #plot_drs_model_parameter_hist()
    #plot_drs_model_parameter_fact_cam()
    #plot_drs_fit_value_residual_hist()
    #plot_residual_fact_cam()

    #plot_pedestel_mean_or_std_vs_temp()
