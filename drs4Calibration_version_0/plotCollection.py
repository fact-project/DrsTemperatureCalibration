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

    versions = [['baseline/std_hist_baseline_.png', [1, 2, 3], 'Baseline', [0.0, 4.0]],
                ['gain/std_hist_gain_.png', [1, 2, 3], 'Gain', [0.0, 4.5]],
                ['roiOffset/std_hist_roiOffset_.png', [1, 2, 3], 'ROIOffset', [0.0, 5.0]]]

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
    versions = [['gain/std_hist_gain_chid250_cell250.png', [1, 2, 3], 'Gain', 250, 250, 2, [0.0, 4.5]]]

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
    versions = [['gain/fitparameter_gain_chid250_cell250_interval1_2_3.png', [1, 2, 3], 'Gain', 250, 250]]

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
    print('Version 0: drs_model_parameter_hist')

    versions = [['baseline/fit_parameter_offset_hist_baseline.png', 'Baseline', 'Offset', [-840, -980]],
                ['baseline/fit_parameter_slope_hist_baseline.png', 'Baseline', 'Slope', [-0.65, 0.4]],
                ['gain/fit_parameter_offset_hist_gain.png', 'Gain', 'Offset', [0.85, 0.98]],
                ['gain/fit_parameter_slope_hist_gain.png', 'Gain', 'Slope', [-0.95, 0.95]],
                ['roiOffset/fit_parameter_offset_hist_roiOffset.png', 'ROIOffset', 'Offset', [-5.0, 17.0]],
                ['roiOffset/fit_parameter_slope_hist_roiOffset.png', 'ROIOffset', 'Slope', [-0.25, 0.3]]]

    for store_path, drs_value_type, fit_parameter_type, x_lim in versions:
        plot.drs_model_parameter_hist_(fit_file_path_array,
                                       (pre_store_path+'model/'+store_path),
                                       drs_value_type,
                                       fit_parameter_type,
                                       x_lim)


################################################################################
def plot_drs_model_parameter_fact_cam():
    print('Version 0: drs_model_parameter_fact_cam')

    versions = [['baseline/fit_parameter_offset_fact_cam_baseline.png', 'Baseline', 'Offset'],
                ['baseline/fit_parameter_slope_fact_cam_baseline.png', 'Baseline', 'Slope'],
                ['gain/fit_parameter_offset_fact_cam_gain.png', 'Gain', 'Offset'],
                ['gain/fit_parameter_slope_fact_cam_gain.png', 'Gain', 'Slope'],
                ['roiOffset/fit_parameter_offset_fact_cam_roiOffset.png', 'ROIOffset', 'Offset'],
                ['roiOffset/fit_parameter_slope_fact_cam_roiOffset.png', 'ROIOffset', 'Slope']]

    for store_path, drs_value_type, fit_parameter_type in versions:
        plot.drs_model_parameter_fact_cam_(fit_file_path_array,
                                           (pre_store_path+'model/'+store_path),
                                           drs_value_type,
                                           fit_parameter_type)


################################################################################
def plot_drs_fit_value_residual_hist():
    print('Version 0: drs_fit_value_residual_hist')

    versions = [['baseline/residual_hist_baseline.png', 'Baseline', [0.0, 1.0]],
                ['gain/residual_hist_gain.png', 'Gain', [0.0, 0.4]],
                ['roiOffset/residual_hist_roiOffset.png', 'ROIOffset', [0.0, 1.0]]]

    for store_path, drs_value_type, x_lim in versions:
        plot.drs_fit_value_residual_hist_(fit_file_path_array,
                                          (pre_store_path+'model/'+store_path),
                                          drs_value_type,
                                          x_lim)


################################################################################
def plot_residual_fact_cam():
    print('Version 0: residual_fact_cam')

    versions = [['baseline/residual_fact_cam_int1_baseline.png', 1, 'Baseline', np.append([966], np.arange(432, 440+1))],
                ['baseline/residual_fact_cam_int2_baseline.png', 2, 'Baseline', np.append([966], np.arange(432, 440+1))],
                ['baseline/residual_fact_cam_int3_baseline.png', 3, 'Baseline', np.append([738, 966], np.arange(1080, 1439+1))],
                ['gain/residual_fact_cam_int1_gain.png', 1, 'Gain', [566]],
                ['gain/residual_fact_cam_int2_gain.png', 2, 'Gain', [566]],
                ['gain/residual_fact_cam_int3_gain.png', 3, 'Gain', np.arange(1080, 1439+1)],
                ['roiOffset/residual_fact_cam_int1_roiOffset.png', 1, 'ROIOffset', [566]],
                ['roiOffset/residual_fact_cam_int2_roiOffset.png', 2, 'ROIOffset', [566]],
                ['roiOffset/residual_fact_cam_int3_roiOffset.png', 3, 'ROIOffset', np.arange(1080, 1439+1)]]

    for store_path, interval_nr, drs_value_type, worst_chids in versions:
        plot.residual_fact_cam_(fit_file_path_array[interval_nr-1],
                                (pre_store_path+'model/'+store_path),
                                drs_value_type,
                                worst_chids)
      # np.append([966], np.arange(432, 440+1)) np.arange(1080, 1439+1)


################################################################################
def plot_pedestel_mean_or_std_vs_temp():
    print('Version 0: pedestel_mean_or_std_vs_temp')

    versions = [['calibratedPedestelDataDistribution_mean.png', 'Mean', 2, ['2014-05-20 12', '2015-05-26 12']],
                ['calibratedPedestelDataDistribution_std.png', 'Std', 2, ['2014-05-20 12', '2015-05-26 12']],
                ['calibratedPedestelDataDistribution_mean.png', 'Mean', 3, ['2015-05-26', '2017-10-01']],
                ['calibratedPedestelDataDistribution_std.png', 'Std', 3, ['2015-05-26', '2017-10-01']]]

    for store_path, calibrated_type, interval_nr, datetime_limits in versions:
        plot.pedestel_mean_or_std_vs_temp_((pre_path+'../validating/version_0/meanAndStd/interval{}/'.format(interval_nr)),
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
    #plot_drs_model_parameter_fact_cam()
    #plot_drs_fit_value_residual_hist()
    plot_residual_fact_cam()

    #plot_pedestel_mean_or_std_vs_temp()

# old version 0 stuff

    #plot_drs_value_std_hist()
    #plot_residual_fact_cam()
    #plot_worst_cell_collection_selected_by_residual()

# ################################################################################
# def plot_drs_value_std_hist():
#     print('drs_value_std_hist')
#     plot.drs_value_std_hist_(data_collection_path,
#                                interval_file_path,
#                              (pre_store_path+'drsValues/baseline/std_hist.jpg'),
#                              [1, 2, 3],
#                              'Baseline')
#     plot.drs_value_std_hist_(data_collection_path,
#                              interval_file_path,
#                              (pre_store_path+'drsValues/gain/std_hist.jpg'),
#                              [1, 2, 3],
#                              'Gain')
#     plot.drs_value_std_hist_(data_collection_path,
#                              interval_file_path,
#                              (pre_store_path+'drsValues/roiOffset/std_hist.jpg'),
#                              [1, 2, 3],
#                              'ROIOffset')
#
#
# ################################################################################
# def plot_residual_fact_cam_old():
#     print('residual_fact_cam')
#     plot.residual_fact_cam_(fit_file_path_array,
#                             (pre_store_path+'residual/baseline/fact_cam_residual_int1.jpg'),
#                             1,
#                             'Baseline')
#     plot.residual_fact_cam_(fit_file_path_array,
#                             (pre_store_path+'residual/baseline/fact_cam_residual_int2.jpg'),
#                             2,
#                             'Baseline')
#     plot.residual_fact_cam_(fit_file_path_array,
#                             (pre_store_path+'residual/baseline/fact_cam_residual_int3.jpg'),
#                             3,
#                             'Baseline')
#     plot.residual_fact_cam_(fit_file_path_array,
#                             (pre_store_path+'residual/gain/fact_cam_residual_int1.jpg'),
#                             1,
#                             'Gain')
#     plot.residual_fact_cam_(fit_file_path_array,
#                             (pre_store_path+'residual/gain/fact_cam_residual_int2.jpg'),
#                             2,
#                             'Gain')
#     plot.residual_fact_cam_(fit_file_path_array,
#                             (pre_store_path+'residual/gain/fact_cam_residual_int3.jpg'),
#                             3,
#                             'Gain')
#     plot.residual_fact_cam_(fit_file_path_array,
#                             (pre_store_path+'residual/roiOffset/fact_cam_residual_int1.jpg'),
#                             1,
#                             'ROIOffset')
#     plot.residual_fact_cam_(fit_file_path_array,
#                             (pre_store_path+'residual/roiOffset/fact_cam_residual_int2.jpg'),
#                             2,
#                             'ROIOffset')
#     plot.residual_fact_cam_(fit_file_path_array,
#                             (pre_store_path+'residual/roiOffset/fact_cam_residual_int3.jpg'),
#                             3,
#                             'ROIOffset')
#     # bad pixel
#     # Baseline:
#     # I1:[966] or [738, 966],
#     # I2:np.arange(720, 728+1) or np.arange(720, 755+1)
#     # I3:[738, 966]
#
#     # Gain:
#     # I1:np.arange(0, 1440, 9)+8
#     # I2:np.arange(720, 755+1) or np.append(np.arange(720, 755+1), np.arange(0, 1440, 9)+8)
#
#     plot.residual_fact_cam_(fit_file_path_array,
#                             (pre_store_path+'residual/baseline/fact_cam_residual_int1_without_chid_966.jpg'),
#                             1,
#                             'Baseline',
#                             [966])
#     plot.residual_fact_cam_(fit_file_path_array,
#                             (pre_store_path+'residual/baseline/fact_cam_residual_int2_without_chid_720-728.jpg'),
#                             2,
#                             'Baseline',
#                             np.arange(720, 728+1))
#     plot.residual_fact_cam_(fit_file_path_array,
#                             (pre_store_path+'residual/baseline/fact_cam_residual_int2_without_chid_720-755.jpg'),
#                             2,
#                             'Baseline',
#                             np.arange(720, 755+1))
#     plot.residual_fact_cam_(fit_file_path_array,
#                             (pre_store_path+'residual/baseline/fact_cam_residual_int3_without_chid_738_966.jpg'),
#                             3,
#                             'Baseline',
#                             [738, 966])
#     plot.residual_fact_cam_(fit_file_path_array,
#                             (pre_store_path+'residual/gain/fact_cam_residual_int1_without_timemarkerchannel.jpg'),
#                             1,
#                             'Gain',
#                             np.arange(0, 1440, 9)+8)
#     plot.residual_fact_cam_(fit_file_path_array,
#                             (pre_store_path+'residual/gain/fact_cam_residual_int2_without_chid_720-755.jpg'),
#                             2,
#                             'Gain',
#                             np.arange(720, 755+1))
#     plot.residual_fact_cam_(fit_file_path_array,
#                             (pre_store_path+'residual/gain/fact_cam_residual_int2_without_chid_720-755_timemarkerchannel.jpg'),
#                             2,
#                             'Gain',
#                             np.append(np.arange(720, 755+1), np.arange(0, 1440, 9)+8))
#
#     plot.residual_fact_cam_(fit_file_path_array,
#                             (pre_store_path+'residual/roiOffset/fact_cam_residual_int1_without_chid-422.jpg'),
#                             1,
#                             'ROIOffset',
#                             np.arange(0, 422+1))
#
#
# ################################################################################
# def plot_worst_cell_collection_selected_by_residual():
#     print('worst_cell_collection_selected_by_residual')
#     plot.worst_cell_collection_selected_by_residual_(
#                                         data_collection_path,
#                                         interval_file_path,
#                                         fit_file_path_array[0],
#                                         (pre_store_path+'residual/baseline/worst_100_cell_collection_I1.pdf'),
#                                         1,
#                                         'Baseline',
#                                         [])
#     plot.worst_cell_collection_selected_by_residual_(
#                                         data_collection_path,
#                                         interval_file_path,
#                                         fit_file_path_array[0],
#                                         (pre_store_path+'residual/baseline/worst_100_cell_collection_I1_without_chid_966_738.pdf'),
#                                         1,
#                                         'Baseline',
#                                         [966, 738])
#     plot.worst_cell_collection_selected_by_residual_(
#                                         data_collection_path,
#                                         interval_file_path,
#                                         fit_file_path_array[1],
#                                         (pre_store_path+'residual/gain/worst_100_cell_collection_I2.pdf'),
#                                         2,
#                                         'Gain',
#                                         [])
#
#     plot.worst_cell_collection_selected_by_residual_(
#                                         data_collection_path,
#                                         interval_file_path,
#                                         fit_file_path_array[1],
#                                         (pre_store_path+'residual/gain/worst_100_cell_collection_I2_without_chid_720-755.pdf'),
#                                         2,
#                                         'Gain',
#                                         np.arange(720, 755+1))
#
#     plot.worst_cell_collection_selected_by_residual_(
#                                         data_collection_path,
#                                         interval_file_path,
#                                         fit_file_path_array[1],
#                                         (pre_store_path+'residual/gain/worst_100_cell_collection_I2_without_chid_720-755_timemarkerchannel.pdf'),
#                                         2,
#                                         'Gain',
#                                         np.append(np.arange(720, 755+1), np.arange(0, 1440, 9)+8))
#
#     plot.worst_cell_collection_selected_by_residual_(
#                                         data_collection_path,
#                                         interval_file_path,
#                                         fit_file_path_array[0],
#                                         (pre_store_path+'residual/roiOffset/worst_100_cell_collection_I1.pdf'),
#                                         1,
#                                         'ROIOffset',
#                                         [])
#     plot.worst_cell_collection_selected_by_residual_(
#                                         data_collection_path,
#                                         interval_file_path,
#                                         fit_file_path_array[0],
#                                         (pre_store_path+'residual/roiOffset/worst_100_cell_collection_I1_without_chid_566.pdf'),
#                                         1,
#                                         'ROIOffset',
#                                         [566])
