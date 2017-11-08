import plotDrsAttributes as plot
import numpy as np

drs_file_path = ('/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                 'calibration/calculation/version_0/drsData.h5')

interval_file_path = ('/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                      'calibration/calculation/version_0/intervalIndices.h5')

fit_file_path = ('/net/big-tank/POOL/projects/fact/drs4_calibration_data/' +
                 'calibration/calculation/version_0/drsFitParameter.fits')
pre_store_path = '/home/fschulz/plots/version_0/'


################################################################################
def plot_drs_value_std_hist():
    print('drs_value_std_hist')
    plot.drs_value_std_hist_(drs_file_path,
                             interval_file_path,
                             (pre_store_path+'drsValues/baseline/std_hist.jpg'),
                             [1, 2, 3],
                             'Baseline')
    plot.drs_value_std_hist_(drs_file_path,
                             interval_file_path,
                             (pre_store_path+'drsValues/gain/std_hist.jpg'),
                             [1, 2, 3],
                             'Gain')
    plot.drs_value_std_hist_(drs_file_path,
                             interval_file_path,
                             (pre_store_path+'drsValues/roiOffset/std_hist.jpg'),
                             [1, 2, 3],
                             'ROIOffset')


################################################################################
def plot_drs_fit_value_residual_hist():
    print('drs_fit_value_residual_hist')
    plot.drs_fit_value_residual_hist_(fit_file_path,
                                      (pre_store_path+'residual/baseline/residual_hist.jpg'),
                                      [1, 2, 3],
                                      'Baseline')
    plot.drs_fit_value_residual_hist_(fit_file_path,
                                      (pre_store_path+'residual/gain/residual_hist.jpg'),
                                      [1, 2, 3],
                                      'Gain')
    plot.drs_fit_value_residual_hist_(fit_file_path,
                                      (pre_store_path+'residual/roiOffset/residual_hist.jpg'),
                                      [1, 2, 3],
                                      'ROIOffset')


################################################################################
def plot_residual_fact_cam():
    print('residual_fact_cam')
    plot.residual_fact_cam_(fit_file_path,
                            (pre_store_path+'residual/baseline/fact_cam_residual_int1.jpg'),
                            1,
                            'Baseline')
    plot.residual_fact_cam_(fit_file_path,
                            (pre_store_path+'residual/baseline/fact_cam_residual_int2.jpg'),
                            2,
                            'Baseline')
    plot.residual_fact_cam_(fit_file_path,
                            (pre_store_path+'residual/baseline/fact_cam_residual_int3.jpg'),
                            3,
                            'Baseline')
    plot.residual_fact_cam_(fit_file_path,
                            (pre_store_path+'residual/gain/fact_cam_residual_int1.jpg'),
                            1,
                            'Gain')
    plot.residual_fact_cam_(fit_file_path,
                            (pre_store_path+'residual/gain/fact_cam_residual_int2.jpg'),
                            2,
                            'Gain')
    plot.residual_fact_cam_(fit_file_path,
                            (pre_store_path+'residual/gain/fact_cam_residual_int3.jpg'),
                            3,
                            'Gain')
    plot.residual_fact_cam_(fit_file_path,
                            (pre_store_path+'residual/roiOffset/fact_cam_residual_int1.jpg'),
                            1,
                            'ROIOffset')
    plot.residual_fact_cam_(fit_file_path,
                            (pre_store_path+'residual/roiOffset/fact_cam_residual_int2.jpg'),
                            2,
                            'ROIOffset')
    plot.residual_fact_cam_(fit_file_path,
                            (pre_store_path+'residual/roiOffset/fact_cam_residual_int3.jpg'),
                            3,
                            'ROIOffset')
    # bad pixel
    # Baseline:
    # I1:[966] or [738, 966],
    # I2:np.arange(720, 728+1) or np.arange(720, 755+1)
    # I3:[738, 966]

    # Gain:
    # I1:np.arange(0, 1440, 9)+8
    # I2:np.arange(720, 755+1) or np.append(np.arange(720, 755+1), np.arange(0, 1440, 9)+8)

    plot.residual_fact_cam_(fit_file_path,
                            (pre_store_path+'residual/baseline/fact_cam_residual_int1_without_chid_966.jpg'),
                            1,
                            'Baseline',
                            [966])
    plot.residual_fact_cam_(fit_file_path,
                            (pre_store_path+'residual/baseline/fact_cam_residual_int2_without_chid_720-728.jpg'),
                            2,
                            'Baseline',
                            np.arange(720, 728+1))
    plot.residual_fact_cam_(fit_file_path,
                            (pre_store_path+'residual/baseline/fact_cam_residual_int2_without_chid_720-755.jpg'),
                            2,
                            'Baseline',
                            np.arange(720, 755+1))
    plot.residual_fact_cam_(fit_file_path,
                            (pre_store_path+'residual/baseline/fact_cam_residual_int3_without_chid_738_966.jpg'),
                            3,
                            'Baseline',
                            [738, 966])
    plot.residual_fact_cam_(fit_file_path,
                            (pre_store_path+'residual/gain/fact_cam_residual_int1_without_timemarkerchannel.jpg'),
                            1,
                            'Gain',
                            np.arange(0, 1440, 9)+8)
    plot.residual_fact_cam_(fit_file_path,
                            (pre_store_path+'residual/gain/fact_cam_residual_int2_without_chid_720-755.jpg'),
                            2,
                            'Gain',
                            np.arange(720, 755+1))
    plot.residual_fact_cam_(fit_file_path,
                            (pre_store_path+'residual/gain/fact_cam_residual_int2_without_chid_720-755_timemarkerchannel.jpg'),
                            2,
                            'Gain',
                            np.append(np.arange(720, 755+1), np.arange(0, 1440, 9)+8))

    plot.residual_fact_cam_(fit_file_path,
                            (pre_store_path+'residual/roiOffset/fact_cam_residual_int1_without_chid_0-422.jpg'),
                            1,
                            'ROIOffset',
                            np.arange(0, 422+1))


################################################################################
def plot_worst_cell_collection_selected_by_residual():
    print('worst_cell_collection_selected_by_residual')
    plot.worst_cell_collection_selected_by_residual_(
                                        drs_file_path,
                                        interval_file_path,
                                        fit_file_path,
                                        (pre_store_path+'residual/baseline/worst_100_cell_collection_I1.pdf'),
                                        1,
                                        'Baseline',
                                        [])
    plot.worst_cell_collection_selected_by_residual_(
                                        drs_file_path,
                                        interval_file_path,
                                        fit_file_path,
                                        (pre_store_path+'residual/baseline/worst_100_cell_collection_I1_without_chid_966_738.pdf'),
                                        1,
                                        'Baseline',
                                        [966, 738])
    plot.worst_cell_collection_selected_by_residual_(
                                        drs_file_path,
                                        interval_file_path,
                                        fit_file_path,
                                        (pre_store_path+'residual/gain/worst_100_cell_collection_I2.pdf'),
                                        2,
                                        'Gain',
                                        [])

    plot.worst_cell_collection_selected_by_residual_(
                                        drs_file_path,
                                        interval_file_path,
                                        fit_file_path,
                                        (pre_store_path+'residual/gain/worst_100_cell_collection_I2_without_chid_720-755.pdf'),
                                        2,
                                        'Gain',
                                        np.arange(720, 755+1))

    plot.worst_cell_collection_selected_by_residual_(
                                        drs_file_path,
                                        interval_file_path,
                                        fit_file_path,
                                        (pre_store_path+'residual/gain/worst_100_cell_collection_I2_without_chid_720-755_timemarkerchannel.pdf'),
                                        2,
                                        'Gain',
                                        np.append(np.arange(720, 755+1), np.arange(0, 1440, 9)+8))

    plot.worst_cell_collection_selected_by_residual_(
                                        drs_file_path,
                                        interval_file_path,
                                        fit_file_path,
                                        (pre_store_path+'residual/roiOffset/worst_100_cell_collection_I1.pdf'),
                                        1,
                                        'ROIOffset',
                                        [])
    plot.worst_cell_collection_selected_by_residual_(
                                        drs_file_path,
                                        interval_file_path,
                                        fit_file_path,
                                        (pre_store_path+'residual/roiOffset/worst_100_cell_collection_I1_without_chid_566.pdf'),
                                        1,
                                        'ROIOffset',
                                        [566])


if __name__ == '__main__':
    plot_drs_value_std_hist()
    #plot_drs_fit_value_residual_hist()
    #plot_residual_fact_cam()
    #plot_worst_cell_collection_selected_by_residual()
