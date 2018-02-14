import drs4Calibration as calib
import plotDrsAttributes as plot
import writeSecondaryFiles as write

#calib.search_drs_run_files()
#calib.store_drs_attributes()
#calib.store_source_based_interval_indices()
#calib.calculate_fit_values()

write.drs_pedestal_run_mean_and_std()
#write.drs_pedestal_run_mean_and_std_outside()

#plot.drs_value_std_hist()
#plot.drs_model_parameter_hist()

#plot.pedestel_mean_or_std_vs_temp_()
#plot.pedestel_mean_or_std_vs_temp_outside()

#plot.drs_value_cell()
#plot.chid_startcell_distribution()
#plot.drs_fit_value_residual_hist()
#plot.residual_fact_cam()
#plot.worst_cell_collection_selected_by_residual()
#plot.drs_values_chi2_outlier_cell_collection()
