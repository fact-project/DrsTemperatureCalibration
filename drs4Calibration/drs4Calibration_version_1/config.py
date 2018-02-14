from drs4Calibration.drs4Calibration_version_1.constants import NRCHID, NRCELL, ROI, NRTEMPSENSOR


# data_collection .h5 stuff
class data_collection_config:

    column_names = ['TimeBaseline', 'TempBaseline', 'Baseline',
                    'TimeGain', 'TempGain', 'Gain', 'GainStd']

    column_dtype = {'TimeBaseline': 'float32',
                    'TempBaseline': 'float32',
                    'Baseline': 'float32',
                    'TimeGain': 'float32',
                    'TempGain': 'float32',
                    'Gain': 'float32',
                    'GainStd': 'float16'}

    column_length = {'TimeBaseline': 1,
                     'TempBaseline': NRTEMPSENSOR,
                     'Baseline': NRCHID*NRCELL*ROI,
                     'TimeGain': 1,
                     'TempGain': NRTEMPSENSOR,
                     'Gain': NRCHID*NRCELL,
                     'GainStd': NRCHID*NRCELL}


class fit_value_config:

    drs_value_types = ['Baseline', 'Gain']

    class interval_indice_config:

        # hardware_boundaries
        #
        # 20.05.2014 Camera repair, Replacement of Capacitors
        # 26.5.2015 Replacement FAD board (crate 2, board 0)
        #
        # See also 'https://trac.fact-project.org/wiki/Protected/SystemChanges'
        hardware_boundaries = (['2014-05-20 12',
                                '2015-05-26 12'])

        # There are two reasons for calculate no mask for Baseline values.
        # 1. No sufficient standard deviation of the Baseline mean exist.
        # 2. Baseline mask does not fit in ram.

        # All Gain-values with a larger error (std dev of the mean)
        # than the 'CutOffErrorFactor' multiplied with the mean of the error
        # from all collected Gain-values for one capacitor will not used for the fit
        cut_off_error_factor = {'Gain': 2}

    drs_values_per_cell = {'Baseline': ROI,
                           'Gain': 1}

    value_units = {'Baseline': 'mV',
                   'Gain': '1'}



nrCellsPerChid = {'Baseline': NRCELL,
                 'Gain': NRCELL}
