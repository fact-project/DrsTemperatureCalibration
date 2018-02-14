from drs4Calibration.drs4Calibration_version_0.constants import NRCELL, ROI

# Dont shuffle drsValueTypes
# The order have to be the same as the of the 'RUNIDs'
# in the drsFiles

drsValueTypes = ['Baseline',
                 'Gain',
                 'TriggerOffset']
renamedDrsValueTypes = ['Baseline',
                        'Gain',
                        'ROIOffset']  # renamed TriggerOffset
nrCellsPerChid = {'Baseline': NRCELL,
                  'Gain': NRCELL,
                  'ROIOffset': ROI}
cutOffErrorFactor = {'Baseline': 2,
                     'Gain': 2,
                     'ROIOffset': 2}
hardwareBoundaries = ['2014-05-20 12',
                      '2015-05-26 12']

# hardwareBoundaries
#
# 20.05.2014 Camera repair, Replacement of Capacitors
# 26.5.2015 Replacement FAD board (crate 2, board 0)
#
# See also 'https://trac.fact-project.org/wiki/Protected/SystemChanges'
