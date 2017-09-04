from setuptools import setup

setup(
    name='drs4Calibration',
    version='0.0.1',
    description='model development for DRS4 amplitude calibration',
    url='http://github.com/fact-project/drs4Calibration',
    author='Florian Schulz, Dominik Neise',
    author_email='florian2.schulz@tu-dortmund.de',
    license='MIT',
    packages=[
        'drs4Calibration',
    ],
    package_data={
        '': []
    },
    tests_require=['pytest>=3.0.0'],
    setup_requires=['pytest-runner'],
    install_requires=[
        'pandas',
        'numpy',
        'click',
        'pyyaml',
        'h5py',
        'tqdm',
        'astropy',
        'pyfact',
        'scipy',
        'matplotlib>=1.4',
    ],
    entry_points={'console_scripts': [
        ('drsTempCalib_search_drs_files = ' +
            'drs4Calibration.drs4CalibrationTool:search_drs_files'),
        ('drsTempCalib_save_drs_attributes =' +
            'drs4Calibration.drs4CalibrationTool:save_drs_attributes'),
        ('drsTempCalib_store_interval_indices =' +
            'drs4Calibration.drs4CalibrationTool:store_source_based_interval_indices'),
        ('drsTempCalib_save_fit_values =' +
            'drs4Calibration.drs4CalibrationTool:save_fit_values'),
        'drsTempCalib_make_plots = drs4Calibration.do:main',
        ('drsTempCalib_make_plots.pixel_capacitor_drs_values =' +
            'drs4Calibration.plotDrsAttributes:pixel_capacitor_drs_values')
    ]},
    zip_safe=False,
)
