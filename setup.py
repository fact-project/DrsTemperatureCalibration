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
        ('drsCalib_v1_search_drs_files = ' +
            'drs4Calibration.drs4Calibration_version_0.' +
            'drs4Calibration_drsFileBased:search_drs_files'),
        ('drsCalib_v1_save_drs_attributes =' +
            'drs4Calibration.drs4Calibration_version_0.' +
            'drs4Calibration_drsFileBased:store_drs_attributes'),
        ('drsCalib_v1_store_interval_indices =' +
            'drs4Calibration.drs4Calibration_version_0.' +
            'drs4Calibration_drsFileBased:store_source_based_interval_indices'),
        ('drsCalib_v1_save_fit_values =' +
            'drs4Calibration.drs4Calibration_version_0.' +
            'drs4Calibration_drsFileBased:calculate_fit_values'),
        ('drsCalib_v2_search_drs_files = ' +
            'drs4Calibration.drs4Calibration_version_1.' +
            'drs4Calibration:search_drs_run_files'),
        ('drsCalib_v2_save_drs_attributes =' +
            'drs4Calibration.drs4Calibration_version_1.' +
            'drs4Calibration:store_drs_attributes'),
        ('drsCalib_v2_store_interval_indices =' +
            'drs4Calibration.drs4Calibration_version_1.' +
            'drs4Calibration:store_source_based_interval_indices'),
        ('drsCalib_v2_save_fit_values =' +
            'drs4Calibration.drs4Calibration_version_1.' +
            'drs4Calibration:calculate_fit_values')
    ]},
    zip_safe=False,
)
