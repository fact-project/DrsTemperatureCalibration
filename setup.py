from setuptools import setup

setup(
    name='drsTemperatureCalibration',
    version='0.0.1',
    description='model development for DRS4 amplitude calibration',
    url='http://github.com/fact-project/drsTemperatureCalibration',
    author='Florian Schulz, Dominik Neise',
    author_email='florian2.schulz@tu-dortmund.de',
    license='MIT',
    packages=[
        'drsTemperatureCalibration',
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
            'drsTemperatureCalibration.drsFitTool:search_drs_files'),
        ('drsTempCalib_save_drs_attributes =' +
            'drsTemperatureCalibration.drsFitTool:save_drs_attributes'),
        ('drsTempCalib_store_interval_indices =' +
            'drsTemperatureCalibration.drsFitTool:store_source_based_interval_indices'),
        ('drsTempCalib_save_fit_values =' +
            'drsTemperatureCalibration.drsFitTool:save_fit_values'),
        'drsTempCalib_make_plots = drsTemperatureCalibration.do:main',
        ('drsTempCalib_make_plots.pixel_capacitor_drs_values =' +
            'drsTemperatureCalibration.plotDrsAttributes:pixel_capacitor_drs_values')
    ]},
    zip_safe=False,
)
