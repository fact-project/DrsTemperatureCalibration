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
    install_requires=[  # add logging, fact, yaml, h5py, os
        'pandas',
        'numpy',
        'scipy',
        'tqdm',
        'click',
        'astropy',
        'matplotlib>=1.4',
    ],
    entry_points={'console_scripts': [
        'drsTempCalib_search_drs_files = drsTemperatureCalibration.drsFitTool:search_drs_files',
        'drsTempCalib_save_drs_attributes = drsTemperatureCalibration.drsFitTool:save_drs_attributes',
        'drsTempCalib_save_fit_values = drsTemperatureCalibration.drsFitTool:save_fit_values',
        'drsTempCalib_make_plots = drsTemperatureCalibration.do:main',
    ]},
    zip_safe=False,
)
