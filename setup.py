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
        'numpy',
        'scipy',
        'matplotlib>=1.4',
    ],
    zip_safe=False,
)
