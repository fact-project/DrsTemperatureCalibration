from astropy.io import fits
from calc_calib_constants import read_pixel, f
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import click

plt.style.use('ggplot')


@click.command()
@click.argument('datafile',
                default="/net/big-tank/POOL/" +
                        "projects/fact/drs4_calibration_data/" +
                        "calibration/calculation/time/temp/timeCalibrationData20160817_017_newVersion.fits",
                type=click.Path(exists=True))
@click.argument('chid',
                default=15)
@click.argument('cell',
                default=753)
def main(datafile: str, chid: int, cell: int):

    fits_file = fits.open(datafile,
                          mmap=True,
                          mode="denywrite",
                          ignoremissing=True,
                          ignore_missing_end=True
                          )

    pixel_data = read_pixel(fits_file, chid).query('cell == @cell')

    plt.title('Pixel {}, Cell {}'.format(chid, cell))

    mask1 = pixel_data['sample'] > 10 # 9
    mask2 = pixel_data['sample'] <= 50 # 240
    mask = pixel_data['sample'] == 100
    a = plt.scatter(
        'delta_t', 'adc_counts',
        lw=0, s=5,
        data=pixel_data[mask1 & mask2],
        label='10 ≤ sample ≤ 240',
        c=pixel_data['sample'][mask1 & mask2],
    )
    cbar = plt.colorbar(a)
    # plt.scatter(
    #     'delta_t', 'adc_counts',
    #     lw=0, s=5,
    #     data=pixel_data[~mask1],
    #     label='sample < 10',
    #     color='gray',
    # )
    # plt.scatter(
    #     'delta_t', 'adc_counts',
    #     lw=0, s=5,
    #     data=pixel_data[~mask2],
    #     label='sample > 240',
    #     color='black',
    # )

    low = pixel_data.adc_counts.min()  # quantile(0.01)
    high = pixel_data.adc_counts.max()  # quantile(0.99)

    r = high - low

    plt.ylim(
        low - 0.05 * r,
        high + 0.05 * r,
    )

    plt.xscale('log')
    plt.xlim(1e-4, 1e0)
    plt.xlabel('$\Delta t \,/\, \mathrm{s}$')
    plt.ylabel('$\mathrm{mV}$')
    plt.legend()

    plt.savefig("../../plots/test_sample_ID_10.jpg")
    plt.show()


if __name__ == '__main__':
    main()
