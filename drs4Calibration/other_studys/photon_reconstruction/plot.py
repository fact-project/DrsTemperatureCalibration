import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import griddata


# Define a class that forces representation of float to look a certain way
# This remove trailing zero so '1.0' becomes '1'
class nf(float):
    def __repr__(self):
        str = '%.1f' % (self.__float__(),)
        if str[-1] == '0':
            return '%.0f' % self.__float__()
        else:
            return '%.1f' % self.__float__()

mv_to_spe = 0.1

level = [75, 80, 85, 90, 93, 95, 97, 98, 99, 99.4, 99.7, 99.9]  # in %

step_size = 0.005
offset_range = np.arange(-0.2, 0.2+step_size, step_size)

method = 'cubic'  # methods:'nearest', 'linear', 'cubic')):

fig, axarr = plt.subplots(3, sharex=True, figsize=(6, 8))
plt.subplots_adjust(hspace=0.02)

# Label levels with specially formatted floats
if plt.rcParams["text.usetex"]:

    fmt = r'%r \%%'
else:
    fmt = '%r %%'

################################
# 1_photon_reconstruction_rate #
################################
load_file = '1_photon_reconstruction_rate'
with pd.HDFStore(load_file+'.h5') as store:
    df = store['df']

X, Y = np.meshgrid(offset_range,
                   np.linspace(0.1, 0.4, len(offset_range)))

px = np.array(df['offset'])*mv_to_spe
py = np.array(df['noise'])*mv_to_spe
Z = np.array(df['reconstruction_rate'])

Ti = griddata((px, py), Z, (X, Y), method=method)

axarr[2].contourf(X, Y, Ti, alpha=.75, levels=level, cmap='bone')
CS_1 = axarr[2].contour(X, Y, Ti, levels=level, colors='black')

# Recast levels to new class
CS_1.levels = [nf(val) for val in CS_1.levels]

axarr[2].clabel(CS_1, CS_1.levels, inline=True, fmt=fmt, fontsize=10)
axarr[2].set_ylabel('Standardabweichung / Spe')
axarr[2].set_yticks(np.arange(0.15, 0.35+0.1, 0.1))
axarr[2].set_ylim(0.1, 0.4)
axarr[2].tick_params(direction='inout', top=False, right=False, which='both')

################################
# 2_photon_reconstruction_rate #
################################
load_file = '2_photon_reconstruction_rate'
with pd.HDFStore(load_file+'.h5') as store:
    df = store['df']

X, Y = np.meshgrid(offset_range,
                   np.linspace(0.2, 0.5, len(offset_range)))

px = np.array(df['offset'])*mv_to_spe
py = np.array(df['noise'])*mv_to_spe
Z = np.array(df['reconstruction_rate'])

Ti = griddata((px, py), Z, (X, Y), method=method)

axarr[1].contourf(X, Y, Ti, alpha=.75, levels=level, cmap='bone')
CS_2 = axarr[1].contour(X, Y, Ti, levels=level, colors='black')

# Recast levels to new class
CS_2.levels = [nf(val) for val in CS_2.levels]

axarr[1].clabel(CS_2, CS_2.levels, inline=True, fmt=fmt, fontsize=10)
axarr[1].set_ylabel('Standardabweichung / Spe')
axarr[1].set_yticks(np.arange(0.25, 0.45+0.1, 0.1))
axarr[1].set_ylim(0.2, 0.5)
axarr[1].tick_params(direction='inout', right=False, which='both')

################################
# 3_photon_reconstruction_rate #
################################
load_file = '3_photon_reconstruction_rate'
with pd.HDFStore(load_file+'.h5') as store:
    df = store['df']

X, Y = np.meshgrid(offset_range,
                   np.linspace(0.3, 0.6, len(offset_range)))
px = np.array(df['offset'])*mv_to_spe
py = np.array(df['noise'])*mv_to_spe
Z = np.array(df['reconstruction_rate'])

Ti = griddata((px, py), Z, (X, Y), method=method)

axarr[0].contourf(X, Y, Ti, alpha=.75, levels=level, cmap='bone')
CS_3 = axarr[0].contour(X, Y, Ti, levels=level, colors='black')

# Recast levels to new class
CS_3.levels = [nf(val) for val in CS_3.levels]

axarr[0].clabel(CS_3, CS_3.levels, inline=True, fmt=fmt, fontsize=10)
axarr[0].set_ylabel('Standardabweichung / Spe')
axarr[0].set_yticks(np.arange(0.35, 0.55, 0.1))
axarr[0].set_ylim(0.3, 0.6)

axarr[2].set_xlabel('Y-Achsenversatz / Spe')
axarr[2].set_xticks(np.arange(-0.2, 0.2+0.1, 0.05))
axarr[2].set_xlim(-0.2, 0.2)
axarr[2].tick_params(direction='inout', right=False, which='both')
plt.tight_layout()
plt.savefig('photon_reconstruction_rate.png', dpi=200)
plt.show()
