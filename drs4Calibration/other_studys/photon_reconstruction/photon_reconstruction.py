import numpy as np
import pandas as pd

from tqdm import tqdm

np.random.seed(0)


mv_to_spe = 0.1
time_range_in_ns = np.arange(0, 100, 0.5)  # time in ns
pulse_pos_in_ns = 25
integration_window_len = 30  # slices
extraction_window_start = 35  # slice index
extraction_window_len = 90  # slices
time_extraction_window = time_range_in_ns[extraction_window_start:extraction_window_start+extraction_window_len]


def fact_pulse_in_mV(x, x0=0):
    p = (1.239*(1-1/(1+np.exp(((x-x0)-2.851)/1.063)))*np.exp(-((x-x0)-2.851)/19.173))
    p *= 10  # 1 spe (10mV)
    return p


def basic_extraction(data):
    maxPos = np.argmax(data)
    maxHalf = data[maxPos] / 2.
    half_pos = np.where(data[:maxPos+1] < maxHalf)[0]
    if len(half_pos):
        half_pos = half_pos[-1]
    else:
        half_pos = extraction_window_start
    integral = data[half_pos:half_pos+30].sum()
    return {
        'arrivalTime': time_range_in_ns[extraction_window_start + half_pos],
        'integral': integral,
    }


def basic_extraction_normalized(data):
    be = basic_extraction(data)
    be['integral'] /= gain
    be['arrivalTime'] -= true_arrival_time
    return be

window_pulse = fact_pulse_in_mV(time_extraction_window, pulse_pos_in_ns)
gain = basic_extraction(window_pulse)['integral']
true_arrival_time = basic_extraction(window_pulse)['arrivalTime']

df = pd.DataFrame(columns=['noise', 'offset', 'reconstruction_rate'])
true_number_of_photons = 3
nr_runs = 50000
store_str = '{}_photon_reconstruction_rate'.format(true_number_of_photons)
print(store_str, ', with {} rep.'.format(nr_runs))
step_size = 0.05
offset_range = np.arange(-2, 2+step_size, step_size)
noise_range = np.linspace(3, 6, len(offset_range))  # np.arange(1e-7, 3, 0.2)
for offset in tqdm(offset_range):
    for noise in noise_range:
        value_list = np.zeros(nr_runs)
        for i in range(nr_runs):
            y = true_number_of_photons * fact_pulse_in_mV(time_extraction_window, pulse_pos_in_ns)
            y += offset + np.random.normal(0, noise, size=len(y))
            d = basic_extraction_normalized(y)
            value_list[i] = (d['integral'].round().astype(int) == true_number_of_photons).astype(int)
        reconstruction_rate = np.round(value_list.sum()/nr_runs*100, 2)
        df = df.append({'noise': noise,
                        'offset': offset,
                        'reconstruction_rate': reconstruction_rate},
                       ignore_index=True)
with pd.HDFStore(store_str+'.h5') as store:
    store['df'] = df
