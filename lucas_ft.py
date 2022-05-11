import numpy
import scipy.fft as fft
import numpy as np 

def fourier_transform_signal(signal, times, abs=False):
    ft = fft.fft(signal)
    freq = fft.fftfreq(len(times), (times[-1] - times[0]) / len(times))
    if abs: 
        ft = np.abs(ft)
    return freq, ft