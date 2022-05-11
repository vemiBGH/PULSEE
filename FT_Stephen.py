import numpy as np

def FT_Stephen(signal, times):
    nt = len(times) #number of points
    
    # zero pad the ends to "interpolate" in frequency domain
    zn = 10; # power of zeros
    N_z = 2 * (2 ** zn) + nt # number of elements in padded array
    zero_pad = np.zeros(N_z, dtype=complex)
    
    M0_trunc_z = zero_pad
    num = 2 ** zn
    M0_trunc_z[num:(num + nt)] = signal
    
    # figure out the "frequency axis" after the FFT
    dt = times[2] - times[1]
    Fs = 1.0 / (dt); # max frequency sampling

    # axis goes from -Fs/2 to Fs/2, with N_z steps
    freq_ax = ((np.linspace(0, N_z, N_z) - 1/2) / N_z - 1/2) * Fs
    
    M_fft = np.fft.fftshift(np.fft.fft(np.fft.fftshift(M0_trunc_z)))
    
    return(freq_ax, M_fft)