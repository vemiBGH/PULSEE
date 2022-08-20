from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom 

n = 3/2
(Ix, Iy, Iz) = spin_J_set(n)
Ix2 = Ix**2
Iy2 = Iy**2
Iz2 = Iz**2
Im = Ix - 1j*Iy
Ip = Ix + 1j*Iy
I2 = Ix**2+Iy**2

def solveH(H, A1, times, dissipation, operators, opts = None, store_final_state = False):
    #no dissolve function
    if (opts is None):
        opts = Options()
        opts.atol=1e-12
        opts.rtol=1e-10
        opts.nsteps = 5000
    opts.store_final_state = store_final_state
    return mesolve(H, A1, times, dissipation, operators, options = opts)

def spinSys(i):
    global Ix, Iy, Iz, Ix2, Iy2, Iz2, Im, I2, Ip, n
    n = i
    (Ix, Iy, Iz) = spin_J_set(n)
    Ix2 = Ix**2
    Iy2 = Iy**2
    Iz2 = Iz**2
    Im = Ix - 1j*Iy
    Ip = Ix + 1j*Iy
    I2 = Ix**2+Iy**2

#program to plot the histogram of a state
def plotHisto(state, lim):
    #lim is the maximum value of the matrix [0, lim]
    xlabels = ["skip"]
    xlabels = []
    i = n
    while i >= -n:
        if(type(i)==float):
            j = i.as_integer_ratio()
            xlabels.append(str(j[0])+'/'+str(j[1]))
        else:
            xlabels.append(i)
        i-=1

    fig, ax = matrix_histogram_complex(state, xlabels, xlabels, limits=lim, colorbar=True)
    ax.view_init(azim=-45, elev=35)
    #plt.savefig('books_read.png')
    plt.show()
    

def FID_signal(H0, dm, acquisition_time, T2=0.1e-3, theta=0, phi=0, reference_frequency=0, n_points=10000): 
    times = np.linspace(start=0, stop=acquisition_time, num=n_points)
    Iplus = Ix+1j*Iy
    # Computes the FID assuming that the detection coils record the time-dependence of the
    # magnetization on the plane perpendicular to (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta))
    I_plus_rotated = (1j*phi*Iz).expm()*(1j*theta*Iy).expm()*Iplus*(-1j*theta*Iy).expm()*(-1j*phi*Iz).expm()
    
    result = solveH(H0, dm, times, [], I_plus_rotated)
    
    FID = np.array(result.expect[0]) * np.exp(-times/T2)*np.exp(-1j*2*np.pi*reference_frequency*times)

    return np.array(FID), times


def FT_Stephen(signal, times, abs=False, padding=None):
    """
    Computes the Fourier transform of the passed time-dependent signal.

    Parameters
    ----------
    - `signal`: array-like:
              Sampled signal to be transformed in the frequency domain (in a.u.).
    - `times`: array-like
             Sampled time domain (in microseconds).
    - `abs`: Boolean 
             Whether to return the absolute value of the computer Fourier transform. 
    - `padding`: Integer
             Amount of zero-padding to add to signal in the power of zeroes.
    
    Returns
    -------
    The frequency and fourier-transformed signal as a tuple (f, ft)
    """
    if padding is not None: 
        # This code by Stephen Carr
        nt = len(times) #number of points
        
        # zero pad the ends to "interpolate" in frequency domain
        zn = padding # power of zeros
        N_z = 2 * (2 ** zn) + nt # number of elements in padded array
        zero_pad = np.zeros(N_z, dtype=complex)
        
        M0_trunc_z = zero_pad
        num = 2 ** zn
        M0_trunc_z[num:(num + nt)] = signal
        
        # figure out the "frequency axis" after the FFT
        dt = times[2] - times[1]
        Fs = 1.0 / dt # max frequency sampling

        # axis goes from - Fs / 2 to Fs / 2, with N_z steps
        freq_ax = ((np.linspace(0, N_z, N_z) - 1/2) / N_z - 1/2) * Fs
        
        M_fft = np.fft.fftshift(np.fft.fft(np.fft.fftshift(M0_trunc_z)))
        if abs:
            M_fft = np.abs(M_fft)
        return freq_ax, M_fft

    ft = np.fft.fft(signal)
    freq = np.fft.fftfreq(len(times), (times[-1] - times[0]) / len(times))
    if abs: 
        ft = np.abs(ft)
    return freq, ft

def calcSPLIT(H0, rho_0, rho_xTh, acquisition_time = 150e-3, T2 = 0.1e-3, n_points=10000):
    fid, times = FID_signal(H0, rho_0, acquisition_time, T2, n_points=n_points)
    fr, ft = FT_Stephen(fid, times, True, 10)
    
    if rho_xTh is not None:
        fid, times = FID_signal(H0, rho_xTh, acquisition_time, T2, n_points=n_points)
        fr1, ft1 = FT_Stephen(fid, times, True, 10)
        return fr, ft, fr1, ft1
 
    return fr, ft
    
def injectT2(result, T2):
    result.expect = result.expect * np.exp(-result.times/T2)
    return result
    