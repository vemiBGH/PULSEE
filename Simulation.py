import numpy as np
import pandas as pd
import math

import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import xticks, yticks
from matplotlib.axes import Axes

from Operators import *

from Nuclear_Spin import *

from Hamiltonians import *


def nuclear_system_setup(spin_par, zeem_par, quad_par, initial_state='canonical', temperature=1e-4):
    
    spin = Nuclear_Spin(spin_par['quantum number'], \
                        spin_par['gamma/2pi'])
    
    h_z = h_zeeman(spin, zeem_par['theta_z'], \
                         zeem_par['phi_z'], \
                         zeem_par['field magnitude'])
    
    h_q = h_quadrupole(spin, quad_par['coupling constant'], \
                             quad_par['asymmetry parameter'], \
                             quad_par['alpha_q'], \
                             quad_par['beta_q'], \
                             quad_par['gamma_q'])
    
    h_unperturbed = Observable(h_z.matrix + h_q.matrix)
    
    if isinstance(initial_state, str) and initial_state == 'canonical':
        dm_initial = canonical_density_matrix(h_unperturbed, temperature)
    else:
        dm_initial = Density_Matrix(initial_state)
    
    return spin, h_unperturbed, dm_initial


def evolve(spin, h_unperturbed, dm_initial, \
           mode, pulse_time, \
           picture='RRF', RRF_par={'nu_RRF': 0,
                                   'theta_RRF': 0,
                                   'phi_RRF': 0}, \
           n_points=10, order=2):
    
    if pulse_time == 0 or np.all(np.absolute((dm_initial-Operator(spin.d)).matrix)<1e-10):
        return dm_initial
    
    if picture == 'IP':
        o_change_of_picture = h_unperturbed
    else:
        o_change_of_picture = RRF_operator(spin, RRF_par)
    
    times, time_step = np.linspace(0, pulse_time, num=max(2, int(pulse_time*n_points)), retstep=True)
    h_new_picture = []
    for t in times:
        h_new_picture.append(h_changed_picture(spin, mode, h_unperturbed, o_change_of_picture, t))
    
    magnus_exp = magnus_expansion_1st_term(h_new_picture, time_step)
    if order>1:
        magnus_exp = magnus_exp + magnus_expansion_2nd_term(h_new_picture, time_step)
        if order>2:
            magnus_exp = magnus_exp + magnus_expansion_3rd_term(h_new_picture, time_step)

    dm_evolved_new_picture = dm_initial.sim_trans(-magnus_exp, exp=True)

    dm_evolved = dm_evolved_new_picture.changed_picture(o_change_of_picture, pulse_time, invert=True)
    
    return Density_Matrix(dm_evolved.matrix)


# Operator which generates a change of picture equivalent to moving to the rotating reference frame
# (RRF)
def RRF_operator(spin, RRF_par):
    nu = RRF_par['nu_RRF']
    theta = RRF_par['theta_RRF']
    phi = RRF_par['phi_RRF']
    RRF_o = nu*(spin.I['z']*math.cos(theta) + \
                spin.I['x']*math.sin(theta)*math.cos(phi) + \
                spin.I['y']*math.sin(theta)*math.sin(phi))
    return Observable(RRF_o.matrix)


def plot_real_part_density_matrix(dm, show=True, save=False, name='RealPartDensityMatrix', destination=''):
    
    real_part = np.vectorize(np.real)
    data_array = real_part(dm.matrix)
    
    # Create a figure for plotting the data as a 3D histogram.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create an X-Y mesh of the same dimension as the 2D data
    # You can think of this as the floor of the plot
    x_data, y_data = np.meshgrid(np.arange(data_array.shape[1])+0.25,
                                 np.arange(data_array.shape[0])+0.25)
    
    # Set width of the vertical bars
    dx = dy = 0.5

    # Flatten out the arrays so that they may be passed to "ax.bar3d".
    # Basically, ax.bar3d expects three one-dimensional arrays: x_data, y_data, z_data. The following
    # call boils down to picking one entry from each array and plotting a bar from (x_data[i],
    # y_data[i], 0) to (x_data[i], y_data[i], z_data[i]).
    x_data = x_data.flatten()
    y_data = y_data.flatten()
    z_data = data_array.flatten()
    ax.bar3d(x_data,
             y_data,
             np.zeros(len(z_data)),
             dx, dy, z_data)
    
    s = (data_array.shape[0]-1)/2

    xticks(np.arange(start=0.5, stop=data_array.shape[0]+0.5), np.arange(start=s, stop=-s-1, step=-1))
    yticks(np.arange(start=0.5, stop=data_array.shape[0]+0.5), np.arange(start=s, stop=-s-1, step=-1))
    
    ax.set_xlabel("m")    
    ax.set_ylabel("m")
    ax.set_zlabel("Re(\N{GREEK SMALL LETTER RHO})")
    
    if save:
        plt.savefig(destination + name)
    
    if show:
        plt.show()
        
    return fig
    

def power_absorption_spectrum(spin, h_unperturbed, normalized=True, dm_initial='none'):
    
    energies, o_change_of_basis = h_unperturbed.diagonalisation()
    
    transition_frequency = []
    
    transition_intensity = []
    
    d = h_unperturbed.dimension()
    
    for i in range(d):
        for j in range(d):
            if i < j:
                nu = np.absolute(energies[j] - energies[i])
                transition_frequency.append(nu)
                
                # Operator of the magnetic moment of the spin expressed in the basis of energy
                # eigenstates, defined in order to extract the matrix elements required by Fermi's
                # golden rule
                magnetization_in_basis_of_eigenstates=\
                    spin.gyro_ratio_over_2pi*spin.I['x'].sim_trans(o_change_of_basis)
                
                intensity_nu = nu*\
                    (np.absolute(magnetization_in_basis_of_eigenstates.matrix[j, i]))**2
                
                if not normalized:
                    p_i = dm_initial.matrix[i, i]
                    p_j = dm_initial.matrix[j, j]
                    intensity_nu = np.absolute(p_i-p_j)*intensity_nu
                    
                transition_intensity.append(intensity_nu)
            else:
                pass
    
    return transition_frequency, transition_intensity


def plot_power_absorption_spectrum(frequencies, intensities, show=True, save=False, name='PowerAbsorptionSpectrum', destination=''):
    fig = plt.figure()
    
    plt.vlines(frequencies, 0, intensities, colors='b')
    
    plt.xlabel("\N{GREEK SMALL LETTER NU} (MHz)")    
    plt.ylabel("Power absorption (a. u.)")
    
    if save: plt.savefig(destination + name)
    
    if show: plt.show()
        
    return fig


def FID_signal(spin, h_unperturbed, dm, acquisition_time, T2=100, theta=0, phi=0, n_points=10):
    
    times = np.linspace(start=0, stop=acquisition_time, num=int(acquisition_time*n_points))
    
    FID = []
    
    # Computes the FID assuming that the detection coils record the time-dependence of the
    # magnetization on the plane perpendicular to (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta))
    Iz = spin.I['z']
    Iy = spin.I['y']
    I_plus_rotated = (1j*phi*Iz).exp()*(1j*theta*Iy).exp()*spin.I['+']*(-1j*theta*Iy).exp()*(-1j*phi*Iz).exp()
    for t in times:
        dm_t = dm.free_evolution(h_unperturbed, t)
        FID.append((dm_t*I_plus_rotated*np.exp(-t/T2)).trace())
    
    return times, np.array(FID)


def plot_real_part_FID_signal(times, FID, show=True, save=False, name='FIDSignal', destination=''):
    fig = plt.figure()
    
    plt.plot(times, np.real(FID), label='Real part')
        
    plt.xlabel("time (\N{GREEK SMALL LETTER MU}s)")    
    plt.ylabel("Re(FID) (a. u.)")
    
    if save: plt.savefig(destination + name)
    
    if show: plt.show()
    
    return fig


# When opposite_frequency is True, the function computes the Fourier spectrum also in the range of
# frequencies opposite to those specified as inputs and returns it
def fourier_transform_signal(signal, times, frequency_start, frequency_stop, opposite_frequency=False):
    
    dt = times[1]-times[0]
    
    frequencies = np.linspace(start=frequency_start, stop=frequency_stop, num=1000)
        
    fourier = [[], []]
    
    if opposite_frequency == False:
        sign_options = 1
    else:
        sign_options = 2
    
    for s in range(sign_options):
        for nu in frequencies:
            integral = np.zeros(sign_options, dtype=complex)
            for t in range(len(times)):
                integral[s] = integral[s] + np.exp(1j*2*math.pi*(1-2*s)*nu*times[t])*signal[t]*dt
            fourier[s].append(integral[s])
        
    if opposite_frequency == False:
        return frequencies, np.array(fourier[0])
    else:
        return frequencies, np.array(fourier[0]), np.array(fourier[1])


# Finds out the phase responsible for the displacement of the real and imaginary parts of the Fourier
# spectrum of the FID with respect to the ideal absorptive/dispersive lorentzian shapes
def fourier_phase_shift(frequencies, fourier, peak_frequency_hint, search_window=0.1):

    peak_pos = 0
    
    # Range where to look for the maximum of the square modulus of the Fourier spectrum
    search_range = np.nonzero(np.isclose(frequencies, peak_frequency_hint, atol=search_window/2))[0]
    
    # Search of the maximum of the square modulus of the Fourier spectrum
    fourier2_max=0
    for i in search_range:
        if (np.absolute(fourier[i])**2)>fourier2_max:
            fourier2_max = np.absolute(fourier[i])
            peak_pos = i
        
    re = np.real(fourier[peak_pos])
    
    im = np.imag(fourier[peak_pos])
    
    if im >= 0:
        phase = math.atan(-im/re)
    else:
        phase = math.atan(-im/re) + math.pi
    
    return phase

# If another set of data is passed as fourier_neg, the function plots a couple of graphs, with the
# one at the top interpreted as the NMR signal produced by a magnetization rotating counter-clockwise,
# the one at the bottom corresponding to the opposite sense of rotation
def plot_fourier_transform(frequencies, fourier, fourier_neg=None, square_modulus=False, show=True, save=False, name='FTSignal', destination=''):
    
    if fourier_neg is None:
        n_plots = 1
        fourier_data = [fourier]
    else:
        n_plots = 2
        fourier_data = [fourier, fourier_neg]
        plot_title = ["Counter-clockwise precession", "Clockwise precession"]
    
    fig, ax = plt.subplots(n_plots, 1, gridspec_kw={'hspace':0.5})
    
    if fourier_neg is None:
        ax = [ax]
        
    for i in range(n_plots):
        if not square_modulus:
            ax[i].plot(frequencies, np.real(fourier_data[i]), label='Real part')
            ax[i].plot(frequencies, np.imag(fourier_data[i]), label='Imaginary part')
        else:
            ax[i].plot(frequencies, np.absolute(fourier_data[i])**2, label='Square modulus')
        
        if n_plots>1:
            ax[i].title.set_text(plot_title[i])
        
        ax[i].legend(loc='upper left')
        ax[i].set_xlabel("frequency (MHz)")    
        ax[i].set_ylabel("FT signal (a. u.)")  
         
    if save: plt.savefig(destination + name)
        
    if show: plt.show()
        
    return fig






