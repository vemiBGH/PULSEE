import numpy as np
import pandas as pd
import math

import matplotlib.pylab as plt

from Operators import *

from Nuclear_Spin import *

from Hamiltonians import *
    
# Computes the energy spectrum of a spin 3/2 nucleus where the quadrupole interaction is a small
# perturbation of the Zeeman energy levels, and plots the difference between the satellite and central
# transition frequencies as a function of the angle between the magnetic field and the crystal axis
def Quadrupole_Perturbation_Satellite_Frequency_Shift():
    spin = Nuclear_Spin(3/2, 1.)
    
    h_z = h_zeeman(spin, 0., 0., 5.)
    
    field_crystal_angles = np.linspace(0, math.pi, num=50)
    
    frequency_shift = {}
    
    x = 360*field_crystal_angles/(2*math.pi)
    y = []
    
    for theta_q in field_crystal_angles:
        
        h_q = h_quadrupole(spin, 0.1, 0., 0., theta_q, 0.)
        
        h_unperturbed = Observable(h_z.matrix + h_q.matrix)
        
        energy_spectrum = h_unperturbed.diagonalisation()[0]
        
        energy_spectrum = np.sort(energy_spectrum)
        
        satellite_frequency = energy_spectrum[3] - energy_spectrum[2]
        
        central_frequency = energy_spectrum[2] - energy_spectrum[1]
        
        frequency_shift[theta_q] = satellite_frequency - central_frequency
        
        y.append(frequency_shift[theta_q])

    plt.plot(x, np.real(y))
    
    plt.xlabel("\N{GREEK SMALL LETTER THETA} (\N{DEGREE SIGN})")    
    plt.ylabel("\N{GREEK SMALL LETTER NU}3/2 - \N{GREEK SMALL LETTER NU}1/2 (MHz)")
    
    plt.savefig('SatelliteFrequencyShift')
    
    plt.show()
    

# Computes the energy spectrum of a spin 5/2 nucleus where the quadrupole interaction is a small
# perturbation of the Zeeman energy levels, and plots central transition frequency as a function of the
# angle between the magnetic field and the crystal axis
def Quadrupole_Perturbation_Central_Frequency_Shift():
    spin = Nuclear_Spin(5/2, 1.)
    
    h_z = h_zeeman(spin, 0., 0., 5.)
    
    field_crystal_angles = np.linspace(0, 2*math.pi, num=100)
    
    central_frequency = {}
    
    x = 360*field_crystal_angles/(2*math.pi)
    y = []
    
    for theta_q in field_crystal_angles:
        
        h_q = h_quadrupole(spin, 0.1, 0., 0., theta_q, 0.)
        
        h_unperturbed = Observable(h_z.matrix + h_q.matrix)
        
        energy_spectrum = h_unperturbed.diagonalisation()[0]
        
        energy_spectrum = np.sort(energy_spectrum)
                
        central_frequency[theta_q] = energy_spectrum[3] - energy_spectrum[2]
        
        y.append(central_frequency[theta_q])

    plt.plot(x, np.real(y))
    
    plt.xlabel("\N{GREEK SMALL LETTER THETA} (\N{DEGREE SIGN})")    
    plt.ylabel("\N{GREEK SMALL LETTER NU}1/2 (MHz)")
    
    plt.savefig('CentralFrequencyShift')

    plt.show()