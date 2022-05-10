import numpy as np
import pandas as pd
import math

from qutip import mesolve, Qobj, rand_dm, tensor

import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import xticks, yticks

import hypothesis.strategies as st
from hypothesis import given, assume

from pulsee.operators import random_density_matrix, changed_picture

from pulsee.nuclear_spin import NuclearSpin

from pulsee.hamiltonians import h_j_coupling

from pulsee.simulation import nuclear_system_setup, \
                       power_absorption_spectrum, \
                       evolve, \
                       RRF_operator, \
                       FID_signal, \
                       fourier_transform_signal, \
                       fourier_phase_shift

def test_null_zeeman_contribution_for_0_gyromagnetic_ratio():
    spin_par = {'quantum number' : 3/2,
                'gamma/2pi' : 0.}    
    zeem_par = {'field magnitude' : 10.,
                'theta_z' : 0,
                'phi_z' : 0}
    
    quad_par = {'coupling constant' : 0.,
                'asymmetry parameter' : 0.,
                'alpha_q' : 0,
                'beta_q' : 0,
                'gamma_q' : 0}
    
    nuclear_system_setup(spin_par, quad_par, zeem_par)

    h_unperturbed = Qobj(np.sum(nuclear_system_setup(spin_par, quad_par, 
                                zeem_par)[1], axis=0))
    
    null_matrix = np.zeros((4, 4))
    
    assert np.all(np.isclose(h_unperturbed, null_matrix, rtol=1e-10))
    
    
@given(s = st.integers(min_value=1, max_value=14))
def test_correct_number_lines_power_absorption_spectrum(s):
    
    spin_par = {'quantum number' : s/2,
                'gamma/2pi' : 1.}

    zeem_par = {'field magnitude' : 10.,
                'theta_z' : math.pi/4,
                'phi_z' : 0}
    
    quad_par = {'coupling constant' : 5.,
                'asymmetry parameter' : 0.3,
                'alpha_q' : math.pi/3,
                'beta_q' : math.pi/5,
                'gamma_q' : 0}
    
    spin, h_unperturbed, dm_0 = nuclear_system_setup(spin_par, quad_par, zeem_par)
    
    f, p = power_absorption_spectrum(spin, Qobj(np.sum(h_unperturbed, axis=0)), 
                        normalized=False, dm_initial=dm_0)
    
    assert len(f)==(spin.d)*(spin.d-1)/2
    

def test_pi_pulse_yields_population_inversion():
    spin_par = {'quantum number' : 5/2,
                'gamma/2pi' : 1.}
    
    zeem_par = {'field magnitude' : 10.,
                'theta_z' : 0,
                'phi_z' : 0}
    
    quad_par = {'coupling constant' : 0.,
                'asymmetry parameter' : 0.,
                'alpha_q' : 0,
                'beta_q' : 0,
                'gamma_q' : 0}
    
    initial_state = np.zeros((6, 6))
    initial_state[0, 0] = 1
    
    spin, h_unperturbed, dm_initial = nuclear_system_setup(spin_par, quad_par, zeem_par, \
                                                           initial_state=initial_state)
    
    mode = pd.DataFrame([(10., 1., 0., math.pi/2, 0)], 
                        columns=['frequency', 'amplitude', 'phase', 'theta_p', 'phi_p'])
    
    # For now test using mesolve master equation solver
    # TODO test fails because evolved density matrix has one eigenvalue JUST 
    # below positivity thresh. consider more forgiving threshold.
    dm_evolved = evolve(spin, h_unperturbed, dm_initial,
                                    mode=mode, pulse_time=2 * np.pi, picture='IP')
    
    assert np.all(np.isclose(dm_evolved[5, 5], 1, rtol=1e-1))
    

def test_evolution_goes_fine_for_low_pulse_duration():
    spin_par = {'quantum number' : 1,
                'gamma/2pi' : 1.}
    
    zeem_par = {'field magnitude' : 10.,
                'theta_z' : 0,
                'phi_z' : 0}
    
    quad_par = {'coupling constant' : 0.,
                'asymmetry parameter' : 0.,
                'alpha_q' : 0,
                'beta_q' : 0,
                'gamma_q' : 0}
    
    spin, h_unperturbed, dm_initial = nuclear_system_setup(spin_par, quad_par, zeem_par, \
                                                           initial_state='canonical')
    
    mode = pd.DataFrame([(10., 1., 0., math.pi/2, 0)], 
                        columns=['frequency', 'amplitude', 'phase', 'theta_p', 'phi_p'])
    
    dm_evolved = evolve(spin, h_unperturbed, dm_initial,
                        mode, pulse_time=0.01, \
                        picture='IP')
    
    
def test_j_coupling_refocusing_sequence():
    spin_par1 = {'quantum number' : 1/2,
                 'gamma/2pi' : 1.}
    
    spin_par2 = {'quantum number' : 1/2,
                 'gamma/2pi' : 5.}
    
    zeem_par = {'field magnitude' : 5.,
                'theta_z' : 0,
                'phi_z' : 0}
    
    initial_state1 = random_density_matrix(2)
    initial_state2 = random_density_matrix(2)
    initial_state = tensor(initial_state1, initial_state2)
    
    spins, h_zeeman, initial_dm = nuclear_system_setup([spin_par1, spin_par2], \
                                                       None, \
                                                       zeem_par, \
                                                       initial_state=initial_state)
    
    j_matrix = np.zeros((2, 2))
    j_matrix[0, 1] = 0.1
    
    h_j = h_j_coupling(spins, j_matrix)
    
    h_unperturbed = h_zeeman + [Qobj(h_j)]
    
    #no_pulse = pd.DataFrame([(0., 0., 0., 0., 0.)], 
    #                        columns=['frequency', 'amplitude', 'phase', 'theta_p', 'phi_p'])
    
    pulse_mode = pd.DataFrame([(25., 2., 0., math.pi/2, 0)], 
                              columns=['frequency', 'amplitude', 'phase', 'theta_p', 'phi_p'])
        
    dm_1 = evolve(spins, h_unperturbed, initial_dm, 
                  None, pulse_time=1, \
                  picture='IP')
        
    dm_2 = evolve(spins, h_unperturbed, dm_1, 
                  pulse_mode, pulse_time=0.1, \
                  picture='IP')
        
    dm_3 = evolve(spins, h_unperturbed, dm_2, 
                  None, pulse_time=1, \
                  picture='IP')
        
    final_dm = evolve(spins, h_unperturbed, dm_3, 
                      pulse_mode, pulse_time=0.1, \
                      picture='IP')
        
    j_evolved_dm = changed_picture(final_dm, Qobj(np.array(h_zeeman[0])), 2.2)
        
    assert np.all(np.isclose(initial_dm.full(), j_evolved_dm.full(), rtol=1, atol=5e-2))


def test_RRF_operator_proportional_to_Iz_for_theta_0():
    
    spin = NuclearSpin(3/2, 1.)
    
    RRF_par = {'nu_RRF': 10,
              'theta_RRF': 0,
              'phi_RRF': 0}
    
    RRF_o = RRF_operator(spin, RRF_par)
    
    RRF_matrix = RRF_o
    Iz_matrix = spin.I['z']
    
    c = RRF_matrix[0, 0]/Iz_matrix[0, 0]
    
    assert np.all(np.isclose(RRF_matrix, c*Iz_matrix, rtol=1e-10))

    
def test_FID_signal_decays_fast_for_small_relaxation_time():
    spin_par = {'quantum number' : 2,
                'gamma/2pi' : 1.}
    
    zeem_par = {'field magnitude' : 10.,
                'theta_z' : math.pi/4,
                'phi_z' : 0}
    
    quad_par = {'coupling constant' : 5.,
                'asymmetry parameter' : 0.3,
                'alpha_q' : math.pi/3,
                'beta_q' : math.pi/5,
                'gamma_q' : 0}
    
    initial_matrix = np.zeros((5, 5))
    initial_matrix[0, 0] = 1
    
    spin, h_unperturbed, dm_0 = nuclear_system_setup(spin_par, quad_par, zeem_par,
                                                     initial_state=initial_matrix)
    
    mode = pd.DataFrame([(10., 1., 0., math.pi/2, 0)], 
                        columns=['frequency', 'amplitude', 'phase', 'theta_p', 'phi_p'])
    
    dm_evolved = evolve(spin, h_unperturbed, dm_0, 
                        mode, pulse_time=math.pi, \
                        picture='IP', \
                        n_points=10)
    
    h_unperturbed_op = Qobj(np.array(h_unperturbed[0]))
    for i in h_unperturbed[1:]:
        i += Qobj(np.array(h_unperturbed_op))

    t, signal = FID_signal(spin, [h_unperturbed_op], dm_evolved, acquisition_time=100, T2=1)
    
    assert np.absolute(signal[-1])<1e-10
    

def test_pure_zeeman_FID_is_periodic_for_long_relax_times():
    spin_par = {'quantum number' : 3/2,
                'gamma/2pi' : 1.}
    
    H_0 = 1
    
    zeem_par = {'field magnitude' : H_0,
                'theta_z' : 0,
                'phi_z' : 0}
    
    quad_par = {'coupling constant' : 0.,
                'asymmetry parameter' : 0.,
                'alpha_q' : 0.,
                'beta_q' : 0.,
                'gamma_q' : 0.}
    
    spin, h_unperturbed, dm_0 = nuclear_system_setup(spin_par, quad_par, zeem_par)
    
    nu = spin_par['gamma/2pi']*H_0
    
    mode = pd.DataFrame([(nu, 1., 0., math.pi/2, 0)], 
                        columns=['frequency', 'amplitude', 'phase', 'theta_p', 'phi_p'])
    
    dm_evolved = evolve(spin, h_unperturbed, dm_0, \
                        mode, pulse_time=1/4, \
                        picture='IP')
    
    t, signal = FID_signal(spin, h_unperturbed, dm_evolved, acquisition_time=1, T2=1e3, n_points=1e3)
        
    t1 = 0
    t2 = 0
    for i in range(len(t)):
        t2 = t[i]
        if np.absolute(t2-1/nu) < 1e-3:
            assert (np.absolute(signal[i]-signal[0])) < 1e-2
            return
    
    raise AssertionError("The sampling of the acquisition time window isn't dense enough to reproduce the periodicity of the FID signal")


def test_opposite_fourier_transform_when_FID_differ_of_pi():
    spin_par = {'quantum number' : 3,
                'gamma/2pi' : 1.}
    
    zeem_par = {'field magnitude' : 10.,
                'theta_z' : math.pi/4,
                'phi_z' : 0}
    
    quad_par = {'coupling constant' : 5.,
                'asymmetry parameter' : 0.3,
                'alpha_q' : math.pi/3,
                'beta_q' : math.pi/5,
                'gamma_q' : 0}
    
    initial_matrix = np.zeros((7, 7))
    initial_matrix[0, 0] = 1
    
    spin, h_unperturbed, dm_0 = nuclear_system_setup(spin_par, quad_par, zeem_par,
                                                     initial_state=initial_matrix)
    
    mode = pd.DataFrame([(10., 1., 0., math.pi/2, 0)], 
                        columns=['frequency', 'amplitude', 'phase', 'theta_p', 'phi_p'])
    
    dm_evolved = evolve(spin, h_unperturbed, dm_0, \
                        mode, pulse_time=math.pi, \
                        picture='IP', \
                        n_points=10)
    
    t, signal1 = FID_signal(spin, h_unperturbed, dm_evolved, acquisition_time=250, T2=100)
    t, signal2 = FID_signal(spin, h_unperturbed, dm_evolved, acquisition_time=250, T2=100, phi=math.pi)
    
    f, fourier1 = fourier_transform_signal(t, signal1, 7.5, 12.5)
    f, fourier2 = fourier_transform_signal(t, signal2, 7.5, 12.5)
    
    assert np.all(np.isclose(fourier1, -fourier2, rtol=1e-10))

# Checks that the Fourier transform of the signal has the same shape both after adding the phase
# computed by fourier_phase_shift directly to the FID signal and by rotating the detection coil by the
# same angle
def test_two_methods_phase_adjustment():
    spin_par = {'quantum number' : 3/2,
                'gamma/2pi' : 1.}
    
    zeem_par = {'field magnitude' : 10.,
                'theta_z' : 0,
                'phi_z' : 0}
    
    quad_par = {'coupling constant' : 0,
                'asymmetry parameter' : 0,
                'alpha_q' : 0,
                'beta_q' : 0,
                'gamma_q' : 0}
    
    initial_matrix = np.zeros((4, 4))
    initial_matrix[0, 0] = 1
    
    spin, h_unperturbed, dm_0 = nuclear_system_setup(spin_par, quad_par, zeem_par,
                                                     initial_state=initial_matrix)
    
    mode = pd.DataFrame([(10., 1., 0., math.pi/2, 0)], 
                        columns=['frequency', 'amplitude', 'phase', 'theta_p', 'phi_p'])
    
    dm_evolved = evolve(spin, h_unperturbed, dm_0, \
                        mode, pulse_time=math.pi, \
                        picture='IP', \
                        n_points=10)
    
    t, fid = FID_signal(spin, h_unperturbed, dm_evolved, acquisition_time=500)
    f, fourier0 = fourier_transform_signal(t, fid, 9, 11)
            
    phi = fourier_phase_shift(f, fourier0, peak_frequency=10, int_domain_width=1)
    fourier1 = np.exp(1j*phi)*fourier0
            
    t, fid_rephased = FID_signal(spin, h_unperturbed, dm_evolved, acquisition_time=500, phi=phi)
    f, fourier2 = fourier_transform_signal(t, fid_rephased, 9, 11)
        
    assert np.all(np.isclose(fourier1, fourier2, rtol=1e-10))


    
    
    
    