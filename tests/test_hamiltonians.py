import numpy as np
import pandas as pd

import hypothesis.strategies as st
from hypothesis import given, note

from qutip import Qobj, rand_dm

from pulsee import random_operator, random_observable

from pulsee import ptrace_subspace

from pulsee.nuclear_spin import NuclearSpin, ManySpins

from pulsee.hamiltonians import h_zeeman, h_quadrupole, \
                         v0_EFG, v1_EFG, v2_EFG, \
                         h_single_mode_pulse, \
                         h_multiple_mode_pulse, \
                         h_changed_picture, \
                         h_j_coupling, h_tensor_coupling

@given(par = st.lists(st.floats(min_value=0, max_value=20), min_size=3, max_size=3))
def test_zeeman_hamiltonian_changes_sign_when_magnetic_field_is_flipped(par):
    spin = NuclearSpin()
    h_z1 = h_zeeman(spin, par[0], par[1], par[2])
    h_z2 = h_zeeman(spin, np.pi-par[0], par[1]+np.pi, par[2])
    note("h_zeeman(theta, phi) = %r" % (h_z1))
    note("h_zeeman(pi-theta, phi+pi) = %r" % (h_z2))
    note("h_zeeman(pi-theta, phi+pi)+h_zeeman(theta, phi) = %r" % (np.absolute(h_z1.full() + h_z2.full())))
    assert np.all(np.absolute(h_z1.full() + h_z2.full()) < 1e-10)
    
@given(gamma = st.lists(st.floats(min_value=0, max_value=2*np.pi), min_size=2, max_size=2))
def test_h_quadrupole_independent_of_gamma_when_EFG_is_symmetric(gamma):
    spin = NuclearSpin()
    h_q1 = h_quadrupole(spin, 1, 0, 1, 1, gamma[0])
    h_q2 = h_quadrupole(spin, 1, 0, 1, 1, gamma[1])
    note("h_quadrupole(gamma1) = %r" % (h_q1))
    note("h_quadrupole(gamma2) = %r" % (h_q2))
    assert np.all(np.absolute(h_q1.full() - h_q2.full()) < 1e-10)
    
@given(eta = st.floats(min_value=0, max_value=1))
def test_v0_reduces_to_one_half_when_angles_are_0(eta):
    v0 = v0_EFG(eta, 0, 0, 0)
    assert np.isclose(1/2, v0, rel_tol=1e-10)
    
def test_v1_reduces_to_0_when_angles_are_0():
    for sign in [-1, +1]:
        v1 = v1_EFG(sign, 0.5, 0, 0, 0)
        assert np.absolute(v1) < 1e-10
        
@given(eta = st.floats(min_value=0, max_value=1))
def test_v2_becomes_proportional_to_eta_when_angles_are_0(eta):
    for sign in [-2, +2]:
        v2 = v2_EFG(sign, eta, 0, 0, 0)
        assert np.isclose(v2, eta/(2*np.sqrt(6)), rtol=1e-10)
        
@given(n = st.integers(min_value=-20, max_value=20))
def test_periodicity_pulse_hamiltonian(n):
    spin = NuclearSpin(1., 1.)
    nu = 5.
    t1 = 1.
    t2 = t1 + 2 * np.pi * n/nu 
    h_p1 = h_single_mode_pulse(spin, nu, 10., 0, np.pi/2, 0, t1)
    h_p2 = h_single_mode_pulse(spin, nu, 10., 0, np.pi/2, 0, t2)
    note("h_single_mode_pulse(t1) = %r" % (h_p1))
    note("h_single_mode_pulse(t2) = %r" % (h_p2))
    assert np.all(np.isclose(h_p1.full(), h_p2.full(), rtol=1e-10))

# Checks that the superposition of two orthogonal pulses with the same frequency and a phase difference
# of pi/2 is equivalent to the time-reversed superposition of the two same pulses with one of them
# changed by sign
@given(t = st.floats(min_value=0, max_value=20))
def test_time_reversal_equivalent_opposite_circular_polarization(t):
    spin = NuclearSpin(1., 1.)
    mode_forward = pd.DataFrame([(5., 10., 0., 0., 0.),
                                 (5., 10., np.pi/2, np.pi/2, 0.)], 
                                columns=['frequency', 'amplitude', 'phase', 'theta_p', 'phi_p'])
    mode_backward = pd.DataFrame([(5., 10., 0., 0., 0.),
                                  (5., 10., -np.pi/2, np.pi/2, 0.)], 
                                 columns=['frequency', 'amplitude', 'phase', 'theta_p', 'phi_p'])
    h_p_forward = h_multiple_mode_pulse(spin, mode_forward, t)
    h_p_backward = h_multiple_mode_pulse(spin, mode_backward, -t)
    assert np.all(np.isclose(h_p_forward, h_p_backward, rtol=1e-10))
    
# Checks that the Hamiltonian of the pulse expressed in the interaction picture is equal to that in the
# Schroedinger picture when it commutes with the unperturbed Hamiltonian
def test_interaction_picture_leaves_pulse_hamiltonian_unaltered_when_commutative_property_holds():
    spin = NuclearSpin(1., 1.)
    mode = pd.DataFrame([(5., 10., 0., np.pi/2, 0.)], 
                        columns=['frequency', 'amplitude', 'phase', 'theta_p', 'phi_p'])
    h_unperturbed = 5.*spin.I['x']
    h_pulse = h_multiple_mode_pulse(spin, mode, 10.)
    h_pulse_ip = h_changed_picture(spin, mode, h_unperturbed, h_unperturbed, 10.)
    assert np.all(np.isclose(h_pulse, h_pulse_ip, rtol=1e-10))
    
def test_ptrace_subspace_j_coupling_hamiltonian_over_non_interacting_spins_subspaces():
    spins = []
    for i in range(4):
        spins.append(NuclearSpin())
    
    spin_system = ManySpins(spins)
    
    j_matrix = np.zeros((4, 4))
    
    for i in range(3):
        j_matrix[i, i+1]
        
    h_j = h_j_coupling(spin_system, j_matrix)
    
    h_j_1 = ptrace_subspace(h_j, [3, 3, 3, 3], 1)
    h_j_2 = ptrace_subspace(h_j, [3, 3, 3, 3], 2)
    
    assert np.all(np.isclose(h_j_2, Qobj(np.zeros((27, 27))), rtol = 1e-10))
    
def test_h_tensor_j_coupling_two_half_spin_system():
    spins = ManySpins([NuclearSpin(0.5), NuclearSpin(0.5)])

    # https://www.weizmann.ac.il/chembiophys/assaf_tal/sites/chemphys.assaf_tal/files/uploads/lecture_ii_-_nmr_interactions.pdf
    j_coeff = 275
    j = 2 * np.pi * j_coeff * np.eye(3) # Hz
    computed_h_j_coupling = h_tensor_coupling(spins, j)
    expected_h_j_coupling = np.array([[j_coeff * np.pi / 2, 0, 0, 0],
                                      [0, - np.pi * j_coeff / 2, np.pi * j_coeff, 0],
                                      [0, np.pi * j_coeff, - np.pi * j_coeff / 2, 0], 
                                     [0, 0, 0, np.pi * j_coeff / 2]])
    assert np.array_equal(computed_h_j_coupling, expected_h_j_coupling)


