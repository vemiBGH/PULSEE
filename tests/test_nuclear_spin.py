import math
import numpy as np

import hypothesis.strategies as st
from hypothesis import given, settings, note

from pulsee.operators import random_operator, random_density_matrix, \
                             random_observable, commutator

from pulsee.nuclear_spin import NuclearSpin, ManySpins
        

def test_spin_quantum_number_initialisation_with_non_half_integer():
    wrong_input = 2.6
    try:
        I = NuclearSpin(wrong_input)
        raise AssertionError
    except ValueError as ve:
        if not "The given spin quantum number is not a half-integer number" in ve.args[0]:
            raise AssertionError("No ValueError caused by the initialisation of the spin quantum number with a non-half-integer number")
    except AssertionError:
        raise AssertionError("No ValueError caused by the initialisation of the spin quantum number with a non-half-integer number")
        
@given(s = st.integers(min_value=1, max_value=14))
def test_spin_raising_lowering_operators_are_hermitian_conjugate(s):
    n_s = NuclearSpin(s/2)
    raising_dagger = n_s.I['+'].dag()
    lowering = n_s.I['-']
    note("Adjoint of I_raising = %r" % (raising_dagger.full()))
    note("I_lowering = %r" % (lowering.full()))
    assert np.all(np.isclose(raising_dagger.full(), lowering.full(), rtol=1e-10))

# Checks the well-known commutation relation
# [I_x, I_y] = i I_z
# between the cartesian components of the spin
@given(s = st.integers(min_value=1, max_value=14))
def test_spin_commutation_relation(s):
    n_s = NuclearSpin(s/2)
    left_hand_side = commutator(n_s.I['x'], n_s.I['y'])
    right_hand_side = 1j*n_s.I['z']
    note("[I_x, I_y] = %r" % (left_hand_side.full()))
    note("i I_z = %r" % (right_hand_side.full()))
    assert np.all(np.isclose(left_hand_side.full(), right_hand_side.full(), rtol=1e-10))

@given(dim = st.lists(st.integers(min_value=2, max_value=4), min_size=3, max_size=3))
def test_dimensions_many_spin_operator(dim):
    spin = []
    for i in range(3):
        spin.append(NuclearSpin((dim[i]-1)/2))
        
    many_spins = ManySpins([spin[0], spin[1], spin[2]])
    note("spin[0].I[-] = %r" % (spin[0].I['-'].full()))
    note("spin[1].I[-] = %r" % (spin[1].I['-'].full()))
    note("spin[2].I[-] = %r" % (spin[2].I['-'].full()))
    note("ManySpins.I[-] = %r" % (many_spins.I['-'].full()))
    assert many_spins.d == many_spins.I['-'].shape[0]

# Checks the well-known result of basic quantum mechanics according to which the total angular momentum
# of a system of two spins with quantum number s = 3/2, 5/2 respectively has the possible eigenvalues
# 1, 2, 3, 4
def test_angular_momentum_sum_rules():
    spin1 = NuclearSpin(3/2)
    spin2 = NuclearSpin(5/2)
    
    spin_system = ManySpins([spin1, spin2])
    
    I_sq_mod = spin_system.I['x']**2 + spin_system.I['y']**2 + spin_system.I['z']**2
    
    eig = I_sq_mod.eigenstates()[0]
    eig = np.sort(np.real(eig))
    
    # The eigenvalues of the square modulus of the total angular momentum are calculated as
    # I(I+1)
    # where I is the quantum number of the total angular momentum
    expected_eig = np.array([2, 2, 2, \
                             6, 6, 6, 6, 6, \
                             12, 12, 12, 12, 12, 12, 12, \
                             20, 20, 20, 20, 20, 20, 20, 20, 20])
    
    assert np.all(np.isclose(eig, expected_eig, rtol=1e-10))
    
