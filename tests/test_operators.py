import math
from numpy import log
import numpy as np
from scipy import linalg
from scipy.linalg import eig, LinAlgError
from scipy.integrate import quad
from scipy.constants import Planck, Boltzmann

from qutip import Qobj, expect

import hypothesis.strategies as st
from hypothesis import given, settings, note, assume

from pulsee.operators import random_operator, random_density_matrix, random_observable, \
                            commutator, magnus_expansion_1st_term, \
                            magnus_expansion_2nd_term, \
                            magnus_expansion_3rd_term, \
                            canonical_density_matrix, changed_picture, free_evolution,\
                            positivity, unit_trace

@given(d = st.integers(min_value=1, max_value=16))
@settings(deadline = None)
def test_opposite_operator(d):
    o = random_operator(d)
    note("o = %r" % (o.full()))
    assert np.all(np.isclose((o-o).full(), np.zeros((d, d)), rtol=1e-10))

@given(d = st.integers(min_value=1, max_value=16))
@settings(deadline = None)
def test_associativity_sum_operators(d):
    a = random_operator(d)
    b = random_operator(d)
    c = random_operator(d)
    left_sum = (a+b)+c
    right_sum = a+(b+c)
    note("a = %r" % (a.full()))
    note("b = %r" % (b.full()))
    note("c = %r" % (c.full()))
    note("(a+b)+c = %r" % (left_sum.full()))
    note("a+(b+c) = %r" % (right_sum.full()))
    assert np.all(np.isclose(left_sum.full(), right_sum.full(), rtol=1e-10))

@given(d = st.integers(min_value=1, max_value=16))
@settings(deadline = None)
def test_associativity_product_operators(d):
    a = random_operator(d)
    b = random_operator(d)
    c = random_operator(d)
    left_product = (a*b)*c
    right_product = a*(b*c)
    note("a = %r" % (a.full()))
    note("b = %r" % (b.full()))
    note("c = %r" % (c.full()))
    note("(a*b)*c = %r" % (left_product.full()))
    note("a*(b*c) = %r" % (right_product.full()))
    assert np.all(np.isclose(left_product.full(), right_product.full(), rtol=1e-10))

@given(d = st.integers(min_value=1, max_value=16))
@settings(deadline = None)
def test_distributivity_operators(d):
    a = random_operator(d)
    b = random_operator(d)
    c = random_operator(d)
    left_hand_side = a*(b+c)
    right_hand_side = a*b+a*c
    note("a = %r" % (a.full()))
    note("b = %r" % (b.full()))
    note("c = %r" % (c.full()))
    note("a*(b+c) = %r" % (left_hand_side.full()))
    note("a*b+a*c = %r" % (right_hand_side.full()))
    assert np.all(np.isclose(left_hand_side.full(), right_hand_side.full(), rtol=1e-10))
    
@given(d = st.integers(min_value=1, max_value=16))
@settings(deadline = None)
def test_operator_trace_normalisation(d):
    o = random_operator(d)
    o_trace = o.tr()
    o_norm = o/o_trace
    o_norm_trace = o_norm.tr()
    note("o = %r" % (o.full()))
    note("Trace of o = %r" % (o_trace))
    note("Trace-normalised o = %r" % (o_norm))
    note("Trace of trace-normalised o = %r" % (o_norm_trace))
    assert np.all(np.isclose(o_norm_trace, 1, rtol=1e-10))

# Checks the fact that the eigenvalues of the exponential of an Operator o are the exponentials of
# o's eigenvalues
@given(d = st.integers(min_value=1, max_value=4))
@settings(deadline = None)
def test_exponential_operator_eigenvalues(d):
    o = random_operator(d)
    o_e = o.eigenstates()[0]
    exp_e = o.expm().eigenstates()[0]
    sorted_exp_o_e = np.sort(np.exp(o_e))
    sorted_exp_e = np.sort(exp_e)
    note("o = %r" % (o.full()))
    note("exp(o) = %r" % (o.expm().full()))
    note("Eigenvalues of o = %r" % (np.sort(o_e)))
    note("Exponential of the eigenvalues of o = %r" % (sorted_exp_o_e))
    note("Eigenvalues of exp(o) = %r" % (sorted_exp_e))
    assert np.all(np.isclose(sorted_exp_o_e, sorted_exp_e, rtol=1e-10))
    
@given(d = st.integers(min_value=1, max_value=16))
@settings(deadline = None)
def test_observable_real_eigenvalues(d):
    o = random_observable(d)
    eig = o.eigenstates()[0]
    note("Eigenvalues of o = %r" % (eig))
    assert np.all(np.absolute(np.imag(eig)) < 1e-10)

# Checks that the adjoint of an Operator o's exponential is the exponential of the adjoint of o
@given(d = st.integers(min_value=1, max_value=16))
@settings(deadline = None)
def test_adjoint_exponential(d):
    o = random_operator(d)
    o_exp = o.expm()
    left_hand_side = (o_exp.dag()).full()
    right_hand_side = ((o.dag()).expm()).full()
    note("(exp(o))+ = %r" % (left_hand_side))
    note("exp(o+) = %r" % (right_hand_side))    
    assert np.all(np.isclose(left_hand_side, right_hand_side, rtol=1e-10))
    
@given(d = st.integers(min_value=2, max_value=4))
@settings(deadline = None)
def test_reversibility_change_picture(d):
    o = random_operator(d)
    h = random_operator(d)
    o_ip = changed_picture(o, h, 1, invert=False)
    o1 = changed_picture(o_ip, h, 1, invert=True)
    note("o = %r" % (o.full()))
    note("o in the changed picture = %r" % (o_ip.full()))
    note("o brought back from the changed picture = %r" % (o1.full()))
    assert np.all(np.isclose(o.full(), o1.full(), rtol=1))

@given(d = st.integers(min_value=2, max_value=8))
@settings(deadline = None)
def test_free_evolution_conserves_dm_properties(d):
    dm = random_density_matrix(d)
    h = random_observable(d)
    try:
        evolved_dm = free_evolution(dm, h, 4)
    except ValueError as ve:
        if "The input array lacks the following properties: \n" in ve.args[0]:
            error_message = ve.args[0][49:]
            error_message = "The evolved DensityMatrix lacks the following properties: \n" + error_message
            note("Initial DensityMatrix = %r" % (dm.full()))
            note("Hamiltonian = %r" % (h.full()))
            raise AssertionError(error_message)

@given(d = st.integers(min_value=1, max_value=16))
@settings(deadline = None)
def test_random_observable_is_hermitian(d):
    try:
        ob_random = random_observable(d)
    except ValueError as ve:
        if "The input array is not hermitian" in ve.args[0]:
            # note("Operator returned by random_observable = %r" % (ob_random.full()))
            raise AssertionError("random_observable fails in the creation of hermitian matrices")

@given(d = st.integers(min_value=1, max_value=16))
@settings(deadline = None)
def test_random_density_matrix_satisfies_dm_properties(d):
    try:
        dm_random = random_density_matrix(d)
    except ValueError as ve:
        if "The input array lacks the following properties: \n" in ve.args[0]:
            error_message = ve.args[0][49:]
            error_message = "The generated random DensityMatrix lacks the following properties: \n" + error_message
            raise AssertionError(error_message)

# Checks that the space of density matrices is a convex space, i.e. that the linear combination
# a*dm1 + b*dm2
# where dm1, dm2 are density matrices, a and b real numbers, is a density matrix if
# a, b in [0, 1] and a + b = 1
@given(d = st.integers(min_value=2, max_value=16))
@settings(deadline = None)
def test_convexity_density_matrix_space(d):
    dm1 = random_density_matrix(d)
    dm2 = random_density_matrix(d)
    a = np.random.random()
    b = 1-a
    hyp_dm = a*dm1 + b*dm2
    assert positivity(hyp_dm) and unit_trace(hyp_dm)

@given(d = st.integers(min_value=2, max_value=16))
@settings(deadline = None)
def test_linearity_evolution(d):
    dm1 = random_density_matrix(d)
    dm2 = random_density_matrix(d)
    h = random_observable(d)
    dm_sum = 0.5*(dm1+dm2)
    evolved_dm_sum = free_evolution(dm_sum, h, 5)
    evolved_dm1 = free_evolution(dm1, h, 5)
    evolved_dm2 = free_evolution(dm2, h, 5)
    left_hand_side = evolved_dm_sum.full()
    right_hand_side = (0.5 * (evolved_dm1 + evolved_dm2)).full()
    note("dm1 = %r" % (dm1.full()))
    note("dm2 = %r" % (dm2.full()))
    note("Evolved dm1+dm2 = %r" % (left_hand_side))
    note("Evolved dm1 + evolved dm2 = %r" % (right_hand_side))
    assert np.all(np.isclose(left_hand_side, right_hand_side, rtol=1e-10))

    
# Checks the well-known relation
# <(O-<O>)^2> = <O^2> - <O>^2
# where O is an observable, and the angular brackets indicate the expectation value over some state
@given(d = st.integers(min_value=2, max_value=16))
@settings(deadline = None)
def test_variance_formula(d):
    ob = random_observable(d)
    i = Qobj(np.eye(d))
    dm = random_density_matrix(d)
    ob_ev = expect(ob, dm)
    sq_dev = (ob - ob_ev*i) ** 2
    left_hand_side = expect(sq_dev, dm)
    right_hand_side = expect(ob ** 2, dm)-ob_ev**2
    assert np.all(np.isclose(left_hand_side, right_hand_side, 1e-10))

def observable_function(x):
    matrix = np.array([[x, 1 + 1j * x ** 2],[1 - 1j * x ** 2, x ** 3]])
    o = Qobj(matrix)
    return o


# NOTE: Deprecated tests - Magnus expansion no longer used in evolution scheme
#       as of QuTiP integration.
# def test_antihermitianity_magnus_1st_term():
#     times, time_step = np.linspace(0, 20, num=2001, retstep=True)
#     t_dep_hamiltonian = observable_function
#     sampled_hamiltonian = t_dep_hamiltonian(times)
#     magnus_1st = magnus_expansion_1st_term(sampled_hamiltonian, time_step)
#     magnus_1st_dagger = magnus_1st.dag()
#     assert np.all(np.isclose(magnus_1st_dagger.full(), - magnus_1st.full(), 1e-10))

# def test_antihermitianity_magnus_2nd_term():
#     times, time_step = np.linspace(0, 5, num=501, retstep=True)
#     t_dep_hamiltonian = np.vectorize(observable_function)
#     sampled_hamiltonian = t_dep_hamiltonian(times)
#     magnus_2nd = magnus_expansion_2nd_term(sampled_hamiltonian, time_step)
#     magnus_2nd_dagger = magnus_2nd.dag()
#     assert np.all(np.isclose(magnus_2nd_dagger.full(), -magnus_2nd.full(), 1e-10))
    
# def test_antihermitianity_magnus_3rd_term():
#     times, time_step = np.linspace(0, 1, num=101, retstep=True)
#     t_dep_hamiltonian = np.vectorize(observable_function)
#     sampled_hamiltonian = t_dep_hamiltonian(times)
#     magnus_3rd = magnus_expansion_3rd_term(sampled_hamiltonian, time_step)
#     magnus_3rd_dagger = magnus_3rd.dag()
#     assert np.all(np.isclose(magnus_3rd_dagger.full(), -magnus_3rd.full(), 1e-10))
    
# Checks that the canonical density matrix computed with the function canonical_density_matrix reduces
# to (1 - h*H_0/(k_B*T))/Z when the temperature T gets very large
@given(d = st.integers(min_value=1, max_value=16))
@settings(deadline = None)
def test_canonical_density_matrix_large_temperature_approximation(d):
    h0 = random_observable(d)
    can_dm = canonical_density_matrix(h0, 300)
    exp = -(Planck*h0*1e6)/(Boltzmann*300)
    num = exp.expm()
    can_partition_function = num.tr()   
    can_dm_apx = (Qobj(np.eye(d))+exp)/can_partition_function
    assert np.all(np.isclose(can_dm.full(), can_dm_apx.full(), rtol=1e-10))







