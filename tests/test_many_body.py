import numpy as np

import hypothesis.strategies as st
from hypothesis import given, settings, note

from qutip import tensor, rand_dm

from pulsee import random_operator, random_observable

from pulsee import ptrace_subspace

@given(d = st.integers(min_value=2, max_value=8))
@settings(deadline = None)
def test_tensor_product_conserves_density_matrix_properties(d):
    A = rand_dm(d)
    B = rand_dm(d)
    
    try:
        C = tensor(A, B)
    except ValueError as ve:
        if "The input array lacks the following properties: \n" in ve.args[0]:
            error_message = ve.args[0][49:]
            error_message = "The tensor product of two DensityMatrix objects lacks the following properties: \n" + error_message
            note("A = %r" % (A.full()))
            note("B = %r" % (B.full()))
            raise AssertionError(error_message)

            
@given(d = st.integers(min_value=3, max_value=6))
@settings(deadline = None)
def test_ptrace_subspace_is_inverse_tensor_product(d):
    A = random_operator(d - 1)
    A = A / A.tr()
    B = random_operator(d)
    B = B / B.tr()
    C = random_operator(d+1)
    C = C / C.tr()
    
    AB = tensor(A, B)
    BC = tensor(B, C)
    ABC = tensor(AB, C)
    
    p_t = ptrace_subspace(ABC, [d-1, d, d + 1], 0)
    
    assert np.all(np.isclose(p_t.full(), BC.full(), rtol=1e-10))

