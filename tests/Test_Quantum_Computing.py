import sys
import numpy as np
sys.path.insert(1, '../Code')

from pulsee.exceptions.quantum_computing import MatrixRepresentationError
from pulsee.quantum_computing import CompositeQubitSpace, QubitSpace 

import hypothesis.strategies as st
from hypothesis import given, settings, note, assume


@given(a = st.integers(min_value=0, max_value=1), 
	   b = st.integers(min_value=0, max_value=1))
@settings(deadline = None)
def test_composite_qubit_space_basis_matrix(a, b):
	"""
	Test of qubit composition matrix as defined by Scherer. 
	"""
	qs = CompositeQubitSpace(2)
	basis_matrix = qs.basis_from_indices([a, b])
	expected_matrix = np.zeros(2 ** 2)
	expected_matrix[a * 2 + b] = 1
	assert expected_matrix.tolist() == basis_matrix.tolist()


@given(indices = st.lists(st.integers()))
def test_n_fold_composite_basis_matrix(indices):
	n = len(indices)
	assume(n > 0)
	qs = CompositeQubitSpace(n)
	try:
		qs.basis_from_indices(indices)
	except MatrixRepresentationError: 
		assume(False)

	
@given(alpha = st.floats(), beta = st.floats())
def test_make_qubit_from_angles(alpha, beta):
	"""
	NOTE: it known that certain values raise numpy warnings; it is assumed that 
	the input float size is reasonable such that they will not cause
	under/overflow errors.  
	"""
	qs = QubitSpace() 
	qs.make_state(alpha=alpha, beta=beta)


@given(coeffs = st.lists(st.integers(min_value=-1, max_value=2), min_size=0, max_size=3))
def test_make_qubit_from_coeffs(coeffs):
	qs = QubitSpace() 
	caught = False 
	try:
		qs.make_state(coeffs=coeffs)
		caught = True 
	except MatrixRepresentationError: 
		caught = True 
	
	assert caught 


def test_make_qubit_invalid_args():
	qs = QubitSpace() 
	
	caught = False 

	try:
		qs.make_state(alpha=1)
	except MatrixRepresentationError:
		caught = True 

	try:
		qs.make_state(beta=1)
	except MatrixRepresentationError:
		caught = caught and True 

	try:
		qs.make_state()
	except MatrixRepresentationError:
		caught = caught and True 

	# Ensure the below don't throw exceptions.
	qs.make_state(coeffs=[1, 2], alpha=1)
	qs.make_state(coeffs=[1, 3], beta=1)

	assert caught 