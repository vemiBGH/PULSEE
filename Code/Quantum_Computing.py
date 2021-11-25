from Operators import Observable, Operator
import numpy as np 


def adjoint(a):
	"""
	Convenience method for quickly taking conjugate transpose of array.

	Params
	------
	- `a`: an array
	
	Returns
	------
	- the conjugate transpose of a.
	"""

	return np.array(np.matrix(a).getH())


# Define quantum gates
class N_Gate(Operator):
	"""
	Quantum n-gate as defined in Scherer pg. 169: a unitary operator U: H^n -> 
	H^n. If n = 1, one has a unary gate; if n = 2, one has a binary gate. 
	n specified by the given qubit space. 

	Params
	------
	- `x`: the array representing this operator. 
	- `qs`: the qubit space on which this operator/gate acts. 
	"""
	def __init__(self, x, qs):
		self._qs = qs 
		self._n = qs.n
		x = np.array(x)
		if not np.shape(x) == (2 ** self._n, 2 ** self._n):
			raise ValueError(f'Input array shape {np.shape(x)} invalid for ' \
				+ f'qubit space H^{self._n}')
		if np.array_equal(np.matmul(adjoint(x), x), np.identity(self._n)):
			raise ValueError('Input array must be unitary.')

		super().__init__(x)
	
	def apply(self, state):
		assert self._n == state.n
		return np.matmul(self.matrix, state.matrix)


class Qubit_State:
	def __init__(self, qs, matrix):
		"""
		Params
		------
		- `qs`: the qubit space of which this qubit state is an element.
		- `matrix`: an 1 by n matrix, where n is the number of factors of the 
		            (possibly composite) qubit space of which this qubit state 
					is an element.
		"""
		assert np.log2(np.shape(matrix)[0]) == qs.n
		self._matrix = matrix
		self._qs = qs 

	@property 
	def matrix(self): 
		return self._matrix 


class Composite_Qubit_Space:
	"""
	Implementation of an n-fold tensor product of qubit spaces (as defined in 
	Scherer 85) 
	"""
	def __init__(self, n): 
		"""
		Params
		------
		- n: number of qubit spaces of which this is a composition. 
		"""
		self._n = n 
	
	def basis_from_indices(self, indices):
		assert np.all([i <= 1 for i in indices]) and len(indices) == self._n
		basis = np.zeros((2 ** self._n))
		k = 1
		i = indices[0]
		for ind in indices[1:]: 
			i += (ind) * 2 ** k 
			k += 1 
		basis[i] = 1
		return basis

	@property
	def n(self): 
		return self._n


class Qubit_Space(Composite_Qubit_Space):
	"""
	Implementation of a qubit space as specified by Scherer : a two-dimensional 
	Hilbert space equipped with an eigenbasis |0> and |1> and a corresponding 
	observable A such that A|0> = 1 |0> and A|1> = - 1 |1>. 
	"""
	def __init__(self):
		super().__init__(1)
		# Define base qubits 
		self._base_zero = super().basis_from_indices([0])
		self._base_one = super().basis_from_indices([1])


	# Defined observable for this qubit
	_observable = Observable(np.array([[1, 0], [0, -1]]))

	def make_state(self, alpha=None, beta=None, coeffs=None):
		if alpha is not None and beta is not None: 
			matrix = np.cos(beta / 2) * self._base_zero + np.sin(beta / 2) \
					* np.exp(1j * alpha) * self._base_one 
			return Qubit_State(self, matrix)

		elif coeffs is not None: 
			assert np.conjugate(coeffs[0]) * coeffs[0] + \
				   np.conjugate(coeffs[1]) * coeffs[1] == 1 and len(coeffs) == 2
			matrix = coeffs[0] * self._base_zero + coeffs[1] * self._base_one
			return Qubit_State(self, matrix)

		else: 
			raise ValueError("State vector must be created using either " \
						   + "coefficients or polar and azimuthal angles.")


# Defined quantum gates.
HADAMARD = N_Gate((1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]]), Qubit_Space())

	
	