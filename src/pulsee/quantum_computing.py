from typing import Iterable

import numpy as np
from qutip import Qobj
from qutip import tensor


class MatrixRepresentationError(Exception):
    pass


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

    return np.array(a).transpose().conjugate()


def normalize(a):
    """
    Normalize an array. 

    Params
    ------
    - `a`: an array 

    Returns
    ------
    A divided by its norm. 
    """
    a = np.array(a)
    norm = np.linalg.norm(a)
    return a / norm


class CompositeQubitSpace:
    """
    Implementation of an n-fold tensor product of qubit spaces (as defined in 
    Scherer 85) 
    """

    def __init__(self, n: int):
        """
        Params
        ------
        - n: number of qubit spaces of which this is a composition. 
        """
        if n <= 0:
            raise ValueError(f'Invalid qubit space composition: {n}.')
        self._n = n

    def basis_ket_from_indices(self, indices: Iterable[int]):
        """
        Generate the QubitState of the basis ket of the desired 
        composition of basis qubits. E.g., if given `[0, 1, 0]` generates the 
        state ket |010⟩ as a QubitState.

        Params
        ------
        - `indices`: iterable of one of {0, 1}. States of qubits of which this 
                     basis is a composition. 

        Returns
        ------
        An ndarray representing the matrix representation of this composite 
        state ket. 
        """
        return QubitState(self, self.basis_from_indices(indices))

    def basis_from_indices(self, indices):
        """
        Generate the matrix representation of the basis ket of the desired 
        composition of basis qubits. E.g., if given `[0, 1, 0]` generates the 
        state ket |010⟩ as a matrix. 

        Params
        ------
        - `indices`: iterable of one of {0, 1}. States of qubits of which this 
                     basis is a composition. 

        Returns
        ------
        An ndarray representing the matrix representation of this composite 
        state ket. 
        """
        # Ensure that no index is greater than 1
        if not np.all([0 <= i <= 1 for i in indices]):
            raise MatrixRepresentationError(
                f'{indices} has one index greater than 1.')
        if not len(indices) == self._n:
            raise MatrixRepresentationError(f'Number of indices in {indices} ' +
                                            f'does not match dimension of composite qubit ' +
                                            f'space: {self._n}.')
        basis = np.zeros((2 ** self._n))
        k = 1
        indices.reverse()
        i = indices[0]
        for ind in indices[1:]:
            i += (ind) * 2 ** k
            k += 1
        basis[i] = 1
        return basis

    def onb_matrices(self):
        """
        Obtain the orthonormal basis of this qubit space as matrix 
        representations.

        Returns
        ------
        A list containing the orthonormal basis as matrix representations. 
        """

        matrices = []

        def add_bit(bit, vec):
            new_vec = vec + [bit]
            if len(new_vec) == self.n:
                matrices.append(self.basis_from_indices(new_vec))
            else:
                add_bit(0, new_vec)
                add_bit(1, new_vec)

        add_bit(0, [])
        add_bit(1, [])
        return matrices

    def onb(self):
        """
        Obtain the orthonormal basis of this qubit space as `QubitState` 
        objects.

        Returns
        ------
        A list containing the orthonormal basis.
        """
        return [QubitState(self, m) for m in self.onb_matrices()]

    def __eq__(self, other):
        if isinstance(other, CompositeQubitSpace):
            return self.n == other.n
        return False

    @property
    def n(self):
        return self._n


def n_gate(op):
    """
    Converts an operator to an NGate

    Params
    ------
    - op: an `Operator` object.

    Returns
    ------
    An `NGate` object     
    """
    if np.shape(op)[0] != np.shape(op)[0] \
            or np.log2(np.shape(op)[0]) % 1 != 0 \
            or np.log2(np.shape(op))[0] % 1 != 0:
        raise MatrixRepresentationError()

    qs = CompositeQubitSpace(int(np.log2(np.shape(op)[0])))
    return NGate(op, qs)


class NGate(Qobj):
    """
    Quantum n-gate as defined in Scherer pg. 169: a unitary operator U: H^n → 
    H^n. If n = 1, one has a unary gate; if n = 2, one has a binary gate. 
    n specified by the given qubit space. 

    Params
    ------
    - `x`: the array representing this operator. 
    - `qubit_space`: the qubit space on which this operator/gate acts. 
    """

    def __init__(self, x, qubit_space: CompositeQubitSpace):
        self._qubit_space = qubit_space
        self._n = qubit_space.n
        if isinstance(x, Qobj):
            x_array = x.full()
        else:
            x_array = np.array(x)
        if not np.shape(x_array) == (2 ** self._n, 2 ** self._n):
            raise MatrixRepresentationError(f'Input array shape {np.shape(x_array)} '
                                            + f'invalid for qubit space H^{self._n}')
        if np.array_equal(np.matmul(adjoint(x_array), x_array), np.identity(self._n)):
            raise MatrixRepresentationError('Input array must be unitary.')

        super().__init__(x)

    def __call__(self, qubit):
        return self.apply(qubit)

    def apply(self, state):
        if self._n != state.n:
            raise MatrixRepresentationError(f'Dimension {self._n} of gate ' +
                                            f'does not match dimension {state.n} of state.')

        return QubitState(self._qubit_space, np.matmul(self.full(), state.matrix))

    @property
    def n(self):
        return self._n


class QubitState(Qobj):
    def __init__(self, qubit_space: CompositeQubitSpace, matrix):
        """
        Params
        ------
        - `qubit_space`: the qubit space of which this qubit state is an element.
        - `matrix`: an 1 by n matrix, where n is the number of factors of the 
                    (possibly composite) qubit space of which this qubit state 
                    is an element.
        """
        if np.shape(matrix)[0] != 2 ** qubit_space.n:
            raise MatrixRepresentationError(f'Improper array shape ' +
                                            f'{np.shape(matrix)[0]} for matrix representation ' +
                                            f'of {2 ** qubit_space.n}-dimensional qubit space.')

        super().__init__(matrix)
        self._matrix = matrix
        self._qubit_space = qubit_space

    @property
    def matrix(self):
        return self._matrix

    def get_density_matrix(self):
        """
        Returns
        ------
        This instance's density matrix as a DensityMatrix object.
        """
        density_matrix = np.outer(self.matrix, self.matrix)
        return Qobj(density_matrix)

    @property
    def density_matrix(self):
        return self.get_density_matrix()

    @property
    def n(self):
        return self._qubit_space.n

    @property
    def qubit_space(self):
        return self._qubit_space

    @property
    def subqubits(self):
        return self._subqubits

    def get_reduced_density_matrix(self, index):
        """
        The reduced density matrix of this qubit as specified by Baaquie (2013)
        page 103. 

        Params
        ------
        - `subqubit_index`: index of the subqubit to "trace out"

        Returns
        ------
        The reduced density matrix of this QubitState.
        """
        if self.qubit_space.n != 2:
            raise MatrixRepresentationError('Reduced density matrix currently ' +
                                            'unsupported for n > 2.')

        density_matrix = self.density_matrix
        reduced_density_matrix = np.zeros((2, 2), dtype="complex_")

        # https://physics.stackexchange.com/questions/179671/how-to-take-partial-trace
        for k in QubitSpace().onb_matrices():
            k_otimes_identity = np.kron(k, np.eye(2, dtype="complex_"))
            if index == 1:
                k_otimes_identity = np.kron(np.eye(2, dtype="complex_"), k)

            reduced_density_matrix += np.matmul(k_otimes_identity,
                                                np.matmul(density_matrix.full(), np.transpose(k_otimes_identity)))

        return Qobj(reduced_density_matrix)

    def __mul__(self, other):
        return tensor_product(self, other)

    def __add__(self, other):
        if self.n != other.n:
            raise MatrixRepresentationError(f'Cannot cast qubit with n = {self.n}' +
                                            f' to qubit with n = {other.n}.')
        return QubitState(self.qubit_space, normalize(self + other))

    def __sub__(self, other):
        if self.n != other.n:
            raise MatrixRepresentationError(f'Cannot cast qubit with n = {self.n}' +
                                            f' to qubit with n = {other.n}.')
        return QubitState(self.qubit_space, normalize(self - other))


class QubitSpace(CompositeQubitSpace):
    """
    Implementation of a qubit space as specified by Scherer: a two-dimensional 
    Hilbert space equipped with an eigenbasis |0⟩ and |1⟩ and a corresponding 
    observable A such that A|0⟩ = 1 |0⟩ and A|1⟩ = - 1 |1⟩.
    """

    def __init__(self):
        super().__init__(1)
        # Define base qubits
        self._base_zero = super().basis_from_indices([0])
        self._base_one = super().basis_from_indices([1])

    # Define observable for this qubit
    _observable = Qobj(np.array([[1, 0], [0, -1]]))

    def make_state(self, azimuthal=None, polar=None, coeffs: Iterable[float] = None):
        """
        Produces a QubitState according to the provided parameters. Accepts 
        either the polar and azimuthal angles of the Bloch sphere representation 
        or the coefficients of the orthonormal basis expansion of the state. 

        If all parameters are given, prioritizes angles; if one of the angles 
        is not given, uses coefficients. Normalizes coefficients. 

        Params
        ------
        - `azimuthal`: the azimuthal angle of the Bloch sphere representation of 
                   this state. 
        - `polar`: the polar angle of the Bloch sphere representation of this 
                   this state. 
        - `coeffs`: the coefficients of the orthonormal basis expansion of this 
                    state. 

        Returns
        ------
        A QubitState. 
        """
        if azimuthal is not None and polar is not None:
            matrix = np.cos(polar / 2) * self._base_zero + np.sin(polar / 2) \
                     * np.exp(1j * azimuthal) * self._base_one
            return QubitState(self, matrix)

        elif coeffs is not None:
            if len(coeffs) != 2:
                raise MatrixRepresentationError

            # Normalize coefficients
            coeffs = normalize(coeffs)

            matrix = coeffs[0] * self._base_zero + coeffs[1] * self._base_one
            return QubitState(self, matrix)

        else:
            raise MatrixRepresentationError("State vector must be created using either "
                                            + "coefficients or polar and azimuthal angles.")


def tensor_product(q1, q2):
    """
    Take the tensor product of two qubit states.

    Params
    ------
    - `q1`: a QubitState
    - `q2`: a QubitState

    Returns
    ------
    A QubitState in the CompositeQubitSpace of dimension n1 * n2 where 
    n1 is the dimension of q1's qubit space and n2 is the dimension of q2's 
    qubit space. 
    """
    # Find number of qubit spaces that this composite qubit's qubit space is
    # comprised of
    n = q1.n + q2.n

    # Make matrix representation of tensor product.
    prod = np.kron(q1, q2)
    return QubitState(CompositeQubitSpace(n), prod)


def gate_tensor_product(g1, g2):
    """
    Take the tensor product of two NGates.     

    Params
    ------
    - `g1`, `g2`: two NGates

    Returns
    ------
    An NGate object representing the tensor product of `g1` and `g2`.     
    """
    n = g1.n + g2.n
    prod = tensor(g1, g2)
    return NGate(prod, CompositeQubitSpace(n))


def gate_tensor_pow(g, pow):
    """
    Take the tensor product of an NGate `pow` times. 

    Params
    ------
    - `q`: an NGate 
    - `pow`: an integer 

    Returns
    -------
    The tensor product of `q` with itself `pow` times. 
    """
    i = 0
    prod = g
    while i < pow - 1:
        prod = gate_tensor_product(g, prod)
        i += 1
    return prod


# Define quantum gates.
identity = NGate(np.eye(2), QubitSpace())
hadamard = NGate((1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]]),
                 QubitSpace())
not_gate = NGate(np.array([[0, 1], [1, 0]]), QubitSpace())
cnot = NGate(np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 1],
                       [0, 0, 1, 0]]),
             CompositeQubitSpace(2))
pauli_x = NGate(np.array([[0, 1], [1, 0]]), QubitSpace())
pauli_y = NGate(np.array([[0, -1j], [1j, 0]]), QubitSpace())
pauli_z = NGate(np.array([[1, 0], [0, -1]]), QubitSpace())
phase = NGate(np.array([[1, 0], [0, 1j]]), QubitSpace())
pi_8 = NGate(np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]]), QubitSpace())
toffoli = NGate(np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0, 0, 1, 0]]), CompositeQubitSpace(3))

cnotnot = NGate([[1, 0, 0, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0, 0]], CompositeQubitSpace(3))
