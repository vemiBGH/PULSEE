from pulsee.quantum_computing import * 

# Create qubit space object. 
qs = QubitSpace()

# Create a qubit from this qubit space
q = qs.basis_ket_from_indices([0])

# The QubitState class implements a `density_matrix` property which can be 
# used to study that state's density matrix. 
print("Density Matrix: ", q.density_matrix.matrix, "\n")

# Apply Pauli-X gate
print("Pauli-X applied to q: ", pauli_x(q).matrix, "\n")

# Let's study the effect that certain gates have on the density matrix of a state. 
# The Hadamard gate creates a superposition out of a basis qubit; we thus expect 
# the density matrix to have off-diagonal elements:

print("Hadamard applied to q: ", hadamard(q).density_matrix.matrix, "\n")


# Experiment with composite qubit spaces.
cqs = CompositeQubitSpace(2) # choose n = 2 for the composition of two spaces
							 # (i.e., two particles).

# Create composite qubit. 
q = cqs.basis_ket_from_indices([1, 0])
print("Example composite qubit matrix: ", q.matrix, "\n")

# Create state |00> and act on it with CNOT gate
control = qs.basis_ket_from_indices([0])
target = qs.basis_ket_from_indices([0])
product_qubit = control * target # pulsee overrides multiplication operation for 
								 # QubitState objects.

product_qubit.matrix

# we see that applying the CNOT gate has no effect because the control is |0>:
print("CNOT applied to |00>: ", cnot(product_qubit).matrix, "\n")

# However, if we instead use the state |10>, we expect to obtain |11>:
product_qubit = qs.basis_ket_from_indices([1]) * qs.basis_ket_from_indices([0])
print("CNOT applied to |11>: ", cnot(product_qubit).matrix, "\n")


# Investigate entanglement. Create superposition of |0> and |1> and use as 
# control qubit. 
control = qs.make_state(coeffs=[1 / np.sqrt(2), 1 / np.sqrt(2)])
target = qs.basis_ket_from_indices([0])

product_qubit = control * target 
density_matrix = cnot(product_qubit).density_matrix.matrix

# take real component to interpret more easily
print("Density matrix of entangled qubits: ", np.real(density_matrix))


