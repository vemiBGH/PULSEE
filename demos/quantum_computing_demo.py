# %% [markdown]
# # PULSEE Quantum Computing Demo
# ### by Lucas Brito for PHYS2050
# This is a demo of the quantum computing module of PULSEE. If the `pulsee` 
# package has been installed in this environment, we can import the quantum
# computing module with 

# %%
from pulsee.Quantum_Computing import * 

# %% [markdown]
# We can create a `QubitSpace` object; this is an implementation of the 
# two-dimensional Hilbert space which contains qubits. 

# %%
qs = QubitSpace()

# %% [markdown]
# From this `QubitSpace` we can generate a basis ket out of the bits which it 
# represents. For example, if we want the ket $|0 \rangle$:

# %%
q = qs.basis_ket_from_indices([0])

# %% [markdown]
# This object is of the type `QubitState`:

# %%
type(q)

# %% [markdown]
# The `QubitState` class implements a `density_matrix` property which can be 
# used to study that state's density matrix. 

# %%
q.density_matrix.matrix
# or 	
q.get_density_matrix().matrix

# %% [markdown]
# Let us study the effect of applying particular gates to this qubit. As a sanity 
# check, we can apply the Pauli-X gate (`Quantum_Computing.pauli_x`) to see that, 
# in this simulation, it indeed functions as quantum analogue of the classical NOT
# gate (i.e., it transforms $|0\rangle \doteq [1 \; 0 ]$ to $|1\rangle \doteq [ 0 \; 1]$): 

# %%
pauli_x(q).matrix

# %% [markdown]
# Let's study the effect that certain gates have on the density matrix of a state. 
# The Hadamard gate creates a superposition out of a basis qubit; we thus expect 
# the density matrix to have off-diagonal elements:

# %%
hadamard(q).density_matrix.matrix


# %% [markdown]
# Indeed we obtain the matrix 
# $$
# \begin{bmatrix}
# 1/2 & 1/2\\ 1/2 & 1/2
# \end{bmatrix}
# $$
# as is to be expected from the ket $(1/\sqrt{2})(|0\rangle + |1 \rangle)$. We see 
# that if our observable of choice is spin, the Hadamard gate has the effect of 
# rotating the Bloch sphere vector by $\pi/2$ about the $y$-axis.
# 
# Now create a composite qubit space (the tensor product of two qubit spaces, used 
# treat, for example, the total spin of two particles) in order to study some more
# complicated gates. 

# %%
cqs = CompositeQubitSpace(2) # choose n = 2 for the composition of two spaces
							 # (i.e., two particles).

# %% [markdown]
# We can create the state ket $|10\rangle$: one spin-up particle and one spin-down 
# particle. 

# %%
q = cqs.basis_ket_from_indices([1, 0])
q.matrix

# %% [markdown]
# A commonly used binary gate is the controlled-NOT (CNOT) gate:
# $$
# \text{CNOT}
# =
# \begin{bmatrix}
# 1 & 0 & 0 & 0 \\
# 0 & 1 & 0 & 0 \\
# 0 & 0 & 0 & 1 \\
# 0 & 0 & 1 & 0 
# \end{bmatrix}
# $$
# which is known to entangle a "target" qubit and a "controller" qubit; see
# Barnett (2009) pg. 247.  The action of the CNOT gate on the target qubit depends
# on the state of the controller qubit: If the control is $|0\rangle$ the target
# is unchanged; if the control is $|1\rangle$, the CNOT gate acts on the target as
# if it is a NOT (Pauli-X) gate. Let us see this in action. Construct the states

# %%

control = qs.basis_ket_from_indices([0])
target = qs.basis_ket_from_indices([0])

# %% [markdown]
# and take the tensor product to create a composite state:

# %%
product_qubit = tensor_product(control, target)
# or 
product_qubit = control * target # pulsee overrides multiplication operation for 
								 # QubitState objects.

product_qubit.matrix

# %% [markdown]
# we see that applying the CNOT gate has no effect because the control is
# $|0\rangle$: 

# %%
cnot(product_qubit).matrix

# %% [markdown]
# However, if we instead use the state $|10\rangle$, we expect to obtain
# $|11\rangle$:

# %%
product_qubit = qs.basis_ket_from_indices([1]) * qs.basis_ket_from_indices([0])
cnot(product_qubit).matrix

# %% [markdown]
# It is not difficult to see how the CNOT gate creates entangled states. Loosely 
# speaking, whether the target qubit was "flipped" depends on the state of the 
# controller qubit. Thus providing a superposed controller qubit and
# subsequently collapsing said qubit leads to a collapse of the target 
# qubit; the value of this collapsed target qubit depends on observed value of the 
# controller qubit.
# 
# Then if we construct the states $(1/\sqrt{2})(|0\rangle + |0 \rangle)$ and 
# $|0\rangle $ and take their product:

# %%
control = qs.make_state(coeffs=[1 / np.sqrt(2), 1 / np.sqrt(2)])
target = qs.basis_ket_from_indices([0])

product_qubit = control * target 

# %% [markdown]
# then apply the CNOT gate and inspect the resulting density matrix:

# %%
density_matrix = cnot(product_qubit).density_matrix.matrix
# take real component to interpret more easily
np.real(density_matrix)

# %% [markdown]
# The diagonal elements of this matrix are as expected—we have nonzero entries 
# $1/2$ in the top left and bottom right, indicating that this is an entangled 
# stat; see Baaquie (2013) pages 93-11.
# 


