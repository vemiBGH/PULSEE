import numpy as np
from pulsee.simulation import FID_signal, fourier_transform_signal, nuclear_system_setup, evolve, plot_fourier_transform
from pulsee.quantum_computing import QubitSpace, QubitState, CompositeQubitSpace, \
                              cnot, hadamard, tensor_product
import pandas as pd 


CORRELATED = True 
# Constants, taken from Candoli thesis (4.1.1), Section 4.1
QUANTUM_NUMBER = 1.5 
GAMMA_2PI = 4.00 
B0 = 9 
B1 = 1e-2*B0
gr = 11.26

USE_QUAD_INTER = True 
COUPLING_CONSTANT = 56.2
ASYMMETRY_PARAMETER = 0
ALPHA_Q = 0
BETA_Q = 0
GAMMA_Q = 0

USE_ZEEM_INTERACTION = False
B_THETA_Z = np.pi / 2 # magnetic field oriented along x axis
B_PHI_Z = 0
B = ...

# J-coupling matrix  no need to worry about this 
J_MATRIX = None 

# Chemical shift parameters 
USE_CS = False
DELTA_ISO = ...

# Dipolar interaction parameters 
USE_D1 = False
B_D1 = ... 
THETA1 = ... 

USE_D2 = False
B_D2 = ... 
THETA2 = ... 

# Hyperfine interaction parameters 
USE_HF = False
A = ... 
B = ... 

# J-coupling secular approximation
USE_J_SEC = False
J = ... 

TEMPERATURE = 1e-4 # default as defined by Davide 
T2 = 100
ACQUISITION_TIME = 10 

# Initialize a QubitSpace and a control and target qubit, both in the pure state 
# |0⟩. 
qs = QubitSpace()
control_qubit = qs.basis_ket_from_indices([0])
target_qubit =qs.basis_ket_from_indices([0])

initial_dm = 'canonical'
if CORRELATED: 
    # Create a correlated state by applying a Hadamard gate to the control qubit 
    # then applying a CNOT gate (see quantum computing module report section 1.3). 
    control_qubit = hadamard(control_qubit)
    composite_state = tensor_product(control_qubit, target_qubit) 
    corr_state = cnot(composite_state) # correlated state 

    initial_dm = corr_state.density_matrix.matrix


# Use Cl (taken from `nuclear_species.txt`)
spin_par = {'quantum number': QUANTUM_NUMBER, 'gamma/2pi': GAMMA_2PI}
quad_par = None 
zeem_par = None 
cs_param = None 
D1_param = None 
D2_param = None 
hf_param = None 
j_sec_param = None

if USE_QUAD_INTER:
    quad_par = {'coupling constant': COUPLING_CONSTANT,
                'asymmetry parameter': ASYMMETRY_PARAMETER,
                'alpha_q': ALPHA_Q,
                'beta_q': BETA_Q,
                'gamma_q': GAMMA_Q}

if USE_ZEEM_INTERACTION: 
    zeem_par = {'theta_z': B_THETA_Z, 'phi_z': B_PHI_Z, 'field magnitude': B}

if USE_CS: 
    cs_param = {'delta_iso': DELTA_ISO}

if USE_D1: 
    D1_param = {'b_d': B_D1, 'theta': THETA1}

if USE_D2: 
    D2_param = {'b_d': B_D2, 'theta': THETA2}

if USE_HF:
    hf_param = {'A': A, 'B': B}

if USE_J_SEC: 
    j_sec_param = {'J': J}

# Set up nuclear system 
spin, h_unperturbed, dm_initial = nuclear_system_setup(spin_par=spin_par,
                                                       quad_par=quad_par, 
                                                       zeem_par=zeem_par, 
                                                       D1_param=D1_param, 
                                                       D2_param=D2_param,
                                                       hf_param=hf_param, 
                                                       j_sec_param=j_sec_param, 
                                                       j_matrix=J_MATRIX, 
                                                       initial_state=initial_dm)

# Evolve 
mode = pd.DataFrame([(gr*B0, 2*B1, 0., np.pi / 2, 0)], 
                    columns=['frequency', 'amplitude', 'phase', 'theta_p', 'phi_p'])
evolved_dm = evolve(spin, h_unperturbed, dm_initial, mode=mode, 
                    pulse_time=1/(4* gr * B1))

# Obtain FID signal
t, fid = FID_signal(spin, h_unperturbed, evolved_dm,
                    acquisition_time=ACQUISITION_TIME, T2=T2)

# Obtain fourier transform of FID signal; i.e., NMR spectrum
f, ft = fourier_transform_signal(t, fid, -0.15, 0.15)
plot_fourier_transform(f, ft)

