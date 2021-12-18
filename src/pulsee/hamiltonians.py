import math
from cmath import exp
import numpy as np
import pandas as pd

from .operators import Operator, DensityMatrix, Observable

from .many_body import tensor_product

from .nuclear_spin import NuclearSpin, ManySpins

def h_zeeman(spin, theta_z, phi_z, B_0):
    """
    Computes the term of the Hamiltonian associated with the Zeeman interaction
    between the nuclear spin and the external static field.
    
    Parameters
    ----------
    - spin: NuclearSpin
            Spin under study;
    - theta_z: float
               Polar angle of the magnetic field in the laboratory coordinate
               system (expressed in radians);
    - phi_z: float
             Azimuthal angle of the magnetic field in the laboratory coordinate
             system (expressed in radians);
    - B_0: non-negative float
           Magnitude of the external magnetic field (expressed in tesla).
    
    Returns
    -------
    An Observable object which represents the Zeeman Hamiltonian in the
    laboratory reference frame (expressed in MHz).
    
    Raises
    ------
    ValueError, when the passed B_0 is a negative number.
    """
    if B_0 < 0: raise ValueError("The modulus of the magnetic field must " \
        + "be a non-negative quantity")
    h_z = - spin.gyro_ratio_over_2pi * B_0 *  \
          (math.sin(theta_z) * math.cos(phi_z) * spin.I['x'] + \
           math.sin(theta_z) * math.sin(phi_z) * spin.I['y'] + \
           math.cos(theta_z) * spin.I['z'])
    return Observable(h_z.matrix)

def h_quadrupole(spin, e2qQ, eta, alpha_q, beta_q, gamma_q):
    """
    Computes the term of the Hamiltonian associated with the quadrupolar
    interaction.  
    
    Parameters
    ----------
    - spin: NuclearSpin
            Spin under study;
    - e2qQ: float
            Product of the quadrupole moment constant, eQ, and the eigenvalue of
            the EFG tensor which is greatest in absolute value, eq. e2qQ is
            measured in MHz;
    - eta: float in the interval [0, 1]
           Asymmetry parameter of the EFG;
    - alpha_q, beta_q, gamma_q: float
                                Euler angles for the conversion from the system
                                of the principal axes of the EFG tensor (PAS) to
                                the lab system (LAB) (expressed in radians).
                                
                                
    Returns
    -------
    If the quantum number of the spin is 1/2, the whole calculation is skipped
    and a null Observable object is returned.  Otherwise, the function returns
    the Observable object which correctly represents the quadrupolar Hamiltonian
    in the laboratory reference frame (expressed in MHz).

    """
    if math.isclose(spin.quantum_number, 1/2, rel_tol=1e-10):
        return Observable(spin.d) * 0
    I = spin.quantum_number
    h_q = (e2qQ/(I * (2 * I - 1))) *  \
          ((1 / 2) * (3 * (spin.I['z'] ** 2) - Operator(spin.d) * I * (I+1)) * v0_EFG(eta, alpha_q, beta_q, gamma_q)+\
           (math.sqrt(6)/4) * 
           ((spin.I['z'] * spin.I['+'] + spin.I['+'] * spin.I['z']) * \
                             v1_EFG(-1, eta, alpha_q, beta_q, gamma_q) + \
            (spin.I['z'] * spin.I['-'] + spin.I['-'] * spin.I['z']) * \
                             v1_EFG(+ 1, eta, alpha_q, beta_q, gamma_q) + \
            (spin.I['+'] ** 2) * \
            v2_EFG(-2, eta, alpha_q, beta_q, gamma_q) + \
            (spin.I['-'] ** 2) * \
            v2_EFG(2, eta, alpha_q, beta_q, gamma_q)))
    return Observable(h_q.matrix)

def v0_EFG(eta, alpha_q, beta_q, gamma_q):
    """
    Returns the component V0 of the EFG tensor (divided by eq) as seen in the
    LAB system. This quantity is expressed in terms of the Euler angles which
    relate PAS and LAB systems and the parameter eta.
    
    Parameters
    ----------
    - eta: float in the interval [0, 1]
           Asymmetry parameter of the EFG;
    - alpha_q, beta_q, gamma_q: float
                                Euler angles connecting the system of the
                                principal axes of the EFG tensor (PAS) to the
                                lab system (LAB) (expressed in radians).
                                
    Returns
    -------
    A float representing the component V0 (divided by eq) of the EFG tensor
    evaluated in the LAB system.
    
    Raises
    ValueError, when the passed eta is not in the interval [0, 1].
    """
    if eta < 0 or eta > 1: raise ValueError("The asymmetry parameter must fall in " 
        + "the interval [0, 1]")
    v0 = (1 / 2) * (((3 * (math.cos(beta_q)) ** 2 - 1) / 2) - (eta * 
            (math.sin(beta_q)) ** 2) * (math.cos(2 * gamma_q))/2)
    return v0

def v1_EFG(sign, eta, alpha_q, beta_q, gamma_q):
    """
    Returns the components V+/-1 of the EFG tensor (divided by eq) as seen in
    the LAB system. These quantities are expressed in terms of the Euler angles
    which relate PAS and LAB systems and the parameter eta.
    
    Parameters
    ----------
    - sign: float
            Specifies wether the V+1 or the V-1 component is to be computed;
    - eta: float in the interval [0, 1]
           Asymmetry parameter of the EFG;
    - alpha_q, beta_q, gamma_q: float
                                Euler angles connecting the system of the
                                principal axes of the EFG tensor (PAS) to the
                                lab system (LAB) (expressed in radians).
                                
    Returns
    -------
    A complex number representing the component:
    - V<sup>+1</sup>, if sign is positive;
    - V<sup>-1</sup>, if sign is negative.
    of the EFG tensor (divided by eq).
    
    Raises
    ------
    ValueError, when the passed eta is not in the interval [0, 1].
    """
    if eta < 0 or eta > 1: raise ValueError("The asymmetry parameter must fall " \
        + "within the interval [0, 1]")
    sign = np.sign(sign)
    v1 = (1 / 2) * \
         (
          -1j * sign * math.sqrt(3/8) * math.sin(2 * beta_q) * exp(sign * 1j * alpha_q)+\
            1j * (eta/(math.sqrt(6))) * math.sin(beta_q) * \
          (
           ((1 + sign * math.cos(beta_q)) / 2) * exp(1j * (sign * alpha_q + 2 * gamma_q))-\
            ((1 - sign * math.cos(beta_q)) / 2) * exp(1j * (sign * alpha_q - 2 * gamma_q))
          ) 
         )
    return v1

def v2_EFG(sign, eta, alpha_q, beta_q, gamma_q):
    """
    Returns the components V+/-2 of the EFG tensor (divided by eq) as seen in
    the LAB system. These quantities are expressed in terms of the Euler angles
    which relate PAS and LAB systems and the parameter eta.
    
    Parameters
    ----------
    - sign: float
            Specifies wether the V+2 or the V-2 component is to be returned;
    - eta: float in the interval [0, 1]
           Asymmetry parameter of the EFG tensor;
    - alpha_q, beta_q, gamma_q: float
                                Euler angles connecting the system of the
                                principal axes of the EFG tensor (PAS) to the
                                lab system (LAB) (expressed in radians).
                                
    Returns
    -------
    A float representing the component:
    - V+2, if sign is positive;
    - V-2, if sign is negative.
    of the EFG tensor (divided by eq).
    
    Raises
    ------
    ValueError, when the passed eta is not in the interval [0, 1].
    """
    if eta<0 or eta>1: raise ValueError("The asymmetry parameter must fall in" \
        + " the interval [0, 1]")
    sign = np.sign(sign)
    v2 = (1 / 2) * \
         (math.sqrt(3/8) * ((math.sin(beta_q)) ** 2) * exp(sign * 2j * alpha_q) + \
          (eta/math.sqrt(6)) * exp(sign * 2j * alpha_q) * \
           (
            exp(2j * gamma_q) * ((1+sign * math.cos(beta_q)) ** 2) / 4 + \
            exp(-2j * gamma_q) * ((1-sign * math.cos(beta_q)) ** 2) / 4
           )
         )
    return v2

def h_single_mode_pulse(spin, frequency, B_1, phase, theta_1, phi_1, t):
    """
    Computes the term of the Hamiltonian describing the interaction with a
    monochromatic and linearly polarized electromagnetic pulse.
    
    Parameters
    ----------
    - spin: NuclearSpin
            Spin under study.
    - frequency: non-negative float
                 Frequency of the monochromatic wave (expressed in MHz).
    - phase: float
             Inital phase of the wave (at t=0) (expressed in radians).
    - B_1: non-negative float
           Maximum amplitude of the oscillating magnetic field (expressed in tesla).
    - theta_1, phi_1: float
                      Polar and azimuthal angles of the direction of
                      polarization of the magnetic wave in the LAB frame
                      (expressed in radians); 
    - t: float
         Time of evaluation of the Hamiltonian (expressed in microseconds).
    
    Returns
    -------
    An Observable object which represents the Hamiltonian of the coupling with
    the electromagnetic pulse evaluated at time t (expressed in MHz).
    
    Raises
    ------
    ValueError, in two distinct cases:
    1. When the passed frequency parameter is a negative quantity;
    2. When the passed B_1 parameter is a negative quantity.
    """
    if frequency < 0: raise ValueError("The modulus of the angular frequency" \
        + " of the electromagnetic wave must be a positive quantity")
    if B_1 < 0: raise ValueError("The amplitude of the electromagnetic" \
        + " wave must be a positive quantity")
    h_pulse = -spin.gyro_ratio_over_2pi * B_1 * \
              (math.sin(theta_1) * math.cos(phi_1) * spin.I['x'] +\
               math.sin(theta_1) * math.sin(phi_1) * spin.I['y'] +\
               math.cos(theta_1) * spin.I['z']
               ) * \
               math.cos(2 * math.pi * frequency * t-phase)
    return Observable(h_pulse.matrix)

def h_multiple_mode_pulse(spin, mode, t):
    """
    Computes the term of the Hamiltonian describing the interaction with a
    superposition of single-mode electromagnetic pulses. If the passed argument
    spin is a NuclearSpin object, the returned Hamiltonian will describe the
    interaction between the pulse of radiation and the single spin; if it is a
    ManySpins object, it will represent the interaction with the whole system
    of many spins.
    
    Parameters
    ----------
    - spin: NuclearSpin or ManySpins
            Spin or spin system under study;
    - mode: pandas.DataFrame
            Table of the parameters of each electromagnetic mode in the
            superposition. It is organised according to the following template:
            
    | index |  'frequency'  |  'amplitude'  |  'phase'  |  'theta_p'  |  'phi_p'  |
    | ----- | ------------- | ------------- | --------- | ----------- | --------- |
    |       |     (MHz)     |      (T)      |   (rad)   |    (rad)    |   (rad)   |
    |   0   |    omega_0    |      B_0      |  phase_0  |   theta_0   |   phi_0   |
    |   1   |    omega_1    |      B_1      |  phase_1  |   theta_1   |   phi_1   |
    |  ...  |      ...      |      ...      |    ...    |     ...     |    ...    |
    |   N   |    omega_N    |      B_N      |  phase_N  |   theta_N   |   phi_N   |
  
    where the meaning of each column is analogous to the corresponding
    parameters in h_single_mode_pulse.
    
    - t: float
         Time of evaluation of the Hamiltonian (expressed in microseconds).
         
    Returns
    -------
    An Observable object which represents the Hamiltonian of the coupling with
    the superposition of the given modes evaluated at time t (expressed in MHz).
    """
    h_pulse = Operator(spin.d) * 0
    omega = mode['frequency']
    B = mode['amplitude']
    phase = mode['phase']
    theta = mode['theta_p']
    phi = mode['phi_p']
    if isinstance(spin, ManySpins):
        for i in mode.index:
            h_pulse = Operator(spin.d) * 0
            for n in range(spin.n_spins):
                term_n = h_single_mode_pulse(spin.spin[n], omega[i], B[i],
                                             phase[i], theta[i], phi[i], t)
                for m in range(spin.n_spins)[:n]:
                    term_n = tensor_product(Operator(spin.spin[m].d), term_n)
                for l in range(spin.n_spins)[n+1:]:
                    term_n = tensor_product(term_n, Operator(spin.spin[l].d))
                h_pulse = h_pulse + term_n
    elif isinstance(spin, NuclearSpin):
        for i in mode.index:
            h_pulse = h_pulse + h_single_mode_pulse(spin, omega[i], B[i], 
                                                phase[i], theta[i], phi[i], t)
    return Observable(h_pulse.matrix)

# Global Hamiltonian of the system (stationary term + pulse term) cast in the picture generated by
# the Operator h_change_of_picture
def h_changed_picture(spin, mode, h_unperturbed, h_change_of_picture, t):
    """
    Returns the global Hamiltonian of the system, made up of the time-dependent
    term h_multiple_mode_pulse(spin, mode, t) and the stationary term
    h_unperturbed, cast in the picture generated by h_change_of_picture.
    
    Parameters
    ----------
    - spin, mode, t: same meaning as the corresponding arguments of 
                     h_multiple_mode_pulse;
    - h_unperturbed: Operator
                     Stationary term of the global Hamiltonian (in MHz);
    - h_change_of_picture: Operator
                           Operator which generates the new picture (in MHz).
                           
    Returns
    -------
    Observable object representing the Hamiltonian of the pulse evaluated at
    time t in the new picture (in MHz).  
    """
    h_cp = (h_unperturbed + h_multiple_mode_pulse(spin, mode, t) - \
            h_change_of_picture).changed_picture(h_change_of_picture, t)
    return Observable(h_cp.matrix)

def h_j_coupling(spins, j_matrix):
    """
    Returns the term of the Hamiltonian describing the J-coupling between the
    spins of a system of many nuclei.  
    
    Parameters
    ----------
    - spins: ManySpins
             Spins' system under study;
             
    - j_matrix: np.ndarray
                Array storing the coefficients Jmn which enter the formula for
                the computation of the Hamiltonian for the j-coupling.  Remark:
                j_matrix doesn't have to be symmetric, since the function reads
                only those elements located in the upper half with respect to
                the diagonal. This means that the elements j_matrix[m, n] which
                matter are those for which m<n.
                
    Returns
    -------
    Observable object acting on the full Hilbert space of the spins' system
    representing the Hamiltonian of the J-coupling between the spins.  
    """
    h_j = Operator(spins.d) * 0
    
    for m in range(j_matrix.shape[0]):
        for n in range(m):            
            term_nm = j_matrix[n, m] * spins.spin[n].I['z']
            for l in range(n):
                term_nm = tensor_product(Operator(spins.spin[l].d), term_nm)
            for k in range(m)[n + 1:]:
                term_nm = tensor_product(term_nm, Operator(spins.spin[k].d))
            term_nm = tensor_product(term_nm, spins.spin[m].I['z'])
            for j in range(spins.n_spins)[m + 1:]:
                term_nm = tensor_product(term_nm, Operator(spins.spin[j].d))
                            
            h_j = h_j + term_nm
            
    return h_j.cast_to_observable()
        


