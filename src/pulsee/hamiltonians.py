import numpy as np
from tqdm import trange
from qutip import Qobj, tensor
from qutip.solver import Result

from .nuclear_spin import NuclearSpin, ManySpins
from .operators import changed_picture, commutator


def h_zeeman(spin, theta_z, phi_z, B_0):
    """
    Computes the term of the Hamiltonian associated with the Zeeman interaction between the nuclear spin and the
     external static field.

    Parameters
    ----------
    - spin: NuclearSpin
            Spin under study;
    - theta_z: float
               Polar angle of the magnetic field in the laboratory coordinate system (expressed in radians);
    - phi_z: float
             Azimuthal angle of the magnetic field in the laboratory coordinate system (expressed in radians);
    - B_0: non-negative float
           Magnitude of the external magnetic field (expressed in tesla).

    Returns
    -------
    An Observable object which represents the Zeeman Hamiltonian in the laboratory reference frame
    (expressed in rad/sec).
    Raises
    ------

    ValueError, when the passed B_0 is a negative number.
    """
    if B_0 < 0: raise ValueError("The modulus of the magnetic field must be a non-negative quantity")

    h_z = -spin.gyro_ratio_over_2pi * B_0 * \
          (np.sin(theta_z) * np.cos(phi_z) * spin.I['x'] +
           np.sin(theta_z) * np.sin(phi_z) * spin.I['y'] +
           np.cos(theta_z) * spin.I['z'])
    return Qobj(h_z)


def v0_EFG(eta, alpha_q, beta_q, gamma_q):
    """
    Returns the component V0 of the EFG tensor (divided by eq) as seen in the LAB system. This quantity is expressed
    in terms of the Euler angles which relate PAS and LAB systems and the parameter eta.

    Parameters
    ----------
    - eta: float in the interval [0, 1]
           Asymmetry parameter of the EFG;
    - alpha_q, beta_q, gamma_q: float
                                Euler angles connecting the system of the principal axes of the EFG tensor (PAS) to
                                the lab system (LAB) (expressed in radians).
                                alpha_q not used.

    Returns
    -------
    A float representing the component V0 (divided by eq) of the EFG tensor evaluated in the LAB system.

    Raises
    ValueError, when the passed eta is not in the interval [0, 1].
    """
    if eta < 0 or eta > 1:
        raise ValueError(
            "The asymmetry parameter must fall in the interval [0, 1]")
    v0 = (1 / 2) * (((3 * (np.cos(beta_q)) ** 2 - 1) / 2) -
                    (eta * (np.sin(beta_q)) ** 2) * (np.cos(2 * gamma_q)) / 2)
    return v0


def h_quadrupole(spin, e2qQ, eta, alpha_q, beta_q, gamma_q, component_order=0):
    """
    Computes the term of the Hamiltonian associated with the quadrupolar interaction.

    Parameters
    ----------
    - spin: NuclearSpin
            Spin under study;
    - e2qQ: float
            Product of the quadrupole moment constant, eQ, and the eigenvalue of the EFG tensor
            which is greatest in absolute value, eq. e2qQ is measured in MHz;
    - eta: float in the interval [0, 1]
           Asymmetry parameter of the EFG;
    - alpha_q, beta_q, gamma_q: float
                                Euler angles for the conversion from the system of the principal
                                axes of the EFG tensor (PAS) to the lab system (LAB) (expressed in radians).
    - component_order = 0: Int
            Order of the quadrupolar interaction in the LAB frame.
    Returns
    -------
    If the quantum number of the spin is 1/2, the whole calculation is skipped and a null Observable object is returned.
    Otherwise, the function returns the Observable object which correctly represents the quadrupolar Hamiltonian in the
    laboratory reference frame (expressed in MHz).

    """
    if np.isclose(spin.quantum_number, 1 / 2, rtol=1e-10):
        return Qobj(spin.d) * 0
    I = spin.quantum_number
    h_q = (e2qQ / (I * (2 * I - 1))) * \
          ((1 / 2) * (3 * (spin.I['z'] ** 2) - Qobj(np.eye(spin.d)) * I * (I + 1)) * v0_EFG(eta, alpha_q, beta_q,
                                                                                            gamma_q))
    if component_order > 0:
        h_q += (e2qQ / (I * (2 * I - 1))) * ((np.sqrt(6) / 4) *
                                             ((spin.I['z'] * spin.I['+'] + spin.I['+'] * spin.I['z']) *
                                              v1_EFG(-1, eta, alpha_q, beta_q, gamma_q) +
                                              (spin.I['z'] * spin.I['-'] + spin.I['-'] * spin.I['z']) *
                                              v1_EFG(+1, eta, alpha_q, beta_q, gamma_q)))
    if component_order > 1:
        h_q += (e2qQ / (I * (2 * I - 1))) * ((spin.I['+'] ** 2) *
                                             v2_EFG(-2, eta, alpha_q, beta_q, gamma_q) +
                                             (spin.I['-'] ** 2) *
                                             v2_EFG(2, eta, alpha_q, beta_q, gamma_q))
    return Qobj(h_q)


def v1_EFG(sign, eta, alpha_q, beta_q, gamma_q):
    """
    Returns the components V+/-1 of the EFG tensor (divided by eq) as seen in the LAB system.
    These quantities are expressed in terms of the Euler angles which relate PAS and LAB systems and the parameter eta.

    Parameters
    ----------
    - sign: float
            Specifies wether the V+1 or the V-1 component is to be computed;
    - eta: float in the interval [0, 1]
           Asymmetry parameter of the EFG;
    - alpha_q, beta_q, gamma_q: float
                                Euler angles connecting the system of the principal axes of the EFG tensor (PAS)
                                to the lab system (LAB) (expressed in radians).

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
    if eta < 0 or eta > 1:
        raise ValueError(
            "The asymmetry parameter must fall within the interval [0, 1]")
    sign = np.sign(sign)
    v1 = (1 / 2) * (- 1j * sign * np.sqrt(3 / 8) * np.sin(2 * beta_q) * np.exp(sign * 1j * alpha_q) +
                    1j * (eta / (np.sqrt(6))) * np.sin(beta_q) *
                    (((1 + sign * np.cos(beta_q)) / 2) * np.exp(1j * (sign * alpha_q + 2 * gamma_q)) -
                     ((1 - sign * np.cos(beta_q)) / 2) * np.exp(1j * (sign * alpha_q - 2 * gamma_q))))
    return v1


def v2_EFG(sign, eta, alpha_q, beta_q, gamma_q):
    """
    Returns the components V+/-2 of the EFG tensor (divided by eq) as seen in the LAB system. These quantities are
    expressed in terms of the Euler angles which relate PAS and LAB systems and the parameter eta.

    Parameters
    ----------
    - sign: float
            Specifies wether the V+2 or the V-2 component is to be returned;
    - eta: float in the interval [0, 1]
           Asymmetry parameter of the EFG tensor;
    - alpha_q, beta_q, gamma_q: float
                                Euler angles connecting the system of the principal axes of the EFG tensor (PAS) to
                                the lab system (LAB) (expressed in radians).

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
    if eta < 0 or eta > 1:
        raise ValueError(
            "The asymmetry parameter must fall in the interval [0, 1]")
    sign = np.sign(sign)
    v2 = (1 / 2) * \
         (np.sqrt(3 / 8) * ((np.sin(beta_q)) ** 2) * np.exp(sign * 2j * alpha_q) +
          (eta / np.sqrt(6)) * np.exp(sign * 2j * alpha_q) *
          (
                  np.exp(2j * gamma_q) * ((1 + sign * np.cos(beta_q)) ** 2) / 4 +
                  np.exp(-2j * gamma_q) * ((1 - sign * np.cos(beta_q)) ** 2) / 4
          )
          )
    return v2


def pulse_time_dep_coeff(frequency, phase):
    """
    Return the time-dependent coefficient of a pulse Hamiltonian. 

    Parameters
    ----------
    - frequency: non-negative float
                 Frequency of the monochromatic wave (expressed in rad/sec).
    - phase: float
             Inital phase of the wave (at t=0) (expressed in radians).

    Returns
    -------
    Function with signature f(t: float, args: iterable) -> float
    """

    def time_dependence_function(t, args):
        return np.cos(frequency * t - phase)

    return time_dependence_function


def pulse_t_independent_op(spin, B_1, theta_1, phi_1):
    """
    Computes the time-independent portion of the Hamiltonian interaction with a
    monochromatic and linearly polarized electromagnetic pulse.

    Parameters
    ----------
    - spin: NuclearSpin
            Spin under study.
    - phase: float
             Inital phase of the wave (at t=0) (expressed in radians).
    - B_1: non-negative float
           Maximum amplitude of the oscillating magnetic field (expressed in tesla).
    - theta_1, phi_1: float
                      Polar and azimuthal angles of the direction of polarization of
                      the magnetic wave in the LAB frame (expressed in radians);
    """
    return - 2 * spin.gyro_ratio_over_2pi * B_1 \
        * (np.sin(theta_1) * np.cos(phi_1) * spin.I['x']
           + np.sin(theta_1) * np.sin(phi_1) * spin.I['y']
           + np.cos(theta_1) * spin.I['z'])


def h_single_mode_pulse(spin, frequency, B_1, phase, theta_1, phi_1, t,
                        factor_t_dependence=False):
    """
    Computes the term of the Hamiltonian describing the interaction with a monochromatic
    and linearly polarized electromagnetic pulse.

    Parameters
    ----------
    - spin: NuclearSpin
            Spin under study.
    - frequency: non-negative float
                 Frequency of the monochromatic wave (expressed in rad/sec).
    - phase: float
             Inital phase of the wave (at t=0) (expressed in radians).
    - B_1: non-negative float
           Maximum amplitude of the oscillating magnetic field (expressed in tesla).
    - theta_1, phi_1: float
                      Polar and azimuthal angles of the direction of polarization of the magnetic
                      wave in the LAB frame (expressed in radians);
    - t: float
         Time of evaluation of the Hamiltonian (expressed in microseconds).
    - factor_t_dependence: bool
                           If true, return tuple (H, f(t)) where f(t) is the 
                           time-dependence of the Hamiltonian as a function.
                           Does not evaluate f(t) at the given time.

    Returns
    -------
    An Observable object which represents the Hamiltonian of the coupling with the electromagnetic pulse
    evaluated at time t (expressed in rad/sec).

    Raises
    ------
    ValueError
    2. When the passed B_1 parameter is a negative quantity.
    """
    # if frequency < 0: raise ValueError("The modulus of the angular frequency of the electromagnetic wave
    # must be a positive quantity")
    if B_1 < 0:
        raise ValueError(
            "The amplitude of the electromagnetic wave must be a positive quantity")

    # Notice the following does not depend on spin
    t_dependence = pulse_time_dep_coeff(
        frequency, phase)  # this variable is a function!
    h_t_independent = pulse_t_independent_op(spin, B_1, theta_1, phi_1)
    if factor_t_dependence:
        return Qobj(h_t_independent), t_dependence
    else:
        # Need pass empty list because of QuTiP compatability necessary
        # arguments of t_dependence; `args` argument functionless
        return Qobj((t_dependence(t, []) * h_t_independent))


def h_multiple_mode_pulse(spin, mode, t, factor_t_dependence=False):
    """
    Computes the term of the Hamiltonian describing the interaction with a superposition of single-mode electromagnetic
    pulses. If the passed argument spin is a NuclearSpin object, the returned Hamiltonian will describe the interaction
    between the pulse of radiation and the single spin; if it is a ManySpins object, it will represent the interaction
     with the whole system of many spins.

    Parameters
    ----------
    - spin: NuclearSpin or ManySpins
            Spin or spin system under study;
    - mode: pandas.DataFrame
            Table of the parameters of each electromagnetic mode in the superposition. It is organised according to the following template:
    - factor_t_dependence: bool
                           If true, return tuple (H, f(t)) where f(t) is the 
                           time-dependence of the Hamiltonian as a function.
                           Does not evaluate f(t) at the given time.

    | index |  'frequency'  |  'amplitude'  |  'phase'  |  'theta_p'  |  'phi_p'  |
    | ----- | ------------- | ------------- | --------- | ----------- | --------- |
    |       |   (rad/sec)   |      (T)      |   (rad)   |    (rad)    |   (rad)   |
    |   0   |    omega_0    |      B_0      |  phase_0  |   theta_0   |   phi_0   |
    |   1   |    omega_1    |      B_1      |  phase_1  |   theta_1   |   phi_1   |
    |  ...  |      ...      |      ...      |    ...    |     ...     |    ...    |
    |   N   |    omega_N    |      B_N      |  phase_N  |   theta_N   |   phi_N   |

    where the meaning of each column is analogous to the corresponding parameters in h_single_mode_pulse.

    - t: float
         Time of evaluation of the Hamiltonian (expressed in microseconds).

    Returns
    -------
    An Observable object which represents the Hamiltonian of the coupling with
    the superposition of the given modes evaluated at time t (expressed in rad/sec).
    OR 
    A list of tuples of the form (H_m, f_m(t)) for each mode m. 
    """

    omega = mode['frequency']
    B = mode['amplitude']
    phase = mode['phase']
    theta = mode['theta_p']
    phi = mode['phi_p']
    if factor_t_dependence:
        # Create list of Hamiltonians with unique time dependencies
        mode_hamiltonians = []
        if isinstance(spin, ManySpins):
            for i in mode.index:
                t_dependence = pulse_time_dep_coeff(omega[i], phase[i])
                # dimensions of vector inputs to tensor; should be same as dual vector
                # inputs, i.e., tensor valence/rank should be (r, k) with r = k. equiv.
                # to matrix being square.
                dims = [s.d for s in spin.spin]
                h_t_independent = Qobj(np.zeros((spin.d, spin.d)), dims=[dims, dims])

                # Construct tensor product of operators acting on each spin.
                # Take a tensor product where every operator except the nth
                # is the identity, add those together
                for n in range(spin.n_spins):
                    term_n = pulse_t_independent_op(spin.spin[n], B[i],
                                                    theta[i], phi[i] + angle)
                    for m in range(spin.n_spins)[:n]:
                        term_n = tensor(Qobj(np.eye(spin.spin[m].d)), term_n)
                    for l in range(spin.n_spins)[n + 1:]:
                        term_n = tensor(term_n, Qobj(np.eye(spin.spin[l].d)))
                    h_t_independent += term_n

                # Append total hamiltonian for this mode to mode_hamiltonians
                mode_hamiltonians.append([Qobj(h_t_independent), t_dependence])
        elif isinstance(spin, NuclearSpin):
            for i in mode.index:
                # Ix term
                mode_hamiltonians.append(h_single_mode_pulse(spin, omega[i], B[i], phase[i], theta[i], phi[i],
                                                             t, factor_t_dependence=True))
                # for a simple pulse in the transverse plane: [(-gamma/2pi * B1 * Ix, 'time_dependence_function'
                # (which returns cos(w0*t)))]

        return mode_hamiltonians
    else:
        h_pulse = Qobj(np.zeros((spin.d, spin.d)))
        if isinstance(spin, ManySpins):
            for i in mode.index:
                # dimensions of vector inputs to tensor; should be same as dual vector
                # inputs, i.e., tensor valence/rank should be (r, k) with r = k. equiv.
                # to matrix being square.
                dims = [s.d for s in spin.spin]
                h_pulse = Qobj(np.zeros((spin.d, spin.d)), dims=[dims, dims])

                # Construct tensor product of operators acting on each spin.
                # Take a tensor product where every operator except the nth
                # is the identity, add those together
                for n in range(spin.n_spins):
                    term_n = h_single_mode_pulse(spin.spin[n], omega[i], B[i],
                                                 phase[i], theta[i], phi[i], t, factor_t_dependence=False)
                    for m in range(spin.n_spins)[:n]:
                        term_n = tensor(Qobj(np.eye(spin.spin[m].d)), term_n)
                    for l in range(spin.n_spins)[n + 1:]:
                        term_n = tensor(term_n, Qobj(np.eye(spin.spin[l].d)))
                    h_pulse += term_n
        elif isinstance(spin, NuclearSpin):
            for i in mode.index:
                h_pulse += h_single_mode_pulse(spin, omega[i], B[i], phase[i], theta[i], phi[i], t)

        return Qobj(h_pulse)


# Global Hamiltonian of the system (stationary term + pulse term) cast in the picture generated by
# the Operator h_change_of_picture
def h_changed_picture(spin, mode, h_unperturbed, h_change_of_picture, t):
    """
    Returns the global Hamiltonian of the system, made up of the time-dependent term h_multiple_mode_pulse(spin, mode, t)
    and the stationary term h_unperturbed, cast in the picture generated by h_change_of_picture.

    Parameters
    ----------
    - spin, mode, t: same meaning as the corresponding arguments of h_multiple_mode_pulse;
    - h_unperturbed: Operator
                     Stationary term of the global Hamiltonian (in MHz);
    - h_change_of_picture: Operator
                           Operator which generates the new picture (in MHz).

    Returns
    -------
    Observable object representing the Hamiltonian of the pulse evaluated at time t in the new picture (in MHz).
    # """
    h_pulse = h_multiple_mode_pulse(spin, mode, t)
    h_cp = changed_picture((h_unperturbed - h_change_of_picture + h_pulse),
                           h_change_of_picture, t)
    return Qobj(h_cp)


def h_changed_picture_func(spin, mode, h_unperturbed, h_change_of_picture, t):
    """
    Returns the global Hamiltonian of the system, made up of the time-dependent term h_multiple_mode_pulse(spin, mode, t)
     and the stationary term h_unperturbed, cast in the picture generated by h_change_of_picture.

    Parameters
    ----------
    - spin, mode, t: same meaning as the corresponding arguments of h_multiple_mode_pulse;
    - h_unperturbed: Operator
                     Stationary term of the global Hamiltonian (in MHz);
    - h_change_of_picture: Operator
                           Operator which generates the new picture (in MHz).

    Returns
    -------
    Observable object representing the Hamiltonian of the pulse evaluated at time t in the new picture (in MHz).
    # """

    def func(t, args):
        h_pulse = h_multiple_mode_pulse(spin, mode, t)
        h_cp = changed_picture((h_unperturbed + h_pulse - h_change_of_picture),
                               h_change_of_picture, t)
        return Qobj(h_cp)

    return func


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
    # dimensions of vector inputs to tensor; should be same as dual vector
    # inputs, i.e., tensor valence/rank should be (r, k) with r = k. equiv.
    # to matrix being square.
    dims = [s.d for s in spins.spin]
    h_j = Qobj(np.zeros((spins.d, spins.d)), dims=[dims, dims])

    # row
    for m in range(j_matrix.shape[0]):
        # column
        for n in range(m):
            term_nm = j_matrix[n, m] * spins.spin[n].I['z']
            for l in range(n):
                term_nm = tensor(Qobj(np.eye(spins.spin[l].d)), term_nm)
            for k in range(m)[n + 1:]:
                term_nm = tensor(term_nm, Qobj(np.eye(spins.spin[k].d)))
            term_nm = tensor(term_nm, spins.spin[m].I['z'])
            for j in range(spins.n_spins)[m + 1:]:
                term_nm = tensor(term_nm, Qobj(np.eye(spins.spin[j].d)))

            h_j = h_j + term_nm

    return h_j


def h_CS_isotropic(spin, delta_iso, B_0):
    """
    Computes the term of the Hamiltonian associated with the chemical shift
    interaction in the secular approximation for isotropic liquids between the
    nuclear spin and the external static field.

    Parameters
    ----------
    - spin: NuclearSpin
            Spin under study;
    - delta_iso: float
                   Magnitude of the chemical shift in Hz - H_CS = -delta_iso\omgega_0 Iz 
    - B_0: non-negative float
           Magnitude of the external magnetic field (expressed in tesla).

    Returns
    -------
    An Observable object which represents the Zeeman Hamiltonian in the laboratory reference
    frame (expressed in MHz).

    Raises
    ------
    ValueError, when the passed B_0 is a negative number.
    """
    if B_0 < 0:
        raise ValueError(
            "The modulus of the magnetic field must be a non-negative quantity")
    h_cs = -delta_iso * spin.gyro_ratio_over_2pi * B_0 \
           * spin.I['z']
    return Qobj(h_cs)


def h_D1(spins, b_D, theta):
    """
    Computes the term of the Hamiltonian associated with the dipolar interaction in the secular
    approximation for homonuclear & heteronuclear spins.
    {H}_{D1} \approx  b_D \frac{3\cos^2\theta-1}{2})[3I_{1z}I_{2z} -\mathbf{I}_1\cdot  \mathbf{I}_2].

    Parameters
    ----------
    - spin: ManySpins
            2 Spins in the system under study;
    - b_D: float
           Magnitude of dipolar constant, b_D\equiv \frac{\mu_0\gamma_1\gamma_2}{4\pi r^3_{21}}.
    - theta: float
           Polar angle between the two spins (expressed in radians).

    Returns
    -------
   Observable object acting on the full Hilbert space of the 2-spin system representing the Hamiltonian.

    """
    h_d1 = b_D * (1 / 2) * (3 * (np.cos(theta) ** 2) - 1) * \
           (2 * tensor(spins.spin[0].I['z'], spins.spin[1].I['z']) -
            tensor(spins.spin[0].I['x'], spins.spin[1].I['x']) -
            tensor(spins.spin[0].I['y'], spins.spin[1].I['y']))
    return Qobj(h_d1)


def h_D2(spins, b_D, theta):
    """
    Computes the term of the Hamiltonian associated with the dipolar interaction in the secular
    approximation for heteronuclear spins. H_{D2} \approx \hslash b_D (3\cos^2\theta-1)I_{1z}I_{2z}.

    Parameters
    ----------
    - spin: ManySpins
            2 Spins in the system under study;
    - b_D: float
           Magnitude of dipolar constant, b_D\equiv \frac{\mu_0\gamma_1\gamma_2}{4\pi r^3_{21}}.
    - theta: float
           Polar angle between the two spins (expressed in radians).

    Returns
    -------
   Observable object acting on the full Hilbert space of the 2-spin system representing the Hamiltonian.

    """
    h_d2 = b_D * (3 * (np.cos(theta) ** 2) - 1) * (
        tensor(spins.spin[0].I['z'], spins.spin[1].I['z']))
    return Qobj(h_d2)


def h_HF_secular(spins, A, B):
    """
    Computes the term of the Hamiltonian associated with the hyperfine interaction in the secular
     approximation for between two spins. H_{D2} \approx A S_{z}I_{z} + B S_{z}I_{x}  .

    Parameters
    ----------
    - spin: ManySpins
            2 Spins in the system under study;
    - A: float
         Constant, see paper.
    - B: float
         Constant, see paper.

    Returns
    -------
   Observable object acting on the full Hilbert space of the 2-spin system representing the Hamiltonian.

    """
    h_hf = A * tensor(spins.spin[0].I['z'], spins.spin[1].I['z']) + \
           B * tensor(spins.spin[0].I['z'], spins.spin[1].I['x'])
    return Qobj(h_hf)


def h_j_secular(spins, J):
    """
    Computes the term of the Hamiltonian associated with the J-coupling in the secular approximation  between two spins.

    Parameters
    ----------
    - spin: ManySpins
            2 Spins in the system under study;
    - J: float
         The J-coupling constant in MHz.

    Returns
    -------
   Observable object acting on the full Hilbert space of the 2-spin system representing the Hamiltonian.

    """
    h_j = 2 * np.pi * J * tensor(spins.spin[0].I['z'], spins.spin[1].I['z'])
    return Qobj(h_j)


def h_tensor_coupling(spins, t):
    """
    Yields Hamiltonian representing an interaction of the form I_1 A I_2 where
    I_i are spin operators and A is a rank-2 tensor. Could for example be used 
    to obtain [hyperfine interaction Hamiltonian.]
    (https://epr.ethz.ch/education/basic-concepts-of-epr/int--with-nucl--spins/hyperfine-interaction.html)
    Author: Lucas Brito

    NOTE: This can be used for any coupling that has a tensor interaction, such
    as the full chemical shift, and full dipolar interaction, where the
    appropriate matrix is passed.

    Params
    ------
    - spins: ManySpins
             2-spin system under study 
    - t: a numpy ndarray representing the interaction tensor of this 
              Hamiltonian in MHz.

    Returns
    ------
    Observable object acting on full Hilbert space of the 2-spin system 
    representing the Hamiltonian of this interaction. 
    """

    i_1 = spins.spin[0].cartesian_operator()
    i_2 = spins.spin[1].cartesian_operator()

    # Initialize empty operator of appropriate dimension as base case for
    # for loop.
    dims = [s.d for s in spins.spin]
    h = Qobj(np.zeros((spins.d, spins.d)), dims=[dims, dims])

    for m in range(len(i_1)):
        for n in range(len(i_2)):
            h += t[m, n] * tensor(i_1[m], i_2[n])

    return h


def h_userDefined(matrix):
    """
    Rehashes the numpy array of the user defined hamiltonian as an Operator.

    Important! --> Make sure to give the Hamiltonian in MHz!

    Parameters
    ----------
    - matrix: numpy array
            Square matrix array which will give the hamiltonian of the system, adding to
      previous terms (if any).

    Returns
    -------
   Observable object representing the Hamiltonian.

    """
    return Qobj(matrix)


# TODO: Better way to calcualte Magnus terms...
def magnus(h_total, rho0, tlist, order, spin, mode, o_change_of_picture):
    """
    Magnus expansion solver, up to 3rd order.
    Integration by the trapezoid rule.

    Parameters
    ----------
    - 'h_total' : np.ndarray of Observable
         Time-independent Hamiltonian (expressed in MHz). Technically, an array of Observable
          objects which correspond to the Hamiltonian evaluated at successive instants of time.
          The start and end points of the array are taken as the extremes of integration 0 and t;
    - `rho0`: Qobj
              Initial density matrix
    - `tlist`: Iterable[float]
            List of times at which the system will be solved.
    - `order`: int
                the order number for magnus
    - `spin`: NuclearSpin
            Spin under study.
    - `mode`: pandas.DataFrame
            Table of the parameters of each electromagnetic mode in the pulse.
    - 'o_change_of_picture': Qobj
                Operator which generates the change to the new picture.

    Returns
    -------
    qutip.Result instance with the evolved density matrix.
    """
    if order > 3:
        raise ValueError('Magnus expansion solver does not support order > 3. ' +
                         f'Given order {order}.')

    output = Result()
    output.times = tlist
    output.solver = 'magnus'
    time_step = (tlist[-1] - tlist[0]) / (len(tlist) - 1)
    h = []
    integral = 0
    for t in trange(len(tlist)):
        H = h_changed_picture(
            spin, mode, h_total, o_change_of_picture, tlist[t])
        # Trapezoid Rule
        factor = 1
        if t == 0 or t == len(tlist) - 1:
            factor *= 1
        else:
            factor *= 2

        integral += factor * H * 2j * np.pi * time_step / 2

        if order >= 2:
            h.append(H)
            for t2 in range(t + 1):
                factor = 1

                if t == 0 or t == len(tlist) - 1:
                    factor *= 1
                else:
                    factor *= 2

                if t2 == 0 or t2 == t:
                    factor *= 1
                else:
                    factor *= 2

                integral += factor * (commutator(h[t], h[t2])) * \
                            ((2 * np.pi * time_step) ** 2) * (1 / 2)

            if order >= 3:
                for t3 in range(1, t2 + 1):
                    factor = 1

                    if t == 0 or t == len(tlist) - 1:
                        factor *= 1
                    else:
                        factor *= 2

                    if t2 == 0 or t2 == t:
                        factor *= 1
                    else:
                        factor *= 2

                    if t3 == 0 or t3 == t2:
                        factor *= 1
                    else:
                        factor *= 2

                    integral += \
                        factor * (commutator(h[t], commutator(h[t2], h[t3])) +
                                  commutator(h[t3], commutator(h[t2], h[t]))) * \
                        ((2 * np.pi * time_step) ** 3) * (-1j / 6)

    dm_evolved_new_picture = rho0.transform(- integral.expm())
    output.states = [rho0, dm_evolved_new_picture]
    return output
