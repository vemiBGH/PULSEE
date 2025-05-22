from collections.abc import Iterable
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from qutip import Qobj, QobjEvo, commutator, qeye, tensor
from qutip.solver import Result
from tqdm import trange

from .nuclear_spin import ManySpins, NuclearSpin
from .operators import apply_exp_op, changed_picture
from .pulses import Pulses

""" The Main Hamiltonian Building Functions """


def make_h_unperturbed(
    spin_system,
    spin_par,
    quad_par,
    zeem_par,
    cs_param,
    j_matrix,
    D1_param,
    D2_param,
    hf_param,
    h_tensor_inter,
    j_secular,
    h_user,
) -> list[Qobj]:
    """
    Helper for 'nuclear_system_setup' in simulation.py
    """
    h_unperturbed = []
    h_quad = []
    h_zeem = []
    if isinstance(spin_system, ManySpins):
        spins = spin_system.spins
        n_spins = spin_system.n_spins
    else:
        spins = [spin_system]
        n_spins = 1

    for i in range(len(spin_par)):
        if quad_par is None:
            h_quad.append(h_quadrupole(spins[i], 0.0, 0.0, 0.0, 0.0, 0.0))
        else:
            h_quad.append(
                h_quadrupole(
                    spins[i],
                    quad_par[i]["coupling constant"],
                    quad_par[i]["asymmetry parameter"],
                    quad_par[i]["alpha_q"],
                    quad_par[i]["beta_q"],
                    quad_par[i]["gamma_q"],
                    quad_par[i]["order"],
                )
            )

        if zeem_par is None:
            h_zeem.append(h_zeeman(spins[i], 0.0, 0.0, 0.0))
        else:
            h_zeem.append(h_zeeman(spins[i], zeem_par["theta_z"], zeem_par["phi_z"], zeem_par["field magnitude"]))

        if (cs_param is not None) and (cs_param != 0.0):
            h_zeem.append(h_CS_isotropic(spins[i], cs_param["delta_iso"], zeem_par["field magnitude"]))

    for i in range(n_spins):
        h_i = h_quad[i] + h_zeem[i]
        for j in range(i):
            h_i = tensor(qeye(spin_system.spins[j].d), h_i)
        for k in range(n_spins)[i + 1 :]:
            h_i = tensor(h_i, qeye(spin_system.spins[k].d))
        h_unperturbed.append(Qobj(h_i))

    if j_matrix is not None:
        h_j = h_j_coupling(spin_system, j_matrix)
        h_unperturbed.append(Qobj(h_j))

    if D1_param is not None:
        if (D1_param["b_D"] == 0.0) and (D1_param["theta"] == 0.0):
            pass
        else:
            h_d1 = h_D1(spin_system, D1_param["b_D"], D1_param["theta"])
            h_unperturbed.append(Qobj(h_d1))

    if D2_param is not None:
        if (D2_param["b_D"] == 0.0) and (D2_param["theta"] == 0.0):
            pass
        else:
            h_d2 = h_D2(spin_system, D2_param["b_D"], D2_param["theta"])
            h_unperturbed.append(Qobj(h_d2))

    if hf_param is not None:
        if (hf_param["A"] == 0.0) and (hf_param["B"] == 0.0):
            pass
        else:
            h_hf = h_HF_secular(spin_system, hf_param["A"], hf_param["B"])
            h_unperturbed.append(Qobj(h_hf))

    if j_secular is not None:
        h_j = h_j_secular(spin_system, j_secular)
        h_unperturbed.append(Qobj(h_j))

    if h_tensor_inter is not None:
        if not isinstance(h_tensor_inter, list):
            h_unperturbed += [Qobj(h_tensor_coupling(spin_system, h_tensor_inter))]
        else:
            for hyp_ten in h_tensor_inter:
                h_unperturbed += [Qobj(h_tensor_coupling(spin_system, hyp_ten))]

    if h_user is not None:
        h_unperturbed += Qobj(h_user)

    return h_unperturbed


def h_multiple_mode_pulse(
    spin: NuclearSpin | ManySpins, mode: Pulses, t: float = None, factor_t_dependence: bool = True
) -> Qobj | list:
    """
    Computes the term of the Hamiltonian describing the interaction with
    a superposition of single-mode electromagnetic pulses.
    If the passed argument spin is a NuclearSpin object, the returned
    Hamiltonian will describe the interaction between the pulse of radiation
    and the single spin;
    if it is a ManySpins object, it will represent the interaction
    with the whole system of many spins.

    Parameters
    ----------
    spin : NuclearSpin or ManySpins
        Spin or spin system under study;
    mode : Pulses
        Pulses dataclass that contains the following list attributes, with each index c
        orresponding to a different pulse in the sequence. The shape of the pulse applies to
        all pulses in the sequence, not to each individually.

        |'frequency'|'amplitude'| 'phase' |'theta_p'|'phi_p'|'pulse_time'|'shape' |'sigma'|
        |-----------|-----------|---------|---------|-------|------------|--------|-------|
        | (rad/sec) |    (T)    |  (rad)  |  (rad)  | (rad) |   (mus)    |square  | (sec) |
        |  omega_0  |    B_0    | phase_0 | theta_0 | phi_0 |   tau_0    |gaussian| sig_0 |
        |  omega_1  |    B_1    | phase_1 | theta_1 | phi_1 |   tau_1    |        | sig_1 |
        |    ...    |    ...    |   ...   |   ...   |  ...  |    ...     |        |  ...  |
        |  omega_N  |    B_N    | phase_N | theta_N | phi_N |   tau_N    |        | sig_N |


        where the meaning of each column is analogous to the corresponding parameters in h_single_mode_pulse.

    t : float
        Time of evaluation of the Hamiltonian (expressed in microseconds).

    factor_t_dependence : bool
    If true, return tuple (H, f(t)) where f(t) is the
    time-dependence of the Hamiltonian as a function.
    Does not evaluate f(t) at the given time.

    Returns
    -------
    A Qobj which represents the Hamiltonian of the coupling with
    the superposition of the given modes evaluated at time t (expressed in rad/sec).
    OR
    A list of tuples of the form (H_m, f_m(t)) for each mode m.
    """
    dims = spin.dims

    omegas = mode.frequencies
    amplitudes = mode.amplitudes
    phases = mode.phases
    thetas = mode.theta_p
    phis = mode.phi_p
    pulse_times = mode.pulse_times
    sigmas = mode.sigma
    if factor_t_dependence:
        # Create list of Hamiltonians with unique time dependencies
        mode_hamiltonians = []
        if isinstance(spin, ManySpins):
            for i in range(mode.size):
                if mode.shape == "square":
                    t_dependence = cosine_wrapper(omegas[i], phases[i], pulse_times[i])
                elif mode.shape == "gaussian":
                    if sigmas is None:
                        raise ValueError("A valid sigma must be provided for a Gaussian pulse.")
                    sigma = sigmas[i]
                    if sigma <= 0:
                        raise ValueError("A valid sigma must be provided for a Gaussian pulse.")
                    t_dependence = gaussian_wrapper(omegas[i], phases[i], pulse_times[i], sigma)
                else:
                    raise ValueError("Unsupported pulse shape. Use 'square' or 'gaussian'.")
                h_t_independent = Qobj(np.zeros((spin.d, spin.d)), dims=dims)

                # Construct tensor product of operators acting on each spin.
                # Take a tensor product where every operator except the nth
                # is the identity, add those together
                for n in range(spin.n_spins):
                    term = pulse_t_independent_op(spin.spins[n], amplitudes[i], thetas[i], phis[i])
                    ops = []
                    for m in range(spin.n_spins):
                        if m == n:
                            ops.append(term)
                        else:
                            ops.append(qeye(spin.spins[m].d))
                    term_n = tensor(ops)
                    h_t_independent += term_n

                # Append total hamiltonian for this mode to mode_hamiltonians
                mode_hamiltonians.append([Qobj(h_t_independent), t_dependence])

        elif isinstance(spin, NuclearSpin):
            for i in range(mode.size):
                # Ix term
                mode_hamiltonians.append(
                    h_single_mode_pulse(
                        spin,
                        omegas[i],
                        amplitudes[i],
                        phases[i],
                        thetas[i],
                        phis[i],
                        t,
                        pulse_times[i],
                        factor_t_dependence=True,
                        sigma=sigmas[i],
                        pulse_shape=mode.shape,
                    )
                )
                # for a simple pulse in the transverse plane: [(-gamma/2pi * B1 * Ix, 'time_dependence_function'
                # (which returns cos(w0*t)))]

        return mode_hamiltonians
    else:
        h_pulse = Qobj(np.zeros((spin.d, spin.d)), dims=dims)
        if isinstance(spin, ManySpins):
            for i in range(mode.size):
                # Construct tensor product of operators acting on each spin.
                # Take a tensor product where every operator except the nth
                # is the identity, add those together
                for n in range(spin.n_spins):
                    term = h_single_mode_pulse(
                        spin.spins[n],
                        omegas[i],
                        amplitudes[i],
                        phases[i],
                        thetas[i],
                        phis[i],
                        t,
                        pulse_times[i],
                        factor_t_dependence=False,
                        sigma=sigmas[i],
                        pulse_shape=mode.shape,
                    )
                    ops = []
                    for m in range(spin.n_spins):
                        if m == n:
                            ops.append(term)
                        else:
                            ops.append(qeye(spin.spins[m].d))
                    term_n = tensor(ops)
                    h_pulse += term_n
        elif isinstance(spin, NuclearSpin):
            for i in range(mode.size):
                h_pulse += h_single_mode_pulse(
                    spin,
                    omegas[i],
                    amplitudes[i],
                    phases[i],
                    thetas[i],
                    phis[i],
                    t,
                    pulse_times[i],
                    factor_t_dependence=False,
                    sigma=sigmas[i],
                    pulse_shape=mode.shape,
                )
        return Qobj(h_pulse)


def h_single_mode_pulse(
    spin: NuclearSpin,
    frequency: float,
    B_1: float,
    phase: float,
    theta_1: float,
    phi_1: float,
    t: float = None,
    pulse_time: float = None,
    factor_t_dependence: bool = True,
    sigma: float = None,
    pulse_shape: str = "square",
):
    """
    Computes the term of the Hamiltonian describing the interaction with a monochromatic
    and linearly polarized electromagnetic pulse.

    Parameters
    ----------
    spin : NuclearSpin
        Spin under study.
    frequency : non-negative float
        Frequency of the monochromatic wave (expressed in rad/sec).
    phase : float
        Initial phase of the wave (at t=0) (expressed in radians).
    B_1 : non-negative float
        Maximum amplitude of the oscillating magnetic field (expressed in tesla).
    theta_1, phi_1 : float
        Polar and azimuthal angles of the direction of polarization of the magnetic
        wave in the LAB frame (expressed in radians);
    t : float
        Time of evaluation of the Hamiltonian (expressed in microseconds).
    pulse_time : float
        Time duration of the pulse
    factor_t_dependence : bool
        If true, return tuple (H, f(t)) where f(t) is the  time-dependence of
        the Hamiltonian as a function.
        Does not evaluate f(t) at the given time.

    Returns
    -------
    A Qobj which represents the Hamiltonian of the coupling with
    the electromagnetic pulse evaluated at time t (expressed in rad/sec).

    Raises
    ------
    ValueError
    When the passed B_1 parameter is a negative quantity.
    """
    # if frequency < 0: raise ValueError("The modulus of the angular frequency of the electromagnetic wave
    # must be a positive quantity")
    if B_1 < 0:
        raise ValueError("The amplitude of the electromagnetic wave must be positive.")
    # Notice the following does not depend on spin
    if pulse_shape == "square":
        t_dependence = cosine_wrapper(frequency, phase, pulse_time)
    elif pulse_shape == "gaussian":
        if sigma is None or sigma <= 0:
            raise ValueError("A valid sigma must be provided for a Gaussian pulse.")
        t_dependence = gaussian_wrapper(frequency, phase, pulse_time, sigma)
    else:
        raise ValueError("Unsupported pulse shape. Use 'square' or 'gaussian'.")
    h_t_independent = pulse_t_independent_op(spin, B_1, theta_1, phi_1)
    if factor_t_dependence:
        return [Qobj(h_t_independent), t_dependence]
    else:
        return Qobj((t_dependence(t) * h_t_independent))


""" Helper Functions that creates Hamiltonian terms in the QuTiP format """


def cosine_wrapper(frequency: float, phase: float, pulse_time: float) -> Callable[[float], float]:
    """
    Return the time-dependent coefficient of a pulse Hamiltonian.

    Parameters
    ----------
    frequency : non-negative float
        Frequency of the monochromatic wave (expressed in rad/sec).
    phase : float
        Initial phase of the wave (at t=0) (expressed in radians).
    pulse_time : float
        Time duration of the pulse (in microseconds).

    Returns
    -------
    A function with signature f(t: float, args: iterable) -> float
    """

    # The second argument 'args' is added to match qutip's documentation convention
    def time_dependence_function(t):
        if t <= pulse_time:
            return np.cos(frequency * t - phase)
        else:
            return 0

    return time_dependence_function


def pulse_t_independent_op(spin: NuclearSpin, B_1: float, theta_1: float, phi_1: float) -> Qobj:
    """
    Computes the time-independent portion of the Hamiltonian interaction with a
    monochromatic and linearly polarized electromagnetic pulse.

    Parameters
    ----------
    spin : NuclearSpin
        Spin under study.
    B_1 : non-negative float
        Maximum amplitude of the oscillating magnetic field (expressed in tesla).
    theta_1, phi_1 : float
        Polar and azimuthal angles of the direction of polarization of
        the magnetic wave in the LAB frame (expressed in radians);

    Returns
    -------
    Qobj
    """
    return (
        -2
        * spin.gyro_ratio_over_2pi
        * B_1
        * (
            np.sin(theta_1) * np.cos(phi_1) * spin.I["x"]
            + np.sin(theta_1) * np.sin(phi_1) * spin.I["y"]
            + np.cos(theta_1) * spin.I["z"]
        )
    )


""" magnus function """


def magnus(
    h_total: Qobj,
    rho0: Qobj,
    tlist: NDArray,
    order: int,
    spin: NuclearSpin,
    mode: Pulses,
    o_change_of_picture: Qobj,
) -> Result:
    """
    Magnus expansion solver, up to 3rd order.
    Integration by the trapezoid rule.

    Parameters
    ----------
    h_total : Qobj
        Time-independent Hamiltonian (expressed in MHz). Technically, an array
        of Observable objects which correspond to the Hamiltonian evaluated at
        successive instants of time. The start and end points of the array are
        taken as the extremes of integration 0 and t;
    rho0 : Qobj
        Initial density matrix
    tlist : Iterable[float]
        List of times at which the system will be solved.
    order : int
        The order number for magnus
    spin : NuclearSpin
        Spin under study.
    mode : Pulses
        Dataclass of the parameters of each electromagnetic mode in the pulse.
        Refer to parameter mode in the function h_multiple_mode_pulse for more
        information.
    o_change_of_picture : Qobj
        Operator which generates the change to the new picture.

    Returns
    -------
    qutip.Result instance with the evolved density matrix.
    """
    if order > 3:
        raise ValueError("Magnus expansion solver does not support order > 3. " + f"Given order {order}.")

    # output = Result()
    # output.times = tlist
    # output.solver = "magnus"
    time_step = (tlist[-1] - tlist[0]) / (len(tlist) - 1)
    h = []
    integral = 0
    # Use trange for progress bar
    for t in range(len(tlist)):
        H = h_changed_picture(spin, mode, h_total, o_change_of_picture, tlist[t])
        # Trapezoid Rule
        factor = 1
        if t != 0 and t != len(tlist) - 1:
            factor *= 2
        integral += factor * H * 2j * np.pi * time_step / 2

        if order >= 2:
            h.append(H)
            for t2 in range(t + 1):
                factor = 1
                if t != 0 and t != len(tlist) - 1:
                    factor *= 2
                if t2 != 0 and t2 != t:
                    factor *= 2
                integral += factor * commutator(h[t], h[t2]) * ((2 * np.pi * time_step) ** 2) * (1 / 2)

            # TODO: is this supposed to be inside the for loop? It's weird to reference `t2` outside the loop.
            if order >= 3:
                for t3 in range(1, t2 + 1):
                    factor = 1
                    if t != 0 and t != len(tlist) - 1:
                        factor *= 2
                    if t2 != 0 and t2 != t:
                        factor *= 2
                    if t3 != 0 and t3 != t2:
                        factor *= 2
                    integral += (
                        factor
                        * ((2 * np.pi * time_step) ** 3)
                        * (-1j / 6)
                        * (commutator(h[t], commutator(h[t2], h[t3])) + commutator(h[t3], commutator(h[t2], h[t])))
                    )

    # dm_evolved_new_picture = rho0.transform((- integral).expm())
    # dm_evolved_new_picture = (- integral).expm() * rho0 * ((- integral).expm()).dag()
    dm_evolved_new_picture = apply_exp_op(rho0, -integral)
    # output.states = [rho0, dm_evolved_new_picture]
    # return output
    return dm_evolved_new_picture


""" Mostly mathematical functions that constructs a specific term for the Hamiltonian """


def h_zeeman(
    spin: NuclearSpin, 
    theta_z: float, 
    phi_z: float, 
    B_0: float
) -> Qobj:
    """
    Computes the term of the Hamiltonian associated with the Zeeman interaction between the nuclear spin and the
     external static field.

    Parameters
    ----------
    spin : NuclearSpin
        Spin under study.
    theta_z : float
        Polar angle of the magnetic field in the laboratory coordinate system (expressed in radians).
    phi_z : float
        Azimuthal angle of the magnetic field in the laboratory coordinate system (expressed in radians).
    B_0 : non-negative float
        Magnitude of the external magnetic field (expressed in tesla).

    Returns
    -------
    Qobj
    An Observable object which represents the Zeeman Hamiltonian
    in the laboratory reference frame (expressed in MHZ).

    Raises
    ------
    ValueError, when the passed B_0 is a negative number.
    """
    if B_0 < 0:
        raise ValueError("The modulus of the magnetic field must be a non-negative quantity")

    h_z = (
        -spin.gyro_ratio_over_2pi
        * B_0
        * (
            np.sin(theta_z) * np.cos(phi_z) * spin.I["x"]
            + np.sin(theta_z) * np.sin(phi_z) * spin.I["y"]
            + np.cos(theta_z) * spin.I["z"]
        )
    )
    return Qobj(h_z)


def h_quadrupole(
    spin: NuclearSpin, 
    e2qQ: float, 
    eta: float, 
    alpha_q: float, 
    beta_q: float, 
    gamma_q: float, 
    component_order: int = 0
) -> Qobj:
    """
    Computes the term of the Hamiltonian associated with the quadrupolar interaction.

    Parameters
    ----------
    spin: NuclearSpin
        Spin under study.
    e2qQ: float
        Product of the quadrupole moment constant, eQ, and the eigenvalue of the EFG tensor
        which is greatest in absolute value, eq. e2qQ is measured in MHz.
    eta: float in the interval [0, 1]
        Asymmetry parameter of the EFG.
    alpha_q, beta_q, gamma_q: float
        Euler angles for the conversion from the system of the principal axes
        of the EFG tensor (PAS) to the lab system (LAB) (expressed in radians).
    component_order: int
        Order of the quadrupolar interaction in the LAB frame.

    Returns
    -------
    Qobj
    If the quantum number of the spin is 1/2, the whole calculation is skipped and a null Observable object is returned.
    Otherwise, the function returns the Observable object which correctly represents the quadrupolar Hamiltonian in the
    laboratory reference frame (expressed in MHz).

    """
    if np.isclose(spin.quantum_number, 1 / 2, rtol=1e-10):
        qobj_array = np.zeros((spin.d, spin.d))
        return Qobj(qobj_array)
    I = spin.quantum_number
    h_q = (
        e2qQ
        / (I * (2 * I - 1))
        * (1 / 2 * (3 * (spin.I["z"] ** 2) - qeye(spin.d) * I * (I + 1)) * v0_EFG(eta, alpha_q, beta_q, gamma_q))
    )
    if component_order > 0:
        h_q += (
            e2qQ
            / (I * (2 * I - 1))
            * np.sqrt(6)
            / 4
            * (
                (spin.I["z"] * spin.I["+"] + spin.I["+"] * spin.I["z"]) * v1_EFG(-1, eta, alpha_q, beta_q, gamma_q)
                + (spin.I["z"] * spin.I["-"] + spin.I["-"] * spin.I["z"]) * v1_EFG(+1, eta, alpha_q, beta_q, gamma_q)
            )
        )

    if component_order > 1:
        h_q += (
            e2qQ
            / (I * (2 * I - 1))
            * (
                (spin.I["+"] ** 2) * v2_EFG(-2, eta, alpha_q, beta_q, gamma_q)
                + (spin.I["-"] ** 2) * v2_EFG(2, eta, alpha_q, beta_q, gamma_q)
            )
        )
    return Qobj(h_q)


def v0_EFG(eta: float, alpha_q: float, beta_q: float, gamma_q: float) -> float:
    """
    Returns the component V0 of the EFG tensor (divided by eq) as seen in the LAB system. This quantity is expressed
    in terms of the Euler angles which relate PAS and LAB systems and the parameter eta.

    Parameters
    ----------
    eta : float in the interval [0, 1]
        Asymmetry parameter of the EFG.
    alpha_q, beta_q, gamma_q : float
        Euler angles connecting the system of the principal axes of the EFG tensor (PAS) to
        the lab system (LAB) (expressed in radians).
        alpha_q is not used.

    Returns
    -------
    A float representing the component V0 (divided by eq) of the EFG tensor evaluated in the LAB system.

    Raises
    ValueError, when the passed eta is not in the interval [0, 1].
    """
    if eta < 0 or eta > 1:
        raise ValueError("The asymmetry parameter must fall in the interval [0, 1]")
    v0 = (1 / 2) * (((3 * (np.cos(beta_q)) ** 2 - 1) / 2) - (eta * (np.sin(beta_q)) ** 2) * (np.cos(2 * gamma_q)) / 2)
    return v0


def v1_EFG(sign: float, eta: float, alpha_q: float, beta_q: float, gamma_q: float) -> complex:
    """
    Returns the components V+/-1 of the EFG tensor (divided by eq) as seen in the LAB system.
    These quantities are expressed in terms of the Euler angles which relate PAS and LAB systems and the parameter eta.

    Parameters
    ----------
    sign : float
        Specifies whether the V+1 or the V-1 component is to be computed.
    eta : float in the interval [0, 1]
        Asymmetry parameter of the EFG.
    alpha_q, beta_q, gamma_q : float
        Euler angles connecting the system of the principal axes of the EFG tensor (PAS)
        to the lab system (LAB) (expressed in radians).

    Returns
    -------
    A complex number representing the component:
        V<sup>+1</sup>, if sign is positive;
        V<sup>-1</sup>, if sign is negative.
    of the EFG tensor (divided by eq).

    Raises
    ------
    ValueError, when the passed eta is not in the interval [0, 1].
    """
    if eta < 0 or eta > 1:
        raise ValueError("The asymmetry parameter must fall within the interval [0, 1]")
    sign = np.sign(sign)
    v1 = (
        1
        / 2
        * (
            -1j * sign * np.sqrt(3 / 8) * np.sin(2 * beta_q) * np.exp(sign * 1j * alpha_q)
            + 1j
            * (eta / (np.sqrt(6)))
            * np.sin(beta_q)
            * (
                ((1 + sign * np.cos(beta_q)) / 2) * np.exp(1j * (sign * alpha_q + 2 * gamma_q))
                - ((1 - sign * np.cos(beta_q)) / 2) * np.exp(1j * (sign * alpha_q - 2 * gamma_q))
            )
        )
    )
    return v1


def v2_EFG(sign: float, eta: float, alpha_q: float, beta_q: float, gamma_q: float) -> float:
    """
    Returns the components V+/-2 of the EFG tensor (divided by eq) as seen in the LAB system.
    These quantities are expressed in terms of the Euler angles which
    relate PAS and LAB systems and the parameter eta.

    Parameters
    ----------
    sign : float
        Specifies whether the V+2 or the V-2 component is to be returned.
    eta : float in the interval [0, 1]
        Asymmetry parameter of the EFG tensor.
    alpha_q, beta_q, gamma_q : float
        Euler angles connecting the system of the principal axes of the
        EFG tensor (PAS) to the lab system (LAB) (expressed in radians).

    Returns
    -------
    A float representing the component:
        V+2, if sign is positive;
        V-2, if sign is negative.
    of the EFG tensor (divided by eq).

    Raises
    ------
    ValueError, when the passed eta is not in the interval [0, 1].
    """
    if eta < 0 or eta > 1:
        raise ValueError("The asymmetry parameter must fall in the interval [0, 1]")
    sign = np.sign(sign)
    v2 = (
        1
        / 2
        * (
            np.sqrt(3 / 8) * ((np.sin(beta_q)) ** 2) * np.exp(sign * 2j * alpha_q)
            + (eta / np.sqrt(6))
            * np.exp(sign * 2j * alpha_q)
            * (
                np.exp(2j * gamma_q)
                * (1 + sign * np.cos(beta_q)) ** 2
                / 4
                * np.exp(-2j * gamma_q)
                * (1 - sign * np.cos(beta_q)) ** 2
                / 4
            )
        )
    )

    return v2


def gaussian_wrapper(frequency: float, phase: float, pulse_time: float, sigma: float) -> Callable[[float], float]:
    """
    Return the time-dependent coefficient of a Gaussian-shaped pulse Hamiltonian.

    Parameters
    ----------
    frequency : non-negative float
        Frequency of the monochromatic wave (expressed in rad/sec).
    phase : float
        Initial phase of the wave (at t=0) (expressed in radians).
    pulse_time : float
        Time duration of the pulse (in microseconds).
    sigma : float
        Standard deviation of the Gaussian envelope (in microseconds).

    Returns
    -------
    A function with signature f(t: float, args: iterable) -> float
    """

    # The second argument 'args' is added to match QuTiP's documentation convention
    def time_dependence_function(t):
        if t <= pulse_time:
            # Gaussian envelope multiplied by the cosine
            gaussian_envelope = np.exp(-((t - pulse_time / 2) ** 2) / (2 * sigma**2))
            return gaussian_envelope * np.cos(frequency * t - phase)
        else:
            return 0

    return time_dependence_function


# Global Hamiltonian of the system (stationary term + pulse term) cast in the picture generated by
# the Operator h_change_of_picture
def h_changed_picture(
    spin: NuclearSpin | ManySpins, mode: Pulses, h_unperturbed: Qobj, h_change_of_picture: Qobj, t: float
) -> Qobj:
    """
    Returns the global Hamiltonian of the system, made up of the time-dependent
    term `h_multiple_mode_pulse(spin, mode, t)` and the stationary term
    h_unperturbed, cast in the picture generated by `h_change_of_picture`.

    Parameters
    ----------
    spin, mode, t :
        same meaning as the corresponding arguments of h_multiple_mode_pulse.
    h_unperturbed : Qobj
        Stationary term of the global Hamiltonian (in MHz).
    h_change_of_picture : Qobj
        Operator which generates the new picture (in MHz).

    Returns
    -------
    A Qobj representing the Hamiltonian of the pulse evaluated at
    time t in the new picture (in MHz).
    """
    h_pulse = h_multiple_mode_pulse(spin, mode, t)
    # If h_pulse is of the form [(H_time_independent, H_time_dependent_function), ...],
    # convert it to a Qobj evaluated at time t
    if isinstance(h_pulse, list):
        assert not isinstance(h_pulse, Qobj)
        if len(h_pulse) == 1:
            h_pulse = h_pulse[0]
        h_pulse = QobjEvo(h_pulse).__call__(t=t)

    h_cp = changed_picture(h_unperturbed + h_pulse - h_change_of_picture, h_change_of_picture, t)
    return Qobj(h_cp)


def h_changed_picture_func(
    spin: NuclearSpin | ManySpins, mode: Pulses, h_unperturbed: Qobj, h_change_of_picture: Qobj, t: float
) -> Callable:
    """
    Returns the global Hamiltonian of the system, made up of the time-dependent
    term `h_multiple_mode_pulse` and the stationary term `h_unperturbed`,
    cast in the picture generated by `h_change_of_picture`.

    Parameters
    ----------
    spin, mode, t :
        same meaning as the corresponding arguments of h_multiple_mode_pulse.
    h_unperturbed : Qobj
        Stationary term of the global Hamiltonian (in MHz).
    h_change_of_picture : Qobj
        Operator which generates the new picture (in MHz).

    Returns
    -------
    A function representing the Hamiltonian of the pulse evaluated at
    time t in the new picture (in MHz).
    """

    def func(t, args):
        h_pulse = h_multiple_mode_pulse(spin, mode, t)
        h_cp = changed_picture((h_unperturbed + h_pulse - h_change_of_picture), h_change_of_picture, t)
        return Qobj(h_cp)

    return func


def h_j_coupling(spins: ManySpins, j_matrix: NDArray) -> Qobj:
    """
    Returns the term of the Hamiltonian describing the J-coupling between the
    spins of a system of many nuclei.

    Parameters
    ----------
    spins: ManySpins
        Spins' system under study.

    j_matrix: np.ndarray
        Array storing the coefficients Jmn which enter the formula for
        the computation of the Hamiltonian for the j-coupling.

        Remark: j_matrix doesn't have to be symmetric, since the function reads
        only those elements located in the upper half with respect to the
        diagonal. This means that the elements j_matrix[m, n] which matter
        are those for which m<n.

    Returns
    -------
    A Qobj acting on the full Hilbert space of the spins' system
    representing the Hamiltonian of the J-coupling between the spins.
    """
    # dimensions of vector inputs to tensor; should be same as dual vector
    # inputs, i.e., tensor valence/rank should be (r, k) with r = k. equiv.
    # to matrix being square.
    dims = spins.dims
    h_j = Qobj(np.zeros((spins.d, spins.d)), dims=dims)

    # row
    for m in range(j_matrix.shape[0]):
        # column
        for n in range(m):
            term_nm = j_matrix[n, m] * spins.spins[n].I["z"]
            for l in range(n):
                term_nm = tensor(qeye(spins.spins[l].d), term_nm)
            for k in range(m)[n + 1 :]:
                term_nm = tensor(term_nm, qeye(spins.spins[k].d))
            term_nm = tensor(term_nm, spins.spins[m].I["z"])
            for j in range(spins.n_spins)[m + 1 :]:
                term_nm = tensor(term_nm, qeye(spins.spins[j].d))

            h_j = h_j + term_nm

    return h_j


def h_CS_isotropic(spin: NuclearSpin, delta_iso: float, B_0: float) -> Qobj:
    """
    Computes the term of the Hamiltonian associated with the chemical shift
    interaction in the secular approximation for isotropic liquids between the
    nuclear spin and the external static field.

    Parameters
    ----------
    spin : NuclearSpin
        Spin under study.
    delta_iso : float
        Magnitude of the chemical shift in Hz: H_CS = -delta_iso\omega_0 Iz
    B_0 : non-negative float
        Magnitude of the external magnetic field (expressed in tesla).

    Returns
    -------
    A Qobj which represents the Zeeman Hamiltonian in the laboratory reference
    frame (expressed in MHz).

    Raises
    ------
    ValueError, when the passed B_0 is a negative number.
    """
    if B_0 < 0:
        raise ValueError("The magnitude of the magnetic field must be a non-negative quantity")
    h_cs = -delta_iso * spin.gyro_ratio_over_2pi * B_0 * spin.I["z"]
    return Qobj(h_cs)


def h_D1(spins: ManySpins, b_D: float, theta: float) -> Qobj:
    """
    Computes the term of the Hamiltonian associated with the dipolar interaction in the secular
    approximation for homonuclear & heteronuclear spins.
    {H}_{D1} \approx  b_D \frac{3\cos^2\theta-1}{2})[3I_{1z}I_{2z} -\mathbf{I}_1\cdot  \mathbf{I}_2].

    Parameters
    ----------
    spins : ManySpins
        2 Spins in the system under study;
    b_D : float
        Magnitude of dipolar constant,
        b_D\equiv \frac{\mu_0\gamma_1\gamma_2}{4\pi r^3_{21}}.
    theta : float
        Polar angle between the two spins (expressed in radians).

    Returns
    -------
    A Qobj operator acting on the full Hilbert space of the 2-spin system
    representing the Hamiltonian.

    """
    h_d1 = (
        b_D
        * 1
        / 2
        * (3 * (np.cos(theta) ** 2) - 1)
        * (
            2 * tensor(spins.spins[0].I["z"], spins.spins[1].I["z"])
            - tensor(spins.spins[0].I["x"], spins.spins[1].I["x"])
            - tensor(spins.spins[0].I["y"], spins.spins[1].I["y"])
        )
    )
    return Qobj(h_d1)


def h_D2(spins: ManySpins, b_D: float, theta: float) -> Qobj:
    """
    Computes the term of the Hamiltonian associated with the dipolar interaction in the secular
    approximation for heteronuclear spins. H_{D2} \approx \hslash b_D (3\cos^2\theta-1)I_{1z}I_{2z}.

    Parameters
    ----------
    spins : ManySpins
        2 Spins in the system under study.
    b_D : float
        Magnitude of dipolar constant,
        b_D\equiv \frac{\mu_0\gamma_1\gamma_2}{4\pi r^3_{21}}.
    theta : float
        Polar angle between the two spins (expressed in radians).

    Returns
    -------
    A Qobj operator acting on the full Hilbert space of the 2-spin system
    representing the Hamiltonian.

    """
    h_d2 = b_D * (3 * (np.cos(theta) ** 2) - 1) * (tensor(spins.spins[0].I["z"], spins.spins[1].I["z"]))
    return Qobj(h_d2)


def h_HF_secular(spins: ManySpins, A: float, B: float) -> Qobj:
    """
    Computes the term of the Hamiltonian associated with the hyperfine interaction in the secular
    approximation for between two spins. H_{D2} \approx A S_{z}I_{z} + B S_{z}I_{x}  .

    Parameters
    ----------
    spins : ManySpins
        2 Spins in the system under study.
    A : float
        Constant, see paper.
    B : float
        Constant, see paper.

    Returns
    -------
    A Qobj operator acting on the full Hilbert space of the 2-spin system
    representing the Hamiltonian.

    """
    h_hf = A * tensor(spins.spins[0].I["z"], spins.spins[1].I["z"]) + B * tensor(
        spins.spins[0].I["z"], spins.spins[1].I["x"]
    )
    return Qobj(h_hf)


def h_j_secular(spins: ManySpins, j_secular: dict[tuple[int, int], float]) -> Qobj:
    """
    Computes the term of the Hamiltonian associated with the J-coupling in the secular approximation  between two spins.

    Parameters
    ----------
    spins : ManySpins
        2 Spins in the system under study.
    j_secular : dict
        (copy & pasted from `nuclear_system_setup`)
        Map containing information about the J-couping in the secular
        approximation. The keys and values required to this argument are shown in
        the table below:

        |         key         |       value      |
        |         ---         |       -----      |
        |     (i,j): tuple    |      J: float    |

        where (i,j) is a tuple of spin indices i and j, between which J-coupling is present, and
        J is the J-coupling value in Hz.

    Returns
    -------
    A Qobj operator acting on the full Hilbert space of the 2-spin system
    representing the Hamiltonian.

    """
    h_j = None
    for i, j in j_secular.keys():
        # construct the term: (where id is the identity matrix and "x" is the tensor product)
        # term_tensored = id_0 x id_1 x ... x Iz_i ... x Iz_j x ... x id_(n-1) x id_n
        # spin operator Iz at index i and j, but the identity operator at every other index.
        # else.
        term_tensored = None
        for k in range(spins.n_spins):
            if k == i or k == j:
                term = spins.spins[k].I["z"]
            else:
                term = qeye(spins.dims[0][k])

            if term_tensored is None:  # initialize
                term_tensored = term
            else:
                term_tensored = tensor(term_tensored, term)

        J = j_secular[(i, j)] * 1e-6  # Convert Hz to MHz
        if h_j is None:  # initialize
            h_j = J * term_tensored
        else:
            h_j += J * term_tensored

    return Qobj(h_j)


def h_tensor_coupling(spins: ManySpins, t: NDArray) -> Qobj:
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
    spins: ManySpins
        2-spin system under study
    t: a numpy ndarray representing the interaction tensor of this
        Hamiltonian in MHz.

    Returns
    ------
    A Qobj operator acting on full Hilbert space of the 2-spin system
    representing the Hamiltonian of this interaction.
    """

    spin_0_ops = [spins.spins[0].I[key] for key in ["x", "y", "z"]]
    spin_1_ops = [spins.spins[1].I[key] for key in ["x", "y", "z"]]

    # Initialize empty operator of appropriate dimension as base case for the for loop.
    h = Qobj(np.zeros((spins.d, spins.d)), dims=spins.dims)

    for m, op_1 in enumerate(spin_0_ops):
        for n, op_2 in enumerate(spin_1_ops):
            h += t[m, n] * tensor(op_1, op_2)
    return h


""" Helper Functions To Manipulate Hamiltonians """


def multiply_by_2pi(h_unscaled: list[Qobj]) -> list[Qobj]:
    """
    Multiples the input 'hamiltonian structure' by 2 pi to correctly scale
    the Hamiltonian (which will get passed into qutip's mesolve).

    Parameters
    ----------
    h_unscaled : list(Qobj)
        Assumed to be in the form of the input Hamiltonian for qutip's mesolve
        function: [H0, [H1, f1(t)], [H2, f2(t)], ...]

    Returns
    -------
    The input Hamiltonian structure multiplied by 2pi.
    """
    h_scaled = []
    for h in h_unscaled:
        if isinstance(h, list) or isinstance(h, tuple):  # of the form: (Hm, fm(t))
            h_scaled.append([h[0] * 2 * np.pi, h[1]])
        else:  # of the form: H0
            h_scaled.append(2 * np.pi * h)
    return h_scaled


def rotating_frame_h(h: QobjEvo, w_ref: float, spins: ManySpins | NuclearSpin) -> QobjEvo:
    """
    Transforms the Hamiltonain H into the rotating frame:
    Rz(-angle) * H * Rz(angle) - w_ref * IZ, where 'angle' = w_ref * t + phi_ref
    For more details, see 15.3 (pg375) of Spin Dynamics - Levitt

    Parameters
    ----------
    h: Qobj
        the Hamiltonian to transform into the rotating frame
        Assuming the form of: [H0, [H1, f1(t)], [H2, f2(t)], ...]
        (refer to QuTiP's mesolve documentation for further detail)
    w_ref: float
        The reference frequency of the rotating frame.
        ** IN THE UNITS OF RAD/SEC !! **
    spins: ManySpins or NuclearSpins
        The spin system
    """
    if spins.gyro_ratio_over_2pi > 0:
        phi_ref = np.pi
    else:
        phi_ref = 0

    # construct the rotation operator
    if isinstance(spins, ManySpins):
        Iz = spins.many_spin_operator(component="z", spin_target="all")
    elif isinstance(spins, NuclearSpin):
        Iz = spins.I["z"]
    else:
        raise TypeError("`spins` must be of type ManySpins or NuclearSpin")

    def op_rotation(t: float) -> Qobj:
        return (-1j * (w_ref * t + phi_ref) * Iz).expm()

    # Need to create a 'master function' that takes time as argument and returns a Qobj
    def h_rotated(t: float) -> Qobj:
        return op_rotation(t) * h.__call__(t) * op_rotation(t).dag() - w_ref * Iz

    return QobjEvo(h_rotated, compress=True)

    # # Need to create a 'master function' that takes time as argument and returns a Qobj
    # def h_rotated(t: float) -> Qobj:
    #     h_final = Qobj(np.zeros(spins.shape), dims=spins.dims)
    #     for i, h_term in enumerate(h.to_list()):
    #         if isinstance(h_term, Qobj):  # time-independent term (H0)
    #             if is_diagonal(h_term):  # If H0 is diagonal, it will not be affected by the transformation
    #                 h_final += h_term  # this will save computation time
    #             else:
    #                 h_final += op_rotation(t) * h_term * op_rotation(t).dag()
    #         elif isinstance(h_term, Iterable):  # time-dependent term [Hm, fm(t)]
    #             h_product = h_term[0] * h_term[1](t)
    #             if is_diagonal(h_product):  # If H0 is diagonal, it will not be affected by the transformation
    #                 h_final += h_product  # this will save computation time
    #             else:
    #                 h_final += op_rotation(t) * h_product * op_rotation(t).dag()
    #
    #     return h_final - w_ref * Iz
    #
    # return h_rotated

    # Transform the Hamiltonian, term by term
    # h_rot = []
    # for i, h_term in enumerate(h.to_list()):
    #     if isinstance(h_term, Qobj):  # time-independent term (H0)
    #         if is_diagonal(h_term):  # If H0 is diagonal, it will not be affected by the transformation
    #             h_rot.append(h_term)  # this will save computation time
    #         else:
    #             # Convert to a [identity, Rz * H0 * Rz.dag()] term
    #             h_rot.append([Qobj(qeye(spins.d), dims=spins.dims),  # identity
    #                           lambda t: op_rotation(t) * h_term * op_rotation(t).dag()])
    #     elif isinstance(h_term, Iterable):  # time-dependent term [Hm, fm(t)]
    #         t_ind_term, t_dep_function = h_term
    #         if is_diagonal(t_ind_term): # If H0 is diagonal, it will not be affected by the transformation
    #             h_rot.append(h_term) # this will save computation time
    #         else:
    #             def t_dep_rotated(t):
    #                 return op_rotation(t) * t_dep_function(t) * op_rotation(t).dag()
    #             h_rot.append([t_ind_term, t_dep_rotated])
    #     else:
    #         raise TypeError("Every element of the 'Hamiltonian object' should either be a Qobj or a list, with the "
    #                         "form [H0, [H1, f1(t)], [H2, f2(t)], ...]")
    #
    # # Now the first time-independent term H0 will be: H0 = -w_ref * Iz
    # h_rot.insert(0, -w_ref * Iz)
    #
    # return QobjEvo(h_rot, compress=True)
