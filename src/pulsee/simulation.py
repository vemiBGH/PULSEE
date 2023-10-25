import numpy as np
import pandas as pd
from fractions import Fraction
import sys
from tqdm import tqdm, trange
from scipy.fft import fft, fftfreq, fftshift

from qutip import Options, mesolve, Qobj, tensor, expect, qeye, spin_coherent
from qutip.parallel import parallel_map
from qutip.ipynbtools import parallel_map as ipynb_parallel_map

import matplotlib.pylab as plt
from matplotlib import colors as clrs
from matplotlib import colorbar as clrbar
from matplotlib.pyplot import xticks, yticks
from matplotlib.patches import Patch

from pulsee.operators import canonical_density_matrix, \
    evolve_by_hamiltonian, \
    changed_picture, exp_diagonalize, \
    apply_exp_op, apply_op

from pulsee.nuclear_spin import NuclearSpin, ManySpins

from pulsee.hamiltonians import h_zeeman, h_quadrupole, \
    h_multiple_mode_pulse, \
    h_j_coupling, \
    h_CS_isotropic, h_D1, h_D2, \
    h_HF_secular, h_j_secular, h_tensor_coupling, \
    h_userDefined, multiply_by_2pi, \
    magnus

from pulsee.spin_squeezing import CSS

def nuclear_system_setup(spin_par, quad_par=None, zeem_par=None, j_matrix=None,
                         cs_param=None, D1_param=None, D2_param=None,
                         hf_param=None, h_tensor_inter=None, j_sec_param=None,
                         h_userDef=None, initial_state='canonical',
                         temperature=1e-4):
    """
    Sets up the nuclear system under study, returning the objects representing
    the spin (either a single one or a multiple spins' system), the unperturbed
    Hamiltonian (made up of the Zeeman, quadrupolar and J-coupling
    contributions) and the initial state of the system.

    Parameters
    ----------
    spin_par : dict / list of dict
        Map/list of maps containing information about the nuclear spin/spins under
        consideration. The keys and values required to each dictionary in this
        argument are shown in the table below.

        |           key          |         value        |
        |           ---          |         -----        |
        |    'quantum number'    |  half-integer float  |
        |       'gamma/2pi'      |         float        |

        The second item is the gyromagnetic ratio over 2 pi, measured in MHz/T.
        E.g., spin_par = {'quantum number': 1 / 2, 'gamma2/pi': 1}

    quad_par : dict / list of dict
        Map/maps containing information about the quadrupolar interaction between
        the electric quadrupole moment and the EFG for each nucleus in the system.
        The keys and values required to each dictionary in this argument are shown
        in the table below:

        |           key           |       value        |
        |           ---           |       -----        |
        |   'coupling constant'   |       float        |
        |  'asymmetry parameter'  |   float in [0, 1]  |
        |        'alpha_q'        |       float        |
        |        'beta_q'         |       float        |
        |        'gamma_q'        |       float        |
        |         'order'         |       int          |

        where 'coupling constant' stands for the product e2qQ in the expression of
        the quadrupole term of the Hamiltonian (to be provided in MHz), 'asymmetry
        parameter' refers to the same-named property of the EFG, and 'alpha_q',
        'beta_q' and 'gamma_q' are the Euler angles for the conversion from the LAB
        coordinate system to the system of the principal axes of the EFG tensor
        (PAS) (to be expressed in radians).

        When it is None, the quadrupolar interaction of all the spins in the
        system is not taken into account.
        Default value is None.

    zeem_par : dict
        Map containing information about the magnetic field interacting with the
        magnetic moment of each nucleus in the system. The keys and values
        required to this argument are shown in the table below:

        |         key         |       value      |
        |         ---         |       -----      |
        |      'theta_z'      |       float      |
        |       'phi_z'       |       float      |
        |  'field magnitude'  |  positive float  |

        where 'theta_z' and 'phi_z' are the polar and azimuthal angles of the
        magnetic field with respect to the LAB system (to be measured in radians),
        while field magnitude is to be expressed in tesla.

        When it is None, the Zeeman interaction is not taken into account.
        Default value is None.

    j_matrix : np.ndarray
        Array whose elements represent the coefficients Jmn which determine the
        strength of the J-coupling between each pair of spins in the system. For
        the details on these data, see the description of the same-named parameter
        in the docstrings of the function h_j_coupling in the module
        Hamiltonians.py.

        When it is None, the J-coupling effects are not taken into account.      
        Default value is None.

    cs_param : dict
        Map containing information about the chemical shift. The keys and values
        required to this argument are shown in the table below:

        |         key         |       value      |
        |         ---         |       -----      |
        |      'delta_iso'    |       float      |

        where delta_iso is the magnitude of the chemical shift in Hz.

        When it is None, the chemical shift is not taken into account.      
        Default value is None.

    D1_param : dict
        Map containing information about the dipolar interaction in the secular
        approximation for homonuclear & heteronuclear spins. The keys and values
        required to this argument are shown in the table below:

        |         key         |       value      |
        |         ---         |       -----      |
        |        'b_d'        |       float      |
        |       'theta'       |       float      |

        where b_d is the magnitude of dipolar constant,  b_D\equiv
        \frac{\mu_0\gamma_1\gamma_2}{4\pi r^3_{21}}, and theta is the polar angle
        between the two spins (expressed in radians).

        When it is None, the dipolar interaction in the secular approximation
        for homonuclear & heteronuclear spins is not taken into account. 
        Default value is None.

    D2_param : dict
        Map containing information about the dipolar interaction in the secular
        approximation for heteronuclear spins. The keys and values required to this
        argument are shown in the table below:

        |         key         |       value      |
        |         ---         |       -----      |
        |        'b_d'        |       float      |
        |       'theta'       |       float      |

        where b_d is the magnitude of dipolar constant,  b_D\equiv
        \frac{\mu_0\gamma_1\gamma_2}{4\pi r^3_{21}}, and theta is the polar angle
        between the two spins (expressed in radians).

        When it is None, the dipolar interaction in the secular approximation
        for heteronuclear spins is not taken into account. 
        Default value is None.

    hf_param : dict
        Map containing information about the hyperfine interaction in the secular
        approximation between two spins. The keys and values required to this
        argument are shown in the table below:

        |         key         |       value      |
        |         ---         |       -----      |
        |         'A'         |       float      |
        |         'B'         |       float      |

        where A, B are constant of the hyperfine interaction inthe secular
        approximation, see paper. 

        When it is None, the hyperfine interaction in the secular approximation
        between two spins is not taken into account.      Default value is None.

    h_tensor_inter : numpy.ndarray  or a list of numpy.ndarrays
        Rank-2 tensor describing a two-spin interaction of the form 
        $\mathbf{I}_1 J \mathbf{I}_2$ where $J$ is the tensor and $\mathbf{I}_i$
        are vector spin operators.

        When it is None, the interaction is not taken into account. 
        Default value is None.

    j_sec_param : dict
        Map containing information about the J-couping in the secular
        approximation. The keys and values required to this argument are shown in
        the table below:

        |         key         |       value      |
        |         ---         |       -----      |
        |         'J'         |       float      |

        where J is the J-coupling constant in Hz.

        When it is None, the J-couping in the secular approximation is not taken
        into account. Default value is None.

    h_userDef : numpy.ndarray
        Square matrix array which will give the hamiltonian of the system, adding to
        previous terms (if any). When passing, must ensure compability with the rest
        of the system.
        Default value is None.

    initial_state : string or numpy.ndarray or dict
        Specifies the state of the system at time t=0.

        If the keyword canonical is passed, the function will return a
        Qobj representing the state of thermal equilibrium at the
        temperature specified by the same-named argument.

        If a dictionary {'theta' : rad, 'phi' : rad} is passed, a spin coherent
        state is created. Can pass a list of dictionaries for a ManySpins system
        to create a tensor product state.

        If a square complex array is passed, the function will return a
        Qobj directly initialised with it.

        Default value is 'canonical'.

    temperature : float
        Temperature of the system (in kelvin).
        Default value is 1e-4.

    Returns
    -------
    [0]: NuclearSpin / ManySpins
        The single spin/spin system subject to the NMR/NQR experiment.

    [1]: List[Qobj]
        The unperturbed Hamiltonian, consisting of the Zeeman, quadrupolar
        and J-coupling terms (expressed in MHz).

    [2]: Qobj
        The density matrix representing the state of the system at time t=0,
        initialised according to initial_state.
    """

    if not isinstance(spin_par, list):
        spin_par = [spin_par]
    if quad_par is not None and not isinstance(quad_par, list):
        quad_par = [quad_par]

    if quad_par is not None and len(spin_par) != len(quad_par):
        raise IndexError("The number of passed sets of spin parameters must be" +
                         " equal to the number of the quadrupolar ones.")

    spins = []
    h_q = []
    h_z = []

    for i in range(len(spin_par)):
        spins.append(NuclearSpin(spin_par[i]['quantum number'],
                                 spin_par[i]['gamma/2pi']))

        if quad_par is not None:
            h_q.append(h_quadrupole(spins[i], quad_par[i]['coupling constant'],
                                    quad_par[i]['asymmetry parameter'],
                                    quad_par[i]['alpha_q'],
                                    quad_par[i]['beta_q'],
                                    quad_par[i]['gamma_q'],
                                    quad_par[i]['order']))
        else:
            h_q.append(h_quadrupole(spins[i], 0., 0., 0., 0., 0.))

        if zeem_par is not None:
            h_z.append(h_zeeman(spins[i], zeem_par['theta_z'],
                                zeem_par['phi_z'], zeem_par['field magnitude']))
        else:
            h_z.append(h_zeeman(spins[i], 0., 0., 0.))

        if (cs_param is not None) and (cs_param != 0.0):
            h_z.append(h_CS_isotropic(spins[i], cs_param['delta_iso'], 
                                      zeem_par['field magnitude']))

    spin_system = ManySpins(spins)
    h_unperturbed = []

    for i in range(spin_system.n_spins):
        h_i = h_q[i] + h_z[i]
        for j in range(i):
            h_i = tensor(qeye(spin_system.spin[j].d), h_i)
        for k in range(spin_system.n_spins)[i + 1:]:
            h_i = tensor(h_i, qeye(spin_system.spin[k].d))
        h_unperturbed = h_unperturbed + [Qobj(h_i)]

    if j_matrix is not None:
        h_j = h_j_coupling(spin_system, j_matrix)
        h_unperturbed = h_unperturbed + [Qobj(h_j)]

    if D1_param is not None:
        if (D1_param['b_D'] == 0.) and (D1_param['theta'] == 0.):
            pass
        else:
            h_d1 = h_D1(spin_system, D1_param['b_D'],
                        D1_param['theta'])
            h_unperturbed = h_unperturbed + [Qobj(h_d1)]

    if D2_param is not None:
        if (D2_param['b_D'] == 0.) and (D2_param['theta'] == 0.):
            pass
        else:
            h_d2 = h_D2(spin_system, D2_param['b_D'], D2_param['theta'])
            h_unperturbed = h_unperturbed + [Qobj(h_d2)]

    if hf_param is not None:
        if (hf_param['A'] == 0.) and (hf_param['B'] == 0.):
            pass
        else:
            h_hf = h_HF_secular(spin_system, hf_param['A'],
                                hf_param['B'])
            h_unperturbed = h_unperturbed + [Qobj(h_hf)]

    if j_sec_param is not None:
        if j_sec_param['J'] == 0.0:
            pass
        else:
            h_j = h_j_secular(spin_system, j_sec_param['J'])
            h_unperturbed = h_unperturbed + [Qobj(h_j)]

    if h_tensor_inter is not None:
        if type(h_tensor_inter) != list:
            h_unperturbed += [Qobj(h_tensor_coupling(spin_system, h_tensor_inter))]
        else:
            for hyp_ten in h_tensor_inter:
                h_unperturbed += [Qobj(h_tensor_coupling(spin_system, hyp_ten))]

    if h_userDef is not None:
        h_unperturbed += (h_userDefined(h_userDef))

    if isinstance(initial_state, str) and initial_state == 'canonical':
        dm_initial = canonical_density_matrix(Qobj(sum(h_unperturbed)), temperature)

    elif 'theta' in np.all(initial_state) and 'phi' in np.all(initial_state):
        dm_initial = CSS(spin_system, initial_state)

    else:  # initial_state is a np array
        if isinstance(initial_state, Qobj) or isinstance(initial_state, np.ndarray):
            dm_initial = Qobj(initial_state)
        else:
            raise ValueError("Please check the type of the initial state passed.")

    if len(spins) == 1:
        return spins[0], h_unperturbed, dm_initial
    else:
        return spin_system, h_unperturbed, dm_initial


def power_absorption_spectrum(spin, h_unperturbed, normalized=True, dm_initial=None):
    """
    Computes the spectrum of power absorption of the system due to x-polarized
    monochromatic pulses.

    Parameters
    ----------
    spin : NuclearSpin / ManySpins
        Single spin/spin system under study.

    h_unperturbed : Operator
        Unperturbed Hamiltonian of the system (in MHz).

    normalized : bool
        Specifies whether the difference between the states' populations are 
        to be taken into account in the calculation of the line intensities. 
        When normalized=True, they are not, when normalized=False, 
        the intensities are weighted by the differences p(b)-p(a) 
        just like in the formula above.
        Default value is True.

    dm_initial : Qobj or None
        Density matrix of the system at time t=0, just before the
        application of the pulse.

        The default value is None, and it should be left so only when
        normalized=True, since the initial density matrix is not needed.

    Action
    ------
    Diagonalises h_unperturbed and computes the frequencies of transitions
    between its eigenstates.

    Then, it determines the relative proportions of the power absorption for
    different lines applying the formula derived from Fermi golden rule (taking
    or not taking into account the states' populations, according to the value
    of normalized).

    Returns:
    -------
    [0]: The list of the frequencies of transition between the eigenstates of
         h_unperturbed (in MHz);

    [1]: The list of the corresponding intensities (in arbitrary units).
    """
    # dims = [s.d for s in spin.spin]
    dims = h_unperturbed[0].dims
    shape = h_unperturbed[0].shape
    h_unperturbed = Qobj(sum(h_unperturbed), dims=dims)
    energies, o_change_of_basis = h_unperturbed.eigenstates()
    transition_frequency = []
    transition_intensity = []

    # assume that this Hamiltonian is a rank-1 tensor
    d = sum(h_unperturbed.dims[0])
    # Operator of the magnetic moment of the spin system
    if isinstance(spin, ManySpins):
        magnetic_moment = Qobj(np.zeros(shape), dims=dims)
        for i in range(spin.n_spins):
            mm_i = spin.spin[i].gyro_ratio_over_2pi * spin.spin[i].I['x']
            for j in range(i):
                mm_i = tensor(Qobj(qeye(spin.spin[j].d)), mm_i)
            for k in range(spin.n_spins)[i + 1:]:
                mm_i = tensor(mm_i, Qobj(qeye(spin.spin[k].d)))
            magnetic_moment += mm_i
    else:
        magnetic_moment = spin.gyro_ratio_over_2pi * spin.I['x']

    mm_in_basis_of_eigenstates = magnetic_moment.transform(o_change_of_basis)

    for i in range(d):
        for j in range(d):
            if i < j:
                nu = np.absolute(energies[j] - energies[i])
                transition_frequency.append(nu)
                intensity_nu = nu * np.absolute(mm_in_basis_of_eigenstates[j, i]) ** 2
                if not normalized:
                    p_i = dm_initial[i, i]
                    p_j = dm_initial[j, j]
                    intensity_nu = np.absolute(p_i - p_j) * intensity_nu
                transition_intensity.append(intensity_nu)
            else:
                pass
    return transition_frequency, transition_intensity


def plot_power_absorption_spectrum(frequencies, intensities, show=True,
                                   xlim=None, ylim=None, fig_dpi=400, save=False,
                                   name='PowerAbsorptionSpectrum', destination=''):
    """
    Plots the power absorption intensities as a function of the corresponding
    frequencies.

    Parameters
    ----------
    frequencies : array-like
        Frequencies of the transitions (in MHz).

    intensities : array-like
        Intensities of the transitions (in a.u.).

    show : bool
        When False, the graph constructed by the function will not be
        displayed.

        Default value is True.

    xlim : 2-element iterable or `None`
        Lower and upper x-axis limits of the plot.
        When `None` uses `matplotlib` default.

    ylim : 2-element iterable or `None`
        Lower and upper y-axis limits of the plot.
        When `None` uses `matplotlib` default.

    fig_dpi : int
        Image quality of the figure when showing and saving. Useful for
        publications. Default set to very high value.

    save : bool
        When False, the plotted graph will not be saved on disk. When True,
        it will be saved with the name passed as name and in the directory
        passed as destination.

        Default value is False.

    name : string
        Name with which the graph will be saved.
        Default value is 'PowerAbsorptionSpectrum'.

    destination : string
        Path of the directory where the graph will be saved (starting
        from the current directory). The name of the directory must
        be terminated with a slash /.

        Default value is the empty string (current directory).

    Action
    ------
    If show=True, generates a graph with the frequencies of transition on the x
    axis and the corresponding intensities on the y axis.

    Returns
    -------
    An object of the class matplotlib.figure.Figure representing the figure
    built up by the function.  
    """
    fig = plt.figure()

    plt.vlines(frequencies, 0, intensities, colors='b')
    plt.xlabel("\N{GREEK SMALL LETTER NU} (MHz)")
    plt.ylabel("Power absorption (a. u.)")

    if xlim is not None:
        plt.xlim(left=xlim[0], right=xlim[1])
    if ylim is not None:
        plt.xlim(left=ylim[0], right=ylim[1])
    if save:
        plt.savefig(destination + name, dpi=fig_dpi)
    if show:
        plt.show()

    return fig


def evolve(spin, h_unperturbed, dm_initial, solver=mesolve, mode=None,
           evolution_time=0., picture='IP', RRF_par=None,
           times=None, n_points=30, order=None, opts=None,
           return_allstates=False, display_progress=True):
    """
    Simulates the evolution of the density matrix of a nuclear spin under the
    action of an electromagnetic pulse in a NMR/NQR experiment.

    Parameters
    ----------
    spin : NuclearSpin
        Spin under study.

    h_unperturbed : List[Qobj] or List[(Qobj, function)]
        Hamiltonian of the nucleus at equilibrium (in MHz).

    dm_initial : Qobj
        Density matrix of the system at time t=0, just before the application 
        of the pulse.

    solver : function: (Qobj, Qobj, ndarray, **kwargs) -> qutip.solver.Result
             OR 
             string
        Solution method to be used when calculating time evolution of
        state. If string, must be either `mesolve` or `magnus.`

    mode : pandas.DataFrame
        Table of the parameters of each electromagnetic mode in the pulse.
        It is organised according to the following template:

        |index|'frequency'|'amplitude'| 'phase' |'theta_p'|'phi_p'|'pulse_time'|
        |-----|-----------|-----------|---------|---------|-------|------------|
        |     | (rad/sec) |    (T)    |  (rad)  |  (rad)  | (rad) |   (mus)    |
        |  0  |  omega_0  |    B_0    | phase_0 | theta_0 | phi_0 |   tau_0    |
        |  1  |  omega_1  |    B_1    | phase_1 | theta_1 | phi_1 |   tau_1    |
        | ... |    ...    |    ...    |   ...   |   ...   |  ...  |    ...     |
        |  N  |  omega_N  |    B_N    | phase_N | theta_N | phi_N |   tau_N    |

        where the meaning of each column is analogous to the corresponding
        parameters in h_single_mode_pulse.

        Important: The amplitude value is B_1, not 2*B_1. The code will
        automatically multiply by 2!

        When it is None, the evolution of the system is performed for the
        given time duration without any applied pulse.

        The default value is None.

    evolution_time : float
        Duration of the evolution (in microseconds).

        The default value is 0 or the max of pulses specified in mode,
        whichever is bigger.

    picture : string
        Sets the dynamical picture where the density matrix of the system
        is evolved for the `magnus` solver. May take the values:
            1.'IP', which sets the interaction picture;
            2.'RRF' (or anything else), which sets the picture corresponding to a
            rotating reference frame whose features are specified in argument
            RRF_par.

        The default value is 'IP'. For LAB frame, use picture='RRF' and
        give no RRF_par.
        The choice of picture has no effect on solvers other than `magnus`.

    RRF_par : dict
        Specifies the properties of the rotating reference frame where
        evolution is carried out when picture='RRF.' The
        keys and values required to this argument are shown in the table below:
        |      key      |  value  |
        |      ---      |  -----  |
        |'nu_RRF' (MHz) |  float  |
        |  'theta_RRF'  |  float  |
        |   'phi_RRF'   |  float  |
        where 'nu_RRF' is the frequency of rotation of the RRF (in MHz), while
        'theta_RRF' and 'phi_RRF' are the polar and azimuthal angles of the normal
        to the plane of rotation in the LAB frame (in radians).
        By default, all the values in this map are set to 0 (RRF equivalent
        to the LAB frame).

    times : list or np.array
        Pass in the array of times that the numerical calculation are performed for.
        Default is None.

    n_points : float
        Factor that multiplies the number of points, (points = [pulse_time * n_points])
        in which the time interval [0, pulse_time] is sampled in the discrete approximation
        of the time-dependent Hamiltonian of the system.
        Default value is 10.

    order : float
        The order of the simulation method to use. For `magnus` must be <= 3. 
        Defaults to 1 for `magnus` and 12 for `mesolve` and any other solver.

    return_allstates : boolean
        Specify whether to return every calculated state or only last one.
        Default False --> returns only last state.
        [Magnus solver only returns final state]

    display_progress: True or None
        True: display progress bar for the mesolve method
        None: don't display progress bar

    Action
    ------
    If
    - evolution_time=0 AND mode=None, or
    - dm_initial is very close to the identity 
      (with an error margin of 1e-10 for each element)

        the function returns dm_initial without performing any evolution.

    Otherwise, 
    evolution is carried out in the picture determined by the
    same-named parameter. The evolution operator is built up appealing to the
    Magnus expansion of the full Hamiltonian of the system (truncated to the
    order specified by the same-named argument).

    Note:
    QuTiP should be the preferred solver.

    Magnus works best if each pulse is evaluated individually because it is dependent on the
    time array. Advised  not use evolution_time with magnus,  rather call another instance
    of the evolve function.

    Returns
    -------
    The Qobj  representing the state of the system (in the
    Schroedinger picture) evolved through a time pulse_time under the action of
    the specified pulse.  
    """
    dims = spin.dims

    if mode is None:
        mode = pd.DataFrame([(0., 0., 0., 0., 0., 0.)],
                            columns=['frequency', 'amplitude', 'phase', 'theta_p', 'phi_p', 'pulse_time'])
    if np.min(mode['pulse_time']) < 0:
        raise ValueError(
            'Pulse duration must be a non-negative number. Given:' + str(np.min(mode['pulse_time'])))

    # In order to use the right hand rule convention, for positive gamma,
    # we 'flip' the pulse by adding pi to the phase,
    # Refer to section 10.6 (pg 244) of 'Spin Dynamics - Levitt' for more detail.
    if spin.gyro_ratio_over_2pi > 0:
        mode = mode.copy()  # in case the user wants to use same 'mode' variable for later uses.
        mode.loc[:, 'phase'] = mode.loc[:, 'phase'].add(np.pi)

    pulse_time = max(np.max(mode['pulse_time']), evolution_time)
    if (pulse_time == 0.) or \
            np.all(np.absolute((dm_initial.full() - np.eye(spin.d))) < 1e-10):
        return dm_initial

    if order is None and (solver == magnus or solver == 'magnus'):
        order = 1

    # match tolerance to operators.positivity tolerance.
    if opts is None:
        opts = Options(atol=1e-14, rtol=1e-14, rhs_reuse=False)

    if times is None:
        times = np.linspace(0, pulse_time, num=max(3, int(n_points)))

    if solver == magnus or solver == 'magnus':
        if picture == 'IP':
            o_change_of_picture = Qobj(
                sum(h_unperturbed), dims=dims)
        elif picture == 'RRF':
            if RRF_par is None:
                RRF_par = {'nu_RRF': 0, 'theta_RRF': 0, 'phi_RRF': 0}
            o_change_of_picture = RRF_operator(spin, RRF_par)
        else:
            raise ValueError("This value of argument 'picture' is not supported."
                             "Must be either 'IF' or 'RRF'.")
        h_total = Qobj(sum(h_unperturbed), dims=dims)
        result = magnus(h_total, Qobj(dm_initial), times,
                        order, spin, mode, o_change_of_picture)
        if return_allstates:
            raise NotImplementedError('Return all states not implemented with Magnus. '
                                      'Use mesolve instead.')
        else:
            dm_evolved = changed_picture(
                result.states[-1], o_change_of_picture, pulse_time, invert=True)
            # TODO: Problem of the conj
            # return dm_evolved.conj()
        return dm_evolved

    # Split into operator and time-dependent coefficient as per QuTiP scheme.
    h_perturbation = h_multiple_mode_pulse(spin, mode, t=0,
                                           factor_t_dependence=True)

    # Given that H = H0 + H1*f1(t) + H2*f1(t) + ...,
    # h_unscaled is of the form [H0, [H1, f1(t)], [H2, f2(t)], ...]
    # (refer to QuTiP's mesolve documentation for further detail)
    h_unscaled = h_unperturbed + h_perturbation

    if solver == mesolve or solver == 'mesolve':
        # Magnus expansion solver includes 2 pi factor in exponentiations;
        # scale Hamiltonians by this factor for `mesolve` for consistency.
        h_scaled = multiply_by_2pi(h_unscaled)

        result = mesolve(h_scaled, Qobj(dm_initial), times,
                         options=opts, progress_bar=display_progress)
        if return_allstates:
            return result.states
        else:
            # return last time step of density matrix evolution.
            return result.states[-1]


    elif isinstance(solver, str):
        raise ValueError(f'Invalid solver: {solver}')

    else:
        result = solver(h_unscaled, Qobj(dm_initial), times, options=opts)
        final_state = result.states[-1]
        # return last time step of density matrix evolution.
        return final_state


# Operator which generates a change of picture equivalent to moving to the rotating reference frame
# (RRF)
def RRF_operator(spin, RRF_par):
    """
    Returns the operator for the change of picture equivalent to moving to the RRF.

    Parameters
    ----------
    spin : NuclearSpin
        Spin under study.
            
    RRF_par : dict
        Specifies the properties of the rotating reference frame. The
        keys and values required to this argument are shown in the table
        below:

        |      key      |  value  |
        |      ---      |  -----  |
        |    'nu_RRF'   |  float  |
        |  'theta_RRF'  |  float  |
        |   'phi_RRF'   |  float  |
        
        where 'nu_RRF' is the frequency of rotation of the RRF (in MHz), while
        'theta_RRF' and 'phi_RRF' are the polar and azimuthal angles of the normal
        to the plane of rotation in the LAB frame (in radians).

    Returns
    -------
    An Observable object representing the operator which generates the change 
    to the RRF picture.
    """
    nu = RRF_par['nu_RRF']
    theta = RRF_par['theta_RRF']
    phi = RRF_par['phi_RRF']
    # The minus sign is to take care of the `Interaction picture' problem when rotating
    # the system
    RRF_o = -nu * (spin.I['z'] * np.cos(theta) +
                   spin.I['x'] * np.sin(theta) * np.cos(phi) +
                   spin.I['y'] * np.sin(theta) * np.sin(phi))
    return Qobj(RRF_o)


def plot_real_part_density_matrix(dm, many_spin_indexing=None,
                                  show=True, fig_dpi=400,
                                  save=False, xmin=None, xmax=None,
                                  ymin=None, ymax=None, show_legend=True,
                                  name='RealPartDensityMatrix', destination=''):
    """
    Generates a 3D histogram displaying the real part of the elements of the
    passed density matrix.

    Parameters
    ----------
    dm : Qobj / numpy array as a square matrix
        Density matrix to be plotted.

    many_spin_indexing : None or list
        If not None, the density matrix dm is interpreted as the state of 
        a many spins' system, and this parameter provides the list of the 
        dimensions of the subspaces of the full Hilbert space related to the
        individual nuclei of the system. 
        The ordering of the elements of many_spin_indexing should match that of
        the single spins' density matrices in their tensor product 
        resulting in dm. Default value is None.

    show : bool
        When False, the graph constructed by the function will not be
        displayed.
        Default value is True.

    fig_dpi : int
        Image quality of the figure when showing and saving. Useful for
        publications. Default set to very high value.

    save : bool
        When False, the plotted graph will not be saved on disk. When True,
        it will be saved with the name passed as name and in the directory
        passed as destination.
        Default value is False.

    xmin, xmax, ymin, ymax : float
        Set axis limits of the graph.

    name : string
        Name with which the graph will be saved.
        Default value is 'RealPartDensityMatrix'.

    destination : string
        Path of the directory where the graph will be saved (starting
        from the current directory). The name of the directory must
        be terminated with a slash /.
        Default value is the empty string (current directory).

    Action
    ------
    If show=True, draws a histogram on a 2-dimensional grid representing the
    density matrix, with the real part of each element indicated along the z
    axis. Blue bars indicate the positive matrix elements, red bars indicate the
    negative elements in absolute value.

    Returns
    -------
    An object of the class matplotlib.figure.Figure and an object of the class
    matplotlib.axis.Axis representing the figure built up by the function.

    """
    real_part = np.vectorize(np.real)
    if isinstance(dm, Qobj):
        data_array = real_part(dm)
    else:
        data_array = real_part(dm)

    # Create a figure for plotting the data as a 3D histogram.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create an X-Y mesh of the same dimension as the 2D data
    # You can think of this as the floor of the plot
    x_data, y_data = np.meshgrid(np.arange(data_array.shape[1]) + 0.25,
                                 np.arange(data_array.shape[0]) + 0.25)

    # Set width of the vertical bars
    dx = dy = 0.5

    # Flatten out the arrays so that they may be passed to "ax.bar3d".
    # Basically, ax.bar3d expects three one-dimensional arrays: x_data, y_data, z_data. The following
    # call boils down to picking one entry from each array and plotting a bar from (x_data[i],
    # y_data[i], 0) to (x_data[i], y_data[i], z_data[i]).
    x_data = x_data.flatten()
    y_data = y_data.flatten()
    z_data = data_array.flatten()

    bar_color = np.zeros(len(z_data), dtype=object)

    for i in range(len(z_data)):
        if z_data[i] < -1e-10:
            bar_color[i] = 'tab:red'
        else:
            bar_color[i] = 'tab:blue'

    ax.bar3d(x_data,
             y_data,
             np.zeros(len(z_data)),
             dx, dy, np.absolute(z_data), color=bar_color)

    d = data_array.shape[0]
    tick_label = []

    if not many_spin_indexing:
        many_spin_indexing = dm.dims[0]

    d_sub = many_spin_indexing
    n_sub = len(d_sub)
    m_dict = []

    for i in range(n_sub):
        m_dict.append({})
        for j in range(d_sub[i]):
            m_dict[i][j] = str(Fraction((d_sub[i] - 1) / 2 - j))

    for i in range(d):
        tick_label.append('>')

    for i in range(n_sub)[::-1]:
        d_downhill = int(np.prod(d_sub[i + 1:n_sub]))
        d_uphill = int(np.prod(d_sub[0:i]))

        for j in range(d_uphill):
            for k in range(d_sub[i]):
                for l in range(d_downhill):
                    comma = ', '
                    if j == n_sub - 1:
                        comma = ''
                    tick_label[j * d_sub[i] * d_downhill +
                               k * d_downhill + l] = m_dict[i][k] + \
                                                     comma + tick_label[j * d_sub[i] * d_downhill +
                                                                        k * d_downhill + l]

    for i in range(d):
        tick_label[i] = '|' + tick_label[i]

    ax.tick_params(axis='both', which='major', labelsize=6)
    xticks(np.arange(start=0.5, stop=data_array.shape[0] + 0.5), tick_label)
    yticks(np.arange(start=0.5, stop=data_array.shape[0] + 0.5), tick_label)

    ax.set_zlabel("Re(\N{GREEK SMALL LETTER RHO})")
    legend_elements = [Patch(facecolor='tab:blue', label='<m|\N{GREEK SMALL LETTER RHO}|m> > 0'),
                       Patch(facecolor='tab:red', label='<m|\N{GREEK SMALL LETTER RHO}|m> < 0')]
    if (show_legend):
        ax.legend(handles=legend_elements, loc='upper left')

    if (xmin is not None) and (xmax is not None):
        plt.xlim(xmin, xmax)
    if (ymin is not None) and (ymax is not None):
        plt.ylim(ymin, ymax)

    if save:
        plt.savefig(destination + name, dpi=fig_dpi)

    if show:
        plt.show()

    return fig, ax


def complex_phase_cmap():
    """
    Create a cyclic colormap for representing the phase of complex variables

    From QuTiP 4.0:
    https://qutip.org

    Returns
    -------
    cmap : A matplotlib linear segmented colormap.
    """
    cdict = {'blue': ((0.00, 0.0, 0.0),
                      (0.25, 0.0, 0.0),
                      (0.50, 1.0, 1.0),
                      (0.75, 1.0, 1.0),
                      (1.00, 0.0, 0.0)),
             'green': ((0.00, 0.0, 0.0),
                       (0.25, 1.0, 1.0),
                       (0.50, 0.0, 0.0),
                       (0.75, 1.0, 1.0),
                       (1.00, 0.0, 0.0)),
             'red': ((0.00, 1.0, 1.0),
                     (0.25, 0.5, 0.5),
                     (0.50, 0.0, 0.0),
                     (0.75, 0.0, 0.0),
                     (1.00, 1.0, 1.0))}
    
    cmap = clrs.LinearSegmentedColormap('phase_colormap', cdict, 256)
    return cmap


def plot_complex_density_matrix(dm, many_spin_indexing=None, show=True,
                                phase_limits=None, phi_label=r'$\phi$', show_legend=True, fig_dpi=400,
                                save_to="", figsize=None, labelsize=6, elev=45, azim=-15):
    """
    Generates a 3D histogram displaying the amplitude and phase (with colors)
    of the elements of the passed density matrix.

    Inspired by QuTiP 4.0's matrix_histogram_complex function.
    https://qutip.org

    Parameters
    ----------
    dm : Qobj
        Density matrix to be plotted.

    many_spin_indexing : None or list
        If not None, the density matrix dm is interpreted as the state of 
        a many spins' system, and this parameter provides the list of the 
        dimensions of the subspaces of the full Hilbert space related to the
        individual nuclei of the system. 
        The ordering of the elements of many_spin_indexing should match that of
        the single spins' density matrices in their tensor product resulting in dm.
        
        For example, a system of [spin-1/2 x spin-1 x spin-3/2] will correspond to:
        many_spin_indexing = [2, 3, 4]
        
        Default value is None.

    show : bool
        When False, the graph constructed by the function will not be
        displayed.
        Default value is True.

    phase_limits : list/array of two floats
        The phase-axis (colorbar) limits [min, max]

    phi_label : str
        Label for the legend for the angle of the complex number.

    show_legend : bool
        Show the legend for the complex angle.

    fig_dpi : int
        Image quality of the figure when showing and saving. Useful for
        publications. Default set to very high value.

    save_to : str
        If this is not the empty string, the plotted graph will be saved to the
        path ('directory/filename') described by this string.

        Default value is the empty string.

    figsize :  (float, float)
         Width, height in inches.
         Default value is the empty string.

    labelsize : int
         Default is 6

    (azim, elev) : (float, float)
         Angle of viewing for the 3D plot.
         Default is (45 deg, -15 deg)

    Action
    ------
    If show=True, draws a histogram on a 2-dimensional grid representing the
    density matrix, with phase sentivit data.

    Returns
    -------
    An object of the class matplotlib.figure.Figure and an object of the class 
    matplotlib.axis.Axis representing the figure built up by the function.

    """
    if not isinstance(dm, Qobj):
        raise TypeError('First argument must be an instance of Qobj!')

    if not many_spin_indexing:
        many_spin_indexing = dm.dims[0]

    dm = np.array(dm)

    # Create a figure for plotting the data as a 3D histogram.
    fig = plt.figure()
    if (figsize):
        fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Create an X-Y mesh of the same dimension as the 2D data
    # You can think of this as the floor of the plot
    x_data, y_data = np.meshgrid(np.arange(dm.shape[1]) + 0.25,
                                 np.arange(dm.shape[0]) + 0.25)

    # Set width of the vertical bars
    dx = dy = 0.5

    # Flatten out the arrays so that they may be passed to "ax.bar3d".
    # Basically, ax.bar3d expects three one-dimensional arrays: x_data, y_data, z_data.
    # The following call boils down to picking one entry from each array and plotting a bar from 
    # (x_data[i], y_data[i], 0) to (x_data[i], y_data[i], z_data[i]).
    x_data = x_data.flatten()
    y_data = y_data.flatten()
    z_data = dm.flatten()

    if phase_limits:  # check that limits is a list type
        phase_min = phase_limits[0]
        phase_max = phase_limits[1]
    else:
        phase_min = -np.pi
        phase_max = np.pi

    norm = clrs.Normalize(phase_min, phase_max)
    cmap = complex_phase_cmap()
    colors = cmap(norm(np.angle(z_data)))

    ax.bar3d(x_data, y_data, np.zeros(len(z_data)),
             dx, dy, np.absolute(z_data), color=colors, shade=True)
    ax.view_init(elev=elev, azim=azim)  # rotating the plot so the "diagonal" direction is more clear

    d = dm.shape[0]
    tick_label = []

    d_sub = many_spin_indexing
    n_sub = len(d_sub)
    m_dict = []  # dictionary of labels for the spin orientation "m"

    # For example, for a two spin-1/2 system:
    # m_dict = [{0: '1/2', 1:'-1/2'}, {0: '1/2', 1:'-1/2'}]
    for i in range(n_sub):
        m_dict.append({})
        for j in range(d_sub[i]):
            m_dict[i][j] = str(Fraction((d_sub[i] - 1) / 2 - j))

    for i in range(d):
        tick_label.append('>')

    for i in range(n_sub)[::-1]:
        d_downhill = int(np.prod(d_sub[i + 1:]))
        d_uphill = int(np.prod(d_sub[0:i]))

        for j in range(d_uphill):
            for k in range(d_sub[i]):
                for l in range(d_downhill):
                    comma = ', '
                    if j == n_sub - 1: comma = ''
                    tick_label[(j * d_sub[i] + k) * d_downhill + l] = (
                            m_dict[i][k] + comma + tick_label[(j * d_sub[i] + k) * d_downhill + l])

    for i in range(d):
        tick_label[i] = '|' + tick_label[i]

    ax.tick_params(axis='both', which='major', labelsize=labelsize)

    xticks(np.arange(start=0.5, stop=dm.shape[0] + 0.5), tick_label)
    yticks(np.arange(start=1., stop=dm.shape[0] + 1.), tick_label)
    if show_legend:
        cax, kw = clrbar.make_axes(ax, location='right', shrink=.75, pad=.06)
        cb = clrbar.ColorbarBase(cax, cmap=cmap, norm=norm)
        cb.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        cb.set_ticklabels(
            (r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'))
        cb.set_label(phi_label)

    if save_to != '':
        plt.savefig(save_to, dpi=fig_dpi)

    if show:
        plt.show()

    return fig, ax


def FID_signal(spin, h_unperturbed, dm, acquisition_time, T2=100, theta=0,
               phi=0, ref_freq=0, n_points=1000, pulse_mode=None,
               opts=None, display_progress=None):
    """ 
    Simulates the free induction decay signal (FID) measured after the shut-off
    of the electromagnetic pulse, once the evolved density matrix of the system,
    the time interval of acquisition, the relaxation time T2 and the direction
    of the detection coils are given.

    Parameters
    ----------
    spin : NuclearSpin
        Spin under study.

    h_unperturbed : List[Qobj]
        Unperturbed Hamiltonian of the system (in MHz). (Most likely an output
        of the 'nuclear_system_setup' function above)

    dm : Qobj
        Density matrix representing the state of the system at the beginning
        of the acquisition of the signal.

    acquisition_time : float
        Duration of the acquisition of the signal, expressed in
        microseconds.

    T2 : float, or
         iterable[float], or 
         function with signature (float) -> float, or
         iterable[function with signature (float) -> float]

        If float, characteristic time of relaxation of the component of the
        magnetization on the plane of detection vanishing, i.e., T2.
        If function, the decay envelope. 
        If iterable, total decay envelope will be product of decays in list.

        In units of microseconds. Default value is 100 (microseconds).

    theta, phi : float
        Polar and azimuthal angles which specify the normal to the
        plane of detection of the FID signal (in radians).
        Default values are theta=0, phi=0.

    ref_freq : float
        Specifies the frequency of rotation of the measurement apparatus
        with respect to the LAB system. (in MHz)
        Default value is 0.

    n_points : float
        The total number of samples for the signal.
        Default value is 1000.

    pulse_mode : pandas.DataFrame
        The user can decide to apply a pulse during the measurement of the FID.
        Although unusual, this is necessary for axion simulations.
        Refer to the argument 'mode' in the function evolve() for details about 
        this pulse_mode argument.

    display_progress: bool
        True will display a progress bar for the mesolve function.
        False will not display a progress bar.
        
    Action
    ------
    Samples the time interval [0, acquisition_time] with n_points points per
    microsecond.

    The FID signal is simulated under the assumption that it is directly related
    to the time-dependent component of the magnetization on the plane specified
    by (theta, phi) of the LAB system.

    Returns
    -------
    [0] : numpy.ndarray
        Vector of equally spaced sampled instants of time in the interval [0,
        acquisition_time] (in microseconds).

    [1] : numpy.ndarray
        FID signal evaluated at the discrete times reported in the first output
        (in arbitrary units). This is the expectation value of the spin in the
        direction defined by the angles (theta, phi) in the input.
    """

    times = np.linspace(start=0, stop=acquisition_time, num=n_points)

    decay_functions = []
    try:
        for d in T2:
            if not callable(d):  # T2 is a list of floats
                decay_functions.append(lambda t: np.exp(-t / d))
            else:  # T2 is a list of functions given by the user
                decay_functions.append(d)
    except TypeError:
        if not callable(T2):  # T2 is a float
            decay_functions.append(lambda t: np.exp(-t / T2))
        else:  # T2 is a function given by user
            decay_functions.append(T2)

    decay_array = []
    for decay_fun in decay_functions:
        decay_array.append(decay_fun(times))
    # decay_array is now a 2D array with shape (len(decay_functions), len(times))
    # Now, multiply all the decay functions together to make it into 1D array with same length as times
    decay_t = np.prod(np.array(decay_array), axis=0)

    # Define the direction of measurement
    Ix, Iy, Iz = spin.I['x'], spin.I['y'], spin.I['z']
    rot_y, rot_z = (-1j * theta * Iy), (-1j * phi * Iz)
    Ix_rotated = apply_exp_op(apply_exp_op(Ix, rot_y), rot_z)

    if pulse_mode is not None:
        # copying the method from function 'evolve()' above
        h_perturbation = h_multiple_mode_pulse(spin, pulse_mode, t=0,
                                               factor_t_dependence=True)
        hamiltonian = h_unperturbed + h_perturbation
    else:
        hamiltonian = h_unperturbed

    h_scaled = multiply_by_2pi(hamiltonian)

    # Measuring the expectation value of Ix rotated:
    if opts is None:
        opts = Options(atol=1e-14, rtol=1e-14, rhs_reuse=False)
    if not display_progress:
        display_progress = None  # qutip takes in a None instead of False for some reason (bad type check)

    result = mesolve(h_scaled, dm, times, e_ops=[Ix_rotated],
                     progress_bar=display_progress, options=opts)

    measurement_direction = np.exp(-1j * 2 * np.pi * ref_freq)
    fid = np.array(result.expect)[0] * decay_t * measurement_direction
    if np.max(fid) < 0.09:
        import warnings
        warnings.warn('Unreliable FID: Weak signal, check simulation!', stacklevel=0)

    return result.times, fid


def plot_real_part_FID_signal(
        times, FID, show=True, fig_dpi=400, save=False,
        name='FIDSignal', destination='', xlim=None, ylim=None, figure = None):
    """
    Plots the real part of the FID signal as a function of time.

    Parameters
    ----------
    time : array-like
        Sampled instants of time (in microseconds).

    FID : array-like
        Sampled FID values (in arbitrary units).

    show : bool
        When False, the graph constructed by the function will not be
        displayed.
        Default value is True.

    fig_dpi : int
        Image quality of the figure when showing and saving. Useful for
        publications. Default set to very high value.

    save : bool
        When False, the plotted graph will not be saved on disk. When True,
        it will be saved with the name passed as name and in the directory
        passed as destination.
        Default value is False.

    name : string
        Name with which the graph will be saved.
        Default value is 'FIDSignal'.

    destination : string
        Path of the directory where the graph will be saved (starting
        from the current directory). The name of the directory must
        be terminated with a slash /.
        Default value is the empty string (current directory).

    xlim: tuple
        x limits of plot

    ylim: tuple
        y limits of plot
        
    fig: plt.figure
        figure to plot FID signal on
        
        
    Action
    ------ 
    If show=True, generates a plot of the FID signal as a function of time.

    Returns
    -------
    An object of the class matplotlib.figure.Figure representing the figure
    built up by the function.
    """
    if figure is None:
        fig, ax = plt.subplots()
    else: 
        fig, ax = figure
    
    ax.plot(times, np.real(FID), label='Real part')
    ax.set_title('FID signal')
    ax.set_xlabel("time (\N{GREEK SMALL LETTER MU}s)")
    ax.set_ylabel("Real(FID) (a.u.)")

    
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if save:
        plt.savefig(destination + name, dpi=fig_dpi)
    if show:
        plt.show()

    return fig, ax


def fourier_transform_signal(signal, times, abs=False, padding=None):
    """
    Computes the Fourier transform of the passed time-dependent signal using
    the scipy library.

    Parameters
    ----------
    signal : array-like
        Sampled signal to be transformed in the frequency domain (in a.u.).
    times : array-like
        Sampled time domain (in microseconds).
    abs : Boolean 
        Whether to return the absolute value of the computer Fourier transform. 
    padding : Integer
        Amount of zero-padding to add to signal in the power of zeroes.

    Returns
    -------
    The frequency (in MHz) and fourier-transformed signal as a tuple (f, ft)
    """
    if padding is not None:
        # This code by Stephen Carr
        nt = len(times)  # number of points

        # zero pad the ends to "interpolate" in frequency domain
        zn = padding  # power of zeros
        N_z = 2 * (2 ** zn) + nt  # number of elements in padded array
        zero_pad = np.zeros(N_z, dtype=complex)

        M0_trunc_z = zero_pad
        num = 2 ** zn
        M0_trunc_z[num:(num + nt)] = signal

        # figure out the "frequency axis" after the FFT
        dt = times[2] - times[1]
        Fs = 1.0 / dt  # max frequency sampling

        # axis goes from - Fs / 2 to Fs / 2, with N_z steps
        freq_ax = ((np.linspace(0, N_z, N_z) - 1 / 2) / N_z - 1 / 2) * Fs

        M_fft = fftshift(fft(fftshift(M0_trunc_z)))
        if abs:
            M_fft = np.abs(M_fft)
        return freq_ax, M_fft

    ft = fftshift(fft(signal))
    freq = fftshift(fftfreq(len(times), (times[-1] - times[0]) / len(times)))
    if abs:
        ft = np.abs(ft)
    return freq, ft


# Finds out the phase responsible for the displacement of the real and imaginary parts of the Fourier
# spectrum of the FID with respect to the ideal absorptive/dispersive lorentzian shapes
def fourier_phase_shift(frequencies, fourier, fourier_neg=None, peak_frequency=0, int_domain_width=.5):
    """
    Computes the phase factor which must multiply the Fourier spectrum
    (`fourier`) in order to have the real and imaginary part of the adjusted
    spectrum showing the conventional dispersive/absorptive shapes at the peak
    specified by `peak_frequency`.

    Parameters
    ----------
    frequencies : array-like
        Sampled values of frequency (in MHz). 

    fourier : array-like
        Values of the Fourier transform of the signal (in a.u.) sampled
        at the frequencies passed as the first argument.

    fourier_neg : array-like
        Values of the Fourier transform of the signal (in a.u.) sampled at 
        the opposite of the frequencies passed as the first argument. 
        When fourier_neg is passed, it is possible to specify a peak_frequency 
        located in the range frequencies changed by sign.
        Default value is None.

    peak_frequency : float
        Position of the peak of interest in the Fourier spectrum.
        Default value is 0.

    int_domain_width : float
        Width of the domain (centered at peak_frequency) where the 
        real and imaginary parts of the Fourier spectrum will be integrated.
        Default value is .5.

    Action
    ------  
    The function integrates both the real and the imaginary parts of the
    spectrum over an interval of frequencies centered at peak_frequency whose
    width is given by int_domain_width. Then, it computes the phase shift.

    Returns
    -------
    A float representing the desired phase shift (in radians).
    """

    if fourier_neg is not None:
        fourier = np.concatenate((fourier, fourier_neg))
        frequencies = np.concatenate((frequencies, -frequencies))

    integration_domain = np.nonzero(np.isclose(
        frequencies, peak_frequency, atol=int_domain_width / 2))[0]

    int_real_fourier = 0
    int_imag_fourier = 0

    for i in integration_domain:
        int_real_fourier = int_real_fourier + np.real(fourier[i])
        int_imag_fourier = int_imag_fourier + np.imag(fourier[i])

    if np.absolute(int_real_fourier) < 1e-10:
        if int_imag_fourier > 0:
            return 0
        else:
            return np.pi

    atan = np.arctan(- int_imag_fourier / int_real_fourier)

    if int_real_fourier > 0:
        phase = atan + np.pi / 2
    else:
        phase = atan - np.pi / 2

    return phase


# If another set of data is passed as fourier_neg, the function plots a couple of graphs, with the
# one at the top interpreted as the NMR signal produced by a magnetization rotating counter-clockwise,
# the one at the bottom corresponding to the opposite sense of rotation
def plot_fourier_transform(
        frequencies, fourier, fourier_neg=None, square_modulus=False,
        xlim=None, ylim=None, scaling_factor=None, norm=True, fig_dpi=400,
        show=True, save=False, name='FTSignal', destination='', figure = None, my_label = ""):
    """
    Plots the Fourier transform of a signal as a function of the frequency.

    Parameters
    ----------
    frequencies : array-like
        Sampled values of frequency (in MHz).

    fourier : array-like
        Sampled values of the Fourier transform (in a.u.).

    fourier_neg : array-like
        Sampled values of the Fourier transform (in a.u.) evaluated
        at the frequencies in frequencies changed by sign.
        Default value is `None`.

    square_modulus : bool
        When True, makes the function plot the square modulus of
        the Fourier spectrum rather than the separate real and
        imaginary parts, which is the default option (by default,
        `square_modulus=False`).

    xlim (ylim) : 2-element iterable or `None`
        Lower and upper x-axis (y-axis) limits of the plot.
        When `None` uses `matplotlib` default.

    scaling_factor : float
        When it is not None, it specifies the scaling factor which
        multiplies the data to be plotted. 
        It applies simultaneously to all the plots in the resulting figure.

    norm : Boolean 
        Whether to normalize the fourier transform; i.e.,
        scale it such that its maximum value is 1.

    fig_dpi : int
        Image quality of the figure when showing and saving. Useful for
        publications. Default set to very high value.

    show : bool
        When False, the graph constructed by the function will not be
        displayed.
        Default value is `True`.

    save : bool
        When `False`, the plotted graph will not be saved on disk. When `True`,
        it will be saved with the name passed as name and in the directory
        passed as destination.
        Default value is False.

    name : string
        Name with which the graph will be saved.
        Default value is `'FTSignal'`.

    destination : string
        Path of the directory where the graph will be saved (starting from 
        the current directory). The name of the directory must be terminated 
        with a slash /.
        Default value is the empty string (current directory).
        
    figure : plt.subplot
        Plot to plot on the fourier transformed frequency

    Action
    ------
    Builds up a plot of the Fourier transform of the passed complex signal as a function of the frequency.
    If fourier_neg is different from None, two graphs are built up which
    represent respectively the Fourier spectra for counter-clockwise and
    clockwise rotation frequencies.

    If show=True, the figure is printed on screen.  

    Returns
    -------
    An object of the class matplotlib.figure.Figure and an object of the class 
    matplotlib.axis.Axis representing the figure 
    built up by the function.
    """
    fourier = np.array(fourier)
    frequencies = np.array(frequencies)

    if fourier_neg is None:
        n_plots = 1
        fourier_data = [fourier]
    else:
        n_plots = 2
        fourier_data = [fourier, fourier_neg]
        plot_title = ["Counter-clockwise precession", "Clockwise precession"]

    if norm:
        for i in range(n_plots):
            fourier_data[i] = (fourier_data[i] /
                               np.amax(np.abs(fourier_data[i])))

    if scaling_factor is not None:
        for i in range(n_plots):
            fourier_data[i] = scaling_factor * fourier_data[i]
    if figure is None:
        fig , ax = plt.subplots(n_plots, 1, sharey=True, gridspec_kw={'hspace': 0.5})
    else:
        fig = figure[0]
        ax = figure[1]
    if fourier_neg is None:
        ax = [ax]

    for i in range(n_plots):
        if not square_modulus:
            ax[i].plot(frequencies, np.real(fourier_data[i]), label='Real part ' + my_label)
            ax[i].plot(frequencies, np.imag(fourier_data[i]), label='Imaginary part ' + my_label)
        else:
            ax[i].plot(frequencies, np.abs(fourier_data[i]) ** 2,
                       label='Square modulus ' + my_label)

        if n_plots > 1:
            ax[i].title.set_text(plot_title[i])
        else:
            ax[i].set_title('Frequency Spectrum')

        ax[i].legend(loc='upper left')
        ax[i].set_xlabel("Frequency (MHz)")
        ax[i].set_ylabel("FT signal (a. u.)")

        if xlim is not None:
            ax[i].set_xlim(*xlim)

        if ylim is not None:
            ax[i].set_ylim(*ylim)

    if save:
        plt.savefig(destination + name, dpi=fig_dpi)

    if show:
        plt.show()

    return fig, ax


def _ed_evolve_solve_t(t, h, rho0, e_ops):
    """
    Helper function for `ed_evolve`; uses exact diagonalization to evolve 
    the given initial state rho0 by a time `t`. 

    Params
    ------
    t : float
        The time up to which to evolve.
    h : Qobj or List[Qobj]:
        The Hamiltonian describing the system in MHz.
    rho0 : Qobj
        The initial state of the system as a density matrix. 
    e_ops : List[Qobj]:
        List of operators for which to return the expectation values. 

    Returns
    ------
    The evolved density matrix at the time specified by `t,' and the expectation
    values of each operartor in `e_ops` at `t`. The latter is in the format
    [e_op1[t], e_op2[t], ..., e_opn[t]]. 
    """
    u1, d1, d1exp = exp_diagonalize(1j * 2 * np.pi * h * t)
    u2, d2, d2exp = exp_diagonalize(-1j * 2 * np.pi * h * t)

    rho_t = u1 * d1exp * u1.inv() * rho0 * u2 * d2exp * u2.inv()

    if e_ops == None:
        return rho_t

    exp = np.transpose([[expect(op, rho_t) for op in e_ops]])

    return rho_t, exp


def ed_evolve(h, rho0, spin, tlist, e_ops=[], state=True, fid=False, parallel=False,
              all_t=False, T2=100):
    """
    Evolve the given density matrix with the interactions given by the provided 
    Hamiltonian using exact diagonalization.

    ipyparallel must be present for Jupyter notebooks.

    Params
    ------
    h : Qobj or List[Qobj]:
        The Hamiltonian describing the system in MHz.
    rho0 : Qobj
        The initial state of the system as a density matrix. 
    spin : NuclearSpin
        The NuclearSpin object representing the system under study. 
    tlist : List[float]
        List of times at which the system will be evolved. 
    e_ops : List[Qobj]:
        List of operators for which to return the expectation values. 
    state : Boolean 
        Whether to return the density matrix at all. Default `True`.
    fid : Boolean
        Whether to return the free induction decay (FID) signal as 
        an expectation value. If True, appends FID signal to the end of 
        the `e_ops` expectation value list. 
    par : Boolean
        Whether to use QuTiP's parallel computing implementation `parallel_map` 
        to evolve the system.
    all_t : Boolean 
        Whether to return the density matrix and for all times in the
        evolution (as opposed to the last state)
    T2 : iterable[float or function with signature (float) -> float] or 
         float or 
         function with signature (float) -> float

        If float, characteristic time of relaxation of the component of the
        magnetization on the plane of detection vanishing, i.e., T2. It is
        measured in
        microseconds.

        If function, the decay envelope. 

        If iterable, total decay envelope will be product of decays in list.

        Default value is 100 (microseconds).

    Returns
    ------
    [0]: The density matrix at time `tlist[-1]` OR the evolved density matrix
        at times specified by `tlist`. 

    [1]: the expectation values of each operator in `e_ops` at the times in
        `tlist`. The latter is in the format `[[e_op1[t1], e_op1[t2], ...] , 
        [e_op2[t1], e_op2[t2]], ..., [e_opn[t1], e_opn[t2], ...]]`. 

    OR 

    The expectation values of each operator in `e_ops` at the times in `tlist`.
    """
    if type(h) is not Qobj and type(h) is list:
        h = Qobj(sum(h), dims=h[0].dims)
    if fid:
        e_ops.append(Qobj(np.array(spin.I['+']), dims=h.dims))

    decay_envelopes = []
    try:
        for d in T2:
            if not callable(d):  # T2 is a list of floats
                decay_envelopes.append(lambda t: np.exp(-t / d))
            else:  # T2 is a list of functions
                decay_envelopes.append(d)
    except TypeError:
        if not callable(T2):  # T2 is a float
            decay_envelopes.append(lambda t: np.exp(-t / T2))
        else:  # T2 is a function given by user
            decay_envelopes.append(T2)

    rho_t = []
    e_ops_t = []

    if parallel:
        # Check if Jupyter notebook to use QuTiP's Jupyter-optimized parallelization
        # Better method than calling 'get_ipython()' since this requires calling un un-imported function
        if 'ipykernel' in sys.modules:
            # make sure to have a running cluser:
            try:
                res = ipynb_parallel_map(_ed_evolve_solve_t,
                                         tlist, (h, rho0, e_ops), progress_bar=True)
            except OSError:
                raise OSError('Make sure to have a running cluster. ' +
                              'Try opening a new cmd and running ipcluster start.')

        else:
            res = parallel_map(_ed_evolve_solve_t, tlist,
                               (h, rho0, e_ops), progress_bar=True)

        for r, e in res:
            rho_t.append(r)
            e_ops_t.append(e)

        e_ops_t = np.concatenate(e_ops_t, axis=1)

    elif not e_ops:
        for t in tlist:
            rho_t.append(_ed_evolve_solve_t(t, h, rho0, None))
        e_ops_t = []

    else:
        e_ops_t = [[] for _ in range(len(e_ops))]
        for t in tqdm(tlist):
            rho, exp = _ed_evolve_solve_t(t, h, rho0, e_ops)
            e_ops_t = np.concatenate([e_ops_t, exp], axis=1)
            rho_t.append(rho)

    if fid:
        fid_exp = []
        fids = e_ops_t[-1]
        for i in trange(len(fids)):
            # Obtain total decay envelope at that time.
            envelope = 1
            for decay in decay_envelopes:
                # Different name to avoid bizarre variable scope bug
                envelope *= decay(tlist[i])
                # (can't have same name as iteration var in line 1117.)
            fid_exp.append(fids[i] * envelope)

        e_ops_t[-1] = fid_exp

    if not state:
        return e_ops_t

    if all_t:
        return rho_t, e_ops_t
    else:
        return rho_t[-1], e_ops_t


def apply_rot_pulse(rho, duration, rot_axis):
    """
    Apply a "pulse" to the given state by rotating the given state by  
    the given duration. I.e., transforms the density matrix by 
    `U = exp(- i * duration * rot_axis)`:
        `U * rho * U.dag()`

    Parameters:
    -----------
    rho : Qobj
        The density matrix of the state to apply the pulse to.
    duration : float
        The duration of the applied pulse as an angle in radians.
    rot_axis : Qobj
        Angular momentum operator for the corresponding axis of rotation. 

    Returns:
    --------
    The transformed density matrix as a Qobj. 
    """

    rot_op = (-1j * duration * rot_axis).expm()
    return apply_op(rho, rot_op)
    # return rho.transform((-1j * duration * rot_axis).expm())
