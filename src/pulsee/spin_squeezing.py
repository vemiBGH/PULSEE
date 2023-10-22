import numpy as np

import matplotlib.pyplot as plt
from qutip import tensor, spin_coherent

from pulsee.nuclear_spin import NuclearSpin, ManySpins
from pulsee.operators import calc_e_ops


class useful_sqz_ops:
    """
    Class that holds useful operators for spin squeezing calculations.

    Parameters
    ----------
    spin_system : NuclearSpin
    """

    def __init__(self, spin_system):
        # Global variable
        (self.Ix, self.Iy, self.Iz) = spin_system.spin_J_set()
        self.Ix2 = self.Ix ** 2
        self.Iy2 = self.Iy ** 2
        self.Iz2 = self.Iz ** 2

        self.Im = spin_system.I['-']
        self.Ip = spin_system.I['+']

        self.Ip2 = self.Ip ** 2
        self.Ip_2Iz = self.Ip * (2 * self.Iz + 1)

        self.IyIz = self.Iy * self.Iz
        self.IzIy = self.Iz * self.Iy

        # Store their average values
        self.avIx, self.avIy, self.avIz = 3 * [None]
        self.avIx2, self.avIy2, self.avIz2 = 3 * [None]
        self.avIm, self.avIp, self.avIp2, self.avIp_2Iz = 4 * [None]
        self.avIyIz, self.avIzIy = 2 * [None]

    def __repr__(self):
        return 'The class contains: Ix, Iy, Iz, Ix2, Iy2, ' \
               'Iz2, Im, Ip, Ip2, Ip_2Iz, IyIz, IzIy, ' \
               'and their average values, av[op].'


def CSS(spin_system, initial_state):
    """
    If a dictionary {'theta' : rad, 'phi' : rad} is passed, a spin coherent
    state is created. Can pass a list of dictionaries for a ManySpins system
    to create a tensor product state.

    Wrapper for qutip spin_coherent.

    Parameters
    ----------
    spin_system : NuclearSpin
    initial_state : Qobj

    Returns
    -------
    [0]: Qobj
        The density matrix representing the state of the system at time t=0,
        initialised according to initial_state.
    """
    if 'theta' in np.all(initial_state) and 'phi' in np.all(initial_state):
        if isinstance(spin_system, ManySpins):
            assert isinstance(initial_state, list), 'The initial_state for CSS for ManySpins must be specified ' \
                                                    'as a list with theta and phi specified for each spin.'
            dm = spin_coherent(spin_system.spin[0].I['I'], initial_state[0]['theta'], initial_state[0]['phi'],
                               type='dm')

            for i in range(1, spin_system.n_spins):
                dm = tensor(dm, spin_coherent(spin_system.spin[i].I['I'], initial_state[i]['theta'],
                                              initial_state[i]['phi'], type='dm'))
        else:
            assert isinstance(spin_system, NuclearSpin), 'Must give a NuclearSpin type.'
            assert isinstance(initial_state, dict), 'The initial_state for CSS must be given as a dictionary, ' \
                                                    'with theta and phi specified'

            dm = spin_coherent(spin_system.I['I'], initial_state['theta'], initial_state['phi'], type='dm')
    else:
        raise ValueError("CSS: Please check that both angles, theta and phi, are given for all the ManySpins.")

    return dm


def populate_averge_values(dms, sqz_ops):
    """
    Populates the class useful_sqz_ops with the average value of the operators in the states
    [density matrices] given.

    Parameters
    ----------
    dms : list of Qobj
    sqz_ops : useful_sqz_ops

    Returns
    -------
    Populated class useful_sqz_ops
    """
    res = calc_e_ops(dms, [sqz_ops.Ix, sqz_ops.Iy, sqz_ops.Iz,
                           sqz_ops.Ix2, sqz_ops.Iy2, sqz_ops.Iz2,
                           sqz_ops.Im, sqz_ops.Ip, sqz_ops.Ip2, sqz_ops.Ip_2Iz,
                           sqz_ops.IyIz, sqz_ops.IzIy])

    sqz_ops.avIx, sqz_ops.avIy, sqz_ops.avIz = (res[0], res[1], res[2])
    sqz_ops.avIx2, sqz_ops.avIy2, sqz_ops.avIz2 = (res[3], res[4], res[5])
    sqz_ops.avIm, sqz_ops.avIp, sqz_ops.avIp2, sqz_ops.avIp_2Iz = (res[6], res[7],
                                                                   res[8], res[9])
    sqz_ops.avIyIz, sqz_ops.avIzIy = (res[10], res[11])

    return sqz_ops


def calc_squeez_param(sqz_ops, I, xi_sq=False, return_av_spher=False):
    """
    Calculates the generalized squeezing paramter and the squeezing angle.

    Parameters
    ----------
    sqz_ops : useful_sqz_ops
        the class useful_sqz_ops

    I : int
        quantum number of the system

    xi_sq : bool
        If false, it won't square the squeezing paramter

    return_av_spher : bool
        Return the average value of the spin operators in spherical coords

    Returns
    -------
    [0] : xi (squeezing parameter)
    [1] : alpha (squeezing angle)
    Or
    [0] : xi (squeezing parameter)
    [1] : alpha (squeezing angle)
    [2] : Jn_1 average value of the operator in spherical coords
    [3] : Jn_2
    [4] : Jn_3
    """

    Jx = sqz_ops.avIx
    Jy = sqz_ops.avIy
    Jz = sqz_ops.avIz

    Jz2 = sqz_ops.avIz2

    Jp2 = sqz_ops.avIp2
    Jp_2Jz = sqz_ops.avIp_2Iz

    # Working in spherical coordinates
    r = np.sqrt(abs(Jx) ** 2 + abs(Jy) ** 2)
    # R = np.sqrt(abs(Jx) ** 2 + abs(Jy) ** 2 + abs(Jz) ** 2)

    th = np.arctan2(np.array(r, dtype=np.float64), np.array(Jz, dtype=np.float64))
    phi = np.arctan2(np.array(Jy, dtype=np.float64), np.array(Jx, dtype=np.float64))

    Jn_1 = - Jx * np.sin(phi) + Jy * np.cos(phi)
    Jn_2 = - Jx * np.cos(th) * np.cos(phi) - Jy * np.sin(phi) * np.cos(th) + Jz * np.sin(th)
    Jn_3 = Jx * np.sin(th) * np.cos(phi) + Jy * np.sin(phi) * np.sin(th) + Jz * np.cos(th)

    A = (1 / 2) * (np.sin(th) ** 2 * (I * (I + 1) - 3 * Jz2) - (1 + np.cos(th) ** 2) * (
            Jp2.imag * np.sin(2 * phi) + Jp2.real * np.cos(2 * phi)) + \
                   np.sin(2 * th) * (Jp_2Jz.imag * np.sin(phi) + Jp_2Jz.real * np.cos(phi)))

    C = I * (I + 1) - Jz2 - Jp2.imag * np.sin(2 * phi) - Jp2.real * np.cos(2 * phi) - A

    B = np.cos(th) * (Jp2.real * np.sin(2 * phi) - Jp2.imag * np.cos(2 * phi)) + np.sin(th) * (
            -Jp_2Jz.real * np.sin(phi) + Jp_2Jz.imag * np.cos(phi))

    if xi_sq:
        xi = (C - np.sqrt(A ** 2 + B ** 2)) / I
    else:
        xi = np.sqrt(C - np.sqrt(A ** 2 + B ** 2)) / np.sqrt(I)

    alpha = (1 / 2) * np.arctan(B / A)

    if return_av_spher:
        return xi, alpha, Jn_1, Jn_2, Jn_3
    else:
        return xi, alpha


def plot_values(vals, times, num_plots, axis_scaler, title='Mean values of magnetization',
                x_label='Time (MHz)', y_label='Mangetization (Arb.)', labels=['I_x', 'I_y', 'I_z', 'I_T'],
                colors=['b', 'r', 'g', 'y'], put_brackets=True):
    """
    Helper plotting function for the squeezing module
    Parameters
    ----------
    vals: list of list of average operators
        E.g. signal of Ix at each time step.

    times: list

    num_plots : int or list
        How many plots to plot

    axis_scaler : float
        Usuful to plot in terms of natural frequency of the system

    title : str

    x_label : str

    y_label : str

    labels : list
        List of labels for each average value plotted

    colors : list
        list of colors in string format

    put_brackets : bool
        Whether to put average value brackets

    """

    brackets = ["\langle ", " \\rangle"]
    times = times / axis_scaler

    if (isinstance(num_plots, int)):
        fig, axs = plt.subplots(num_plots)

        for i in range(len(vals)):
            if put_brackets:
                axs.plot(times, vals[i], colors[i], label=r"${} {} {}$".format(brackets[0], labels[i], brackets[1]))
            else:
                axs.plot(times, vals[i], colors[i], label=r"${}$".format(labels[i]))
        axs.set_xlabel(x_label)
        axs.set_ylabel(y_label)

    elif (num_plots[0] == 1 or num_plots[1] == 1):
        fig, axs = plt.subplots(*num_plots)
        for i in range(max(num_plots[0], num_plots[1])):
            if put_brackets:
                axs[i].plot(times, vals[i], colors[i], label=r"${} {} {}$".format(brackets[0], labels[i], brackets[1]))
            else:
                axs[i].plot(times, vals[i], colors[i], label=r"${}$".format(labels[i]))
        axs.flat[1].set_xlabel(x_label)
        axs.flat[0].set_ylabel(y_label)
    else:
        fig, axs = plt.subplots(*num_plots, sharey=False, sharex=True)
        cnt = 0
        for i in range(num_plots[0]):
            for j in range(num_plots[1]):
                if put_brackets:
                    axs[i, j].plot(times, vals[cnt], colors[cnt],
                                   label=r"${} {} {}$".format(brackets[0], labels[cnt], brackets[1]))
                else:
                    axs[i, j].plot(times, vals[cnt], colors[cnt], label=r"${}$".format(labels[cnt]))
                cnt += 1
        axs.flat[2].set_xlabel(x_label)
        axs.flat[0].set_ylabel(y_label)
    fig.suptitle(title)
    fig.legend()

    plt.show()