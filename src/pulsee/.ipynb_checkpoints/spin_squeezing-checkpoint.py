import matplotlib.pyplot as plt
import numpy as np
from qutip import Qobj, spin_coherent, tensor

from .nuclear_spin import ManySpins, NuclearSpin
from .operators import calc_e_ops


class UsefulSqzOps:
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


def coherent_spin_state(spin_system : NuclearSpin, initial_state: list[dict]) -> Qobj:
    """
    If a dictionary {'theta' : rad, 'phi' : rad} is passed, a spin coherent
    state is created. Can pass a list of dictionaries for a ManySpins system
    to create a tensor product state.

    Wrapper for qutip spin_coherent.

    Parameters
    ----------
    spin_system : NuclearSpin

    initial_state : list of dictionaries. Each dictionary must have keys
    'theta' and 'phi'.

    Returns
    -------
    [0]: Qobj
        The density matrix representing the state of the system at time t=0,
        initialised according to initial_state.
    """
    for d in initial_state:
        if ('theta' not in d.keys()) or ('phi' not in d.keys()):
            raise ValueError("Please check that both 'theta' and 'phi' are given for all the spins.")

    if isinstance(spin_system, NuclearSpin) and not isinstance(spin_system, ManySpins):
        assert len(initial_state) == 1, "length of `initial_state` should be 1 since `spin_system` only has 1 spin!"
        dm = spin_coherent(spin_system.I['I'], initial_state[0]['theta'], initial_state[0]['phi'], type='dm')
        return dm

    assert isinstance(spin_system, ManySpins), "Not a valid type of `spin_system`!"
    assert len(initial_state) == spin_system.n_spins, "Length of `initial_state` must match the number of spins!"

    dm = spin_coherent(spin_system.spins[0].I['I'], initial_state[0]['theta'], initial_state[0]['phi'], type='dm')
    for i in range(1, spin_system.n_spins):
        dm = tensor(dm, spin_coherent(spin_system.spins[i].I['I'], initial_state[i]['theta'],
                                      initial_state[i]['phi'], type='dm'))
    return dm


def populate_averge_values(dms : list[Qobj], sqz_ops : UsefulSqzOps):
    """
    Populates the class useful_sqz_ops with the average value of the operators in the states
    [density matrices] given.

    Parameters
    ----------
    dms : list of Qobj
    sqz_ops : UsefulSqzOps

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


def calc_squeez_param(sqz_ops : UsefulSqzOps, I : int, xi_sq : bool =False, return_av_sphere : bool =False) -> tuple:
    """
    Calculates the generalized squeezing parameter and the squeezing angle.

    Parameters
    ----------
    sqz_ops : UsefulSqzOps
        the class useful_sqz_ops

    I : int
        quantum number of the system

    xi_sq : bool
        If false, it won't square the squeezing parameter

    return_av_sphere : bool
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
            Jp2.imag * np.sin(2 * phi) + Jp2.real * np.cos(2 * phi)) +
                   np.sin(2 * th) * (Jp_2Jz.imag * np.sin(phi) + Jp_2Jz.real * np.cos(phi)))

    C = I * (I + 1) - Jz2 - Jp2.imag * np.sin(2 * phi) - Jp2.real * np.cos(2 * phi) - A

    B = np.cos(th) * (Jp2.real * np.sin(2 * phi) - Jp2.imag * np.cos(2 * phi)) + np.sin(th) * (
            -Jp_2Jz.real * np.sin(phi) + Jp_2Jz.imag * np.cos(phi))

    if xi_sq:
        xi = (C - np.sqrt(A ** 2 + B ** 2)) / I
    else:
        xi = np.sqrt(C - np.sqrt(A ** 2 + B ** 2)) / np.sqrt(I)

    alpha = (1 / 2) * np.arctan(B / A)

    if return_av_sphere:
        return xi, alpha, Jn_1, Jn_2, Jn_3
    else:
        return xi, alpha


def plot_values(vals, times, num_plots, axis_scaler, title='Mean values of magnetization',
                x_label='Time (MHz)', y_label='Magnetization (Arb.)', labels=None,
                colors=None, put_brackets=True):
    """
    Helper plotting function for the squeezing module
    Parameters
    ----------
    vals: list of list of average operators
        E.g. signal of Ix at each time step.

    times: numpy array

    num_plots : int or list
        How many plots to plot

    axis_scaler : float
        Useful to plot in terms of natural frequency of the system

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

    if colors is None:
        colors = ['b', 'r', 'g', 'y']
    if labels is None:
        labels = ['I_x', 'I_y', 'I_z', 'I_T']

    brackets = ["\langle ", " \\rangle"]
    times = times / axis_scaler

    if isinstance(num_plots, int):
        fig, axs = plt.subplots(num_plots)

        # axis is a list here! Maybe meant axs[i] ?
        for i in range(len(vals)):
            if put_brackets:
                axs.plot(times, vals[i], colors[i], label=r"${} {} {}$".format(brackets[0], labels[i], brackets[1]))
            else:
                axs.plot(times, vals[i], colors[i], label=r"${}$".format(labels[i]))
        axs.set_xlabel(x_label)
        axs.set_ylabel(y_label)

    elif isinstance(num_plots, list) and (num_plots[0] == 1 or num_plots[1] == 1):
        fig, axs = plt.subplots(*num_plots)
        for i in range(max(num_plots[0], num_plots[1])):
            if put_brackets:
                axs[i].plot(times, vals[i], colors[i], label=r"${} {} {}$".format(brackets[0], labels[i], brackets[1]))
            else:
                axs[i].plot(times, vals[i], colors[i], label=r"${}$".format(labels[i]))
        axs.flat[1].set_xlabel(x_label)
        axs.flat[0].set_ylabel(y_label)

    elif isinstance(num_plots, list):
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

    else:
        raise TypeError('`num_plot` should either be type int or list!')

    fig.suptitle(title)
    fig.legend()

    plt.show()
