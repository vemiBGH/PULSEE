from fractions import Fraction

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colorbar as clrbar, colors as clrs
from matplotlib.patches import Patch
from qutip import Qobj
# import proplot as pplt



def plot_power_absorption_spectrum(frequencies : np.ndarray, intensities : np.ndarray, show : bool =True,
                                   xlim : list[float, float] | None =None, ylim : list[float, float] | None =None, 
                                   fig_dpi : int =400, save : bool =False, name : str ='PowerAbsorptionSpectrum', 
                                   destination : str ='') -> plt.Figure:
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
    If show=True, generates a graph with the frequencies of transition on the
    x-axis and the corresponding intensities on the y-axis.

    Returns
    -------
    An object of the class matplotlib.figure.Figure representing the figure
    built up by the function.
    """
    fig = plt.figure()

    plt.vlines(frequencies, 0, intensities, colors="b")
    plt.xlabel("\N{GREEK SMALL LETTER NU} (MHz)")
    plt.ylabel("Power absorption (a.u.)")

    if xlim is not None:
        plt.xlim(left=xlim[0], right=xlim[1])
    if ylim is not None:
        plt.xlim(left=ylim[0], right=ylim[1])
    if save:
        plt.savefig(destination + name, dpi=fig_dpi)
    if show:
        plt.show()

    return fig


def plot_real_part_density_matrix(dm : Qobj | np.ndarray, many_spin_indexing : list | None =None,
                                  show : bool =True, fig_dpi : int=400,
                                  save : bool =False, xmin : float =None, xmax : float =None,
                                  ymin : float =None, ymax : float =None, show_legend : bool =True,
                                  name : str ='RealPartDensityMatrix', destination : str ='', label_size : float | None =None) -> tuple[plt.Figure, plt.Axes]:
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

    show_legend : float
        Whether to show the phase legend or not.

    name : string
        Name with which the graph will be saved.
        Default value is 'RealPartDensityMatrix'.

    destination : string
        Path of the directory where the graph will be saved (starting
        from the current directory). The name of the directory must
        be terminated with a slash /.
        Default value is the empty string (current directory).

    label_size : float
        Font size of the indices labels

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
    if not isinstance(dm, Qobj):
        raise TypeError("The matrix must be an instance of Qobj!")

    if many_spin_indexing is None:
        many_spin_indexing = dm.dims[0]

    real_part = np.vectorize(np.real)
    dmr = real_part(dm.full())

    # Create a figure for plotting the data as a 3D histogram.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Create an X-Y mesh of the same dimension as the 2D data
    # You can think of this as the floor of the plot
    x_data, y_data = np.meshgrid(np.arange(dmr.shape[1]) + 0.25, np.arange(dmr.shape[0]) + 0.25)

    # Set width of the vertical bars
    dx = dy = 0.5

    # Flatten out the arrays so that they may be passed to "ax.bar3d".
    # Basically, ax.bar3d expects three one-dimensional arrays: x_data, y_data, z_data. The following
    # call boils down to picking one entry from each array and plotting a bar from (x_data[i],
    # y_data[i], 0) to (x_data[i], y_data[i], z_data[i]).
    x_data = x_data.flatten()
    y_data = y_data.flatten()
    z_data = dmr.flatten()
    bar_color = np.zeros(len(z_data), dtype=object)

    for i in range(len(z_data)):
        if z_data[i] < -1e-10:
            bar_color[i] = "tab:red"
        else:
            bar_color[i] = "tab:blue"

    ax.bar3d(x_data, y_data, np.zeros(len(z_data)), dx, dy, np.absolute(z_data), color=bar_color)

    label_indices(ax, dm, label_size, many_spin_indexing)

    ax.set_zlabel("Re(\N{GREEK SMALL LETTER RHO})")
    legend_elements = [
        Patch(facecolor="tab:blue", label="<m|\N{GREEK SMALL LETTER RHO}|m> > 0"),
        Patch(facecolor="tab:red", label="<m|\N{GREEK SMALL LETTER RHO}|m> < 0"),
    ]
    if show_legend:
        ax.legend(handles=legend_elements, loc="upper left")

    if (xmin is not None) and (xmax is not None):
        plt.xlim(xmin, xmax)
    if (ymin is not None) and (ymax is not None):
        plt.ylim(ymin, ymax)

    if save:
        plt.savefig(destination + name, dpi=fig_dpi)

    if show:
        plt.show()

    return fig, ax


def complex_phase_cmap() -> clrs.LinearSegmentedColormap:
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


def plot_complex_density_matrix(
        dm : Qobj, many_spin_indexing : list | None = None,
        show : bool = True,
        phase_limits : list | np.ndarray | None = None,
        phi_label : str = r'$\phi$',
        show_legend : bool = True,
        fig_dpi : int = 400,
        save_to : str = "",
        fig_size : tuple[float, float] | None =None,
        label_size : int =6,
        label_qubit : bool = False,
        view_angle : tuple[float] =(45, -15),
        zlim : tuple[float, float] | None = None
) -> tuple[plt.Figure, plt.Axes]:

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

    show_legend : bool
        Show the legend for the complex angle.

    fig_dpi : int
        Image quality of the figure when showing and saving. Useful for
        publications. Default set to very high value.

    save_to : str
        If this is not the empty string, the plotted graph will be saved to the
        path ('directory/filename') described by this string.

        Default value is the empty string.

    fig_size :  (float, float)
         Width, height in inches.
         Default value is the empty string.

    label_size : int
         Default is 6

    label_qubit : bool
        Whether to show the labels in the qubit convention:
        ex) |01> as opposed to |1/2, -1/2>.
        Default is False

    view_angle : (float, float)
         A tuple of (azimuthal, elevation) viewing angles for the 3D plot.
         Default is (45 deg, -15 deg)
         
    zlim : (int, int)
        The z axis limits of the plot.

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
        raise TypeError("The matrix must be an instance of Qobj!")

    if many_spin_indexing is None:
        many_spin_indexing = dm.dims[0]

    dm = dm.full()

    n = np.size(dm)
    # Create an X-Y mesh of the same dimension as the 2D data. You can think of this as the floor of the plot
    xpos, ypos = np.meshgrid(range(dm.shape[0]), range(dm.shape[1]))
    xpos = xpos.T.flatten() - 0.5
    ypos = ypos.T.flatten() - 0.5
    zpos = np.zeros(n)
    # Set width of the vertical bars
    dx = dy = 0.5 * np.ones(n)
    dm_data = dm.flatten()
    dz = np.abs(dm_data)

    # make small numbers real, to avoid random colors
    idx, = np.where(abs(dm_data) < 0.001)
    dm_data[idx] = abs(dm_data[idx])

    if phase_limits:  # check that limits is a list type
        phase_min = phase_limits[0]
        phase_max = phase_limits[1]
    else:
        phase_min = -np.pi
        phase_max = np.pi

    norm = clrs.Normalize(phase_min, phase_max)
    # cmap = pplt.Colormap('vikO', shift=-90)  # Using 'VikO' colormap from ProPlot
    # cmap = plt.get_cmap('twilight_shifted')
    cmap = rotate_colormap(plt.get_cmap('twilight'), angle=90, flip=True)
    colors = cmap(norm(np.angle(dm_data)))

    # Create a figure for plotting the data as a 3D histogram.
    fig = plt.figure(constrained_layout=False)
    if fig_size:
        fig.set_size_inches(fig_size)

    ax = fig.add_subplot(111, projection="3d")
    if zlim is not None:
        ax.set_zlim(zlim)
    elif label_qubit:  # To display as figure in a paper.
        ax.set_zlim(0, 1)
        ax.set_zticks([0, 0.5, 1], [0, 0.5, 1], fontsize=label_size, verticalalignment='center')
    # Adjust the z tick label locations to they line up better with the ticks
    ax.tick_params('z', pad=0)

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, shade=True)
    # TODO: change light source? Make color more consistent between elements
    ax.view_init(elev=view_angle[0], azim=view_angle[1])  # rotating the plot so the "diagonal" direction is more clear
    if label_qubit:
        label_qubit_indices(ax, label_size, xpos, ypos)
    else:
        label_indices(ax, dm, label_size, many_spin_indexing)

    if show_legend:
        cax, kw = clrbar.make_axes(ax, location="right", shrink=0.75, pad=0.06)
        cb = clrbar.ColorbarBase(cax, cmap=cmap, norm=norm)
        cb.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        cb.set_ticklabels((r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"))
        cb.set_label("Phase")

    if save_to != "":
        plt.savefig(save_to, dpi=fig_dpi, bbox_inches='tight')

    if show:
        plt.show()

    return fig, ax


def rotate_colormap(
        cmap: matplotlib.colors.Colormap,
        angle: float,
        flip: bool = False) -> matplotlib.colors.Colormap:
    """
    Helper function for `plot_complex_density_matrix`.

    Parameters
    ----------
    cmap: Colormap
        The colormap class to be shifted.
    angle: float
        IN DEGREES!
    flip: bool
        Whether to flip the color wheel. Note the flip is done AFTER the rotation.

    Returns
    -------
    a newly shifted colormap. Note that the flip is done AFTER the rotation.
    """
    n = 256
    nums = np.linspace(0, 1, n)
    shifted_nums = np.roll(nums, int(n * angle / 360))
    if flip:
        shifted_nums = np.flip(shifted_nums)
    shifted_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(f"{cmap.name}_new", cmap(shifted_nums))
    return shifted_cmap


def label_indices(
        ax: plt.Axes,
        dm: Qobj,
        label_size: float,
        many_spin_indexing: list[int]):
    """
    Helper function for `plot_complex_density_matrix` and `plot_real_part_density_matrix`.
    """
    # x & y axex tick labelling:
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
        tick_label.append(r"$\rangle$")
    for i in range(n_sub)[::-1]:
        d_downhill = int(np.prod(d_sub[i + 1:]))
        d_uphill = int(np.prod(d_sub[0:i]))

        for j in range(d_uphill):
            for k in range(d_sub[i]):
                for l in range(d_downhill):
                    comma = ", "
                    if j == n_sub - 1:
                        comma = ""
                    tick_label[(j * d_sub[i] + k) * d_downhill + l] = (
                            m_dict[i][k] + comma + tick_label[(j * d_sub[i] + k) * d_downhill + l]
                    )
    for i in range(d):
        tick_label[i] = "|" + tick_label[i]

    ax.tick_params(axis="both", which="major", labelsize=label_size)
    tick_locations = np.arange(start=0.5, stop=dm.shape[0] + 0.5)
    ax.set(xticks=tick_locations - 1.5, xticklabels=tick_label,
           yticks=tick_locations - 0.5, yticklabels=tick_label)


def label_qubit_indices(ax: plt.Axes, label_size: float, xpos, ypos):
    # convert the 16 coordinates into 4
    xpos = np.sort(np.unique(np.array(xpos))) + 0.25
    ypos = np.sort(np.unique(np.array(ypos))) + 0.25
    # adapted from qutip's `matrix_histogram_complex`
    labels = [r"$|$00$\rangle$", r"$|$01$\rangle$", r"$|$10$\rangle$", r"$|$11$\rangle$"]
    # ax.axes.xaxis.set_major_locator(plt.IndexLocator(1, -0.5))
    # ax.set_xticklabels(labels)
    # ax.tick_params(axis='x', labelsize=label_size)
    #
    # ax.axes.yaxis.set_major_locator(plt.IndexLocator(1, -0.5))
    # ax.set_yticklabels(labels)
    # ax.tick_params(axis='y', labelsize=label_size)

    ax.set_xticks(xpos, labels=labels, fontsize=label_size, va='bottom')
    ax.set_yticks(ypos, labels=labels, fontsize=label_size)
    ax.tick_params(axis='x', direction='in', pad=7)
    ax.tick_params(axis='y', direction='in', pad=-3)


def plot_real_part_FID_signal(
        times : np.ndarray, FID : np.ndarray, show : bool =True, fig_dpi : int =400, save : bool =False,
        name : str ='FIDSignal', destination : str ='', xlim : tuple | None =None, ylim : tuple | None =None, 
        figure : plt.Figure | None =None) -> plt.Figure:
    """
    Plots the real part of the FID signal as a function of time.

    Parameters
    ----------
    times : array-like
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

    figure: plt.figure
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

    ax.plot(times, np.real(FID), label="Real part")
    ax.set_title("FID signal")
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


# If another set of data is passed as fourier_neg, the function plots a couple of graphs, with the
# one at the top interpreted as the NMR signal produced by a magnetization rotating counter-clockwise,
# the one at the bottom corresponding to the opposite sense of rotation
def plot_fourier_transform(
        frequencies : np.ndarray, fourier : np.ndarray, fourier_neg : np.ndarray =None, square_modulus : bool =False,
        xlim : list[float, float] =None, ylim : list[float, float] =None, scaling_factor : float =None, norm : bool =True, 
        fig_dpi : int =400, show : bool =True, save : bool =False, name : str ='FTSignal', destination : str ='', 
        figure : tuple[plt.Figure, plt.Axes] | None =None, my_label : str ="") -> tuple[plt.Figure, plt.Axes]:
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

    xlim, ylim : 2-element iterable or `None`
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
        plot_title = "Frequency Spectrum"
    else:
        n_plots = 2
        fourier_data = [fourier, fourier_neg]
        plot_title = ["Counter-clockwise precession", "Clockwise precession"]

    if norm:
        for i in range(n_plots):
            fourier_data[i] = fourier_data[i] / np.amax(np.abs(fourier_data[i]))

    if scaling_factor is not None:
        for i in range(n_plots):
            fourier_data[i] = scaling_factor * fourier_data[i]
    if figure is None:
        fig, ax = plt.subplots(n_plots, 1, sharey=True, gridspec_kw={"hspace": 0.5})
    else:
        fig = figure[0]
        ax = figure[1]
    if fourier_neg is None:
        ax = [ax]

    for i in range(n_plots):
        if not square_modulus:
            ax[i].plot(frequencies, np.real(fourier_data[i]), label="Real part " + my_label)
            ax[i].plot(frequencies, np.imag(fourier_data[i]), label="Imaginary part " + my_label)
        else:
            ax[i].plot(frequencies, np.abs(fourier_data[i]) ** 2, label="Square modulus " + my_label)

        if n_plots > 1:
            ax[i].title.set_text(plot_title[i])
        else:
            ax[i].set_title(plot_title)

        ax[i].legend(loc="upper left")
        ax[i].set_xlabel("Frequency (MHz)")
        ax[i].set_ylabel("FT signal (a.u.)")

        if xlim is not None:
            ax[i].set_xlim(*xlim)

        if ylim is not None:
            ax[i].set_ylim(*ylim)

    if save:
        plt.savefig(destination + name, dpi=fig_dpi)

    if show:
        plt.show()

    return fig, ax
