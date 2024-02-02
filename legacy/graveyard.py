# class NuclearSpin:
# '''Depreciated:'''

# def raising_operator(self):
#     """
#     Returns an Operator object representing the raising operator I+ of the spin,
#      expressing its matrix attribute with respect to the basis of the eigenstates of Iz.
#     """
#     I_raising = np.zeros(self.shape)
#     for m in range(self.d):
#         for n in range(self.d):
#             if n - m == 1:
#                 I_raising[m, n] = np.sqrt(self.quantum_number*(self.quantum_number+1) - (self.quantum_number-n)*(self.quantum_number-n + 1))
#     return Qobj(I_raising, dims=self.dims)
#
# def lowering_operator(self):
#     """
#     Returns an Operator object representing the lowering operator I- of the spin,
#     expressing its matrix attribute with respect to the basis of the eigenstates of Iz.
#     """
#     I_lowering = np.zeros(self.shape)
#     for m in range(self.d):
#         for n in range(self.d):
#             if n - m == -1:
#                 I_lowering[m, n] = np.sqrt(self.quantum_number*(self.quantum_number+1) - (self.quantum_number-n)*(self.quantum_number-n - 1))
#     return Qobj(I_lowering, dims=self.dims)

# def cartesian_operator(self):
#     """
#     Returns a list of 3 Observable objects representing in the order the x, y and z
#     components of the spin. The first two are built up exploiting their relation with
#      the raising and lowering operators (see the formulas above), while the third is
#      simply expressed in its diagonal form, since the chosen basis of representation
#      is made up of its eigenstates.
#
#     Returns
#     -------
#     - [0]: an Observable object standing for the x component of the spin;
#     - [1]: an Observable object standing for the y component of the spin;
#     - [2]: an Observable object standing for the z component of the spin;
#     """
#     I = [Qobj((self.raising_operator() + self.lowering_operator()) / 2, dims=self.dims),
#          Qobj((self.raising_operator() - self.lowering_operator()) / 2j, dims=self.dims), qeye(self.dims)]
#     for m in range(self.d):
#         I[2].data[m, m] = self.quantum_number - m
#     return I

'''Old calculate tensor'''
# for m in range(spin.n_spins)[:n]:
#     term_n = tensor(qeye(spin.spin[m].d), term_n)
# for l in range(spin.n_spins)[n + 1:]:
#     term_n = tensor(term_n, qeye(spin.spin[l].d))


# def random_observable(d):
#     """
#     Returns a randomly generated observable of dimensions d. Wrapper for QuTiP's
#     `rand_herm()`.

#     Parameters
#     ----------
#     d : int
#         Dimensions of the Observable to be generated.

#     Returns
#     -------
#     An Observable object whose matrix is d-dimensional and has random complex elements with real
#     and imaginary parts in the half-open interval [-10., 10.].
#     """
#     return rand_herm(d)

''' legay fourier transform '''

# def legacy_fourier_transform_signal(times, signal, frequency_start,
#                                     frequency_stop, opposite_frequency=False):
#     """
#     Deprecated since QuTiP integration; see simulation.fourier_transform_signal.

#     Computes the Fourier transform of the passed time-dependent signal over the
#     frequency interval [frequency_start, frequency_stop]. The implemented
#     Fourier transform operation is

#     where S is the original signal and T is its duration. In order to have a
#     reliable Fourier transform, the signal should be very small beyond time T.

#     Parameters
#     ----------
#     times : array-like
#         Sampled time domain (in microseconds).

#     signal : array-like
#         Sampled signal to be transformed in the frequency domain (in a.u.).

#     frequency_start, frequency_stop : float
#         Left and right bounds of the frequency interval of interest, 
#         respectively (in MHz).

#     opposite_frequency : bool
#         When it is True, the function computes the Fourier spectrum of the 
#         signal in both the intervals 
#         frequency_start -> frequency_stop and 
#         -frequency_start -> -frequency_stop 
#         (the arrow specifies the ordering of the Fourier transform's values 
#         when they are stored in the arrays to be returned).

#     Returns
#     -------
#     [0]: numpy.ndarray
#         Vector of 1000 equally spaced sampled values of frequency in the
#         interval [frequency_start, frequency_stop] (in MHz).

#     [1]: numpy.ndarray
#         Fourier transform of the signal evaluated at the discrete frequencies
#         reported in the first output (in a.u.).

#     If opposite_frequency=True, the function also returns:

#     [2]: numpy.ndarray
#         Fourier transform of the signal evaluated at the discrete frequencies
#         reported in the first output changed by sign (in a.u.).  
#     """
#     dt = times[1] - times[0]

#     frequencies = np.linspace(start=frequency_start,
#                               stop=frequency_stop, num=1000)

#     fourier = [[], []]

#     if not opposite_frequency:
#         sign_options = 1
#     else:
#         sign_options = 2

#     for s in range(sign_options):
#         for nu in frequencies:
#             integral = np.zeros(sign_options, dtype=complex)
#             for t in range(len(times)):
#                 integral[s] = integral[s] + \
#                     np.exp(-1j * 2 * np.pi * (1 - 2 * s) *
#                            nu * times[t]) * signal[t] * dt
#             fourier[s].append(integral[s])

#     if not opposite_frequency:
#         return frequencies, np.array(fourier[0])
#     else:
#         return frequencies, np.array(fourier[0]), np.array(fourier[1])


''' complex_phase_cmap() '''
# def complex_phase_cmap():
#     """
#     Create a cyclic colormap for representing the phase of complex variables
#
#     From QuTiP 4.0:
#     https://qutip.org
#
#     Returns
#     -------
#     cmap : A matplotlib linear segmented colormap.
#     """
#     cdict = {
#         "blue": ((0.00, 0.0, 0.0), (0.25, 0.0, 0.0), (0.50, 1.0, 1.0), (0.75, 1.0, 1.0), (1.00, 0.0, 0.0)),
#         "green": ((0.00, 0.0, 0.0), (0.25, 1.0, 1.0), (0.50, 0.0, 0.0), (0.75, 1.0, 1.0), (1.00, 0.0, 0.0)),
#         "red": ((0.00, 1.0, 1.0), (0.25, 0.5, 0.5), (0.50, 0.0, 0.0), (0.75, 0.0, 0.0), (1.00, 1.0, 1.0)),
#     }
#
#     cmap = clrs.LinearSegmentedColormap("phase_colormap", cdict, 256)
#     return cmap