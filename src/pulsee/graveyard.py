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