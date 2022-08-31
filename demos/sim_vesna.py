import numpy as np
import pandas as pd
from pulsee.operators import * 
from pulsee.simulation import *
import qutip as qt
import matplotlib.pyplot as plt
qn = 3/2
gr = 1
b0 = 1
e2qQ = 0.2
eta = 1

spin_par = {'quantum number' : qn,
			'gamma/2pi' : gr}

zeem_par = {'field magnitude' : b0,
			'theta_z' : 0,
			'phi_z' : 0}


quad_par = {'coupling constant' : e2qQ,
			'asymmetry parameter' : eta,
			'alpha_q' : 0.,
			'beta_q' : math.pi/2,
			'gamma_q' : 0.}

b1 = 1e-2 * b0

mode = pd.DataFrame([(2 * np.pi * gr * b0, 2 * b1, 0., math.pi/2, 0)], 
					columns=['frequency', 'amplitude', 'phase', 'theta_p', 'phi_p'])

RRF_par = {'nu_RRF': 2 * np.pi * gr * b0,
			'theta_RRF': math.pi,
			'phi_RRF': 0.}

dm_initial = Qobj([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
spin, h_unperturbed, dm_initial = nuclear_system_setup(spin_par, quad_par=quad_par, 
									zeem_par=zeem_par, initial_state='canonical')

opts = qt.Options(nsteps=1500)
dm_evolved = evolve(spin, h_unperturbed, dm_initial, solver='mesolve', \
					mode=mode, pulse_time=1 / (4 * gr * b1), \
					picture='IP', n_points=50, opts=opts)

t, FID = FID_signal(spin, h_unperturbed, dm_evolved, acquisition_time=50, \
		T2=[lambda t: 1.0 / t, lambda t: np.exp(-t)], reference_frequency=0, n_points=100)


f, ft = fourier_transform_signal(FID, t, padding=10)


	