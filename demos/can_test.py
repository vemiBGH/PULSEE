import numpy as np
import pandas as pd
import math
from pulsee.operators import * 
from pulsee.simulation import *
import matplotlib.pyplot as plt

qn = 3/2
gr = 1
B0 = 1

spin_par = {'quantum number' : qn,
			'gamma/2pi' : gr}

zeem_par = {'field magnitude' : B0,
			'theta_z' : 0,
			'phi_z' : 0}

fig = plt.figure()

set_e2qQ = np.linspace(0, 0.2, num=5)

eta = 0

quad_par = {'coupling constant' : 0.2,
			'asymmetry parameter' : eta,
			'alpha_q' : 0.,
			'beta_q' : math.pi/2,
			'gamma_q' : 0.}

B1 = 1e-2*B0

mode = pd.DataFrame([(2 * np.pi * gr * B0, 2 * B1, 0., math.pi/2, 0)], 
					columns=['frequency', 'amplitude', 'phase', 'theta_p', 'phi_p'])

RRF_par = {'nu_RRF': 2 * np.pi * gr*B0,
			'theta_RRF': math.pi,
			'phi_RRF': 0.}

dm_initial = Qobj([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
spin, h_unperturbed, dm_initial = nuclear_system_setup(spin_par, quad_par=quad_par, 
									zeem_par=zeem_par, initial_state='canonical')
print(dm_initial)

#plot_real_part_density_matrix(dm_initial)
dm_evolved = evolve(spin, h_unperturbed, dm_initial, solver=magnus, \
					mode=mode, pulse_time= 1 / (4 * gr * B1), \
					picture='IP', n_points=50, order=2)

print(np.round(dm_evolved, 2))
#plot_real_part_density_matrix(dm_evolved)

t, FID = FID_signal(spin, h_unperturbed, dm_evolved, acquisition_time=50, T2=lambda t: np.exp(-t /2), reference_frequency=gr*B0, n_points=100)




		
