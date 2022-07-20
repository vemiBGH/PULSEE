import numpy as np
import pandas as pd
import math
from pulsee.operators import * 
from pulsee.simulation import *

import matplotlib.pyplot as plt


def sim_vesna():
	
	qn = 3/2
	gr = 11.26
	B0 = 9.
	
	spin_par = {'quantum number' : qn,
				'gamma/2pi' : gr}
	
	zeem_par = {'field magnitude' : B0,
				'theta_z' : 0,
				'phi_z' : 0}

	fig = plt.figure()
	
	set_e2qQ = np.linspace(0, 0.2, num=5)
	
	set_eta = np.linspace(0, 1, num=1)

	for eta in set_eta:
		print(eta)
		quad_par = {'coupling constant' : 0.2,
					'asymmetry parameter' : eta,
					'alpha_q' : 0.,
					'beta_q' : math.pi/2,
					'gamma_q' : 0.}
		
		B1 = 1e-2*B0
		
		mode = pd.DataFrame([(gr*B0, 2*B1, 0., math.pi/2, 0)], 
							columns=['frequency', 'amplitude', 'phase', 'theta_p', 'phi_p'])
		
		RRF_par = {'nu_RRF': gr*B0,
				   'theta_RRF': math.pi,
				   'phi_RRF': 0.}
		
		spin, h_unperturbed, dm_initial = nuclear_system_setup(spin_par, quad_par, zeem_par, \
														 initial_state='canonical')
		print(dm_initial)
		# print(h_unperturbed.diagonalisation()[0])
		
		#plot_real_part_density_matrix(dm_initial)

		 # TODO POSSIBLY PULSE TIME IS WRONG?
		dm_evolved = evolve(spin, h_unperturbed, dm_initial, solver=magnus, \
							mode=mode, pulse_time=1/(4*gr*B1), \
							picture='IP', n_points=1e2)
		
		print(dm_evolved)
		#plot_real_part_density_matrix(dm_evolved)
		
		t, FID = FID_signal(spin, h_unperturbed, dm_evolved, acquisition_time=50, T2=10, reference_frequency=-gr*B0, n_points=10)
		
		# plot_real_part_FID_signal(t, FID)
		
		f, ft = fourier_transform_signal(FID, t)
		
		plt.plot(f, np.absolute(ft)**2, label="\N{GREEK SMALL LETTER ETA} = " + str(np.round(eta, 2)))

	
	plt.figtext(.65, .8, 'e2qQ = ' + str(np.round(0.2, 3)) + ' MHz')
	plt.legend(loc='upper left')
	plt.xlabel("frequency (MHz)")    
	plt.ylabel("FT signal (a. u.)")
	
	plt.savefig(f'./demos/testfigs/increasing_eta.pdf')
	plt.show()

sim_vesna()


			
