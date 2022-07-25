from qutip import * 
import numpy as np
from pulsee.simulation import *
import pandas as pd

sz = sigmaz()

init_ket = Qobj([[1], [0], [0], [0]]) # Bloch vector is unit x
init_dm = init_ket * init_ket.dag()

b_0 = 10
b_1 = b_0 / 10
gam = 1
h_zeem = gam * b_0 * sz
h = [h_zeem]

larmor_freq = gam * b_0 
period = 2 * np.pi / (larmor_freq)
t = 1 / (2 * b_1)

spin_par = {'quantum number': 3/2, 'gamma/2pi': gam}
zeem_par = {'theta_z': 0, 'phi_z': 0, 'field magnitude': b_0}

spin, h_unpert, dm = nuclear_system_setup(spin_par=spin_par, zeem_par=zeem_par, initial_state=init_dm)


mode = pd.DataFrame([(2 * np.pi * gam * b_0, 2 * b_1, 0., np.pi / 2, 0)], 
					columns=['frequency', 'amplitude', 'phase', 'theta_p', 'phi_p'])

n_points = 20
print(np.round(evolve(spin, h_unpert, dm, solver='magnus', picture='IP', mode=mode, pulse_time=t, n_points=n_points), 3))
print(np.round(evolve(spin, h_unpert, dm, solver='mesolve', picture='IP', mode=mode, pulse_time=t, n_points=n_points), 3))
# low number of steps reproduces sim_vesna result