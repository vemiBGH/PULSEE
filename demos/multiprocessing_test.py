from pulsee.simulation import ed_evolve, nuclear_system_setup
from qutip import *
import numpy as np 

h = Qobj([[1,0],[0,-1]])
r = (1/2) * Qobj([[1, 1], [1, 1]])

spin_par = {'quantum number' : 1/2,
            'gamma/2pi' : 1.}

zeem_par = {'field magnitude' : 10.,
            'theta_z' : 0,
            'phi_z' : 0}

spin, h, rho0 = nuclear_system_setup(spin_par, quad_par=None, zeem_par=zeem_par, initial_state=r)
t = np.linspace(0, 1, 100)

if __name__ == '__main__':
	ed_evolve(h, rho0, spin, t, e_ops=[], fid=True, par=True)