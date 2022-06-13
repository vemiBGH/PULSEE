from qutip import *
from helperNumPy import apply_pulse
from pulsee.simulation import apply_rot_pulse, ed_evolve 
import numpy as np 

# a = [[1,3],[3,2]]
# q = Qobj(a)
# iz = [[1,0],[0,-1]]
# izq = Qobj(iz)

# print(apply_pulse(a, 10, iz))
# print(apply_rot_pulse(q, 10, izq))


# a = [[], [], []]

# b = [[1,2,3]]
# c = [[3,4,5]]

# a = np.concatenate([a, np.transpose(b)], axis=1)
# print(np.concatenate([a, np.transpose(c)], axis=1))

# print([[] for i in range(3)])


t = np.linspace(0, 10, 100)
rho0 = (1/2) * Qobj([[1, 1],[1, 1]])
h = Qobj([[1,0],[0,-1]])
hlist = [h]
e_ops = [h, -h]
print(ed_evolve(hlist, rho0, t, e_ops))