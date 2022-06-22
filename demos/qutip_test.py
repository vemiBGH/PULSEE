from qutip import * 
import numpy as np

sz = sigmaz()

init_ket = 2 ** (-1/2) * Qobj([[1], [1]]) # Bloch vector is unit x
init_dm = init_ket * init_ket.dag()

b_0 = 1
gam = 1
h_zeem = gam * b_0 * sz
h = [h_zeem]

larmor_freq = gam * b_0 
period = 2 * np.pi / (larmor_freq)
t = np.linspace(0, period / 2, 100)

print(mesolve(h, init_dm, t, options=Options()).states[-1])