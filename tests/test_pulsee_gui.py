import sys
sys.path.insert(1, '../Code')

# Generic python imports
import math
import numpy as np
import pandas as pd
from fractions import Fraction
from functools import partial

# Generic graphics imports
import matplotlib
import matplotlib.pylab as plt

from pulsee.PULSEE_CMP_GUI import null_string, Simulation_Manager, System_Parameters
    
def test_null_string():
    test_text = 'test_text'
    test_text1 = null_string(test_text)
    assert test_text1 == test_text

def test_formula_nu_q():
    sim_man = Simulation_Manager()
    sim_man.spin_par = {'quantum number' : 3/2,
                        'gamma/2pi' : 1.}
    
    sim_man.quad_par = {'coupling constant' : 2.,
                        'asymmetry parameter' : 0.,
                        'alpha_q' : 0,
                        'beta_q' : 0,
                        'gamma_q' : 0}
    sys_par = System_Parameters(sim_man)
    
    sys_par.store_and_write_nu_q(sim_man)
    
    assert np.isclose(sim_man.nu_q, 1, rtol=1e-10)