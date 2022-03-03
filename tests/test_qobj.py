import hypothesis.strategies as st
from pulsee.qobj import Qobj 
from hypothesis.extra import numpy 
import numpy as np

import hypothesis.strategies as st
from hypothesis import given 

@given(o = numpy.arrays(np.csingle, st.lists(st.integers(min_value=1, max_value=10), 
											min_size=2, 
											max_size=2)))
def test_isdm_fuzz(o):
	"""
	General fuzz test for Qobj.isdm wrapper property.
	"""
	Qobj(o).isdm