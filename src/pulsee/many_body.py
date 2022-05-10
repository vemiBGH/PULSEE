from qutip import Qobj
import numpy as np

def partial_trace(operator, subspaces_dimensions, index_position):
    """
    Returns the partial trace of an operator over the specified subspace of the Hilbert space.
    Different from QuTip.Qobj.ptrace() in that it traces over the given subspace as opposed to keeping the given subspace.
    
    Parameters
    ----------
    
    - operator: Qobj
                Operator to be sliced through the partial trace operation.
    - subspaces_dimensions: list
                            List of the dimensions of the subspaces whose direct sum is the Hilbert space where operator acts.
    - index_position: int
                      Indicates the subspace over which the partial trace of operator is to be taken by referring to the corresponding position along the list subspace_dimensions.

    Returns
    -------
    There are 3 possibilities:
    1. If operator belongs to type DensityMatrix, the function returns a DensityMatrix object representing the desired partial trace;
    2. If operator belongs to type Observable, the function returns a Observable object representing the desired partial trace;
    3. Otherwise, the function returns a generic Operator object representing the desired partial trace.
    """

    m = operator.full()
    d = subspaces_dimensions

    n = len(d)
    i = index_position
    
    d_downhill = int(np.prod(d[i+1:n]))
        
    d_block = d[i]*d_downhill
        
    d_uphill = int(np.prod(d[0:i]))
        
    partial_trace = np.empty((d_downhill, d_downhill*(d_uphill+1)), dtype=np.ndarray)
    for j in range(d_uphill):
        p_t_row = np.empty((d_downhill, d_downhill), dtype=np.ndarray)
        for k in range(d_uphill):
            block = m[j*d_block:(j+1)*d_block, k*d_block:(k+1)*d_block]
            p_t_block = np.zeros((d_downhill, d_downhill))
            for l in range(d[i]):
                p_t_block = p_t_block + block[l*d_downhill:(l+1)*d_downhill, \
                                            l*d_downhill:(l+1)*d_downhill]
            p_t_row = np.concatenate((p_t_row, p_t_block), axis=1)
        partial_trace = np.concatenate((partial_trace, p_t_row), axis=0)
        
    partial_trace = Qobj(partial_trace[d_downhill:,d_downhill:])
    return partial_trace
    
    
