import math
import numpy as np

from .operators import Operator, DensityMatrix, Observable

def tensor_product(A, B):
    """
    Returns the tensor product of two operators.
    
    Parameters
    ----------
    - A: Operator
         Left factor in the product.
    - B: Operator
         Right factor in the product.
    
    Returns
    -------
    1. If both A and B belong to type DensityMatrix, the function returns a
       DensityMatrix object initialised with such a matrix; 
    2. If both A and B belong to type Observable, the function returns an
       Observable object initialised with such a matrix;
    3. Otherwise, the function returns a generic Operator object initialised
    with such a matrix.  
    """
    
    d_A = A.dimension()
    d_B = B.dimension()
    
    cast_array_into_complex = np.vectorize(complex)
    A_tensor_B = cast_array_into_complex(np.zeros([d_A * d_B, d_A * d_B]))
    
    for i in range(d_A):
        for j in range(d_A):
            for ii in range(d_B):
                for jj in range(d_B):
                    A_tensor_B[i * d_B + ii, j * d_B + jj] = A.matrix[i, j] * B.matrix[ii, jj]
    
    if isinstance(A, DensityMatrix) and isinstance(B, DensityMatrix):
        return DensityMatrix(A_tensor_B)
    
    elif isinstance(A, Observable) and isinstance(B, Observable):
        return Observable(A_tensor_B)
    
    # TODO add isinstance(Ket) and implement some sort of state ket class which 
    # qubits inherit. 
    else:
        return Operator(A_tensor_B)
    
def partial_trace(operator, subspaces_dimensions, index_position):
    """
    Returns the partial trace of an operator over the specified subspace of the
    Hilbert space.
    
    Parameters
    ----------
    
    - operator: Operator
                Operator to be sliced through the partial trace operation.
    - subspaces_dimensions: list
                            List of the dimensions of the subspaces whose direct
                            sum is the Hilbert space where operator acts.
    - index_position: int
                      Indicates the subspace over which the partial trace of 
                      operator is to be taken by referring to the corresponding 
                      position along the list subspace_dimensions.

    Returns
    -------
    There are 3 possibilities:
    1. If operator belongs to type DensityMatrix, the function returns a
       DensityMatrix object representing the desired partial trace; 
    2. If operator belongs to type Observable, the function returns a Observable
       object representing the desired partial trace;
    3. Otherwise, the function returns a generic Operator object representing
       the desired partial trace.  
    """
    m = operator.matrix
    
    d = subspaces_dimensions
    
    i = index_position
    
    n = len(d)
    
    d_downhill = int(np.prod(d[i + 1:n]))
        
    d_block = d[i] * d_downhill
        
    d_uphill = int(np.prod(d[0:i]))
        
    partial_trace = np.empty((d_downhill, d_downhill * (d_uphill + 1)), dtype=np.ndarray)
    for j in range(d_uphill):
        p_t_row = np.empty((d_downhill, d_downhill), dtype=np.ndarray)
        for k in range(d_uphill):
            block = m[j * d_block:(j + 1) * d_block, k * d_block:(k + 1) * d_block]
            p_t_block = np.zeros((d_downhill, d_downhill))
            for l in range(d[i]):
                p_t_block = p_t_block  +  block[l * d_downhill:(l + 1) * d_downhill, \
                                              l * d_downhill:(l + 1) * d_downhill]
            p_t_row = np.concatenate((p_t_row, p_t_block), axis=1)
        partial_trace = np.concatenate((partial_trace, p_t_row), axis=0)
        
    partial_trace = partial_trace[d_downhill:, d_downhill:]
    
    if isinstance(operator, DensityMatrix):
        return DensityMatrix(partial_trace)
    
    elif isinstance(operator, Observable):
        return Observable(partial_trace)
    
    else:
        return Operator(partial_trace)
    
    
