import numpy
from qutip import Qobj as QutipQobj
import numpy as np
from scipy.linalg import eig


class Qobj(QutipQobj):
    """
    Wrapper for QuTiP Qobj class. Implements the properties `positivity`, 
    `trace`, `isdm`. 

    From the QuTiP Qobj documentation: A class for representing quantum objects,
    such as quantum operators and states.

    The Qobj class is the QuTiP representation of quantum operators and state
    vectors. This class also implements math operations +,-,* between Qobj
    instances (and / by a C-number), as well as a collection of common
    operator/state operations.  The Qobj constructor optionally takes a
    dimension ``list`` and/or shape ``list`` as arguments.

    Parameters
    ----------
    inpt : array_like
        Data for vector/matrix representation of the quantum object.
    dims : list
        Dimensions of object used for tensor products.
    shape : list
        Shape of underlying data structure (matrix shape).
    copy : bool
        Flag specifying whether Qobj should get a copy of the
        input data, or use the original.
    fast : bool
        Flag for fast qobj creation when running ode solvers.
        This parameter is used internally only.


    Attributes
    ----------
    data : array_like
        Sparse matrix characterizing the quantum object.
    dims : list
        List of dimensions keeping track of the tensor structure.
    shape : list
        Shape of the underlying `data` array.
    type : str
        Type of quantum object: 'bra', 'ket', 'oper', 'operator-ket',
        'operator-bra', or 'super'.
    superrep : str
        Representation used if `type` is 'super'. One of 'super'
        (Liouville form) or 'choi' (Choi matrix with tr = dimension).
    isherm : bool
        Indicates if quantum object represents Hermitian operator.
    isunitary : bool
        Indictaes if quantum object represents unitary operator.
    iscp : bool
        Indicates if the quantum object represents a map, and if that map is
        completely positive (CP).
    ishp : bool
        Indicates if the quantum object represents a map, and if that map is
        hermicity preserving (HP).
    istp : bool
        Indicates if the quantum object represents a map, and if that map is
        trace preserving (TP).
    iscptp : bool
        Indicates if the quantum object represents a map that is completely
        positive and trace preserving (CPTP).
    isket : bool
        Indicates if the quantum object represents a ket.
    isbra : bool
        Indicates if the quantum object represents a bra.
    isoper : bool
        Indicates if the quantum object represents an operator.
    issuper : bool
        Indicates if the quantum object represents a superoperator.
    isoperket : bool
        Indicates if the quantum object represents an operator in column vector
        form.
    isoperbra : bool
        Indicates if the quantum object represents an operator in row vector
        form.
    positivity* : bool 
        Indicates if the quantum object is positive semi-definite. 
    trace* : numpy.complex128
        Return the trace of the given operator. 
    isdm* : bool
        Indicates if the operator satisfies the characteristic properties of a 
        density matrix; i.e., positive semi-definite, unit trace, and
        hermiticity. 
        
    Methods
    -------
    copy()
        Create copy of Qobj
    conj()
        Conjugate of quantum object.
    cosm()
        Cosine of quantum object.
    dag()
        Adjoint (dagger) of quantum object.
    dnorm()
        Diamond norm of quantum operator.
    dual_chan()
        Dual channel of quantum object representing a CP map.
    eigenenergies(sparse=False, sort='low', eigvals=0, tol=0, maxiter=100000)
        Returns eigenenergies (eigenvalues) of a quantum object.
    eigenstates(sparse=False, sort='low', eigvals=0, tol=0, maxiter=100000)
        Returns eigenenergies and eigenstates of quantum object.
    expm()
        Matrix exponential of quantum object.
    full(order='C')
        Returns dense array of quantum object `data` attribute.
    groundstate(sparse=False, tol=0, maxiter=100000)
        Returns eigenvalue and eigenket for the groundstate of a quantum
        object.
    inv()
        Return a Qobj corresponding to the matrix inverse of the operator.
    matrix_element(bra, ket)
        Returns the matrix element of operator between `bra` and `ket` vectors.
    norm(norm='tr', sparse=False, tol=0, maxiter=100000)
        Returns norm of a ket or an operator.
    permute(order)
        Returns composite qobj with indices reordered.
    proj()
        Computes the projector for a ket or bra vector.
    ptrace(sel)
        Returns quantum object for selected dimensions after performing
        partial trace.
    sinm()
        Sine of quantum object.
    sqrtm()
        Matrix square root of quantum object.
    tidyup(atol=1e-12)
        Removes small elements from quantum object.
    tr()
        Trace of quantum object.
    trans()
        Transpose of quantum object.
    transform(inpt, inverse=False)
        Performs a basis transformation defined by `inpt` matrix.
    trunc_neg(method='clip')
        Removes negative eigenvalues and returns a new Qobj that is
        a valid density operator.
    unit(norm='tr', sparse=False, tol=0, maxiter=100000)
        R

    *Implemented by PULSEE wrapper.
    """

    @property
    def positivity(self):
        """
        Returns a boolean which expresses whether the matrix is a positive
        operator, i.e. its matrix has only non-negative eigenvalues (taking the 0
        with an error margin of 10^(-10)).

        Modified from pulsee.operators.Operator.positivity in legacy version.
        
        Returns
        -------
        True, when positivity is verified.
        
        False, when positivity is not verified.
        """
        eigenvalues = eig(self)[0]
        return np.all(np.real(eigenvalues) >= -1e-10)

    @property
    def trace(self):
        """
        Returns the trace of the operator (which is a complex number).
        """
        trace = 0
        for i in range(self.shape[0]):
            trace = trace + self[i, i]
        return trace
    
    @property
    def isdm(self): 
        return self.isherm and self.positivity and self.trace == 1



