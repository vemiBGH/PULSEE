import math
import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import expm, eig
from scipy.constants import Planck, Boltzmann
from scipy.integrate import quad

class Operator:
    
    """
    The objects of this class represent linear applications which transform
    vectors inside a given linear space. For the purposes of our simulation,
    where the systems under study are nuclear spins, we consider operators
    acting in finite-dimensional Hilbert spaces. The main advantage of
    finite-dimensional operators is that they admit a matrix representation,
    which depends on the chosen basis set of vectors.
    
    Attributes
    ----------
    - matrix: numpy.ndarray

      Square array of complex numbers providing the matrix representation of the
      operator in the desired basis set.    
      
    Methods
    -------
    """
    
    def __init__(self, x):
        """
        Constructs an instance of Operator.
  
        Parameters
        ----------
        - x: either int or numpy.ndarray
             When x is a positive integer, the constructor initializes matrix as
             an x-dimensional identity array.  When x is an array, it is
             assigned directly to matrix.  In this case, the constructor checks
             that the given object is a square array, and raises an appropriate
             error if it is not.
    
        Returns
        -------
        The initialised Operator object.
    
        Raises
        ------
        IndexError, when the passed array x is not a 2D square array.
        """
        if isinstance(x, np.ndarray):
            if len(x.shape) != 2 or x.shape[0] != x.shape[1]:
                raise IndexError("An Operator object should"
                + " be initialised with a 2D square array")
            cast_array_into_complex = np.vectorize(complex)
            input_array = cast_array_into_complex(x)
            self.matrix = input_array # Matrix representation of the operator 
                                      # (in the desired basis).
        else:
            d = int(x)
            self.matrix = np.identity(d, dtype=complex)
    
    def dimension(self):
        """
        Returns the dimension of the matrix, i.e. the dimensionality of the
        Hilbert space where the operator acts.  
        """
        return self.matrix.shape[0]
    
    def __add__(self, addend):
        """
        Returns the sum of two Operator objects.
  
        Parameters
        ----------
        - o_left, o_right: Operator
                           Addends of the sum.
        
        Returns
        -------
        A new Operator object initialised with the sum of the addends' matrices.
        """
        return Operator(self.matrix + addend.matrix)

    def __sub__(self, subtrahend):
        """
        Returns the difference of two Operator objects.
  
        Parameters
        ----------
        - o_left, o_right: Operator
                           Minuend and subtrahend in the difference, respectively.
  
        Returns
        -------
        A new Operator object initialised with the difference of the arguments' matrices.
        """
        return Operator(self.matrix - subtrahend.matrix)
    
    def __neg__(self):
        """
        Returns the opposite of the Operator.
        """
        return Operator(- self.matrix)
    
    def __mul__(self, factor):
        """
        Returns the Operator resulting either from: (1) the product of two
        Operator objects; (2) the multiplication of an Operator by a scalar.
        
        Parameters
        ----------
        - o: Operator
        - a: (1) Operator; (2) complex number.

        Returns
        -------
        A new Operator object initialised either with: (1) the product between
        the matrices of o and a, in reading order; (2) the matrix of o
        multiplied by the scalar a.  
        """
        if isinstance(factor, Operator):
            return Operator(self.matrix @ factor.matrix)
        else:
            factor = complex(factor)
            return Operator(self.matrix * factor)
                
    def __rmul__(self, factor):
        """
        Returns the Operator resulting either from: (1) the product of two
        Operator objects; (2) the multiplication of an Operator by a scalar.
        
        Parameters
        ----------
        - o: Operator
        - a: (1) Operator; (2) complex number.

        Returns
        -------
        A new Operator object initialised either with: (1) the product between
        the matrices of o and a, in reading order; (2) the matrix of o
        multiplied by the scalar a.  
        """
        factor = complex(factor)
        return Operator(factor * self.matrix)
            
    def __truediv__(self, divisor):
        """
        Returns the operator divided by a complex number.
  
        Parameters
        ----------
        - o: Operator
        - c: complex number.
  
        Returns
        -------
        A new Operator object initialised with the division of the matrix of o
        by the quantity c.
        
        Raises
        ------
        ZeroDivisionError, when c is cast to 0.
        """
        try:
            divisor = complex(divisor)
            if divisor == 0:
                raise ZeroDivisionError                
            return Operator(self.matrix / divisor)
        except ZeroDivisionError:
            raise ZeroDivisionError("The division of an Operator by 0 makes"
                + " no sense")

    def __pow__(self, exponent):
        """
        Returns the operator raised to the power of exponent.

        Parameters
        ----------
        - exponent: int
        
        Returns
        -------
        A new Operator object initialised with the matrix of o raised to the
        power of exponent.  
        """
        return Operator(np.linalg.matrix_power(self.matrix, exponent))
    
    def exp(self):
        """
        Returns a new Operator object representing the exponential of the
        operator. The program exploits the Padè approximation for the
        calculation of matrix exponentials.
        """
        exp_matrix = expm(self.matrix)
        return Operator(exp_matrix)
    
    def diagonalisation(self):
        """
        Diagonalises the operator, returning its eigenvalues and eigenvectors.
        
        Returns
        -------
        - [0]: An array listing the eigenvalues of the Operator's matrix;
        - [1]: An Operator object whose matrix columns are the eigenvectors of
               the considered operator, appearing in the same order as the
               corresponding eigenvalues in the first output. 
        """
        eigenvalues, change_of_basis = np.linalg.eig(self.matrix)
        return eigenvalues, Operator(change_of_basis)

    def sim_trans(self, change_of_basis_operator, exp=False):
        """
        Returns the Operator resulting from the application of the similarity
        transformation P^(-1)*M*P to the Operator M which owns the method.
        
        Parameters
        ----------
        - change_of_basis_operator: Operator
                                    Operator which enters expression
                                    P<sup>-1</sup>MP as P.
        - exp: bool
                Specifies whether the change of basis is performed using as P
                the change_of_basis_operator by itself (exp=False) or its
                exponential change_of_basis_operator.exp() (exp=True). Default
                value is set to False.
        """
        if exp == True:
            left_exp = (-change_of_basis_operator).exp()
            right_exp = change_of_basis_operator.exp()
            new_basis_operator = left_exp * self * right_exp
        else:
            new_basis_operator = (change_of_basis_operator ** (-1)) * self * change_of_basis_operator
        return new_basis_operator

    def trace(self):
        """
        Returns the trace of the operator (which is a complex number).
        """
        trace = 0
        for i in range(self.dimension()):
            trace = trace + self.matrix[i, i]
        return trace

    def dagger(self):
        """
        Returns a new Operator object initialised with the adjoint (complex
        conjugate transposed) of the matrix of the owner object.  
        """
        adjoint_matrix = (np.conj(self.matrix)).T
        return Operator(adjoint_matrix)

    def changed_picture(self, h_change_of_picture, time, invert=False):
        """
        Casts the operator either in a new picture generated by the Operator
        h_change_of_picture or back to the Schroedinger picture, according to
        the parameter invert.
        
        Parameters
        ----------
        - h_change_of_picture: Operator
                                 Operator which generates the change to the new
                                 picture. Typically, this operator is a term of
                                 the Hamiltonian (measured in MHz).
        - time: float
                  Instant of evaluation of the operator in the new picture,
                  expressed in microseconds.
                  
        - invert: bool
                     When it is False, the owner Operator object is assumed to
                     be expressed in the Schroedinger picture and is converted
                     into the new one.  When it is True, the owner object is
                     thought in the new picture and the opposite operation is
                     performed.

        Returns
        -------
        A new Operator object equivalent to the owner object but expressed in a
        different picture.
        """
        T = -1j * 2 * math.pi * h_change_of_picture * time
        if invert: T = -T
        return self.sim_trans(T, exp=True)

    def hermitianity(self):
        """
        Returns a boolean which expresses whether the operator is equal to its
        adjoint, comparing their matrices element-wise with a relative error
        tolerance of 10^(-10).
        
        Returns
        -------
        True, when hermitianity is verified.
        
        False, when hermitianity is not verified.
        """
        return np.all(np.isclose(self.matrix, self.dagger().matrix, rtol=1e-10))

    def unit_trace(self):
        """
        Returns a boolean which expresses whether the trace of the operator is
        equal to 1, within a relative error tolerance of 10<sup>-10</sup>.
        
        Returns
        -------
        True, when unit trace is verified.
        
        False, when unit trace is not verified.
        """
        return np.isclose(self.trace(), 1, rtol=1e-10)

    def positivity(self):
        """
        Returns a boolean which expresses whether the operator is a positive
        operator, i.e. its matrix has only non-negative eigenvalues (taking the
        0 with an error margin of 10^(-10)).
        
        Returns
        -------
        True, when positivity is verified.
        
        False, when positivity is not verified.
        """
        eigenvalues = eig(self.matrix)[0]
        return np.all(np.real(eigenvalues) >= -1e-10)
    
    def cast_to_density_matrix(self):
        """
        Returns an object of the class DensityMatrix initialised with the
        matrix of the owner Operator object, if all the properties of a density
        matrix are satisfied.
        
        Raises
        ------
        ValueError, when any of the three properties of density matrices is
        missing in the matrix of the owner object. Also, an error message
        explaining which properties are not satisfied is shown.  
        """
        return DensityMatrix(self.matrix)
    
    def cast_to_observable(self):
        """
        Returns an object of the class Observable initialised with the matrix of
        the owner Operator object, if it is hermitian.
        
        Raises
        ------
        ValueError, when the owner Operator object is not hermitian.
        """
        return Observable(self.matrix)

    def free_evolution(self, stat_hamiltonian, time):
        """
        Tries to cast the operator into the type DensityMatrix (using method
        cast_to_density_matrix), and in case of success returns this object
        evolved through the time time under the effect of stat_hamiltonian,
        calling the method of DensityMatrix with the same name.
        
        See the description of DensityMatrix.free_evolution below for details.
        """
        dm = self.cast_to_density_matrix()
        return dm.free_evolution(stat_hamiltonian, time)
    
    def expectation_value(self, density_matrix):
        """
        Tries to cast the Operator into the type Observable (using method
        cast_to_observable), and in case of success returns its expectation
        value in the state represented by density_matrix, calling the method of
        Observable with the same name.
        
        See the description of Observable.expectation_value below for details.
        """
        ob = self.cast_to_observable()
        return ob.expectation_value(density_matrix)


# Objects of the class DensityMatrix are special Operator objects characterised
# by the following properties: 
# i)   Hermitianity;
# ii)  Unit trace;
# iii) Positivity
class DensityMatrix(Operator):
    """
    A density matrix is a formal representation of the state of a quantum system
    which assigns a unique operator to each state. Density matrices associated
    to a well-defined (pure) state of the system are equivalent to the projector
    over the subspace generated by the vector describing that state in Dirac's
    notation.  Density matrix formalism is particularly suitable for the
    representation of mixed states, which encode the (classical) distribution of
    the states in an ensemble of identical systems.  The axiomatic definition of
    density matrix is based on the following properties:
    1. Hermitianity
    2. Unit trace
    3. Positivity
    
    Methods
    -------
    """
    
    def __init__(self, x):
        """
        Constructs an instance of DensityMatrix.
  
        Parameters
        ----------
        - x: either int or ndarray.
             When x is an integer, the constructor initialises matrix as an
             x-dimensional maximally entangled density matrix (identity(x)/x).
             When x is an array, it is assigned directly to matrix. In this
             case, the constructor checks that the given object is a square
             array, and raises appropriate errors if it is not. Also, the
             defining properties of density matrices are checked and errors are
             raised if any of them is not satisfied.   
        
        Returns
        -------
        The initialised DensityMatrix object.
        
        Raises
        ------
        - ValueError, when x is an array but some of the three definining
          properties (hermitianity, unit trace and positivity) are not verified.
          Also, an error message displaying which properties are missing is shown;
        - IndexError, when the passed array x is not a 2D square array.
        """
        d_m_operator = Operator(x)
        if isinstance(x, np.ndarray):
            error_message = "The input array lacks the following properties: \n"
            em = error_message
            if not d_m_operator.hermitianity():
                em = em + "- hermitianity \n"
            if not d_m_operator.unit_trace():
                em = em + "- unit trace \n"                
            if not d_m_operator.positivity():
                em = em + "- positivity \n"
            if em != error_message:
                raise ValueError(em)
        else:
            d = int(x)
            d_m_operator = d_m_operator * (1 / d)
        self.matrix = d_m_operator.matrix

    def free_evolution(self, static_hamiltonian, time):
        """
        Returns the density matrix represented by the owner object evolved
        through a time interval time under the action of the stationary
        Hamiltonian static_hamiltonian.
        
        Parameters
        ----------
        - static_hamiltonian: Observable or in general a hermitian Operator
                               Time-independent Hamiltonian of the system, in MHz.
        - time: float
                  Duration of the evolution, expressed in microseconds.
        
        Returns
        -------
        A DensityMatrix object representing the evolved state.
        """
        iHt = (1j * 2 * math.pi * static_hamiltonian * time)
        evolved_dm = self.sim_trans(iHt, exp=True)
        return DensityMatrix(evolved_dm.matrix)


# Objects of the class Observable are hermitian operators representing the measurable properties of the
# system.
class Observable(Operator):
    
    """
    Observables are the measurable properties of a physical system, and in
    quantum mechanics are represented by hermitian operators.
    
    The expectation value of an observable of a quantum system in a certain
    state is conventionally computed as the bra-operator-ket product of the
    observable's operator in between the state vector. Alternatively, one can
    exploit the density matrix representation of the state and find the
    expectation value as the trace of the product of the density matrix and the
    observable's operator.  
    """
    
    def __init__(self, x):
        """
        Constructs an instance of Observable.
        
        Parameters
        ----------
        - x: either int or ndarray
               When x is an integer, the constructor initialises matrix as an
               x-dimensional identity matrix.  When x is an array, it is
               assigned directly to matrix. In this case, the constructor checks
               that the given object is a hermitian square matrix, and raises
               appropriate errors if it is not.
        
        Returns
        -------
        The initialised Observable object.
        
        Raises
        ------
        - ValueError, when x is a square array but it is not hermitian;
        - IndexError, when the passed array x is not a 2D square array.
        """
        ob_operator = Operator(x)
        if isinstance(x, np.ndarray):
            if not ob_operator.hermitianity():
                raise ValueError("The input array is not hermitian")
        self.matrix = ob_operator.matrix
        
    def expectation_value(self, density_matrix):
        """
        Returns the expectation value of the observable calculated in the state
        represented by density_matrix.
        
        Parameters
        ----------
        - density_matrix: DensityMatrix (or any Operator which can be cast to
                          DensityMatrix) State of the system.
                          
        Returns
        -------
        In general, a complex number representing the expectation value of the
        observable for the given density matrix. When the imaginary part of this
        number is smaller than 10^(-10) (in absolute value), only the real part
        is retained.  
        """
        dm = density_matrix.cast_to_density_matrix()
        exp_val = (self * dm).trace()
        if np.absolute(np.imag(exp_val)) < 1e-10: exp_val = np.real(exp_val)
        return exp_val


def random_operator(d):
    """
    Returns a randomly generated operator object of dimensions d.
    
    Parameters
    ----------
    - d: int
         Dimensions of the Operator to be generated.
           
    Returns
    -------
    An Operator object whose matrix is d-dimensional and has random complex
    elements with real and imaginary parts in the half-open interval [-10.,
    10.].  
    """
    round_elements = np.vectorize(round)
    real_part = round_elements(20 * (np.random.random_sample(size=(d, d))-1/2), 2)
    imaginary_part = 1j * round_elements(20 * (np.random.random_sample(size=(d, d))-1/2), 2)
    random_array = real_part + imaginary_part
    return Operator(random_array)


def random_observable(d):
    """
    Returns a randomly generated observable of dimensions d.
  
    Parameters
    ----------
    - d: int
         Dimensions of the Observable to be generated.
          
    Returns
    -------
    An Observable object whose matrix is d-dimensional and has random complex
    elements with real and imaginary parts in the half-open interval [-10.,
    10.].  
    """
    o_random = random_operator(d)
    o_hermitian_random = (o_random + o_random.dagger())*(1/2)
    return Observable(o_hermitian_random.matrix)


def random_density_matrix(d):
    """
    Returns a randomly generated density matrix of dimensions d.
    
    Parameters
    ----------
    
    - d: int
         Dimensions of the DensityMatrix to be generated.
    
    Returns
    -------
    A DensityMatrix object whose matrix is d-dimensional and has randomly
    generated eigenvalues.  
    """
    spectrum = np.random.random(d)
    spectrum_norm = spectrum/(spectrum.sum())
    dm_diag = DensityMatrix(np.diag(spectrum_norm))
    cob = (1j * random_observable(d))  # The exponential of this matrix is a 
                                       # generic unitary transformation
    dm = dm_diag.sim_trans(cob, exp=True)
    return DensityMatrix(dm.matrix)


def commutator(A, B):
    """
    Returns the commutator of operators A and B.
    
    Parameters
    ----------
    - A, B: Operator
    
    Returns
    -------
    An Operator representing the commutator of A and B.
    """
    return A*B - B*A


def magnus_expansion_1st_term(h, time_step):
    """
    Returns the 1st order term of the Magnus expansion of the passed
    time-dependent Hamiltonian.
    
    Parameters
    ----------
    - h: np.ndarray of Observable
         Time-dependent Hamiltonian (expressed in MHz). Technically, an array of
         Observable objects which correspond to the Hamiltonian evaluated at
         successive instants of time. The start and end points of the array are
         taken as the extremes of integration 0 and t;
         
    - time_step: float 
                 Time difference between adjacent points of the array h,
                 expressed in microseconds.
                 
    Returns
    -------
    An adimensional Operator object resulting from the integral of h over the
    whole array size, multiplied by -1j*2*math.pi. The integration is carried
    out through the traditional trapezoidal rule.  
    """
    integral = h[0].matrix
    for t in range(len(h)-2):
        integral = integral + 2 * h[t+1].matrix
    integral = (integral + h[-1].matrix) * (time_step)/2
    magnus_1st_term = Operator(-1j * 2 * math.pi * integral)
    return magnus_1st_term


def magnus_expansion_2nd_term(h, time_step):
    """
    Returns the 2nd order term of the Magnus expansion of the passed
    time-dependent Hamiltonian.
    
    Parameters
    ----------
    - h: np.ndarray of Observable
         Time-dependent Hamiltonian (expressed in MHz). Technically, an array of
         Observable objects which correspond to the Hamiltonian evaluated at
         successive instants of time. The start and end points of the array are
         taken as the extremes of integration 0 and t;
         
    - time_step: float
                 Time difference between adjacent points of the array h,
                 expressed in microseconds.
                 
    Returns
    -------
    An adimensional Operator object representing the 2nd order Magnus term of
    the Hamiltonian, calculated applying Commutator to the elements in h and
    summing them.
    """
    integral = (h[0] * 0).matrix
    for t1 in range(len(h)-  1):
        for t2 in range(t1 + 1):
            integral = integral + (commutator(h[t1], h[t2]).matrix)*(time_step**2)
    magnus_2nd_term = ((2 * math.pi) ** 2) * Operator(-(1/2) * integral)
    return magnus_2nd_term


def magnus_expansion_3rd_term(h, time_step):
    """
    Returns the 3rd order term of the Magnus expansion of the passed
    time-dependent Hamiltonian.
    
    Parameters
    ----------
    
    - h: np.ndarray of Observable
         Time-dependent Hamiltonian (expressed in MHz). Technically, an array of
         Observable objects which correspond to the Hamiltonian evaluated at
         successive instants of time. The start and end points of the array are
         taken as the extremes of integration 0 and t;
         
    - time_step: float
                 Time difference between adjacent points of the array h,
                 expressed in microseconds.
                 
    Returns
    -------
    An adimensional Operator object representing the 3rd order Magnus term of
    the Hamiltonian, calculated applying nested Commutator to the elements in h
    and summing them.
    """
    integral = (h[0] * 0).matrix
    for t1 in range(len(h) - 1):
        for t2 in range(t1 + 1):
            for t3 in range(t2 + 1):
                integral = integral + \
                           ((commutator(h[t1], commutator(h[t2], h[t3])) + \
                             commutator(h[t3], commutator(h[t2], h[t1]))).matrix) * (time_step ** 3)
    magnus_3rd_term = Operator((1j/6) * ((2 * math.pi) ** 3) * integral)
    return magnus_3rd_term


def canonical_density_matrix(hamiltonian, temperature):
    """
    Returns the density matrix of a canonical ensemble of quantum systems at
    thermal equilibrium.
    
    Parameters
    ----------
    - hamiltonian: Operator
                   Hamiltonian of the system at equilibrium, expressed in MHz.
    - temperature: positive float
                   Temperature of the system in kelvin.

    Returns
    -------
    A DensityMatrix object which embodies the canonical density matrix.
    
    Raises
    ------
    ValueError, if temperature is negative or equal to zero.
    """
    if temperature <= 0:
        raise ValueError("The temperature must take a positive value")
    
    exponent = -(Planck * hamiltonian * 1e6)/(Boltzmann * temperature)
    numerator = exponent.exp()
    canonical_partition_function = numerator.trace()
    canonical_dm = numerator/canonical_partition_function
    return DensityMatrix(canonical_dm.matrix)








