# PULSEE (Program for the simULation of nuclear Spin Ensemble Evolution)
## Development branch for Quantum Computing Module [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lzawbrito/PULSEE/lbrito-quantum-computing?labpath=demos%2Fquantum_computing_demo.ipynb)
Authors: Davide Candoli (Università di Bologna) and Lucas Brito (Brown University)

For the past few decades, quantum
computing and quantum information science have remained active and
fertile fields, and the prospect of expediting certain computational
tasks using quantum algorithms is expected to continue to motivate
research in the field for many more years. Encouraged by this persistent
and widespread interest in quantum computing---and aiming to continue
PULSEE's mission to bridge condensed matter theory and experiment
through simulations of spin evolution phenomena---I have implemented the
fundamental components of quantum circuits in the form of a new PULSEE
module titled `Quantum_Computing`.

Installation 
===========

To use this development version of PULSEE, first install the dependencies as 
outlined by `requirements.txt`: 
```
>>> pip install -r requirements.txt
```

or, with Anaconda: 
```
conda install --file conda-requirements.txt
```

Navigate to the directory containing `setup.py` (this should simply be PULSEE), 
and run `pip install -e .` to perform a development installation of the package. 
This lets your environment know that you will continue to edit the source code 
of the package, so changes made to any files will be trickled down to any 
imports. Enjoy! 

## Demos
See `/demos` for some demonstrations of using PULSEE (as Python scripts and 
Jupyter notebooks).

Implementation
==============

Design
------

The `Quantum_Computing` module comprises four classes, a handful of functions,
and implementations of selected quantum gates as instances of the `NGate`
class. The module was implemented with a hybrid functional and object
oriented approach: the core mathematical objects are implemented as
Python classes and auxiliary functionalities are implemented as Python
functions. The class structure is as follows:

-   `QubitState`: The `QubitState` class implements the quantum
    computational qubit state in an arbitrary composition of qubit
    spaces. That is, the qubit state is a $2^n$-dimensional state ket in
    an $n$-fold composition of qubit spaces. A general qubit state class
    is defined (as opposed to, say, a composite qubit state and a child
    qubit class or vice versa) for two reasons: there is no
    functionality required of qubits that is not also required of
    composite qubits and vice versa, and the module's design centers
    qubit spaces as the generators of qubit states, thus eliminating the
    need to specify whether a qubit is in a composite qubit space.

    The `QubitState` constructor takes two arguments: the
    `CompositeQubitSpace` instance representing the qubit space this
    qubit state is in and the matrix representation of this qubit. The
    constructor ensures that the matrix representation has the
    appropriate dimensionality for the given qubit space, and throws a
    `MatrixRepresentationError` if this not the case.

    For syntactic sugar, the `QubitState` class implements the binary
    arithmetic operations `*`, `+`, and `-`. The multiplication
    operation calls the `tensor_product` method, which itself returns a
    new `QubitState` object represented by the Kronecker product of the
    two multiplicands' matrices. The addition and subtraction
    operators, similarly, sum and subtract the matrix representations of
    the qubit states and return a corresponding new `QubitState`.

    For convenience, the `QubitState` class is equipped with a
    `get_density_matrix` method as well as a functionally identical
    `density_matrix` property. This returns a `DensityMatrix` operator
    as defined in `operators.py`, the matrix representation of which is
    computed with respect to the object's qubit space computational
    basis. The method `QubitState.get_reduced_density_matrix` computes
    the parcial trace of this density matrix with respect to a state
    given by an index 0 or 1  (note that there is no support for
    reduced density matrices of states in larger composite qubit spaces;
    see Known Bugs and Limitations below. 

-   `CompositeQubitSpace`: The `CompositeQubitState` class implements a
    $n$-fold tensor product of qubit spaces. This is the most general
    implementation of qubit spaces (i.e., it is the parent class of the
    two-dimensional non-composite `QubitSpace`). The class hierarchy is
    designed in this fashion because there are functionalities and
    properties of composite qubit spaces that become special cases in
    the `QubitSpace` class (and not vice versa); separation of concerns
    inspires us to write one implementation of the most general form of
    these functionalities.

    The `CompositeQubitSpace` constructor takes as a required argument
    `n` the number of qubit spaces out of which this instance is
    composed. Being the only identifying property of a qubit space
    instance, the equality operation `__eq__` for two
    `CompositeQubitSpace` instances is overriden to check whether
    `self.n == other.n`.

    The central functionality of `CompositeQubitSpace` instances is
    obtaining the computational basis for this qubit space composition.
    The method `basis_from_indices` returns, as a `ndarray`, the matrix
    representation of the the computational basis ket specified by the
    given list of indices; i.e., if given `[1, 0, 1]`, the matrix
    representatin of the ket $|101\rangle$. The method
    `basis_ket_from_indices` is identical but wraps the matrix
    representation in a `QubitState` object. Lastly, `onb_matrices` uses a
    binary recursion to generate all possible combinations of $n$ zeros and
    ones, calls `basis_from_indices` on each, then returns these `ndarray`
    matrix representations in a list, effectively producing a list of spanning
    orthonormal basis kets for this composite qubit space.

-   `QubitSpace` (inherits `CompositeQubitSpace`): Motivated by the fact
    that a qubit space can appropriately be thought of as a $1$-fold
    composite qubit space, the `QubitSpace` class is a child class of
    `CompositeQubit` `Space` and thus inherits all functionality
    specified above.

    One feature of two-dimensional qubit spaces that is not possessed by
    composite qubit spaces is the ability to capture all information
    about a state ket using a Bloch sphere representation. The
    `QubitSpace` implementation leverages this picture in the
    `make_state` method, which produces `Qubit` `State` instanaces and
    takes in either `alpha` and `beta` keyword arguments representing
    the azimuthal and polar angles, or a `coeffs` keyword argument
    representing the coefficients of the superposition of basis kets
    $|0\rangle $ and $|1\rangle$ (which the method normalizes). If given all
    keyword arguments, the method prioritizes angles; if one angle is
    missing, the method uses the coefficients.

-   `NGate` (inherits `Operator`): the `NGate` class implements quantum
    $n$-gates---i.e., unary gates (2-by-2 matrices), binary gates
    (4-by-4 matrices), and virtually any other `Operator` on the
    appropriate qubit space. The class inherits the `Operator` class as
    gates can be regarded as quantum operators; its constructor takes in
    an array `x` as the matrix representation of this operator and the
    `CompositeQubit` `Space` on which this operator acts, ensures that
    the given matrix has the appropriate dimensions for this qubit space
    and verifies that the matrix is unitary, throwing
    `MatrixRepresentationError` in the case either of these requirements
    is not fulfilled.

    `NGate` implements `__call__`, meaning that instances of this class
    are callable. Calling a `NGate` instance on a `QubitState` is
    equivalent to calling the method `NGate.apply` on that `QubitState`,
    which is also publicly accessible. The `apply` method multiplies the
    provided `QubitState`'s matrix representation by the instance's own
    matrix representation, and returns the resulting matrix wrapped in a
    `QubitState`.

Known Bugs and Limitations 
--------------------------

-   The current reduced density matrix implementaton does not support
    qubit states composed of more than one qubit. It is posited by the
    author that the partial trace is well-defined for larger
    compositions of qubit states. It remains to determine an algorithm
    to do so and implement it.

-   `CompositeQubitSpace.onb_matrices` is perhaps functionally
    equivalent to simply generating the columns of the $2^n$-by-$2^n$
    identity matrix; further investigation needed.

-   Fuzz-testing revealed certain usages of analytical functions in
    `QubitSpace.make_state` gave rise to Numpy underflow errors. It is
    assumed that given values will not be sufficiently small to cause
    unexpected behavior; if they are, appropriate unit conversions must
    be made.

-   The present test suite for the quantum computing module
    (`tests/Test_Quantum_Computing.py`) is quite limited due to time
    constraints; in particular, `QubitState.get_density_matrix`,
    `.get_reduced` `_density_matrix`, `.__add__`, `.__sub__`,
    `tensor_product`, and `NGate` have not been tested. It remains to
    write a more thorough test suite and produce a follow-up report
    including a summary of said test suite.

Refactoring
-----------

The previous PULSEE code-base has undergone some refactoring in
preparation for the distribution of the software as a Python package.
Source code has been relocated to the directory `src/pulsee`, following
the "src" layout cited by [Pytest's integration
practices](https://docs.pytest.org/en/stable/goodpractices.html). File
(module) names have been made all-lowercase per compliance with the
[Python style guide](https://www.python.org/dev/peps/pep-0008/). The
author also intends to, in the future, change class names to comply with
the "CapWords" convention, also per Python's style guide.