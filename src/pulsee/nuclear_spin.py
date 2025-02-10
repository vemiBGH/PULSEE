import numpy as np
from qutip import Qobj, qeye, spin_J_set, tensor


class NuclearSpin:
    """
    An instance of the following class is to be thought as an all-round representation of the
    nuclear spin angular momentum. Indeed, it includes all the operators typically associated
    with the spin and also specific parameters like the spin quantum number and the spin multiplicity.
    
    Attributes
    ----------
    quantum_number : float
        Half-integer spin quantum number.
    d : int
        Dimensions of the spin Hilbert space.
    gyro_ratio_over_2pi : float
        Gyromagnetic ratio (over 2 pi) of the nuclear spin measured in units of
        MHz/T. 
        The gyromagnetic ratio is the constant of proportionality  between the 
        intrinsic magnetic moment and the spin angular momentum of a particle.
    I : dict
        Dictionary whose values are Operator objects representing the cartesian
        components of the spin.

        I['+'] : spin raising operator;
        I['-'] : spin lowering operator;
        I['x'] : spin x component;
        I['y'] : spin y component;
        I['z'] : spin z component;

    shape
    """

    def __init__(self, s: float = 1, gamma_over_2pi: float = 1):
        """
        Constructs an instance of NuclearSpin.
        
        Parameters
        ----------
        s : float
            Spin quantum number. The constructor checks if it is a half-integer, and raises
            appropriate errors in case this condition is not obeyed.
            Default value is 1;
        gamma_over_2pi : float
            Gyromagnetic ratio over 2 pi (in units of MHz/T).
        
        Action
        ------
        Assigns the passed argument s to the attribute quantum_number.
        Assigns the passed argument gamma to the attribute gyromagnetic_ratio.
        Initialises the attribute d with the method multiplicity() (see below).
        Initialises the elements of the dictionary I using the methods
        described later, according to the following correspondence.

        |  I  | Method                  |
        | --- | ------------------------|
        |  x  | cartesian_operator()[0] |
        |  y  | cartesian_operator()[1] |
        |  z  | cartesian_operator()[2] |
        |  +  | raising_operator()      |
        |  -  | lowering_operator()     |
        |  I  | quantum number          |

        Returns
        -------
        The initialised NuclearSpin object.

        Raises
        ------
        ValueError, when the argument s is not a half-integer number (within a relative
        tolerance of 10^(-10)).
        """
        s = float(s)
        if not np.isclose(int(2 * s), 2 * s, rtol=1e-10):
            raise ValueError("The given spin quantum number is not a half-integer number")
        self.quantum_number = s
        self.d = self.multiplicity()

        # Spin operators
        self.Ix, self.Iy, self.Iz = spin_J_set(self.quantum_number)
        # Raising & lowering operators
        self.Ip, self.Im = (self.Ix + 1j * self.Iy, self.Ix - 1j * self.Iy)
        # Pack everything into a dict
        self.I = {'-': self.Im,
                  '+': self.Ip,
                  'x': self.Ix,
                  'y': self.Iy,
                  'z': self.Iz,
                  'I': self.quantum_number}

        self.gyro_ratio_over_2pi = float(gamma_over_2pi)
        # Helper dimension size of the space
        self.shape = self.I['-'].shape
        self.dims = self.I['-'].dims

    def __repr__(self):
        return (f'quantum_number: {self.quantum_number}, '
                f'multiplicity: {self.d}, shape: {self.shape}, dims: {self.dims}')

    def multiplicity(self):
        """
        Returns the spin states' multiplicity: 2*quantum_number+1 (cast to int).
        """
        return int(2 * self.quantum_number + 1)

    def spin_J_set(self):
        """
        Returns the Ix, Iy, and Iz operators.
        """
        return self.I['x'], self.I['y'], self.I['z']


class ManySpins(NuclearSpin):
    """
    An instance of this class represents a system made up of many nuclear spins,
    and its attributes include the individual NuclearSpin objects,
    the dimensions of the full Hilbert space and the components of the overall spin operator.
    """

    def __init__(self, spins: list[NuclearSpin]):
        """
        Constructs an instance of ManySpins.
  
        Parameters
        ----------
        spins : list[NuclearSpin]
            A list of the NuclearSpin objects which represent the spins in the system.
        
        Action
        ------
        Stores the NuclearSpin objects contained in the spins argument into the
        attribute spin, maintaining their original ordering.
  
        Initialises the attribute d with the product of each spin's dimensions d.
  
        Initialises the elements of the dictionary I from the corresponding 
        attributes of its spin components by calling the method many_spin_operator.

        Returns
        -------
        The initialised ManySpins object.
        """
        self.n_spins = len(spins)

        self.spins = spins
        self.d = np.prod([spin.d for spin in spins])  # multiply all the d's together

        self.dims = spins[0].dims  # updated below!
        for s in spins[1:]:
            self.dims = np.concatenate([self.dims, s.dims], axis=1)
            # Careful of qutip's convention for `dims`. For example,
            # A tensor product of a 2x2 matrix (dims = [[2],[2]]) with a 3x3 matrix (dims = [[3],[3]])
            # will result in a dims = [[2,3], [2,3]].
        if not isinstance(self.dims, list):
            self.dims = self.dims.tolist()

        self.shape = (self.d, self.d)
        self.I = {'-': self.many_spin_operator('-'),
                  '+': self.many_spin_operator('+'),
                  'x': self.many_spin_operator('x'),
                  'y': self.many_spin_operator('y'),
                  'z': self.many_spin_operator('z')}

        # For now, we are using just the first gamma value. This should be fixed later.
        self.gyro_ratio_over_2pi = spins[0].gyro_ratio_over_2pi

    def __repr__(self):
        return f' shape: {self.shape}, dims: {self.dims}'

    def many_spin_operator(
            self,
            component: str | list[str] = "z",
            spin_target: str | int | list[int] = "all") -> Qobj:
        """
        Returns a spin operator with the dimension of the ManySpins system, with the specified components at
        specified indices (details below).
        If spin_target == 'all' it applies the specified spin component to all the spins;
        otherwise, they're applied only to the specified spins.

        In the most general case (both component and spin_targets are lists), this functions outputs the operator:
        I_{component[0]} (x) Id (x) Id (x) ... (x) Id
        + Id (x) I_{component[1]} (x) Id (x) Id ... (x) Id
        + Id (x) Id (x) I_{component[2]} (x) Id ... (x) Id
        + ...
        + Id (x) Id (x) Id (x) ... (x) I_{component[-1]}

        where (x) is the tensor product, 'Id' is the identity operator in whatever dimension it appears in.
        Important details in the Parameters documentation below.

        Parameters
        ----------
        component: string or list[strings]
            Specifies which component of the spin operator is to be computed, following the key-value correspondence
            of the attribute `I` of `NuclearSpin`.

            If string, must be one of: {'x', 'y', 'z', '+', '-'}.
            Passing in a string will apply this spin component operator to all the spins.

            If list, must be a list consisting of: {'x', 'y', 'z', '+', '-', None},
            with the length equal to the number of  spins in this system ('n_spins').
            A user should pass in a list only if they want to apply different component operators at different spins.
            Passing in a list will apply the operator I_{component[i]} to the spin at index i.
            If the value at component[i] is None, no operator is applied to the spin at index i.

        spin_target: string or int or list[ints]
            The target spin(s) that the spin operator component is applied to

            If string, must be 'all', which applies the same spin operator component to all the spins in `ManySpins`.
            If int, just applies the operator to that single spin at that index value.
            If list, must be a list of index values to which the operators should apply.

            If `component` is a list, `spin_target` gets overwritten as a list of index values at which components
            are not None.

            Default is 'all'.

        Returns
        -------
        A Qobj operator with the same dimension of the ManySpins system, with details given above.
        """
        # Processing the parameters
        if isinstance(component, list):
            if not len(component) == self.n_spins:
                raise ValueError(
                f"The length of `component` ({len(component)}) must be equal to the number of spins ({self.n_spins}). "
                "You must specify the spin component for every spin. At positions where you do not want to apply an "
                "operator, put a dummy placeholder in that position, such as None.")

            # If component is a list, overwrite spin_target as a list of indices with proper component values
            spin_target = []
            for i, comp in enumerate(component):
                if comp in self.I:  # {'x', 'y', 'z', '+', '-'}
                    spin_target.append(self.I[comp])
                elif comp is not None:  # To prevent unexpected bugs when a user makes a typo
                    raise ValueError(f"Every element of `component` must be one of: ('x', 'y', 'z', '+', '-', None)")
        elif isinstance(component, str):  # A string specifying the component, which will be applied to ALL spins.
            component = self.n_spins * [component]
        else:
            raise TypeError(f"The argument `component` must be a string or a list of strings.")

        if isinstance(spin_target, int):
            spin_target = [spin_target]
        elif isinstance(spin_target, str):
            assert spin_target == "all", "If `spin_target` is a string, must be 'all'."
        elif isinstance(spin_target, list):
            for s in spin_target:
                assert isinstance(s, int), "If `spin_target` is a list, must be a list of integers!"
        else:
            raise TypeError(f"The argument `spin_target` must be a string, int, or a list of ints.")

        # Constructing the operator
        many_spin_op = Qobj(np.zeros(self.shape), dims=self.dims)
        for i in range(self.n_spins):
            if spin_target == 'all':
                # Apply the spin operator component to all the spins
                term = self.spins[i].I[component[i]]
            elif isinstance(spin_target, list) and (i in spin_target):
                # Only apply the spin operator component to the spin specified in spin_target
                term = self.spins[i].I[component[i]]
            else:
                # Apply nothing to the given spin, and make the current term 0
                term = Qobj(0 * qeye(self.spins[i].d))

            terms_to_tensor = []
            for j in range(self.n_spins):
                if i == j:
                    terms_to_tensor.append(term)
                else:
                    terms_to_tensor.append(qeye(self.spins[j].d))

            terms_tensored = tensor(terms_to_tensor)
            assert terms_tensored.dims == self.dims
            many_spin_op += terms_tensored

        assert isinstance(many_spin_op, Qobj)  # prevent weird Qobj bugs
        return many_spin_op

    def spin_J_set(self):
        """
        Returns the Ix, Iy, and Iz operators.
        """
        return self.I['x'], self.I['y'], self.I['z']
